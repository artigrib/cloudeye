"""app/services/stage_timings.py: parsing (re-exported by scripts/nightly_batch.py,
already covered there against a shared WORKER_LOG_SAMPLE - not duplicated here) plus
the on-disk persistence (`stage_timings.json`) and the
app.services.scene_service.build_status_response helper both routers'
`GET .../status`-shaped endpoints share. Pure filesystem tests, no DB/GPU/network -
matches this suite's existing pure-unit convention (see conftest.py)."""

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.services.stage_timings import (
    StageTiming,
    read_stage_timings,
    refresh_stage_timings_from_log,
    stage_timings_path,
    write_stage_timings,
)

WORKER_LOG_SAMPLE = """\
2026-08-31 05:55:03,709 INFO [app.services.pipeline_orchestrator] scene=e1e21716 stage=keyframes (1/7) state=running message=starting keyframes
2026-08-31 05:55:19,637 INFO [app.services.pipeline_orchestrator] scene=e1e21716 stage=infer (3/7) state=running message=starting infer
2026-08-31 05:57:46,365 INFO [app.services.pipeline_orchestrator] scene=e1e21716 stage=align (4/7) state=running message=starting align
"""


# --- write_stage_timings / read_stage_timings --------------------------------------


def test_write_then_read_round_trips(tmp_path: Path):
    timings = [
        StageTiming(stage="keyframes", started_at="2026-08-31T05:55:03", duration_sec=15.9),
        StageTiming(stage="infer", started_at="2026-08-31T05:55:19", duration_sec=None),
    ]
    write_stage_timings(tmp_path, timings)
    assert stage_timings_path(tmp_path).exists()
    assert read_stage_timings(tmp_path) == timings


def test_read_missing_file_returns_empty_list(tmp_path: Path):
    assert read_stage_timings(tmp_path / "does-not-exist") == []


def test_read_corrupt_file_returns_empty_list_not_raise(tmp_path: Path):
    stage_timings_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    stage_timings_path(tmp_path).write_text("{not valid json")
    assert read_stage_timings(tmp_path) == []


def test_write_is_atomic_no_tmp_file_left_behind(tmp_path: Path):
    write_stage_timings(tmp_path, [StageTiming(stage="keyframes", started_at="2026-08-31T05:55:03")])
    assert not stage_timings_path(tmp_path).with_suffix(".json.tmp").exists()


# --- refresh_stage_timings_from_log --------------------------------------------------


def test_refresh_parses_log_and_persists(tmp_path: Path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "worker.log").write_text(WORKER_LOG_SAMPLE)

    timings = refresh_stage_timings_from_log(tmp_path)

    assert [t.stage for t in timings] == ["keyframes", "infer", "align"]
    assert timings[-1].duration_sec is None  # last observed stage, no next boundary yet
    assert timings[0].duration_sec == pytest.approx(15.928, abs=1e-3)
    # Persisted, and a subsequent plain read (no re-parse) sees the same thing.
    assert read_stage_timings(tmp_path) == timings


def test_refresh_missing_log_returns_empty_and_does_not_raise(tmp_path: Path):
    assert refresh_stage_timings_from_log(tmp_path) == []
    assert not stage_timings_path(tmp_path).exists()


def test_refresh_overwrites_previous_persisted_file(tmp_path: Path):
    write_stage_timings(tmp_path, [StageTiming(stage="stale", started_at="2020-01-01T00:00:00")])
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "worker.log").write_text(WORKER_LOG_SAMPLE)

    refresh_stage_timings_from_log(tmp_path)

    stages = [t.stage for t in read_stage_timings(tmp_path)]
    assert "stale" not in stages
    assert stages == ["keyframes", "infer", "align"]


# --- scene_service.build_status_response --------------------------------------------


def _fake_scene(status: str, error_message: str | None = None):
    from app.models import Scene

    return Scene(
        id=uuid.uuid4(),
        video_id=uuid.uuid4(),
        project_id=uuid.uuid4(),
        status=status,
        error_message=error_message,
        is_reflected=False,
        reflective_floor_suspected=False,
        created_at=datetime.now(timezone.utc),
        completed_at=None,
    )


def test_build_status_response_processing_reflects_live_log(tmp_path, monkeypatch):
    from app.config import settings
    from app.services import scene_service

    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
    scene = _fake_scene("processing")
    d = scene_service.scene_dir(scene.id)
    (d / "logs").mkdir(parents=True)
    (d / "logs" / "worker.log").write_text(WORKER_LOG_SAMPLE)

    resp = scene_service.build_status_response(scene)

    assert resp.status == "processing"
    assert resp.stage == "align"  # last transition observed
    assert [t.stage for t in resp.stage_timings] == ["keyframes", "infer", "align"]


def test_build_status_response_terminal_reads_persisted_file_not_log(tmp_path, monkeypatch):
    from app.config import settings
    from app.services import scene_service
    from app.services.stage_timings import write_stage_timings

    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
    scene = _fake_scene("done")
    d = scene_service.scene_dir(scene.id)
    write_stage_timings(d, [StageTiming(stage="finalize", started_at="2026-08-31T06:05:51", duration_sec=1.2)])
    # No logs/worker.log at all - a terminal scene must not need it.

    resp = scene_service.build_status_response(scene)

    assert resp.status == "done"
    assert resp.stage == "finalize"
    assert resp.stage_timings[0].duration_sec == 1.2


def test_build_status_response_failed_scene_carries_validator_error_message(tmp_path, monkeypatch):
    from app.config import settings
    from app.services import scene_service

    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
    msg = "scene failed validation: ceiling_above_cameras: ceiling_y=2.105 vs max camera Y=3.570"
    scene = _fake_scene("failed", error_message=msg)

    resp = scene_service.build_status_response(scene)

    assert resp.status == "failed"
    assert resp.error_message == msg


def test_build_status_response_no_timings_leaves_stage_none(tmp_path, monkeypatch):
    from app.config import settings
    from app.services import scene_service

    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
    scene = _fake_scene("queued")

    resp = scene_service.build_status_response(scene)

    assert resp.stage is None
    assert resp.stage_timings == []
