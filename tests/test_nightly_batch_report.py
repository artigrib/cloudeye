"""scripts/nightly_batch.py's report-building logic, tested against synthetic
SceneRunResult fixtures - no DB, no GPU, no network, no real pipeline run (matching
this suite's existing pure-unit convention, see conftest.py). Only the formatting
functions (`build_results_json`, `build_report_md`, `write_report`) and the pure
worker.log stage-timing parser (`parse_stage_timings`) are exercised here; the
HTTP-driving functions (upload_scene, poll_scene_status, run_one_scene, ...) need a
real backend + GPU pipeline and are deliberately not covered by this file.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.nightly_batch import (
    SceneRunResult,
    StageTiming,
    _is_regression_project_name,
    build_report_md,
    build_results_json,
    derive_validator_status,
    detect_retain_per_view_setting,
    parse_stage_timings,
    scene_result_to_dict,
    write_report,
)

WORKER_LOG_SAMPLE = """\
2026-08-31 05:55:03,709 INFO [app.services.pipeline_orchestrator] scene=e1e21716 stage=keyframes (1/7) state=running message=starting keyframes
2026-08-31 05:55:19,637 INFO [app.services.pipeline_orchestrator] scene=e1e21716 stage=infer (3/7) state=running message=starting infer
2026-08-31 05:57:46,365 INFO [app.services.pipeline_orchestrator] scene=e1e21716 stage=align (4/7) state=running message=starting align
2026-08-31 05:58:34,767 INFO [app.services.pipeline_orchestrator] scene=e1e21716 stage=objects (5/7) state=running message=starting objects
2026-08-31 06:05:03,058 INFO [app.services.pipeline_orchestrator] scene=e1e21716 stage=export_glb (7/7) state=running message=starting export_glb
2026-08-31 06:05:51,666 INFO [app.services.pipeline_orchestrator] scene=e1e21716 stage=finalize (7/7) state=done message=pipeline complete
"""


def _done_result(name="02_modular_home") -> SceneRunResult:
    return SceneRunResult(
        name=name,
        project_id="9a8e6ec6-41ee-4738-ae3a-88af3a37b878",
        video_id="62cabce3-4ad3-49fc-ac2b-4bc37ad990d8",
        scene_id="e1e21716-841e-449e-b973-07c9020094e2",
        status="done",
        validator_status="passed",
        error_message=None,
        floor_y=0.0,
        ceiling_y=2.7076803001776657,
        ceiling_height_m=2.7076803001776657,
        num_objects=40,
        isaac_validate={"status": "pass", "exit_code": 0},
        stage_timings=[
            StageTiming(stage="keyframes", started_at="2026-08-31T05:55:03", duration_sec=15.9),
            StageTiming(stage="finalize", started_at="2026-08-31T06:05:51", duration_sec=None),
        ],
        total_duration_sec=650.0,
        started_at="2026-08-31T05:55:02+00:00",
        finished_at="2026-08-31T06:05:52+00:00",
    )


def _failed_result(name="room") -> SceneRunResult:
    return SceneRunResult(
        name=name,
        project_id="79834afb-9f42-4ceb-aaac-4953ee5d96ec",
        video_id="89e97f15-9b60-410f-9595-e9232e70e05e",
        scene_id="cafd5865-2de6-4c16-bbf6-32382c45f9ae",
        status="failed",
        validator_status="failed",
        error_message="ceiling_above_cameras: ceiling_y=2.105 vs max camera Y=3.570",
        floor_y=None,
        ceiling_y=None,
        num_objects=None,
        isaac_validate=None,
        stage_timings=[],
        total_duration_sec=210.5,
        started_at="2026-09-05T02:00:00+00:00",
        finished_at="2026-09-05T02:03:30+00:00",
    )


# --- parse_stage_timings ----------------------------------------------------------


def test_parse_stage_timings_computes_deltas_between_transitions():
    timings = parse_stage_timings(WORKER_LOG_SAMPLE)
    stages = [t.stage for t in timings]
    assert stages == ["keyframes", "infer", "align", "objects", "export_glb", "finalize"]
    # keyframes -> infer: 05:55:19.637 - 05:55:03.709 = 15.928s
    assert timings[0].duration_sec == 15.928
    # last observed transition (finalize) has no next boundary
    assert timings[-1].duration_sec is None


def test_parse_stage_timings_empty_text_returns_empty_list():
    assert parse_stage_timings("") == []


def test_parse_stage_timings_ignores_non_stage_lines():
    text = "2026-08-31 05:55:03,709 INFO [x] some unrelated line\n" + WORKER_LOG_SAMPLE
    timings = parse_stage_timings(text)
    assert len(timings) == 6  # the unrelated line contributes nothing


# --- derive_validator_status --------------------------------------------------------


def test_derive_validator_status_done_is_passed():
    assert derive_validator_status("done", None) == "passed"


def test_derive_validator_status_failed_with_known_validator_marker():
    assert derive_validator_status("failed", "ceiling_above_cameras: ...") == "failed"
    assert derive_validator_status("failed", "height_prior_majority: ...") == "failed"


def test_derive_validator_status_failed_without_validator_marker_is_pipeline_error():
    assert derive_validator_status("failed", "GPU box unreachable: connection refused") == "pipeline_error"


def test_derive_validator_status_non_terminal_is_unknown():
    assert derive_validator_status("timeout", None) == "unknown"
    assert derive_validator_status("processing", None) == "unknown"


# --- _is_regression_project_name ----------------------------------------------------


def test_numbered_corpus_names_are_regression_scenes():
    assert _is_regression_project_name("01_hotel_room")
    assert _is_regression_project_name("02_modular_home")
    assert _is_regression_project_name("15_sea_grill_restaurant")


def test_canonical_hand_named_scenes_are_regression_scenes():
    assert _is_regression_project_name("room")
    assert _is_regression_project_name("street1")
    assert _is_regression_project_name("street2")
    assert _is_regression_project_name("ikport")


def test_rerun_and_retest_duplicates_are_excluded():
    assert not _is_regression_project_name("room-rerun-49271597")
    assert not _is_regression_project_name("ikport-rerun-49271597")
    assert not _is_regression_project_name("room_retest_20260904")
    assert not _is_regression_project_name("corridor-retest")
    assert not _is_regression_project_name("room-rerun-guard-fix")


def test_unrelated_auto_named_projects_are_excluded():
    assert not _is_regression_project_name("VID_20260901_155452724")
    assert not _is_regression_project_name("66db17e2-99d9-47a2-9e2d-7d7098fc8399")
    assert not _is_regression_project_name("smoke-test")


# --- build_results_json / scene_result_to_dict --------------------------------------


def test_build_results_json_shape_and_summary_counts():
    results = [_done_result(), _failed_result()]
    payload = build_results_json(
        run_date="2026-09-05",
        generated_at="2026-09-05T23:10:00+00:00",
        retain_per_view_artifacts_effective=False,
        results=results,
    )
    assert payload["schema"] == "cloudeye.nightly_batch/1"
    assert payload["run_date"] == "2026-09-05"
    assert payload["retain_per_view_artifacts_effective"] is False
    assert payload["summary"] == {"total_scenes": 2, "done": 1, "failed_or_incomplete": 1}
    assert len(payload["scenes"]) == 2
    assert payload["scenes"][0]["name"] == "02_modular_home"
    assert payload["scenes"][0]["ceiling_height_m"] == 2.7076803001776657
    assert payload["scenes"][1]["status"] == "failed"


def test_build_results_json_is_actually_json_serializable():
    results = [_done_result(), _failed_result()]
    payload = build_results_json(
        run_date="2026-09-05",
        generated_at="2026-09-05T23:10:00+00:00",
        retain_per_view_artifacts_effective=True,
        results=results,
    )
    # Must round-trip cleanly - this is exactly what gets written to results.json.
    reparsed = json.loads(json.dumps(payload))
    assert reparsed == payload


def test_scene_result_to_dict_includes_stage_timings():
    d = scene_result_to_dict(_done_result())
    assert d["stage_timings"] == [
        {"stage": "keyframes", "started_at": "2026-08-31T05:55:03", "duration_sec": 15.9},
        {"stage": "finalize", "started_at": "2026-08-31T06:05:51", "duration_sec": None},
    ]


# --- build_report_md -----------------------------------------------------------------


def test_report_md_contains_scene_names_and_key_fields():
    md = build_report_md(
        run_date="2026-09-05",
        generated_at="2026-09-05T23:10:00+00:00",
        retain_per_view_artifacts_effective=False,
        results=[_done_result(), _failed_result()],
    )
    assert "# Nightly batch report - 2026-09-05" in md
    assert "02_modular_home" in md
    assert "room" in md
    assert "2.708" in md  # ceiling_height_m, formatted to 3 decimals
    assert "1/2 scenes done." in md


def test_report_md_warns_when_retain_per_view_is_off():
    md = build_report_md(
        run_date="2026-09-05",
        generated_at="2026-09-05T23:10:00+00:00",
        retain_per_view_artifacts_effective=False,
        results=[_done_result()],
    )
    assert "WARNING" in md
    assert "retain_per_view_artifacts" in md


def test_report_md_no_warning_when_retain_per_view_is_on():
    md = build_report_md(
        run_date="2026-09-05",
        generated_at="2026-09-05T23:10:00+00:00",
        retain_per_view_artifacts_effective=True,
        results=[_done_result()],
    )
    assert "WARNING" not in md
    assert "currently **True**" in md


def test_report_md_shows_failed_scene_error_message():
    md = build_report_md(
        run_date="2026-09-05",
        generated_at="2026-09-05T23:10:00+00:00",
        retain_per_view_artifacts_effective=True,
        results=[_failed_result()],
    )
    assert "ceiling_above_cameras" in md


# --- write_report (real filesystem I/O, tmp_path - still no network/DB/GPU) ----------


def test_write_report_writes_both_files(tmp_path):
    out_dir = tmp_path / "nightly" / "2026-09-05"
    write_report(
        out_dir,
        run_date="2026-09-05",
        generated_at="2026-09-05T23:10:00+00:00",
        retain_per_view_artifacts_effective=False,
        results=[_done_result(), _failed_result()],
    )
    assert (out_dir / "report.md").is_file()
    assert (out_dir / "results.json").is_file()

    payload = json.loads((out_dir / "results.json").read_text())
    assert payload["summary"]["total_scenes"] == 2

    report_text = (out_dir / "report.md").read_text()
    assert "02_modular_home" in report_text
    assert "room" in report_text


# --- detect_retain_per_view_setting (reads a given .env-shaped file, still no I/O
# against the real repo/service) -----------------------------------------------------


def test_detect_retain_per_view_true(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("DATABASE_URL=x\nRETAIN_PER_VIEW_ARTIFACTS=true\n")
    assert detect_retain_per_view_setting(env_file) is True


def test_detect_retain_per_view_false_when_absent(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("DATABASE_URL=x\n")
    assert detect_retain_per_view_setting(env_file) is False


def test_detect_retain_per_view_false_when_file_missing(tmp_path):
    assert detect_retain_per_view_setting(tmp_path / "does_not_exist.env") is False


def test_detect_retain_per_view_ignores_comments(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("# RETAIN_PER_VIEW_ARTIFACTS=true\nRETAIN_PER_VIEW_ARTIFACTS=false\n")
    assert detect_retain_per_view_setting(env_file) is False
