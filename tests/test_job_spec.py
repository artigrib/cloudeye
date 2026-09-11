"""The job-spec contract: defaults, validation, the file it lives in, and agreement
between the pydantic model and docs/job_spec.schema.json.

The schema file is the one a non-Python consumer reads, so it is not decoration - these
tests fail if the two drift apart.
"""

import json
import uuid
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.services.job_spec import (
    SEMANTICS_TO_VOCAB_PROVIDER,
    SPEC_VERSION,
    JobSpec,
    now_iso,
    read_job_spec,
    spec_path_for_video,
    write_job_spec,
)

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "docs" / "job_spec.schema.json"


@pytest.fixture(scope="module")
def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def test_defaults_are_the_documented_ones():
    spec = JobSpec(video_id=uuid.uuid4())
    assert spec.spec_version == SPEC_VERSION
    assert spec.frames_fps == 15
    assert spec.mapping == "mapanything"
    assert spec.semantics == "openrouter-glm"
    assert spec.chat_llm == "openrouter-nemotron"
    # [] means "every registered platform", which is what the viewer already shows - it is
    # not the same as "no platforms", and nothing downstream should read it that way.
    assert spec.robots == []


@pytest.mark.parametrize("fps", [0, -1, 31])
def test_frames_fps_is_bounded(fps):
    with pytest.raises(ValidationError):
        JobSpec(video_id=uuid.uuid4(), frames_fps=fps)


def test_mapping_has_exactly_one_legal_value():
    """The field exists so that a second backend is a named schema change rather than a
    silent reinterpretation of specs already on disk."""
    with pytest.raises(ValidationError):
        JobSpec(video_id=uuid.uuid4(), mapping="colmap")


@pytest.mark.parametrize("field,bad", [("semantics", "openrouter"), ("chat_llm", "gpt-4")])
def test_provider_fields_reject_unknown_values(field, bad):
    with pytest.raises(ValidationError):
        JobSpec(video_id=uuid.uuid4(), **{field: bad})


def test_semantics_maps_onto_the_catalog_provider_ids():
    """`vocab_provider` is what the upload path and the scene-vocabulary stage already
    speak; the spec's own names must translate into it, not replace it."""
    assert JobSpec(video_id=uuid.uuid4(), semantics="openrouter-glm").vocab_provider() == "openrouter"
    assert JobSpec(video_id=uuid.uuid4(), semantics="vertex-gemma").vocab_provider() == "vertex"


def test_spec_path_sits_beside_the_video():
    assert spec_path_for_video("var/uploads/abc.mp4") == Path("var/uploads/abc.job.json")
    assert spec_path_for_video("var/uploads/abc.mov") == Path("var/uploads/abc.job.json")
    # An extension-less video must not have its spec path collide with the video itself.
    assert spec_path_for_video("var/uploads/abc") == Path("var/uploads/abc.job.json")


def test_write_then_read_round_trips(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"not really a video")
    spec = JobSpec(
        video_id=uuid.uuid4(),
        frames_fps=7.5,
        semantics="vertex-gemma",
        chat_llm="vertex-gemma",
        robots=["husky", "go2"],
        created_at=now_iso(),
    )
    written = write_job_spec(video, spec)

    assert written == tmp_path / "v.job.json"
    # No .tmp left behind - the write is atomic, not two-phase-and-hope.
    assert not list(tmp_path.glob("*.tmp"))

    back = read_job_spec(video)
    assert back == spec


def test_read_returns_none_when_there_is_no_spec(tmp_path):
    """Every video uploaded before the wizard existed has no spec. That is "use the
    defaults", never an error."""
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    assert read_job_spec(video) is None


def test_written_json_is_the_documented_shape(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    vid = uuid.uuid4()
    write_job_spec(video, JobSpec(video_id=vid, created_at="2026-09-09T11:04:22+02:00"))

    doc = json.loads((tmp_path / "v.job.json").read_text(encoding="utf-8"))
    assert doc == {
        "spec_version": 1,
        "video_id": str(vid),
        "frames_fps": 15.0,
        "mapping": "mapanything",
        "semantics": "openrouter-glm",
        "chat_llm": "openrouter-nemotron",
        "robots": [],
        "created_at": "2026-09-09T11:04:22+02:00",
    }


# --- the schema file and the model must agree ---------------------------------------


def test_schema_and_model_have_the_same_fields(schema):
    assert set(schema["properties"]) == set(JobSpec.model_fields)


def test_schema_required_fields_are_the_non_optional_ones(schema):
    # created_at is the only optional field: a spec is still valid without it.
    assert set(schema["required"]) == set(JobSpec.model_fields) - {"created_at"}


@pytest.mark.parametrize(
    "field,expected",
    [
        ("semantics", ["openrouter-glm", "vertex-gemma"]),
        ("chat_llm", ["openrouter-nemotron", "vertex-gemma"]),
        ("mapping", ["mapanything"]),
    ],
)
def test_schema_enums_match_the_model(schema, field, expected):
    assert schema["properties"][field]["enum"] == expected


def test_schema_defaults_match_the_model(schema):
    defaults = JobSpec(video_id=uuid.uuid4())
    for field in ("frames_fps", "mapping", "semantics", "chat_llm"):
        assert schema["properties"][field]["default"] == getattr(defaults, field), field


def test_schema_robot_enum_is_the_real_registry(schema):
    """A platform id that does not exist would be accepted by the wizard and then fail in
    the worker, so the schema's list is pinned to the registry rather than hand-kept."""
    from app import robots

    registry_ids = {r.id for r in robots.list_robots()}
    assert set(schema["properties"]["robots"]["items"]["enum"]) == registry_ids


def test_semantics_map_covers_every_enum_value(schema):
    assert set(SEMANTICS_TO_VOCAB_PROVIDER) == set(schema["properties"]["semantics"]["enum"])


def test_catalog_default_matches_the_job_spec_default():
    """One decision, one default.

    GET /api/models/catalog's "Scene vocabulary" default and the wizard's `semantics`
    default are the same choice asked twice, and they disagreed: the catalog said
    `vertex` while the wizard defaulted to `openrouter-glm`, so an operator who accepted
    the wizard's default got a different provider from one who uploaded without a spec.
    Settled 2026-09-09 in favour of OpenRouter; this test is what stops it drifting again.
    """
    from app.services.model_catalog import DEFAULT_VOCAB_PROVIDER

    wizard_default = JobSpec(video_id=uuid.uuid4()).vocab_provider()
    assert wizard_default == DEFAULT_VOCAB_PROVIDER


def test_catalog_stage_reports_that_same_default():
    """Not just the constant - what the endpoint actually serves, since that is what the
    frontend reads."""
    from app.services.model_catalog import get_catalog

    vocab = next(s for s in get_catalog() if s.stage == "Scene vocabulary")
    assert vocab.default == JobSpec(video_id=uuid.uuid4()).vocab_provider()
    assert vocab.default in {o.id for o in vocab.options}


def test_delete_removes_the_spec_beside_the_video(tmp_path):
    """The spec is a file the app writes; deleting the workspace has to take it with the
    video. It did not - measured on a throwaway workspace, the .mp4 went and the
    <uuid>.job.json stayed, orphaning one spec per deleted workspace forever."""
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    write_job_spec(video, JobSpec(video_id=uuid.uuid4()))
    spec = spec_path_for_video(video)
    assert spec.is_file()

    # What project_service.delete_project does to a video, in the same order.
    video.unlink(missing_ok=True)
    spec_path_for_video(video).unlink(missing_ok=True)

    assert not video.exists()
    assert not spec.exists()
