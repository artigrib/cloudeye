"""Reading pipeline-v1's substage fields out of status.json.

`state` says whether a step finished; `substage` says what it is doing inside that, which
is the difference between "GPU_UP_NV has been running eleven minutes because it is broken"
and "...because vast has no 4090 free and it is on attempt 4 of 12"
(run_pipeline.SUBSTAGES, and its own comment saying exactly that).

Every one of these fields is OPTIONAL, and the run this repo actually has on disk -
own_0901_161054, 2026-09-09 - carries none of them: it was written by a build that
predates them. So "absent stays absent" is not a corner case here, it is the common one,
and it is what these tests spend most of their assertions on.
"""

import json

from app.services.processing_status import SUBSTAGES, _rows_from_status


def _doc(step: dict) -> dict:
    return {"state": "GPU_UP_NV", "steps": {"GPU_UP_NV": step}}


def test_the_vocabulary_matches_the_writers():
    # Copied from run_pipeline.SUBSTAGES. If the writer grows a value and this does not,
    # the reader still passes it through (see below) but stops being able to say whether
    # it was agreed to.
    assert SUBSTAGES == (
        "searching",
        "creating",
        "provisioning",
        "waiting_capacity",
        "transfer_gate",
        "refused",
        "waiting_manual",
        "skipped",
    )


def test_a_step_with_no_substage_fields_yields_none_for_every_one_of_them():
    # Exactly the shape of the recorded own_0901_161054 run.
    row = _rows_from_status(_doc({
        "state": "done",
        "started": "2026-09-09T13:46:39+02:00",
        "finished": "2026-09-09T13:47:11+02:00",
        "duration_s": 32.18,
        "error": None,
    }))[0]

    assert (row.substage, row.detail, row.since, row.next_retry) == (None, None, None, None)
    assert row.attempt is None
    assert row.needs_attention is False
    # ...and the fields that ARE there still arrive.
    assert (row.stage, row.state, row.duration_sec) == ("GPU_UP_NV", "done", 32.18)


def test_a_waiting_step_carries_its_substage_detail_since_attempt_and_retry():
    row = _rows_from_status(_doc({
        "state": "running",
        "started": "2026-09-09T13:46:39+02:00",
        "finished": None,
        "duration_s": None,
        "error": None,
        "substage": "waiting_capacity",
        "detail": "no box available on attempt 4: no capacity",
        "since": "2026-09-09T13:50:10+02:00",
        "attempt": 4,
        "next_retry": "2026-09-09T13:51:10+02:00",
    }))[0]

    assert row.substage == "waiting_capacity"
    assert row.detail == "no box available on attempt 4: no capacity"
    # `since` is the SUBSTAGE's clock, not the step's - they differ here by 3m31s, which
    # is the whole reason the field exists.
    assert row.since == "2026-09-09T13:50:10+02:00"
    assert row.started_at == "2026-09-09T13:46:39+02:00"
    assert row.attempt == 4
    assert row.next_retry == "2026-09-09T13:51:10+02:00"
    assert row.needs_attention is False


def test_attempt_zero_survives_as_zero_rather_than_becoming_absent():
    # `Status.enter` initialises attempt to 0. `or None` would erase it, and 0 and None
    # mean different things to a client deciding whether to render the row: one is "the
    # writer said nothing", the other is "the writer said no attempts yet".
    row = _rows_from_status(_doc({"state": "running", "attempt": 0}))[0]
    assert row.attempt == 0


def test_refused_and_waiting_manual_are_the_two_that_need_attention():
    for substage in SUBSTAGES:
        row = _rows_from_status(_doc({"state": "running", "substage": substage}))[0]
        assert row.needs_attention is (substage in ("refused", "waiting_manual"))


def test_a_substage_the_reader_does_not_know_is_passed_through_not_dropped():
    # Same convention `state` already follows. Dropping it would turn a typo in the
    # writer into an invisible one here.
    row = _rows_from_status(_doc({"state": "running", "substage": "teleporting"}))[0]
    assert row.substage == "teleporting"
    assert row.needs_attention is False


def test_the_recorded_run_reads_back_with_no_substage_anywhere():
    """The real thing, not a fixture of it: the file this screen was built against."""
    source = "var/scratch/pipeline_live/scene161054/status.json"
    try:
        doc = json.loads(open(source).read())
    except OSError:
        import pytest

        pytest.skip(f"{source} not on this box")

    rows = _rows_from_status(doc)
    assert [r.stage for r in rows][:3] == ["QUEUED", "FRAMES", "GPU_UP_MA"]
    assert len(rows) == 15
    assert all(r.substage is None for r in rows)
    assert all(r.attempt is None for r in rows)
    assert all(not r.needs_attention for r in rows)
    # The states it does carry, verbatim - including two this app had never enumerated.
    assert {r.state for r in rows} == {"done", "skipped", "skipped_precomputed", "stubbed"}
