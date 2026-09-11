# Known test failures

Tests that are red on `pipeline-v1` on purpose, with the reason and who should decide.
A test is listed here only when leaving it red is a deliberate call — never as a place to
park something nobody looked at. Each entry says what would have to change to make it green,
so "fix it" is a decision someone can take rather than a mystery.

Current: **2 failing, 888 passing, 3 skipped** (`uv run --project . pytest tests/ -q`).
The offline pipeline suite (`python3 pipeline/tests/run_tests.py`) is **117/117**.

---

## `tests/test_pathfinding_reachability_cache.py::test_cache_key_already_includes_radius`

**Since:** 2026-09-09, commit `bb10570` (port of demo-take1's pathfinding).

**Why it fails.** Its premise is that a larger robot radius makes an object *less* reachable,
so it asserts `"far_side" not in high.reachable_ids` after asserting it is reachable at the
smaller radius. `resolve_start_for_radius`, which arrived with that port, re-derives the
robot's start **per radius** into the largest connected component at that radius. At
r=0.15 the start moves 0.583 m into a component that can still see `far_side`, so the
object stays reachable and the assertion fails.

**It is not a regression introduced here.** It fails identically on `demo-take1`, the branch
the code came from — verified by running it there:

    uv run --project <repo> \
        pytest .../tests/test_pathfinding_reachability_cache.py::test_cache_key_already_includes_radius
    1 failed

**Left red deliberately.** Making it green means rewriting the assertion, and the thing the
test is named for — that the cache key includes the radius — is not what is broken. Whether
the fixture should be reshaped so a bigger radius really does disconnect `far_side`, or the
assertion should be dropped, is a call for whoever owns that port.

---

## `tests/test_msa_validate.py::test_recompute_gaps_matches_golden_for_unchanged_objects`

**Since:** before this branch's work began.

**Why it fails.** The recomputed object-pair gap set does not match the golden fixture —
extra pairs appear on the left (`toilet_0/towel_2`, `toilet_4/wall_0`, `chair_11/table_2`,
`chair_9/wall_2`, `chair_9/table_1`, and more).

**Pre-existing, confirmed.** Verified before any §9.3 or §9.5 change by stashing the working
tree and running it on the clean checkout: **1 failed, 27 passed**, the same case. Nothing in
this branch's work touches `scripts/msa` or the golden fixture.

**Left red deliberately.** It belongs to whoever owns the MSA golden data; regenerating the
golden to match the code would erase the signal that they disagree.
