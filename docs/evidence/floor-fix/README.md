# Floor-selection fix: before/after evidence

Both images are the same real clip, `01_hotel_room.mp4` (10.66s, the video named in
`/workspace/incoming_videos/manifest.csv` on the GPU box). Each shows a real, actually-run
GPU pipeline point cloud, not a synthetic/simulated example. Rendered with `matplotlib`
(`Agg` backend, offscreen; `open3d`'s GPU-accelerated `OffscreenRenderer` was tried first
and is unavailable on this VPS - `libEGL.so.1` is missing - so this uses the matplotlib
fallback, a real 3D scatter of a 200k-point random subsample of each run's actual
`aligned_room.ply`, colored by the point cloud's own RGB). Red triangles are the real
camera positions from that run's `cameras_aligned.json`. The cyan plane/line is `y=0`, the
aligned frame's chosen floor reference (both images use the SAME convention: Y is up,
`y=0` is wherever that run's algorithm decided the floor was - the whole point of these
images is that in `before.png`, `y=0` is NOT actually the real floor).

## before.png

**Run:** 2026-09-02 real GPU pipeline execution against this exact clip (Workstream C's
validation spike, `spike-poseval-20260902` in the code, output fixture still on this VPS
at `/data/spike_fixtures/pose_validation_hotel_room_20260902/`). Pipeline code at that
time was the **original single-plane-per-band floor logic** (`alignment.json`
`schema_version: 3`) - this predates even the first floor-selection fix (multi-candidate
`select_floor_candidate`, added 2026-09-03), let alone the `isaac-hardening` branch. This
is real, actual pre-fix code output, not a synthetic ablation.

**Numbers (from that run's real `alignment.json`):**
- `floor_y = -0.8536` (raw, pre-shift), `points_below_floor_frac = 18.4%` (555,094 points -
  vs. the pipeline's own 3% reflective-floor-suspect threshold), `reflective_floor_suspected
  = true`.
- Camera height above the chosen floor: **only ~0.85-0.93m** (see the red triangles sitting
  right against the cyan `y=0` line in `before.png`'s side view) - implausibly low for
  handheld/phone walkthrough footage (typically 1.2-1.6m), the visible symptom of the floor
  plane being picked too high.
- In the side view, the point cloud has a hard, flat cutoff exactly at `y=0` with almost
  nothing below it - this is the below-floor phantom-point clip (already present in the
  pipeline by this date) having already removed the 18.4% of points that were the REAL
  floor, because from that plane's perspective they looked like reflection artifacts below
  it. The room's actual geometry (visible above `y=0`) starts at what is really
  furniture/table height, not floor height.

## after.png

**Run:** 2026-09-04, job `verify-floorfix-20260904`, full 7-stage pipeline on the
`isaac-hardening` branch's current code (multi-candidate `select_floor_candidate` +
band-restricted RANSAC + the 2026-09-03 real-scene recalibration), same clip. Artifacts at
`/workspace/data/jobs/verify-floorfix-20260904/` on the `gpu` host (see
`docs/DECISIONS.md`'s 2026-09-04 entry for the full run record).

**Numbers:** `floor_y = -1.4693` (raw), `points_below_floor_frac = 0.25%`,
`reflective_floor_suspected = false`, `floor_likely_wrong = false`. Cameras sit at
**~1.46-1.54m** above the chosen floor (a plausible walking/handheld height - visibly
higher above the cyan line than in `before.png`). The side view shows a dense, continuous
floor band right at `y=0` across nearly the full footprint, with the room's geometry
correctly bounded down to real floor level.

## Validator readout

**After (real, full readout - `app/services/scene_validator.validate_scene()` run against
this job's actual `alignment.json`/`cameras_aligned.json`/`scene_objects.json`/
`occupancy_meta.json`):**

```
floor_below_cameras            passed=True  fatal=True  floor_y=0.000 vs min camera Y=1.460 (margin=1.460)
ceiling_above_cameras          passed=True  fatal=True  ceiling_y=2.865 vs max camera Y=1.539 (margin=1.326)
room_dimensions                passed=True  fatal=True  footprint 5.45x4.10m (expect 1.5-30.0m each), ceiling 2.87m (expect 2.0-5.0m)
object_height:bed              passed=True  fatal=False bed (furniture) y=0.638, expected [0.25,0.9]
object_height:chair            passed=True  fatal=False chair (furniture) y=0.635, expected [0.25,0.9]
object_height:table            passed=True  fatal=False table (furniture) y=0.678, expected [0.25,0.9]
object_height:table            passed=True  fatal=False table (furniture) y=0.545, expected [0.25,0.9]
height_prior_majority          passed=True  fatal=True  0/4 typed objects violate their height prior (0%)
relative_order_counter_below_wall passed=True  fatal=False insufficient data (0 counter, 0 wall objects), skipped
object_max_extent              passed=True  fatal=False no object exceeds the plausible-size threshold
reflection_flag                passed=True  fatal=False det(R)=1.0000 (proper rotation)

overall: PASSED
```

**Before - honest limitation, read this before using this pair as a "validator rejected
it" claim:** the 2026-09-02 fixture is a `stage_align`-only spike (Workstream C never ran
`stage_vocab`/`stage_objects`/`stage_occupancy` against it) - there is no `scene_objects.json`
or `occupancy_meta.json` for it, ever. That means **the specific check that would formally
catch this bug, `height_prior_majority`, cannot be run against this exact clip's pre-fix
output** - it was never computed at the time, and can't be reconstructed now without
re-running the old code (which no longer exists as the checked-out state of any branch,
only as this preserved point-cloud fixture). Searched this repo's own logs
(`docs/GPU_SETUP_LOG.md`) for a historical case of `height_prior_majority` actually
rejecting this specific hotel-room bug: the one recorded `height_prior_majority` failure
in this repo's history is a **different** bug (a reflective-corridor-floor "doormat" case),
not this one - so there is no real historical "validator rejected this scene" log entry to
show for the hotel-room case either.

What real, non-fabricated checks *could* be run against the before fixture's actual
`alignment.json` + `cameras_aligned.json` (`room_dimensions` skipped - no
`occupancy_meta.json`; object-level checks skipped - no `scene_objects.json`):

```
floor_below_cameras            passed=True  fatal=True  floor_y=0.000 vs min camera Y=0.850 (margin=0.850)
ceiling_above_cameras          passed=True  fatal=True  ceiling_y=2.237 vs max camera Y=0.925 (margin=1.311)
reflection_flag                passed=True  fatal=False det(R)=1.0000 (proper rotation)
```

These two geometric checks **pass even on the buggy run** - they only verify cameras sit
between whatever plane was chosen as floor and whatever was chosen as ceiling, which is
true by construction of the alignment algorithm (it orients "up" from the floor plane
toward the cameras), regardless of whether that floor plane is the real floor. They are
structurally incapable of catching this specific bug. The real, available evidence that
the before-scene is wrong is exactly the alignment-level diagnostics above:
`points_below_floor_frac=18.4%` (vs. a 3% suspect threshold), `reflective_floor_suspected
=true`, and the implausibly low ~0.85-0.93m camera-above-floor height - all real numbers
from that real run, plus the visual "hard cutoff at furniture height" in `before.png`.

**For a validator-level (object-height-prior) before/after with a REAL floor-selection
failure**, see `docs/DECISIONS.md`'s 2026-09-04 entry: the old-params ablation there
reproduces `find_floor_and_ceiling` raising outright (`RuntimeError`, no floor candidate)
against this same clip's `per_view/*.npz` - a harder failure than a wrong-but-plausible
floor, and one directly attributable (via ablation) to the RANSAC-scope change. That
ablation is alignment-only too (no `stage_objects` re-run), for the same reason: re-running
`stage_objects` under a synthetically-reverted floor was out of scope for that
verification task.

## Source data

| | before.png | after.png |
|---|---|---|
| run | 2026-09-02 Workstream C spike | 2026-09-04 job `verify-floorfix-20260904` |
| code | pre-fix, single-plane-per-band (`schema_version: 3`) | `isaac-hardening` current |
| artifacts | `/data/spike_fixtures/pose_validation_hotel_room_20260902/` (this VPS) | `/workspace/data/jobs/verify-floorfix-20260904/` (`gpu` host) |
| full pipeline run? | no - alignment only | yes - all 7 stages |
