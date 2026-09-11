# msa/02_modular_home fixture

Source: real scene `e1e21716-841e-449e-b973-07c9020094e2` (project "02_modular_home"),
processed 2026-08-31. Pulled 2026-09-06 via `scripts/msa/make_test_fixture.py`
(re-run that script to regenerate `occupancy.npy`/`occupancy_meta.json`/
`scene_objects_hulls.json` from the live scene directory, e.g. after an
occupancy-classification change).

- `occupancy.npy` / `occupancy_meta.json` - unmodified copies from the scene's own
  `gpu/stage_occupancy.py` output (5cm grid, 159x93 cells).
- `scene_meta.json` - `floor_y=0.0`, `ceiling_y=2.7076803001776657`, taken from the
  DB `scenes` row (NOT from `alignment.json`'s own `floor_y`/`ceiling_y` fields -
  `alignment.json`'s `floor_y` is in a pre-normalization frame: it reads -1.279 on
  this scene while every object's `bbox_min_y` and the occupancy grid's own height
  band are already floor-zeroed. Using `alignment.json` directly here would
  silently misplace every wall/floor extrusion - see `docs/DECISIONS.md`).
- `scene_objects_hulls.json` - a compact stand-in for `scene_objects.json` +
  per-object `.ply` point clouds: for each of the 67 real detected objects, the
  convex hull of that object's own point cloud projected to XZ (typically 10-40
  vertices), plus `bbox_min`/`bbox_max`/mean `color_rgb`. The real per-object PLY
  files are tens to hundreds of MB each (open3d binary point clouds, one per
  object) - far over the ≤1MB-per-fixture budget - so this fixture stores only
  what `oriented_min_area_rect` actually needs (the hull), not the raw points.
  The hull-from-points -> oriented-rectangle logic itself is already covered
  end-to-end on synthetic point sets in `tests/test_msa_geometry.py`
  (`TestOrientedMinAreaRect`), so this is a faithful substitution, not a weaker
  test path.
- `golden_gaps.json` - `gaps.json` produced by `scripts.msa.bootstrap.run_bootstrap`
  on this exact fixture as of the commit that added it (`msa/A0-bootstrap`, see
  `docs/DECISIONS.md`). `tests/test_msa_bootstrap.py`'s integration test
  regenerates gaps.json from the fixture and asserts each entry's `width_m` is
  within 1cm of this golden file, matched by the `(a, b)` obstacle-id pair,
  independent of ordering. Regenerate it by running `run_bootstrap` on this
  fixture dir (with `object_inputs=load_object_inputs_from_hulls_json(...)`, as
  `bootstrap_report` in `tests/test_msa_bootstrap.py` does) and copying the
  resulting `gaps.json` over this file.

  **2026-09-06 delta (wall-backfill Tier 1):** `compute_object_footprints` gained
  a `walls` parameter (SPEC: extend a footprint edge outward to touch a wall
  polygon when that edge already sits within 0.10m of it - closes the sliver a
  wall-flush object like a wardrobe or headboard leaves because the camera never
  captured the floor right against the wall). On this fixture, backfill moved 12
  object footprints out to a nearby wall: `bed_1`, `pillow_0`, `rug_0`, `rug_3`,
  `refrigerator_0`, `counter_0`, `coffee maker_1`, `coffee maker_2`, `sink_0`,
  `toilet_0`, `toilet_1`, `toilet_3`. That shifted 6 of 59 gap entries by more
  than 1cm (regenerated and reverified against a fresh `run_bootstrap` pass):
  - `counter_0`-`refrigerator_0` (0.10m gap) disappeared and `counter_0`-`rug_0`
    (0.1195m) appeared instead - both objects moved toward walls near each
    other, changing which pair is nearest.
  - `chair_0`-`rug_0`: 0.3384m -> 0.2914m (`rug_0` backfilled toward its wall,
    `chair_0` unaffected).
  - `table_0`-`rug_0`: 0.2118m -> 0.1207m (same cause).
  - `stool_1`-`rug_0`: 0.5328m -> 0.4183m (same cause).
  - `toilet_1`-`wall_0`: 0.10m -> 0.1207m (`toilet_1` itself backfilled toward a
    *different* wall than `wall_0`, moving it slightly further from `wall_0`).
  All other gap widths are unchanged (<1cm), and the total gap count stayed at
  59. See `scripts/msa/bootstrap.py`'s `_backfill_footprint_to_walls` for the
  exact rule.

  **2026-09-06 delta (small/short-footprint filter + floor-plate union):**
  `run_bootstrap` now drops any object footprint with area < 0.04m^2 OR height
  < 0.10m, unless it touches a wall or another object's footprint (see
  `scripts/msa/bootstrap.py`'s `_filter_small_isolated_objects` docstring for
  the "touching floor/wall/object" interpretation - short version: everything
  reaching this point is already floor-level, since overhead objects are
  excluded earlier, so the operative check is wall/object adjacency, not a
  separate floor check). On this fixture 5 of the 63 non-overhead objects were
  dropped as isolated noise: `refrigerator_1`, `refrigerator_2`, `broom_0`,
  `towel_1`, `toilet_4` (all duplicate/spurious detections sitting alone in
  open floor space, not touching anything). That changed which obstacle
  sources feed gap computation, so `golden_gaps.json` was regenerated on top
  of T4's version:
  - Gap count dropped from 59 to 47: the 14 removed gap pairs each involve one
    of the 5 dropped objects (e.g. `refrigerator_0`-`refrigerator_1`,
    `broom_0`-`wall_1`, `toilet_0`-`toilet_4`, ...).
  - 2 new pairs appeared where a dropped object used to be the nearest
    neighbor: `rug_3`-`wall_1` (0.25m, previously `rug_3` was nearest to the
    now-removed `refrigerator_2`) and `sink_2`-`wall_0` (0.3803m, same cause).
  - Among the 45 pairs common to both golden files, only `wall_0`-`wall_1`
    changed by more than 1cm (1.00m -> 0.4562m) - removing the 5 objects
    changed the free-space topology enough to reveal a narrower true passage
    between those two walls that was previously not the reported measurement
    point. All other common pairs are unchanged (<1cm).

  Separately, `run_bootstrap`'s USD floor mesh and `export_glb.build_scene`'s
  GLB "floor" node now extrude from `union(floor_polygon, every kept object's
  footprint)` (`export_glb._floor_plate_polygon`) instead of `floor_polygon`
  alone, so an object footprint that pokes slightly past the free-space
  polygon boundary (wall-backfilled objects in particular) still has solid
  floor under it in the visual exports. This doesn't change `golden_gaps.json`
  (gaps are computed from the occupancy grid + object masks, not from the
  floor mesh) and doesn't touch the `Plan/floor` outline curve or
  `export_dxf.py`'s floor plan (neither draws a filled floor surface, so
  neither has a missing-fill problem to fix - see `_floor_plate_polygon`'s
  docstring).

  **2026-09-06 delta (reflection/through-window guard, room-clipping):**
  `run_bootstrap` now runs `_drop_or_clip_objects_outside_room` right after the
  small-object filter above (see that function's docstring in
  `scripts/msa/bootstrap.py`): an object footprint more than 50% outside
  `floor_polygon` (the free-space boundary from `extract_floor_polygon`) is
  dropped as a likely mirror-reflection/through-a-window false detection; one
  <=50% outside is clipped to `floor_polygon` and refit to a new oriented
  rectangle via `oriented_min_area_rect`.

  **This fixture is a real, unusually strong stress test of that rule, for a
  reason worth calling out explicitly:** `extract_floor_polygon` returns only
  the single *largest* connected free-space component (pre-existing behavior,
  unchanged by this delta), and this "modular_home" scene's occupancy grid
  free space is genuinely split across several disconnected modules/rooms
  (multiple toilets/sinks/dining areas well outside the main room's x/z
  extent - confirmed by inspecting `floor_polygon`'s bounds against individual
  object footprints, not a coordinate bug). So on this fixture the >50% rule
  drops a large batch of real furniture that merely lives in a *different,
  disconnected* module - not reflections at all: 28 objects (`bed_0`,
  `pillow_0`, `pillow_1`, `blanket_0`, `blanket_1`, `rug_2`, `chair_4`
  through `chair_11`, `chair_13` through `chair_15`, `table_1` through
  `table_6`, `table_9`, `bench_0`, `bench_1`, `towel_0`, `toilet_2`), leaving
  30 of the original 58 post-small-filter objects. None of the remaining
  objects were partially-outside/clipped on this fixture (every survivor was
  either fully inside `floor_polygon`, or the one genuinely wall-backfilled
  and reflection-guarded case landed on the "drop" side rather than a
  partial-clip case) - the clip path is exercised only by the synthetic unit
  tests in `tests/test_msa_bootstrap.py::TestOutsideRoomFilter`, not this
  fixture. This is flagged as a design tension worth the orchestrator's
  attention (whether `floor_polygon`/`extract_floor_polygon` should union all
  sufficiently-large free-space components instead of only the largest) but
  is out of scope for this change, which implements the >50%/refit rule
  exactly as specified against the existing `floor_polygon` definition.

  Despite the large object-count impact, the `golden_gaps.json` delta is
  small: gap count dropped from 47 to 42 (regenerated and reverified against
  a fresh `run_bootstrap` pass) - only 5 pairs disappeared, all involving the
  now-dropped `bed_0` or `blanket_0`: `bed_0`-`blanket_0` (0.1207m),
  `bed_0`-`rug_0` (0.9217m), `bed_0`-`rug_1` (0.6325m), `bed_0`-`wall_0`
  (0.10m), `blanket_0`-`wall_0` (0.10m). No new pairs appeared, and all 42
  pairs common to both golden files are unchanged (<1cm) - the other 27
  dropped objects apparently weren't anyone's nearest-obstacle pair in
  `gaps.json` to begin with, so their removal didn't otherwise perturb gap
  computation.

All four files together are ~144KB, well under the 1MB/file budget.

**2026-09-06 delta (yaw normalization post-step):** `run_bootstrap` gained a
post-step (`scripts/msa/geometry.py`'s `compute_dominant_wall_yaw` +
`scripts/msa/bootstrap.py`'s `_yaw_rotation_center`/`_rotate_*` helpers) that
histograms every wall-polygon edge's direction mod 90 degrees (2-degree bins,
length-weighted) and, if one bin holds >= 60% of the total weighted length,
rotates the whole scene (walls, floor_polygon, objects incl. `hull_xz`, gaps,
and - best-effort - camera poses from `cameras_aligned.json` if present) about
the floor-polygon centroid to axis-align that dominant direction. This exists
because the GPU pipeline never corrects the room's yaw about the vertical
axis, so MSA's occupancy grid comes in at an arbitrary rotation that varies
run to run.

**This fixture does not qualify, and that's the correct, expected outcome for
it:** `extract_wall_polygons` finds 4 wall components on this fixture, and
`compute_dominant_wall_yaw`'s peak bin captures only ~32% of the total
weighted edge length (well under the 60% confidence gate) - consistent with
the "modular_home" scene's free space being genuinely split across several
disconnected modules/rooms at different orientations, already documented
above in the reflection/through-window-guard delta. `run_bootstrap`'s report
and `scene_meta.json` both come back with `yaw_applied: false` and
`yaw_correction_rad: 0.0` on this fixture - **the golden fixture files
(`golden_gaps.json`, `occupancy.npy`, `occupancy_meta.json`,
`scene_meta.json`, `scene_objects_hulls.json`) are all unchanged by this
delta**, since normalization is a no-op whenever it doesn't apply. The
detection algorithm's positive case (clearly axis-misaligned walls) and
negative case (no dominant direction, e.g. an octagonal room) are both
covered by synthetic unit tests in `tests/test_msa_yaw_normalization.py`
instead, since this fixture happens to only exercise the "correctly declines
to act" path.

**2026-09-06 delta (T3' rework - polygon-based yaw is now primary, no
confidence gate):** the yaw-normalization post-step's primary method changed
from `compute_dominant_wall_yaw`'s wall-segment-angle histogram (described
immediately above) to `scripts/msa/geometry.py`'s new
`compute_room_polygon_yaw`: it fits a minimum-area bounding rectangle (via
the same `oriented_min_area_rect` rotating-calipers routine already used for
every object footprint) to `floor_polygon` itself, and rotates the whole
scene so that rectangle's *long* side lands on +X. Unlike the histogram
method, there is no confidence gate - a bounding rectangle always has a
well-defined long axis except in the explicit degenerate near-square/
near-circular case (not applicable here). `compute_dominant_wall_yaw` is kept
and still computed on every run, but now only as a secondary/diagnostic
cross-check logged in `scene_meta.json`/`report.json` alongside the
polygon-derived result - it no longer decides whether normalization is
applied.

**This fixture is exactly the case this rework targets:** as documented
immediately above, `compute_dominant_wall_yaw`'s peak bin only ever captured
~32% of the total weighted wall-edge length here (well under the old 60%
gate), so the previous (histogram-gated) implementation left this fixture
completely unrotated. The new primary method has no such gate: `floor_polygon`
(the largest connected free-space component, unaffected by this delta)
bounds to a `7.95m x 4.65m` rectangle whose long side sits at
`long_axis_angle_deg=-90.0` (i.e. already axis-aligned, just onto Z instead of
X) - `compute_room_polygon_yaw` reports `correction_deg=90.0`,
`is_degenerate_square=False`, and `run_bootstrap` actually applies a clean
90-degree rotation this time (`yaw_applied: true`, vs. `false` before). The
secondary histogram cross-check still reports "no clear direction" on this
fixture (`yaw_histogram_gate_passed: false`, `yaw_histogram_correction_deg:
null`) - visible confirmation, side by side in `scene_meta.json`, that the two
methods can legitimately disagree on whether *they* have a confident answer,
even when the polygon method is able to act anyway.

`golden_gaps.json` **was regenerated** (unlike the "no change" outcome from
the previous delta): a rigid 90-degree rotation about the floor-polygon
centroid preserves every distance, so the gap *pair set* (42 pairs, same as
before) and every `width_m` value are byte-for-byte identical to the prior
golden file - but every `measurement_point_xy` moved (a clean 90-degree
rotation swaps X/Z-ish components about the rotation center), so the file's
raw JSON content differs in all 42 entries even though nothing the existing
`TestGapsMatchGolden` test actually asserts on (`width_m`, the pair set)
changed. Re-verified against a fresh `run_bootstrap` pass:
`tests/test_msa_bootstrap.py` passes unmodified against the regenerated file.
`occupancy.npy`, `occupancy_meta.json`, `scene_meta.json`, and
`scene_objects_hulls.json` are all unchanged by this delta (the rotation is
purely a `run_bootstrap` post-step over these inputs, not a change to any of
them).

**2026-09-06 delta (T3'' rework - yaw source switched from `floor_polygon`
(FREE+UNKNOWN) to a new FREE-only polygon, cross-checked against the
wall-vertex convex hull):** the previous delta above already flagged the
concern later confirmed on the hero scene (`own_0901_173903`): `floor_polygon`
is built from the largest connected FREE+**UNKNOWN** component, and
`gpu/stage_occupancy.py` builds the occupancy grid as a percentile-trimmed
axis-aligned bounding box, so UNKNOWN cells always touch the grid array's own
edges. Wherever the room's FREE interior connects to that UNKNOWN halo through
a doorway/gap, `floor_polygon`'s bounding rectangle silently measures the
*grid array's own extent* instead of the true room shape - confirmed directly
on the hero scene, where the previous method's result exactly equalled
`occupancy_meta.json`'s `width*resolution x height*resolution`.
`scripts/msa/geometry.py`'s new `extract_free_only_room_polygon` (FREE cells
only, closed with `scipy.ndimage.binary_closing` before component selection
to bridge sub-2-cell gaps) is now the primary yaw source, cross-checked
against the independent `compute_wall_hull_yaw` (bounding rect of the pooled
wall-vertex convex hull, which never touches the occupancy grid at all) -
`bootstrap._reconcile_yaw_estimates` falls back to the wall-hull angle if the
two disagree by more than 10 degrees.

**This fixture's applied correction did not change**, and required no
`golden_gaps.json` regeneration - a byte-for-byte diff of a fresh
`run_bootstrap` run's `gaps.json` against the existing golden file is empty.
This is a meaningfully different (and reassuring) outcome from "nothing
changed because the code path wasn't exercised": `run_bootstrap`'s report now
shows `yaw_free_only_long_axis_angle_deg: -90.0` from a `7.75m x 4.45m`
free-only bounding rect - close to, but *not* exactly, the grid's own
`7.95m x 4.65m` extent (`159 x 93` cells at 5cm resolution) that the old
buggy method matched exactly - and the independent wall-hull cross-check
reports `-84.29 degrees`, a `5.71`-degree disagreement, comfortably inside the
10-degree fallback threshold. Two independent signals (one from the occupancy
grid's free space, one purely from measured wall vertices) agreeing on "close
to axis-aligned onto Z, needs ~90 degrees of correction" is real corroborating
evidence that this fixture's room genuinely is oriented that way - not a
restatement of the old bug's coincidental exact-grid-extent match. The
old delta's speculation ("this fixture's earlier 90-degree correction was the
same grid-extent artifact, not real signal") is superseded: the fixture's
90-degree correction was actually correct, it was just derived from the wrong
evidence before. `occupancy.npy`, `occupancy_meta.json`, `scene_meta.json`,
`scene_objects_hulls.json`, and `golden_gaps.json` are all unchanged by this
delta.

**2026-09-06 delta (T15a - one room polygon for everything: largest FREE
component union floor-standing object footprints, closed, outer contour):**
`run_bootstrap` no longer uses `extract_floor_polygon` (FREE+UNKNOWN) as the
room reference, nor the yaw-only `extract_free_only_room_polygon`. The new
`scripts/msa/geometry.py::extract_room_polygon` builds union(largest
8-connected FREE component, rasterized footprints of every object whose
`bbox_min_y - floor_y < 0.15m`), closes it (3x3, 2 iterations), keeps the
component containing the FREE region with holes filled, traces the outer
contour and DP-simplifies it. That single `room_polygon` now feeds yaw
estimation, `_drop_or_clip_objects_outside_room`, the floor plate, the USD/GLB
floor mesh and the exported `floor_polygon`; `scene_meta.json` additionally
carries `room_polygon` ([x, z] pairs, already rotated) and
`room_polygon_method`.

**`golden_gaps.json` was regenerated** (from a fresh `run_bootstrap` pass over
this fixture, copied verbatim). Delta against the T3'' golden:
- The outside-room drop list grew from 28 to 31 objects: `mirror_0`
  (`bbox_min_y=1.09`), `sink_2` (`0.41`) and `towel_2` (`0.71`) are now
  dropped. All three are *elevated* fixtures in a bathroom module whose floor
  is not part of the largest FREE component; the old FREE+UNKNOWN
  `floor_polygon` leaked across the UNKNOWN halo far enough to contain them,
  the new polygon does not. No previously-dropped object came back: the 28
  earlier drops are floor-standing furniture in *disconnected* modules (their
  footprints never touch the main FREE component, so `build_room_mask` leaves
  them out of the room and the >50%-outside rule still drops them - the
  design tension flagged in the room-clipping delta above is unchanged).
  `n_objects_kept` 30 -> 27; 29 of the 58 post-small-filter objects were
  classed floor-standing (`room_polygon_floor_standing_object_ids`).
- Gap count 42 -> 36: 7 pairs disappeared, every one involving a newly-dropped
  object (`mirror_0`-`sink_2`, `mirror_0`-`wall_1`, `sink_2`-`towel_2`,
  `sink_2`-`wall_0`, `sink_2`-`wall_1`, `toilet_0`-`towel_2`,
  `towel_2`-`wall_0`), and 1 pair appeared where `towel_2` used to be
  `toilet_0`'s nearest obstacle (`toilet_0`-`wall_1`). All 35 pairs common to
  both golden files have `width_m` unchanged (<1cm).
- Applied yaw correction changed from exactly `90.0` to `84.49` degrees: the
  room polygon's min-area rect (`7.70m x 4.30m`) sits at `-84.49`, and the
  independent wall-hull cross-check (`8.02m x 4.28m`) at `-84.29` - a `0.21`
  degree disagreement, down from `5.71` under T3''. The old exact `-90.0` was
  the rasterized FREE blob's axis-aligned staircase edge; the furniture
  footprints now unioned in carry the room's actual slight skew.
`occupancy.npy`, `occupancy_meta.json`, `scene_meta.json` and
`scene_objects_hulls.json` are unchanged by this delta.
`tests/test_msa_validate.py::test_recompute_gaps_matches_golden_for_unchanged_objects`
was already failing before T15a and still fails after the regeneration
(same test, not introduced here).

**2026-09-07 delta (morning-1 - the wall-hull angle is the applied yaw):**
`bootstrap._reconcile_yaw_estimates` now APPLIES `compute_wall_hull_yaw`'s angle to
the whole scene and treats the room polygon's own min-area-rect angle as the
fallback (no walls / degenerate hull) and otherwise a diagnostic; the T15g
Manhattan snap happens in the applied (wall-hull) frame. On this fixture the
applied correction moves `3.33` -> **`5.18`** deg (`yaw_method` =
`wall_hull_min_area_rect`, polygon `-3.33` logged as `yaw_free_only_*`,
disagreement `1.84` deg). `golden_gaps.json` **was regenerated**: the pair set is
identical (52 pairs) and no shared pair's `width_m` moved by > 0.1 mm (a rigid
rotation preserves widths), but every `measurement_point_xy` is rotated by the
extra 1.84 deg about the room centroid (max move 0.74 m, mean 0.07 m) and the
snapped outline is now axis-aligned in the wall-hull frame (still 20 vertices).
Objects kept 51 -> 53 (`refrigerator_1` and `toilet_1` are no longer > 50 %
outside the re-snapped outline; `table_4` and the other 6 T15b drops still
drop), walls 3 (unchanged),
`floor_standing_inside_fraction_min` `0.915` (see the note below on why the
>= 0.95 assertion is not made on this fixture). New meta/report keys
`yaw_applied_source` ("wall_hull" | "room_polygon"), `yaw_fallback_to_polygon`;
`yaw_fallback_to_wall_hull` removed. `occupancy.npy`, `occupancy_meta.json`,
`scene_meta.json`, `scene_objects_hulls.json` unchanged.

**2026-09-07 delta (T15g - Manhattan room outline, grid padding, one wall band):**
`run_bootstrap` now (a) pads the occupancy grid with UNKNOWN so every
floor-standing footprint fits before the room mask is built (here `[0, 0, 2, 4]`
cells - two footprints reached 10-20 cm past the grid's z edges), (b) estimates
the yaw ANGLE from the closed polygon, snaps the outline to 0/90 degree edges in
that frame (`geometry.regularize_room_polygon`, notch 0.30 m: 86 raw vertices ->
20, every interior angle 90/270), (c) runs the outside-room drop/clip against the
SNAPPED polygon, and (d) applies the rotation to everything last. The applied
yaw is unchanged (`3.33` deg - the angle comes from the unsnapped polygon, as
before). `golden_gaps.json` **was regenerated**: 53 -> 52 gaps, one pair gone
(`table_4`-`wall_2`) because `table_4` (a 0.01 m^2 sliver, 66% outside under
T15b and clipped) is now 100% outside the snapped outline and dropped; no other
pair changed and no shared pair's `width_m` moved by > 1cm. Objects kept 52 ->
51. `floor_standing_inside_fraction_min` on this fixture is `0.923` (`broom_0`, a
small object that straddles the room edge, is clipped and its refit rectangle
pokes back out) - the >= 0.95 assertion is made on the hero and the synthetic
room in `tests/test_msa_room_regularize.py`, not on this multi-module fixture.
The GLB/USD now carry one `wall_outline` band (0.10 m thick, floor -> ceiling)
instead of `wall_i` meshes; `Plan/wall_i` curves and gaps.json obstacle sources
still list the 3 raw fragments. `occupancy.npy`, `occupancy_meta.json`,
`scene_meta.json`, `scene_objects_hulls.json` unchanged.

  **2026-09-06 delta (T15b: grid-frame fix, prim drop/clip, wall tracer):**
  `golden_gaps.json` regenerated from a fresh `run_bootstrap` pass. This is the
  largest delta so far because `scripts/msa/` had been reading `occupancy.npy`
  **transposed** (row -> z, col -> x, while the file is `[ix, iz]` - this
  fixture's grid is 159 x 93 = `width x height`): every wall polygon and the
  room polygon were reflected across the x = z diagonal relative to the object
  footprints, which are computed from world-frame hulls and were never
  transposed. With walls and footprints finally in one frame:
  - Applied yaw correction `84.49` -> `3.33` degrees (room polygon min-area
    rect `-3.33`, wall-hull cross-check `-5.18`, disagreement `1.84`, was
    `0.21`). Consistent with the old value being the transposed angle
    (reflection maps an angle a to 90 - a: 90 - 84.49 = 5.51).
  - `n_objects_kept` 27 -> 52, outside-room drops 31 -> 8 (`bed_1`,
    `refrigerator_1`, `mirror_0`, `sink_2`, `towel_1`, `towel_2`, `toilet_1`,
    `toilet_3`): the room polygon now actually sits over the furniture, so the
    28 "disconnected module" drops T15a documented were mostly the transposed
    polygon missing them. The bathroom/laundry items still drop (they are in
    another room of this multi-room home, beyond the room polygon).
  - Walls: 4 -> 3. `wall_3` (3.06 m^2 of cells) has 63.2% of its area more
    than `WALL_CLIP_MARGIN_M` = 0.20 m outside the room polygon (it belongs to
    a neighbouring module) and is dropped by the new prim rule
    (`report.json` `dropped_prims`); the surviving walls are renumbered
    `wall_0..wall_2` in both the exports and gaps.json, in step.
  - Gaps 36 -> 53: 22 pairs disappeared (every one involves the dropped
    `wall_3`, a dropped object, or a wall whose polygon moved), 39 appeared
    (mostly furniture-to-wall pairs that only exist now that walls are on the
    correct side of the furniture, e.g. `bed_0`-`wall_1`, `counter_0`-`wall_1`,
    `rug_2`-`wall_0`). Of the 14 pairs common to both files, 5 changed by more
    than 1 cm (`refrigerator_0`-`wall_0` 0.10 -> 0.1207, `rug_0`-`stool_1`
    0.4183 -> 0.3803, `rug_0`-`table_0` 0.1207 -> 0.20, `rug_1`-`stool_1`
    0.5193 -> 0.3803, `stool_1`-`wall_0` 0.3606 -> 0.3864) - wall-backfill now
    extends footprints toward the correctly placed walls.
  - No wall on this fixture needed the T15b tracer fix (none is pinch-connected
    into a degenerate ring). The doorway-spill opening (0.45 m radius) does act
    here: without it the room polygon's min-area rect is `7.63 x 4.22 m` at
    `-9.71` degrees (a spill through an opening into the next module), with it
    `7.11 x 4.25 m` at `-3.33` - the applied `3.33` above - which is what brings
    the polygon to within 1.84 degrees of the wall-hull cross-check.
`occupancy.npy`, `occupancy_meta.json`, `scene_meta.json` and
`scene_objects_hulls.json` are unchanged by this delta.

  **2026-09-07 delta (M7 de-clutter: duplicate merge -> support rule -> class
  caps):** `run_bootstrap`'s stage order is now footprints (+ backfill) ->
  `dedupe.merge_duplicates` -> small-object filter -> outside-room drop/clip ->
  `support_rule.apply_support_rule` -> `class_caps.apply_class_caps` -> gaps ->
  yaw -> exports (`report.json["stage_counts"]`). On this fixture: 63 footprints
  -> **40** after the merge (9 clusters: `chair_1`+2, `chair_4`+3, `chair_7`+5 (IoU
  0.32), `table_0`+2, `table_1`+1, `table_2`+4, `bench_1`+`bench_0` (IoU 0.40),
  `stool_1`+3, `coffee maker_1`+2 - the merged object keeps the highest-support
  member's id and the union hull) -> 37 (small filter) -> 31 (outside-room) ->
  **28** after the support rule (`pillow_1`, `blanket_1`, `towel_0` are
  floating: their `bbox_min_y` 0.52-0.54 m sits 0.18-0.20 m above `bed_0`'s bbox
  top, beyond the 0.10 m stacking tolerance, and they touch neither floor nor
  wall; kept: 19 floor / 8 wall / 1 stacked) -> **26** after the class caps
  (`chair` 6 -> 4: `chair_9`, `chair_12` ranked 5/6 - this hull-only fixture has
  no `point_count`, so support is the hull-area x height fallback). `golden_gaps.json`
  regenerated: 52 -> **51** pairs, 41 common; 11 pairs gone (every one names a
  merged-away member - `bench_0`, `chair_5`, `chair_9`/`chair_11`, `stool_0`,
  `table_2`'s old partner - or a capped chair), 10 appeared (the merged objects'
  larger hulls are now the nearest obstacle: `bench_1`-`chair_7`, `bench_1`-`table_2`,
  `chair_0`-`stool_1`, `table_1`-`wall_2`, ...); 5 common pairs moved by more than
  1 cm, all against a merged hull (`rug_0`-`stool_1` 0.3803 -> 0.505,
  `rug_1`-`stool_1` 0.5193 -> 0.505, `bed_0`-`stool_1` 0.8303 -> 0.8772,
  `blanket_0`-`stool_1` 0.6708 -> 0.7413, `bed_0`-`chair_4` 3.3666 -> 3.3845).
  Objects kept 51 -> 26, walls 3, applied yaw 5.18 unchanged. The other fixture
  files are unchanged.
