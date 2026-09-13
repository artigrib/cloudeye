# Frame rate and what it buys: MapAnything pass A, nvblox pass B

Six view counts from one video, measured end to end. Pass A was run on a B200 on 2026-09-08;
pass B was run on an RTX 4090 on 2026-09-13 from the stored pass-A bundles, so no pass A was
repeated for this study.

## Method

**Input.** One video: 2295 frames, native 29.98897 fps, 76.53 s. One phone, one room.

**Rungs.** For five of six the rung is an integer decimation step over native fps and the
view count is derived; T1 lists views, fps and step per point.

All six rungs are the same video file, confirmed through the scene's video row.

Three sampling rules, not one. The 74-view scene asked for 2 fps keyframes and then let the
product extractor's blur and similarity filters run, which is why 76.53 s at 2 fps yields 74
views and not 153: those filters drop 6-77 % depending on content. The 327 rung came from a
motion heuristic. Only 574-2295 are integer decimation. Every figure caption repeats this,
because these two rungs sit at the left edge of every curve.

**Pass B.** `pipeline/run_pipeline.py` per point, `--from-pulled` the stored bundle,
`--nvblox-source live`, one 4090. Direct driver runs: artifacts only, no database rows.
Identical parameters across all five.

**Colour is off for all five.** `PACK_RGB` re-derives RGB with an ffmpeg modulo `select`,
which reproduces a frame list only when it is an arithmetic run; on the motion-sampled 327
rung it refuses rather than pair one frame's colour with another frame's depth. Four coloured
and one not would break identical parameters, so colour is off everywhere. It changes vertex
colour and transfer volume, not occupancy, components or routes.

**The 74-view point is taken as it stands** and was not re-run. Same band (0.1-1.5 m), same
0.05 m resolution, same `min_points_per_cell` as the five pass-B runs. It differs in one way
that matters, stated in Limitations: its grid is in the aligned product frame.

**Fixed parameters**, held across every point:

| parameter | value | source |
|---|---|---|
| nvblox TSDF voxel | 0.03 m | `pipeline/steps/nvblox_scenes.py:731` |
| band | 0.1-1.5 m | driver default, recorded per run |
| occupancy resolution | 0.05 m | `gpu/job_io.py:30` |
| `min_points_per_cell` | 3 | `gpu/stage_occupancy.py:44` |
| UNKNOWN cost multiplier | 1.5x | `app/services/pathfinding.py:35` |
| robot registry | 8 platforms, radii 0.100-0.5528 m | `app/robots.py` |

**Fixed target.** One app-frame coordinate on the 74-view scene: the `desk_1` point cluster
centroid, x 2.5926, y 0.7326, z -2.2641, cell **(81, 76)**.

**Corrected study-design error.** The first version of this study used a different cell, (81,
78), picked as the nearest FREE cell to that centroid in the scene's **0.1-0.5 m** occupancy
grid (`occupancy.npy` at the scene root) - the grid the map page draws. Every route in this
study is planned on the **0.1-1.5 m** band layer (`layers/backfill/`), and the two grids
disagree about both cells: (81, 78) is FREE in the 0.1-0.5 m grid and OBSTACLE in the band
layer at `obstacle_min_height` 0.1683 m, while the centroid's own cell (81, 76) is OBSTACLE in
the 0.1-0.5 m grid and FREE in the band layer. The target was therefore chosen on a grid no
route was ever planned on, and no view count and no radius could have reached it. **The target
is now chosen on the band layer**, which makes the nearest FREE cell the centroid's own cell,
0.000 m away. The route measurement that follows from it is [T3-74](#t3-74-route-to-the-desk-on-the-74-view-point);
what the old choice produced is kept in the appendix.

- Start: the 74-view scene's recorded robot start, x 0.00096, z -0.00119, cell (29, 121). This
  is exactly that scene's first camera - `cameras_aligned.json[0]` agrees in XZ to 1e-9.
- Both cells are in the 74-view grid, and only the 74-view grid. The desk exists as an object
  on that scene alone, and the five pass-B points are in unregistered frames, so this
  coordinate is not carried to them; see the appendix for what happens when it is.

**No semantics.** No object detection, vocabulary stage or VLM. Objects exist only on the
74-view scene, so no reach-by-object number is reported anywhere else.

## Results

### Observation

![views vs coverage_raw](img/fps_coverage_raw.png)

`coverage_raw` is a relative metric: it sizes its grid from each run's own point-cloud bbox,
a fixed K, and rewards spread. The percentage peaks at 765 views and falls; the count of
5 cm cells holding data rises the whole way. Only the four integer-decimation rungs carry
this measurement.

The tri-state grid from this study's own pass-B runs moves the other way and keeps moving:
unobserved falls from 42.95 % at 74 views to 19.21 % at 2295, non-monotone in the middle where
327, 574 and 765 cluster at 30-32 %. The gain from 4 fps to 10 fps is inside this scene's
noise; the step to every frame is not.

### T1 tri-state and unobserved

| views | fps | step | grid | FREE | OBSTACLE | UNKNOWN | unobserved % | grid-origin offset vs hero-74 (m) |
|---:|---:|---:|---|---:|---:|---:|---:|---|
| 74 | 2.0 | - | 111x142 | 6082 | 2910 | 6770 | 42.95 | (+0.000, +0.000) |
| 327 | 4.27 | - | 75x139 | 5436 | 1533 | 3456 | 30.17 | (-0.706, +2.420) |
| 574 | 7.50 | 4 | 76x143 | 5822 | 1355 | 3691 | 31.16 | (-0.741, +2.068) |
| 765 | 10.00 | 3 | 75x159 | 5898 | 1819 | 4208 | 32.34 | (-0.718, +1.304) |
| 1148 | 14.99 | 2 | 75x141 | 5979 | 1517 | 3079 | 26.32 | (-0.433, +3.208) |
| 2295 | 29.99 | 1 | 76x160 | 7472 | 1900 | 2788 | 19.21 | (-0.677, +2.716) |

### Drivable space

![views vs largest drivable component](img/fps_drivable_component.png)

The largest drivable component grows with views for the six smallest platforms. The measured
74-to-2295 ratios are 2.09x (burger), 3.43x (waffle_pi), 4.51x (turtlebot4), 4.38x (limo),
3.57x (rosbot_xl_arm) and 4.34x (go2) - a range of **2.1x to 4.5x**, not a doubling. The two
largest platforms barely move: jackal 1.63x, husky 1.17x. Two readings the table does not give
on its own:

- **Husky never gets a usable component.** At radius 0.5528 m the best rung produces 0.27 m²;
  three rungs produce 0.01 m² or less. Frames do not open a room too narrow for the robot.
- **Component growth is not reachable growth.** How much of a component a robot can actually
  use depends on where it starts and where it is going, and neither survives the move between
  rungs: the six points are in six unregistered frames. Everything start- or target-dependent
  is in [the appendix](#appendix-fixed-target-routes-attempted-frames-not-registered), out of
  Results. What stays here is the component size itself, which each point measures in its own
  frame and which therefore compares.

### T2 largest drivable component per robot, m2

| robot | radius m | 74 | 327 | 574 | 765 | 1148 | 2295 |
|---|---:|---:|---:|---:|---:|---:|---:|
| burger | 0.1000 | 6.49 | 9.44 | 10.11 | 9.96 | 9.71 | 13.54 |
| waffle_pi | 0.1500 | 3.53 | 7.92 | 8.13 | 6.59 | 8.48 | 12.11 |
| turtlebot4 | 0.1750 | 1.60 | 3.99 | 4.68 | 4.56 | 4.68 | 7.20 |
| limo | 0.1940 | 1.69 | 4.15 | 4.72 | 3.82 | 5.41 | 7.41 |
| rosbot_xl_arm | 0.2186 | 1.84 | 4.88 | 4.22 | 3.89 | 5.19 | 6.58 |
| go2 | 0.2496 | 0.83 | 2.97 | 3.04 | 2.86 | 2.36 | 3.61 |
| jackal | 0.3340 | 0.92 | 1.70 | 1.86 | 1.47 | 1.15 | 1.49 |
| husky | 0.5528 | 0.23 | 0.01 | 0.08 | 0.01 | 0.00 | 0.27 |

Largest connected component of cells that are observed and at least one radius clear of an
obstacle at that robot's nav height. It is a property of one grid in one frame, so it compares
across rungs without any registration between them. The column that used to sit beside it -
the component containing the start - depended on placing one frame's coordinate in another's
grid and has moved to the appendix.

### Peak VRAM

![views vs peak VRAM](img/fps_peak_vram.png)

Five pass-A runs, residuals against `4704.6 + 67.083*n` MiB of +0.26 to +1.21 MiB. The points
are collinear because the allocator sizes the buffer from `n_views`; the line predicts
allocation, not demand, and a run cannot fall off it. Its one operational use is the ceiling:
2295 views allocates 158 661 MiB, which fits a 183 359 MiB B200 and no smaller card here.

### Cost and wall time

Pass B for all five rungs cost **$0.1487** on one RTX 4090 at $0.3556/hr.

**"NVBLOX scales with transfer, not compute" was a hypothesis, and the bytes do not support
it.** NVBLOX goes 171.8 s at 327 views to 304.2 s at 2295, a 1.77x rise across a 7x rise in
views. Over the same span the bytes it moves rise 6.30x. If transfer set the time the two
would rise together. They do not, and seconds per pushed MiB falls fourfold across the sweep.

### T4 pass-B wall time and cost

| views | PACK s | NVBLOX s | LAYERS s | REACH s | total s | cost USD |
|---:|---:|---:|---:|---:|---:|---:|
| 327 | 6.61 | 171.79 | 19.38 | 3.68 | 285.91 | 0.0274 |
| 574 | 15.06 | 182.3 | 8.65 | 3.51 | 279.0 | 0.0254 |
| 765 | 32.23 | 210.11 | 7.95 | 3.53 | 322.64 | 0.0281 |
| 1148 | 26.19 | 219.41 | 8.13 | 3.21 | 323.04 | 0.0287 |
| 2295 | 58.26 | 304.23 | 7.88 | 3.31 | 459.94 | 0.0391 |

### T5 nvblox payload per rung

| views | push MiB | pull MiB | total MiB | total / 327 | NVBLOX s | NVBLOX s / 327 | s per pushed MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 327 | 95.3 | 13.85 | 109.1 | 1.00x | 171.79 | 1.00x | 1.803 |
| 574 | 167.3 | 14.65 | 181.9 | 1.67x | 182.30 | 1.06x | 1.090 |
| 765 | 222.9 | 15.45 | 238.4 | 2.18x | 210.11 | 1.22x | 0.943 |
| 1148 | 334.5 | 16.06 | 350.6 | 3.21x | 219.41 | 1.28x | 0.656 |
| 2295 | 668.7 | 18.41 | 687.1 | 6.30x | 304.23 | 1.77x | 0.455 |

Push is the size of `packed/<scene>/` - `depth_u16.npy` plus `meta.json` - and pull the size of
`nvblox/`; both are the on-disk artifacts, measured after the run. The driver's own logged
`NVBLOX: transfer estimate` line is deliberately not used: at 2295 views it predicted 406.2 s
of push against a 304.23 s stage total, so it is a probe-rate extrapolation, not a measurement.

A least-squares fit over the five points gives `NVBLOX s = 145.9 + 0.2285 x total MiB`, with
residuals of -6.6 to +9.7 s. Read that as a description, not a mechanism: on five points it
cannot separate a fixed per-job cost from anything else flat in views. What it does say is
that about 146 s of the 304 s at 2295 views - 48 % - moves with neither bytes nor views. PACK
is the stage that does track bytes: `2.9 + 0.0832 x pushed MiB`, residuals -4.5 to +10.8 s.
LAYERS and REACH stay flat.

## Limitations

1. **One video, one phone, one room**, a single 76.5 s walkthrough. Nothing separates what
   frame rate does from what this room does.
2. **Three sampling rules.** 74 = 2 fps keyframes then blur/similarity filtering, 327 =
   motion heuristic, 574-2295 = integer decimation. The two leftmost points on every curve
   were chosen differently from the other four; read the curves as series sharing an axis.
3. **The frames are not one frame.** The five pass-B points sit in their own pass-A frames,
   each with its own MapAnything rotation, scale and drift; the 74-view point is in the aligned
   product frame. Nothing in Results crosses between them - every metric there is measured
   inside one grid - which is why Results survives the gap and the fixed-target work does not.
   Registering the six clouds is the missing step, and it is recorded as the appendix's own
   conclusion rather than as a caveat on numbers that do not depend on it.
4. **Pass A was not repeated.** Bundles were produced on 2026-09-08 across two sessions and
   two boxes; pass-A wall time and peak VRAM are read from those runs.
5. **`coverage_raw` sizes its own grid.** The percentage is not comparable across rungs; the
   cell count is. Both are plotted for that reason.
6. **One pass B per point**, so no variance estimate. Two runs of the 1148 rung on
   2026-09-10 produced identical mesh triangle counts on different boxes, which bounds
   run-to-run variation for that rung only.
7. **UNKNOWN is traversable at 1.5x, not blocked.** A component can grow because cells became
   FREE or because they became UNKNOWN. Read the unobserved fraction beside every component.

## What this does not show

- **No lighting or texture ablation.** One capture, one exposure. "More views help" and
  "these particular frames help" are not separated.
- **Confidence masks are unused.** Per-view `conf` is stored raw and unthresholded and no
  stage here filters on it; the 327 and 2295 bundles do not carry `conf` at all.
- **No semantics outside the 74-view point.** Object counts and every by-object route belong
  to that scene only.
- **No pass-A quality metric.** Nothing here re-derives reconstruction accuracy against
  ground truth.
- **No claim about a knee.** The saturation threshold that picked 1148 views elsewhere is an
  operator stopping rule, not a measurement, and is not tested here.
- **No cross-rung route comparison.** Route length, tightest gap and reachability are reported
  for the 74-view point only. Comparing them across rungs needs the six clouds registered to a
  common frame - rigid plus scale - and that registration does not exist yet. Until it does,
  no route number here may be read as a function of view count.

## Appendix: Fixed-target routes: attempted, frames not registered

The five pass-B points sit in independent MapAnything frames that differ from the 74-view
scene and from each other by rotation and scale as well as translation, so shifting a
coordinate by the difference of two grid origins is not a transform between them. Registering
the clouds to one frame - rigid plus scale, the way the viewer's mesh mirror was settled by
scoring candidate transforms against the cloud actually drawn rather than against a bounding
box (`pipeline/steps/ingest_scene.py:73-79`) - is the prerequisite for any cross-rung route
number, and it was not done.

What follows is therefore kept out of Results: the one route measurement that needs no
registration because it never leaves the 74-view grid (T3-74), and the attempt that did leave
it, reported as measured so the failure is legible.

### T3-74 route to the desk on the 74-view point

Target: the nearest FREE cell to the `desk_1` centroid **in the 0.1-1.5 m band layer**, the
layer routes are planned on. That is cell **(81, 76)**, world x 2.6043, z -2.2714 - the
centroid's own cell, 0.000 m away. Start is cell (29, 121). One frame, one grid, no
translation of any kind; this is the study's only clean fixed-target measurement.

| robot | radius m | nav height m | clearance at start m | clearance at target m | reaches target | route m | tightest gap m | nearest cell free for this radius |
|---|---:|---:|---:|---:|---|---|---|---|
| burger | 0.1000 | 0.1920 | 0.250 | 0.050 | no | - | - | (79, 80), 0.224 m from the target |
| waffle_pi | 0.1500 | 0.1410 | 0.250 | 0.050 | no | - | - | (78, 81), 0.292 m |
| turtlebot4 | 0.1750 | 0.3467 | 0.150 | 0.000 | no | - | - | (77, 82), 0.361 m |
| limo | 0.1940 | 0.2514 | 0.150 | 0.050 | no | - | - | (77, 82), 0.361 m |
| rosbot_xl_arm | 0.2186 | 0.1320 | 0.250 | 0.050 | no | - | - | (77, 83), 0.403 m |
| go2 | 0.2496 | 0.4000 | 0.150 | 0.000 | no | - | - | (77, 84), 0.447 m |
| jackal | 0.3340 | 0.2490 | 0.150 | 0.000 | no | - | - | (78, 87), 0.570 m |
| husky | 0.5528 | 0.3963 | 0.150 | 0.000 | no | - | - | (102, 62), 1.262 m |

**0/8, and this time the reason is measured rather than an artefact.** The target cell has
0.000-0.050 m of clearance for every robot, against a smallest radius of 0.100 m: it is a cell
on the desk's own footprint, and a robot cannot stand on the desk. Route length and tightest
gap are consequently undefined for all eight, so those columns stay empty.

Five of the eight also fail earlier, at the start: the recorded robot start has 0.150 m of
clearance for `turtlebot4`, `limo`, `go2`, `jackal` and `husky`, under each of their radii.
Clearance is per robot because a cell counts as an obstacle only if its lowest obstacle surface
sits at or below that robot's nav height plus the 0.05 m margin - which is why
`rosbot_xl_arm`, at radius 0.2186 m but only 0.132 m tall, has a free start where the 0.175 m
`turtlebot4` does not.

The last column is the diagnostic the empty route columns cannot give: how far from the desk
each robot would have to stop. Between 0.224 m and 0.570 m for seven of the eight, 1.262 m for
Husky. **Choosing a target a robot can occupy - a stand-off pose beside the desk rather than a
cell on it - is what would make this table carry routes.** That is a further design change and
has not been made; the target here is the one the instruction specifies, the nearest FREE cell
in the band layer.

### The attempt to carry the target to the other five rungs

Everything below was computed before the frames question was settled. It is kept because it is
the evidence for the appendix's first paragraph: it shows what a grid-origin shift does and
does not buy, with numbers.

#### The alignment that was attempted

The first version of this document printed a grid-origin offset per point and then indexed the
74-view world coordinate straight into each point's grid without using it. This section applies
it and recomputes the start column and T3 from the stored layers. No GPU work was
repeated; every number below comes from `layers/<scene>/` and `cameras_aligned.json` of runs
already on disk.

**Sign convention.** Write `o_p` for a point's grid origin and `o_h` for the 74-view grid's.
T1's offset column is

    D = o_p - o_h

and a 74-view app-frame coordinate is carried into that point's frame by **subtracting** it:

    p_point = p_hero - D          then    cell = round((p_point - o_p) / resolution)

**Verification.** The 74-view recorded start is that scene's first camera, and every pass-A
frame is anchored at its own first camera, so the translated start must land on or beside the
point's own recorded first camera. On the **327-view point** it does: translated start
(+0.7070, -2.4210) against a first camera at (+0.4996, -2.6144), a miss of **0.284 m**, about
6 cells of 0.05 m in a grid 3.75 x 6.95 m. That fixes the sign - the other two candidates miss
by 2.660 m and 5.175 m on the same point (table B).

**The verification does not hold up as well on the other four.** The miss grows to 0.774 m at
574, and to about 1.45 m at 765, 1148 and 2295; at 2295 the uncorrected placement is actually
the closer of the two. `D` is the difference of two independently sized grid origins, not a
measured frame transform, and the pass-A frames differ by rotation and MapAnything scale as
well as translation - the 74-view room bbox spans 6.72 m in x against 4.43 m at 327 - which no
single 2-D offset can express. The corrected numbers below are therefore better than the
uncorrected ones and still carry a metre-scale placement error. They are reported with it
stated rather than without.

##### A. translated coordinates, per point

| views | T1 offset D (m) | translated start (m) | start cell | translated target (m) | target cell | in grid? | first camera (m) | start miss |
|---:|---|---|---|---|---|---|---|---:|
| 74 | (+0.0000, +0.0000) | (+0.0010, -0.0012) | (29, 121) | (+2.6043, -2.1714) | (81, 78) | start yes / target yes | (+0.0010, -0.0012) | 0.000 m |
| 327 | (-0.7060, +2.4198) | (+0.7070, -2.4210) | (57, 25) | (+3.3103, -4.5912) | (109, -19) | start yes / **target no** | (+0.4996, -2.6144) | 0.284 m |
| 574 | (-0.7412, +2.0681) | (+0.7422, -2.0693) | (59, 39) | (+3.3455, -4.2395) | (111, -5) | start yes / **target no** | (+0.4724, -2.7946) | 0.774 m |
| 765 | (-0.7177, +1.3036) | (+0.7186, -1.3048) | (58, 69) | (+3.3220, -3.4750) | (110, 26) | start yes / **target no** | (+0.5005, -2.7448) | 1.456 m |
| 1148 | (-0.4330, +3.2077) | (+0.4339, -3.2089) | (46, -7) | (+3.0373, -5.3791) | (98, -50) | **start no** / **target no** | (+0.7765, -1.8072) | 1.443 m |
| 2295 | (-0.6768, +2.7155) | (+0.6778, -2.7167) | (56, 13) | (+3.2811, -4.8869) | (108, -31) | start yes / **target no** | (+0.5762, -1.2551) | 1.465 m |

**The translated target still falls outside all five pass-B grids, and it falls further
outside than before.** The grids are 75 or 76 cells wide and the translated target's x index
is 98 to 111 in every one of them, 24 to 36 cells past the right edge - 1.20 to 1.80 m. On
four of the five it also leaves the grid in z, at indices -50, -31, -19 and -5. Measured as
distance from the grid rectangle, the uncorrected indexing put the target 0.80 to 1.05 m
outside; after translation it is 1.80 to 2.77 m outside, on every point:

| views | uncorrected target cell | distance outside | translated target cell | distance outside |
|---:|---|---:|---|---:|
| 327 | (95, 30) | 1.05 m | (109, -19) | 1.99 m |
| 574 | (96, 37) | 1.05 m | (111, -5) | 1.82 m |
| 765 | (95, 52) | 1.05 m | (110, 26) | 1.80 m |
| 1148 | (90, 14) | 0.80 m | (98, -50) | 2.77 m |
| 2295 | (95, 24) | 1.00 m | (108, -31) | 2.26 m |

This is not a contradiction of the verification: `-D` pulls the start towards the point's first
camera and pushes the desk target away from the grid, because it is one rigid 2-D shift applied
to a pair of points that the real transform relates by rotation and scale as well. It is the
clearest single statement of why the fixed target cannot be rescued by an offset.

##### B. sign convention, three candidates

Miss between the translated start and the point's recorded first camera:

| views | `p - D` (used) | `p` (uncorrected) | `p + D` |
|---:|---:|---:|---:|
| 74 | 0.000 m | 0.000 m | 0.000 m |
| 327 | **0.284 m** | 2.660 m | 5.175 m |
| 574 | **0.774 m** | 2.833 m | 5.010 m |
| 765 | **1.456 m** | 2.789 m | 4.226 m |
| 1148 | **1.443 m** | 1.965 m | 5.157 m |
| 2295 | 1.465 m | **1.380 m** | 4.162 m |

##### T3 route to the fixed target, after alignment

| views | robots reaching target | route m (min-max) | tightest gap m (min-max) | reason when unreachable |
|---:|---|---|---|---|
| 74 | 0/8 (none) | - | - | target cell not free for this radius (3/8: burger, waffle_pi, rosbot_xl_arm); start cell not free for this radius (5/8) |
| 327 | 0/8 (none) | - | - | target outside this point's grid, cell (109, -19), width 75 (8/8) |
| 574 | 0/8 (none) | - | - | target outside this point's grid, cell (111, -5), width 76 (8/8) |
| 765 | 0/8 (none) | - | - | target outside this point's grid, cell (110, 26), width 75 (8/8) |
| 1148 | 0/8 (none) | - | - | start and target outside this point's grid, cells (46, -7) and (98, -50) (8/8) |
| 2295 | 0/8 (none) | - | - | target outside this point's grid, cell (108, -31), width 76 (8/8) |

Still 0/8 at every rung, so the route-length and tightest-gap columns stay empty; what changes
is the reason, and one of them is not about frames at all.

**On the 74-view point the target was never routable.** The offset there is zero, so nothing in
this section moves it - and the three robots whose start cell is free still fail, because cell
(81, 78) is OBSTACLE in the 0.1-1.5 m band layer the planner reads, at `obstacle_min_height`
0.168 m. It is FREE only in the 0.1-0.5 m scene grid it was chosen on. The lowest nav-height
threshold in the registry is 0.182 m, so no radius and no view count would have reached it.

**The fixed target was mis-chosen**, and re-running it at more views would not fix it. That
is the study-design error recorded under Method, and it has since been corrected: the target
is now picked on the band layer the planner routes on, and the result is
[T3-74](#t3-74-route-to-the-desk-on-the-74-view-point). Everything in this subsection uses the
old cell (81, 78) and is kept only as the record of the attempt.

On the other five the reason is still the frames. The T1 offsets reach 3.2 m in z against
grids about 7 m deep, so one world coordinate names six different physical places, and that
has not changed. What the correction buys is a start that is placed to within 0.28-1.47 m
instead of 1.38-2.83 m, which is enough to make the start column mean something; it buys
nothing at all for the target, which ends up further outside every grid than it began. The
honest summary is that the offset is a partial correction on one of the two coordinates and a
worse one on the other, and that the fixed-target design cannot be repaired by arithmetic on
the stored layers.

### T2 start column, after alignment

The component containing the translated start, m2. This is the column that used to sit beside
the largest component in Results. It depends on placing the 74-view start in each point's grid,
so it inherits the 0.28-1.47 m placement error in table B and does not compare across rungs.

| robot | radius m | 74 | 327 | 574 | 765 | 1148 | 2295 |
|---|---:|---:|---:|---:|---:|---:|---:|
| burger | 0.1000 | 2.13 | 9.44 | 10.11 | 9.96 | 0.00 | 13.54 |
| waffle_pi | 0.1500 | 1.20 | 7.92 | 8.13 | 6.59 | 0.00 | 12.11 |
| turtlebot4 | 0.1750 | 0.00 | 0.00 | 0.00 | 4.56 | 0.00 | 7.20 |
| limo | 0.1940 | 0.00 | 0.60 | 0.00 | 3.82 | 0.00 | 7.41 |
| rosbot_xl_arm | 0.2186 | 0.14 | 4.88 | 0.00 | 0.00 | 0.00 | 6.58 |
| go2 | 0.2496 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.42 |
| jackal | 0.3340 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.01 |
| husky | 0.5528 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

The 1148 column is 0.00 throughout because the translated start lands at cell (46, -7), outside
that grid.

#### Before alignment: the tables as first published

The tables as first published, kept so the correction is visible. Both were produced by
indexing the 74-view world coordinates straight into each point's grid with that point's own
origin and **without** applying the grid-origin offset the study printed alongside them - the
`p` column of table B, which misses the recorded first camera by 1.38 to 2.83 m on the five
pass-B points. The largest-component column of T2 is unaffected by the translation and is
identical in both versions.

##### T2 before alignment: largest / containing the start, m2

| robot | radius m | 74 | 327 | 574 | 765 | 1148 | 2295 |
|---|---:|---:|---:|---:|---:|---:|---:|
| burger | 0.1000 | 6.49 / 2.13 | 9.44 / 9.44 | 10.11 / 10.11 | 9.96 / 9.96 | 9.71 / 9.71 | 13.54 / 0.00 |
| waffle_pi | 0.1500 | 3.53 / 1.20 | 7.92 / 0.00 | 8.13 / 8.13 | 6.59 / 0.00 | 8.48 / 8.48 | 12.11 / 0.00 |
| turtlebot4 | 0.1750 | 1.60 / 0.00 | 3.99 / 0.00 | 4.68 / 0.00 | 4.56 / 0.00 | 4.68 / 0.68 | 7.20 / 0.00 |
| limo | 0.1940 | 1.69 / 0.00 | 4.15 / 0.00 | 4.72 / 0.00 | 3.82 / 0.00 | 5.41 / 5.41 | 7.41 / 0.00 |
| rosbot_xl_arm | 0.2186 | 1.84 / 0.14 | 4.88 / 0.00 | 4.22 / 0.00 | 3.89 / 0.00 | 5.19 / 0.00 | 6.58 / 0.00 |
| go2 | 0.2496 | 0.83 / 0.00 | 2.97 / 0.00 | 3.04 / 0.00 | 2.86 / 0.00 | 2.36 / 0.00 | 3.61 / 0.00 |
| jackal | 0.3340 | 0.92 / 0.00 | 1.70 / 0.00 | 1.86 / 0.00 | 1.47 / 0.00 | 1.15 / 0.00 | 1.49 / 0.00 |
| husky | 0.5528 | 0.23 / 0.00 | 0.01 / 0.00 | 0.08 / 0.00 | 0.01 / 0.00 | 0.00 / 0.00 | 0.27 / 0.00 |

##### T3 before alignment

| views | robots reaching target | route m (min-max) | tightest gap m (min-max) | dominant reason when not |
|---:|---|---|---|---|
| 74 | 0/8 (none) | - | - | start cell not free for this radius |
| 327 | 0/8 (none) | - | - | start or target outside this point's grid |
| 574 | 0/8 (none) | - | - | start or target outside this point's grid |
| 765 | 0/8 (none) | - | - | start or target outside this point's grid |
| 1148 | 0/8 (none) | - | - | start or target outside this point's grid |
| 2295 | 0/8 (none) | - | - | start or target outside this point's grid |

##### What the old narrative claimed

> No robot reaches the fixed target at any of the six rungs, and the reason differs by rung. On
> the 74-view scene the start cell is not free at radius 0.175 m and above, and for the two
> smallest robots the start and the target fall in different components: the desk is not
> connected to the start. On the other five the target lands outside the grid entirely.

Three things in that paragraph were wrong, beyond the missing translation. The two smallest
robots did not fail on connectivity: their start was free and the *target* cell was not, at
every radius, for the band-layer reason set out above - the stored `analyse.json` said so at
the time. "Start or target outside this point's grid" merged two distinct cases that the
corrected T3 separates. And the 0/8 result was read as a frame-alignment consequence
throughout, when on the 74-view point, where the offset is zero, it is a target-selection
consequence instead.

The corrected T2's right-hand column also contradicts the old claim that the start component
is "zero for five of eight robots at every rung, and zero for all eight at 2295": after
translation 2295 is the rung where seven of eight robots have a non-zero start component, and
1148 is the rung where all eight are zero, because the translated start leaves the grid.
