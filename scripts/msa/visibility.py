"""T15i: camera visibility for the MSA presentation shot - pure stdlib, no numpy/trimesh.

Shared by `scripts/msa/export_presentation.py` (repo venv: solves and bakes the
`/World/PresentationCamera`) and `demo/record_isaac.py`'s Phase-2 worker (Isaac Sim's
own interpreter: re-checks the baked camera against the LARGER platform's ring and
re-solves if needed). Both sides must apply the identical rule, so it lives in one
module the worker can load either as `scripts.msa.visibility` (repo root on
sys.path) or from a `visibility.py` copied next to `record_isaac.py` on the Isaac host.

Why (docs/DECISIONS.md, "T15e run2"): the frustum-only `framing_ok` said the route was
in frame while every ray from the eye to the route/robots crossed the near 1.2 m wall
stub at 0.2-1.0 m height - i.e. behind an OPAQUE stub. The occlusion model here is a
set of oriented boxes (the remaining wall stub edges after the dollhouse cull + the
non-translucent object placeholder parts); a check point is visible iff the segment
eye -> point hits none of them. Boxes are conservative for a placeholder mesh (its
AABB contains it), so "box says visible" implies "mesh says visible".

Frame-agnostic: every function takes an explicit `up_axis` (1 = the MSA modules' Y-up
frame, 2 = Isaac's Z-up) and a `box` is `{"id", "center", "axes", "half"}` with `axes`
three orthonormal rows in whatever frame the caller uses.
"""

from __future__ import annotations

import math

# Soft furnishings / reflective surfaces that get 30 % opacity in every renderer
# (render_perspective still, web viewer, the USD's UsdPreviewSurface) and therefore
# never count as occluders here. `render_perspective.TRANSLUCENT_OBJECT_CLASSES`
# re-exports this; frontend/src/lib/msaLayers.ts mirrors it.
TRANSLUCENT_OBJECT_CLASSES = frozenset({"curtain", "drape", "blind", "mirror"})
TRANSLUCENT_OPACITY = 0.3
# Floor coverings (rug/mat placeholders a few cm tall) are not occluders: the robot
# drives over them and nothing taller than them can hide behind them, yet their box
# "contains" the floor-level check points (route at +0.03 m). Parts whose top is
# within this height of the floor are skipped by both exporters' occluder lists.
LOW_OCCLUDER_MAX_TOP_M = 0.1

FRAMING_MARGIN_FRAC = 0.05  # every framed point keeps >= this fraction of the frame per side
# The room outline (floor + stub-top vertices) only has to be IN the frame - "the room
# fills the frame" means its edges run close to the picture edges; the 5 % margin is
# for the things that must read clearly (robot, path, ring). Hero: with 5 % on the
# outline too the deep room forced a 0.4 m back-off and 59.8 % width coverage.
OUTLINE_MARGIN_FRAC = 0.02
FRAMING_NEAR_M = 0.05
HEIGHT_STEP_M = 0.25  # the eye is raised in these steps ...
MAX_EYE_HEIGHT_ABOVE_FLOOR_M = 4.5  # ... up to here before backing off
BACKOFF_STEP_M = 0.1
MAX_BACKOFF_M = 10.0
RING_VISIBILITY_SEGMENTS = 8  # points per clearance ring in the visibility check
# A clearance ring lies ON the floor and, at the goal, sits 0.2 m from the target's
# footprint (often between two objects), so from any elevated camera the neighbouring
# furniture hides the ring points nearest to it - demanding 8/8 is unsatisfiable there
# (02_modular_home: the table's leg hides ring_goal_7 from every candidate, the stool
# 3 more from anything closer than 3.4 m). Ring points are therefore "soft": a ring
# passes when at least HALF of its points are visible - the visible half still reads
# as the ring, the hidden half is behind the very object the robot stands next to;
# the route start, every path point, the goal and the robot tops are hard (all must
# be visible).
RING_MIN_VISIBLE_POINTS = 4


# --- tiny vector helpers -------------------------------------------------------------


def v_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v_scale(a, s: float):
    return (a[0] * s, a[1] * s, a[2] * s)


def v_dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def v_norm(a):
    n = math.sqrt(v_dot(a, a))
    return (a[0] / n, a[1] / n, a[2] / n) if n > 1e-12 else (0.0, 0.0, 0.0)


def unit_axis(axis: int):
    v = [0.0, 0.0, 0.0]
    v[axis] = 1.0
    return tuple(v)


def object_class_from_id(obj_id: str) -> str:
    """`curtain_0` -> `curtain` (same rule as render_perspective._object_class_from_id
    and the frontend's objectClassFromNodeName): strip one trailing `_<digits>`."""
    stem = obj_id.split("/", 1)[0]
    i = stem.rfind("_")
    if i > 0 and stem[i + 1 :].isdigit():
        return stem[:i]
    return stem


def is_translucent_class(obj_id_or_class: str) -> bool:
    return object_class_from_id(obj_id_or_class).lower() in TRANSLUCENT_OBJECT_CLASSES


# --- boxes ----------------------------------------------------------------------------


def make_box(box_id: str, center, axes, half) -> dict:
    """An oriented box: `axes` = three orthonormal unit vectors (rows), `half` = half
    extents along them."""
    return {
        "id": str(box_id),
        "center": tuple(float(v) for v in center),
        "axes": tuple(tuple(float(c) for c in ax) for ax in axes),
        "half": tuple(float(v) for v in half),
    }


def aabb_box(box_id: str, lo, hi) -> dict:
    center = tuple((float(lo[i]) + float(hi[i])) / 2.0 for i in range(3))
    half = tuple(max(0.0, (float(hi[i]) - float(lo[i])) / 2.0) for i in range(3))
    return make_box(box_id, center, ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), half)


def box_to_list(box: dict) -> list:
    """15 floats: center(3), axes(9), half(3) - the flat form baked into the USD."""
    return [*box["center"], *box["axes"][0], *box["axes"][1], *box["axes"][2], *box["half"]]


def boxes_from_flat(values, ids=None) -> list[dict]:
    vals = [float(v) for v in values]
    if len(vals) % 15:
        raise ValueError(f"flat box list has {len(vals)} values, not a multiple of 15")
    out = []
    for k in range(len(vals) // 15):
        c = vals[15 * k : 15 * k + 15]
        box_id = ids[k] if ids is not None and k < len(ids) else f"box_{k}"
        out.append(make_box(box_id, c[0:3], (c[3:6], c[6:9], c[9:12]), c[12:15]))
    return out


def transform_box(box: dict, fn) -> dict:
    """Apply a proper rotation `fn(xyz) -> xyz` (e.g. usd_export.to_isaac) to a box."""
    origin = fn((0.0, 0.0, 0.0))
    axes = tuple(v_norm(v_sub(fn(ax), origin)) for ax in box["axes"])
    return make_box(box["id"], fn(box["center"]), axes, box["half"])


def segment_hits_box(p0, p1, box: dict, *, eps: float = 1e-9) -> bool:
    """Slab test of the segment p0 -> p1 (parameter t in (0, 1)) against an oriented
    box, in the box's own frame. A segment that merely touches a face at its
    endpoint (a check point ON the box, e.g. a route point on the floor next to a
    hull) counts as a hit only if it actually enters the box."""
    rel = v_sub(p0, box["center"])
    d = v_sub(p1, p0)
    t_min, t_max = 0.0, 1.0
    for ax, h in zip(box["axes"], box["half"]):
        o = v_dot(rel, ax)
        dd = v_dot(d, ax)
        if abs(dd) < eps:
            if abs(o) > h:
                return False
            continue
        t0 = (-h - o) / dd
        t1 = (h - o) / dd
        if t0 > t1:
            t0, t1 = t1, t0
        t_min = max(t_min, t0)
        t_max = min(t_max, t1)
        if t_min > t_max:
            return False
    # Inside the slab intersection for t in [t_min, t_max]; require a real interval
    # strictly inside the open segment (endpoint grazes don't occlude).
    return t_max - t_min > 1e-6 and t_max > 1e-6 and t_min < 1.0 - 1e-6


def first_occluder(p0, p1, boxes) -> str | None:
    for box in boxes:
        if segment_hits_box(p0, p1, box):
            return box["id"]
    return None


def point_inside_box(p, box: dict, *, tol: float = 1e-6) -> bool:
    rel = v_sub(p, box["center"])
    return all(abs(v_dot(rel, ax)) <= h - tol for ax, h in zip(box["axes"], box["half"]))


def points_inside_boxes(labelled_points, boxes) -> list[dict]:
    """Check points that lie INSIDE an opaque box - no camera anywhere can see them,
    so the solver reports them separately instead of scanning every candidate for
    a pose that cannot exist (a placeholder overhanging the route, a modelling gap)."""
    out = []
    for label, p in labelled_points:
        for box in boxes:
            if point_inside_box(p, box):
                out.append({"label": label, "point": [float(v) for v in p], "inside": box["id"]})
                break
    return out


def ring_group(label: str) -> str | None:
    """`ring_goal_7` -> `ring_goal`; None for a non-ring label."""
    if not label.startswith("ring_"):
        return None
    return label.rsplit("_", 1)[0]


def visibility_report(eye, labelled_points, boxes, *, ring_min_visible: int = RING_MIN_VISIBLE_POINTS) -> dict:
    """`labelled_points` = [(label, xyz), ...]. Every point's eye ray is tested against
    every box. `ok` = no HARD point (anything not labelled `ring_*`) is occluded and
    every ring keeps >= `ring_min_visible` visible points. Returns `{ok, n_points,
    n_occluded, n_hard_occluded, occluded: [{label, point, by}], rings: {group:
    {n, visible, ok}}}`."""
    occluded = []
    rings: dict = {}
    n_hard = 0
    for label, p in labelled_points:
        group = ring_group(label)
        if group is not None:
            rings.setdefault(group, {"n": 0, "visible": 0})
            rings[group]["n"] += 1
        by = first_occluder(eye, p, boxes)
        if by is not None:
            occluded.append({"label": label, "point": [float(v) for v in p], "by": by})
            if group is None:
                n_hard += 1
        elif group is not None:
            rings[group]["visible"] += 1
    for g in rings.values():
        g["ok"] = g["visible"] >= min(ring_min_visible, g["n"])
    ok = n_hard == 0 and all(g["ok"] for g in rings.values())
    return {
        "ok": ok, "n_points": len(labelled_points), "n_occluded": len(occluded), "n_hard_occluded": n_hard,
        "occluded": occluded, "rings": rings, "ring_min_visible": ring_min_visible,
    }


# --- check points ---------------------------------------------------------------------


def ring_points(center, radius_m: float, *, ground_axes=(0, 1), n: int = RING_VISIBILITY_SEGMENTS) -> list:
    pts = []
    for i in range(n):
        theta = 2 * math.pi * i / n
        p = list(center)
        p[ground_axes[0]] = center[ground_axes[0]] + radius_m * math.cos(theta)
        p[ground_axes[1]] = center[ground_axes[1]] + radius_m * math.sin(theta)
        pts.append(tuple(p))
    return pts


def visibility_check_points(start_xyz, path_points_xyz, goal_xyz, ring_radius_m: float, robot_height_m: float, *,
                            ground_axes=(0, 1), up_axis: int = 2) -> list:
    """What must be unoccluded (T15i): the route start, every path point, the goal,
    the robot proxy's top at the start and at the goal, and RING_VISIBILITY_SEGMENTS
    points on the clearance ring at both. Returned as `[(label, xyz), ...]`."""
    lift = [0.0, 0.0, 0.0]
    lift[up_axis] = robot_height_m
    pts = [("route_start", tuple(start_xyz))]
    pts += [(f"path_{i}", tuple(p)) for i, p in enumerate(path_points_xyz)]
    pts.append(("goal", tuple(goal_xyz)))
    pts.append(("robot_top_start", v_add(start_xyz, lift)))
    pts.append(("robot_top_goal", v_add(goal_xyz, lift)))
    pts += [(f"ring_start_{k}", p) for k, p in enumerate(ring_points(start_xyz, ring_radius_m, ground_axes=ground_axes))]
    pts += [(f"ring_goal_{k}", p) for k, p in enumerate(ring_points(goal_xyz, ring_radius_m, ground_axes=ground_axes))]
    return pts


# --- camera basis / framing -----------------------------------------------------------


def camera_basis(forward, up_axis: int = 2):
    """(forward, right, up) - the same convention as scripts.msa.render_perspective
    (right = forward x world_up, up = right x forward; +right = image right)."""
    f = v_norm(forward)
    r = v_cross(f, unit_axis(up_axis))
    if v_dot(r, r) < 1e-18:
        r = (1.0, 0.0, 0.0)
    r = v_norm(r)
    u = v_norm(v_cross(r, f))
    return f, r, u


def camera_basis_looking_at(eye, aim, up_axis: int = 2):
    return camera_basis(v_sub(aim, eye), up_axis)


def aim_point_on_floor(eye, forward, floor_level: float, up_axis: int = 2, *, fallback_distance_m: float = 5.0):
    """Where the central ray meets the floor plane (`coord[up_axis] == floor_level`);
    if it does not (pitch >= 0), the point `fallback_distance_m` ahead."""
    f = v_norm(forward)
    if f[up_axis] < -1e-6:
        t = (floor_level - eye[up_axis]) / f[up_axis]
        if t > 0:
            return v_add(eye, v_scale(f, t))
    return v_add(eye, v_scale(f, fallback_distance_m))


def pitch_deg(forward, up_axis: int = 2) -> float:
    f = v_norm(forward)
    return math.degrees(math.asin(max(-1.0, min(1.0, f[up_axis]))))


def points_in_frame(points, eye, forward, right, up, fov_v_deg: float, aspect: float, *,
                    margin_frac: float = FRAMING_MARGIN_FRAC, near_m: float = FRAMING_NEAR_M) -> bool:
    """True iff every point projects inside a pinhole frame (vertical FOV `fov_v_deg`,
    width/height `aspect`) with >= `margin_frac` of the frame left on each side and
    sits in front of the near plane."""
    return first_point_out_of_frame(points, eye, forward, right, up, fov_v_deg, aspect, margin_frac=margin_frac, near_m=near_m) is None


def first_point_out_of_frame(points, eye, forward, right, up, fov_v_deg: float, aspect: float, *,
                             margin_frac: float = FRAMING_MARGIN_FRAC, near_m: float = FRAMING_NEAR_M):
    tan_v = math.tan(math.radians(fov_v_deg) / 2.0)
    tan_h = tan_v * aspect
    usable = 1.0 - 2.0 * margin_frac
    for p in points:
        rel = v_sub(p, eye)
        z = v_dot(rel, forward)
        if z <= near_m:
            return tuple(p)
        if abs(v_dot(rel, right)) > tan_h * z * usable:
            return tuple(p)
        if abs(v_dot(rel, up)) > tan_v * z * usable:
            return tuple(p)
    return None


def projected_bbox(points, eye, forward, right, up, fov_v_deg: float, aspect: float, *, near_m: float = FRAMING_NEAR_M):
    """Normalised-device bbox `(min_x, min_y, max_x, max_y)` of the points in front of
    the near plane (frame spans [-1, 1] in both axes), or None. `width_frac` of the
    frame = (max_x - min_x) / 2."""
    tan_v = math.tan(math.radians(fov_v_deg) / 2.0)
    tan_h = tan_v * aspect
    xs, ys = [], []
    for p in points:
        rel = v_sub(p, eye)
        z = v_dot(rel, forward)
        if z <= near_m:
            continue
        xs.append(v_dot(rel, right) / (z * tan_h))
        ys.append(v_dot(rel, up) / (z * tan_v))
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def centre_points_in_frame(eye, points, fov_v_deg: float, aspect: float, *, up_axis: int = 2, iterations: int = 12):
    """The forward direction that puts the projected bbox of `points` at the frame
    centre (yaw and pitch both solved; roll zero). Starts by looking at the points'
    mean and re-aims at the bbox centre until it converges. Returns
    `(forward, bbox_ndc)`; the bbox is None if nothing is in front of the eye."""
    n = len(points)
    mean = (sum(p[0] for p in points) / n, sum(p[1] for p in points) / n, sum(p[2] for p in points) / n)
    forward = v_norm(v_sub(mean, eye))
    bbox = None
    tan_v = math.tan(math.radians(fov_v_deg) / 2.0)
    tan_h = tan_v * aspect
    for _ in range(iterations):
        f, r, u = camera_basis(forward, up_axis)
        bbox = projected_bbox(points, eye, f, r, u, fov_v_deg, aspect)
        if bbox is None:
            break
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        if abs(cx) < 1e-6 and abs(cy) < 1e-6:
            break
        forward = v_norm(v_add(f, v_add(v_scale(r, cx * tan_h), v_scale(u, cy * tan_v))))
    return forward, bbox


# --- morning-4: the path ribbon with a robot standing anywhere on the route -------------

PATH_RIBBON_WIDTH_M = 0.03
PATH_RIBBON_HEIGHT_ABOVE_FLOOR_M = 0.03
ROUTE_ROBOT_SAMPLE_STEP_M = 0.25  # a robot box is placed on the route every this much ...
RIBBON_SAMPLE_STEP_M = 0.05  # ... and the ribbon is checked at points this far apart (0.05: a 95 % bar on a 1.4 m
# uncovered ribbon is 0.07 m - at 0.10 m spacing the fraction could only step by 1/16)
PATH_VISIBILITY_MIN_FRACTION = 0.95  # >= this fraction of ribbon points visible in the WORST robot placement


def polyline_samples(points, step_m: float) -> list:
    """Points along a polyline every `step_m` (every vertex included) with the
    unit tangent of the segment they lie on: `[(point, tangent), ...]`."""
    out = []
    if not points:
        return out
    pts = [tuple(float(v) for v in p) for p in points]
    if len(pts) == 1:
        return [(pts[0], (1.0, 0.0, 0.0))]
    for a, b in zip(pts, pts[1:]):
        d = v_sub(b, a)
        seg = math.sqrt(v_dot(d, d))
        if seg < 1e-9:
            continue
        t_hat = v_scale(d, 1.0 / seg)
        n = max(1, int(math.ceil(seg / step_m)))
        if not out:
            out.append((a, t_hat))
        for k in range(1, n + 1):
            out.append((v_add(a, v_scale(d, k / n)), t_hat))
    return out


def robot_box_on_route(box_id: str, point, tangent, length_m: float, width_m: float, height_m: float, *,
                       floor_level: float, up_axis: int = 2) -> dict:
    """An oriented robot body box standing at `point` (a floor point) heading along
    `tangent`: length along the tangent, width across it in the ground plane, height
    up from `floor_level`."""
    up = unit_axis(up_axis)
    t = tuple(0.0 if i == up_axis else tangent[i] for i in range(3))
    t = v_norm(t) if v_dot(t, t) > 1e-12 else tuple(1.0 if i == (0 if up_axis != 0 else 1) else 0.0 for i in range(3))
    side = v_norm(v_cross(up, t))
    centre = list(point)
    centre[up_axis] = floor_level + height_m / 2.0
    return make_box(box_id, tuple(centre), (t, side, up), (length_m / 2.0, width_m / 2.0, height_m / 2.0))


def route_robot_box_sets(path_points, platforms, *, floor_level: float, up_axis: int = 2, step_m: float = ROUTE_ROBOT_SAMPLE_STEP_M) -> list:
    """For every route sample (every `step_m`) and every platform `{"id", "length_m",
    "width_m", "height_m"}`: one oriented box - `[{"sample": k, "platform": id,
    "point": xyz, "box": box}, ...]`. The path-visibility check takes the WORST case
    over these placements."""
    out = []
    for k, (p, t) in enumerate(polyline_samples(path_points, step_m)):
        for plat in platforms:
            box = robot_box_on_route(f"robot:{plat['id']}@{k}", p, t, float(plat["length_m"]), float(plat["width_m"]), float(plat["height_m"]),
                                     floor_level=floor_level, up_axis=up_axis)
            out.append({"sample": k, "platform": plat["id"], "point": [float(v) for v in p], "box": box})
    return out


def ribbon_sample_points(path_points, *, floor_level: float, up_axis: int = 2, step_m: float = RIBBON_SAMPLE_STEP_M,
                         height_above_floor_m: float = PATH_RIBBON_HEIGHT_ABOVE_FLOOR_M) -> list:
    """The ribbon's check points: the path sampled every `step_m`, lifted to the ribbon height."""
    pts = []
    for p, _t in polyline_samples(path_points, step_m):
        q = list(p)
        q[up_axis] = floor_level + height_above_floor_m
        pts.append(tuple(q))
    return pts


def point_under_box_footprint(p, box: dict, ground_axes=(0, 1)) -> bool:
    """True when `p`'s ground projection lies inside the box's ground footprint (the
    box's two ground-plane axes; the third axis is ignored) - a ribbon point the
    robot is standing ON."""
    rel = v_sub(p, box["center"])
    for ax, h in zip(box["axes"], box["half"]):
        if abs(ax[ground_axes[0]]) < 1e-9 and abs(ax[ground_axes[1]]) < 1e-9:
            continue  # the up axis
        if abs(v_dot(rel, ax)) > h:
            return False
    return True


def path_visibility_worst_case(eye, ribbon_points, static_boxes, robot_box_sets, *, ground_axes=(0, 1)) -> dict:
    """Fraction of the ribbon visible from `eye` past `static_boxes` (stubs + parts)
    in the WORST placement of a robot box along the route (`route_robot_box_sets`,
    one box at a time - a robot stands at ONE point of the route at any frame).
    Ribbon points UNDER the robot's footprint (`point_under_box_footprint`) are
    COVERED, not occluded: a robot on the route necessarily hides the ribbon it
    stands on (Husky's 0.985 m body covers 42 % of the hero's 2.34 m route by itself),
    so they leave the denominator; what the criterion measures is the ribbon hidden
    BEHIND the robot's body from the camera (T15e run3: the whole route, from a
    -35 deg camera looking along it). Static occlusion is evaluated once per point;
    each placement only adds its own box. Returns `{fraction, static_fraction,
    worst: {sample, platform, point, visible, covered}, n_points, n_placements}`."""
    n = len(ribbon_points)
    if n == 0:
        return {"fraction": 1.0, "static_fraction": 1.0, "worst": None, "n_points": 0, "n_placements": len(robot_box_sets)}
    static_visible = [first_occluder(eye, p, static_boxes) is None for p in ribbon_points]
    static_fraction = sum(static_visible) / n
    worst = None
    worst_fraction = static_fraction
    for placement in robot_box_sets:
        box = placement["box"]
        covered = [point_under_box_footprint(p, box, ground_axes) for p in ribbon_points]
        n_free = n - sum(covered)
        visible = sum(1 for p, ok, c in zip(ribbon_points, static_visible, covered) if not c and ok and not segment_hits_box(eye, p, box))
        frac = visible / n_free if n_free > 0 else 1.0
        if worst is None or frac < worst_fraction:
            worst_fraction = frac
            worst = {"sample": placement["sample"], "platform": placement["platform"], "point": placement["point"], "visible": visible,
                     "covered": sum(covered), "n_uncovered": n_free}
    return {"fraction": worst_fraction, "static_fraction": static_fraction, "worst": worst, "n_points": n, "n_placements": len(robot_box_sets),
            "rule": "ribbon points under the robot footprint are covered (excluded); fraction = visible / uncovered, worst placement"}


# --- camera clearance (user addendum 2026-09-07): the eye keeps >= this from every prim --

CAMERA_MIN_CLEARANCE_M = 0.3


def aabb_distance(point, lo, hi) -> float:
    """Euclidean distance from `point` to the axis-aligned box [lo, hi] (0 inside)."""
    d2 = 0.0
    for i in range(3):
        v = float(point[i])
        if v < lo[i]:
            d2 += (lo[i] - v) ** 2
        elif v > hi[i]:
            d2 += (v - hi[i]) ** 2
    return math.sqrt(d2)


def camera_clearance_violation(eye, ranges, min_clearance_m: float = CAMERA_MIN_CLEARANCE_M) -> dict | None:
    """`ranges` = `[{"prim", "min", "max"}, ...]` (world AABBs). The nearest prim
    closer than `min_clearance_m` to `eye` (or containing it) as `{prim, distance_m,
    inside}`, else None."""
    worst = None
    for r in ranges:
        d = aabb_distance(eye, r["min"], r["max"])
        if d < min_clearance_m and (worst is None or d < worst["distance_m"]):
            worst = {"prim": r["prim"], "distance_m": d, "inside": d <= 0.0, "min_clearance_m": min_clearance_m}
    return worst


def segment_clearance_violation(p0, p1, ranges, min_clearance_m: float = CAMERA_MIN_CLEARANCE_M, *, n_samples: int = 24) -> dict | None:
    """The clearance check along a straight camera move (the wide -> close transition),
    sampled at `n_samples` points including both ends."""
    for k in range(n_samples + 1):
        t = k / n_samples
        p = v_add(p0, v_scale(v_sub(p1, p0), t))
        v = camera_clearance_violation(p, ranges, min_clearance_m)
        if v is not None:
            return {**v, "t": t, "point": [float(c) for c in p]}
    return None


# --- the solver -----------------------------------------------------------------------


def solve_visible_camera(
    eye0, aim, fov_v_deg: float, aspect: float, frame_points, labelled_vis_points, boxes, *,
    up_axis: int = 2, floor_level: float = 0.0,
    height_step_m: float = HEIGHT_STEP_M, max_height_above_floor_m: float = MAX_EYE_HEIGHT_ABOVE_FLOOR_M,
    backoff_step_m: float = BACKOFF_STEP_M, max_backoff_m: float = MAX_BACKOFF_M,
    margin_frac: float = FRAMING_MARGIN_FRAC, outline_points=(), outline_margin_frac: float = OUTLINE_MARGIN_FRAC,
    ribbon_points=(), robot_box_sets=(), path_visibility_min_fraction: float = PATH_VISIBILITY_MIN_FRACTION,
    clearance_ranges=(), min_eye_clearance_m: float = CAMERA_MIN_CLEARANCE_M,
    move_in_max_m: float = 0.0, min_aim_distance_m: float = 0.5, min_outline_width_frac: float = 0.0,
    transition_from=None, outline_frame_required: bool = True,
) -> dict:
    """The T15i camera rule, shared by export_presentation and record_isaac's worker.

    Starting from `eye0` looking at `aim` (a floor point - the room's frame centre),
    find the first pose, scanning back-off OUTER and eye height INNER, at which (a)
    every `frame_points` point is in frame with `margin_frac` per side (and every
    `outline_points` point - the room polygon at floor/stub height - with the looser
    `outline_margin_frac`) and (b) no
    hard `labelled_vis_points` point is occluded by a box and every ring keeps
    RING_MIN_VISIBLE_POINTS visible points (`visibility_report`): for each back-off step (0,
    `backoff_step_m`, ... along the horizontal aim -> eye direction) the eye height
    above `floor_level` is raised from its starting value in `height_step_m` steps up
    to `max_height_above_floor_m`, the pitch re-solved each time so `aim` stays at the
    frame centre. Deterministic. If nothing within the limits works, the candidate
    with the fewest occluded points (then the smallest back-off) is returned with
    `ok = False` - the caller logs it and the shot is still rendered.

    HARD check points that lie inside a box (`points_inside_boxes`) can never be seen:
    they are excluded from the search, listed under `visibility["inside"]`, and make
    `ok` False. A RING point inside a box (Husky's 0.6 m ring at a goal 0.2 m from the
    target) simply counts as one of that ring's hidden points.

    Morning-4 (`ribbon_points` + `robot_box_sets` given): the candidate must also keep
    >= `path_visibility_min_fraction` of the path ribbon's points visible with a robot
    box standing at ANY sampled point of the route (`path_visibility_worst_case`,
    worst case over `route_robot_box_sets`) - the T15e run3 clip lost the path for
    45 % of its frames under Husky. Camera higher / farther is the remedy, both
    explicitly allowed. Addendum: with `clearance_ranges` (world AABBs) the eye of a
    candidate must keep >= `min_eye_clearance_m` from every prim
    (`camera_clearance_violation`), else the candidate is skipped.

    Steep search (orchestrator 2026-09-07, for the Husky-pass pose): `move_in_max_m` > 0
    also scans NEGATIVE back-offs (the eye moves horizontally TOWARD the aim, never
    closer than `min_aim_distance_m`), so a high eye can look near top-down; the
    back-off order is 0, -step, +step, -2 step, ... (least horizontal change first,
    heights inner as before). `min_outline_width_frac` > 0 additionally requires the
    room outline's projected bbox to span that fraction of the frame width
    (`projected_bbox`). `transition_from` (an eye) rejects candidates whose straight
    camera move from there violates the clearance (`segment_clearance_violation`).
    `outline_frame_required=False` drops the "whole outline in frame" test (the width
    fraction is still measured on `outline_points`) - the Husky-pass pose looks down
    at the route, not at the whole room.

    Returns eye/forward/right/up, `backoff_m`, `height_above_floor_m`,
    `height_raise_m`, `pitch_deg`, `framing_ok`, `visibility` (the per-point report),
    `path_visibility` (worst case, or None when not requested), `eye_clearance`,
    `outline_width_frac`, `n_candidates` and `tried` (one row per candidate)."""
    inside = points_inside_boxes([(label, p) for label, p in labelled_vis_points if ring_group(label) is None], boxes)
    inside_labels = {row["label"] for row in inside}
    solvable_points = [(label, p) for label, p in labelled_vis_points if label not in inside_labels]
    away = v_sub(eye0, aim)
    away = tuple(0.0 if i == up_axis else away[i] for i in range(3))
    if v_dot(away, away) < 1e-12:
        f0 = v_sub(aim, eye0)
        away = tuple(0.0 if i == up_axis else -f0[i] for i in range(3))
    away = v_norm(away)
    h0 = float(eye0[up_axis]) - float(floor_level)
    heights = [h0]
    while heights[-1] + height_step_m <= max_height_above_floor_m + 1e-9:
        heights.append(heights[-1] + height_step_m)
    if max_height_above_floor_m > heights[-1] + 1e-9:
        heights.append(float(max_height_above_floor_m))  # the cap itself is always a candidate

    n_back = int(math.floor(max_backoff_m / backoff_step_m + 1e-9))
    d_aim0 = math.sqrt(sum((eye0[i] - aim[i]) ** 2 for i in range(3) if i != up_axis))
    n_in = int(math.floor(min(move_in_max_m, max(0.0, d_aim0 - min_aim_distance_m)) / backoff_step_m + 1e-9)) if move_in_max_m > 0 else 0
    backoffs = [0.0]
    for k in range(1, max(n_back, n_in) + 1):
        if k <= n_in:
            backoffs.append(-k * backoff_step_m)
        if k <= n_back:
            backoffs.append(k * backoff_step_m)
    tried = []
    best = None
    for backoff in backoffs:
        base = v_add(eye0, v_scale(away, backoff))
        for h in heights:
            eye = list(base)
            eye[up_axis] = float(floor_level) + h
            eye = tuple(eye)
            f, r, u = camera_basis_looking_at(eye, aim, up_axis)
            framing_ok = points_in_frame(frame_points, eye, f, r, u, fov_v_deg, aspect, margin_frac=margin_frac) and (
                not outline_frame_required or points_in_frame(outline_points, eye, f, r, u, fov_v_deg, aspect, margin_frac=outline_margin_frac)
            )
            outline_width = None
            if outline_points and min_outline_width_frac > 0:
                bbox = projected_bbox(outline_points, eye, f, r, u, fov_v_deg, aspect)
                outline_width = (bbox[2] - bbox[0]) / 2.0 if bbox is not None else 0.0
                framing_ok = framing_ok and outline_width >= min_outline_width_frac - 1e-9
            vis = visibility_report(eye, solvable_points, boxes)
            solvable_ok = vis["ok"]
            vis["inside"] = inside
            vis["n_points"] = len(labelled_vis_points)
            vis["ok"] = solvable_ok and not inside
            path_vis = None
            path_ok = True
            if ribbon_points:
                path_vis = path_visibility_worst_case(eye, ribbon_points, boxes, robot_box_sets)
                path_vis["min_fraction"] = path_visibility_min_fraction
                path_vis["ok"] = path_vis["fraction"] >= path_visibility_min_fraction - 1e-9
                path_ok = path_vis["ok"]
            clearance = camera_clearance_violation(eye, clearance_ranges, min_eye_clearance_m) if clearance_ranges else None
            if clearance is None and transition_from is not None and clearance_ranges:
                clearance = segment_clearance_violation(tuple(transition_from), eye, clearance_ranges, min_eye_clearance_m)
                if clearance is not None:
                    clearance = {**clearance, "on_transition": True}
            clearance_ok = clearance is None
            row = {"backoff_m": backoff, "height_above_floor_m": h, "framing_ok": framing_ok, "n_occluded": vis["n_occluded"],
                   "path_visibility": None if path_vis is None else path_vis["fraction"], "clearance_ok": clearance_ok}
            tried.append(row)
            cand = {
                "ok": framing_ok and vis["ok"] and path_ok and clearance_ok, "eye": eye, "forward": f, "right": r, "up": u, "aim": tuple(aim),
                "backoff_m": backoff, "height_above_floor_m": h, "height_raise_m": h - h0,
                "pitch_deg": pitch_deg(f, up_axis), "framing_ok": framing_ok, "visibility": vis,
                "path_visibility": path_vis, "eye_clearance": {"ok": clearance_ok, "violation": clearance, "min_clearance_m": min_eye_clearance_m,
                                                                "n_ranges": len(clearance_ranges)},
                "outline_width_frac": outline_width,
                "solved_for_solvable_points": framing_ok and solvable_ok and path_ok and clearance_ok,
            }
            if cand["solved_for_solvable_points"]:
                cand["n_candidates"] = len(tried)
                cand["tried"] = tried
                return cand
            score = (0 if clearance_ok else 1, 0 if framing_ok else 1, vis["n_hard_occluded"], vis["n_occluded"],
                     0.0 if path_vis is None else -path_vis["fraction"], backoff, h)
            if best is None or score < best[0]:
                best = (score, cand)
    _, cand = best
    cand["n_candidates"] = len(tried)
    cand["tried"] = tried
    return cand
