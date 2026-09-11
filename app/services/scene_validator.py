"""Mandatory validation gate, run before any scene data is written to the database.

Not optional: the GPU pipeline has produced plausible-looking but wrong scenes twice
during this project's development (an inverted Y axis, and RANSAC finding the ceiling
instead of the floor) - both times every individual number looked internally
consistent, and only a check that understood what TYPE of object something is (a sink
should not be taller than a mirror) caught it. This module is that check, made
permanent and automatic instead of a one-off debugging step.

Note on `floor_y`/`ceiling_y` semantics: the GPU pipeline always normalizes the aligned
point cloud so the floor sits at Y=0 by construction (see gpu/stage_align.py). So
"floor" in these checks is always the constant 0.0, not a value read from the artifacts
- what varies scene-to-scene is `ceiling_y` (the ceiling's height above that floor) and
the camera positions, both read from the aligned-frame artifacts.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field

from app.services.scene_ingest import AlignmentInfo, GridMeta, ParsedObject

logger = logging.getLogger(__name__)

ALIGNED_FLOOR_Y = 0.0

# object-type -> (min_y, max_y) plausible centroid height above the floor, in meters.
# "counter" is anchored to this project's own validated reference: a real pedestal
# sink's rim measured at 0.8375m via SAM3-segmentation + real-world cross-check.
OBJECT_HEIGHT_PRIORS: dict[str, tuple[float, float]] = {
    "floor": (0.00, 0.15),
    "furniture": (0.25, 0.90),
    "counter": (0.65, 1.00),
    "wall": (1.10, 2.00),
}

_KEYWORDS_BY_CLASS: dict[str, tuple[str, ...]] = {
    "floor": ("floor", "rug", "carpet", "mat", "tile"),
    "counter": ("sink", "basin", "counter", "countertop", "stove", "vanity", "washbasin"),
    "wall": (
        "mirror", "cabinet", "shelf", "switch", "outlet", "dispenser", "window",
        "picture", "clock", "light fixture", "light",
    ),
    "furniture": (
        "chair", "table", "desk", "sofa", "couch", "bed", "toilet", "bin",
        "trash can", "stool", "bench",
    ),
}

ROOM_MIN_DIM_M = 1.5
ROOM_MAX_DIM_M = 30.0
CEILING_MIN_M = 2.0
CEILING_MAX_M = 5.0

# Horizontal (X or Z) bounding-box extent above which a single object is flagged as
# implausible. Confirmed real failure mode, not a guess: hotel-own-batch room 2
# (2026-09-01) reported a "bed" object at 2.92x3.31m against a real ~1x2m single bed -
# almost certainly two adjacent objects (two beds, or a bed plus nearby furniture)
# merged into one DBSCAN cluster. Non-fatal (see _check_object_max_extent) - unlike
# `height_prior_majority`, a single oversized object isn't strong enough evidence of a
# systemically wrong reconstruction to reject the whole scene, it just needs a visible
# flag so it doesn't ship silently.
OBJECT_MAX_EXTENT_M = 2.5

# M7b: curtains legitimately span a whole wall/window (the hero scene's real curtain
# measures 3.77m) - that is not the "two objects merged into one" failure mode
# OBJECT_MAX_EXTENT_M targets, so it's exempted from the oversized-object warning.
# Keep in sync with any other class-specific size handling (e.g. the class caps in
# scripts/msa/class_caps.py use a similar exemption idea, though that pipeline is
# unrelated to this one - the prod DB scene path here has no per-class limit map).
OVERSIZE_EXEMPT_CLASSES = {"curtain"}

MIN_VIEWS_FOR_HEIGHT_CHECK = 2  # single-view/fragment objects would poison the medians


class SceneValidationError(Exception):
    """Raised by the caller (pipeline_orchestrator) when a fatal check fails - carries
    the report's summary as its message."""


@dataclass(frozen=True)
class ValidationCheck:
    name: str
    passed: bool
    fatal: bool
    detail: str
    values: dict[str, float] = field(default_factory=dict)


@dataclass
class ValidationReport:
    checks: list[ValidationCheck]

    @property
    def passed(self) -> bool:
        return not self.failures()

    def failures(self) -> list[ValidationCheck]:
        return [c for c in self.checks if c.fatal and not c.passed]

    def summary(self) -> str:
        fails = self.failures()
        if not fails:
            return "all validation checks passed"
        return "; ".join(f"{c.name}: {c.detail}" for c in fails)


def classify_object(name: str) -> str | None:
    """Map an object name to a height-prior class via substring match, or None if it
    doesn't confidently match any class (e.g. "door" - spans floor to ceiling, no
    single plausible centroid height) - unmatched objects are skipped, not failed."""
    lowered = name.lower()
    for cls, keywords in _KEYWORDS_BY_CLASS.items():
        if any(kw in lowered for kw in keywords):
            return cls
    return None


def _check_floor_below_cameras(cameras: list[tuple[float, float, float]]) -> ValidationCheck:
    if not cameras:
        return ValidationCheck(
            "floor_below_cameras", passed=True, fatal=False,
            detail="no camera track available, skipped",
        )
    cam_ys = [c[1] for c in cameras]
    min_cam_y = min(cam_ys)
    passed = ALIGNED_FLOOR_Y < min_cam_y
    return ValidationCheck(
        "floor_below_cameras",
        passed=passed,
        fatal=True,
        detail=(
            f"floor_y={ALIGNED_FLOOR_Y:.3f} vs min camera Y={min_cam_y:.3f} "
            f"(margin={min_cam_y - ALIGNED_FLOOR_Y:.3f})"
        ),
        values={"floor_y": ALIGNED_FLOOR_Y, "min_camera_y": min_cam_y},
    )


def _check_ceiling_above_cameras(
    alignment: AlignmentInfo, cameras: list[tuple[float, float, float]]
) -> ValidationCheck:
    if not cameras:
        return ValidationCheck(
            "ceiling_above_cameras", passed=True, fatal=False,
            detail="no camera track available, skipped",
        )
    cam_ys = [c[1] for c in cameras]
    max_cam_y = max(cam_ys)
    passed = alignment.ceiling_y > max_cam_y
    return ValidationCheck(
        "ceiling_above_cameras",
        passed=passed,
        fatal=True,
        detail=(
            f"ceiling_y={alignment.ceiling_y:.3f} vs max camera Y={max_cam_y:.3f} "
            f"(margin={alignment.ceiling_y - max_cam_y:.3f})"
        ),
        values={"ceiling_y": alignment.ceiling_y, "max_camera_y": max_cam_y},
    )


# Deliberately NOT implemented here: a "floor plane tilt beyond tolerance -> flag"
# check. It was proposed alongside object_max_extent above, but a direct measurement
# across 5 scenes with recoverable geometry (2026-09-01 floor-plane diagnosis) found
# tilt uncorrelated with correctness in this pipeline: basement_apartment shipped fine
# at a 27.05deg chosen-candidate tilt (its own "low, less-tilted" alternative was
# 5.83deg - picking the LOW-tilt candidate there would have made a good 2.22m result
# 24% worse, not better); hotel room 2 is correct at 20.62deg. Meanwhile two scenes
# that both failed height_prior_majority for real (11_real_estate_fpv_axis,
# corridor-retest) had LOW chosen-candidate tilts (2.26deg, 0.36deg). A tilt-threshold
# flag here would false-positive on good scenes like basement_apartment and miss both
# real bad ones. Not implementing a check the evidence contradicts - `room_dimensions`
# below already catches the ceiling-height and footprint ranges this spec also asked
# for, as a stronger (fatal) form.


def _check_room_dimensions(alignment: AlignmentInfo, grid: GridMeta) -> ValidationCheck:
    width_m = grid.width * grid.resolution
    height_m = grid.height * grid.resolution
    ceiling_ok = CEILING_MIN_M <= alignment.ceiling_y <= CEILING_MAX_M
    dims_ok = all(ROOM_MIN_DIM_M <= d <= ROOM_MAX_DIM_M for d in (width_m, height_m))
    passed = ceiling_ok and dims_ok
    return ValidationCheck(
        "room_dimensions",
        passed=passed,
        fatal=True,
        detail=(
            f"footprint {width_m:.2f}x{height_m:.2f}m (expect "
            f"{ROOM_MIN_DIM_M}-{ROOM_MAX_DIM_M}m each), ceiling {alignment.ceiling_y:.2f}m "
            f"(expect {CEILING_MIN_M}-{CEILING_MAX_M}m)"
        ),
        values={"width_m": width_m, "height_m": height_m, "ceiling_y": alignment.ceiling_y},
    )


def _eligible_objects(objects: list[ParsedObject]) -> list[tuple[ParsedObject, str]]:
    """Objects confidently classified and reliable enough (not a fragment, >=2 views) to
    participate in the height-prior checks."""
    out = []
    for obj in objects:
        if obj.is_fragment or obj.num_views < MIN_VIEWS_FOR_HEIGHT_CHECK:
            continue
        cls = classify_object(obj.name)
        if cls is not None:
            out.append((obj, cls))
    return out


def _check_object_height_priors(objects: list[ParsedObject]) -> list[ValidationCheck]:
    """One non-fatal check per eligible object - individual misses are just a warning,
    not grounds to fail the whole scene (a single mis-segmented object shouldn't block
    an otherwise-good reconstruction)."""
    checks = []
    for obj, cls in _eligible_objects(objects):
        lo, hi = OBJECT_HEIGHT_PRIORS[cls]
        y = obj.pos[1]
        passed = lo <= y <= hi
        checks.append(
            ValidationCheck(
                f"object_height:{obj.name}",
                passed=passed,
                fatal=False,
                detail=f"{obj.name} ({cls}) y={y:.3f}, expected [{lo},{hi}]",
                values={"y": y, "lo": lo, "hi": hi},
            )
        )
    return checks


def _check_height_prior_majority(objects: list[ParsedObject]) -> ValidationCheck:
    """Fatal if MOST confidently-typed objects violate their own height prior - this is
    the aggregate signal that something is systematically wrong (an axis flip, a wrong
    floor/ceiling pick), as opposed to one object being individually mis-segmented."""
    eligible = _eligible_objects(objects)
    if not eligible:
        return ValidationCheck(
            "height_prior_majority", passed=True, fatal=False,
            detail="no eligible (typed, non-fragment, multi-view) objects to check",
        )
    violations = sum(
        1 for obj, cls in eligible if not (OBJECT_HEIGHT_PRIORS[cls][0] <= obj.pos[1] <= OBJECT_HEIGHT_PRIORS[cls][1])
    )
    frac = violations / len(eligible)
    passed = frac <= 0.5
    return ValidationCheck(
        "height_prior_majority",
        passed=passed,
        fatal=True,
        detail=f"{violations}/{len(eligible)} typed objects violate their height prior ({frac:.0%})",
        values={"violation_fraction": frac, "n_eligible": len(eligible)},
    )


def _check_relative_order(objects: list[ParsedObject]) -> ValidationCheck:
    """The specific check that caught both real failures this project hit: a counter
    fixture (sink) must not be, on median, taller than a wall fixture (mirror)."""
    eligible = _eligible_objects(objects)
    counter_ys = [obj.pos[1] for obj, cls in eligible if cls == "counter"]
    wall_ys = [obj.pos[1] for obj, cls in eligible if cls == "wall"]

    if not counter_ys or not wall_ys:
        return ValidationCheck(
            "relative_order_counter_below_wall", passed=True, fatal=False,
            detail=f"insufficient data ({len(counter_ys)} counter, {len(wall_ys)} wall objects), skipped",
        )

    counter_median = statistics.median(counter_ys)
    wall_median = statistics.median(wall_ys)
    passed = counter_median < wall_median
    return ValidationCheck(
        "relative_order_counter_below_wall",
        passed=passed,
        fatal=True,
        detail=f"median counter-fixture y={counter_median:.3f} vs median wall-fixture y={wall_median:.3f}",
        values={"counter_median_y": counter_median, "wall_median_y": wall_median},
    )


def find_oversized_objects(
    objects: list[ParsedObject], max_extent_m: float = OBJECT_MAX_EXTENT_M
) -> list[str]:
    """Non-fragment objects whose horizontal bounding-box extent (X or Z span, whichever
    is larger) exceeds `max_extent_m`. Returns ready-to-display "name (extent)" strings,
    not raw data - both `_check_object_max_extent` (below, for the validation log) and
    `pipeline_orchestrator._run` (to persist the flag onto the scene) use this directly
    so the two never drift out of sync with each other."""
    out = []
    for obj in objects:
        if obj.is_fragment or obj.name.lower() in OVERSIZE_EXEMPT_CLASSES:
            continue
        extent = max(obj.bbox_max[0] - obj.bbox_min[0], obj.bbox_max[2] - obj.bbox_min[2])
        if extent > max_extent_m:
            out.append(f"{obj.name} ({extent:.2f}m)")
    return out


def _check_object_max_extent(objects: list[ParsedObject]) -> ValidationCheck:
    """Non-fatal: see OBJECT_MAX_EXTENT_M's comment for why this doesn't fail the scene
    outright."""
    oversized = find_oversized_objects(objects)
    passed = not oversized
    return ValidationCheck(
        "object_max_extent",
        passed=passed,
        fatal=False,
        detail=(
            "no object exceeds the plausible-size threshold"
            if passed
            else f"{len(oversized)} object(s) exceed {OBJECT_MAX_EXTENT_M}m horizontal extent: "
            + ", ".join(oversized)
        ),
        values={"n_oversized": len(oversized), "threshold_m": OBJECT_MAX_EXTENT_M},
    )


def _check_reflection(alignment: AlignmentInfo) -> ValidationCheck:
    """Never fatal - a reflection (det(R)<0) is harmless for points/centroids/bboxes,
    it only matters once a real mesh with faces is exported (see gpu/stage_align.py's
    is_reflection field and the process rule in setup-log.md). This check's only job is
    to make the flag visible in the report; `scenes.is_reflected` is set from
    `alignment.is_reflection` directly by the caller, not derived from this check."""
    return ValidationCheck(
        "reflection_flag",
        passed=True,
        fatal=False,
        detail=(
            f"det(R)={alignment.det_r:.4f} "
            f"({'reflection - invert normals/winding before any mesh export' if alignment.is_reflection else 'proper rotation'})"
        ),
        values={"det_r": alignment.det_r},
    )


def validate_scene(
    alignment: AlignmentInfo,
    objects: list[ParsedObject],
    grid: GridMeta,
    cameras: list[tuple[float, float, float]],
) -> ValidationReport:
    """Run every check and return a full report. Log each check's numbers regardless of
    pass/fail - per the spec, "все проверки логировать с цифрами, не просто pass/fail"."""
    checks: list[ValidationCheck] = [
        _check_floor_below_cameras(cameras),
        _check_ceiling_above_cameras(alignment, cameras),
        _check_room_dimensions(alignment, grid),
        *_check_object_height_priors(objects),
        _check_height_prior_majority(objects),
        _check_relative_order(objects),
        _check_object_max_extent(objects),
        _check_reflection(alignment),
    ]

    for c in checks:
        level = logging.INFO if c.passed else (logging.ERROR if c.fatal else logging.WARNING)
        logger.log(level, "validation[%s] passed=%s fatal=%s: %s", c.name, c.passed, c.fatal, c.detail)

    return ValidationReport(checks=checks)
