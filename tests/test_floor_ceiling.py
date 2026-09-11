"""select_floor_candidate is the fix for the hotel-room floor-plane failure: on a
scene where furniture (a bed/table) covers most of the true floor, a single
band-wide RANSAC fit picks whichever plane has more inlier support - which can be
the furniture top, not the sparse real floor. The exported scene then starts at
bed/table height with real floor points below it clipped away as "phantom".

This module tests the pure selection logic in isolation from RANSAC/open3d (which
isn't installed in this venv - see floor_ceiling.fit_plane's docstring for why that
import is deferred): given a set of already-extracted horizontal plane candidates,
`select_floor_candidate` must prefer the true (lower, less-supported) floor over a
higher, better-supported furniture plane, while still satisfying the camera-height
and extent sanity checks.
"""

import numpy as np
import pytest

from gpu.floor_ceiling import PlaneFit, select_floor_candidate

# A room roughly 4m x 3m in XZ, camera walkthrough at realistic standing/handheld
# height (~1.5m) above the true floor.
ROOM_BBOX_XY_AREA = 4.0 * 3.0
CAM_POSITIONS = np.array(
    [
        [0.0, 1.5, 0.0],
        [1.0, 1.4, -1.0],
        [-1.0, 1.6, 1.0],
        [0.5, 1.5, 0.5],
    ]
)


def _square_ring(n_side: int, size: float, y: float, center=(0.0, 0.0)) -> np.ndarray:
    """A grid of points on a horizontal plane at height `y`, spanning `size` meters -
    enough for a well-defined convex hull, not just a handful of collinear points."""
    xs = np.linspace(-size / 2, size / 2, n_side) + center[0]
    zs = np.linspace(-size / 2, size / 2, n_side) + center[1]
    xx, zz = np.meshgrid(xs, zs)
    yy = np.full(xx.shape, y)
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)


def _make_candidate(*, y: float, n_points: int, size: float, n_views: int) -> PlaneFit:
    n_side = max(2, int(np.ceil(np.sqrt(n_points))))
    pts = _square_ring(n_side, size, y)[:n_points]
    return PlaneFit(
        normal=np.array([0.0, 1.0, 0.0]),
        d=-y,
        inlier_idx=np.arange(len(pts)),
        inlier_points=pts,
        tilt_deg=0.0,
        support=len(pts),
        n_views_supporting=n_views,
        plane_y=y,
        hull_area=size * size,
    )


def test_regression_furniture_plane_outsupports_real_floor_but_floor_is_chosen():
    """The exact hotel-room shape: true floor at y=0.0 is heavily occluded (only
    visible through narrow gaps scattered around the room, so its point count is far
    lower than the furniture's, even though those gaps span nearly the whole footprint),
    a bed/table top at y=0.55 is fully, densely visible. Old max-support logic would
    pick the bed. The fix must pick the floor, since it still clears the 20%-of-best
    support floor and passes both sanity checks."""
    real_floor = _make_candidate(y=0.0, n_points=900, size=2.5, n_views=10)
    bed_top = _make_candidate(y=0.55, n_points=4000, size=3.5, n_views=20)

    assert bed_top.support > real_floor.support  # sanity: furniture really does win on raw support

    chosen, warnings = select_floor_candidate(
        [real_floor, bed_top],
        extreme="min",
        cam_positions=CAM_POSITIONS,
        room_bbox_xy_area=ROOM_BBOX_XY_AREA,
    )

    assert chosen is real_floor
    assert chosen.plane_y == pytest.approx(0.0)
    assert warnings == []  # floor passed both sanity checks on the first try, nothing to explain


def test_old_max_support_logic_would_have_picked_the_furniture_plane():
    """Documents the bug this replaces: picking by raw max(support) alone - what the
    single-plane-per-band RANSAC fit effectively did - lands on the furniture plane,
    not the floor, for the exact same candidate set used above."""
    real_floor = _make_candidate(y=0.0, n_points=900, size=2.5, n_views=10)
    bed_top = _make_candidate(y=0.55, n_points=4000, size=3.5, n_views=20)

    old_logic_choice = max([real_floor, bed_top], key=lambda c: c.support)

    assert old_logic_choice is bed_top
    assert old_logic_choice.plane_y != pytest.approx(0.0)


def test_low_support_candidate_below_20pct_threshold_is_excluded():
    """A near-floor-height candidate with too little support to trust (e.g. a handful
    of noise points that happened to fit a horizontal plane) must not beat out a
    legitimate, better-supported floor just for being lower."""
    noise_candidate = _make_candidate(y=-0.3, n_points=20, size=0.3, n_views=1)
    real_floor = _make_candidate(y=0.0, n_points=2000, size=3.5, n_views=15)

    chosen, _warnings = select_floor_candidate(
        [noise_candidate, real_floor],
        extreme="min",
        cam_positions=CAM_POSITIONS,
        room_bbox_xy_area=ROOM_BBOX_XY_AREA,
    )

    assert chosen is real_floor


def test_camera_height_prior_rejects_implausible_floor_candidate():
    """A candidate that would put every camera 3m+ off the ground (e.g. a ceiling
    fixture or light fitting mistakenly landing in the floor-side group) fails the
    camera-height prior and is skipped in favor of a plausible one."""
    implausible = _make_candidate(y=-1.6, n_points=2000, size=3.5, n_views=15)  # cams ~3.1m above
    plausible = _make_candidate(y=0.0, n_points=1800, size=3.5, n_views=15)  # cams ~1.5m above

    chosen, warnings = select_floor_candidate(
        [implausible, plausible],
        extreme="min",
        cam_positions=CAM_POSITIONS,
        room_bbox_xy_area=ROOM_BBOX_XY_AREA,
    )

    assert chosen is plausible
    assert any("camera height" in w for w in warnings)


def test_extent_check_rejects_small_footprint_candidate():
    """A small, flat, well-supported surface (a nightstand top, densely sampled) below
    a large real floor must lose the extent check even though it's lower and has
    plenty of support."""
    nightstand = _make_candidate(y=-0.1, n_points=2000, size=0.4, n_views=10)  # tiny footprint
    real_floor = _make_candidate(y=0.0, n_points=1800, size=3.6, n_views=15)  # covers the room

    chosen, warnings = select_floor_candidate(
        [nightstand, real_floor],
        extreme="min",
        cam_positions=CAM_POSITIONS,
        room_bbox_xy_area=ROOM_BBOX_XY_AREA,
    )

    assert chosen is real_floor
    assert any("hull_area" in w for w in warnings)


def test_extreme_max_side_picks_highest_eligible_candidate():
    """The ceiling side of the room uses extreme='max' - sanity-check the direction
    flag actually flips the ordering, not just floor-side min."""
    lower = _make_candidate(y=2.0, n_points=2000, size=3.5, n_views=15)
    higher = _make_candidate(y=2.4, n_points=1900, size=3.5, n_views=15)

    cams_below_ceiling = np.array([[0.0, 0.9, 0.0], [0.5, 1.0, -0.5]])
    chosen, _warnings = select_floor_candidate(
        [lower, higher],
        extreme="max",
        cam_positions=cams_below_ceiling,
        room_bbox_xy_area=ROOM_BBOX_XY_AREA,
    )

    assert chosen is higher


def test_no_eligible_candidate_passes_checks_falls_back_with_warning():
    """If every candidate fails the sanity checks, fall back to max-support rather than
    raising - a low-confidence floor with clear diagnostics beats a hard failure."""
    only_candidate = _make_candidate(y=-1.6, n_points=2000, size=0.3, n_views=15)

    chosen, warnings = select_floor_candidate(
        [only_candidate],
        extreme="min",
        cam_positions=CAM_POSITIONS,
        room_bbox_xy_area=ROOM_BBOX_XY_AREA,
    )

    assert chosen is only_candidate
    assert any("LOW CONFIDENCE" in w for w in warnings)


def test_empty_candidate_list_raises():
    with pytest.raises(ValueError):
        select_floor_candidate(
            [], extreme="min", cam_positions=CAM_POSITIONS, room_bbox_xy_area=ROOM_BBOX_XY_AREA
        )
