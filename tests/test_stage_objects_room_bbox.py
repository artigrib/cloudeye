"""room_bbox_point_mask() is the per-point guard against SAM3/MapAnything's mask-leak
failure mode: a mask for an open doorway or mirror can include the opening itself, and
MapAnything's monocular depth has no anchor past it, producing points far outside the
room. A cluster's centroid can still land safely inside room_bbox even when a chunk of
its points are leaked meters past a wall - centroid-only checks miss this entirely
(confirmed on real production data: scene 68027f2a's kept, non-fragment door_0 has bbox
depth 2.52m with an in-bounds centroid). This guard has to reject the leaked points
themselves, before they can pollute bbox/point_count/the written .ply, without also
discarding legitimate points on an object that's flush against a wall.
"""

import numpy as np

from gpu.stage_objects import POINT_ROOM_MARGIN_M, room_bbox_point_mask

ROOM_BBOX_MIN = np.array([-1.0, -0.3, -6.0])
ROOM_BBOX_MAX = np.array([2.5, 3.0, 0.5])


def test_points_outside_room_are_dropped():
    # Mirrors the real defect: a leaked tail meters beyond the far wall (room's own
    # z max is 0.5m - this point is ~11m past it, in line with the room.mp4 finding).
    leaked = np.array([[0.0, 1.0, 11.5]])
    mask = room_bbox_point_mask(leaked, ROOM_BBOX_MIN, ROOM_BBOX_MAX)
    assert mask.tolist() == [False]


def test_object_entirely_inside_room_is_unaffected():
    inside = np.array(
        [
            [0.0, 1.0, -3.0],
            [-0.5, 0.5, -1.0],
            [2.0, 2.0, 0.0],
        ]
    )
    mask = room_bbox_point_mask(inside, ROOM_BBOX_MIN, ROOM_BBOX_MAX)
    assert mask.tolist() == [True, True, True]


def test_point_within_margin_of_wall_is_kept():
    # A real object (e.g. a door frame) can legitimately have points just past the
    # room's point-cloud-derived boundary due to reconstruction noise, not a mask leak.
    just_past_wall = np.array([[0.0, 1.0, ROOM_BBOX_MAX[2] + POINT_ROOM_MARGIN_M / 2]])
    mask = room_bbox_point_mask(just_past_wall, ROOM_BBOX_MIN, ROOM_BBOX_MAX)
    assert mask.tolist() == [True]


def test_point_beyond_margin_is_dropped():
    just_beyond_margin = np.array([[0.0, 1.0, ROOM_BBOX_MAX[2] + POINT_ROOM_MARGIN_M * 2]])
    mask = room_bbox_point_mask(just_beyond_margin, ROOM_BBOX_MIN, ROOM_BBOX_MAX)
    assert mask.tolist() == [False]


def test_bbox_computed_from_filtered_points_excludes_leak():
    """The actual bug: bbox_min/bbox_max computed from unfiltered points carry the
    leaked tail (scene 68027f2a's door_0: 2.52m bbox depth for a real door). Filtering
    first must produce a bbox that reflects only the in-room points."""
    cluster = np.array(
        [
            [0.0, 1.0, -0.4],  # near the wall, real door geometry
            [0.05, 1.0, -0.3],
            [-0.05, 1.2, -0.35],
            [0.0, 1.0, 11.5],  # leaked point, same "cluster" pre-filter
        ]
    )
    mask = room_bbox_point_mask(cluster, ROOM_BBOX_MIN, ROOM_BBOX_MAX)
    filtered = cluster[mask]

    assert len(filtered) == 3
    bbox_depth = filtered[:, 2].max() - filtered[:, 2].min()
    assert bbox_depth < 0.5  # real door depth, not the 11.9m the unfiltered cluster would give


def test_margin_is_applied_symmetrically_on_every_axis():
    lo_corner = ROOM_BBOX_MIN - POINT_ROOM_MARGIN_M / 2
    hi_corner = ROOM_BBOX_MAX + POINT_ROOM_MARGIN_M / 2
    mask = room_bbox_point_mask(np.array([lo_corner, hi_corner]), ROOM_BBOX_MIN, ROOM_BBOX_MAX)
    assert mask.tolist() == [True, True]
