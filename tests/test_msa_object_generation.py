import numpy as np

from scripts.msa.object_generation import (
    bbox_volume,
    containment_ratio,
    crop_from_mask,
    match_best_detection,
    passes_containment,
    project_points_to_view,
    to_raw_frame,
)


def _identity_camera_looking_along_z():
    """camera_pose = camera-to-world identity: camera sits at world origin,
    looking down +Z, no rotation. A point at world (0, 0, 5) is 5m in front."""
    return np.eye(4)


def test_project_points_to_view_simple_pinhole():
    camera_pose = _identity_camera_looking_along_z()
    intrinsics = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])
    # Point directly ahead on the optical axis projects to the principal point.
    xyz = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]])
    uv, in_front = project_points_to_view(xyz, camera_pose, intrinsics)
    assert in_front.all()
    assert np.allclose(uv[0], [50, 50])
    # A point offset in +X projects to the right of the principal point.
    assert uv[1, 0] > uv[0, 0]


def test_project_points_to_view_behind_camera_excluded():
    camera_pose = _identity_camera_looking_along_z()
    intrinsics = np.eye(3)
    xyz = np.array([[0.0, 0.0, -1.0]])  # behind the camera
    _, in_front = project_points_to_view(xyz, camera_pose, intrinsics)
    assert not in_front[0]


def test_to_raw_frame_identity_transform_is_noop():
    R = np.eye(3)
    floor_y = 0.0
    aligned = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]])
    raw = to_raw_frame(aligned, R, floor_y)
    assert np.allclose(raw, aligned)


def test_to_raw_frame_restores_floor_offset():
    R = np.eye(3)
    floor_y = -1.5
    aligned = np.array([[0.0, 0.0, 0.0]])  # floor-zeroed in the aligned frame
    raw = to_raw_frame(aligned, R, floor_y)
    assert np.allclose(raw, [[0.0, floor_y, 0.0]])


def test_to_raw_frame_inverts_rotation():
    # A rotation R with raw = aligned @ R must satisfy: if aligned = raw @ R.T,
    # then to_raw_frame(aligned, R, 0) recovers the original raw point.
    theta = 0.3
    R = np.array([[np.cos(theta), 0, -np.sin(theta)], [0, 1, 0], [np.sin(theta), 0, np.cos(theta)]])
    raw_true = np.array([[2.0, 0.0, 5.0]])
    aligned = raw_true @ R.T
    raw_recovered = to_raw_frame(aligned, R, 0.0)
    assert np.allclose(raw_recovered, raw_true, atol=1e-9)


def test_match_best_detection_picks_closest_and_ignores_far_or_tiny():
    intrinsics = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])
    view_data = {"v0": (np.eye(4), intrinsics, (100, 100))}
    # Object centroid at world (0,0,2) projects to (50,50) - the principal point.
    centroid_raw = np.array([0.0, 0.0, 2.0])
    detections = [
        {"view_id": "v0", "centroid_uv": [52.0, 51.0], "npix": 500},  # close match
        {"view_id": "v0", "centroid_uv": [90.0, 90.0], "npix": 500},  # far, should lose
        {"view_id": "v0", "centroid_uv": [50.5, 50.5], "npix": 5},  # closer but too small (noise)
    ]
    match = match_best_detection(centroid_raw, detections, view_data, min_npix=200)
    assert match["centroid_uv"] == [52.0, 51.0]
    assert match["match_dist_px"] < 5.0


def test_match_best_detection_none_when_nothing_close_enough():
    intrinsics = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])
    view_data = {"v0": (np.eye(4), intrinsics, (100, 100))}
    centroid_raw = np.array([0.0, 0.0, 2.0])
    detections = [{"view_id": "v0", "centroid_uv": [99.0, 99.0], "npix": 500}]
    assert match_best_detection(centroid_raw, detections, view_data, max_dist_px=10.0) is None


def test_match_best_detection_ignores_view_not_in_view_data():
    intrinsics = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])
    view_data = {"v0": (np.eye(4), intrinsics, (100, 100))}
    centroid_raw = np.array([0.0, 0.0, 2.0])
    detections = [{"view_id": "v_missing", "centroid_uv": [50.0, 50.0], "npix": 500}]
    assert match_best_detection(centroid_raw, detections, view_data) is None


def test_crop_from_mask_bbox_and_alpha():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 0] = 128
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 30:50] = 255
    rgba = crop_from_mask(img, mask, pad_frac=0.0)
    assert rgba is not None
    assert rgba.shape == (19, 19, 4)  # max index inclusive: mask[40:60,30:50] -> xs/ys max at 49/59
    assert (rgba[..., 3] == 255).all()  # no padding, whole crop is the mask


def test_crop_from_mask_empty_returns_none():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)
    assert crop_from_mask(img, mask) is None


def test_bbox_volume():
    assert bbox_volume([0, 0, 0], [2, 3, 4]) == 24.0
    assert bbox_volume([1, 1, 1], [0, 0, 0]) == 0.0  # inverted bbox clamps to 0


def test_containment_ratio_and_pass():
    # Measured object is a cube; generated mesh is the same cube shape at a
    # totally different absolute scale (TRELLIS.2's own frame is arbitrary,
    # not metric) - a scale-invariant shape comparison must still PASS.
    ratio = containment_ratio([0, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 1])
    assert ratio == 1.0
    assert passes_containment(ratio)

    tiny_but_same_shape = containment_ratio([0, 0, 0], [0.1, 0.1, 0.1], [0, 0, 0], [50, 50, 50])
    assert tiny_but_same_shape == 1.0
    assert passes_containment(tiny_but_same_shape)

    # Generated mesh proportions way off (near-flat slab) vs a real cube-ish
    # measured object - reject on SHAPE, not absolute size.
    flat_ratio = containment_ratio([0, 0, 0], [1, 1, 0.02], [0, 0, 0], [1, 1, 1])
    assert passes_containment(flat_ratio) is False

    # Generated mesh far more elongated than the measured object - reject.
    elongated_ratio = containment_ratio([0, 0, 0], [1, 0.05, 0.05], [0, 0, 0], [1, 0.9, 0.8])
    assert passes_containment(elongated_ratio) is False


def test_containment_ratio_orientation_invariant():
    # Same elongated shape, axes permuted (TRELLIS has no guaranteed axis
    # correspondence to the measured object's world-frame axes) - must still
    # compare as identical shapes.
    a = containment_ratio([0, 0, 0], [2, 1, 0.5], [0, 0, 0], [2, 1, 0.5])
    b = containment_ratio([0, 0, 0], [0.5, 2, 1], [0, 0, 0], [2, 1, 0.5])
    assert a == b == 1.0


def test_containment_ratio_zero_measured_volume_is_infinite_and_rejected():
    ratio = containment_ratio([0, 0, 0], [1, 1, 1], [0, 0, 0], [0, 0, 0])
    assert ratio == float("inf")
    assert passes_containment(ratio) is False
