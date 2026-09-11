"""Unit tests for gpu/plane_regularize.py's `regularize_planes` - the production port
of the Workstream A spike's `mode=project` result (see
docs/SPRINT-poisson-meshing.md and gpu/spikes/plane_regularize_spike.py). Runs in the
main app `uv` venv, which has neither open3d nor sklearn installed - `regularize_planes`
must therefore never be exercised with its real `_segment_plane_o3d` default here; every
test below supplies a fake `segment_plane` via the keyword test seam instead (this repo
uses zero unittest.mock/monkeypatch anywhere in tests/, by design - see the module's own
docstring).

The fake segmenter below deliberately recomputes inliers from whatever points subset
it's actually called with (never a precomputed global answer), so the local-index
contract `regularize_planes` relies on (indices 0..k-1 into the CURRENT remaining
subset, not global indices into the original cloud) is genuinely exercised rather than
accidentally satisfied.
"""

import numpy as np

from gpu.plane_regularize import MIN_INLIERS_ABSOLUTE, regularize_planes


def make_fake_segmenter(models, fake_dist_thresh=None):
    """Factory for a fake `segment_plane`. `models` is a list of ground-truth
    (a, b, c, d) plane tuples, popped off in order (one per call). Once the queue is
    empty, returns a dummy model with an empty inlier array - this is what drives
    `regularize_planes`'s "not enough inliers" loop-exit in several tests below.

    `fake_dist_thresh`, if given, overrides the `dist_thresh` argument the fake uses
    internally to decide inliers (used to simulate a segmenter with a looser internal
    threshold than what the caller actually passed - regularize_planes's own local
    guardrail is what's supposed to catch the mismatch, not this fake).

    The returned callable exposes `.calls` (an int count) so tests can assert it was
    never invoked (empty/tiny clouds).
    """
    queue = list(models)
    state = {"calls": 0}

    def segmenter(points, dist_thresh):
        state["calls"] += 1
        if not queue:
            return (1.0, 0.0, 0.0, 0.0), np.zeros(0, dtype=np.int64)
        a, b, c, d = queue.pop(0)
        abc = np.array([a, b, c], dtype=np.float64)
        norm = np.linalg.norm(abc)
        thresh = fake_dist_thresh if fake_dist_thresh is not None else dist_thresh
        signed = (points @ abc + d) / norm
        local_inlier_indices = np.flatnonzero(np.abs(signed) <= thresh)
        return (a, b, c, d), local_inlier_indices

    segmenter.calls = state
    return segmenter


def _n_calls(segmenter):
    return segmenter.calls["calls"]


def _tilted_plane_points(rng, n_pts, n0, d0, centroid, extent=1.0, noise_scale=0.02):
    """Sample n_pts points on/near the plane {x : n0.x + d0 = 0} (n0 assumed unit),
    passing through `centroid`, plus small noise strictly bounded to
    +/- noise_scale along n0 (never touching the default 0.04 dist_thresh)."""
    tmp = np.array([1.0, 0.0, 0.0]) if abs(n0[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n0, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(n0, u)
    u_coord = rng.uniform(-extent, extent, n_pts)
    v_coord = rng.uniform(-extent, extent, n_pts)
    noise = rng.uniform(-noise_scale, noise_scale, n_pts)
    pts = centroid[None, :] + u_coord[:, None] * u[None, :] + v_coord[:, None] * v[None, :] + noise[:, None] * n0[None, :]
    assert abs(n0 @ centroid + d0) < 1e-9  # sanity: centroid actually lies on the plane
    return pts, noise


def test_synthetic_noisy_plane_is_flattened():
    rng = np.random.default_rng(0)
    n0 = np.array([1.0, 2.0, 3.0])
    n0 /= np.linalg.norm(n0)
    centroid = np.array([0.5, 1.0, -0.3])
    d0 = -float(n0 @ centroid)
    points, _ = _tilted_plane_points(rng, 300, n0, d0, centroid, noise_scale=0.02)

    segmenter = make_fake_segmenter([(n0[0], n0[1], n0[2], d0)])
    out, reports = regularize_planes(points, dist_thresh=0.04, segment_plane=segmenter)

    assert len(reports) == 1
    assert reports[0]["n_inliers_projected"] == 300
    residual_after = out @ n0 + d0
    assert np.max(np.abs(residual_after)) < 1e-9


def test_far_away_blob_is_returned_bit_identical():
    rng = np.random.default_rng(1)
    blob = rng.uniform(-1, 1, size=(120, 3)) + np.array([50.0, 50.0, 50.0])
    # Empty model queue -> fake immediately returns 0 inliers -> loop exits before
    # ever touching `blob`.
    segmenter = make_fake_segmenter([])
    out, reports = regularize_planes(blob, dist_thresh=0.04, segment_plane=segmenter)

    assert reports == []
    assert np.array_equal(out, blob)


def test_moved_points_shift_only_along_normal_and_within_dist_thresh():
    rng = np.random.default_rng(2)
    n0 = np.array([0.2, -0.4, 0.9])
    n0 /= np.linalg.norm(n0)
    centroid = np.array([1.0, -2.0, 0.5])
    d0 = -float(n0 @ centroid)
    points, _ = _tilted_plane_points(rng, 200, n0, d0, centroid, noise_scale=0.03)

    tmp = np.array([1.0, 0.0, 0.0]) if abs(n0[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n0, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(n0, u)

    segmenter = make_fake_segmenter([(n0[0], n0[1], n0[2], d0)])
    out, reports = regularize_planes(points, dist_thresh=0.04, segment_plane=segmenter)
    assert reports[0]["n_inliers_projected"] == 200

    displacement = out - points
    # Parallel to the normal: cross product ~0 for every row.
    cross = np.cross(displacement, n0[None, :])
    assert np.max(np.abs(cross)) < 1e-9
    # In-plane coordinates (u, v) unchanged: displacement has ~zero component along
    # either in-plane basis vector.
    assert np.max(np.abs(displacement @ u)) < 1e-9
    assert np.max(np.abs(displacement @ v)) < 1e-9
    # No shift exceeds the configured distance threshold.
    shift_mag = np.linalg.norm(displacement, axis=1)
    assert np.max(shift_mag) <= 0.04 + 1e-12


def test_non_unit_normal_model_projects_onto_correct_plane():
    # Plane y=2, expressed with a non-unit-length normal (0, 2, 0, -4) i.e. 2y-4=0.
    # A d-normalization bug (using raw d with a normalized (a,b,c)) would instead
    # project points onto y=4 or y=0.
    rng = np.random.default_rng(3)
    points = np.column_stack(
        [
            rng.uniform(-5, 5, 150),
            2.0 + rng.uniform(-0.02, 0.02, 150),
            rng.uniform(-5, 5, 150),
        ]
    )
    segmenter = make_fake_segmenter([(0.0, 2.0, 0.0, -4.0)])
    out, reports = regularize_planes(points, dist_thresh=0.04, segment_plane=segmenter)

    assert len(reports) == 1
    assert reports[0]["n_inliers_projected"] == 150
    assert np.allclose(out[:, 1], 2.0, atol=1e-9)
    assert reports[0]["model_abcd"] == [0.0, 1.0, 0.0, -2.0]


def test_guardrail_rejects_inliers_beyond_dist_thresh():
    rng = np.random.default_rng(4)
    n0 = np.array([0.0, 0.0, 1.0])
    centroid = np.array([0.0, 0.0, 0.75])
    d0 = -0.75
    # Wide spread (up to 0.3m off-plane) - a real dist_thresh=0.04 caller would only
    # want the near ones, but the fake's internal threshold is deliberately looser
    # (0.5) so it hands back everything, exercising regularize_planes's own guardrail.
    points, _ = _tilted_plane_points(rng, 200, n0, d0, centroid, noise_scale=0.3)
    segmenter = make_fake_segmenter([(n0[0], n0[1], n0[2], d0)], fake_dist_thresh=0.5)

    out, reports = regularize_planes(points, dist_thresh=0.04, segment_plane=segmenter)

    true_dist = points @ n0 + d0
    within = np.abs(true_dist) <= 0.04
    beyond = ~within

    assert len(reports) == 1
    assert reports[0]["n_inliers_projected"] == int(within.sum())
    # Points within the real threshold got projected onto the plane.
    assert np.max(np.abs(out[within] @ n0 + d0)) < 1e-9
    # Points beyond the real threshold (even though the fake claimed them as
    # inliers) were left completely untouched.
    assert np.array_equal(out[beyond], points[beyond])


def test_min_inliers_floor_finds_plane_on_538_point_cloud():
    # Mirrors curtain_1.ply's real point count (538) from the spike's follow-up. A
    # ~300-point planar subset should be found now that the absolute floor is 30, not
    # the old buggy 500 (which made min_inlier_frac inert below ~50k points).
    rng = np.random.default_rng(5)
    n0 = np.array([0.0, 0.0, 1.0])
    d0 = -0.75
    planar = np.column_stack(
        [rng.uniform(-1, 1, 300), rng.uniform(-1, 1, 300), 0.75 + rng.uniform(-0.01, 0.01, 300)]
    )
    other = rng.uniform(-5, 5, size=(238, 3)) + np.array([20.0, 20.0, 20.0])
    points = np.vstack([planar, other])
    assert len(points) == 538

    segmenter = make_fake_segmenter([(n0[0], n0[1], n0[2], d0)])
    out, reports = regularize_planes(points, dist_thresh=0.04, min_inlier_frac=0.01, segment_plane=segmenter)

    assert len(reports) == 1
    assert reports[0]["n_inliers_projected"] == 300
    assert MIN_INLIERS_ABSOLUTE == 30  # documents the floor this test depends on


def test_min_inlier_frac_still_governs_on_large_cloud():
    # Companion to the 538-point test: on a 100k-point cloud, min_inlier_frac=0.01
    # requires 1000 inliers - a genuine 500-point planar subset should NOT be enough,
    # proving the fraction (not just the floor of 30) still gates larger clouds.
    rng = np.random.default_rng(6)
    n0 = np.array([0.0, 0.0, 1.0])
    d0 = -0.75
    planar = np.column_stack(
        [rng.uniform(-1, 1, 500), rng.uniform(-1, 1, 500), 0.75 + rng.uniform(-0.01, 0.01, 500)]
    )
    other = np.column_stack(
        [
            rng.uniform(-50, 50, 99_500),
            rng.uniform(-50, 50, 99_500),
            rng.uniform(2.0, 50.0, 99_500),  # always far from z=0.75
        ]
    )
    points = np.vstack([planar, other])
    assert len(points) == 100_000

    segmenter = make_fake_segmenter([(n0[0], n0[1], n0[2], d0)])
    out, reports = regularize_planes(points, dist_thresh=0.04, min_inlier_frac=0.01, segment_plane=segmenter)

    assert reports == []
    assert np.array_equal(out, points)


def test_loop_stops_at_max_planes():
    rng = np.random.default_rng(7)
    n_per = 80
    n_clusters = 5
    clusters = []
    models = []
    for i in range(n_clusters):
        z = 10.0 * i
        pts = np.column_stack(
            [rng.uniform(-1, 1, n_per), rng.uniform(-1, 1, n_per), z + rng.uniform(-0.01, 0.01, n_per)]
        )
        clusters.append(pts)
        models.append((0.0, 0.0, 1.0, -z))
    points = np.vstack(clusters)

    segmenter = make_fake_segmenter(models)  # 5 models queued
    out, reports = regularize_planes(points, dist_thresh=0.04, max_planes=3, segment_plane=segmenter)

    assert len(reports) == 3
    assert [r["plane_id"] for r in reports] == [0, 1, 2]
    # Clusters 4 and 5 (never reached - only 3 of the 5 queued planes were tried)
    # must be completely untouched.
    untried = points[3 * n_per :]
    assert np.array_equal(out[3 * n_per :], untried)


def test_loop_stops_early_when_a_later_plane_has_too_few_inliers():
    rng = np.random.default_rng(8)
    plane_a = np.column_stack(
        [rng.uniform(-1, 1, 100), rng.uniform(-1, 1, 100), 0.75 + rng.uniform(-0.01, 0.01, 100)]
    )
    # Leftover points scattered well away from both plane A's z and the second
    # queued (unmatched) model's z=5.0 - so after plane A is removed, the fake's
    # real recount for model 2 finds ~nothing.
    leftover = np.column_stack([rng.uniform(-1, 1, 50), rng.uniform(-1, 1, 50), rng.uniform(-1.5, -0.5, 50)])
    points = np.vstack([plane_a, leftover])

    segmenter = make_fake_segmenter([(0.0, 0.0, 1.0, -0.75), (0.0, 0.0, 1.0, -5.0)])
    out, reports = regularize_planes(points, dist_thresh=0.04, min_inlier_frac=0.01, segment_plane=segmenter)

    assert len(reports) == 1
    assert reports[0]["n_inliers_projected"] == 100
    # segment_plane WAS called a second time (and returned too few inliers) - confirm
    # the post-segment_plane exit path, not just the pre-check one, was exercised.
    assert _n_calls(segmenter) == 2


def test_point_count_and_row_order_preserved():
    rng = np.random.default_rng(9)
    n0 = np.array([0.0, 0.0, 1.0])
    d0 = -0.75
    points, _ = _tilted_plane_points(rng, 400, n0, d0, np.array([0.0, 0.0, 0.75]), noise_scale=0.01)
    ids = np.arange(len(points))

    segmenter = make_fake_segmenter([(0.0, 0.0, 1.0, -0.75)])
    out, reports = regularize_planes(points, dist_thresh=0.04, segment_plane=segmenter)

    assert out.shape == points.shape
    # Row-by-row correspondence: each output row's xy (untouched, since normal is
    # purely +z) still matches the id-indexed original row's xy.
    assert np.array_equal(out[ids, :2], points[ids, :2])


def test_input_array_is_not_mutated():
    rng = np.random.default_rng(10)
    n0 = np.array([0.0, 0.0, 1.0])
    d0 = -0.75
    points, _ = _tilted_plane_points(rng, 150, n0, d0, np.array([0.0, 0.0, 0.75]), noise_scale=0.01)
    src_before = points.copy()

    segmenter = make_fake_segmenter([(0.0, 0.0, 1.0, -0.75)])
    regularize_planes(points, dist_thresh=0.04, segment_plane=segmenter)

    assert np.array_equal(points, src_before)


def test_empty_and_tiny_clouds_never_call_segmenter():
    segmenter_empty = make_fake_segmenter([(0.0, 0.0, 1.0, -0.75)])
    empty = np.zeros((0, 3))
    out_empty, reports_empty = regularize_planes(empty, dist_thresh=0.04, segment_plane=segmenter_empty)
    assert out_empty.shape == (0, 3)
    assert reports_empty == []
    assert _n_calls(segmenter_empty) == 0

    segmenter_tiny = make_fake_segmenter([(0.0, 0.0, 1.0, -0.75)])
    tiny = np.array([[0.0, 0.0, 0.75], [0.1, 0.1, 0.76], [0.2, -0.1, 0.74], [0.0, 0.2, 0.75], [-0.1, 0.0, 0.75]])
    out_tiny, reports_tiny = regularize_planes(tiny, dist_thresh=0.04, segment_plane=segmenter_tiny)
    assert np.array_equal(out_tiny, tiny)
    assert reports_tiny == []
    assert _n_calls(segmenter_tiny) == 0


def test_report_dict_sanity():
    rng = np.random.default_rng(11)
    n0 = np.array([0.0, 0.0, 1.0])
    d0 = -0.75
    points, _ = _tilted_plane_points(rng, 250, n0, d0, np.array([0.0, 0.0, 0.75]), noise_scale=0.02)

    segmenter = make_fake_segmenter([(0.0, 0.0, 1.0, -0.75)])
    _, reports = regularize_planes(points, dist_thresh=0.04, segment_plane=segmenter)

    assert len(reports) == 1
    for i, r in enumerate(reports):
        assert r["plane_id"] == i
        assert len(r["model_abcd"]) == 4
        assert r["residual_max_m"] <= 0.04 + 1e-12
        assert r["residual_rms_m"] <= r["residual_max_m"] + 1e-12
        assert r["n_inliers_returned"] >= r["n_inliers_projected"]
