"""The contract the scene screen depends on: a scene answers, or says it cannot.

`GET /reachability` and `GET /map` return **200 with a verdict** or **200 with a documented
"layer not available"**. Never 5xx, and never 409 - a client rendering a scene has two
shapes to draw and no error branch to guess at.

This file exists because own_0901_161054 (the old prod scene, `4ae57385-...`) answered
**500** on tag `scene-screen-2026-09-09b` for every robot, and the cause was neither the
scene nor the layer: its first camera pose sits 0.62 m past the grid's z edge, and
`world_to_cell` raises past a 2-cell margin. A scene the operator started from the doorway
took the whole screen down.
"""

import json

import numpy as np
import pytest

from app.services import pathfinding


# --- the anchor chain, which is what actually failed -----------------------------------


def _meta(**kw):
    from app.services.scene_ingest import GridMeta
    base = dict(resolution=0.05, origin_x=-1.389, origin_z=-6.626, width=117, height=120)
    base.update(kw)
    return GridMeta(**base)


def test_the_own_0901_161054_camera_pose_no_longer_raises():
    """The exact point out of the tag -b traceback:

        ValueError: world point (0.000, -0.001) is far outside the grid bounds
                    (origin=(-1.389,-6.626), size=117x120 @ 0.05m)
    """
    meta = _meta()
    z_max = meta.origin_z + meta.height * meta.resolution
    assert -0.001 > z_max                       # genuinely outside, by 0.62 m
    assert pathfinding.clamp_anchor_to_grid((0.0, -0.001), meta) == (27, 119)


def test_reachability_answers_when_the_anchor_is_outside_the_grid():
    """End to end through compute_reachability, not just the clamp: an anchor past the
    edge produces a verdict, not an exception."""
    cells = np.zeros((20, 20), dtype=np.uint8)
    grid = pathfinding.OccupancyGrid(
        cells=cells,
        meta=_meta(resolution=0.1, origin_x=0.0, origin_z=0.0, width=20, height=20),
    )
    obj = pathfinding.ObjectFootprint(x=1.0, z=1.0, bbox_min_x=0.95, bbox_min_z=0.95,
                                      bbox_max_x=1.05, bbox_max_z=1.05)
    outside = (-0.62, 0.5)      # past the z=0 edge, the 161054 shape
    result = pathfinding.compute_reachability(grid, outside, [("a", obj)], robot_radius_m=0.0)
    assert result.start.status in {"original", "moved"}
    assert "a" in result.reachable_ids


# --- the two response shapes ----------------------------------------------------------


@pytest.fixture()
def scene_dir_with_layer(tmp_path):
    d = tmp_path / "layers" / "backfill"
    d.mkdir(parents=True)
    (d / "occupancy_meta.json").write_text(json.dumps({
        "resolution": 0.1, "origin_x": 0.0, "origin_z": 0.0, "width": 8, "height": 8,
        "source": "test",
    }))
    np.save(d / "occupancy.npy", np.zeros((8, 8), dtype=np.uint8))
    return tmp_path


def test_a_scene_with_no_layer_is_a_state_not_an_error(tmp_path):
    """`nav_layer.resolve` raising is what the router turns into the second shape. The
    message must name the command that produces a layer - an operator reading
    "layer not available" on a screen needs to know what to run."""
    from app.services.nav_layer import NavLayerUnavailableError, resolve

    with pytest.raises(NavLayerUnavailableError) as exc:
        resolve(tmp_path)
    assert "backfill_occupancy.py" in str(exc.value)


def test_the_no_layer_reachability_body_is_distinguishable_from_fits_nowhere():
    """Both are all-zero with start_status "none". `layer_available` is the only thing
    separating "this robot fits nowhere in this room" from "there is no room data at all",
    and a screen that rendered the second as "0 of 17 reachable" would be lying with
    arithmetic."""
    import uuid

    from app.routers.scenes import _no_layer_reachability

    body = _no_layer_reachability(uuid.uuid4(), "burger", 0.1, "no navigation layer under …")
    assert body.layer_available is False
    assert body.layer_unavailable_reason
    assert body.reachable_object_ids == [] and body.unreachable_reasons == {}
    assert body.start_status == "none" and body.reachable_cells == []


def test_a_scene_with_a_layer_reports_layer_available(scene_dir_with_layer):
    from app.services import nav_layer

    layer = nav_layer.resolve(scene_dir_with_layer)
    assert layer.meta.width == 8
