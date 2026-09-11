"""build_steps/execute_command plan from whatever start point they're given, not just
(0, 0) - the layer the frontend's double-click / plan-from-current-position feature
depends on. Pure-unit: build_steps takes plain SceneObject rows and an OccupancyGrid,
no DB needed.
"""

import uuid

import numpy as np
import pytest

from app.models import SceneObject
from app.services import command_service
from app.services.pathfinding import OccupancyGrid
from app.services.scene_ingest import GridMeta


def make_grid(rows: list[str], *, resolution=0.1) -> OccupancyGrid:
    """Same convention as test_pathfinding_astar.py's helper: rows are Z (top=iz=0),
    columns are X, one character per cell ('.' free, '#' obstacle)."""
    height = len(rows)
    width = len(rows[0])
    char_to_val = {".": 0, "#": 1}
    cells = np.zeros((width, height), dtype=np.uint8)
    for iz, row in enumerate(rows):
        for ix, ch in enumerate(row):
            cells[ix, iz] = char_to_val[ch]
    meta = GridMeta(resolution=resolution, origin_x=0.0, origin_z=0.0, width=width, height=height)
    return OccupancyGrid(cells=cells, meta=meta)


def make_object(name: str, x: float, z: float) -> SceneObject:
    return SceneObject(
        id=uuid.uuid4(),
        scene_id=uuid.uuid4(),
        name=name,
        pos_x=x, pos_y=0.0, pos_z=z,
        bbox_min_x=x - 0.05, bbox_min_y=0.0, bbox_min_z=z - 0.05,
        bbox_max_x=x + 0.05, bbox_max_y=1.0, bbox_max_z=z + 0.05,
        num_views=5,
        num_points=1000,
        is_fragment=False,
    )


def test_build_steps_goto_starts_from_given_point_not_origin():
    grid = make_grid(["." * 10 for _ in range(10)])
    target = make_object("sink", x=0.95, z=0.05)

    steps, _total_duration, _total_length = command_service.build_steps(
        "goto", (0.85, 0.85), target, None, grid, robot_radius_m=0.05, speed_mps=0.3,
    )

    path = steps[0]["path"]
    # The path must begin near the given start, not near the grid origin (0, 0) that
    # scene.robot_start would default to.
    assert path[0][0] == pytest.approx(0.85, abs=0.15)
    assert path[0][1] == pytest.approx(0.85, abs=0.15)


def test_build_steps_different_starts_produce_different_paths():
    grid = make_grid(["." * 10 for _ in range(10)])
    target = make_object("sink", x=0.05, z=0.95)

    steps_a, _, _ = command_service.build_steps(
        "goto", (0.05, 0.05), target, None, grid, robot_radius_m=0.05, speed_mps=0.3,
    )
    steps_b, _, _ = command_service.build_steps(
        "goto", (0.95, 0.05), target, None, grid, robot_radius_m=0.05, speed_mps=0.3,
    )

    assert steps_a[0]["path"][0] != steps_b[0]["path"][0]


def test_execute_command_uses_start_override_over_scene_robot_start(monkeypatch):
    """execute_command must plan from `start_override`, not scene.robot_start_x/z, when
    given one - the whole point of the "from" override."""
    captured: dict = {}

    def fake_build_steps(action, start, target_obj, dest_obj, grid, *, robot_radius_m, speed_mps, all_objects=()):
        captured["start"] = start
        return [{"type": "move", "path": [list(start)], "duration_sec": 1.0, "length_m": 1.0}], 1.0, 1.0

    monkeypatch.setattr(command_service, "build_steps", fake_build_steps)

    class FakeSession:
        async def commit(self):
            pass

        async def refresh(self, _obj):
            pass

    class FakeScene:
        id = uuid.uuid4()
        robot_start_x = 0.0
        robot_start_z = 0.0

    class FakeCommand:
        id = uuid.uuid4()
        user_text = "goto sink"
        parsed_action = None
        result = None
        status = "pending"
        error_message = None

    target = make_object("sink", x=1.0, z=1.0)

    import asyncio

    result = asyncio.run(
        command_service.execute_command(
            FakeSession(),
            FakeScene(),
            FakeCommand(),
            [target],
            make_grid(["." * 5 for _ in range(5)]),
            parsed_override={"action": "goto", "target": "sink", "destination": None},
            target_override=target,
            start_override=(0.42, 0.77),
        )
    )

    assert captured["start"] == (0.42, 0.77)
    assert result.status == "done"
