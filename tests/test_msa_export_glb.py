"""Stage E0 (SPEC §8): GLB node hierarchy mirrors the USD prim hierarchy -
visual/collision split per object, Plan, optional Path/Target."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from scripts.msa.bootstrap import load_object_inputs_from_hulls_json, run_bootstrap
from scripts.msa.export_glb import build_scene
from scripts.msa.geometry import WallPolygon

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "msa" / "02_modular_home"


@pytest.fixture(scope="module")
def bootstrap_out(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("msa_e0_glb")
    inputs = load_object_inputs_from_hulls_json(FIXTURE_DIR / "scene_objects_hulls.json")
    run_bootstrap(FIXTURE_DIR, out_dir, object_inputs=inputs)
    return out_dir


class TestGlbHierarchy:
    def test_glb_round_trips_with_visual_and_collision_nodes(self, bootstrap_out):
        import trimesh

        scene = trimesh.load(str(bootstrap_out / "scene.glb"))
        nodes = set(scene.graph.nodes)
        object_group_nodes = {n for n in nodes if "/" not in n and n not in ("wall_0", "wall_1", "wall_2", "wall_3", "wall_outline", "floor", "Plan", "Path", "Target", "world")}
        assert object_group_nodes, "expected at least one object group node"
        sample = next(iter(object_group_nodes))
        assert f"{sample}/visual" in nodes
        assert f"{sample}/collision" in nodes or not any(n.startswith(f"{sample}/collision") for n in nodes)

    def test_single_wall_band_replaces_per_fragment_walls(self, bootstrap_out):
        """T15g: one `wall_outline` node (the regularized room outline buffered
        by WALL_THICKNESS_M) is the only wall mesh; the raw fragments keep
        their `Plan/wall_i` curves."""
        import trimesh

        scene = trimesh.load(str(bootstrap_out / "scene.glb"))
        nodes = set(scene.graph.nodes)
        assert "wall_outline" in nodes
        assert not any(n.startswith("wall_") and n != "wall_outline" for n in nodes)
        assert any(n.startswith("Plan/wall_") or n.startswith("Planwall_") for n in nodes)

    def test_plan_nodes_present(self, bootstrap_out):
        import trimesh

        scene = trimesh.load(str(bootstrap_out / "scene.glb"))
        nodes = set(scene.graph.nodes)
        assert any(n.startswith("Plan/") for n in nodes)


class TestPathAndTargetNodes:
    def test_present_when_provided(self):
        wall = WallPolygon(vertices=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], area_m2=1.0)
        scene = build_scene(
            [wall],
            None,
            0.0,
            2.5,
            [],
            path_points_world=[(0.0, 0.05, 0.0), (1.0, 0.05, 1.0)],
            target_point=(1.0, 0.05, 1.0),
        )
        assert "Path" in scene.graph.nodes
        assert "Target" in scene.graph.nodes
        # morning-4: `Path` is the 3 cm ribbon mesh at floor + 0.03 m, the polyline rides as `Path_curve`
        ribbon = scene.geometry[scene.graph["Path"][1]]
        assert isinstance(ribbon, trimesh.Trimesh) and len(ribbon.faces) == 4  # one segment, both windings
        assert np.allclose(ribbon.vertices[:, 1], 0.03)
        assert ribbon.extents[[0, 2]].tolist() == pytest.approx([1.0 + 0.03 * np.cos(np.pi / 4)] * 2, abs=1e-6)  # diagonal run + width
        assert "Path_curve" in scene.graph.nodes

    def test_path_ribbon_mesh_geometry(self):
        from scripts.msa.export_glb import path_ribbon_mesh

        ribbon = path_ribbon_mesh([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 0.0, 1.0)], 0.5, width_m=0.03)
        assert ribbon is not None and np.allclose(ribbon.vertices[:, 1], 0.53)
        # width 3 cm on the straight run; the corner is mitred
        xs = ribbon.vertices[:, 0]
        zs = ribbon.vertices[:, 2]
        assert zs[[0, 3]].tolist() == pytest.approx([0.015, -0.015])  # left/right of the first vertex
        assert xs.max() == pytest.approx(2.015, abs=1e-6) and xs.min() == pytest.approx(0.0)
        assert len(ribbon.faces) == 2 * 4
        assert path_ribbon_mesh([(0.0, 0.0, 0.0)], 0.0) is None and path_ribbon_mesh([(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)], 0.0) is None

    def test_absent_when_not_provided(self):
        wall = WallPolygon(vertices=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], area_m2=1.0)
        scene = build_scene([wall], None, 0.0, 2.5, [])
        assert "Path" not in scene.graph.nodes
        assert "Target" not in scene.graph.nodes
