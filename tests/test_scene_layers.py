"""Matching a scene to its nvblox layer pack.

The rule is the whole test: a pack is served only when the grid it is sampled on equals
the scene's own grid exactly. Names are not the key and must not become one - the hero
video produced two reconstructions whose grids differ in origin, shape and frame, and one
pack directory carries an array for each. Getting this wrong does not fail loudly; it
draws a map half a metre out of place, under a route that is correct.
"""

import json

import numpy as np
import pytest

from app.services.scene_ingest import GridMeta
from app.services.scene_layers import find_pack_for_grid

OUR = {"cell_m": 0.05, "shape_uv": [4, 3], "u_range": [-1.5, 0.0], "v_range": [-6.0, 0.0],
       "height_above_floor_m": 0.3}
FRONT = {"cell_m": 0.05, "origin_x": 2.25, "origin_z": -0.5, "shape_xz": [5, 2]}


def _pack(tmp_path, name="own_test", *, our=OUR, front=FRONT, arrays=True, quality=True):
    d = tmp_path / name
    d.mkdir(parents=True)
    meta = {"our_grid": our, "frontend_grid": front}
    if quality:
        meta["frontend_grid_resample"] = {"quality": {"verdict": "APPROXIMATE", "why": "two solves"}}
    (d / "grid_meta.json").write_text(json.dumps(meta))
    if arrays:
        np.save(d / "esdf_slice_0.3m.npy", np.full((our["shape_uv"][0], our["shape_uv"][1]), 0.2, np.float32))
        np.save(d / "unobserved_mask.npy", np.zeros((our["shape_uv"][0], our["shape_uv"][1]), bool))
        np.save(d / "esdf_slice_frontend_grid.npy",
                np.full((front["shape_xz"][0], front["shape_xz"][1]), 0.1, np.float32))
        np.save(d / "unobserved_mask_frontend_grid.npy",
                np.ones((front["shape_xz"][0], front["shape_xz"][1]), bool))
    return d


def _grid(**kw):
    base = dict(resolution=0.05, origin_x=-1.5, origin_z=-6.0, width=4, height=3)
    base.update(kw)
    return GridMeta(**base)


def test_matches_our_grid_by_cell_origin_and_shape(tmp_path):
    _pack(tmp_path)
    pack = find_pack_for_grid(_grid(), root=tmp_path)
    assert pack is not None
    assert (pack.variant, pack.name) == ("our_grid", "own_test")
    assert pack.esdf_m.shape == (4, 3)
    assert pack.slice_height_m == 0.3
    # our_grid is the pack's own reconstruction, so it carries no resample caveat.
    assert pack.quality is None


def test_matches_the_frontend_grid_and_passes_its_verdict_through(tmp_path):
    _pack(tmp_path)
    pack = find_pack_for_grid(_grid(origin_x=2.25, origin_z=-0.5, width=5, height=2), root=tmp_path)
    assert pack is not None
    assert pack.variant == "frontend_grid"
    assert pack.unobserved.all()
    # The pack's own words about its own resample, not a summary of them.
    assert pack.quality is not None and "APPROXIMATE" in pack.quality


@pytest.mark.parametrize(
    "override",
    [
        {"origin_x": -1.5001},   # 0.1 mm out: two hundredths of a cell, and still wrong
        {"origin_z": -5.9999},
        {"resolution": 0.049},
        {"width": 5},
        {"height": 4},
    ],
    ids=["origin_x", "origin_z", "cell", "width", "height"],
)
def test_any_disagreement_at_all_is_no_match(tmp_path, override):
    _pack(tmp_path)
    assert find_pack_for_grid(_grid(**override), root=tmp_path) is None


def test_a_near_miss_on_one_axis_does_not_borrow_the_other_grids_pack(tmp_path):
    # The real case this guards: a second solve of the same room whose origin_z is the
    # NEGATED far edge rather than the near one. Every other field agrees.
    _pack(tmp_path)
    assert find_pack_for_grid(_grid(origin_z=-(0.0))) is None
    assert find_pack_for_grid(_grid(origin_z=0.0), root=tmp_path) is None


def test_arrays_that_contradict_the_metadata_are_refused(tmp_path):
    d = _pack(tmp_path, arrays=False)
    np.save(d / "esdf_slice_0.3m.npy", np.zeros((9, 9), np.float32))
    np.save(d / "unobserved_mask.npy", np.zeros((9, 9), bool))
    assert find_pack_for_grid(_grid(), root=tmp_path) is None


def test_a_missing_root_or_an_unreadable_pack_is_none_not_an_error(tmp_path):
    assert find_pack_for_grid(_grid(), root=tmp_path / "nope") is None
    bad = tmp_path / "broken"
    bad.mkdir()
    (bad / "grid_meta.json").write_text("{ not json")
    assert find_pack_for_grid(_grid(), root=tmp_path) is None


def test_packs_are_scanned_in_a_stable_order(tmp_path):
    # Two packs, both claiming the same grid: whichever sorts first wins, every time.
    _pack(tmp_path, "own_b")
    _pack(tmp_path, "own_a")
    assert find_pack_for_grid(_grid(), root=tmp_path).name == "own_a"
    assert find_pack_for_grid(_grid(), root=tmp_path).name == "own_a"
