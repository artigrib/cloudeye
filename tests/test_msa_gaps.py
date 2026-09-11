"""Unit tests for scripts/msa/gaps.py on synthetic occupancy grids (SPEC.md A5,
section 11 "gap computation on synthetic grids")."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.msa.gaps import FREE, ObstacleSource, compute_gaps


def test_known_corridor_width_between_two_walls():
    # 40x20 grid, resolution 0.1m: two obstacle blocks with a free corridor of
    # exactly 6 cells = 0.6m between them (rows 0-19 blocked on both ends).
    grid_shape = (20, 40)
    passable = np.full(grid_shape, True)
    wall_a = np.zeros(grid_shape, dtype=bool)
    wall_b = np.zeros(grid_shape, dtype=bool)
    wall_a[:, :10] = True
    wall_b[:, 16:] = True
    passable[wall_a] = False
    passable[wall_b] = False

    sources = [
        ObstacleSource(id="wall_a", kind="wall", mask=wall_a),
        ObstacleSource(id="wall_b", kind="wall", mask=wall_b),
    ]
    gaps = compute_gaps(passable, sources, resolution=0.1, origin_x=0.0, origin_z=0.0)
    assert len(gaps) == 1
    gap = gaps[0]
    assert {gap.a_id, gap.b_id} == {"wall_a", "wall_b"}
    # Corridor is 6 cells = 0.6m wide; EDT-ridge estimate should land within one
    # cell of that.
    assert gap.width_m == pytest.approx(0.6, abs=0.11)


def test_platform_pass_fail_verdicts():
    grid_shape = (10, 30)
    passable = np.full(grid_shape, True)
    wall_a = np.zeros(grid_shape, dtype=bool)
    wall_b = np.zeros(grid_shape, dtype=bool)
    wall_a[:, :10] = True
    wall_b[:, 15:] = True  # 5-cell = 0.25m corridor at resolution 0.05
    passable[wall_a] = False
    passable[wall_b] = False

    sources = [
        ObstacleSource(id="wall_a", kind="wall", mask=wall_a),
        ObstacleSource(id="wall_b", kind="wall", mask=wall_b),
    ]
    gaps = compute_gaps(
        passable,
        sources,
        resolution=0.05,
        origin_x=0.0,
        origin_z=0.0,
        platforms={"burger": 0.20, "go2": 0.40},
    )
    assert len(gaps) == 1
    verdicts = {v.platform_id: v.fits for v in gaps[0].platform_verdicts}
    assert verdicts["burger"] is True  # 0.20 <= ~0.25m corridor
    assert verdicts["go2"] is False  # 0.40 > ~0.25m corridor


def test_no_gap_reported_for_non_adjacent_obstacles():
    # Three obstacles in a row: a and c never directly border a shared free
    # ridge closer than through b, so only (a,b) and (b,c) pairs should appear.
    grid_shape = (10, 60)
    passable = np.full(grid_shape, True)
    wall_a = np.zeros(grid_shape, dtype=bool)
    wall_b = np.zeros(grid_shape, dtype=bool)
    wall_c = np.zeros(grid_shape, dtype=bool)
    wall_a[:, :10] = True
    wall_b[:, 25:35] = True
    wall_c[:, 50:] = True
    passable[wall_a] = False
    passable[wall_b] = False
    passable[wall_c] = False

    sources = [
        ObstacleSource(id="a", kind="wall", mask=wall_a),
        ObstacleSource(id="b", kind="wall", mask=wall_b),
        ObstacleSource(id="c", kind="wall", mask=wall_c),
    ]
    gaps = compute_gaps(passable, sources, resolution=0.1, origin_x=0.0, origin_z=0.0)
    pairs = {frozenset((g.a_id, g.b_id)) for g in gaps}
    assert frozenset(("a", "b")) in pairs
    assert frozenset(("b", "c")) in pairs
    assert frozenset(("a", "c")) not in pairs


def test_empty_sources_returns_no_gaps():
    passable = np.full((5, 5), True)
    assert compute_gaps(passable, [], resolution=0.1, origin_x=0.0, origin_z=0.0) == []
