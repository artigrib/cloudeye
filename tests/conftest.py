"""Shared pytest fixtures. All tests here are pure-unit: no DB, no GPU, no network -
they run against the real (metadata-only) scene fixture captured from an actual GPU
pipeline run, checked into tests/fixtures/.
"""

from pathlib import Path

import pytest

FIXTURES_ROOT = Path(__file__).parent / "fixtures"


@pytest.fixture
def scene_horizontal_dir() -> Path:
    """A real scene's metadata artifacts (alignment.json, occupancy_meta.json,
    occupancy.npy, cameras_aligned.json, scene_objects/scene_objects.json) - captured
    from an actual end-to-end GPU pipeline run against a real room video. Point-cloud
    .ply binaries are deliberately not included (tests never read their content, only
    the path string)."""
    return FIXTURES_ROOT / "scene_horizontal"


@pytest.fixture
def scene_modular_home_dir() -> Path:
    """A real scene's metadata artifacts (alignment.json, occupancy_meta.json,
    occupancy.npy, cameras_aligned.json, scene_objects/scene_objects.json) for
    02_modular_home (scene_id afe4890d-0e5b-4261-891f-150ccb5e55a7, the 2026-09-05
    nightly-batch run - see docs/PROGRESS-footprint-planner.md / this PR's report) -
    used by tests/test_footprint.py as the one real-fixture check for the rectangle-
    footprint planner, alongside scene_horizontal above. 151x92 occupancy grid @
    0.05m resolution, 83 objects. Same "metadata only, no .ply binaries" convention as
    scene_horizontal_dir."""
    return FIXTURES_ROOT / "scene_modular_home"


@pytest.fixture
def scene_hotel_room2_dir() -> Path:
    """occupancy_meta.json + cameras_aligned.json from the hotel-own-batch room 2 scene
    (2026-09-01, id 4ae57385-5cfe-49b1-bc8d-e4c7882313e4) - the real scene whose
    `resolve_robot_start` crashed with `world_to_cell`'s "far outside the grid bounds"
    ValueError. Its first 5 of 47 camera poses (z from -0.001 to -0.605) sit outside the
    occupancy grid's z-range (grid z-max is -0.657) - the walkthrough starts a few
    frames before the operator steps into the reconstructed room. Only the two small
    JSON files this bug needs are checked in, not the full scene."""
    return FIXTURES_ROOT / "scene_hotel_room2"
