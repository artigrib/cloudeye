"""The robot platform registry (app/robots.py): closed-enum validity, on-disk asset
paths for real (non-placeholder) entries, and `resolve_radius_m`'s override precedence
- the function every reachability/path/USD-export call site now goes through instead
of reading `settings.robot_radius_m` directly.
"""

from pathlib import Path

import pytest

from app.config import settings
from app.robots import (
    DEFAULT_ROBOT_ID,
    ROBOTS,
    Kinematics,
    RadiusSource,
    get_robot,
    list_robots,
    resolve_footprint_m,
    resolve_radius_m,
)

# frontend/public/ - mesh_path/license_path are stored relative to this, same
# convention the frontend resolves them against.
FRONTEND_PUBLIC_ROOT = Path(__file__).parent.parent / "frontend" / "public"


def test_default_robot_id_names_a_real_entry():
    assert get_robot(DEFAULT_ROBOT_ID) is not None


def test_list_robots_returns_every_registered_platform():
    ids = {r.id for r in list_robots()}
    assert ids == {
        "burger", "limo", "waffle_pi", "turtlebot4", "jackal", "go2", "husky", "rosbot_xl_arm",
    }


def test_get_robot_unknown_id_returns_none():
    assert get_robot("not_a_real_platform") is None


@pytest.mark.parametrize("platform", ROBOTS, ids=[r.id for r in ROBOTS])
def test_kinematics_and_radius_source_are_valid_enum_members(platform):
    # Guards against a future edit assigning a plain string instead of an enum member -
    # dataclass field types aren't enforced at runtime, so this has to be checked
    # explicitly rather than relying on the type annotation alone.
    assert isinstance(platform.kinematics, Kinematics)
    assert isinstance(platform.radius_source, RadiusSource)


@pytest.mark.parametrize("platform", ROBOTS, ids=[r.id for r in ROBOTS])
def test_measured_platforms_have_real_mesh_and_license_files(platform):
    """Only platforms with a real (non-None) radius_m are expected to have their mesh
    already on disk - the seven PLACEHOLDER entries name where their mesh WILL live,
    not a file that exists yet (mesh measurement is a separate, later agent's job)."""
    if platform.radius_m is None:
        pytest.skip(f"{platform.id} is a placeholder - no mesh landed yet")
    assert (FRONTEND_PUBLIC_ROOT / platform.mesh_path).is_file()
    assert (FRONTEND_PUBLIC_ROOT / platform.license_path).is_file()


def test_placeholder_platforms_are_clearly_marked_not_measured():
    """Any entry still lacking a measured radius must say so plainly in `notes` -
    guards against a future edit leaving `radius_m=None` without explanation. As of
    the robots-integration pass all eight platforms have real measured/vendor data,
    so this currently has nothing to check but stays in place for the next platform
    added ahead of its own mesh-measurement pass."""
    for platform in ROBOTS:
        if platform.radius_m is None:
            assert "PLACEHOLDER" in platform.notes


def test_burger_matches_the_documented_vendor_nav2_value():
    burger = get_robot("burger")
    assert burger.radius_m == pytest.approx(0.10)
    assert burger.radius_source == RadiusSource.VENDOR_NAV2


# --- resolve_radius_m precedence -----------------------------------------------------


def test_resolve_radius_m_explicit_override_wins_over_platform():
    assert resolve_radius_m("burger", 0.33) == pytest.approx(0.33)


def test_resolve_radius_m_uses_platform_radius_when_no_override():
    assert resolve_radius_m("burger", None) == pytest.approx(0.10)


def test_resolve_radius_m_falls_back_to_settings_default_for_placeholder_platform():
    # As of the robots-integration pass every registered platform has a measured
    # radius_m (limo, previously used here as the example placeholder, was filled in
    # by that pass) - this test protects the *fallback path itself* for whichever
    # platform is a placeholder next, so it looks one up dynamically instead of
    # hardcoding an id that may since have been measured.
    placeholder = next((r for r in ROBOTS if r.radius_m is None), None)
    if placeholder is None:
        pytest.skip("no placeholder platform currently registered - all have a measured radius_m")
    assert resolve_radius_m(placeholder.id, None) == pytest.approx(settings.robot_radius_m)


def test_resolve_radius_m_falls_back_to_settings_default_for_unknown_platform():
    assert resolve_radius_m("not_a_real_platform", None) == pytest.approx(settings.robot_radius_m)


def test_resolve_radius_m_falls_back_to_settings_default_when_nothing_given():
    assert resolve_radius_m(None, None) == pytest.approx(settings.robot_radius_m)


# --- resolve_footprint_m precedence --------------------------------------------------


def test_resolve_footprint_m_explicit_override_wins_over_platform():
    assert resolve_footprint_m("burger", 1.0, 2.0) == (pytest.approx(1.0), pytest.approx(2.0))


def test_resolve_footprint_m_partial_override_falls_through_to_platform():
    # Only one of length/width given isn't a usable rectangle override - falls through
    # to the platform's own dimensions_m, same as passing (None, None).
    length_m, width_m = resolve_footprint_m("burger", 1.0, None)
    assert (length_m, width_m) == (pytest.approx(0.138), pytest.approx(0.178))


def test_resolve_footprint_m_uses_platform_dimensions_when_no_override():
    # burger.glb's dimensions_m (see app/robots.py) - mesh-measured, not radius-derived.
    assert resolve_footprint_m("burger", None, None) == (pytest.approx(0.138), pytest.approx(0.178))


@pytest.mark.parametrize("platform", ROBOTS, ids=[r.id for r in ROBOTS])
def test_resolve_footprint_m_matches_registered_dimensions_for_every_platform(platform):
    length_m, width_m = resolve_footprint_m(platform.id, None, None)
    if platform.dimensions_m is not None:
        assert length_m == pytest.approx(platform.dimensions_m.length_m)
        assert width_m == pytest.approx(platform.dimensions_m.width_m)
    else:
        # Placeholder platform: falls back to a circle-derived square, same
        # conservative fallback resolve_footprint_m documents for radius_m=None too.
        radius_m = resolve_radius_m(platform.id, None)
        assert length_m == pytest.approx(2 * radius_m)
        assert width_m == pytest.approx(2 * radius_m)


def test_resolve_footprint_m_falls_back_to_circle_derived_square_for_unknown_platform():
    length_m, width_m = resolve_footprint_m("not_a_real_platform", None, None)
    assert length_m == pytest.approx(2 * settings.robot_radius_m)
    assert width_m == pytest.approx(2 * settings.robot_radius_m)


def test_resolve_footprint_m_falls_back_to_circle_derived_square_when_nothing_given():
    length_m, width_m = resolve_footprint_m(None, None, None)
    assert length_m == pytest.approx(2 * settings.robot_radius_m)
    assert width_m == pytest.approx(2 * settings.robot_radius_m)
