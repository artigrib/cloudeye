"""Registry of robot platforms CloudEye can plan reachability/paths for.

Single source of truth mapping a platform id to the data the rest of the backend
needs: its physical radius (for `pathfinding.inflate`'s obstacle dilation - see
`resolve_radius_m` below), its 3D-viewer assets (`mesh_path`/`license_path`, resolved
by the frontend relative to `frontend/public/`), and where its radius came from.

Chosen as a plain Python module, not a JSON file: this project already keeps a closed
set of values as a typed Python construct rather than a free string wherever one
consumer needs to trust another's spelling (`CommandStep.type: Literal["move", "pick",
"place"]` in schemas.py; `ReachabilityResult.unreachable_reasons` values are documented
as exactly "outside_grid"/"disconnected"/"robot_does_not_fit" even though the dict
itself is `dict[str, str]`). `Kinematics`/`RadiusSource` follow that same pattern here,
as real `enum.Enum` members instead of a docstring's word of honor - a JSON registry
would need a second, separate schema (JSON Schema or a manual validator) to get the
same guarantee, for eight records that only ever change by a person editing this file
directly. A plain module also lets `dimensions_m` be a real dataclass instead of an ad
hoc dict shape.

Real, vendor/mesh-sourced numbers exist for all eight platforms as of the
robots-integration pass (see ~/reports/mesh/*.json for each platform's raw
measurement report and ~/reports/P.md for the consolidated writeup, radius-source
rationale, and vendor-vs-mesh discrepancy table). `burger`'s numbers predate that
pass (see ROBOT_RADIUS_M's long comment in app/config.py for how its 0.10m was
chosen); the other seven were filled in from `frontend/public/models/*.glb` mesh
measurements plus, where available, vendor Nav2 config values - each entry's `notes`
cites its source. Should a future platform be added before its mesh lands, follow
the same PLACEHOLDER convention this file used previously: `radius_m`/`dimensions_m`
left `None`, `notes` saying so, `mesh_path`/`license_path` naming the files the
measurement pass is expected to land under `frontend/public/models/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.config import settings


class Kinematics(str, Enum):
    """Locomotion type - closed set, not a free string (see module docstring)."""

    DIFFERENTIAL_DRIVE = "differential_drive"
    SKID_STEER = "skid_steer"
    ACKERMANN = "ackermann"
    LEGGED = "legged"
    MOBILE_MANIPULATOR = "mobile_manipulator"


class RadiusSource(str, Enum):
    """Where `radius_m` came from (or, for a placeholder, is expected to come from
    once measured) - closed set, not a free string (see module docstring)."""

    VENDOR_NAV2 = "vendor_nav2"  # taken from the vendor's own Nav2 config
    MESH_MEASURED = "mesh_measured"  # half the diagonal of the final mesh's bbox
    GAIT_APPROXIMATE = "gait_approximate"  # quadruped: estimated from body footprint


@dataclass(frozen=True)
class RobotDimensions:
    """Overall footprint, meters. `None` on the platform record until measured."""

    length_m: float
    width_m: float
    height_m: float


@dataclass(frozen=True)
class RobotPlatform:
    """One registry entry. Field order matches the shape described in the robots-
    integration task: id, display_name, vendor, kinematics, mesh_path, license_path,
    dimensions_m, radius_m, radius_source, reach_m (nullable), notes."""

    id: str
    display_name: str
    vendor: str
    kinematics: Kinematics
    mesh_path: str  # relative to frontend/public/, e.g. "models/turtlebot3_burger.glb"
    license_path: str  # same convention as mesh_path
    dimensions_m: RobotDimensions | None
    radius_m: float | None  # None = not yet measured/sourced (placeholder)
    radius_source: RadiusSource
    reach_m: float | None = None  # manipulator reach; only meaningful for an arm-equipped platform
    #: Height for NAVIGATION - what the robot can drive under. Usually the same as
    #: dimensions_m.height_m, but not always: dimensions_m records the MESH as measured
    #: (for go2, the trunk-only bbox posed at the URDF's zero/straight-legs angles), and
    #: that is not the height a body actually presents when the machine is standing. Set
    #: this only where the two genuinely differ, and say why in `notes`. None falls back
    #: to dimensions_m.height_m - see resolve_height_m.
    nav_height_m: float | None = None
    notes: str = ""


DEFAULT_ROBOT_ID = "burger"

ROBOTS: list[RobotPlatform] = [
    RobotPlatform(
        id="burger",
        display_name="TurtleBot3 Burger",
        vendor="ROBOTIS",
        kinematics=Kinematics.DIFFERENTIAL_DRIVE,
        mesh_path="models/turtlebot3_burger.glb",
        license_path="models/turtlebot3_burger.LICENSE.txt",
        dimensions_m=RobotDimensions(length_m=0.138, width_m=0.178, height_m=0.192),
        radius_m=0.10,
        radius_source=RadiusSource.VENDOR_NAV2,
        notes=(
            "turtlebot3_navigation2/param/burger.yaml, robot_radius: 0.1 (identical on "
            "humble/jazzy/master) - see app/config.py's robot_radius_m comment for the "
            "full inflate()-cell-boundary rationale for keeping exactly this value."
        ),
    ),
    RobotPlatform(
        id="limo",
        display_name="AgileX LIMO",
        vendor="AgileX Robotics",
        kinematics=Kinematics.ACKERMANN,
        mesh_path="models/limo.glb",
        license_path="models/limo.LICENSE.txt",
        dimensions_m=RobotDimensions(length_m=0.3215, width_m=0.2173, height_m=0.2514),
        radius_m=0.1940,
        radius_source=RadiusSource.MESH_MEASURED,
        notes=(
            "half_diagonal_m of the final limo.glb footprint (321.5x217.3mm rectangle, "
            "limo_four_diff.xacro configuration - AgileX also ships an Ackermann-steering "
            "xacro variant of the same base mesh, not modelled separately here). No "
            "robot_radius/circumscribed-radius found anywhere in limo_bringup's costmap "
            "params (both diff and ackerman configs use a rectangular footprint polygon "
            "instead, [[-0.16,-0.11],[-0.16,0.11],[0.16,0.11],[0.16,-0.11]] + 0.02m "
            "padding - essentially the same box as the mesh bbox), so vendor_radius_m is "
            "unset and the measured half-diagonal is used per mesh_measured. Source: "
            "~/reports/mesh/limo.json (agilexrobotics/limo_ros@4c78efc, dimensions_match=true)."
        ),
    ),
    RobotPlatform(
        id="waffle_pi",
        display_name="TurtleBot3 Waffle Pi",
        vendor="ROBOTIS",
        kinematics=Kinematics.DIFFERENTIAL_DRIVE,
        mesh_path="models/waffle_pi.glb",
        license_path="models/waffle_pi.LICENSE.txt",
        dimensions_m=RobotDimensions(length_m=0.2739, width_m=0.3062, height_m=0.1410),
        radius_m=0.15,
        radius_source=RadiusSource.VENDOR_NAV2,
        notes=(
            "turtlebot3_navigation2/param/waffle_pi.yaml, robot_radius: 0.15 (default/"
            "main branch, commit fc817ce3073af1d6032397c64504134882af5e9a). CAVEAT: "
            "turtlebot3_navigation2/param/humble/waffle_pi.yaml overrides this to "
            "robot_radius: 0.22 for the same robot on the humble branch - a real "
            "upstream inconsistency, not a transcription error (see M-waffle_pi's mesh "
            "report). Kept the top-level/default 0.15 value as vendor_radius_m per the "
            "mesh agent's own choice; neither vendor figure matches the mesh-measured "
            "half_diagonal_m of 0.2054m - see robots-integration report for the full "
            "vendor-vs-mesh discrepancy table. Source: ~/reports/mesh/waffle_pi.json "
            "(ROBOTIS-GIT/turtlebot3@fc817ce, dimensions_match=true)."
        ),
    ),
    RobotPlatform(
        id="turtlebot4",
        display_name="TurtleBot 4",
        vendor="Clearpath Robotics (iRobot Create 3 base)",
        kinematics=Kinematics.DIFFERENTIAL_DRIVE,
        mesh_path="models/turtlebot4.glb",
        license_path="models/turtlebot4.LICENSE.txt",
        dimensions_m=RobotDimensions(length_m=0.3415, width_m=0.3382, height_m=0.3467),
        radius_m=0.175,
        radius_source=RadiusSource.VENDOR_NAV2,
        notes=(
            "turtlebot4_navigation/config/nav2.yaml, lines 152 and 191 (local_costmap "
            "and global_costmap ros__parameters), branch humble, commit "
            "a6ee13b63cbd524500cc6a68cd20dbef69f326a9 of turtlebot/turtlebot4: literal "
            "line 'robot_radius: 0.175'. Not present on the default jazzy branch (used "
            "for the mesh/URDF), which instead defines an 8-point footprint polygon "
            "with circumradius 0.189m at the same location - 0.175 is the only scalar "
            "robot_radius found upstream, so it is used here despite the branch split. "
            "Mesh-measured half_diagonal_m is 0.24m, noticeably larger - see the "
            "robots-integration report's vendor-vs-mesh discrepancy table. Source: "
            "~/reports/mesh/turtlebot4.json (dimensions_match=true)."
        ),
    ),
    RobotPlatform(
        id="jackal",
        display_name="Clearpath Jackal",
        vendor="Clearpath Robotics",
        kinematics=Kinematics.SKID_STEER,
        mesh_path="models/jackal.glb",
        license_path="models/jackal.LICENSE.txt",
        dimensions_m=RobotDimensions(length_m=0.511, width_m=0.430, height_m=0.249),
        radius_m=0.334,
        radius_source=RadiusSource.MESH_MEASURED,
        notes=(
            "half_diagonal_m of the final jackal.glb footprint (510.6x430.0mm "
            "base+wheels+fenders rectangle). No robot_radius or circumscribed/inscribed "
            "radius value found anywhere in jackal_navigation's costmap/move_base "
            "params - only a rectangular nav footprint polygon "
            "([[-0.21,-0.165],[-0.21,0.165],[0.21,0.165],[0.21,-0.165]], the bare-chassis "
            "collision box, smaller than the full mesh because it excludes the fenders) - "
            "so vendor_radius_m is unset and the measured half-diagonal is used per "
            "mesh_measured. Source: ~/reports/mesh/jackal.json "
            "(jackal/jackal@4ddf9b5, dimensions_match=true)."
        ),
    ),
    RobotPlatform(
        id="go2",
        display_name="Unitree Go2",
        vendor="Unitree Robotics",
        kinematics=Kinematics.LEGGED,
        mesh_path="models/go2.glb",
        license_path="models/go2.LICENSE.txt",
        dimensions_m=RobotDimensions(length_m=0.4600, width_m=0.1940, height_m=0.1847),
        radius_m=0.2496,
        radius_source=RadiusSource.GAIT_APPROXIMATE,
        # 0.40 m, the VENDOR STANDING height (700x310x400mm), not the 0.1847 m mesh figure
        # beside it. dimensions_m is the trunk-only bbox at the URDF's zero/straight-legs
        # joint angles - a pose the robot is never in while driving - so using it as "what
        # can this robot fit under" would let a Go2 walk under a 0.25 m shelf it would hit.
        # The vendor spec was already recorded in `notes` below; this promotes it to a
        # field the navigation layer reads. Crouching is 0.20 m and is NOT used: a robot
        # that could crouch under an obstacle is not a robot that will.
        nav_height_m=0.40,
        notes=(
            "Quadruped - no vendor Nav2 footprint radius exists. radius_m is half the "
            "diagonal of the TRUNK-ONLY bbox (460.0x194.0mm), a corpus/worst-case "
            "approximation of how wide a gap the body could fit through, not a real "
            "gait/footstep-planning clearance radius - real traversability is governed "
            "by gait planning, not a static circle (see M-go2's CIRCULAR-FOOTPRINT "
            "CAVEAT). dimensions_m above is likewise the trunk/corpus-only extent (legs "
            "excluded), posed at the URDF's zero/straight-legs joint angle - NOT the "
            "vendor's whole-robot-with-legs-deployed spec (700x310x400mm standing / "
            "760x310x200mm crouching), which is why dimensions_match is false in the "
            "source report (an apples-to-oranges comparison, not a units bug). Source: "
            "~/reports/mesh/go2.json (unitreerobotics/unitree_ros@7d6075f, "
            "dimensions_match=false, explained above)."
        ),
    ),
    RobotPlatform(
        id="husky",
        display_name="Husky A200",
        vendor="Clearpath Robotics",
        kinematics=Kinematics.SKID_STEER,
        mesh_path="models/husky.glb",
        license_path="models/husky.LICENSE.txt",
        dimensions_m=RobotDimensions(length_m=0.985, width_m=0.6693, height_m=0.3963),
        radius_m=0.5528,
        radius_source=RadiusSource.MESH_MEASURED,
        notes=(
            "dimensions_m is the full rendered husky.glb assembly (base_link + "
            "top_chassis + top_plate + bumpers + wheels), matching Clearpath's "
            "published 990x670x390mm with-bumpers spec to within ~0.5-1.6%. radius_m, "
            "however, is half_diagonal_m computed WITHOUT bumpers (880.0x669.2mm "
            "rectangle -> 0.5528m), per the mesh agent's 'exclude bumpers/attachments "
            "where separable' convention - the with-bumpers half-diagonal would be "
            "~0.5954m (~8% larger); flagged here since a consumer wanting maximum "
            "collision-safety margin may want the larger figure instead. No scalar "
            "robot_radius/circumscribed_radius found anywhere in the husky/husky repo - "
            "husky_navigation/config/costmap_common.yaml defines a rectangular nav "
            "footprint polygon instead ([[-0.5,-0.33],...], padding 0.01), which "
            "corroborates the mesh measurement (0.34m padded half-width vs. this "
            "build's 0.3346m) but is a polygon, not a scalar, so vendor_radius_m stays "
            "unset per mesh_measured. PRACTICAL CONSEQUENCE for /reachability: since "
            "pathfinding.inflate dilates obstacles by this radius_m (bumpers "
            "excluded), the reachable/unreachable classification it returns for Husky "
            "is correspondingly optimistic - cells right at an obstacle's inflated "
            "boundary can be reported reachable even though the ~8% larger "
            "with-bumpers footprint would actually collide there. This is the Husky A200 (classic), not the newer "
            "A300 - see M-husky's REVISION DETERMINATION. NAMING: the on-disk mesh file "
            "was originally built as husky_a200.glb by the mesh-measurement agent and "
            "renamed to husky.glb to match this registry's mesh_path convention (no "
            "content change). Source: ~/reports/mesh/husky.json "
            "(husky/husky@41e15d2, dimensions_match=true)."
        ),
    ),
    RobotPlatform(
        id="rosbot_xl_arm",
        display_name="Husarion ROSbot XL + Arm",
        vendor="Husarion (ROSbot XL base, arm add-on)",
        kinematics=Kinematics.MOBILE_MANIPULATOR,
        mesh_path="models/rosbot_xl_arm.glb",
        license_path="models/rosbot_xl_arm.LICENSE.txt",
        dimensions_m=RobotDimensions(length_m=0.332054, width_m=0.284280, height_m=0.131989),
        radius_m=0.218561,
        radius_source=RadiusSource.MESH_MEASURED,
        reach_m=0.380,
        notes=(
            "dimensions_m/radius_m are for the BASE ONLY (no arm) - matches the task's "
            "instruction to the mesh agent and the vendor's own base-only "
            "332x284x131mm spec to within 0.02-0.76%; this is expected, not incomplete "
            "data. Mecanum-wheeled base plus a mounted manipulator - no single vendor "
            "Nav2 footprint accounts for the arm's added swing, so radius_m is "
            "mesh_measured (half_diagonal_m of the base footprint) rather than a "
            "vendor config value. reach_m=0.380 is OpenMANIPULATOR-X's published "
            "max-reach spec from its own base (link1); NOT measured from this mesh, "
            "whose arm is posed at the URDF's zero/home joint configuration (that "
            "pose's own end-effector distance is only ~0.352m, an under-count vs. true "
            "max reach) - flagged in the source report as 'not currently used by the "
            "system, recorded for future use'. Source: ~/reports/mesh/rosbot_xl_arm.json "
            "(husarion/rosbot_ros@41fad02 base + ROBOTIS-GIT/open_manipulator@84b68b2 "
            "arm (humble branch), dimensions_match=true)."
        ),
    ),
]

_BY_ID: dict[str, RobotPlatform] = {r.id: r for r in ROBOTS}

assert DEFAULT_ROBOT_ID in _BY_ID, "DEFAULT_ROBOT_ID must name a real registry entry"


def list_robots() -> list[RobotPlatform]:
    """All registered platforms, in registry order."""
    return list(ROBOTS)


def get_robot(robot_id: str) -> RobotPlatform | None:
    """A single platform by id, or None if `robot_id` isn't registered."""
    return _BY_ID.get(robot_id)


def resolve_radius_m(robot_id: str | None, explicit_radius_m: float | None) -> float:
    """The effective robot radius for one request, in priority order:

    1. `explicit_radius_m` - an explicit override (the frontend's radius slider, or a
       per-command `radius`/`robot_radius_m` param) always wins, regardless of which
       platform is selected - the slider must keep working exactly as it did before
       platforms existed.
    2. The selected platform's own `radius_m`, if `robot_id` names a registered
       platform that has one.
    3. `settings.robot_radius_m` - the pre-registry default (see config.py), kept as
       the ultimate fallback. Covers both "no platform selected" and a placeholder
       platform whose `radius_m` isn't measured yet (still `None`), so requests for
       any of the seven not-yet-measured platforms keep returning a usable answer
       instead of erroring.
    """
    if explicit_radius_m is not None:
        return explicit_radius_m
    if robot_id is not None:
        platform = get_robot(robot_id)
        if platform is not None and platform.radius_m is not None:
            return platform.radius_m
    return settings.robot_radius_m


def resolve_height_m(robot_id: str | None, explicit_height_m: float | None = None) -> float:
    """The effective robot HEIGHT for one request - what it can drive under.

    Same priority order as `resolve_radius_m`. This exists because the navigation layer
    stopped asking "is something there" and started asking "is something there that this
    robot cannot drive under": app/services/nav_layer.cells_for_height thresholds the
    shipped obstacle_min_height map against this number.

    Measured reason, hero-74 (`7ccaa75d`): a fixed 1.5 m obstacle band turned a bed top at
    0.694 m, a nightstand top at 0.684 m and a hanging curtain at 0.876 m into walls for a
    0.192 m TurtleBot, costing it three otherwise-reachable objects.

    Falls back to the DEFAULT platform's height rather than a settings value, because there
    is no pre-registry height setting to be compatible with - height only ever came from
    `dimensions_m`.
    """
    if explicit_height_m is not None:
        return explicit_height_m

    def _height(p: RobotPlatform | None) -> float | None:
        # nav_height_m wins where it is set: dimensions_m records the mesh as measured, and
        # for a legged platform that is the trunk at zero joint angles, not standing height.
        if p is None:
            return None
        if p.nav_height_m is not None:
            return p.nav_height_m
        return p.dimensions_m.height_m if p.dimensions_m is not None else None

    if robot_id is not None:
        h = _height(get_robot(robot_id))
        if h is not None:
            return h
    return _height(_BY_ID[DEFAULT_ROBOT_ID])


def resolve_footprint_m(
    robot_id: str | None,
    explicit_length_m: float | None = None,
    explicit_width_m: float | None = None,
) -> tuple[float, float]:
    """The effective (length_m, width_m) rectangle for one request - same priority
    order as `resolve_radius_m`, added for the rectangle-footprint planner
    (`app/services/footprint.py`):

    1. An explicit (length_m, width_m) override, if BOTH are given (a partial override
       - only one of the two - falls through, since a length without a width isn't a
       usable rectangle).
    2. The selected platform's own `dimensions_m.length_m`/`.width_m`, if `robot_id`
       names a registered platform whose dimensions are measured.
    3. A circle-derived square fallback: `resolve_radius_m`'s effective radius, doubled
       into a `2*radius_m x 2*radius_m` square. This is deliberately conservative (a
       square circumscribing the same radius has a LARGER area than the circle, so it
       never under-counts collision risk versus the old circle-only planner) and keeps
       every one of the 8 platforms usable even before real mesh dimensions existed for
       all of them (see this module's docstring on the robots-integration pass) - the
       same "always returns a usable answer" guarantee `resolve_radius_m` gives.
    """
    if explicit_length_m is not None and explicit_width_m is not None:
        return explicit_length_m, explicit_width_m
    if robot_id is not None:
        platform = get_robot(robot_id)
        if platform is not None and platform.dimensions_m is not None:
            return platform.dimensions_m.length_m, platform.dimensions_m.width_m
    radius_m = resolve_radius_m(robot_id, None)
    return 2 * radius_m, 2 * radius_m
