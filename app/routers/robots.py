"""Robot platform registry endpoint - backs the frontend's platform selector.

See app/robots.py for the registry itself, and app.robots.resolve_radius_m for how a
selected platform's radius interacts with an explicit radius override (the reachability
slider) elsewhere in the API.
"""

from fastapi import APIRouter

from app.robots import DEFAULT_ROBOT_ID, list_robots
from app.schemas import RobotListResponse, RobotPlatformResponse

router = APIRouter(prefix="/api/robots", tags=["robots"])


@router.get("", response_model=RobotListResponse)
async def get_robots() -> RobotListResponse:
    """List every registered robot platform. All eight now carry real measured/vendor
    numbers (see app/robots.py's module docstring and ~/reports/P.md for provenance);
    a platform added ahead of its own mesh-measurement pass would still show up here
    as a placeholder (`radius_m`/`dimensions_m` null, `notes` explains why)."""
    platforms = list_robots()
    return RobotListResponse(
        default_robot_id=DEFAULT_ROBOT_ID,
        total=len(platforms),
        items=[RobotPlatformResponse.model_validate(p) for p in platforms],
    )
