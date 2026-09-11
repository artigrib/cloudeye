"""Resolve objects by name (plain SQL, no vector search - per explicit spec
constraint), plan robot command steps via pathfinding, and persist the command +
result.
"""

from __future__ import annotations

import logging
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Command, Scene, SceneObject
from app.robots import resolve_radius_m
from app.schemas import CommandResponse, CommandStep
from app.services import pathfinding, vlm_client
from app.services.pathfinding import NoPathError, ObjectFootprint, OccupancyGrid
from app.services.vlm_client import ChatProviderNotConnected, VLMError

logger = logging.getLogger(__name__)

PICK_DURATION_SEC = 2.0
PLACE_DURATION_SEC = 2.0


class CommandFailedError(Exception):
    """Base class for command-planning failures that get persisted as a `failed`
    Command row and reported to the caller, rather than raised as a bare 500."""


class ObjectNotFoundError(CommandFailedError):
    """Raised by `resolve_object` when no object in the scene matches. Handled
    specially by the router (404, not the 200-with-failed-body every other
    CommandFailedError gets) - per spec: "внятная ошибка", not silence."""

    def __init__(self, name: str):
        self.name = name
        super().__init__(f"no object matching {name!r} in this scene")


def resolve_object_by_id(objects: Sequence[SceneObject], object_id) -> SceneObject:
    """Look up an object by id rather than by name - used by the `target_object_id`
    command path, where the object is already known (e.g. a double-clicked marker) and
    there's no ambiguity for `resolve_object`'s name tie-break to resolve."""
    for o in objects:
        if o.id == object_id:
            return o
    raise ObjectNotFoundError(str(object_id))


def resolve_object(objects: Sequence[SceneObject], name: str) -> SceneObject:
    """Plain name matching: exact, then case-insensitive, then substring in either
    direction. Among multiple candidates: prefer non-fragment, then highest num_views,
    then highest num_points - the "more reliable" tie-break from the spec."""
    name_lower = name.strip().lower()

    candidates = [o for o in objects if o.name == name]
    if not candidates:
        candidates = [o for o in objects if o.name.lower() == name_lower]
    if not candidates:
        candidates = [
            o for o in objects if name_lower in o.name.lower() or o.name.lower() in name_lower
        ]
    if not candidates:
        raise ObjectNotFoundError(name)

    candidates.sort(key=lambda o: (o.is_fragment, -o.num_views, -o.num_points))
    return candidates[0]


def _footprint(obj: SceneObject) -> ObjectFootprint:
    return ObjectFootprint(
        x=obj.pos_x,
        z=obj.pos_z,
        bbox_min_x=obj.bbox_min_x,
        bbox_min_z=obj.bbox_min_z,
        bbox_max_x=obj.bbox_max_x,
        bbox_max_z=obj.bbox_max_z,
    )


def build_steps(
    action: str,
    start: tuple[float, float],
    target_obj: SceneObject,
    dest_obj: SceneObject | None,
    grid: OccupancyGrid,
    *,
    robot_radius_m: float,
    speed_mps: float,
    all_objects: Sequence[SceneObject] = (),
) -> tuple[list[dict], float, float]:
    """Plan the concrete steps for one command. Returns (steps, total_duration_sec,
    total_length_m) - the latter is the runtime planner's own route length (metres,
    `pathfinding.PathResult.length_m`), summed across every "move" step, alongside
    the existing duration total (see CommandResponse.total_length_m).

    `all_objects` (default empty, for direct/test callers that don't care - see
    `pathfinding.plan_to_object`'s own docstring) is every object in the scene,
    passed through so the runtime planner rasterizes each one's own bbox as an
    obstacle (`pathfinding.rasterize_footprints_as_obstacles`), not just relying on
    the occupancy grid's own (object-blind) density scan. `execute_command` always
    passes the scene's full object list here."""
    all_footprints = [_footprint(o) for o in all_objects]
    if action in ("goto", "look"):
        result = pathfinding.plan_to_object(
            grid, start, _footprint(target_obj), robot_radius_m=robot_radius_m, speed_mps=speed_mps, objects=all_footprints
        )
        step = {
            "type": "move",
            "path": [list(p) for p in result.points],
            "duration_sec": round(result.duration_sec, 2),
            "length_m": round(result.length_m, 3),
        }
        return [step], result.duration_sec, result.length_m

    if action == "take":
        if dest_obj is None:
            raise CommandFailedError("'take' requires a destination object")

        leg1 = pathfinding.plan_to_object(
            grid, start, _footprint(target_obj), robot_radius_m=robot_radius_m, speed_mps=speed_mps, objects=all_footprints
        )
        leg2 = pathfinding.plan_to_object(
            grid, leg1.points[-1], _footprint(dest_obj), robot_radius_m=robot_radius_m, speed_mps=speed_mps, objects=all_footprints
        )
        steps = [
            {
                "type": "move",
                "path": [list(p) for p in leg1.points],
                "duration_sec": round(leg1.duration_sec, 2),
                "length_m": round(leg1.length_m, 3),
            },
            {
                "type": "pick",
                "object": target_obj.name,
                "position": [target_obj.pos_x, target_obj.pos_y, target_obj.pos_z],
                "duration_sec": PICK_DURATION_SEC,
            },
            {
                "type": "move",
                "path": [list(p) for p in leg2.points],
                "duration_sec": round(leg2.duration_sec, 2),
                "length_m": round(leg2.length_m, 3),
            },
            {"type": "place", "object": target_obj.name, "at": dest_obj.name, "duration_sec": PLACE_DURATION_SEC},
        ]
        total_duration = leg1.duration_sec + PICK_DURATION_SEC + leg2.duration_sec + PLACE_DURATION_SEC
        total_length = leg1.length_m + leg2.length_m
        return steps, total_duration, total_length

    raise CommandFailedError(f"unknown action {action!r}")


async def create_command(session: AsyncSession, scene: Scene, text: str) -> Command:
    command = Command(scene_id=scene.id, user_text=text, status="pending")
    session.add(command)
    await session.commit()
    await session.refresh(command)
    return command


async def execute_command(
    session: AsyncSession,
    scene: Scene,
    command: Command,
    objects: Sequence[SceneObject],
    grid: OccupancyGrid,
    *,
    robot_id: str | None = None,
    robot_radius_m: float | None = None,
    parsed_override: dict | None = None,
    target_override: SceneObject | None = None,
    start_override: tuple[float, float] | None = None,
    chat_provider: str | None = None,
) -> Command:
    """Parse the command via the VLM, resolve objects, plan a path, and persist the
    result on the Command row - success or failure, never discarded. `ObjectNotFoundError`
    is re-raised after persisting (the router turns it into 404); every other failure
    (a bad VLM response, an unreachable path) is stored as `status="failed"` and
    returned normally (200) - per the spec's explicit split between the two.

    `robot_id` selects a platform from the registry (app/robots.py) for its radius;
    `robot_radius_m` lets a single command override that (e.g. from the frontend's
    radius slider) without changing the configured default for every other command -
    see `app.robots.resolve_radius_m` for the exact precedence, and also
    `pathfinding.compute_reachability`, which the same override drives for the
    pre-flight reachability check.

    `parsed_override`/`target_override` skip the VLM call entirely for the
    `target_object_id` command path (the object is already known - see
    `routers/scenes.post_command`); `target_override` is used instead of re-resolving
    `parsed_override["target"]` by name, so a double-clicked object is never swapped for
    a same-named duplicate by `resolve_object`'s tie-break.

    `start_override` plans from an arbitrary world point instead of the scene's fixed
    `robot_start_x/z` - lets the frontend continue from wherever the robot visually is.

    `chat_provider` names the chat LLM to parse `user_text` with, from the workspace's job
    spec (`JobSpec.chat_llm`); None means the default. It is recorded on the command
    either way - including on the failure paths, so "which model was asked" survives a
    failure, which is the case where the question actually gets asked. The
    `parsed_override` path records None, because no model was involved at all."""
    parsed: dict | None = None
    # None for the direct-goto path: nothing was asked of any model, and naming a
    # provider there would claim otherwise.
    provider_used: str | None = None
    try:
        if parsed_override is not None:
            parsed = parsed_override
        else:
            provider_used = vlm_client.resolve_chat_provider(chat_provider)
            parsed = await vlm_client.parse_command(list(objects), command.user_text, provider=provider_used)
            if "error" in parsed:
                raise CommandFailedError(parsed["error"])

        action = parsed["action"]
        target_obj = (
            target_override if target_override is not None else resolve_object(objects, parsed["target"])
        )  # may raise ObjectNotFoundError
        dest_obj = (
            resolve_object(objects, parsed["destination"]) if parsed.get("destination") else None
        )

        start = start_override if start_override is not None else (scene.robot_start_x or 0.0, scene.robot_start_z or 0.0)
        steps, total_duration, total_length = build_steps(
            action,
            start,
            target_obj,
            dest_obj,
            grid,
            robot_radius_m=resolve_radius_m(robot_id, robot_radius_m),
            speed_mps=settings.robot_speed_mps,
            all_objects=objects,
        )

        command.parsed_action = parsed
        command.result = {
            "steps": steps,
            "total_duration_sec": round(total_duration, 2),
            "total_length_m": round(total_length, 3),
            "provider_used": provider_used,
            "error_code": None,
        }
        command.status = "done"
        command.error_message = None
        await session.commit()
        await session.refresh(command)
        return command

    except ObjectNotFoundError as exc:
        command.parsed_action = parsed
        command.result = {"provider_used": provider_used, "error_code": None}
        command.status = "failed"
        command.error_message = str(exc)
        await session.commit()
        await session.refresh(command)
        raise

    except (CommandFailedError, VLMError, NoPathError) as exc:
        logger.warning("command %s failed: %s", command.id, exc)
        command.parsed_action = parsed
        command.result = {
            "provider_used": provider_used,
            # The one failure that is about this deployment, not about this command.
            "error_code": "chat_llm_not_connected" if isinstance(exc, ChatProviderNotConnected) else None,
        }
        command.status = "failed"
        command.error_message = str(exc)
        await session.commit()
        await session.refresh(command)
        return command


def command_to_response(command: Command) -> CommandResponse:
    """Build the API response shape from a Command row - `action`/`steps`/
    `total_duration_sec` live inside the JSONB `parsed_action`/`result` columns, not as
    direct attributes, so this can't just be `CommandResponse.model_validate(command)`."""
    result = command.result or {}
    return CommandResponse(
        command_id=command.id,
        scene_id=command.scene_id,
        user_text=command.user_text,
        action=(command.parsed_action or {}).get("action"),
        parsed_action=command.parsed_action,
        steps=[CommandStep(**s) for s in result.get("steps", [])],
        total_duration_sec=result.get("total_duration_sec"),
        total_length_m=result.get("total_length_m"),
        provider_used=result.get("provider_used"),
        error_code=result.get("error_code"),
        status=command.status,
        error_message=command.error_message,
        created_at=command.created_at,
    )


async def list_commands(
    session: AsyncSession, scene_id, *, skip: int, limit: int
) -> tuple[list[Command], int]:
    from sqlalchemy import func

    total = await session.scalar(select(func.count()).select_from(Command).where(Command.scene_id == scene_id))
    result = await session.execute(
        select(Command)
        .where(Command.scene_id == scene_id)
        .order_by(Command.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    return list(result.scalars().all()), total or 0
