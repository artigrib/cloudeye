"""Scene read endpoints: metadata, status, occupancy map, mesh, objects, and deletion.

Non-`done` scenes return 409 (not 404 - the scene exists, it's just not ready yet) from
`/map`, `/mesh`, `/objects`. `/status` always works regardless of state, since that's
exactly what a frontend polls while waiting - it does read one small local file per
call (this scene's own `logs/worker.log` or `stage_timings.json`, see
get_scene_status/app.services.stage_timings) to surface per-stage timing for the
frontend's ETA display; still no DB joins and no remote/GPU-side I/O.
"""

import json
import logging
import uuid
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models import Scene, SceneObject
from app.robots import resolve_height_m, resolve_radius_m
from app.schemas import (
    CameraPoseResponse,
    CameraTrackResponse,
    CommandCreate,
    CommandListResponse,
    CommandResponse,
    MsaObjectsResponse,
    OccupancyGridResponse,
    ReachabilityResponse,
    SceneLayersResponse,
    SceneObjectListResponse,
    SceneObjectResponse,
    SceneResponse,
    SceneStatusResponse,
)
from app.services import (
    command_service,
    nav_layer,
    scene_ingest,
    scene_layers,
    scene_service,
    usd_export,
    video_service,
)
from app.services.job_spec import read_job_spec
from app.services.command_service import ObjectNotFoundError
from app.services.nav_layer import NavLayerUnavailableError
from app.services.pathfinding import (
    ObjectFootprint,
    OccupancyGrid,
    cell_to_world,
    clamp_anchor_to_grid,
    compute_reachability,
    compute_reachability_cached,
    cut_footprints_from_grid,
    invalidate_grid,
    translate_footprint,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/scenes", tags=["scenes"])


def _layer_or_none(scene_id: uuid.UUID) -> tuple[nav_layer.NavLayer | None, str | None]:
    """This scene's ONE navigation layer, or (None, why not) - for the two READ endpoints
    that a scene screen renders.

    There is no fallback to `Scene.occupancy_path`. That column points at
    gpu/stage_occupancy.py's 0.1-0.5 m density histogram - the layer that could not see a
    table top at 0.9 m and drew route cells straight through furniture. Keeping it as a
    fallback is how a scene silently gets a second answer to "where can this robot stand".

    But "no layer" is a STATE OF THE SCENE, not a failed request. `/reachability` and
    `/map` answer 200 either way, and say which shape they are returning, so a client has
    two renderable answers and no error branch. An action endpoint is different - see
    `_layer_or_409`.
    """
    try:
        return nav_layer.resolve(scene_service.scene_dir(scene_id)), None
    except NavLayerUnavailableError as exc:
        return None, str(exc)


def _layer_or_409(scene_id: uuid.UUID) -> nav_layer.NavLayer:
    """The same layer, for endpoints that DO something rather than describe the scene:
    planning a command, exporting a USD, serving the raw .npy.

    409 is right there and 200 would not be: there is no "here is your export, but empty"
    that a caller could use. The scene-screen reads never reach this - they take
    `_layer_or_none` and render the not-available shape.
    """
    layer, why = _layer_or_none(scene_id)
    if layer is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=why)
    return layer


def _grid_for(layer: nav_layer.NavLayer, robot_id: str | None) -> OccupancyGrid:
    """The scene's grid AS THIS ROBOT SEES IT.

    The layer ships one map - the lowest surface height per cell - and every route derives
    its tri-state grid from it at the requesting robot's own height. That is what keeps the
    single source single while letting a 0.192 m TurtleBot drive under a 0.694 m bed top
    that a 1.5 m fixed band called a wall.
    """
    return OccupancyGrid(cells=nav_layer.cells_for_height(layer, resolve_height_m(robot_id)),
                         meta=layer.meta)


def _no_layer_reachability(scene_id: uuid.UUID, robot_id: str | None,
                           radius: float, why: str | None) -> ReachabilityResponse:
    """The second of the two shapes `/reachability` can return: no layer, so no verdict.

    Every count is zero and `start_status` is "none", which is also what "this robot fits
    nowhere in this room" looks like - so `layer_available` is what separates them. One is
    an answer about the robot; the other is the absence of anything to answer with, and a
    screen that showed "0 of 17 reachable" for the second would be lying with arithmetic.
    """
    return ReachabilityResponse(
        layer_available=False,
        layer_unavailable_reason=why,
        scene_id=scene_id,
        radius=radius,
        robot_id=robot_id,
        component_count=0,
        largest_component_size=0,
        start_component_size=0,
        reachable_object_ids=[],
        unreachable_reasons={},
        path_length_m={},
        moved_object_ids=[],
        reachable_cells=[],
        unknown_cells_on_route=0,
        obstacle_cells_on_route=0,
        start_status="none",
        start_x=None,
        start_z=None,
        start_moved_m=0.0,
    )


async def _get_scene_or_404(session: AsyncSession, scene_id: uuid.UUID) -> Scene:
    scene = await scene_service.get_scene(session, scene_id)
    if scene is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scene not found")
    return scene


def _require_done(scene: Scene) -> None:
    if scene.status != "done":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Scene is not ready (status={scene.status})",
        )


@router.get("/{scene_id}", response_model=SceneResponse)
async def get_scene(
    scene_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> SceneResponse:
    """Full scene metadata plus its detected objects, regardless of status. Also
    carries `stage`/`stage_timings` (see scene_service.apply_stage_timings) so a
    frontend that already has this scene open (ScenePage, polling this same endpoint
    while queued/processing) can show the stage/ETA display without a second poll
    against `/status`."""
    scene = await scene_service.get_scene_with_objects(session, scene_id)
    if scene is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scene not found")
    resp = SceneResponse.model_validate(scene)
    scene_service.apply_stage_timings(resp, scene)
    return resp


@router.get("/{scene_id}/status", response_model=SceneStatusResponse)
async def get_scene_status(
    scene_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> SceneStatusResponse:
    """Status for frontend polling, plus per-stage timing for the stage/ETA display.

    While `status == "processing"`, re-parses this scene's own (small, local)
    `logs/worker.log` on every call so `stage`/`stage_timings` always reflect the
    latest transition - see app.services.stage_timings.refresh_stage_timings_from_log.
    For a terminal scene (`done`/`failed`), reads the stage_timings.json
    `pipeline_orchestrator.run_pipeline_for_scene` persisted once the run finished,
    rather than re-parsing the log on every poll.
    """
    scene = await _get_scene_or_404(session, scene_id)
    return scene_service.build_status_response(scene)


@router.get("/{scene_id}/objects", response_model=SceneObjectListResponse)
async def list_scene_objects(
    scene_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> SceneObjectListResponse:
    scene = await _get_scene_or_404(session, scene_id)
    _require_done(scene)
    objects = await scene_service.list_scene_objects(session, scene_id)
    return SceneObjectListResponse(
        scene_id=scene_id,
        total=len(objects),
        items=[SceneObjectResponse.model_validate(o) for o in objects],
    )


@router.get("/{scene_id}/camera-track", response_model=CameraTrackResponse)
async def get_camera_track(
    scene_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> CameraTrackResponse:
    """The aligned camera trajectory from the capture video, for the frontend's
    (default-off) "camera path" visualisation - see CameraTrackResponse's docstring."""
    scene = await _get_scene_or_404(session, scene_id)
    _require_done(scene)
    scene_dir = scene_service.scene_dir(scene_id)
    cameras = scene_ingest.read_camera_track(scene_dir)
    timestamps = scene_ingest.read_keyframe_timestamps(scene_dir, len(cameras))
    return CameraTrackResponse(
        scene_id=scene_id,
        poses=[
            CameraPoseResponse(
                x=x, y=y, z=z,
                timestamp_sec=timestamps[i] if timestamps is not None else None,
            )
            for i, (x, y, z) in enumerate(cameras)
        ],
    )


@router.get("/{scene_id}/map")
async def get_scene_map(
    scene_id: uuid.UUID,
    format: str = Query("json", pattern="^(json|npy)$"),
    robot_id: str | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
):
    """The occupancy grid: `?format=json` (default - a ~79x79 grid is a few KB, fine
    for a canvas frontend to consume directly) or `?format=npy` (the raw array file).

    `robot_id` matters here: the grid is derived from the layer's obstacle-height map at
    that platform's own height, so the map a user looks at is the map their routes were
    planned on. Unset uses the default platform.

    `unobserved` rides along because the tri-state grid alone cannot answer "was this cell
    ever seen": a cell can read FREE for a short robot and still be ground the scan never
    covered. The 2D view hatches those cells and the verdict counts the ones a route
    crosses, and both need the mask, not an inference from the colour."""
    scene = await _get_scene_or_404(session, scene_id)
    _require_done(scene)
    layer, why = _layer_or_none(scene_id)

    if layer is None:
        if format == "npy":
            # No 200-with-an-empty-array for a binary download: there is no such file, and
            # a caller saving one would get a corrupt .npy. This is the action-shaped half.
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=why)
        return OccupancyGridResponse(
            resolution=0.0, origin_x=0.0, origin_z=0.0, width=0, height=0,
            cells=[], unobserved=[],
            layer_available=False, layer_unavailable_reason=why,
        )

    if format == "npy":
        # The raw file is the DEFAULT-height view; a per-robot npy would be a file that
        # does not exist on disk. JSON is the per-robot answer.
        return FileResponse(layer.grid_path, media_type="application/octet-stream")

    cells = _grid_for(layer, robot_id).cells
    return OccupancyGridResponse(
        resolution=layer.meta.resolution,
        origin_x=layer.meta.origin_x,
        origin_z=layer.meta.origin_z,
        width=layer.meta.width,
        height=layer.meta.height,
        cells=cells.tolist(),
        unobserved=nav_layer.unobserved_mask(layer).tolist(),
    )


@router.get("/{scene_id}/map/preview")
async def get_scene_map_preview(
    scene_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> FileResponse:
    scene = await _get_scene_or_404(session, scene_id)
    _require_done(scene)
    if not scene.map_preview_path or not Path(scene.map_preview_path).is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Map preview not found on disk")
    return FileResponse(scene.map_preview_path, media_type="image/png")


@router.get("/{scene_id}/mesh")
async def get_scene_mesh(
    scene_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> FileResponse:
    """The scene's points-primitive GLB (see gpu/stage_export_glb.py's docstring - this
    is a colored point cloud in glTF form, not a triangulated mesh). Starlette's
    FileResponse handles Range requests natively (verified: 1.6.0+), so a 150MB file
    seeks correctly without a custom implementation."""
    scene = await _get_scene_or_404(session, scene_id)
    _require_done(scene)
    if not scene.mesh_path or not Path(scene.mesh_path).is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mesh not found on disk")
    return FileResponse(scene.mesh_path, media_type="model/gltf-binary")


@router.get("/{scene_id}/cloud")
async def get_scene_cloud(
    scene_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> FileResponse:
    """The scene's binary point cloud (CEPC v1 - header, float32 xyz, uint8 rgb; see
    frontend/src/lib/cloudBin.ts for the layout it is parsed against).

    Deliberately a fixed filename inside the scene dir rather than a Scene column: like
    the MSA export (see msa_glb_path's docstring), a scene either has this file or it
    does not, and there is nothing about it worth a migration. 404 is the ordinary
    answer - every scene the GPU pipeline produced has only the points-primitive GLB at
    /mesh, and the viewer's loader falls back to that on a 404. FileResponse handles
    Range natively, same as /mesh."""
    scene = await _get_scene_or_404(session, scene_id)
    _require_done(scene)
    path = scene_service.scene_dir(scene_id) / "cloud.bin"
    if not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No binary cloud for this scene")
    return FileResponse(path, media_type="application/octet-stream")


@router.get("/{scene_id}/msa-glb")
async def get_scene_msa_glb(
    scene_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> FileResponse:
    """An optional MSA (Measured Scene Assembly, var/scratch/msa/SPEC.md) glTF export
    for this scene - mesh/collision/plan layers generated by scripts/msa/export_*.py,
    meant to be overlaid on top of THIS SAME scene's point-cloud view in the frontend
    (SceneMap3D.tsx), not shown in a separate viewer with its own camera. Prefers the
    textured dollhouse export over the plain untextured bootstrap export - see
    scene_service.MSA_GLB_CANDIDATES for the exact fallback order and why - and 404s if
    the scene has none of those files under `<scene_dir>/msa/` - most scenes won't,
    since MSA is a separate, newer pipeline that hasn't been run for every scene.
    Purely read-only: this endpoint never generates or writes the file itself, only
    serves it if already present. Range-capable via FileResponse, same as /mesh and
    /usd. Reports which candidate was actually served via the `X-Msa-Glb-Variant`
    response header, so callers can tell a textured export from the bootstrap
    fallback without inspecting the glTF itself."""
    scene = await _get_scene_or_404(session, scene_id)
    _require_done(scene)
    resolved = scene_service.resolve_msa_glb(scene_id)
    if resolved is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No MSA export for this scene")
    path, variant = resolved
    return FileResponse(
        path,
        media_type="model/gltf-binary",
        headers={"X-Msa-Glb-Variant": variant},
    )


@router.get("/{scene_id}/msa-objects", response_model=MsaObjectsResponse)
async def get_scene_msa_objects(
    scene_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> MsaObjectsResponse:
    """The canonical, de-cluttered MSA object list (id/label only) for this scene -
    scripts/msa/bootstrap.py's merge + support-rule + class-cap output, and the M7
    audit's manual drops when present - as opposed to `scene.objects` on
    GET /{scene_id} (raw per-detection DB rows, pre-de-clutter). Purely read-only,
    same optional/best-effort shape as /msa-glb: 404s if the scene has no MSA objects
    export under `<scene_dir>/msa/` (most scenes won't). See
    scene_service.MSA_OBJECTS_CANDIDATES for the fallback order and
    `X-Msa-Objects-Variant`-equivalent info via the `variant` field (a body field here
    rather than a header, since this is JSON, not a FileResponse)."""
    scene = await _get_scene_or_404(session, scene_id)
    _require_done(scene)
    resolved = scene_service.resolve_msa_objects(scene_id)
    if resolved is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No MSA objects export for this scene")
    objects, variant = resolved
    return MsaObjectsResponse(objects=objects, variant=variant)


@router.get("/{scene_id}/fit-prob")
async def get_scene_fit_prob(
    scene_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> FileResponse:
    """The Monte-Carlo fit-probability audit (`scripts.audit.fit_prob`) for this
    scene, if one has been run - `<scene_dir>/msa/fit_prob.json` (see
    scene_service.fit_prob_path), 404 when absent (most scenes won't have one: this
    is a separate offline audit, not part of the normal pipeline). Purely read-only,
    same convention as `/msa-glb`: this endpoint never generates the file itself,
    only serves it if already present. The frontend degrades to its existing
    reachable-count-only display on a 404 (see RobotPanel/ReachabilityCard)."""
    scene = await _get_scene_or_404(session, scene_id)
    _require_done(scene)
    path = scene_service.fit_prob_path(scene_id)
    if not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No fit-probability audit for this scene")
    return FileResponse(path, media_type="application/json")


@router.get("/{scene_id}/critic")
async def get_scene_critic(
    scene_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> FileResponse:
    """Optional scene-critic report for this scene - the merged output of
    scripts/audit/critic_rules.py + critic_vlm.py + critic_merge.py
    (var/scratch/QUEUE.md M14 "Scene critic"), served from its fixed
    `<scene_dir>/msa/critic.json` location (see
    app.services.scene_service.msa_critic_json_path). Mirrors get_scene_msa_glb
    above: purely read-only (never generates or writes the file itself, only serves
    it if a critic pass was already run for this scene), 404s when no critic.json
    exists on disk - the common case, since the critic is opt-in and run manually
    per scene, same as MSA itself. No auto-fix and no scene mutation happen through
    this route or anywhere in the critic pipeline - it is strictly a report."""
    scene = await _get_scene_or_404(session, scene_id)
    _require_done(scene)
    path = scene_service.msa_critic_json_path(scene_id)
    if not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No critic report for this scene")
    return FileResponse(path, media_type="application/json")


@router.get("/{scene_id}/msa-meta")
async def get_scene_msa_meta(
    scene_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Sibling of /msa-glb: the bootstrap `scene_meta.json` written next to the served
    MSA GLB, reduced to `scene_service.MSA_META_FIELDS` (`yaw_applied`,
    `yaw_correction_rad`, `yaw_correction_deg`, `yaw_rotation_center_xy`, `yaw_method`,
    `floor_y`, `ceiling_y`, `room_polygon` - whichever are present). The GLB's geometry
    is yaw-normalized by bootstrap (rotated by `yaw_correction_rad` about the XZ pivot
    `yaw_rotation_center_xy`, see scripts/msa/geometry.rotate_point_xz) while the point
    cloud this app serves is NOT, so SceneMap3D.tsx applies the inverse rotation to its
    MSA layer group from these fields to land the MSA walls on the cloud's walls (see
    lib/msaLayers.msaOverlayTransform). Read-only; 404 if the scene has no MSA export or
    its export predates scene_meta.json."""
    scene = await _get_scene_or_404(session, scene_id)
    _require_done(scene)
    meta = scene_service.read_msa_meta(scene_service.msa_meta_path(scene_id))
    if meta is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No MSA scene_meta.json for this scene")
    return meta


@router.get("/{scene_id}/usd")
async def get_scene_usd(
    scene_id: uuid.UUID,
    include_pointcloud: bool = Query(True),
    include_fragments: bool = Query(False),
    robot_id: str | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
) -> FileResponse:
    """An Isaac Sim-ready OpenUSD export: floor/walls as collidable boxes from the
    occupancy grid, objects as collidable convex hulls of their own captured point
    clouds (falling back to an axis-aligned bbox when a cloud is missing or unusable -
    see usd_export.py's docstring for why the raw point cloud itself isn't the
    simulated geometry, and why a hull rather than a bbox). `robot_id` (app/robots.py
    registry, see app.robots.resolve_radius_m) selects which platform's radius is
    stamped into the `cloudeye:robotRadiusM` custom attribute; unset falls back to the
    default platform. Generated lazily on first request per (include_pointcloud,
    include_fragments, robot_id's resolved radius) and cached to disk under
    `<scene_dir>/usd/` - a param change is a different cache file, never a stale one.
    Range-capable via FileResponse, same as /mesh."""
    scene = await _get_scene_or_404(session, scene_id)
    _require_done(scene)

    # Prefer a pre-built nvblox export when the scene has one. It is a real reconstructed
    # mesh (vertex-coloured, z-up with the floor at z=0), where the generated export below
    # is collision geometry only - occupancy-grid boxes plus per-object convex hulls, no
    # visual mesh at all. Dropped in as `<scene_dir>/usd/nvblox.usd` alongside a
    # `source.json` recording which run produced it; there is no DB column for it on
    # purpose, so no migration is involved. `X-Usd-Variant` says which one was served, the
    # same way /msa-glb reports `X-Msa-Glb-Variant`.
    nvblox_usd = scene_service.scene_dir(scene_id) / "usd" / "nvblox.usd"
    if nvblox_usd.is_file():
        headers = {"X-Usd-Variant": "nvblox"}
        source_json = nvblox_usd.with_name("source.json")
        if source_json.is_file():
            try:
                src = json.loads(source_json.read_text())
                headers["X-Usd-Run-Id"] = str(src.get("run_id", ""))
                headers["X-Usd-Scene-Name"] = str(src.get("scene_name", ""))
            except (OSError, ValueError):
                pass    # provenance is nice to have; a broken source.json must not 500
        return FileResponse(nvblox_usd, media_type="model/vnd.usd",
                            filename=f"scene_{scene_id}.usd", headers=headers)

    if scene.ceiling_y is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Scene has no ceiling_y recorded"
        )

    layer = _layer_or_409(scene_id)
    grid = _grid_for(layer, robot_id)

    effective_radius_m = resolve_radius_m(robot_id, None)

    cache_dir = scene_service.scene_dir(scene_id) / "usd"
    cache_path = cache_dir / usd_export.cache_filename(
        include_pointcloud=include_pointcloud,
        include_fragments=include_fragments,
        robot_radius_m=effective_radius_m,
    )

    if not cache_path.is_file():
        objects = await scene_service.list_scene_objects(session, scene_id)
        export_objects = [
            usd_export.UsdExportObject(
                name=o.name,
                bbox_min=(o.bbox_min_x, o.bbox_min_y, o.bbox_min_z),
                bbox_max=(o.bbox_max_x, o.bbox_max_y, o.bbox_max_z),
                num_views=o.num_views,
                num_points=o.num_points,
                is_fragment=o.is_fragment,
                mesh_path=o.mesh_path,
            )
            for o in objects
        ]
        pointcloud_path = None
        if include_pointcloud and scene.pointcloud_path and Path(scene.pointcloud_path).is_file():
            pointcloud_path = scene.pointcloud_path

        robot_start = None
        if scene.robot_start_x is not None and scene.robot_start_z is not None:
            robot_start = (scene.robot_start_x, scene.robot_start_z)

        export_input = usd_export.UsdExportInput(
            grid_cells=grid.cells,
            grid_meta=layer.meta,
            ceiling_y=scene.ceiling_y,
            robot_start=robot_start,
            is_reflected=scene.is_reflected,
            robot_radius_m=effective_radius_m,
            objects=export_objects,
            pointcloud_path=pointcloud_path,
            include_fragments=include_fragments,
        )
        stats = usd_export.export_scene_usd(export_input, cache_path)
        logger.info(
            "USD export for scene %s (pointcloud=%s, fragments=%s): %d obstacle cells -> "
            "%d structure boxes, %d objects (%d fragments total, %d convex hull [%d "
            "decimated], %d bbox fallback), %d point-cloud points",
            scene_id, include_pointcloud, include_fragments,
            stats.obstacle_cell_count, stats.structure_box_count,
            stats.object_count, stats.fragment_count,
            stats.hull_object_count, stats.hull_decimated_object_count,
            stats.bbox_fallback_object_count, stats.pointcloud_point_count,
        )
        await scene_service.set_usd_path(session, scene, str(cache_path))

    return FileResponse(
        cache_path, media_type="model/vnd.usd", filename=f"scene_{scene_id}.usd",
        headers={"X-Usd-Variant": "legacy-collision-export"},
    )


@router.post("/{scene_id}/command", response_model=CommandResponse, status_code=status.HTTP_200_OK)
async def post_command(
    scene_id: uuid.UUID,
    body: CommandCreate,
    session: AsyncSession = Depends(get_session),
) -> CommandResponse:
    """Issue a robot command against a scene: either a natural-language `text` (parsed
    via the VLM) or a `target_object_id` (a direct "goto", skipping the VLM entirely -
    used by the frontend's double-click-to-go interaction). Returns 200 with a
    `status="failed"` body for most planning failures (per spec - a UI history entry,
    not a bare error); an unresolvable object (by name or by id) is the one exception,
    returning 404."""
    scene = await _get_scene_or_404(session, scene_id)
    _require_done(scene)

    objects = await scene_service.list_scene_objects(session, scene_id)
    grid = _grid_for(_layer_or_409(scene_id), body.robot_id)

    parsed_override = None
    target_override = None
    if body.target_object_id is not None:
        try:
            target_override = command_service.resolve_object_by_id(objects, body.target_object_id)
        except ObjectNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        parsed_override = {"action": "goto", "target": target_override.name, "destination": None}
        user_text = body.text or f"goto {target_override.name}"
    else:
        user_text = body.text  # schema guarantees non-empty when target_object_id is absent

    start_override = (body.from_[0], body.from_[1]) if body.from_ is not None else None

    # Which chat LLM parses a TYPED command: the one this workspace's operator chose in
    # the New-workspace wizard, recorded in the job spec beside the uploaded video
    # (docs/JOB_SPEC.md). It was written and then ignored until now, so an operator who
    # picked vertex-gemma silently got OpenRouter. A scene whose video predates the
    # wizard, or has no spec on disk, falls back to the default - `read_job_spec`'s own
    # contract ("callers must treat None as 'use the defaults', never as an error").
    # Skipped entirely for a direct goto: no model is asked, so none is looked up.
    chat_provider: str | None = None
    if body.target_object_id is None:
        video = await video_service.get_video(session, scene.video_id)
        spec = read_job_spec(video.filepath) if video and video.filepath else None
        chat_provider = spec.chat_llm if spec else None

    command = await command_service.create_command(session, scene, user_text)
    try:
        command = await command_service.execute_command(
            session,
            scene,
            command,
            objects,
            grid,
            robot_id=body.robot_id,
            robot_radius_m=body.radius,
            parsed_override=parsed_override,
            target_override=target_override,
            start_override=start_override,
            chat_provider=chat_provider,
        )
    except ObjectNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return command_service.command_to_response(command)


def _footprints(objects: list[SceneObject]) -> list[tuple[str, ObjectFootprint]]:
    return [
        (
            str(o.id),
            ObjectFootprint(
                x=o.pos_x, z=o.pos_z,
                bbox_min_x=o.bbox_min_x, bbox_min_z=o.bbox_min_z,
                bbox_max_x=o.bbox_max_x, bbox_max_z=o.bbox_max_z,
            ),
        )
        for o in objects
        if not o.is_fragment
    ]


def _parse_moves(raw: list[str]) -> dict[str, tuple[float, float]]:
    """`["<object uuid>:<dx>:<dz>", ...]` -> {object id: (dx, dz)}, metres.

    A repeated query parameter rather than a request body, so this stays a GET and stays
    linkable: a moved-furniture what-if is a URL someone can send to a colleague.
    Malformed input is a 400, never a silently dropped move - a what-if that quietly did
    not happen would be answered with the UNMOVED room's numbers, which is the one wrong
    answer this endpoint must never give."""
    moves: dict[str, tuple[float, float]] = {}
    for item in raw:
        parts = item.split(":")
        if len(parts) != 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"move must be '<object_id>:<dx>:<dz>', got {item!r}",
            )
        try:
            object_id = str(uuid.UUID(parts[0]))
            moves[object_id] = (float(parts[1]), float(parts[2]))
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"move must be '<object_id>:<dx>:<dz>' with a uuid and two numbers, got {item!r}",
            ) from exc
    return moves


@router.get("/{scene_id}/reachability", response_model=ReachabilityResponse)
async def get_reachability(
    scene_id: uuid.UUID,
    robot_id: str | None = Query(default=None),
    radius: float | None = Query(default=None, ge=0.02, le=0.6),
    move: list[str] = Query(default=[]),
    session: AsyncSession = Depends(get_session),
) -> ReachabilityResponse:
    """Which objects are reachable at a given robot radius, and the grid's overall
    connectivity there - backs the frontend's radius slider and its pre-flight "this
    object is unreachable" check. Cached per (scene, radius).

    `robot_id` (app/robots.py registry) sets the platform's own radius as the starting
    point; an explicit `radius` (the slider) always overrides it - see
    `app.robots.resolve_radius_m` for the exact precedence. Neither given falls back to
    the default platform, same behavior as before platforms existed."""
    scene = await _get_scene_or_404(session, scene_id)
    _require_done(scene)

    effective_radius = resolve_radius_m(robot_id, radius)

    objects = await scene_service.list_scene_objects(session, scene_id)
    layer, why = _layer_or_none(scene_id)
    if layer is None:
        # 200, not 409. "This scene was never given a navigation layer" is something the
        # screen can render - it is a state of the scene, not a failed request - and a
        # verdict bar that says so beats an error the client has to translate.
        return _no_layer_reachability(scene_id, robot_id, effective_radius, why)
    effective_height = resolve_height_m(robot_id)
    grid = _grid_for(layer, robot_id)

    # The anchor the per-radius start is measured from is the FIRST CAMERA POSE, not the
    # persisted robot_start_x/z. The persisted one was resolved once at scene creation,
    # at whatever radius was default then, and already carries that radius's snapping and
    # relocation baked in - anchoring a fresh per-platform search on it would compound one
    # platform's answer into the next. The camera track is the raw, radius-independent
    # fact: where the operator was actually standing when the scan started. Falls back to
    # the persisted start for a scene with no readable track.
    #
    # Each candidate is CHECKED against the grid before it is used, and the first usable
    # one wins. own_0901_161054's first camera pose sits 0.62 m past the grid's z edge -
    # the operator started in the doorway, outside the reconstruction's percentile-trimmed
    # extent - and feeding it in unchecked raised ValueError out of world_to_cell, so every
    # /reachability call on that scene answered 500 (reproduced on tag
    # scene-screen-2026-09-09b). `clamp_anchor_to_grid` now absorbs a metre of that, and
    # this chain absorbs the rest: a scene answers with a verdict or says it has no layer,
    # and an anchor nobody can use is a logged warning, not a 500.
    try:
        cameras = scene_ingest.read_camera_track(scene_service.scene_dir(scene_id))
    except Exception:  # noqa: BLE001 - a missing/garbled track is not a reachability failure
        cameras = []
    candidates: list[tuple[str, tuple[float, float]]] = []
    if cameras:
        candidates.append(("first camera pose", (cameras[0][0], cameras[0][2])))
    if scene.robot_start_x is not None and scene.robot_start_z is not None:
        candidates.append(("persisted robot start", (scene.robot_start_x, scene.robot_start_z)))
    centre = cell_to_world(grid.meta.width // 2, grid.meta.height // 2, grid.meta)
    candidates.append(("grid centre", centre))

    anchor = centre
    for name, candidate in candidates:
        try:
            clamp_anchor_to_grid(candidate, grid.meta)
        except ValueError as exc:
            logger.warning("reachability anchor: %s unusable for scene %s - %s",
                           name, scene_id, exc)
            continue
        anchor = candidate
        break

    # "What if that were somewhere else": a bbox cut/paste on the occupancy grid, and
    # then the ordinary reachability computation over the result. The cut is blunt on
    # purpose - see cut_footprints_from_grid - which is why the UI labels every answer
    # that used it APPROXIMATE.
    #
    # With no `move` parameters this is byte-for-byte the previous call, cache included:
    # the moved path is a separate branch that never touches the cache, rather than a
    # cache key that grows a field. The scene's real numbers cannot change because
    # somebody once asked a what-if.
    moves = _parse_moves(move)
    footprints = _footprints(objects)
    if moves:
        unknown = set(moves) - {obj_id for obj_id, _fp in footprints}
        if unknown:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"no such object in this scene: {sorted(unknown)}",
            )
        cut = cut_footprints_from_grid(
            grid.cells, grid.meta, [fp for obj_id, fp in footprints if obj_id in moves]
        )
        grid = OccupancyGrid(cells=cut, meta=grid.meta)
        footprints = [
            (obj_id, translate_footprint(fp, *moves[obj_id]) if obj_id in moves else fp)
            for obj_id, fp in footprints
        ]
        result = compute_reachability(
            grid, anchor, footprints, robot_radius_m=effective_radius, scene_id=scene_id
        )
    else:
        result = compute_reachability_cached(
            scene_id, layer.grid_path, grid, anchor, footprints,
            robot_radius_m=effective_radius, robot_height_m=effective_height,
        )
    return ReachabilityResponse(
        start_status=result.start.status,
        start_x=result.start.point[0] if result.start.point else None,
        start_z=result.start.point[1] if result.start.point else None,
        start_moved_m=result.start.moved_m,
        scene_id=scene_id,
        radius=result.radius,
        robot_id=robot_id,
        component_count=result.component_count,
        largest_component_size=result.largest_component_size,
        start_component_size=result.start_component_size,
        reachable_object_ids=[uuid.UUID(i) for i in result.reachable_ids],
        unreachable_reasons={uuid.UUID(i): reason for i, reason in result.unreachable_reasons.items()},
        path_length_m={uuid.UUID(i): length for i, length in result.path_length_m.items()},
        moved_object_ids=[uuid.UUID(i) for i in sorted(moves)],
        reachable_cells=result.reachable_cells.tolist(),
        unknown_cells_on_route=result.unknown_cells_on_route,
        obstacle_cells_on_route=result.obstacle_cells_on_route,
    )


@router.get("/{scene_id}/layers", response_model=SceneLayersResponse)
async def get_scene_layers(
    scene_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> SceneLayersResponse:
    """This scene's nvblox layer pack - the ESDF slice and the unobserved mask - on the
    scene's own occupancy grid.

    404 when no pack is sampled on this scene's grid, which is the common case and not an
    error: the packs are the geometry side's published output for the reconstructions it
    has run, and a scene without one simply has no ESDF layer to draw. The client falls
    back to the occupancy grid it already has."""
    scene = await _get_scene_or_404(session, scene_id)
    _require_done(scene)
    grid_meta = scene_ingest.read_grid_metadata(scene_service.scene_dir(scene_id))
    pack = scene_layers.find_pack_for_grid(grid_meta)
    if pack is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="no nvblox layer pack is sampled on this scene's grid",
        )

    esdf = pack.esdf_m
    observed = np.isfinite(esdf)
    return SceneLayersResponse(
        scene_id=scene_id,
        pack=pack.name,
        variant=pack.variant,
        resolution=grid_meta.resolution,
        origin_x=grid_meta.origin_x,
        origin_z=grid_meta.origin_z,
        width=grid_meta.width,
        height=grid_meta.height,
        slice_height_m=pack.slice_height_m,
        # NaN cannot travel through JSON; null is what "never observed" looks like on the
        # wire, and `unobserved` says the same thing as a mask for a client that wants it
        # without scanning for nulls.
        esdf_m=[[None if not observed[ix, iz] else float(esdf[ix, iz]) for iz in range(grid_meta.height)]
                for ix in range(grid_meta.width)],
        unobserved=[[bool(pack.unobserved[ix, iz]) for iz in range(grid_meta.height)]
                    for ix in range(grid_meta.width)],
        observed_cells=int(observed.sum()),
        obstacle_cells=int((observed & (esdf <= 0)).sum()),
        esdf_max_m=float(esdf[observed].max()) if observed.any() else None,
        quality=pack.quality,
    )


@router.get("/{scene_id}/commands", response_model=CommandListResponse)
async def get_commands(
    scene_id: uuid.UUID,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
) -> CommandListResponse:
    await _get_scene_or_404(session, scene_id)
    commands, total = await command_service.list_commands(session, scene_id, skip=skip, limit=limit)
    return CommandListResponse(
        scene_id=scene_id,
        total=total,
        skip=skip,
        limit=limit,
        items=[command_service.command_to_response(c) for c in commands],
    )


@router.delete("/{scene_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_scene(
    scene_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> None:
    scene = await _get_scene_or_404(session, scene_id)
    invalidate_grid(scene_id)
    await scene_service.delete_scene(session, scene)
