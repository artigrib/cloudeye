"""Scene lifecycle: creation, status transitions, and lookups.

A `Scene` is created exactly once per `Video` (enforced by both `scenes.video_id`
being UNIQUE and `get_or_create_scene`'s `ON CONFLICT DO NOTHING` insert), and moves
through `queued -> processing -> {done, failed}`.
"""

import json
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.models import Scene, SceneObject
from app.schemas import SceneResponse, SceneStatusResponse, StageTimingResponse
from app.services.stage_timings import StageTiming, read_stage_timings, refresh_stage_timings_from_log

logger = logging.getLogger(__name__)


class SceneNotFoundError(Exception):
    """Raised when a scene id doesn't exist."""


class SceneNotReadyError(Exception):
    """Raised when an endpoint that requires a `done` scene is called on one that isn't."""

    def __init__(self, scene: Scene):
        self.scene = scene
        super().__init__(f"Scene {scene.id} is not ready (status={scene.status})")


def scene_dir(scene_id: uuid.UUID) -> Path:
    """The on-disk directory a scene's fetched artifacts live in."""
    return Path(settings.upload_dir) / "scenes" / str(scene_id)


def msa_glb_path(scene_id: uuid.UUID) -> Path:
    """Fixed convention for the plain bootstrap MSA (Measured Scene Assembly,
    var/scratch/msa/SPEC.md) glTF export placed alongside a scene's other artifacts -
    `<scene_dir>/msa/scene.glb`. Deliberately NOT a Scene model column: MSA output is
    file-based script output from a separate pipeline (see docs/DECISIONS.md's "MSA
    Stage E1" entry), not part of this app's DB-backed data model, so there is nothing
    to migrate - a scene either has this file on disk or it doesn't. Pure path
    construction, no filesystem access - callers check `.is_file()` themselves.

    Kept as its own function (rather than folded into MSA_GLB_CANDIDATES) since it
    names the one file every MSA-exported scene is guaranteed to have - the fallback
    of last resort in resolve_msa_glb() below."""
    return scene_dir(scene_id) / "msa" / "scene.glb"


# Resolution order for GET /{scene_id}/msa-glb (app.routers.scenes.get_scene_msa_glb):
# the PLAIN bootstrap export first, textured dollhouse exports after it.
#
# This order was the other way round between 6aa87a5 (2026-09-07) and this change, and
# together with 2cd9fbc - which first made the main viewer request this route at all -
# it cost the main viewer 22 seconds on every scene open. On the hero scene the two
# candidates measure: `scene_assets_textured.glb` 44,195,740 bytes with 23 embedded
# textures (19.8 MB of images), `scene.glb` 99,716 bytes with 0 images - 443x. The
# textured file is parsed by GLTFLoader on the main thread (frontend's lib/glbCache.ts
# uses no Worker and no Draco/Meshopt decoder), so nothing else in the tab - not the
# point cloud's own GLB, not the /reachability request a platform switch issues - could
# start until it finished.
#
# The M6 shot-list finding this order originally came from (docs/DEMO_SHOTLIST.md: the
# live route served an untextured file for a scene that had a textured one) was about
# the DOLLHOUSE viewer, which is off the product surface as of 2026-09-08 (HANDOFF §8)
# and does not use this route anyway - MsaViewerPage takes its GLB from `?glb=` or a
# static fixture. Nothing that ships today wants 42 MB of baked furniture textures on
# scene open. A caller that does want the textured export can name it explicitly; the
# served filename is reported in the `X-Msa-Glb-Variant` response header either way.
#
# First name in this tuple that exists under <scene_dir>/msa/ wins.
MSA_GLB_CANDIDATES = (
    "scene.glb",  # plain bootstrap export, no embedded images - see msa_glb_path()
    "scene_assets_textured.glb",  # de-cluttered, most recent M7 dollhouse export
    "scene_textured.glb",  # earlier textured export naming
    "scene_assets.glb",  # de-cluttered but untextured
)


def resolve_msa_glb(scene_id: uuid.UUID) -> tuple[Path, str] | None:
    """Resolve which MSA glTF export to serve for this scene, walking
    MSA_GLB_CANDIDATES in order and returning the first (path, filename) that exists
    on disk under `<scene_dir>/msa/`, or None if the scene has no MSA export at all."""
    msa_dir = scene_dir(scene_id) / "msa"
    for filename in MSA_GLB_CANDIDATES:
        path = msa_dir / filename
        if path.is_file():
            return path, filename
    return None


# M7b addition (c): resolution order for GET /{scene_id}/msa-objects - the ObjectList
# panel was reading raw per-detection SceneObject DB rows (pre-de-clutter counts, e.g.
# "television (2)"/"lamp (4)" before M7's merge+cap stages ran), not the canonical
# de-cluttered set scripts/msa/bootstrap.py actually shipped. Prefer the audited file
# (M7 audit's manual/rule-based drops, e.g. a bogus mirror-as-television detection)
# over the plain bootstrap output, same "most-corrected-first" idea as
# MSA_GLB_CANDIDATES above - keep the two lists' *reasoning* in sync if either grows
# a new stage output, even though they resolve different file types.
MSA_OBJECTS_CANDIDATES = (
    "objects_audited.json",  # manual-audit-drop-filtered, when an audit ran
    "objects.json",  # plain bootstrap output - every MSA-exported scene has this
)


def resolve_msa_objects(scene_id: uuid.UUID) -> tuple[list[dict], str] | None:
    """Resolve which MSA `objects.json`-shaped export to read for this scene, walking
    MSA_OBJECTS_CANDIDATES in order. Returns (objects, filename) for the first
    candidate that exists AND parses as JSON with an `objects` list, or None if the
    scene has no MSA objects export at all (most scenes - MSA is optional/newer, same
    as resolve_msa_glb). Malformed JSON on an existing file is treated as "not found"
    rather than raising - this is read-only best-effort UI data, never load-bearing,
    so a bad file on disk degrades to "no MSA objects" instead of a 500."""
    msa_dir = scene_dir(scene_id) / "msa"
    for filename in MSA_OBJECTS_CANDIDATES:
        path = msa_dir / filename
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        objects = data.get("objects") if isinstance(data, dict) else None
        if not isinstance(objects, list):
            continue
        summaries = [
            {"id": o["id"], "label": o["label"]}
            for o in objects
            if isinstance(o, dict) and isinstance(o.get("id"), str) and isinstance(o.get("label"), str)
        ]
        return summaries, filename
    return None


def fit_prob_path(scene_id: uuid.UUID) -> Path:
    """Fixed convention for the fit-probability audit output
    (`scripts.audit.fit_prob`) placed alongside a scene's other MSA artifacts -
    `<scene_dir>/msa/fit_prob.json`. Same rationale as `msa_glb_path` above: file-
    based script output from a separate offline audit, not part of this app's
    DB-backed data model, so there is nothing to migrate - a scene either has this
    file on disk or it doesn't (most won't, same as MSA glTF exports). Pure path
    construction, no filesystem access - `app.routers.scenes.get_scene_fit_prob`
    checks `.is_file()` itself and 404s when absent."""
    return scene_dir(scene_id) / "msa" / "fit_prob.json"


def msa_critic_json_path(scene_id: uuid.UUID) -> Path:
    """Fixed convention for the scene critic's (scripts/audit/critic_rules.py +
    critic_vlm.py + critic_merge.py, var/scratch/QUEUE.md M14) merged output -
    `<scene_dir>/msa/critic.json`, mirroring `msa_glb_path()`'s single-fixed-name
    convention above (the critic writes exactly one file here per scene, no
    candidate/fallback list needed the way MSA_GLB_CANDIDATES has several historical
    export names to fall back through). Same "not a DB column" reasoning as
    msa_glb_path: this is file-based script output from a separate, optional
    pipeline stage - a scene either has a critic report on disk or it doesn't, and
    most won't, since the critic is opt-in and run manually per scene, same as MSA
    itself. Pure path construction, no filesystem access - callers check
    `.is_file()` themselves (see get_scene_critic in app.routers.scenes)."""
    return scene_dir(scene_id) / "msa" / "critic.json"


def msa_meta_path(scene_id: uuid.UUID) -> Path:
    """The bootstrap `scene_meta.json` that sits next to `msa_glb_path` - same fixed
    `<scene_dir>/msa/` convention, same "not a DB column" reasoning. It carries the yaw
    normalization the GLB's geometry was rotated by (scripts/msa/bootstrap.py's
    `yaw_correction_rad` about `yaw_rotation_center_xy`), which the frontend needs to
    overlay the GLB back onto the UNROTATED point cloud (see docs/DECISIONS.md's
    "morning-2" entry). Pure path construction, no filesystem access."""
    return scene_dir(scene_id) / "msa" / "scene_meta.json"


# The subset of bootstrap's scene_meta.json that GET /api/scenes/{id}/msa-meta exposes -
# the yaw normalization (everything the inverse overlay transform needs), the vertical
# extent, and the room polygon. The full file also carries every yaw-policy diagnostic
# (wall-hull/histogram cross-checks, rect sizes, ...) and per-object data the viewer has
# no use for; keeping the wire contract to this explicit list means bootstrap can add
# fields freely without them leaking into the API.
MSA_META_FIELDS: tuple[str, ...] = (
    "yaw_applied",
    "yaw_correction_rad",
    "yaw_correction_deg",
    "yaw_rotation_center_xy",
    "yaw_method",
    "floor_y",
    "ceiling_y",
    "room_polygon",
)


def read_msa_meta(path: Path) -> dict | None:
    """Reads a bootstrap `scene_meta.json` and returns only `MSA_META_FIELDS` (absent
    keys are omitted, not null-filled - a pre-yaw meta file simply has no yaw keys and
    the frontend treats a missing `yaw_applied` as "not rotated"). None if the file
    doesn't exist. A malformed file raises (json.JSONDecodeError / ValueError) rather
    than being masked as "no meta" - that's a broken export, not an absent one."""
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object at the top level")
    return {k: data[k] for k in MSA_META_FIELDS if k in data}


async def get_or_create_scene(
    session: AsyncSession, *, video_id: uuid.UUID, project_id: uuid.UUID, label: str | None = None
) -> tuple[Scene, bool]:
    """Get the scene for this video, creating it (status="queued") if it doesn't exist yet.

    Uses `INSERT ... ON CONFLICT (video_id) DO NOTHING` so this is safe even if called
    concurrently (e.g. a retried enqueue racing the first attempt) - there can never be
    two scenes for one video. Returns (scene, created) where `created` is False if a scene
    already existed (whatever its status - the caller decides what to do with an existing
    non-queued scene).
    """
    stmt = (
        pg_insert(Scene)
        .values(video_id=video_id, project_id=project_id, label=label, status="queued")
        .on_conflict_do_nothing(index_elements=["video_id"])
        .returning(Scene.id)
    )
    result = await session.execute(stmt)
    new_id = result.scalar_one_or_none()
    await session.commit()

    if new_id is not None:
        scene = await session.get(Scene, new_id)
        assert scene is not None
        return scene, True

    scene = await get_scene_by_video(session, video_id)
    assert scene is not None, "scene must exist if the insert conflicted on video_id"
    return scene, False


async def mark_processing(session: AsyncSession, scene: Scene) -> None:
    scene.status = "processing"
    scene.error_message = None
    await session.commit()


async def mark_failed(session: AsyncSession, scene: Scene, message: str, *, completed: bool = False) -> None:
    """`completed=True` also stamps `completed_at`, matching mark_done - used where a
    failure is genuinely terminal but wasn't already covered by one of the specific
    recoverable-error branches (pipeline_orchestrator's catch-all), so the row doesn't
    look like it might still be running once error_message is glanced past."""
    logger.error("Scene %s failed: %s", scene.id, message)
    scene.status = "failed"
    scene.error_message = message
    if completed:
        scene.completed_at = datetime.now(timezone.utc)
    await session.commit()


async def mark_done(session: AsyncSession, scene: Scene) -> None:
    scene.status = "done"
    scene.error_message = None
    scene.completed_at = datetime.now(timezone.utc)
    await session.commit()


def _current_stage_timings(scene: Scene) -> list[StageTiming]:
    """This scene's stage timings right now: re-parsed live from `logs/worker.log`
    while still `processing` (so a poll mid-run always reflects the latest
    transition), or read from the persisted `stage_timings.json` once terminal (see
    app.services.stage_timings and pipeline_orchestrator.run_pipeline_for_scene,
    which persists it once the run finishes)."""
    d = scene_dir(scene.id)
    if scene.status == "processing":
        return refresh_stage_timings_from_log(d)
    return read_stage_timings(d)


def apply_stage_timings(resp: SceneResponse | SceneStatusResponse, scene: Scene) -> None:
    """Populate `resp.stage`/`resp.stage_timings` in place - shared by SceneResponse
    (GET /scenes/{id}) and SceneStatusResponse (GET /scenes/{id}/status, GET
    /videos/{id}/scene)."""
    timings = _current_stage_timings(scene)
    resp.stage_timings = [
        StageTimingResponse(stage=t.stage, started_at=t.started_at, duration_sec=t.duration_sec)
        for t in timings
    ]
    if timings:
        resp.stage = timings[-1].stage


def build_status_response(scene: Scene) -> SceneStatusResponse:
    """Shared by GET /scenes/{id}/status and GET /videos/{id}/scene (both return the
    same lightweight polling shape)."""
    resp = SceneStatusResponse.model_validate(scene)
    apply_stage_timings(resp, scene)
    return resp


async def get_scene(session: AsyncSession, scene_id: uuid.UUID) -> Scene | None:
    return await session.get(Scene, scene_id)


async def get_scene_with_objects(session: AsyncSession, scene_id: uuid.UUID) -> Scene | None:
    result = await session.execute(
        select(Scene).where(Scene.id == scene_id).options(selectinload(Scene.objects))
    )
    return result.scalar_one_or_none()


async def get_scene_by_video(session: AsyncSession, video_id: uuid.UUID) -> Scene | None:
    result = await session.execute(select(Scene).where(Scene.video_id == video_id))
    return result.scalar_one_or_none()


async def list_scene_objects(session: AsyncSession, scene_id: uuid.UUID) -> list[SceneObject]:
    result = await session.execute(
        select(SceneObject).where(SceneObject.scene_id == scene_id).order_by(SceneObject.name)
    )
    return list(result.scalars().all())


async def delete_scene(session: AsyncSession, scene: Scene) -> None:
    """Delete a scene's on-disk artifact directory, then its DB row (cascades objects/commands)."""
    d = scene_dir(scene.id)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
        logger.info("Removed scene directory %s", d)
    await session.delete(scene)
    await session.commit()


async def set_usd_path(session: AsyncSession, scene: Scene, path: str) -> None:
    """Record the most recently generated Isaac Sim USD export path (see usd_export.py
    - the actual per-params cache lookup is by filename on disk, not this field)."""
    scene.usd_path = path
    await session.commit()


async def apply_scene_artifacts(session: AsyncSession, scene: Scene, **fields) -> None:
    """Set every scene field the pipeline produced (paths, grid metadata, floor/ceiling,
    robot start) in one go. Does NOT commit or change status - the caller
    (pipeline_orchestrator) does that as part of the same transaction as
    `replace_scene_objects`, so a scene is never left half-populated."""
    for key, value in fields.items():
        setattr(scene, key, value)


async def replace_scene_objects(session: AsyncSession, scene: Scene, objects: list[dict]) -> None:
    """Replace this scene's detected objects: delete whatever's there first, then
    insert the new set. Most scenes are only ever populated once (see
    get_or_create_scene's idempotency), where the delete is just a no-op against zero
    rows - but a `rerun` (pipeline_orchestrator.run_pipeline_for_scene(..., rerun=True))
    processes the same scene id a second time, and without deleting first this would
    silently double up every object row rather than actually replacing them."""
    await session.execute(delete(SceneObject).where(SceneObject.scene_id == scene.id))
    for obj in objects:
        session.add(SceneObject(scene_id=scene.id, **obj))
