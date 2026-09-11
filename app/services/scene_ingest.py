"""Parse GPU pipeline output artifacts into plain dataclasses - deliberately DB-free, so
it's unit-testable against `~/gpu-backup` fixtures without a database or GPU access.

Reads the plain-JSON mirrors (`alignment.json`, `occupancy_meta.json`,
`scene_objects.json`, `cameras_aligned.json`) rather than the `.npz`/binary artifacts
directly - this backend never needs numpy for reading these, they're a direct
`json.load`. (`pathfinding.py` does need numpy, but only for the occupancy grid array
itself, not for these metadata files.)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


class ArtifactParseError(Exception):
    """Raised when a required GPU pipeline artifact is missing or malformed."""


@dataclass(frozen=True)
class ParsedObject:
    name: str
    description: str | None
    pos: tuple[float, float, float]
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    num_views: int
    num_points: int
    is_fragment: bool
    mesh_path: str | None


@dataclass(frozen=True)
class AlignmentInfo:
    floor_y: float
    ceiling_y: float
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    residual_tilt_deg: float
    det_r: float
    is_reflection: bool
    # Below-floor phantom-point clip diagnostics (schema_version 3+, see stage_align.py
    # and setup-log.md's "reflective-floor phantom points" entry). Default to "clean"
    # values for alignment.json written by an older stage_align.py that predates this.
    points_below_floor_frac: float = 0.0
    reflective_floor_suspected: bool = False
    # schema_version 4+: the below-floor clip removed so much more than a reflective
    # floor would (>3x reflective_floor_frac_threshold) that the selected floor plane
    # itself is suspect, not just glossy - see floor_ceiling.py's select_floor_candidate.
    floor_likely_wrong: bool = False


@dataclass(frozen=True)
class GridMeta:
    resolution: float
    origin_x: float
    origin_z: float
    width: int
    height: int


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise ArtifactParseError(f"required artifact missing: {path}")
    return path


def parse_scene_objects(json_path: str | Path, scene_dir: str | Path) -> list[ParsedObject]:
    """Parse scene_objects.json into ParsedObject rows, rebasing each object's point
    cloud path onto `scene_dir` (the local directory the results were fetched into).

    Handles both the current pipeline's relative paths (e.g. "scene_objects/sink_0.ply")
    and an older schema's absolute GPU paths (e.g.
    "/workspace/data/output_horizontal/run2/scene_objects/sink_0.ply") - and an even
    older schema that lacks `likely_fragment` entirely, defaulting it to False.
    """
    json_path = _require_file(Path(json_path))
    scene_dir = Path(scene_dir)

    try:
        raw = json.loads(json_path.read_text())
    except json.JSONDecodeError as exc:
        raise ArtifactParseError(f"malformed JSON in {json_path}: {exc}") from exc
    if not isinstance(raw, list):
        raise ArtifactParseError(f"{json_path} is not a JSON array (schema_version 2+ expected)")

    objects: list[ParsedObject] = []
    for i, entry in enumerate(raw):
        try:
            name = entry["label"]
            centroid = entry["centroid_xyz"]
            bbox_min = entry["bbox_min"]
            bbox_max = entry["bbox_max"]
            num_points = entry["point_count"]
            num_views = entry["source_view_count"]
        except KeyError as exc:
            raise ArtifactParseError(f"{json_path} entry {i} missing required key {exc}") from exc

        is_fragment = bool(entry.get("likely_fragment", False))

        mesh_path: str | None = entry.get("point_cloud_path")
        if mesh_path:
            p = Path(mesh_path)
            if p.is_absolute():
                # Older schema wrote an absolute GPU-host path - only the filename
                # under scene_objects/ is portable.
                mesh_path = str(scene_dir / "scene_objects" / p.name)
            else:
                mesh_path = str(scene_dir / p)

        objects.append(
            ParsedObject(
                name=name,
                description=entry.get("description"),
                pos=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
                bbox_min=(float(bbox_min[0]), float(bbox_min[1]), float(bbox_min[2])),
                bbox_max=(float(bbox_max[0]), float(bbox_max[1]), float(bbox_max[2])),
                num_views=int(num_views),
                num_points=int(num_points),
                is_fragment=is_fragment,
                mesh_path=mesh_path,
            )
        )
    return objects


def read_alignment(scene_dir: str | Path) -> AlignmentInfo:
    scene_dir = Path(scene_dir)
    path = _require_file(scene_dir / "alignment.json")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ArtifactParseError(f"malformed JSON in {path}: {exc}") from exc

    try:
        return AlignmentInfo(
            floor_y=float(raw["floor_y"]),
            ceiling_y=float(raw["ceiling_y"]),
            bbox_min=tuple(float(v) for v in raw["bbox_min"]),
            bbox_max=tuple(float(v) for v in raw["bbox_max"]),
            residual_tilt_deg=float(raw["residual_tilt_deg"]),
            det_r=float(raw["det_r"]),
            is_reflection=bool(raw["is_reflection"]),
            points_below_floor_frac=float(raw.get("points_below_floor_frac", 0.0)),
            reflective_floor_suspected=bool(raw.get("reflective_floor_suspected", False)),
            floor_likely_wrong=bool(raw.get("floor_likely_wrong", False)),
        )
    except KeyError as exc:
        raise ArtifactParseError(f"{path} missing required key {exc}") from exc


def read_grid_metadata(scene_dir: str | Path) -> GridMeta:
    scene_dir = Path(scene_dir)
    path = _require_file(scene_dir / "occupancy_meta.json")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ArtifactParseError(f"malformed JSON in {path}: {exc}") from exc

    try:
        return GridMeta(
            resolution=float(raw["resolution"]),
            origin_x=float(raw["origin_x"]),
            origin_z=float(raw["origin_z"]),
            width=int(raw["width"]),
            height=int(raw["height"]),
        )
    except KeyError as exc:
        raise ArtifactParseError(f"{path} missing required key {exc}") from exc


def read_camera_track(scene_dir: str | Path) -> list[tuple[float, float, float]]:
    scene_dir = Path(scene_dir)
    path = _require_file(scene_dir / "cameras_aligned.json")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ArtifactParseError(f"malformed JSON in {path}: {exc}") from exc
    cameras = raw.get("cameras", [])
    return [(float(c[0]), float(c[1]), float(c[2])) for c in cameras]


_RAW_FRAME_RE = re.compile(r"^frame_(\d+)\.jpg$")


def read_keyframe_timestamps(scene_dir: str | Path, num_cameras: int) -> list[float] | None:
    """Video-playback timestamp (seconds) for each of `cameras_aligned.json`'s poses, in
    the same capture order - backs the camera-path/video-player sync in the 3D and 2D
    views. Display only, and never touches pose data itself.

    Neither `cameras_aligned.json` nor keyframe images carry a timestamp field, so this
    reconstructs one from two artifacts vidmap.py already writes:
      - keyframes.json's "fps": the exact rate `gpu/vidmap.py`'s extract_frames() passed
        to `ffmpeg -vf fps=<fps>`, which resamples onto an exact uniform time grid - raw
        frame N (1-indexed, from its "frame_NNNNNN.jpg" filename) always lands at
        (N-1)/fps seconds of source video, a guarantee of ffmpeg's fps filter, not an
        estimate.
      - manifest.json: vidmap.py's per-raw-frame record of which ones were `kept` as
        keyframes, in original capture order - so the i-th `kept` entry there is
        cameras_aligned.json's i-th pose, *provided* nothing removed keyframes after
        manifest.json was written.

    That proviso is exactly what stage_keyframes.py's own VRAM-budget subsampling does
    when a walkthrough yields more than max_keyframes: it deletes some already-kept
    keyframe files by index, a selection this function does not replicate (unverified
    against real data, unlike the fps-grid reconstruction above). Rather than risk
    silently pairing a timestamp with the wrong pose, this backs off to None (not an
    error) whenever the reconstructed count doesn't match `num_cameras` - along with the
    other honest-degradation cases: manifest.json/keyframes.json missing (all scenes
    ingested before either existed) or malformed, keyframes.json lacking a usable "fps",
    or a manifest entry that doesn't parse as a "frame_NNNNNN.jpg" raw-frame name."""
    manifest_path = Path(scene_dir) / "manifest.json"
    keyframes_path = Path(scene_dir) / "keyframes.json"
    if not manifest_path.exists() or not keyframes_path.exists():
        return None

    try:
        manifest = json.loads(manifest_path.read_text())
        keyframes_meta = json.loads(keyframes_path.read_text())
    except json.JSONDecodeError:
        return None
    if not isinstance(manifest, list):
        return None

    fps = keyframes_meta.get("fps") if isinstance(keyframes_meta, dict) else None
    if not isinstance(fps, (int, float)) or fps <= 0:
        return None

    timestamps: list[float] = []
    for entry in manifest:
        if not isinstance(entry, dict) or not entry.get("kept"):
            continue
        match = _RAW_FRAME_RE.match(entry.get("source_frame", ""))
        if not match:
            return None
        frame_index = int(match.group(1))
        timestamps.append((frame_index - 1) / fps)

    if len(timestamps) != num_cameras:
        return None
    return timestamps


def read_vocab_source(scene_dir: str | Path) -> str | None:
    """vocab.json's "source" field - "openrouter"/"vertex"/"override" going forward
    (see gpu/stage_vocab.py; a provider failure now fails the whole scene instead of
    writing vocab.json at all - see VocabProviderError). Scenes built before that
    change can still carry the retired "default_no_key"/"default_no_frames"/
    "default_after_error" values - not produced anymore, but still read correctly
    here for those older scenes. Unlike the other read_* functions here, a missing or
    malformed file is NOT an
    ArtifactParseError: vocab.json isn't load-bearing for anything else this module
    parses, and a hard failure to timeout in stage_vocab.py itself (see
    gpu/run_pipeline.sh's STAGE_TIMEOUT) means the file may simply never have been
    written for an otherwise-successful-looking scene - that shouldn't turn into a
    scene ingestion failure over one provenance field."""
    path = Path(scene_dir) / "vocab.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    source = raw.get("source")
    return source if isinstance(source, str) else None


def read_vocab_model(scene_dir: str | Path) -> str | None:
    """vocab.json's "model" field - the exact model string actually used (e.g.
    "google/gemma-4-31b-it"), absent for the hardcoded-default sources where no model
    was called at all. Same tolerant-of-missing/malformed-file treatment as
    read_vocab_source() above, for the same reason."""
    path = Path(scene_dir) / "vocab.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    model = raw.get("model")
    return model if isinstance(model, str) else None


def read_vocab_entries(scene_dir: str | Path) -> list[dict] | None:
    """vocab.json's "objects" array, reduced to just {"name", "size_class"} per entry -
    the same shape gpu/stage_vocab.py's `vocab_override` job param accepts (and the
    shape build_vocab_entries() there re-derives the SAM3 threshold fields from), so
    this is exactly what a later `rerun` needs to feed back in as next run's
    vocab_override (see pipeline_orchestrator.resolve_vocab_override) to reuse this
    scene's own vocabulary instead of making a fresh LLM call. The derived
    score_threshold/dbscan_*/min_cluster fields vocab.json also carries per entry are
    dropped here on purpose - they're a pure function of size_class
    (SIZE_CLASS_PARAMS), re-derived on the next run rather than persisted, so a future
    tuning change to those constants still applies even to a cached-vocabulary rerun.

    Same tolerant-of-missing/malformed-file/absent-field treatment as
    read_vocab_source()/read_vocab_model() above, for the same reason - not
    load-bearing for scene ingestion, just a cache of vocab.json's content."""
    path = Path(scene_dir) / "vocab.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    objects = raw.get("objects")
    if not isinstance(objects, list):
        return None
    entries = []
    for item in objects:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name:
            continue
        entry = {"name": name}
        size_class = item.get("size_class")
        if isinstance(size_class, str):
            entry["size_class"] = size_class
        entries.append(entry)
    return entries or None


def choose_robot_start(
    cameras: list[tuple[float, float, float]], grid: GridMeta
) -> tuple[float, float]:
    """Pick a robot starting position: the first camera position that actually falls
    inside the occupancy grid's world-space bounds, clamped in the same sense
    `pathfinding.world_to_cell` accepts (a 2-cell floating-point-slop margin).

    Not simply `cameras[0]`: a walkthrough recording routinely starts a few frames
    before the operator has stepped into the room (still in a doorway/hallway while
    hitting record), and the occupancy grid is built from the room's own reconstructed
    point-cloud extent, not the full camera track - so `cameras[0]` can legitimately
    sit outside the grid even on a perfectly good scene. Confirmed on a real scene: its
    first 5 of 47 camera poses (z from -0.001 to -0.605) were about 0.5-0.65m past the
    grid's near z-edge (-0.657); camera 5 onward (z=-0.895) was inside. Walking forward
    to the first in-bounds pose is a direct fix for that, not a workaround - it's
    picking a *different, more correct* input, not tolerating/catching the failure
    `world_to_cell` correctly raises on invalid input downstream.

    Doesn't snap to a specific free cell here - pathfinding.py's own snap_if_blocked
    (in resolve_robot_start) does that with actual grid cost data, this just needs a
    reasonable in-bounds (x, z) to store on the scene."""
    if not cameras:
        # Fall back to the grid's center if there's no camera track for some reason -
        # better than leaving robot_start null and breaking the first command a user
        # issues.
        return (
            grid.origin_x + grid.width * grid.resolution / 2,
            grid.origin_z + grid.height * grid.resolution / 2,
        )

    margin_m = 2 * grid.resolution
    x_min, x_max = grid.origin_x - margin_m, grid.origin_x + grid.width * grid.resolution + margin_m
    z_min, z_max = grid.origin_z - margin_m, grid.origin_z + grid.height * grid.resolution + margin_m
    for x, _y, z in cameras:
        if x_min <= x <= x_max and z_min <= z <= z_max:
            return (x, z)

    # No camera pose ever entered the grid's bounds - the track and the reconstructed
    # room don't overlap at all. That's a genuinely broken scene, not something to paper
    # over with a synthetic point; keep the old cameras[0] behavior so it fails loudly
    # downstream in resolve_robot_start's world_to_cell call, exactly as that
    # function's own docstring says it must.
    x, _y, z = cameras[0]
    return (x, z)
