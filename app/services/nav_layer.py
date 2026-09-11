"""Where the scene page's 2D navigation grid comes from - and there is exactly one answer.

Until §9.5 there were two. `gpu/stage_occupancy.py` wrote `<scene_dir>/occupancy.npy`, a
0.1-0.5 m point-density histogram, and that is what every router route read. Meanwhile the
pipeline's LAYERS step shipped an ESDF slice, and `nvblox_v2/reachability.py` rasterised a
third thing - a 0.1-1.5 m band - inside a script the front end never runs. Three answers to
"where can this robot stand", for one scene, and the UI only ever saw the weakest of them:
the 0.3 m slice cannot see a table top at 0.9 m, so it called that floor clear. Measured on
own_0901_161054, TurtleBot at r=0.1: 14 cells of route drawn straight through furniture.

So this module resolves ONE layer, the one the worker writes, and refuses rather than
falling back. A fallback is how you get two answers again, silently, on whichever scenes
happen to be old.

Layout, written by pipeline/steps/layers/band_mask.py (live scenes) or
pipeline/steps/layers/backfill_occupancy.py (scenes that predate it):

    <scene_dir>/layers/<name>/occupancy.npy         uint8 [ix, iz], 0 FREE 1 OBSTACLE 2 UNKNOWN
    <scene_dir>/layers/<name>/occupancy_meta.json   resolution / origin_x / origin_z /
                                                    width / height, plus band_min/band_max
                                                    and `source`

`occupancy_meta.json`'s five geometry fields are exactly what `scene_ingest.read_grid_metadata`
already parses, so the GridMeta the app builds is unchanged in shape - only its provenance
moved.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.services.scene_ingest import ArtifactParseError, GridMeta

logger = logging.getLogger(__name__)

LAYERS_DIRNAME = "layers"
GRID_NAME = "occupancy.npy"
HEIGHT_NAME = "obstacle_min_height.npy"
UNOBSERVED_NAME = "unobserved_mask.npy"
META_NAME = "occupancy_meta.json"

FREE, OBSTACLE, UNKNOWN = 0, 1, 2

#: Headroom over a robot's own height before a surface counts as blocking it. Matches
#: pipeline/steps/layers/band_mask.DEFAULT_MARGIN - one cell of vertical slack.
DEFAULT_MARGIN_M = 0.05

_CELLS_CACHE: "OrderedDict[tuple[str, int, float], np.ndarray]" = OrderedDict()
_CELLS_CACHE_MAXSIZE = 16


class NavLayerUnavailableError(Exception):
    """This scene has no band layer. Names the scene and how to produce one - a scene
    processed before §9.5 needs the backfill run over it, and that is a deliberate act with
    the owner's word, not something a GET request performs on its way past."""


@dataclass(frozen=True)
class NavLayer:
    grid_path: Path
    meta: GridMeta
    source: str
    band_m: tuple[float | None, float | None]
    #: The layer's SOURCE OF TRUTH: the lowest surface height per cell, NaN where the cell
    #: has none. Everything tri-state is derived from it, per robot.
    height_path: Path | None = None
    #: Cells nobody ever observed - no points at all, or too few to claim anything
    #: (`min_points_per_cell`). Shipped separately from the tri-state grid because the
    #: grid cannot carry it: derive the grid at a tall robot's height and an unobserved
    #: cell still reads UNKNOWN, but derive it at a short one's and a cell that IS
    #: observed, holding only a high surface, reads FREE - the two questions come apart.
    unobserved_path: Path | None = None
    margin_m: float = DEFAULT_MARGIN_M


def layer_dir(scene_dir: str | Path) -> Path | None:
    """The scene's layer directory, or None.

    A live scene writes `layers/<short scene name>/`; a backfilled one writes
    `layers/backfill/`. Both are single entries in practice; when both exist the one
    band_mask.py wrote wins, because it came from this scene's own mesh rather than from a
    point cloud with no observed/unobserved mask of its own.
    """
    root = Path(scene_dir) / LAYERS_DIRNAME
    if not root.is_dir():
        return None
    candidates = [d for d in sorted(root.iterdir())
                  if d.is_dir() and (d / GRID_NAME).exists() and (d / META_NAME).exists()]
    if not candidates:
        return None
    live = [d for d in candidates if d.name != "backfill"]
    return (live or candidates)[0]


def resolve(scene_dir: str | Path) -> NavLayer:
    """The one nav layer for this scene, or NavLayerUnavailableError naming the fix."""
    d = layer_dir(scene_dir)
    if d is None:
        raise NavLayerUnavailableError(
            f"no navigation layer under {Path(scene_dir) / LAYERS_DIRNAME}/. A scene "
            f"processed before the band layer needs backfilling: "
            f"`python3 pipeline/steps/layers/backfill_occupancy.py --apply` "
            f"(it is a dry run without --apply)."
        )
    raw = json.loads((d / META_NAME).read_text())
    try:
        meta = GridMeta(
            resolution=float(raw["resolution"]),
            origin_x=float(raw["origin_x"]),
            origin_z=float(raw["origin_z"]),
            width=int(raw["width"]),
            height=int(raw["height"]),
        )
    except KeyError as exc:
        raise ArtifactParseError(f"{d / META_NAME} missing required key {exc}") from exc
    hp = d / HEIGHT_NAME
    up = d / UNOBSERVED_NAME
    return NavLayer(grid_path=d / GRID_NAME, meta=meta,
                    source=raw.get("source", "unknown"),
                    band_m=(raw.get("band_min"), raw.get("band_max")),
                    height_path=hp if hp.exists() else None,
                    unobserved_path=up if up.exists() else None,
                    margin_m=float(raw.get("height_margin_m", DEFAULT_MARGIN_M)))


def unobserved_mask(layer: NavLayer) -> np.ndarray:
    """Which cells the scan never covered, as a bool array shaped like the grid.

    Falls back to `shipped grid == UNKNOWN` for a pack written before the mask was a file.
    That is the same set by construction - `backfill_occupancy` applies unobserved LAST,
    over everything else - so the fallback is exact, not an approximation, and it is a
    fallback about a FILE inside the one layer, never a fallback to another layer.
    """
    if layer.unobserved_path is not None:
        return np.load(layer.unobserved_path).astype(bool)
    return np.load(layer.grid_path) == UNKNOWN


def cells_for_height(layer: NavLayer, robot_height_m: float | None) -> np.ndarray:
    """The tri-state grid FOR A ROBOT THIS TALL, derived from the shipped height map.

    A fixed obstacle band asks "is something there". The question a ground robot has is "is
    something there that I cannot drive under", and the two differ by everything above the
    robot's roof. Measured on hero-74 (`7ccaa75d`): a 1.5 m band turned a bed top at
    0.694 m, a nightstand top at 0.684 m and a hanging curtain at 0.876 m into walls for a
    0.192 m TurtleBot, and cost it three otherwise-reachable objects. 3107 of the 3879 cells
    the band newly blocked had nothing below 0.4 m in them at all.

    Deriving this here, from one map, is what keeps the single source single: REACH and the
    scene page call the same function on the same file, so they cannot answer differently.

    `robot_height_m` None falls back to the grid the layer shipped - the default platform's
    view, which the metadata names.
    """
    if layer.height_path is None or robot_height_m is None:
        return np.load(layer.grid_path)
    key = (str(layer.height_path), layer.height_path.stat().st_mtime_ns,
           round(float(robot_height_m), 4))
    hit = _CELLS_CACHE.get(key)
    if hit is not None:
        _CELLS_CACHE.move_to_end(key)
        return hit
    lowest = np.load(layer.height_path)
    base = np.load(layer.grid_path)
    with np.errstate(invalid="ignore"):
        blocks = np.isfinite(lowest) & (lowest <= float(robot_height_m) + layer.margin_m)
    cells = np.full(lowest.shape, FREE, dtype=np.uint8)
    cells[blocks] = OBSTACLE
    # Unobserved is still never free, and it is carried over from the shipped grid rather
    # than recomputed - the height map says nothing about what was observed.
    cells[base == UNKNOWN] = UNKNOWN
    _CELLS_CACHE[key] = cells
    if len(_CELLS_CACHE) > _CELLS_CACHE_MAXSIZE:
        _CELLS_CACHE.popitem(last=False)
    return cells
