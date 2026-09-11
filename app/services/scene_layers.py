"""The nvblox layer pack for a scene: its ESDF slice and its unobserved mask.

`nvblox_v2/results/frontend_layers/<scene>/` is the geometry side's published output -
one directory per reconstruction, holding the layers a viewer can draw that the pipeline's
own `occupancy.npy` cannot express: how far each cell is from the nearest obstacle
(signed, metres) and which cells were never observed at all.

**A scene is matched to a pack by its GRID, never by its name.** The pack's
`grid_meta.json` describes the exact grid each array is sampled on - cell size, origin,
shape - and a pack is served only when that matches the scene's own
`occupancy_meta.json` to the last decimal. Names would be the obvious key and would be
wrong: the hero video produced two reconstructions (a 74-view 2 fps solve and a 1148-view
15 fps one) whose grids differ in origin, shape and frame, and the same pack directory
carries an array for one of them and a resampled array for the other. Matching on the
grid means a mismatch is a 404 rather than a map drawn half a metre out of place.

What the pack does NOT contain, and what this module therefore cannot serve: a BAND
obstacle mask. The pack ships a single horizontal ESDF slice at 0.3 m above the floor.
Measured on the hero grid, `esdf <= 0` marks 1753 cells where the scene's own 0.1-0.5 m
band mask marks 2857, and the slice's ESDF saturates at 0.3421 m - so a robot wider than
that scores no free cell on it at all. This is the risk HANDOFF section 6 records as item
10. The viewer draws the pack's slice AS a slice, labelled, and takes its obstacle cells
from the scene's own band; it does not present one as the other.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import settings
from app.services.scene_ingest import GridMeta

logger = logging.getLogger(__name__)

#: Floats in two JSON files written by different programs; equal to well under a
#: micrometre is equal. Not exact equality: the pack's grid_meta and the scene's
#: occupancy_meta are serialised independently from the same doubles.
_TOL = 1e-9

#: The two grids a pack can describe, and the file names sampled on each. "our_grid" is
#: the pack's own reconstruction; "frontend_grid" is the same layers resampled onto the
#: grid the API serves for a DIFFERENT solve of the same room (see the pack's own
#: `frontend_grid_resample.quality`, which calls that resample approximate).
_GRID_VARIANTS = (
    ("our_grid", "esdf_slice_0.3m.npy", "unobserved_mask.npy"),
    ("frontend_grid", "esdf_slice_frontend_grid.npy", "unobserved_mask_frontend_grid.npy"),
)


@dataclass(frozen=True)
class LayerPack:
    """One scene's layers, already checked to be on that scene's own grid."""

    name: str
    directory: Path
    #: Which block of the pack's grid_meta matched - "our_grid" or "frontend_grid".
    variant: str
    #: Signed distance to the nearest obstacle, metres, [ix][iz]. NaN = unobserved.
    esdf_m: np.ndarray
    #: True where the cell was never observed, [ix][iz].
    unobserved: np.ndarray
    #: Height above the floor the slice was cut at, metres.
    slice_height_m: float | None
    #: The pack's own verdict on this variant, when it records one. Passed through
    #: verbatim rather than summarised - it is the geometry side's statement, not ours.
    quality: str | None


def _grid_block(raw: dict, variant: str) -> tuple[float, float, float, int, int] | None:
    """(cell, origin_x, origin_z, width, height) for one of grid_meta's grid blocks, or
    None when that block is absent or shaped unexpectedly."""
    block = raw.get(variant)
    if not isinstance(block, dict):
        return None
    try:
        cell = float(block["cell_m"])
        if variant == "frontend_grid":
            return cell, float(block["origin_x"]), float(block["origin_z"]), *map(int, block["shape_xz"])
        # our_grid states its extent as ranges in its own (u, v) axes; the origin is the
        # low corner of each.
        return (
            cell,
            float(block["u_range"][0]),
            float(block["v_range"][0]),
            *map(int, block["shape_uv"]),
        )
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def _matches(block: tuple[float, float, float, int, int], grid: GridMeta) -> bool:
    cell, origin_x, origin_z, width, height = block
    return (
        math.isclose(cell, grid.resolution, abs_tol=_TOL)
        and math.isclose(origin_x, grid.origin_x, abs_tol=_TOL)
        and math.isclose(origin_z, grid.origin_z, abs_tol=_TOL)
        and width == grid.width
        and height == grid.height
    )


def _quality_for(raw: dict, variant: str) -> str | None:
    if variant != "frontend_grid":
        return None
    quality = raw.get("frontend_grid_resample", {}).get("quality", {})
    verdict = quality.get("verdict")
    why = quality.get("why")
    if not verdict:
        return None
    return f"{verdict}. {why}" if why else str(verdict)


def find_pack_for_grid(grid: GridMeta, root: str | Path | None = None) -> LayerPack | None:
    """The layer pack sampled on exactly `grid`, or None.

    Scans every pack directory rather than guessing a name from the scene - see the
    module docstring for why. Returns None (never raises) for a missing root, a malformed
    grid_meta, or arrays whose shape contradicts the metadata: a viewer overlay is not
    worth failing a scene over."""
    base = Path(root if root is not None else settings.frontend_layers_dir)
    if not base.is_dir():
        return None

    for directory in sorted(p for p in base.iterdir() if p.is_dir()):
        meta_path = directory / "grid_meta.json"
        if not meta_path.is_file():
            continue
        try:
            raw = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            logger.warning("layer pack %s has an unreadable grid_meta.json", directory.name)
            continue

        for variant, esdf_name, unobserved_name in _GRID_VARIANTS:
            block = _grid_block(raw, variant)
            if block is None or not _matches(block, grid):
                continue
            esdf_path, unobserved_path = directory / esdf_name, directory / unobserved_name
            if not esdf_path.is_file() or not unobserved_path.is_file():
                continue
            try:
                esdf = np.load(esdf_path)
                unobserved = np.load(unobserved_path)
            except (OSError, ValueError):
                logger.warning("layer pack %s has unreadable arrays", directory.name)
                continue
            # The metadata claimed this shape; if the array disagrees, the metadata is
            # not describing the array and nothing here can be trusted.
            if esdf.shape != (grid.width, grid.height) or unobserved.shape != (grid.width, grid.height):
                logger.warning(
                    "layer pack %s/%s: arrays are %s/%s but the grid is %dx%d - skipped",
                    directory.name, variant, esdf.shape, unobserved.shape, grid.width, grid.height,
                )
                continue
            return LayerPack(
                name=directory.name,
                directory=directory,
                variant=variant,
                esdf_m=esdf,
                unobserved=unobserved.astype(bool),
                slice_height_m=_slice_height(raw, directory),
                quality=_quality_for(raw, variant),
            )
    return None


def _slice_height(raw: dict, directory: Path) -> float | None:
    height = raw.get("our_grid", {}).get("height_above_floor_m")
    if isinstance(height, (int, float)):
        return float(height)
    try:
        plane = json.loads((directory / "floor_plane.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    value = plane.get("slice_height_above_floor_m")
    return float(value) if isinstance(value, (int, float)) else None
