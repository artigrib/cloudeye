"""T15f Part A: index of the CC0 furniture asset packs under
`var/assets/furniture` (see `ATTRIBUTION.md` there and
`docs/ASSET_ATTRIBUTION.md`) keyed by a small furniture class vocabulary, so
`asset_fallback.py` can drop a real textured model into any object footprint
the generator (Stage B1) did not produce an accepted mesh for.

Canonical size per candidate (metres, Y-up, `canonical_aabb_m = (x, y, z)`
matching the GLB's own axes):

* **Poly Haven** (primary source): the API's real-world `dimensions` triple
  (millimetres) is authoritative - the GLB's AABB is the same three
  measurements in some axis permutation, so the two are matched *by sorted
  rank* (smallest api value -> the GLB axis with the smallest extent, ...)
  and the permutation is recorded (`axis_permutation`: for GLB axis i, which
  api index was used). `dimension_mismatch_rel` is the worst relative
  disagreement after matching - 0 for 23/24 assets; `desk_lamp_arm_01` is
  posed differently from its API bbox (0.20 m vs 0.41 m on its thin axis) and
  is flagged, not dropped.
* **Kenney Furniture Kit** (fallback only, for classes Poly Haven lacks - floor
  lamp, monitor, pillow, rug, ...): the GLB AABB in the pack's native units
  times `KENNEY_UNIT_TO_M`. The kit is built on a 0.5-unit grid, NOT 1 unit =
  1 m: `bedDouble` is 1.125 u long / 0.956 u wide, `chair` 0.47 u tall,
  `desk` 0.734 u long, `lampRoundFloor` 0.86 u tall, `loungeSofa` 0.98 u
  long - all ~0.55x a plausible real size, so 1 u = 1.8 m (2.0 m bed,
  0.85 m chair, 1.3 m desk, 1.55 m floor lamp, 1.75 m sofa).
* **mastjie household goods**: `obtained: false` in the inventory - listed as
  a source name only, contributes nothing until the pack appears.

Tiers: every furniture candidate is `tier="furniture"`. **T16b** adds
`tier="small"`: Google Scanned Objects (CC BY 4.0, attribution in
`var/assets/small_objects/ATTRIBUTION.md`) indexed from a second root
(`build_index(..., small_root=...)`, `INVENTORY.json` with `models: {name:
{class, glb_path, aabb_extents_m}}`). Canonical size = the converted GLB's own
AABB in metres (GSO is metric; the conversion rotated Gazebo's Z-up to Y-up).
Small-tier candidates are only ever reached through `SMALL_LABEL_SYNONYMS`
(detected label -> small class), and `asset_fallback` refuses them for any
object whose label is not a small-object label - a small asset is never placed
where SAM3 did not detect a small object.

The index is cached as `<assets_root>/index.json` and rebuilt whenever either
`INVENTORY.json` is newer than the cache or `INDEX_SCHEMA_VERSION` changed.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_ASSETS_ROOT = Path("var/assets/furniture")
DEFAULT_SMALL_ROOT = Path("var/assets/small_objects")
INDEX_SCHEMA_VERSION = 2  # 2: small tier (T16b)
INDEX_FILENAME = "index.json"

# See module docstring for the derivation.
KENNEY_UNIT_TO_M = 1.8
UNIT_FACTORS = {"kenney": KENNEY_UNIT_TO_M, "polyhaven": 1.0, "mastjie": 1.0, "gso": 1.0}

# Uniformly scaled, never distorted per-axis (asset_fallback.py). `floor_lamp`
# is the tall-lamp sub-class the "lamp" label resolves to first when the
# measured object is at least FLOOR_LAMP_MIN_HEIGHT_M tall. Cups/bottles are
# round too: a mug fitted per-axis into a thin SAM3 footprint becomes a slab.
ROUND_CLASSES = frozenset({"lamp", "floor_lamp", "cup", "bottle", "glass", "can", "vase"})
FLOOR_LAMP_MIN_HEIGHT_M = 1.0

# Source preference within one class: the first source that has any candidate
# for the class wins outright (Kenney is a fallback, never mixed with Poly Haven).
SOURCE_PRIORITY = ("polyhaven", "mastjie", "kenney", "gso")

TIER_FURNITURE = "furniture"
TIER_SMALL = "small"

# Small-object index classes (tier="small" candidates live only under these).
SMALL_CLASSES = frozenset({"cup", "bottle", "can", "shoe", "book", "bag", "laptop", "phone", "keyboard", "mouse", "toy", "headphones", "box", "bucket", "vase"})

# Detected SAM3 label -> ordered small classes. `glass` (the most frequent small
# detection in the own scenes, 47 in own_0901_155452) has NO drinking glass in
# GSO; a mug/cup is the documented visual proxy. Labels absent here (pillow,
# curtain, lamp, ...) can never reach a small-tier candidate.
SMALL_LABEL_SYNONYMS: dict[str, tuple[str, ...]] = {
    "glass": ("cup",),
    "drinking glass": ("cup",),
    "cup": ("cup",),
    "mug": ("cup",),
    "coffee cup": ("cup",),
    "bottle": ("bottle",),
    "water bottle": ("bottle",),
    "can": ("can", "bottle"),
    "vase": ("vase", "bottle"),
    "shoe": ("shoe",),
    "shoes": ("shoe",),
    "sneaker": ("shoe",),
    "boot": ("shoe",),
    "book": ("book",),
    "books": ("book",),
    "bag": ("bag",),
    "backpack": ("bag",),
    "handbag": ("bag",),
    "laptop": ("laptop",),
    "phone": ("phone",),
    "cell phone": ("phone",),
    "keyboard": ("keyboard",),
    "mouse": ("mouse",),
    "toy": ("toy",),
    "teddy bear": ("toy",),
    "headphones": ("headphones",),
    "box": ("box",),
    "bucket": ("bucket",),
}

# Largest measured extent (m) for which a small-tier asset is still placed. SAM3
# `glass` also fires on window panes (own_0901_155452 glass_19: 1.3 x 1.2 x 2.4 m);
# a mug must never be scaled into one - the placeholder stays instead.
SMALL_TIER_MAX_EXTENT_M = 0.6

# Pipeline label (SAM3 prompt vocabulary, lower-cased, trailing `_<n>` index
# stripped) -> ordered index classes to try. First class with candidates wins.
LABEL_SYNONYMS: dict[str, tuple[str, ...]] = {
    "bed": ("bed",),
    "chair": ("chair",),
    "armchair": ("chair",),
    "office chair": ("chair",),
    "stool": ("stool", "chair"),
    "desk": ("desk", "table"),
    "table": ("table", "desk"),
    "dining table": ("table",),
    "coffee table": ("table",),
    "nightstand": ("nightstand",),
    "night stand": ("nightstand",),
    "side table": ("nightstand", "table"),
    "bedside table": ("nightstand",),
    "lamp": ("lamp",),
    "floor lamp": ("floor_lamp", "lamp"),
    "table lamp": ("lamp",),
    "desk lamp": ("lamp",),
    "television": ("tv",),
    "tv": ("tv",),
    "monitor": ("monitor", "tv"),
    "computer monitor": ("monitor", "tv"),
    "screen": ("monitor", "tv"),
    "laptop": ("laptop",),
    "sofa": ("sofa",),
    "couch": ("sofa",),
    "wardrobe": ("cabinet",),
    "cabinet": ("cabinet",),
    "shelf": ("cabinet",),
    "shelves": ("cabinet",),
    "bookshelf": ("cabinet",),
    "bookcase": ("cabinet",),
    "dresser": ("cabinet",),
    "cupboard": ("cabinet",),
    "curtain": ("curtain",),
    "pillow": ("pillow",),
    "cushion": ("pillow",),
    "rug": ("rug",),
    "carpet": ("rug",),
    "plant": ("plant",),
    "potted plant": ("plant",),
    "bench": ("bench",),
    "toilet": ("toilet",),
    "bathtub": ("bathtub",),
    "shower": ("shower",),
    "sink": ("sink",),
    "mirror": ("mirror",),
    "refrigerator": ("fridge",),
    "fridge": ("fridge",),
    "stove": ("stove",),
    "oven": ("stove",),
    "microwave": ("microwave",),
    "washing machine": ("washer",),
    "washer": ("washer",),
    "dryer": ("washer",),
    "trash can": ("trashcan",),
    "bin": ("trashcan",),
    "box": ("box",),
    "book": ("books",),
    "books": ("books",),
    "speaker": ("speaker",),
    "radio": ("radio",),
    "keyboard": ("keyboard",),
    "mouse": ("mouse",),
    "coat rack": ("coat_rack",),
    "teddy bear": ("toy",),
}

# Poly Haven `class_hint` strings (from INVENTORY.json) -> index class.
POLYHAVEN_CLASS_HINTS = {
    "sofa/couch": "sofa",
    "chair/armchair": "chair",
    "table (dining/coffee)": "table",
    "desk": "desk",
    "nightstand/side table/bedside": "nightstand",
    "bed": "bed",
    "cabinet/shelf/bookshelf/wardrobe/dresser": "cabinet",
    "lamp (floor/table lamp)": "lamp",
    "tv/television/monitor": "tv",
}

# Kenney Furniture Kit GLB stem -> index class. Explicit so nothing is guessed;
# structural pieces (walls, floors, stairs, doorways, paneling) and ceiling/
# wall-mounted fixtures are deliberately absent -> reported under `unmapped`.
KENNEY_CLASSES: dict[str, str] = {
    "bedDouble": "bed",
    "bedSingle": "bed",
    "bedBunk": "bed",
    "chair": "chair",
    "chairCushion": "chair",
    "chairDesk": "chair",
    "chairModernCushion": "chair",
    "chairModernFrameCushion": "chair",
    "chairRounded": "chair",
    "loungeChair": "chair",
    "loungeChairRelax": "chair",
    "loungeDesignChair": "chair",
    "stoolBar": "stool",
    "stoolBarSquare": "stool",
    "loungeSofa": "sofa",
    "loungeSofaLong": "sofa",
    "loungeDesignSofa": "sofa",
    "loungeSofaOttoman": "sofa",
    "loungeSofaCorner": "sofa",
    "loungeDesignSofaCorner": "sofa",
    "table": "table",
    "tableCloth": "table",
    "tableCross": "table",
    "tableCrossCloth": "table",
    "tableGlass": "table",
    "tableRound": "table",
    "tableCoffee": "table",
    "tableCoffeeGlass": "table",
    "tableCoffeeGlassSquare": "table",
    "tableCoffeeSquare": "table",
    "desk": "desk",
    "deskCorner": "desk",
    "sideTable": "nightstand",
    "sideTableDrawers": "nightstand",
    "cabinetBed": "nightstand",
    "cabinetBedDrawer": "nightstand",
    "cabinetBedDrawerTable": "nightstand",
    "bookcaseClosed": "cabinet",
    "bookcaseClosedDoors": "cabinet",
    "bookcaseClosedWide": "cabinet",
    "bookcaseOpen": "cabinet",
    "bookcaseOpenLow": "cabinet",
    "cabinetTelevision": "cabinet",
    "cabinetTelevisionDoors": "cabinet",
    "bathroomCabinet": "cabinet",
    "bathroomCabinetDrawer": "cabinet",
    "kitchenCabinet": "cabinet",
    "kitchenCabinetDrawer": "cabinet",
    "kitchenCabinetCornerInner": "cabinet",
    "kitchenCabinetCornerRound": "cabinet",
    "kitchenCabinetUpper": "cabinet",
    "kitchenCabinetUpperCorner": "cabinet",
    "kitchenCabinetUpperDouble": "cabinet",
    "kitchenCabinetUpperLow": "cabinet",
    "lampRoundTable": "lamp",
    "lampSquareTable": "lamp",
    "lampRoundFloor": "floor_lamp",
    "lampSquareFloor": "floor_lamp",
    "televisionModern": "tv",
    "televisionVintage": "tv",
    "televisionAntenna": "tv",
    "computerScreen": "monitor",
    "laptop": "laptop",
    "computerKeyboard": "keyboard",
    "computerMouse": "mouse",
    "pillow": "pillow",
    "pillowBlue": "pillow",
    "pillowBlueLong": "pillow",
    "pillowLong": "pillow",
    "rugDoormat": "rug",
    "rugRectangle": "rug",
    "rugRound": "rug",
    "rugRounded": "rug",
    "rugSquare": "rug",
    "plantSmall1": "plant",
    "plantSmall2": "plant",
    "plantSmall3": "plant",
    "pottedPlant": "plant",
    "bench": "bench",
    "benchCushion": "bench",
    "benchCushionLow": "bench",
    "toilet": "toilet",
    "toiletSquare": "toilet",
    "bathtub": "bathtub",
    "shower": "shower",
    "showerRound": "shower",
    "bathroomSink": "sink",
    "bathroomSinkSquare": "sink",
    "kitchenSink": "sink",
    "bathroomMirror": "mirror",
    "kitchenFridge": "fridge",
    "kitchenFridgeBuiltIn": "fridge",
    "kitchenFridgeLarge": "fridge",
    "kitchenFridgeSmall": "fridge",
    "kitchenStove": "stove",
    "kitchenStoveElectric": "stove",
    "kitchenMicrowave": "microwave",
    "washer": "washer",
    "dryer": "washer",
    "washerDryerStacked": "washer",
    "trashcan": "trashcan",
    "cardboardBoxClosed": "box",
    "cardboardBoxOpen": "box",
    "books": "books",
    "speaker": "speaker",
    "speakerSmall": "speaker",
    "radio": "radio",
    "coatRack": "coat_rack",
    "coatRackStanding": "coat_rack",
    "bear": "toy",
    "kitchenBar": "counter",
    "kitchenBarEnd": "counter",
    "toaster": "appliance",
    "kitchenBlender": "appliance",
    "kitchenCoffeeMachine": "appliance",
}

_TRAILING_INDEX_RE = re.compile(r"[_\s-]+\d+$")


@dataclass
class AssetCandidate:
    source: str
    id: str
    path: str
    cls: str
    canonical_aabb_m: tuple[float, float, float]  # GLB axes, Y-up
    width_m: float  # canonical X extent
    depth_m: float  # canonical Z extent
    height_m: float  # canonical Y extent
    is_round: bool
    tier: str = "furniture"
    # Provenance of the canonical size (see module docstring).
    size_source: str = "glb_aabb"
    unit_factor: float = 1.0
    glb_aabb_native: tuple[float, float, float] | None = None
    axis_permutation: tuple[int, int, int] | None = None
    dimension_mismatch_rel: float = 0.0

    @property
    def footprint_aspect(self) -> float:
        return footprint_aspect(self.width_m, self.depth_m)


def footprint_aspect(a: float, b: float) -> float:
    """long/short of a footprint - orientation-free, >= 1."""
    lo, hi = sorted((abs(float(a)), abs(float(b))))
    return hi / max(lo, 1e-6)


@dataclass
class AssetIndex:
    classes: dict[str, list[AssetCandidate]] = field(default_factory=dict)
    unmapped: dict[str, list[str]] = field(default_factory=dict)
    sources: dict[str, dict] = field(default_factory=dict)
    assets_root: str = ""
    small_root: str | None = None

    def candidates(self, cls: str) -> list[AssetCandidate]:
        """All candidates of `cls` from the highest-priority source that has
        any - Poly Haven if it covers the class, else the next source down.
        Empty if no source covers it."""
        cands = self.classes.get(cls, [])
        for source in SOURCE_PRIORITY:
            picked = [c for c in cands if c.source == source]
            if picked:
                return picked
        return list(cands)

    def candidates_for_label(self, label: str, height_m: float | None = None) -> tuple[str | None, list[AssetCandidate]]:
        """Resolve a pipeline label through `LABEL_SYNONYMS` and return
        `(index_class, candidates)` for the first class with any candidate;
        `(None, [])` if the label is unknown or no class it maps to is covered.
        A "lamp" measured >= FLOOR_LAMP_MIN_HEIGHT_M tall tries `floor_lamp`
        first (Poly Haven has no floor lamp; Kenney does)."""
        for cls in resolve_label(label, height_m=height_m):
            cands = self.candidates(cls)
            if cands:
                return cls, cands
        return None, []

    def stats(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for cls in sorted(self.classes):
            counts: dict[str, int] = {}
            for c in self.classes[cls]:
                counts[c.source] = counts.get(c.source, 0) + 1
            out[cls] = counts
        return out

    def to_json(self) -> dict:
        return {
            "schema_version": INDEX_SCHEMA_VERSION,
            "assets_root": self.assets_root,
            "small_root": self.small_root,
            "unit_factors": UNIT_FACTORS,
            "sources": self.sources,
            "classes": {cls: [asdict(c) for c in cands] for cls, cands in self.classes.items()},
            "unmapped": self.unmapped,
        }

    @classmethod
    def from_json(cls, data: dict) -> "AssetIndex":
        index = cls(assets_root=data.get("assets_root", ""), small_root=data.get("small_root"), unmapped=data.get("unmapped", {}), sources=data.get("sources", {}))
        for name, cands in data.get("classes", {}).items():
            index.classes[name] = [_candidate_from_dict(c) for c in cands]
        return index


def _candidate_from_dict(d: dict) -> AssetCandidate:
    d = dict(d)
    d["canonical_aabb_m"] = tuple(d["canonical_aabb_m"])
    if d.get("glb_aabb_native") is not None:
        d["glb_aabb_native"] = tuple(d["glb_aabb_native"])
    if d.get("axis_permutation") is not None:
        d["axis_permutation"] = tuple(d["axis_permutation"])
    return AssetCandidate(**d)


def normalize_label(label: str) -> str:
    return _TRAILING_INDEX_RE.sub("", str(label).strip().lower().replace("_", " ")).strip()


def resolve_label(label: str, height_m: float | None = None) -> tuple[str, ...]:
    """Ordered index classes for a pipeline label: furniture `LABEL_SYNONYMS`
    first, then the small-object classes of `SMALL_LABEL_SYNONYMS` (so `laptop`
    tries Kenney's laptop before a GSO one, `glass` only reaches `cup`). Unknown
    labels fall back to their last word (e.g. "wooden chair" -> "chair") and
    finally to the label itself if it names an index class."""
    key = normalize_label(label)
    classes: tuple[str, ...] = LABEL_SYNONYMS.get(key, ())
    small = SMALL_LABEL_SYNONYMS.get(key, ())
    if not classes and not small and " " in key:
        tail = key.rsplit(" ", 1)[-1]
        classes = LABEL_SYNONYMS.get(tail, ())
        small = SMALL_LABEL_SYNONYMS.get(tail, ())
    classes = classes + tuple(c for c in small if c not in classes)
    if not classes:
        classes = (key,)
    if classes and classes[0] == "lamp" and height_m is not None and height_m >= FLOOR_LAMP_MIN_HEIGHT_M:
        classes = ("floor_lamp",) + classes
    return classes


def is_small_label(label: str) -> bool:
    """True when the detected label is in the small-object vocabulary, i.e. the
    only labels for which a `tier="small"` candidate may ever be placed."""
    key = normalize_label(label)
    if key in SMALL_LABEL_SYNONYMS:
        return True
    return " " in key and key.rsplit(" ", 1)[-1] in SMALL_LABEL_SYNONYMS


# --- building -----------------------------------------------------------------


def _glb_aabb_extents(path: Path) -> tuple[float, float, float]:
    import trimesh

    scene = trimesh.load(path.as_posix(), force="scene")
    return tuple(float(v) for v in scene.extents)


def canonical_from_api_dimensions(api_dims_mm, glb_aabb) -> tuple[tuple[float, float, float], tuple[int, int, int], float]:
    """Map the API's (mm) real-world triple onto the GLB's axes by sorted rank.
    Returns (canonical_aabb_m on GLB axes, axis_permutation, worst relative
    mismatch vs the GLB's own AABB)."""
    api_m = np.asarray(api_dims_mm, dtype=float) / 1000.0
    aabb = np.asarray(glb_aabb, dtype=float)
    api_order = np.argsort(api_m)  # ascending api indices
    aabb_rank = np.argsort(np.argsort(aabb))  # rank of each GLB axis
    perm = tuple(int(api_order[r]) for r in aabb_rank)
    canonical = tuple(float(api_m[p]) for p in perm)
    mismatch = float(np.max(np.abs(np.asarray(canonical) - aabb) / np.maximum(aabb, 1e-6)))
    return canonical, perm, mismatch


def _polyhaven_class(class_hint: str) -> str | None:
    if class_hint in POLYHAVEN_CLASS_HINTS:
        return POLYHAVEN_CLASS_HINTS[class_hint]
    head = re.split(r"[/(]", class_hint.strip().lower())[0].strip()
    return head or None


def _index_polyhaven(entries: dict, index: AssetIndex, assets_root: Path) -> None:
    unmapped = []
    for asset_id, entry in entries.items():
        cls = _polyhaven_class(entry.get("class_hint", ""))
        glb_path = Path(entry.get("glb_path") or (assets_root / "polyhaven" / asset_id / f"{asset_id}.glb"))
        if cls is None or not glb_path.exists():
            unmapped.append(asset_id)
            continue
        aabb = entry.get("aabb_extents_m")
        if not aabb:
            aabb = _glb_aabb_extents(glb_path)
        api_dims = entry.get("api_dimensions")
        if api_dims and len(api_dims) == 3 and all(float(d) > 0 for d in api_dims):
            canonical, perm, mismatch = canonical_from_api_dimensions(api_dims, aabb)
            size_source = "polyhaven_api_dimensions_mm"
        else:
            canonical, perm, mismatch = tuple(float(v) for v in aabb), None, 0.0
            size_source = "glb_aabb"
        index.classes.setdefault(cls, []).append(
            AssetCandidate(
                source="polyhaven",
                id=asset_id,
                path=glb_path.as_posix(),
                cls=cls,
                canonical_aabb_m=canonical,
                width_m=canonical[0],
                depth_m=canonical[2],
                height_m=canonical[1],
                is_round=cls in ROUND_CLASSES,
                size_source=size_source,
                unit_factor=1.0,
                glb_aabb_native=tuple(float(v) for v in aabb),
                axis_permutation=perm,
                dimension_mismatch_rel=mismatch,
            )
        )
    index.unmapped["polyhaven"] = unmapped
    index.sources["polyhaven"] = {"obtained": True, "n_models": len(entries), "n_indexed": len(entries) - len(unmapped)}


def _index_kenney(entry: dict, index: AssetIndex, assets_root: Path) -> None:
    kit_dir = assets_root / "kenney_furniture_kit"
    glbs: list[Path] = []
    for f in entry.get("model_files", []):
        if str(f.get("format", "")).upper() in ("GLB", "GLTF") and str(f.get("path", "")).lower().endswith(".glb"):
            glbs.append(kit_dir / f["path"])
    if not glbs and kit_dir.exists():
        glbs = sorted(kit_dir.rglob("*.glb"))
    unmapped = []
    n_indexed = 0
    for glb_path in glbs:
        if not glb_path.exists():
            continue
        stem = glb_path.stem
        cls = KENNEY_CLASSES.get(stem)
        if cls is None:
            unmapped.append(stem)
            continue
        native = _glb_aabb_extents(glb_path)
        canonical = tuple(float(v) * KENNEY_UNIT_TO_M for v in native)
        index.classes.setdefault(cls, []).append(
            AssetCandidate(
                source="kenney",
                id=stem,
                path=glb_path.as_posix(),
                cls=cls,
                canonical_aabb_m=canonical,
                width_m=canonical[0],
                depth_m=canonical[2],
                height_m=canonical[1],
                is_round=cls in ROUND_CLASSES,
                size_source="glb_aabb_x_unit_factor",
                unit_factor=KENNEY_UNIT_TO_M,
                glb_aabb_native=native,
            )
        )
        n_indexed += 1
    index.unmapped["kenney"] = unmapped
    index.sources["kenney"] = {"obtained": bool(entry.get("obtained", bool(glbs))), "n_models": len(glbs), "n_indexed": n_indexed}


def _index_small(inventory: dict, index: AssetIndex, small_root: Path) -> None:
    """T16b: Google Scanned Objects from `<small_root>/INVENTORY.json`
    (`models: {name: {class, glb_path, aabb_extents_m}}`). Canonical size is the
    converted GLB's AABB (metres, Y-up). Models whose class is not in
    `SMALL_CLASSES` or whose GLB is missing are reported under `unmapped`."""
    unmapped = []
    n_indexed = 0
    models = inventory.get("models", {}) or {}
    for name, entry in models.items():
        cls = str(entry.get("class", "")).strip().lower()
        glb_path = Path(entry.get("glb_path") or (small_root / "gso" / name / f"{name}.glb"))
        if cls not in SMALL_CLASSES or not glb_path.exists():
            unmapped.append(name)
            continue
        aabb = entry.get("aabb_extents_m") or _glb_aabb_extents(glb_path)
        canonical = tuple(float(v) for v in aabb)
        index.classes.setdefault(cls, []).append(
            AssetCandidate(
                source="gso",
                id=name,
                path=glb_path.as_posix(),
                cls=cls,
                canonical_aabb_m=canonical,
                width_m=canonical[0],
                depth_m=canonical[2],
                height_m=canonical[1],
                is_round=cls in ROUND_CLASSES,
                tier=TIER_SMALL,
                size_source="glb_aabb",
                unit_factor=1.0,
                glb_aabb_native=canonical,
            )
        )
        n_indexed += 1
    index.unmapped["gso"] = unmapped
    index.sources["gso"] = {"obtained": True, "n_models": len(models), "n_indexed": n_indexed, "license": inventory.get("license", "CC BY 4.0"), "root": small_root.as_posix()}


def build_index(
    assets_root: str | Path = DEFAULT_ASSETS_ROOT, *, small_root: str | Path | None = None, use_cache: bool = True
) -> AssetIndex:
    """Build (or load from `index.json`) the asset index for `assets_root`,
    plus the `tier="small"` GSO candidates from `small_root` when given and
    present (a missing small root is not an error - the small tier is optional).
    The cache is regenerated when either `INVENTORY.json` is newer than it, the
    small root differs, or it was written by a different schema version."""
    assets_root = Path(assets_root)
    inventory_path = assets_root / "INVENTORY.json"
    cache_path = assets_root / INDEX_FILENAME
    if not inventory_path.exists():
        raise FileNotFoundError(f"no INVENTORY.json under {assets_root}")
    small_root = Path(small_root) if small_root is not None else None
    small_inventory_path = small_root / "INVENTORY.json" if small_root is not None else None
    if small_inventory_path is not None and not small_inventory_path.exists():
        small_root, small_inventory_path = None, None
    small_key = small_root.as_posix() if small_root is not None else None

    if use_cache and cache_path.exists():
        newest_input = max([inventory_path.stat().st_mtime] + ([small_inventory_path.stat().st_mtime] if small_inventory_path else []))
        if cache_path.stat().st_mtime >= newest_input:
            try:
                data = json.loads(cache_path.read_text())
                if (
                    data.get("schema_version") == INDEX_SCHEMA_VERSION
                    and data.get("assets_root") == assets_root.as_posix()
                    and data.get("small_root") == small_key
                ):
                    return AssetIndex.from_json(data)
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

    inventory = json.loads(inventory_path.read_text())
    index = AssetIndex(assets_root=assets_root.as_posix(), small_root=small_key)
    _index_polyhaven(inventory.get("polyhaven", {}) or {}, index, assets_root)
    kenney = inventory.get("kenney_furniture_kit", {}) or {}
    if kenney.get("obtained", True) or (assets_root / "kenney_furniture_kit").exists():
        _index_kenney(kenney, index, assets_root)
    mastjie = inventory.get("mastjie_household_goods", {}) or {}
    index.sources["mastjie"] = {"obtained": bool(mastjie.get("obtained", False)), "n_models": int(mastjie.get("count", 0) or 0), "n_indexed": 0}
    if small_inventory_path is not None:
        _index_small(json.loads(small_inventory_path.read_text()), index, small_root)
    for cands in index.classes.values():
        cands.sort(key=lambda c: (SOURCE_PRIORITY.index(c.source) if c.source in SOURCE_PRIORITY else 99, c.id))

    if use_cache:
        try:
            cache_path.write_text(json.dumps(index.to_json(), indent=1))
        except OSError:
            pass  # read-only assets root: the index still works in-memory
    return index


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build/print the furniture asset index.")
    parser.add_argument("--assets-root", type=Path, default=DEFAULT_ASSETS_ROOT)
    parser.add_argument("--small-root", type=Path, default=DEFAULT_SMALL_ROOT, help="T16b small-object (GSO) root; skipped when it has no INVENTORY.json")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    index = build_index(args.assets_root, small_root=args.small_root, use_cache=not args.rebuild)
    if args.rebuild:
        (args.assets_root / INDEX_FILENAME).write_text(json.dumps(index.to_json(), indent=1))
    print(json.dumps({"sources": index.sources, "classes": index.stats(), "unmapped": index.unmapped}, indent=2))


if __name__ == "__main__":
    main()
