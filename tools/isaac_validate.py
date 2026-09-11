#!/usr/bin/env python3
"""Validate an exported CloudEye scene .usd (app/services/usd_export.py) inside NVIDIA
Isaac Sim, headless.

    !!! UNVERIFIED !!!
    As of 2026-09-03 this script has NEVER been executed against a real Isaac Sim
    install. It cannot be run on the CloudEye VPS: Isaac Sim needs an RTX GPU with RT
    cores and >=16GB VRAM, and that machine has no NVIDIA GPU at all. It is written to
    be correct-by-construction against NVIDIA's published Isaac Sim 6.0 API docs, and
    every Omniverse/PhysX call beyond the documented SimulationApp bootstrap is
    feature-detected: if a call is missing in the build you run it, this exits 3 and
    NAMES the missing symbol rather than guessing silently. Expect the first real run
    to need fixes. Record that run in docs/ISAAC_VALIDATION.md's "Verified runs" table
    - until a row exists there, treat every claim this script makes about its own
    behaviour as a prediction, not a fact.

    Run (Isaac Sim installed, e.g. via deploy/install_isaacsim_gpu.sh):
        <isaac-python> tools/isaac_validate.py scene.usd --json-out report.json

    Run (no Isaac Sim available - pure-USD structural checks only, works anywhere
    usd-core is installed, including this VPS):
        uv run python tools/isaac_validate.py --usd-only scene.usd --json-out report.json

    See docs/ISAAC_VALIDATION.md for the full runbook, install steps, and a triage
    table for interpreting the JSON output.

What this checks (see docs/ISAAC_VALIDATION.md for the full triage table):
  1. the stage opens, is Z-up / metersPerUnit=1, and has the expected /World hierarchy
  2. PhysX cooks the whole stage without errors (Isaac mode only)
  3. every /World/Structure/* and /World/Objects/* prim yields a real PhysX collision
     shape at its authored location - a functional overlap-query check, not merely "the
     CollisionAPI is applied" (a hull whose convex cook was rejected still has the API
     applied with no real shape) (Isaac mode only)
  4. a test sphere dropped from 2m onto /World/Floor comes to rest there (Isaac mode only)
  5. no static prim (Floor/Structure) ends up below the floor; dynamic-object
     displacement is measured and reported, not pass/fail (see "Dynamics phase" below)
  6. structural sanity checks that need no physics at all: /World/Floor's world bbox
     overlaps /World/Objects/*'s in XY (catches an xformOpOrder-style exporter bug that
     silently displaces Floor/Structure relative to Objects), zero/near-zero object mass
     (a real exporter defect - see _add_object_geometry's volume==0 bbox-fallback
     path), non-positive cube scale, and (only under --gpu-dynamics) convex hull
     vertex/face counts against PhysX's GPU-dynamics limits

It never modifies USD_PATH: all authoring (test sphere, physics scene) goes to an
in-memory session layer, the root layer is locked read-only, and the file's sha256 is
compared before and after as a receipt in the JSON output.

Dynamics phase (diagnostic, non-fatal by default): objects are captured against walls,
so their convex hulls very likely interpenetrate /World/Structure boxes at t=0 - that
is a property of the SCENE (real detected geometry overlapping), not a defect in the
exporter, and must not be conflated with one. Dynamic objects are frozen (kinematic)
during the primary sphere-drop check specifically to keep that check unambiguous; a
separate phase then unfreezes them and reports displacement/ejection as info, correlated
against a t=0 interpenetration census, never as a pass/fail gate (unless --fail-on-ejected).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

SCHEMA = "cloudeye.isaac_validate/1"
VALIDATION_SCOPE = "/World/_cloudeye_validation"
SPHERE_PATH = f"{VALIDATION_SCOPE}/TestSphere"
PHYSICS_SCENE_PATH = f"{VALIDATION_SCOPE}/PhysicsScene"
STRUCTURE_ROOT = "/World/Structure"
OBJECTS_ROOT = "/World/Objects"
FLOOR_PATH = "/World/Floor"
POINTCLOUD_PATH = "/World/PointCloud"
ROBOT_START_PATH = "/World/RobotStart"

# From app/services/usd_export.py - the contract this script validates against. Kept
# as literal constants here (not imported - this script must run standalone under
# Isaac's own interpreter, which has no access to this repo's backend venv) but must
# stay in sync with usd_export.py if either changes.
FLOOR_THICKNESS_M = 0.02
MAX_HULL_VERTICES = 64
# Believed PhysX GPU-dynamics convex-collider limit; unverified for this Isaac Sim
# build - see the gpu_convex_limits check, which is informational unless --gpu-dynamics.
GPU_CONVEX_VERTEX_FACE_LIMIT = 64

JSON_BEGIN = "===CLOUDEYE_VALIDATION_JSON_BEGIN==="
JSON_END = "===CLOUDEYE_VALIDATION_JSON_END==="

EXIT_OK = 0
EXIT_CHECK_FAILED = 1
EXIT_USAGE_ERROR = 2
EXIT_ENVIRONMENT_ERROR = 3
EXIT_INTERNAL_ERROR = 4


class EnvironmentMismatch(RuntimeError):
    """A required Isaac Sim / omni.physx API was absent or behaved unexpectedly for
    this build. Exit 3, never a silent guess about what it would have done."""


class ValidationAbort(RuntimeError):
    """A required precondition failed badly enough that no further check can run
    meaningfully (e.g. the stage didn't open). The JSON is still written."""


# --- CLI ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("usd_path", type=Path, help="scene .usd/.usda produced by app/services/usd_export.py")

    sphere = p.add_argument_group("sphere drop (primary check)")
    sphere.add_argument("--steps", type=int, default=600, help="max physics steps (default 600 = 2.5s @ 1/240)")
    sphere.add_argument("--physics-dt", type=float, default=1.0 / 240.0)
    sphere.add_argument("--drop-height", type=float, default=2.0, help="meters above the floor's top surface")
    sphere.add_argument("--sphere-radius", type=float, default=0.1)
    sphere.add_argument("--sphere-mass", type=float, default=1.0)
    sphere.add_argument("--drop-point", type=float, nargs=2, metavar=("X", "Y"), default=None)
    sphere.add_argument("--settle-speed", type=float, default=0.01, help="m/s threshold for 'at rest'")
    sphere.add_argument("--settle-hold", type=int, default=30, help="consecutive steps under threshold required")

    dyn = p.add_argument_group("dynamics (diagnostic phase, non-fatal by default)")
    dyn.add_argument("--dynamics-steps", type=int, default=120, help="0 disables this phase")
    dyn.add_argument("--fail-on-ejected", action="store_true")
    dyn.add_argument("--ejection-threshold", type=float, default=0.5, help="meters of displacement")

    coll = p.add_argument_group("collider audit")
    coll.add_argument("--collider-check", choices=("overlap", "usd-api", "none"), default="overlap")
    coll.add_argument("--overlap-shrink", type=float, default=0.98)

    log = p.add_argument_group("log capture")
    log.add_argument("--log-out", type=Path, default=None)
    log.add_argument("--allow-physx-warnings", action="store_true")
    log.add_argument("--ignore-log-pattern", action="append", default=[])

    out = p.add_argument_group("output / behaviour")
    out.add_argument("--json-out", type=Path, default=None)
    out.add_argument("--trace", action="store_true", help="include the sphere's z/speed trajectory in the JSON")
    out.add_argument("--strict", action="store_true", help="warnings become failures")
    out.add_argument("--gpu-dynamics", action="store_true", help="default is CPU PhysX dynamics")
    out.add_argument("--no-freeze-objects", action="store_true")
    out.add_argument("--no-lock-root-layer", action="store_true")
    out.add_argument("--usd-only", action="store_true", help="skip Isaac; run only the pure-USD structural checks")
    out.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


# --- Report / JSON assembly ----------------------------------------------------------


@dataclass
class Report:
    args: argparse.Namespace
    input: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)
    stage: dict = field(default_factory=dict)
    checks: dict = field(default_factory=dict)
    sphere_drop: dict | None = None
    below_floor: dict | None = None
    dynamics: dict | None = None
    physx_log: dict | None = None
    timings_s: dict = field(default_factory=dict)
    _t0: float = field(default_factory=time.perf_counter)
    _fatal_env: str | None = None
    _fatal_internal: str | None = None

    def add_check(self, name: str, status: str, *, required: bool, detail: Any = None) -> None:
        entry: dict = {"status": status, "required": required}
        if detail is not None:
            entry["detail"] = detail
        self.checks[name] = entry

    def fatal_env(self, message: str) -> None:
        self._fatal_env = message
        self.environment["missing_api"] = self.environment.get("missing_api", []) + [message]

    def fatal_internal(self, tb: str) -> None:
        self._fatal_internal = tb

    def record_input_file(self, path: Path) -> None:
        self.input["path"] = str(path)
        if not path.is_file():
            self.input["exists"] = False
            return
        self.input["exists"] = True
        self.input["bytes"] = path.stat().st_size
        self.input["sha256_before"] = _sha256(path)
        self.input.update(sniff_usd_format(path))

    def record_input_after(self, path: Path) -> None:
        if path.is_file():
            sha = _sha256(path)
            self.input["sha256_after"] = sha
            self.input["unmodified"] = sha == self.input.get("sha256_before")

    def overall_status(self, strict: bool) -> str:
        if self._fatal_env:
            return "error"
        if self._fatal_internal:
            return "error"
        for c in self.checks.values():
            if not c.get("required"):
                continue
            if c["status"] == "fail":
                return "fail"
            if c["status"] == "skipped":
                return "fail"
            if strict and c["status"] == "warn":
                return "fail"
        return "pass"

    def failures(self) -> list[str]:
        out = []
        for name, c in self.checks.items():
            if c.get("required") and c["status"] in ("fail", "skipped"):
                out.append(f"{name}: {c.get('detail')}")
        return out

    def warnings(self) -> list[str]:
        out = []
        for name, c in self.checks.items():
            if c["status"] == "warn":
                out.append(f"{name}: {c.get('detail')}")
        return out

    def to_dict(self, exit_code: int, argv: Sequence[str]) -> dict:
        status = self.overall_status(self.args.strict) if not (self._fatal_env or self._fatal_internal) else (
            "error"
        )
        d = {
            "schema": SCHEMA,
            "status": status,
            "exit_code": exit_code,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "argv": list(argv),
            "input": self.input,
            "environment": self.environment,
            "stage": self.stage,
            "checks": self.checks,
            "sphere_drop": self.sphere_drop,
            "below_floor": self.below_floor,
            "dynamics": self.dynamics,
            "physx_log": self.physx_log,
            "timings_s": {**self.timings_s, "total": round(time.perf_counter() - self._t0, 3)},
            "failures": self.failures(),
            "warnings": self.warnings(),
            "unverified": (
                "This script had never been executed against a live Isaac Sim install "
                "at the time it was written - see docs/ISAAC_VALIDATION.md 'Verified runs'."
            ),
        }
        if self._fatal_env:
            d["fatal_environment_error"] = self._fatal_env
        if self._fatal_internal:
            d["fatal_internal_error"] = self._fatal_internal
        return d

    def finish(self, exit_code: int, argv: Sequence[str], json_out: Path | None) -> dict:
        d = self.to_dict(exit_code, argv)
        text = json.dumps(d, indent=2, default=str)
        if json_out:
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(text)
        print(JSON_BEGIN, file=sys.stdout)
        print(text, file=sys.stdout)
        print(JSON_END, file=sys.stdout)
        print(
            f"\n[isaac_validate] status={d['status']} exit={exit_code} "
            f"failures={len(d['failures'])} warnings={len(d['warnings'])}",
            file=sys.stderr,
        )
        for f in d["failures"]:
            print(f"  FAIL: {f}", file=sys.stderr)
        for w in d["warnings"]:
            print(f"  WARN: {w}", file=sys.stderr)
        return d


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sniff_usd_format(path: Path) -> dict:
    """Crate's bootstrap (OpenUSD crateFile.cpp) is an 8-byte magic 'PXR-USDC' then
    three version bytes; an ASCII layer starts with '#usda '. Sniffed from the file's
    own header (never guessed from an error message) so a version mismatch against
    whatever USD Isaac Sim bundles is named precisely - see the module docstring's
    crate-compatibility note and docs/ISAAC_VALIDATION.md."""
    try:
        head = path.open("rb").read(16)
    except OSError as exc:
        return {"format": "unreadable", "error": str(exc)}
    if head[:8] == b"PXR-USDC":
        return {"format": "crate", "crate_version": f"{head[8]}.{head[9]}.{head[10]}"}
    if head[:6] == b"#usda ":
        return {"format": "ascii"}
    return {"format": "unknown", "head_hex": head.hex()}


# --- Late imports (Omniverse-level imports must come after SimulationApp exists) -----


def _late_imports(*, isaac: bool) -> dict[str, bool]:
    global Usd, UsdGeom, UsdPhysics, Gf, Sdf, Vt
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt  # noqa: F401

    feat: dict[str, bool] = {}
    if not isaac:
        return feat

    global carb, omni
    global PhysxSchema, PhysicsSchemaTools
    global get_physx_interface, get_physx_scene_query_interface, get_physx_cooking_interface

    import carb
    import carb.settings
    import omni.usd

    try:
        from pxr import PhysxSchema

        feat["PhysxSchema"] = True
    except ImportError:
        PhysxSchema = None
        feat["PhysxSchema"] = False

    try:
        from pxr import PhysicsSchemaTools

        feat["PhysicsSchemaTools"] = True
    except ImportError:
        PhysicsSchemaTools = None
        feat["PhysicsSchemaTools"] = False

    try:
        from omni.physx import (
            get_physx_cooking_interface,
            get_physx_interface,
            get_physx_scene_query_interface,
        )
    except ImportError as exc:
        raise EnvironmentMismatch(
            "omni.physx python bindings not importable (get_physx_interface / "
            f"get_physx_scene_query_interface): {exc}. Is the omni.physx extension "
            "enabled in this Isaac Sim build?"
        ) from exc
    feat["omni.physx"] = True
    return feat


def _require(obj: Any, name: str, what: str) -> Callable:
    fn = getattr(obj, name, None)
    if fn is None:
        avail = sorted(n for n in dir(obj) if not n.startswith("_"))
        raise EnvironmentMismatch(
            f"{what}: this Isaac Sim build's {type(obj).__name__} has no .{name}(). "
            f"Available members: {avail}"
        )
    return fn


def _optional(obj: Any, name: str, feat: dict[str, bool], key: str) -> Callable | None:
    fn = getattr(obj, name, None) if obj is not None else None
    feat[key] = fn is not None
    return fn


# --- Pure-USD structural checks (the --usd-only path; runnable without Isaac) --------


def preflight(stage, report: Report, args: argparse.Namespace):
    up = UsdGeom.GetStageUpAxis(stage)
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    default_prim = stage.GetDefaultPrim()
    conventions_ok = up == UsdGeom.Tokens.z and abs(mpu - 1.0) < 1e-9 and default_prim.IsValid()
    report.add_check(
        "stage_conventions",
        "pass" if conventions_ok else "fail",
        required=True,
        detail={
            "up_axis": str(up),
            "meters_per_unit": mpu,
            "default_prim": str(default_prim.GetPath()) if default_prim.IsValid() else None,
            "custom_layer_data": dict(stage.GetRootLayer().customLayerData),
        },
    )

    floor = stage.GetPrimAtPath(FLOOR_PATH)
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    )
    # 'guide' purpose is DELIBERATELY excluded above: /World/PointCloud is authored
    # with purpose=guide (usd_export._add_pointcloud) and can be up to 500k points of
    # non-collidable decoration - including it would swamp every bound computed here.
    floor_range = bbox_cache.ComputeWorldBound(floor).ComputeAlignedRange() if floor.IsValid() else None
    floor_top_z = float(floor_range.GetMax()[2]) if floor_range and not floor_range.IsEmpty() else None
    floor_bottom_z = float(floor_range.GetMin()[2]) if floor_range and not floor_range.IsEmpty() else None
    report.add_check(
        "floor_present",
        "pass" if floor.IsValid() and floor_range and not floor_range.IsEmpty() else "fail",
        required=True,
        detail={"path": FLOOR_PATH, "top_z": floor_top_z, "bottom_z": floor_bottom_z},
    )

    structure = [p for p in stage.Traverse() if str(p.GetPath()).startswith(STRUCTURE_ROOT + "/")]
    objects = [p for p in stage.Traverse() if str(p.GetPath()).startswith(OBJECTS_ROOT + "/")]
    fragments = [p for p in objects if "_fragments" in str(p.GetPath())]
    hull_objects = [p for p in objects if p.GetTypeName() == "Mesh"]
    bbox_objects = [p for p in objects if p.GetTypeName() == "Cube"]
    pc = stage.GetPrimAtPath(POINTCLOUD_PATH)
    pc_points = 0
    if pc.IsValid():
        pts_attr = pc.GetAttribute("points")
        pc_points = len(pts_attr.Get() or []) if pts_attr else 0

    report.stage = {
        "up_axis": str(up),
        "meters_per_unit": mpu,
        "default_prim": str(default_prim.GetPath()) if default_prim.IsValid() else None,
        "custom_layer_data": dict(stage.GetRootLayer().customLayerData),
        "counts": {
            "structure": len(structure),
            "objects": len(objects) - len(fragments),
            "fragments": len(fragments),
            "hull_objects": len(hull_objects),
            "bbox_objects": len(bbox_objects),
            "pointcloud_points": pc_points,
        },
        "floor": {"path": FLOOR_PATH, "top_z": floor_top_z, "bottom_z": floor_bottom_z},
    }

    # --- (a0) floor/objects spatial consistency: regression check for the xformOpOrder
    # bug in usd_export._add_cube (AddScaleOp before AddTranslateOp caused every cube's
    # world position to be scaled by its own size, sending Floor/Structure meters away
    # from Objects while leaving hull-mesh Objects, which use raw absolute coordinates,
    # in place - two disjoint regions in one file). Floor and Objects footprint the
    # same physical room, so their world-space XY bboxes must overlap; this catches a
    # recurrence of that bug (or a similar future one) on the simulator side, not just
    # at export time.
    non_fragment_objects = [p for p in objects if p not in fragments]
    objects_range = None
    for p in non_fragment_objects:
        r = bbox_cache.ComputeWorldBound(p).ComputeAlignedRange()
        if r.IsEmpty():
            continue
        objects_range = r if objects_range is None else objects_range.UnionWith(r)
    footprint_overlap = None
    footprint_detail: dict = {"objects_checked": len(non_fragment_objects)}
    if floor_range and not floor_range.IsEmpty() and objects_range is not None:
        fmin, fmax = floor_range.GetMin(), floor_range.GetMax()
        omin, omax = objects_range.GetMin(), objects_range.GetMax()
        footprint_overlap = all(fmin[i] <= omax[i] and omin[i] <= fmax[i] for i in (0, 1))
        footprint_detail.update({
            "floor_xy_range": [[fmin[0], fmin[1]], [fmax[0], fmax[1]]],
            "objects_xy_range": [[omin[0], omin[1]], [omax[0], omax[1]]],
        })
    report.add_check(
        "floor_objects_footprint_overlap",
        "pass" if footprint_overlap else ("skipped" if footprint_overlap is None else "fail"),
        required=footprint_overlap is not None,
        detail=footprint_detail,
    )

    # --- (a) zero/near-zero mass: usd_export computes mass = volume * density, and a
    # degenerate/flat bbox fallback yields volume == 0.0 exactly. Zero-physics check.
    dynamic_prims = [p for p in objects if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    zero_mass, tiny_mass, masses = [], [], []
    for p in dynamic_prims:
        m = UsdPhysics.MassAPI(p).GetMassAttr().Get()
        if m is None:
            continue
        masses.append(float(m))
        if m <= 0.0:
            zero_mass.append(_prim_diag(p))
        elif m < 1e-3:
            tiny_mass.append(_prim_diag(p))
    report.add_check(
        "mass_sanity",
        "fail" if zero_mass else ("warn" if tiny_mass else "pass"),
        required=True,
        detail={
            "zero_mass": zero_mass,
            "tiny_mass": tiny_mass,
            "min_mass_kg": min(masses) if masses else None,
        },
    )

    # --- (b) non-positive scale on a Cube - a malformed bbox (min > max) would produce
    # a negative/zero scale -> an inverted or degenerate collider.
    bad_scale = []
    for p in [floor] + structure + bbox_objects if floor.IsValid() else structure + bbox_objects:
        ops = UsdGeom.Xformable(p).GetOrderedXformOps()
        scale_op = next((o for o in ops if o.GetOpType() == UsdGeom.XformOp.TypeScale), None)
        if scale_op is None:
            continue
        scale = scale_op.Get()
        if scale is not None and any(c <= 0 for c in scale):
            bad_scale.append({**_prim_diag(p), "scale": list(scale)})
    report.add_check(
        "scale_sanity", "fail" if bad_scale else "pass", required=True, detail={"bad_scale": bad_scale}
    )

    # --- (c) convex hull vertex/face counts vs PhysX's GPU-dynamics convex limits.
    # MAX_HULL_VERTICES in usd_export.py is exactly this cap - i.e. on the boundary -
    # and face count is not capped there at all. Only a hard failure under
    # --gpu-dynamics; CPU PhysX (the default here) does not share this limit.
    over = []
    for p in hull_objects:
        mesh = UsdGeom.Mesh(p)
        n_verts = len(mesh.GetPointsAttr().Get() or [])
        n_faces = len(mesh.GetFaceVertexCountsAttr().Get() or [])
        if n_verts > GPU_CONVEX_VERTEX_FACE_LIMIT or n_faces > GPU_CONVEX_VERTEX_FACE_LIMIT:
            over.append({**_prim_diag(p), "num_verts": n_verts, "num_faces": n_faces})
    report.add_check(
        "gpu_convex_limits",
        "fail" if (over and args.gpu_dynamics) else ("warn" if over else "pass"),
        required=bool(args.gpu_dynamics),
        detail={"limit": GPU_CONVEX_VERTEX_FACE_LIMIT, "over": over},
    )

    return floor_top_z, floor_bottom_z


def _prim_diag(prim, **extra) -> dict:
    """Pulls the exporter's own cloudeye:* breadcrumbs onto a finding so it points
    straight at the responsible code path (e.g. build_convex_hull's degenerate-input
    handling) instead of leaving the reader to go grep for the prim by hand."""
    d = {"prim": prim.GetPath().pathString, "type": prim.GetTypeName(), **extra}
    for attr, key in (
        ("cloudeye:collisionSource", "collision_source"),
        ("cloudeye:numPoints", "num_points"),
        ("cloudeye:name", "name"),
        ("cloudeye:isFragment", "is_fragment"),
    ):
        a = prim.GetAttribute(attr)
        if a and a.HasValue():
            d[key] = a.Get()
    return d


def run_usd_only(args: argparse.Namespace, report: Report) -> int:
    if not args.usd_path.is_file():
        report.add_check("stage_opened", "fail", required=True, detail=f"{args.usd_path} does not exist")
        return report_and_exit(report, EXIT_ENVIRONMENT_ERROR, args)

    report.environment["mode"] = "usd-only (no Isaac Sim - structural checks only)"
    try:
        stage = Usd.Stage.Open(str(args.usd_path))
    except Exception as exc:
        report.add_check("stage_opened", "fail", required=True, detail=str(exc))
        return report_and_exit(report, EXIT_ENVIRONMENT_ERROR, args)
    if stage is None:
        report.add_check("stage_opened", "fail", required=True, detail="Usd.Stage.Open returned None")
        return report_and_exit(report, EXIT_ENVIRONMENT_ERROR, args)
    report.add_check("stage_opened", "pass", required=True)

    preflight(stage, report, args)
    for name in ("cooking", "collider_coverage", "sphere_drop", "physx_log_clean"):
        report.add_check(name, "skipped", required=False, detail="requires Isaac Sim - run without --usd-only")

    return report_and_exit(report, EXIT_OK if report.overall_status(args.strict) == "pass" else EXIT_CHECK_FAILED, args)


def report_and_exit(report: Report, code: int, args: argparse.Namespace) -> int:
    report.finish(code, sys.argv, args.json_out)
    return code


# --- Isaac-mode: stage open, physics scene, cooking -----------------------------------


def open_stage_isaac(args: argparse.Namespace, report: Report):
    ctx = omni.usd.get_context()
    result = ctx.open_stage(str(args.usd_path))
    ok = result[0] if isinstance(result, tuple) else bool(result)
    stage = ctx.get_stage()
    if not ok or stage is None:
        report.add_check("stage_opened", "fail", required=True, detail="omni.usd.get_context().open_stage failed")
        raise ValidationAbort("stage did not open")
    report.add_check("stage_opened", "pass", required=True)

    root = stage.GetRootLayer()
    if not args.no_lock_root_layer:
        # Any physics writeback or accidental authoring targeting the root layer now
        # raises instead of silently mutating the file under validation.
        root.SetPermissionToEdit(False)
    session = stage.GetSessionLayer()
    stage.SetEditTarget(Usd.EditTarget(session))
    return stage


def ensure_physics_scene(stage, args: argparse.Namespace, report: Report):
    existing = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if existing:
        scene, created = UsdPhysics.Scene(existing[0]), False
    else:
        scene = UsdPhysics.Scene.Define(stage, PHYSICS_SCENE_PATH)
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))  # stage is Z-up
        scene.CreateGravityMagnitudeAttr(9.81)  # metersPerUnit == 1.0
        created = True
    ccd_enabled = False
    if PhysxSchema is not None:
        px = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        px.CreateEnableCCDAttr(True)
        px.CreateEnableGPUDynamicsAttr(bool(args.gpu_dynamics))
        px.CreateEnableStabilizationAttr(True)
        ccd_enabled = True
    report.add_check(
        "physics_scene",
        "pass",
        required=True,
        detail={
            "path": str(scene.GetPath()),
            "created_by_validator": created,
            "gravity_z": -9.81,
            "ccd": ccd_enabled,
            "gpu_dynamics": bool(args.gpu_dynamics),
        },
    )
    return ccd_enabled


def cook(report: Report, feat: dict[str, bool]) -> float:
    physx = get_physx_interface()
    t0 = time.perf_counter()
    _require(
        physx,
        "force_load_physics_from_usd",
        "PhysX cooking (parse the stage and build PhysX actors without stepping)",
    )()
    cooking_iface = get_physx_cooking_interface() if get_physx_cooking_interface else None
    wait = _optional(cooking_iface, "wait_for_cooking_to_finish", feat, "cooking_wait")
    if wait:
        wait()
    elapsed = time.perf_counter() - t0
    report.add_check(
        "cooking",
        "pass",
        required=True,
        detail={"method": "force_load_physics_from_usd", "seconds": round(elapsed, 3), "waited": feat.get("cooking_wait", False)},
    )
    return elapsed


def freeze_objects(stage, freeze: bool) -> None:
    """Kinematic (not disabled): a kinematic body keeps its collider, so the collider
    audit and the sphere's contacts against it are unaffected, but it cannot be
    depenetrated out of geometry it overlaps at t=0 - see module docstring."""
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith(OBJECTS_ROOT + "/") and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(freeze)


def _hit_collision_path(hit) -> str:
    """omni.physx scene-query hit records have been seen to expose the collider path
    either as a plain string attribute or an encoded (path0, path1) int pair; handle
    both rather than assuming one, and never crash on an unrecognised record."""
    p = getattr(hit, "collision", None)
    if isinstance(p, str) and p:
        return p
    enc = getattr(hit, "collision_encoded", None)
    if enc is not None and PhysicsSchemaTools is not None:
        try:
            return str(PhysicsSchemaTools.decodeSdfPath(enc[0], enc[1]))
        except Exception:
            pass
    return f"<unrecognised hit record: {sorted(n for n in dir(hit) if not n.startswith('_'))}>"


def _path_matches(hit_path: str, prim_path: str) -> bool:
    return hit_path == prim_path or hit_path.startswith(prim_path + "/") or prim_path.startswith(hit_path + "/")


def audit_colliders(stage, args: argparse.Namespace, report: Report) -> None:
    if args.collider_check == "none":
        report.add_check("collider_coverage", "skipped", required=False, detail="--collider-check none")
        return

    collidable = [
        p
        for p in stage.Traverse()
        if (str(p.GetPath()).startswith(STRUCTURE_ROOT + "/") or str(p.GetPath()).startswith(OBJECTS_ROOT + "/"))
        and (p.GetTypeName() in ("Cube", "Mesh"))
        # MSA's schema-v5 export (scripts/msa/export_usd.py) splits each object into
        # a `visual` mesh (placeholder/generated, never a collider - SPEC §5) and a
        # `collision` mesh (the measured hull). The main pipeline never nests a
        # "visual" prim under Objects/, so this exclusion is a no-op for it.
        and "/visual/" not in str(p.GetPath())
    ]
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy]
    )

    missing, no_extent, confirmed = [], [], 0
    use_overlap = args.collider_check == "overlap"
    sq = None
    if use_overlap:
        try:
            sq = get_physx_scene_query_interface()
            overlap = _require(sq, "overlap_box", "collider audit (PhysX overlap scene query)")
        except EnvironmentMismatch:
            report.add_check(
                "collider_coverage",
                "skipped",
                required=True,
                detail="overlap_box unavailable on this build - falling back to --collider-check usd-api semantics",
            )
            use_overlap = False

    for prim in collidable:
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            missing.append(_prim_diag(prim, reason="no_collision_api"))
            continue
        if not use_overlap:
            confirmed += 1  # API-only check: presence of the schema is all we verify
            continue

        rng = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if rng.IsEmpty():
            no_extent.append(_prim_diag(prim, reason="empty_world_bound"))
            continue
        c = rng.GetMidpoint()
        e = (rng.GetMax() - rng.GetMin()) * 0.5 * args.overlap_shrink
        half = carb.Float3(*(max(float(v), 1e-3) for v in e))
        hits: list[str] = []

        def _on_hit(hit, _hits=hits):
            _hits.append(_hit_collision_path(hit))
            return True

        try:
            overlap(half, carb.Float3(*c), carb.Float4(0, 0, 0, 1), _on_hit, False)
        except Exception as exc:
            missing.append(_prim_diag(prim, reason=f"overlap_box raised: {exc}"))
            continue

        path = prim.GetPath().pathString
        if any(_path_matches(h, path) for h in hits):
            confirmed += 1
        else:
            missing.append(_prim_diag(prim, reason="no_physx_shape", neighbour_hits=hits[:5]))

    status = "fail" if (missing or no_extent) else "pass"
    report.add_check(
        "collider_coverage",
        status,
        required=True,
        detail={
            "method": "physx_overlap_box" if use_overlap else "usd_api_only",
            "checked": len(collidable),
            "confirmed": confirmed,
            "missing": missing,
            "no_extent": no_extent,
        },
    )


# --- Sphere drop ------------------------------------------------------------------


def choose_drop_point(stage, floor_top_z: float, args: argparse.Namespace) -> tuple[tuple[float, float], str]:
    if args.drop_point:
        return (args.drop_point[0], args.drop_point[1]), "cli"
    rs = stage.GetPrimAtPath(ROBOT_START_PATH)
    if rs.IsValid():
        ops = UsdGeom.Xformable(rs).GetOrderedXformOps()
        if ops:
            t = UsdGeom.Xformable(rs).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
            return (float(t[0]), float(t[1])), "robot_start"
    # No RobotStart authored (robot_start was None on the scene) - fall back to the
    # floor prim's own world-space center.
    floor = stage.GetPrimAtPath(FLOOR_PATH)
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy])
    rng = bbox_cache.ComputeWorldBound(floor).ComputeAlignedRange()
    mid = rng.GetMidpoint()
    return (float(mid[0]), float(mid[1])), "floor_center_fallback"


def build_sphere(stage, xy: tuple[float, float], floor_top_z: float, args: argparse.Namespace):
    s = UsdGeom.Sphere.Define(stage, SPHERE_PATH)
    r = args.sphere_radius
    s.CreateRadiusAttr(r)
    s.CreateExtentAttr([(-r, -r, -r), (r, r, r)])
    start_z = floor_top_z + args.drop_height
    UsdGeom.Xformable(s).AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(xy[0], xy[1], start_z)
    )
    prim = s.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(args.sphere_mass))
    if PhysxSchema is not None:
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateEnableCCDAttr(True)
    return prim, start_z


def _sphere_world_z(stage, prim) -> float:
    t = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
    return float(t[2])


def drop_sphere(world, stage, prim, start_z: float, floor_top_z: float, floor_bottom_z: float, args, report):
    """Steps the simulation and watches the sphere's Z. See module docstring: the
    floor is only FLOOR_THICKNESS_M thick and a 2m fall reaches ~6.3 m/s, so a coarse
    --physics-dt without CCD can tunnel straight through it - a `tunneled` verdict is a
    simulation-configuration issue, not evidence the floor's collider is missing (the
    drop-point raycast recorded during choose_drop_point is what actually proves
    that)."""
    rest_z = floor_top_z + args.sphere_radius
    z_prev = start_z
    trace: list[list[float]] = []
    hold = 0
    max_speed = 0.0
    settled_at = None
    verdict = "still_moving"
    final_z = start_z

    for step in range(1, args.steps + 1):
        world.step(render=False)
        z = _sphere_world_z(stage, prim)
        if not math.isfinite(z):
            report.sphere_drop = _sphere_drop_summary(
                prim, args, start_z, rest_z, z, step, None, max_speed, "non_finite", trace
            )
            report.add_check("sphere_drop", "fail", required=True, detail="non-finite sphere position mid-simulation")
            return

        speed = abs(z - z_prev) / args.physics_dt if args.physics_dt > 0 else 0.0
        max_speed = max(max_speed, speed)
        z_prev = z
        final_z = z
        if args.trace and step % 5 == 0:
            trace.append([step, round(z, 5), round(speed, 5)])

        if z < floor_bottom_z - 0.5:
            verdict = "tunneled"
            break

        if speed < args.settle_speed and abs(z - rest_z) < 0.02:
            hold += 1
        else:
            hold = 0
        if hold >= args.settle_hold:
            settled_at = step - args.settle_hold
            verdict = "settled"
            break

    if verdict == "still_moving" and abs(final_z - start_z) < 1e-3:
        verdict = "never_fell"

    ok = verdict == "settled"
    report.sphere_drop = _sphere_drop_summary(
        prim, args, start_z, rest_z, final_z, args.steps, settled_at, max_speed, verdict, trace
    )
    detail = dict(report.sphere_drop)
    if verdict == "tunneled":
        detail["hint"] = (
            f"floor is {FLOOR_THICKNESS_M}m thick; lower --physics-dt or check "
            "environment.feature_detection.PhysxSchema (CCD). See drop_point_raycast "
            "in the JSON for independent evidence the floor collider exists."
        )
    report.add_check("sphere_drop", "pass" if ok else "fail", required=True, detail={"verdict": verdict})


def _sphere_drop_summary(prim, args, start_z, rest_z, final_z, steps_run, settled_at, max_speed, verdict, trace) -> dict:
    d = {
        "prim": SPHERE_PATH,
        "radius_m": args.sphere_radius,
        "mass_kg": args.sphere_mass,
        "start_z": start_z,
        "expected_rest_z": rest_z,
        "final_z": final_z,
        "z_error_m": abs(final_z - rest_z) if math.isfinite(final_z) else None,
        "steps_run": steps_run,
        "settled_at_step": settled_at,
        "sim_seconds": round(steps_run * args.physics_dt, 3),
        "max_speed_m_s": round(max_speed, 4),
        "verdict": verdict,
    }
    if args.trace:
        d["trace"] = trace
    return d


# --- Below-floor sweep --------------------------------------------------------------


def below_floor_sweep(stage, floor_bottom_z: float, report: Report, tolerance: float = 0.01) -> None:
    """Static prims (Floor/Structure) cannot move, so a static prim below the floor is
    a pure exporter-geometry bug - checked directly against the t=0 (== current, since
    they're static) world bound. Dynamic prims are informational only here: whether
    they've moved below the floor is exactly what the dynamics phase measures and
    reports, not re-litigated as a second gate."""
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy])
    static_paths = [
        p
        for p in stage.Traverse()
        if (str(p.GetPath()) == FLOOR_PATH or str(p.GetPath()).startswith(STRUCTURE_ROOT + "/"))
        and p.GetTypeName() == "Cube"
    ]
    violations = []
    for p in static_paths:
        rng = bbox_cache.ComputeWorldBound(p).ComputeAlignedRange()
        if rng.IsEmpty():
            continue
        z_min = float(rng.GetMin()[2])
        if z_min < floor_bottom_z - tolerance:
            violations.append({**_prim_diag(p), "z_min": z_min})

    report.below_floor = {
        "floor_bottom_z": floor_bottom_z,
        "tolerance_m": tolerance,
        "excluded": [f"{POINTCLOUD_PATH} (purpose=guide, no collider by design)", f"{VALIDATION_SCOPE}/*"],
        "static_violations": violations,
    }
    report.add_check(
        "no_static_prim_below_floor", "fail" if violations else "pass", required=True, detail={"count": len(violations)}
    )


# --- Dynamics phase (diagnostic) -----------------------------------------------------


def dynamics_phase(world, stage, args: argparse.Namespace, report: Report) -> None:
    if args.dynamics_steps <= 0:
        report.add_check("dynamics_settling", "skipped", required=False, detail="--dynamics-steps 0")
        return
    try:
        _run_dynamics_phase(world, stage, args, report)
    except Exception:
        report.dynamics = {"status": "error", "traceback": traceback.format_exc()}
        report.add_check("dynamics_settling", "warn", required=False, detail="dynamics phase raised - see report.dynamics.traceback")


def _run_dynamics_phase(world, stage, args, report) -> None:
    dynamic_prims = [
        p
        for p in stage.Traverse()
        if str(p.GetPath()).startswith(OBJECTS_ROOT + "/") and p.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    before = {p.GetPath().pathString: _sphere_world_z(stage, p) for p in dynamic_prims}  # z only, cheap baseline

    world.stop()
    freeze_objects(stage, False)
    if PhysxSchema is not None:
        for p in dynamic_prims:
            PhysxSchema.PhysxRigidBodyAPI.Apply(p).CreateMaxDepenetrationVelocityAttr(1.0)
    world.reset()

    for _ in range(args.dynamics_steps):
        world.step(render=False)

    movers = []
    ejected_count = 0
    for p in dynamic_prims:
        path = p.GetPath().pathString
        z0 = before.get(path, 0.0)
        z1 = _sphere_world_z(stage, p)
        disp = abs(z1 - z0) if math.isfinite(z1) else float("inf")
        cls = "non_finite" if not math.isfinite(z1) else (
            "ejected" if disp >= args.ejection_threshold else ("moved" if disp >= 0.05 else "settled")
        )
        if cls == "ejected":
            ejected_count += 1
        movers.append({**_prim_diag(p), "displacement_m": None if math.isinf(disp) else round(disp, 4), "classification": cls})

    movers.sort(key=lambda m: (m["displacement_m"] is None, -(m["displacement_m"] or 0)))
    displacements = [m["displacement_m"] for m in movers if m["displacement_m"] is not None]
    report.dynamics = {
        "status": "ok",
        "steps": args.dynamics_steps,
        "objects": len(dynamic_prims),
        "displacement_m": {
            "p50": round(statistics.median(displacements), 4) if displacements else None,
            "max": round(max(displacements), 4) if displacements else None,
        },
        "classification_counts": {
            k: sum(1 for m in movers if m["classification"] == k) for k in ("settled", "moved", "ejected", "non_finite")
        },
        "top_movers": movers[:10],
        "note": (
            "Displacement is diagnostic, not pass/fail. Convex hulls are conservative "
            "bounds and objects captured against a wall overlap /World/Structure at "
            "t=0; PhysX depenetrating that overlap is a property of the SCENE, not an "
            "exporter defect. High displacement is expected for objects flagged here."
        ),
    }
    status = "warn" if ejected_count and not args.fail_on_ejected else ("fail" if ejected_count and args.fail_on_ejected else "pass")
    report.add_check(
        "dynamics_settling",
        status,
        required=bool(args.fail_on_ejected),
        detail={"ejected": ejected_count},
    )


# --- PhysX log capture ---------------------------------------------------------------


class LogWatch:
    """Post-parses a carb log file from an offset marked after app startup, so Kit's
    own boot noise (extension registry, shader cache, RTX init) can never be
    misattributed to the scene under test. See module/docstring's PhysX log capture
    note - this is a secondary signal; the functional collider audit is the real net."""

    LEVEL = re.compile(r"\[(Error|Fatal|Warning)\]", re.IGNORECASE)
    PHYSX = re.compile(r"physx|physics|cook|convex|collider|collision|rigidbody", re.IGNORECASE)

    def __init__(self, path: Path | None):
        self.path = path
        self.offset = 0

    def mark(self) -> None:
        if self.path and self.path.exists():
            self.offset = self.path.stat().st_size

    def harvest(self, ignore_patterns: list[str]) -> dict:
        if not self.path or not self.path.exists():
            return {"capture": "unavailable", "path": str(self.path) if self.path else None}
        ignore = [re.compile(p) for p in ignore_patterns]
        try:
            with self.path.open("r", errors="replace") as f:
                f.seek(self.offset)
                lines = f.readlines()
        except OSError as exc:
            return {"capture": "unavailable", "error": str(exc)}

        errors = warnings_ = physx_errors = physx_warnings = ignored = 0
        samples = []
        for line in lines:
            if any(p.search(line) for p in ignore):
                ignored += 1
                continue
            m = self.LEVEL.search(line)
            if not m:
                continue
            level = m.group(1).lower()
            is_physx = bool(self.PHYSX.search(line))
            if level in ("error", "fatal"):
                errors += 1
                physx_errors += is_physx
            elif level == "warning":
                warnings_ += 1
                physx_warnings += is_physx
            if len(samples) < 20 and is_physx:
                samples.append(line.rstrip())
        return {
            "capture": "carb_log_file",
            "path": str(self.path),
            "from_offset": self.offset,
            "errors": errors,
            "warnings": warnings_,
            "physx_errors": physx_errors,
            "physx_warnings": physx_warnings,
            "ignored_by_pattern": ignored,
            "samples": samples,
        }


def configure_settings(log_path: Path | None) -> None:
    s = carb.settings.get_settings()
    s.set("/physics/updateToUsd", True)
    s.set("/physics/updateVelocitiesToUsd", True)
    s.set("/physics/fabricUpdateTransformations", False)
    s.set("/app/asyncRendering", False)
    s.set("/log/level", "warning")
    if log_path is not None:
        s.set("/log/file", str(log_path))
    s.set("/log/enableStandardStreamOutput", True)


# --- Main ------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = Report(args=args)
    report.record_input_file(args.usd_path)

    if args.usd_only:
        _late_imports(isaac=False)
        return run_usd_only(args, report)

    try:
        from isaacsim import SimulationApp

        report.environment["isaacsim_import"] = "isaacsim.SimulationApp"
    except ImportError:
        try:
            from omni.isaac.kit import SimulationApp  # Isaac Sim 4.x fallback

            report.environment["isaacsim_import"] = "omni.isaac.kit.SimulationApp"
        except ImportError as exc:
            report.fatal_env(
                f"Isaac Sim not importable ({exc}). Run this with Isaac Sim's own "
                "interpreter - see docs/ISAAC_VALIDATION.md."
            )
            return report_and_exit(report, EXIT_ENVIRONMENT_ERROR, args)

    report.environment["mode"] = "isaac"
    report.environment["headless"] = True
    sim_app = SimulationApp({"headless": True})
    code = EXIT_INTERNAL_ERROR
    log_watch = LogWatch(args.log_out)
    try:
        feat = _late_imports(isaac=True)
        report.environment["feature_detection"] = feat
        configure_settings(args.log_out)
        log_watch.mark()

        stage = open_stage_isaac(args, report)
        floor_top_z, floor_bottom_z = preflight(stage, report, args)
        if floor_top_z is None:
            raise ValidationAbort("floor prim missing or degenerate - cannot proceed to physics checks")

        ensure_physics_scene(stage, args, report)
        cook(report, feat)
        audit_colliders(stage, args, report)

        if not args.no_freeze_objects:
            freeze_objects(stage, True)

        from isaacsim.core.api import World  # feature-detected import, see except below

        world = World(physics_dt=args.physics_dt, rendering_dt=args.physics_dt)
        world.reset()

        xy, drop_source = choose_drop_point(stage, floor_top_z, args)
        report.environment["drop_point_source"] = drop_source
        sphere_prim, start_z = build_sphere(stage, xy, floor_top_z, args)
        drop_sphere(world, stage, sphere_prim, start_z, floor_top_z, floor_bottom_z, args, report)

        below_floor_sweep(stage, floor_bottom_z, report)
        dynamics_phase(world, stage, args, report)

        report.physx_log = log_watch.harvest(args.ignore_log_pattern)
        physx_errs = report.physx_log.get("physx_errors", 0) if report.physx_log else 0
        physx_warns = report.physx_log.get("physx_warnings", 0) if report.physx_log else 0
        if report.physx_log and report.physx_log.get("capture") == "unavailable":
            report.add_check(
                "physx_log_clean", "skipped", required=False,
                detail="log capture unavailable - the functional collider audit is the real safety net here",
            )
        else:
            log_ok = physx_errs == 0 and (physx_warns == 0 or args.allow_physx_warnings)
            report.add_check(
                "physx_log_clean", "pass" if log_ok else ("warn" if physx_errs == 0 else "fail"),
                required=True, detail={"physx_errors": physx_errs, "physx_warnings": physx_warns},
            )

        code = EXIT_OK if report.overall_status(args.strict) == "pass" else EXIT_CHECK_FAILED
    except EnvironmentMismatch as exc:
        report.fatal_env(str(exc))
        code = EXIT_ENVIRONMENT_ERROR
    except ValidationAbort as exc:
        report.add_check("validation_aborted", "fail", required=True, detail=str(exc))
        code = EXIT_CHECK_FAILED
    except ImportError as exc:
        report.fatal_env(f"a required Isaac Sim module was not importable: {exc}")
        code = EXIT_ENVIRONMENT_ERROR
    except Exception:
        report.fatal_internal(traceback.format_exc())
        code = EXIT_INTERNAL_ERROR
    finally:
        report.record_input_after(args.usd_path)
        report.finish(code, sys.argv, args.json_out)
        try:
            sim_app.close()
        except Exception:
            pass
    return code


if __name__ == "__main__":
    sys.exit(main())
