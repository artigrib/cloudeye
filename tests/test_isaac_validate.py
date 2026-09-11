"""tools/isaac_validate.py's Report/overall_status logic, tested at the pure-Python
level - no Isaac Sim needed (the module only imports isaacsim/pxr lazily, inside
function bodies, so importing it here is safe anywhere usd-core+pytest are installed).

2026-09-04: overall_status() treats any *required* check with status "skipped" as a
failure (by design - a required check that didn't run means the overall verdict can't
be trusted). physx_log_clean was marking itself "skipped" with required=True whenever
PhysX log capture was unavailable in the environment - an artifact of the sandbox, not
a real defect - which meant overall_status() could never return "pass" in that
environment regardless of how every other check went. Fixed by marking that particular
skip required=False (the same pattern already used elsewhere in this file for
environment-dependent skips, e.g. the --usd-only and --collider-check none paths).
"""

import argparse

from tools.isaac_validate import Report, _late_imports, audit_colliders


def _report_with(*checks: tuple[str, str, bool]) -> Report:
    r = Report(args=None)
    for name, status, required in checks:
        r.add_check(name, status, required=required)
    return r


def test_physx_log_clean_skipped_and_not_required_does_not_fail_overall():
    r = _report_with(
        ("stage_opened", "pass", True),
        ("collider_coverage", "pass", True),
        ("sphere_drop", "pass", True),
        ("physx_log_clean", "skipped", False),
    )
    assert r.overall_status(strict=False) == "pass"
    assert r.failures() == []


def test_a_required_skip_still_fails_overall_general_semantics_unchanged():
    # Guards the general "required check must actually run" rule that the
    # physx_log_clean fix deliberately opts out of - it should still hold for any
    # other check that's required and gets skipped.
    r = _report_with(
        ("stage_opened", "pass", True),
        ("collider_coverage", "skipped", True),
    )
    assert r.overall_status(strict=False) == "fail"
    assert r.failures() == ["collider_coverage: None"]


def test_physx_log_clean_pass_with_errors_still_fails_overall():
    # The fix only changes the "capture unavailable" skip path - a real, measured
    # PhysX error must still fail the run.
    r = _report_with(
        ("stage_opened", "pass", True),
        ("physx_log_clean", "fail", True),
    )
    assert r.overall_status(strict=False) == "fail"


def test_collider_coverage_ignores_visual_meshes_requires_collision_meshes(tmp_path):
    """2026-09-06 (Stage E2): MSA's schema-v5 export nests a `visual` (never a
    collider - SPEC §5) and a `collision` (measured hull, must be a collider)
    mesh under each object. Before this fix, audit_colliders() flagged EVERY
    Mesh/Cube under /World/Objects/ - including visual/* - as missing
    CollisionAPI, so no MSA scene could ever pass collider_coverage even with
    CollisionAPI correctly applied to collision/*. --collider-check usd-api
    (not "overlap") keeps this a pure-USD check, no Isaac Sim needed - matches
    this file's module docstring."""
    _late_imports(isaac=False)
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Objects")
    UsdGeom.Xform.Define(stage, "/World/Objects/bed_0")

    UsdGeom.Xform.Define(stage, "/World/Objects/bed_0/visual")
    visual_mesh = UsdGeom.Mesh.Define(stage, "/World/Objects/bed_0/visual/part_0")
    visual_mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    visual_mesh.CreateExtentAttr([(0, 0, 0), (1, 1, 0)])
    # deliberately NOT given CollisionAPI - it must not be required to have one.

    UsdGeom.Xform.Define(stage, "/World/Objects/bed_0/collision")
    hull_mesh = UsdGeom.Mesh.Define(stage, "/World/Objects/bed_0/collision/hull")
    hull_mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    hull_mesh.CreateExtentAttr([(0, 0, 0), (1, 1, 0)])
    UsdPhysics.CollisionAPI.Apply(hull_mesh.GetPrim())

    args = argparse.Namespace(collider_check="usd-api")
    report = Report(args=args)
    audit_colliders(stage, args, report)

    assert report.checks["collider_coverage"]["status"] == "pass"


def test_collider_coverage_still_flags_collision_mesh_missing_the_api(tmp_path):
    """The exclusion is scoped to /visual/ only - a collision/* mesh (or any
    other structural/object mesh) missing CollisionAPI must still fail."""
    _late_imports(isaac=False)
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Objects")
    UsdGeom.Xform.Define(stage, "/World/Objects/bed_0")
    UsdGeom.Xform.Define(stage, "/World/Objects/bed_0/collision")
    hull_mesh = UsdGeom.Mesh.Define(stage, "/World/Objects/bed_0/collision/hull")
    hull_mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    hull_mesh.CreateExtentAttr([(0, 0, 0), (1, 1, 0)])
    # deliberately NOT given CollisionAPI.

    args = argparse.Namespace(collider_check="usd-api")
    report = Report(args=args)
    audit_colliders(stage, args, report)

    assert report.checks["collider_coverage"]["status"] == "fail"
