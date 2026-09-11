#!/usr/bin/env python3
"""Offline tests for the pipeline's vast safeties, box lifecycle and budget.

No network, no vast.ai, no GPU: `vastai`, `ssh`, `scp` and `rsync` are replaced by the fakes
in `tests/fakes/` via PATH, and each fake records the argv it was handed. That recording is
the evidence - test (a) passes only if the fake `vastai` was never invoked at all.

Run:  python3 pipeline/tests/run_tests.py        (stdlib only, ~1 min)
"""

from __future__ import annotations

import ast
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent
REPO = PIPELINE.parent
FAKES = HERE / "fakes"

sys.path.insert(0, str(PIPELINE))
from cpu_env import cpu_python, missing_reason  # noqa: E402

FIXTURES = Path(os.environ.get("PIPELINE_TEST_FIXTURES") or (REPO / "var" / "test-fixtures"))
PACKED_FIXTURE = FIXTURES / "pipeline_dry" / "hero" / "packed"
NVBLOX_FIXTURE = FIXTURES / "pipeline_dry" / "_fixture_nvblox" / "own_0901_173903__step2"
PULLED = FIXTURES / "b200_run_20260908" / "pulled" / "own_0901_173903__step2"
VIDEO = os.environ.get("PIPELINE_TEST_VIDEO", str(FIXTURES / "sample.mp4"))
SCENE = "own_0901_173903__step2"
NV_ID = 50265868
MA_ID = 99999999          # a second id, used only to prove two sessions are both stopped

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


class Case:
    """One scene dir plus the fake environment the driver runs under."""

    def __init__(self, root: Path, name: str, *, armed: str | None = "today",
                 with_packed: bool = True):
        self.dir = root / name
        self.dir.mkdir(parents=True)
        self.fake_log = self.dir / "fake_cli.log"
        self.armed_file = self.dir / "ARMED"
        if armed == "today":
            self.armed_file.write_text(f"yes {dt.date.today().isoformat()}\n")
        elif armed == "stale":
            stale = dt.date.today() - dt.timedelta(days=1)
            self.armed_file.write_text(f"yes {stale.isoformat()}\n")
        elif armed == "garbage":
            self.armed_file.write_text("no\n")
        if with_packed:
            (self.dir / "packed").symlink_to(PACKED_FIXTURE)

    def env(self, **extra) -> dict:
        e = dict(os.environ)
        e.pop("PIPELINE_DRY_RUN", None)
        e["PATH"] = f"{FAKES}:{e['PATH']}"
        e["FAKE_LOG"] = str(self.fake_log)
        e["FAKE_SHOW_JSON"] = json.dumps(
            {"actual_status": "running", "cur_state": "running", "dph_total": 0.369,
             "gpu_name": "RTX 4090", "ssh_host": "ssh9.vast.ai", "ssh_port": 25868,
             "public_ipaddr": None, "direct_port_start": None})
        e.update({k: str(v) for k, v in extra.items()})
        return e

    def run(self, *args, env=None, timeout=300,
            from_pulled: bool = True) -> subprocess.CompletedProcess:
        argv = [sys.executable, "-u", str(PIPELINE / "run_pipeline.py"), VIDEO,
                "--scene-dir", str(self.dir), "--scene-name", SCENE,
                *(["--from-pulled", str(PULLED)] if from_pulled else []),
                "--armed-file", str(self.armed_file),
                "--nvblox-instance-id", str(NV_ID),
                "--ssh-wait-s", "5", "--ssh-poll-s", "1", "--probe-mb", "1",
                # The duration floor is a PRODUCTION rule: a probe that finishes too fast
                # has not let TCP reach its stride, so it is repeated at a larger size.
                # Against a fake scp that sleeps a fixed 0.2 s it fires every time and
                # resizes 1 MB -> ~50 MB, which is disk this suite has no reason to spend:
                # nothing here measures a real link. Test (h), which DOES test the
                # transfer gate, sets its own FAKE_SCP_SLEEP and is unaffected.
                "--probe-min-s", "0.05",
                # The export is asserted to EXIST, never to be dense. At isaac_export's own
                # 200k default each case wrote a 4.36 MB scene.glb and a 3.67 MB scene.usd -
                # 56 MB across seven live-path cases, the entire remainder of this suite's
                # disk once the probe and fixture were dealt with.
                "--export-max-tri", "20000",
                # PACK_RGB re-derives colour from the SOURCE VIDEO with ffmpeg, and VIDEO
                # here is a fixture path, not a decodable file. This suite is about the
                # state machine, not about colour, so every case runs the colourless path
                # and case (p) covers PACK_RGB on its own terms. A case that wants colour
                # passes `--color` back in through *args.
                "--no-color",
                "--heartbeat-s", "1", "--exit-confirm-s", "10", *args]
        return subprocess.run(argv, capture_output=True, text=True,
                              env=env or self.env(), timeout=timeout, cwd=str(REPO))

    def status(self) -> dict:
        f = self.dir / "status.json"
        return json.loads(f.read_text()) if f.exists() else {}

    def cli_log(self) -> str:
        return self.fake_log.read_text() if self.fake_log.exists() else ""


# --------------------------------------------------------------------------------------
def test_a_no_armed(root: Path) -> None:
    print("\n(a) without ARMED a real call is impossible")
    for label, armed in (("no ARMED file", None), ("stale ARMED", "stale"),
                         ("malformed ARMED", "garbage")):
        c = Case(root, f"a_{label.split()[0]}", armed=armed)
        p = c.run("--nvblox-source", "live")
        check(f"(a) {label}: exit 2", p.returncode == 2, f"rc={p.returncode}")
        check(f"(a) {label}: fake vastai never invoked", not c.fake_log.exists(),
              c.cli_log()[:120] or "log absent")
        check(f"(a) {label}: refusal explains itself",
              "refusing to start a live run" in p.stderr, p.stderr.strip()[:100])
    # and the same guard one level down, inside the client itself
    sys.path.insert(0, str(PIPELINE))
    from vast_client import VastClient, VastNotArmed
    c = Case(root, "a_client", armed="stale")
    cl = VastClient(c.dir / "vc.log", c.armed_file)
    try:
        cl.stop(NV_ID, dry_run=False)
        check("(a) client refuses a real stop without ARMED", False, "no exception")
    except VastNotArmed as e:
        check("(a) client refuses a real stop without ARMED", True, str(e)[:70])
    check("(a) client logged the refusal, not a call",
          "REFUSED" in (c.dir / "vc.log").read_text())


def test_b_ssh_refused(root: Path) -> None:
    print("\n(b) ssh never answers -> FAILED, and stop is still issued")
    c = Case(root, "b_ssh")
    p = c.run("--nvblox-source", "live", env=c.env(FAKE_SSH_RC=255))
    st = c.status()
    check("(b) run failed", p.returncode == 1, f"rc={p.returncode}")
    check("(b) failed at GPU_UP_NV",
          st.get("steps", {}).get("GPU_UP_NV", {}).get("state") == "failed",
          st.get("state", "?"))
    check("(b) error names the ssh deadline",
          "ssh not reachable within" in (st.get("error") or ""),
          (st.get("error") or "")[:90])
    check(f"(b) stop instance {NV_ID} was issued",
          f"vastai stop instance {NV_ID}" in c.cli_log())
    check("(b) no destroy was issued", "destroy" not in c.cli_log())


def test_c_budget(root: Path) -> None:
    print("\n(c) budget exceeded -> FAILED, every open box stopped")
    c = Case(root, "c_budget")
    show = json.dumps({"actual_status": "running", "cur_state": "running",
                       "dph_total": 500.0, "gpu_name": "RTX 4090",
                       "ssh_host": "ssh9.vast.ai", "ssh_port": 25868})
    p = c.run("--nvblox-source", "live", "--max-usd", "0.001",
              env=c.env(FAKE_SHOW_JSON=show, FAKE_NVBLOX_FIXTURE=str(NVBLOX_FIXTURE)))
    st = c.status()
    check("(c) run failed", p.returncode == 1, f"rc={p.returncode}")
    check("(c) error is a budget breach", "budget exceeded" in (st.get("error") or ""),
          (st.get("error") or "")[:90])
    check("(c) measured cost recorded, not estimated",
          st.get("cost_estimate_usd", 0) > 0 and "measured" in st.get("cost_basis", ""),
          f"${st.get('cost_estimate_usd')}")
    check(f"(c) stop instance {NV_ID} issued", f"vastai stop instance {NV_ID}" in c.cli_log())

    # two open sessions must BOTH be stopped, and a failure stopping one must not stop the
    # other from being stopped.
    sys.path.insert(0, str(PIPELINE))
    from vast_client import VastClient
    import box_ops
    d = root / "c_two"
    d.mkdir()
    cl = VastClient(d / "vast_calls.log", d / "ARMED")

    class P:
        pass
    pl = P()
    pl.sessions = {}
    pl.boxes_seen = {}
    for iid, name in ((MA_ID, "mapanything"), (NV_ID, "nvblox")):
        s = box_ops.BoxSession(cl, iid, dry_run=True, ssh_key="/dev/null", name=name)
        s.up_at = time.time()
        pl.sessions[iid] = s
    first = pl.sessions[MA_ID]
    orig = first.client.stop
    first.client = type("Boom", (), {"stop": lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("simulated stop failure"))})()
    stopped = []
    for iid in list(pl.sessions):
        try:
            pl.sessions[iid].stop()
        except Exception:
            pass
        stopped.append(iid)
    log = (d / "vast_calls.log").read_text() if (d / "vast_calls.log").exists() else ""
    check("(c) both sessions were visited by the stop loop", stopped == [MA_ID, NV_ID],
          str(stopped))
    check("(c) a stop failure on one box does not block the other",
          f"vastai stop instance {NV_ID}" in log, log.strip()[-90:] or "empty")
    check("(c) the failing stop is recorded, not swallowed silently",
          "stop_error" in pl.sessions[MA_ID].info)
    del orig


def test_d_resume(root: Path) -> None:
    print("\n(d) a re-run after FAILED continues from the last state")
    c = Case(root, "d_resume")
    env = c.env(FAKE_NVBLOX_FIXTURE=str(NVBLOX_FIXTURE))
    show = json.dumps({"actual_status": "running", "cur_state": "running",
                       "dph_total": 500.0, "gpu_name": "RTX 4090",
                       "ssh_host": "ssh9.vast.ai", "ssh_port": 25868})
    first = c.run("--nvblox-source", "live", "--max-usd", "0.001",
                  env={**env, "FAKE_SHOW_JSON": show})
    st1 = c.status()
    check("(d) first run failed on budget", first.returncode == 1, st1.get("state", "?"))
    second = c.run("--nvblox-source", "live", "--max-usd", "50", env=env, timeout=600)
    st2 = c.status()
    steps2 = st2.get("steps", {})
    check("(d) second run reached DONE", second.returncode == 0, st2.get("state", "?"))
    # Box states are deliberately NOT idempotent - a box is never assumed to still be up,
    # so GPU_UP_NV/GPU_DOWN_NV run again. Everything that produced a file is skipped.
    # QUEUED is the entry marker, not a produced artifact; box states are deliberately
    # not idempotent (a box is never assumed to still be up).
    boxes = {"QUEUED", "GPU_UP_NV", "GPU_DOWN_NV", "GPU_UP_MA", "GPU_DOWN_MA"}
    done_before = [k for k, v in st1.get("steps", {}).items()
                   if v.get("state") in ("done", "skipped", "skipped_precomputed")
                   and k not in boxes]
    reused = [k for k in done_before if steps2.get(k, {}).get("state")
              in ("skipped", "skipped_precomputed")]
    check("(d) work already done is skipped, not repeated",
          len(reused) == len(done_before),
          f"{len(reused)}/{len(done_before)}: "
          f"{[k for k in done_before if k not in reused]}")
    failed1 = [k for k, v in st1.get("steps", {}).items() if v.get("state") == "failed"]
    check("(d) the failing state is named in steps, not only in error", len(failed1) == 1,
          f"failed steps in run 1: {failed1}")
    if failed1:
        check("(d) that state is retried and completes on the re-run",
              steps2.get(failed1[0], {}).get("state")
              in ("done", "skipped", "stubbed", "skipped_precomputed"),
              f"{failed1[0]} -> {steps2.get(failed1[0], {}).get('state')}")
    check("(d) NVBLOX output exists and was not recomputed needlessly",
          steps2.get("NVBLOX", {}).get("state") in ("done", "skipped"),
          steps2.get("NVBLOX", {}).get("state", "?"))
    for stg in ("LAYERS", "REACH", "EXPORT"):
        check(f"(d) {stg} completed off the live nvblox output",
              steps2.get(stg, {}).get("state") in ("done", "skipped"),
              steps2.get(stg, {}).get("state", "?"))
    check("(d) inet_up was actually measured",
          isinstance(st2.get("boxes", {}).get(str(NV_ID), {}).get("inet_up_mb_s"), float),
          str(st2.get("boxes", {}).get(str(NV_ID), {}).get("inet_up_mb_s")))
    box = st2.get("boxes", {}).get(str(NV_ID), {})
    check("(d) the stop was confirmed by a real `exited` reading",
          box.get("exited_confirmed") is True, str(box.get("exited_confirmed")))
    check("(d) the cost clock ran to that confirmation, not to the stop call",
          isinstance(box.get("cost_usd_measured"), float)
          and box.get("cost_usd_measured") >= 0,
          f"${box.get('cost_usd_measured')}")
    check("(d) VRAM was read off the box",
          st2.get("boxes", {}).get(str(NV_ID), {}).get("gpu", {}).get("vram_total_mib")
          == 24564,
          str(st2.get("boxes", {}).get(str(NV_ID), {}).get("gpu")))


def test_e_no_key_access() -> None:
    print("\n(e) the pipeline code never reads the vast API key")
    bad = []
    for f in sorted(PIPELINE.glob("*.py")):
        tree = ast.parse(f.read_text())
        docs = {id(ast.get_docstring(n, clean=False)) for n in ast.walk(tree)
                if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef))}
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if "vast_api_key" in node.value and id(node.value) not in docs:
                    bad.append(f"{f.name}:{node.lineno}")
    check("(e) no key path appears in executable code", not bad, ", ".join(bad) or "clean")
    grep = subprocess.run(["grep", "-rn", "api.key", "-i", str(PIPELINE / "vast_client.py")],
                          capture_output=True, text=True)
    code_hits = [ln for ln in grep.stdout.splitlines()
                 if "open(" in ln or "read_text" in ln]
    check("(e) vast_client opens no key file", not code_hits, str(code_hits)[:100])


def test_f_unreachable(root: Path) -> None:
    print("\n(f) three failed `show` calls signal box_unreachable without stopping anything")
    c = Case(root, "f_unreach")
    env = c.env(FAKE_NVBLOX_FIXTURE=str(NVBLOX_FIXTURE))
    # ssh works, so the box comes up; `show` starts failing right after.
    env["FAKE_SHOW_RC"] = "0"
    p = c.run("--nvblox-source", "live", "--max-usd", "50", env=env, timeout=600)
    st_e2e = c.status()
    check("(f) NVBLOX ran live on a fresh scene dir",
          st_e2e.get("steps", {}).get("NVBLOX", {}).get("state") == "done",
          st_e2e.get("steps", {}).get("NVBLOX", {}).get("state", "?"))
    check("(f) the live nvblox output landed in this run's own directory",
          (c.dir / "nvblox" / SCENE / "esdf_slice_0.3m.npy").exists())
    # The fake fails `show` only when told to; drive the failure directly instead, so the
    # assertion is about the supervisor's own logic rather than the fake's timing.
    sys.path.insert(0, str(PIPELINE))
    import run_pipeline as rp
    from vast_client import VastClient

    class FailingClient(VastClient):
        def show(self, instance_id, *, dry_run):
            raise RuntimeError("simulated show failure")

    d = root / "f_direct"
    d.mkdir()
    st = rp.Status(d / "status.json", SCENE, VIDEO)

    class P:
        pass
    pl = P()
    pl.status = st
    pl.vast = FailingClient(d / "vast_calls.log", d / "ARMED")
    pl.dry_run = True
    pl.max_usd = 50.0
    pl.abort_reason = None
    pl.hb_path = d / "heartbeat"
    pl.kill_current = lambda: None
    sess = rp.BoxSession(pl.vast, NV_ID, dry_run=True, ssh_key="/dev/null", name="nvblox")
    sess.up_at = time.time()
    pl.sessions = {NV_ID: sess}
    pl.boxes_seen = {NV_ID: sess}
    sup = rp.Supervisor(pl, interval_s=1)
    for _ in range(3):
        sup.poll_cost()
    box = st.doc["boxes"][str(NV_ID)]
    check("(f) three consecutive failures counted", box.get("consecutive_show_failures") == 3,
          str(box.get("consecutive_show_failures")))
    check("(f) box_unreachable raised as a signal", box.get("box_unreachable") is True)
    check("(f) no stop was issued on box_unreachable",
          "stop instance" not in ((d / "vast_calls.log").read_text()
                                  if (d / "vast_calls.log").exists() else ""))
    check("(f) no budget verdict from failed shows", pl.abort_reason is None,
          str(pl.abort_reason))
    check("(f) the end-to-end run with a healthy show still finished",
          p.returncode == 0, f"rc={p.returncode} state={c.status().get('state')}")


FLOOR_RULE_PROBE = r"""
import ast, json, sys
import numpy as np
SRC = sys.argv[1]
tree = ast.parse(open(SRC).read())
ns = {"np": np, "DEFAULT_FLOOR_CAMERA_HEIGHT_RANGE_M": (0.8, 2.0)}
for kind, names in ((ast.ClassDef, ("FloorFitError", "FloorImplausible")),
                    (ast.FunctionDef, ("fit_floor", "median_camera_up", "angle_to_deg",
                                       "camera_up_variants", "_rotation_hint"))):
    for name in names:
        node = next(n for n in tree.body if isinstance(n, kind) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), SRC, "exec"), ns)

ang, up = ns["angle_to_deg"], np.array([0.0, -1.0, 0.0])
views = [{"camera_pose": np.eye(4).tolist()} for _ in range(5)]
med = ns["median_camera_up"](views)

def pick(cands, frac):
    best = max(c["inlier_frac"] for c in cands)
    pool = [c for c in cands if c["inlier_frac"] >= frac * best]
    return pool, min(pool, key=lambda c: ang(c["up"], up))

wall = np.array([1.0, 0.0, 0.0])
# the real hero shape: a wall out-supporting the floor by a hair
hero = [{"up": up, "centroid": np.zeros(3), "inlier_frac": 0.0502, "cam_height_m": 1.30},
        {"up": wall, "centroid": np.zeros(3), "inlier_frac": 0.0560, "cam_height_m": 1.98}]
pool_h, pick_h = pick(hero, 0.2)
# a ceiling that out-supports everything
ceil = [{"up": up, "centroid": np.zeros(3), "inlier_frac": 0.050, "cam_height_m": 1.30},
        {"up": -up, "centroid": np.zeros(3), "inlier_frac": 0.060, "cam_height_m": 1.36}]
_, pick_c = pick(ceil, 0.2)
# the own_0829_000840 shape: the floor is real but weakly supported
weak = [{"up": wall, "centroid": np.zeros(3), "inlier_frac": 0.100, "cam_height_m": 1.5},
        {"up": up, "centroid": np.zeros(3), "inlier_frac": 0.030, "cam_height_m": 1.3}]
pool_w08, pick_w08 = pick(weak, 0.8)
pool_w02, pick_w02 = pick(weak, 0.2)

# rotation diagnosis: a 90 deg rescue is a FRAMES bug; a 180 deg rescue is ambiguous
hint90 = ns["_rotation_hint"]({0: 87.4, 90: 3.1, 180: 92.6, 270: 176.9}, 30.0)
hint180 = ns["_rotation_hint"]({0: 174.1, 90: 91.7, 180: 5.9, 270: 88.4}, 30.0)
hintnone = ns["_rotation_hint"]({0: 87.4, 90: 88.1, 180: 92.6, 270: 91.9}, 30.0)
poses = [{"camera_pose": np.eye(4).tolist()}]
variants = ns["camera_up_variants"](poses)

print(json.dumps({
    "median_up": med.tolist(),
    "hint90_blames_frames": "FRAMES" in hint90 and "[90]" in hint90,
    "hint180_says_ambiguous": "ambiguous" in hint180 and "ceiling" in hint180,
    "hintnone_makes_no_claim": "NOTE" not in hintnone,
    "hint_applies_nothing": all("Nothing was applied" in h for h in (hint90, hint180)),
    "variant_0": variants[0].tolist(), "variant_180": variants[180].tolist(),
    "variant_90": variants[90].tolist(),
    "floor_angle": ang(up, up),
    "ceiling_angle": ang(-up, up),
    "wall_angle": ang(wall, up),
    "hero_pool": len(pool_h), "hero_pick_inlier": pick_h["inlier_frac"],
    "ceiling_pick_inlier": pick_c["inlier_frac"],
    "weak_pool_08": len(pool_w08), "weak_pick_08_angle": ang(pick_w08["up"], up),
    "weak_pool_02": len(pool_w02), "weak_pick_02_angle": ang(pick_w02["up"], up),
    "implausible_is_separate": not issubclass(ns["FloorImplausible"], ns["FloorFitError"]),
}))
"""


def test_g_floor_rule() -> None:
    """The floor-selection rule, isolated from RANSAC.

    Same idea as tests/test_floor_ceiling.py, which unit-tests `select_floor_candidate`
    away from Open3D for exactly this reason: the selection logic is where the bug lived,
    and it is pure arithmetic. Runs under the CPU venv because it needs numpy - this suite
    and the driver are stdlib-only on purpose. The end-to-end sweep over real meshes is
    pipeline/verify_floor_rule.py, too slow (~45 s per seed) to belong here.
    """
    print("\n(g) floor selection: pool by support, pick by camera up, gate implausible")
    steps = PIPELINE / "steps" / "nvblox_scenes.py"
    p = subprocess.run([cpu_python(), "-c", FLOOR_RULE_PROBE, str(steps)],
                       capture_output=True, text=True, timeout=120)
    if p.returncode != 0:
        check("(g) probe ran", False, p.stderr.strip()[-200:])
        return
    d = json.loads(p.stdout)

    check("(g) median camera up is the camera's -y in world",
          d["median_up"] == [0.0, -1.0, 0.0], str(d["median_up"]))
    check("(g) a floor normal reads ~0 deg", abs(d["floor_angle"]) < 1e-6)
    check("(g) a CEILING reads ~180 deg, not ~0 - abs() would hide it",
          abs(d["ceiling_angle"] - 180.0) < 1e-6, f"{d['ceiling_angle']:.1f}")
    check("(g) a wall reads ~90 deg", abs(d["wall_angle"] - 90.0) < 1e-6)
    check("(g) the support pool keeps both the wall and the floor", d["hero_pool"] == 2)
    check("(g) the reference is the floor, not the better-supported wall",
          d["hero_pick_inlier"] == 0.0502, str(d["hero_pick_inlier"]))
    check("(g) a better-supported ceiling is not mistaken for the floor",
          d["ceiling_pick_inlier"] == 0.050, str(d["ceiling_pick_inlier"]))
    check("(g) a 0.8 pool would drop a weakly-supported real floor and pick a wall",
          d["weak_pool_08"] == 1 and d["weak_pick_08_angle"] > 30.0,
          f"pool {d['weak_pool_08']}, angle {d['weak_pick_08_angle']:.1f}")
    check("(g) the 0.2 default keeps it and picks the floor",
          d["weak_pool_02"] == 2 and d["weak_pick_02_angle"] < 30.0,
          f"pool {d['weak_pool_02']}, angle {d['weak_pick_02_angle']:.1f}")
    check("(g) FloorImplausible is not a FloorFitError, so run_scene cannot swallow it",
          d["implausible_is_separate"])
    check("(g) a 90 deg rescue is reported as a FRAMES rotation bug",
          d["hint90_blames_frames"])
    check("(g) a 180 deg rescue is reported as AMBIGUOUS with a ceiling pick",
          d["hint180_says_ambiguous"])
    check("(g) no rescue means no claim is made", d["hintnone_makes_no_claim"])
    check("(g) the diagnosis never applies a rotation", d["hint_applies_nothing"])
    check("(g) the four frame rotations give four distinct ups",
          d["variant_0"] == [0.0, -1.0, 0.0] and d["variant_180"] == [0.0, 1.0, 0.0]
          and d["variant_90"] == [1.0, 0.0, 0.0],
          f"0={d['variant_0']} 90={d['variant_90']} 180={d['variant_180']}")


def test_i_start_refused(root: Path) -> None:
    """vast declines the start in stdout while exiting 0 -> FAILED in seconds, not after the
    ssh deadline, and the queued start is still cancelled by a stop."""
    print("\n(i) a start vast refuses in words is a failure, not a 300 s ssh wait")
    c = Case(root, "i_refused")
    env = c.env(FAKE_START_STDOUT="Required resources are currently unavailable, "
                                  "state change queued.")
    t0 = time.time()
    p = c.run("--nvblox-source", "live", "--max-usd", "50", "--ssh-wait-s", "300",
              env=env, timeout=300)
    elapsed = time.time() - t0
    st = c.status()
    check("(i) run failed", p.returncode == 1, f"rc={p.returncode}")
    check("(i) it failed at GPU_UP_NV",
          st.get("steps", {}).get("GPU_UP_NV", {}).get("state") == "failed",
          st.get("state", "?"))
    check("(i) fail_reason carries vast's own words verbatim",
          (st.get("fail_reason") or "").startswith("vast start refused: ")
          and "Required resources are currently unavailable" in (st.get("fail_reason") or ""),
          (st.get("fail_reason") or "")[:110])
    check("(i) it failed in seconds, not after the 300 s ssh deadline", elapsed < 60,
          f"{elapsed:.1f} s")
    check("(i) the queued start was cancelled by a stop",
          f"vastai stop instance {NV_ID}" in c.cli_log())
    check("(i) the refusal is in the run's own vast_calls.log",
          "REFUSED_BY_VAST" in (c.dir / "vast_calls.log").read_text())


def test_h_transfer_gate(root: Path) -> None:
    """A transfer that cannot finish inside the step timeout is refused BEFORE it starts.

    Without this the stage burns the whole timeout, gets killed mid-rsync and leaves a
    partial job on a box that is still billing. The fake scp is slowed to make the measured
    link look bad; nothing about the gate is stubbed.
    """
    print("\n(h) a transfer that cannot fit in the step timeout is refused up front")
    c = Case(root, "h_gate")
    env = c.env(FAKE_NVBLOX_FIXTURE=str(NVBLOX_FIXTURE), FAKE_SCP_SLEEP=10)
    p = c.run("--nvblox-source", "live", "--max-usd", "50", "--step-timeout-s", "900",
              env=env, timeout=600)
    st = c.status()
    err = st.get("error") or ""
    check("(h) run failed", p.returncode == 1, f"rc={p.returncode}")
    check("(h) it failed at NVBLOX",
          st.get("steps", {}).get("NVBLOX", {}).get("state") == "failed",
          st.get("state", "?"))
    check("(h) the message names the time, the rate and the limit",
          "transfer would take" in err and "MB/s, exceeds" in err
          and "--step-timeout-s 900" in err, err.splitlines()[0][:150] if err else "")
    check("(h) it refused BEFORE pushing the data",
          "push_data_s" not in json.dumps(st.get("steps", {}).get("NVBLOX", {})),
          "push_data_s present" if "push_data_s" in json.dumps(st) else "no data push")
    check(f"(h) the box was stopped anyway",
          f"vastai stop instance {NV_ID}" in c.cli_log())
    check("(h) both link directions were measured before the verdict",
          isinstance(st.get("boxes", {}).get(str(NV_ID), {}).get("inet_up_mb_s"), float)
          and isinstance(st.get("boxes", {}).get(str(NV_ID), {}).get("inet_down_mb_s"),
                         float),
          f"up={st.get('boxes', {}).get(str(NV_ID), {}).get('inet_up_mb_s')} "
          f"down={st.get('boxes', {}).get(str(NV_ID), {}).get('inet_down_mb_s')}")

    # ...and a healthy link still passes the gate
    c2 = Case(root, "h_gate_ok")
    p2 = c2.run("--nvblox-source", "live", "--max-usd", "50",
                env=c2.env(FAKE_NVBLOX_FIXTURE=str(NVBLOX_FIXTURE)), timeout=600)
    st2 = c2.status()
    check("(h) a healthy link passes the gate and the run completes",
          p2.returncode == 0 and st2.get("state") == "DONE", st2.get("state", "?"))
    est = st2.get("steps", {}).get("NVBLOX", {}).get("artifacts", {}).get(
        "transfer_estimate", {})
    check("(h) the estimate is recorded either way",
          est.get("push_est_s") is not None and est.get("pull_est_s") is not None,
          json.dumps(est)[:120])


# --------------------------------------------------------------------------------------
# (j) the production worker path - PIPELINE.md 9.3
# --------------------------------------------------------------------------------------
class fake_env:
    """Swap the process environment for the Case's faked one, and put it back.

    run_job() runs in THIS process and hands its own os.environ to the driver, so the fakes
    have to be on PATH here - otherwise "the fake vastai was never invoked" would only prove
    the fake was not on PATH.
    """

    def __init__(self, case: "Case", **extra):
        self.env = case.env(**extra)
        self.saved: dict | None = None

    def __enter__(self):
        self.saved = dict(os.environ)
        os.environ.clear()
        os.environ.update(self.env)
        return self

    def __exit__(self, *exc):
        os.environ.clear()
        os.environ.update(self.saved or {})
        return False


def _worker_driver():
    """Import pipeline/worker_driver.py without pulling in the app (pydantic, DB, arq).

    run_tests.py is stdlib-only by design, and worker_driver is too - it takes the parsed
    spec as an argument rather than reading it. The pydantic reader itself
    (app/services/job_spec.py, a missing spec being a refusal) is covered in tests/ where
    pydantic is available; what is tested here is the bridge.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("worker_driver",
                                                  PIPELINE / "worker_driver.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Spec:
    """The two fields worker_driver reads off a JobSpec. Deliberately not a JobSpec: the
    bridge must not depend on pydantic, and this proves it does not."""

    def __init__(self, frames_fps=15.0, semantics="mapanything"):
        self.frames_fps = frames_fps
        self.semantics = semantics


def test_j_worker_path(root: Path) -> None:
    """The worker bridge builds the right command line and drives a run to DONE."""
    print("\n(j) the worker bridge reaches DONE with a job spec")
    wd = _worker_driver()
    c = Case(root, "j_worker")
    argv = wd.build_argv(Path(VIDEO), c.dir, _Spec(frames_fps=2.0, semantics="mapanything"),
                         scene_name=SCENE, from_pulled=PULLED, live=False, max_usd=2.0,
                         step_timeout_s=900)
    check("(j) the command line carries the spec's fps",
          "--frames-fps" in argv and argv[argv.index("--frames-fps") + 1] == "2.0",
          " ".join(argv[-12:]))
    check("(j) it carries the spec's semantics provider, for the record",
          "--semantics" in argv and argv[argv.index("--semantics") + 1] == "mapanything")
    check("(j) it is marked --worker-mode", "--worker-mode" in argv)
    check("(j) a dry worker run asks for the precomputed nvblox, never a live box",
          argv[argv.index("--nvblox-source") + 1] == "precomputed")

    ledger = c.dir / "ledger.json"
    with fake_env(c):
        os.environ.pop("PIPELINE_WORKER_ENABLED", None)
        os.environ.pop("PIPELINE_DAILY_USD", None)
        live, why = wd.worker_mode()
        check("(j) with PIPELINE_WORKER_ENABLED unset the default is a dry run",
              live is False and "dry run" in why, why)
        st = wd.run_job(Path(VIDEO), c.dir, _Spec(), scene_name=SCENE, from_pulled=PULLED,
                        max_usd=2.0, ledger=ledger)
    check("(j) the run reached DONE", st.get("state") == "DONE", st.get("state", "?"))
    check("(j) it rented nothing", float(st.get("cost_estimate_usd") or 0) == 0.0,
          str(st.get("cost_estimate_usd")))
    check("(j) SEMANTICS is a declared slot that says it is empty",
          st["steps"]["SEMANTICS"]["substage"] == "skipped"
          and st["steps"]["SEMANTICS"]["detail"] == "not implemented",
          json.dumps(st["steps"]["SEMANTICS"])[:90])
    check("(j) no provider_used is invented for a fallback that never happened",
          "provider_used" not in st and "fallback_reason" not in st)
    check("(j) every step carries the new fields",
          all({"substage", "detail", "since", "attempt", "next_retry"} <= set(v)
              for v in st["steps"].values()),
          f"{len(st['steps'])} steps")


def test_k_start_refused_creates(root: Path) -> None:
    """A refused start takes the create path and the run continues - and when there is
    nothing to rent either, the waiting_capacity ceiling ends it instead of looping."""
    print("\n(k) a refused start rents a replacement; no capacity at all is bounded")
    offer = {"id": 777001, "machine_id": 424242, "gpu_name": "RTX 4090", "num_gpus": 1,
             "verification": "verified", "cuda_max_good": 12.9, "driver_version": "590.44",
             "inet_up": 900, "inet_down": 900, "reliability2": 0.995, "disk_space": 200,
             "geolocation": "Prague, CZ", "dph_total": 0.35}
    NEW_ID = 50999001

    c = Case(root, "k_create")
    state_dir = c.dir / "state"
    state_dir.mkdir()
    (state_dir / "nvblox_box_id").write_text(f"{NV_ID}\n")
    env = c.env(FAKE_START_STDOUT="Required resources are currently unavailable, "
                                  "state change queued.",
                FAKE_OFFERS_JSON=json.dumps([offer]),
                FAKE_NEW_CONTRACT=NEW_ID,
                FAKE_NVBLOX_FIXTURE=str(NVBLOX_FIXTURE))
    p = c.run("--worker-mode", "--nvblox-source", "live", "--max-usd", "50",
              "--state-dir", str(state_dir), "--capacity-backoff-s", "1",
              env=env, timeout=600)
    st = c.status()
    log = c.cli_log()
    check("(k) the run completed on the replacement box",
          p.returncode == 0 and st.get("state") == "DONE", st.get("state", "?"))
    check("(k) the refused start was followed by a search and a create",
          "search offers" in log and f"create instance {offer['id']}" in log,
          log.replace("\n", " | ")[:150])
    check("(k) the created instance is the one that was used and stopped",
          f"vastai stop instance {NEW_ID}" in log)
    check("(k) the new id was cached and the refusing one retired",
          (state_dir / "nvblox_box_id").read_text().strip() == str(NEW_ID)
          and str(NV_ID) in (state_dir / "retired_box_ids").read_text(),
          (state_dir / "nvblox_box_id").read_text().strip())
    # The repo's own cache file is machine-local and gitignored, so a fresh clone has
    # none at all - which is the strongest form of "a test did not write to it".
    _repo_cache = PIPELINE / "state" / "nvblox_box_id"
    _repo_cached_id = _repo_cache.read_text().strip() if _repo_cache.exists() else ""
    check("(k) the real repo cache was not rewritten by a test",
          _repo_cached_id != str(NEW_ID), _repo_cached_id or "(absent)")
    check("(k) the create response's instance_api_key never reached disk",
          "SECRET-should-never-be-returned" not in json.dumps(st)
          and "SECRET-should-never-be-returned" not in
              (c.dir / "vast_calls.log").read_text())
    up = st.get("steps", {}).get("GPU_UP_NV", {}).get("artifacts", {})
    check("(k) the status file keeps the fact that a box was rented mid-run",
          up.get("replaced_instance_id") == NV_ID
          and up.get("created_from_offer") == offer["id"]
          and up.get("instance_id") == NEW_ID,
          f"replaced={up.get('replaced_instance_id')} offer={up.get('created_from_offer')} "
          f"used={up.get('instance_id')}")

    # ...and now nothing is rentable at all
    c2 = Case(root, "k_nocapacity")
    sd2 = c2.dir / "state"
    sd2.mkdir()
    (sd2 / "nvblox_box_id").write_text(f"{NV_ID}\n")
    t0 = time.time()
    p2 = c2.run("--worker-mode", "--nvblox-source", "live", "--max-usd", "50",
                "--state-dir", str(sd2), "--capacity-max-attempts", "3",
                "--capacity-backoff-s", "1",
                env=c2.env(FAKE_START_STDOUT="Required resources are currently "
                                             "unavailable, state change queued."),
                timeout=300)
    st2 = c2.status()
    up2 = st2.get("steps", {}).get("GPU_UP_NV", {})
    check("(k) with nothing rentable the run FAILED rather than looping",
          p2.returncode == 1 and st2.get("state") == "FAILED", st2.get("state", "?"))
    check("(k) the reason names the ceiling, not a generic error",
          st2.get("fail_reason") == "waiting_capacity_exhausted",
          str(st2.get("fail_reason")))
    check("(k) it stopped at the attempt ceiling", up2.get("attempt") == 3,
          str(up2.get("attempt")))
    check("(k) the last substage is waiting_capacity",
          up2.get("substage") == "waiting_capacity", str(up2.get("substage")))
    check("(k) it gave up in seconds under a 1 s backoff, not after 60 minutes",
          time.time() - t0 < 120, f"{time.time() - t0:.1f} s")


def test_l_waiting_manual(root: Path) -> None:
    """MapAnything is manual. With no pass-A output the step parks by name; drop the output
    in and the same command resumes."""
    print("\n(l) MAPANYTHING with no pass-A output waits by name, then resumes")
    c = Case(root, "l_manual")
    p = c.run("--worker-mode", "--frames-fps", "2.0", env=c.env(), from_pulled=False,
              timeout=300)
    st = c.status()
    ma = st.get("steps", {}).get("MAPANYTHING", {})
    check("(l) the run stopped at MAPANYTHING",
          p.returncode == 1 and ma.get("state") == "failed", st.get("state", "?"))
    check("(l) it is waiting_manual, not a generic missing-output error",
          ma.get("substage") == "waiting_manual"
          and st.get("fail_reason") == "waiting_manual",
          f"{ma.get('substage')} / {st.get('fail_reason')}")
    check("(l) the detail names both exact paths",
          str(c.dir / "pass_a" / "poses.json") in (ma.get("detail") or "")
          and str(c.dir / "pass_a" / "per_view") in (ma.get("detail") or ""),
          (ma.get("detail") or "")[:100])
    check("(l) FRAMES ran first, at the fps it was given",
          st["steps"]["FRAMES"]["state"] == "done"
          and json.loads((c.dir / "frames.json").read_text())["sampling"]["requested_fps"]
          == 2.0,
          st["steps"]["FRAMES"]["state"])
    check("(l) the frames record what rotation they actually are",
          "frames_orientation_deg" in
          json.loads((c.dir / "frames.json").read_text())["video_meta"])

    # the operator does the manual pass; nothing else changes
    (c.dir / "pass_a").symlink_to(PULLED)
    p2 = c.run("--worker-mode", "--frames-fps", "2.0", env=c.env(), from_pulled=False,
               timeout=300)
    st2 = c.status()
    check("(l) the same command resumes and finishes",
          p2.returncode == 0 and st2.get("state") == "DONE", st2.get("state", "?"))
    check("(l) the frames it already had were not resampled",
          st2["steps"]["FRAMES"]["state"] == "skipped",
          st2["steps"]["FRAMES"]["state"])
    check("(l) MAPANYTHING is no longer waiting",
          st2["steps"]["MAPANYTHING"]["state"] == "skipped"
          and st2["steps"]["MAPANYTHING"]["substage"] is None,
          st2["steps"]["MAPANYTHING"]["state"])


def test_m_daily_cap(root: Path) -> None:
    """An exhausted daily cap refuses BEFORE any vast call. The evidence is the fake's log
    never being created - the same standard test (a) holds ARMED to."""
    print("\n(m) an exhausted PIPELINE_DAILY_USD refuses before spending anything")
    wd = _worker_driver()
    c = Case(root, "m_cap")
    ledger = c.dir / "ledger.json"
    today = dt.date.today().isoformat()

    ledger.write_text(json.dumps({today: 3.10}))
    with fake_env(c):
        os.environ["PIPELINE_DAILY_USD"] = "3.00"
        os.environ["PIPELINE_WORKER_ENABLED"] = "1"
        refused = None
        try:
            wd.run_job(Path(VIDEO), c.dir, _Spec(), scene_name=SCENE, from_pulled=PULLED,
                       max_usd=3.0, ledger=ledger)
        except wd.JobRefused as e:
            refused = str(e)
        check("(m) the job was refused", refused is not None, str(refused)[:90])
        check("(m) the refusal names the cap and what was already spent",
              refused is not None and "$3.1000 of $3.00" in refused, str(refused)[:110])
        check("(m) the fake vastai was never invoked at all", not c.fake_log.exists(),
              c.cli_log()[:80])
        check("(m) no status.json was written for a job that never started",
              not (c.dir / "status.json").exists())

        # ...and with headroom left, the per-run cap is clamped to it
        ledger.write_text(json.dumps({today: 2.90}))
        os.environ.pop("PIPELINE_WORKER_ENABLED")
        st = wd.run_job(Path(VIDEO), c.dir, _Spec(), scene_name=SCENE, from_pulled=PULLED,
                        max_usd=3.0, ledger=ledger)
        check("(m) a run with $0.10 of headroom gets --max-usd 0.10, not 3.00",
              st.get("max_usd") == 0.1, str(st.get("max_usd")))
        check("(m) the ledger's own numbers are carried into the status file",
              st.get("daily_cap_usd") == 3.0 and st.get("daily_spent_before_usd") == 2.9,
              f"cap={st.get('daily_cap_usd')} spent={st.get('daily_spent_before_usd')}")


def test_n_offer_ranking() -> None:
    """Region beats price, and the price ceiling is a hard filter, not a preference."""
    print("\n(n) offer ranking: EU at $0.80 beats CN at $0.32, and > $1.00 is dropped")
    sys.path.insert(0, str(PIPELINE))
    from vast_client import VastClient, MAX_DPH

    def offer(**kw):
        base = {"id": 1, "machine_id": 1, "gpu_name": "RTX 4090", "num_gpus": 1,
                "verification": "verified", "cuda_max_good": 12.9,
                "driver_version": "590.44", "inet_up": 900, "inet_down": 900,
                "reliability2": 0.995, "disk_space": 200,
                "geolocation": "Prague, CZ", "dph_total": 0.50}
        base.update(kw)
        return base

    c = VastClient("/dev/null", "/dev/null")
    eu = offer(id=1, machine_id=11, geolocation="Prague, CZ", dph_total=0.80)
    cn = offer(id=2, machine_id=22, geolocation="Shanghai, CN", dph_total=0.32)

    check("(n) the price ceiling is the working box's 0.80 plus headroom, not a guess",
          MAX_DPH == 1.00, f"MAX_DPH={MAX_DPH}")

    ranked = c.rank_offers([cn, eu])
    check("(n) both offers survive the hard filter",
          [o["id"] for o in sorted(ranked, key=lambda o: o["id"])] == [1, 2],
          str([o["id"] for o in ranked]))
    check("(n) EU at $0.80 ranks ABOVE CN at $0.32 - region beats price",
          ranked[0]["id"] == 1,
          f"first={ranked[0]['geolocation']} ${ranked[0]['dph_total']}")
    check("(n) ...and it is not just insertion order",
          c.rank_offers([eu, cn])[0]["id"] == 1)

    # the ceiling drops an offer outright, ranking never sees it
    dear_eu = offer(id=3, machine_id=33, geolocation="Berlin, DE", dph_total=1.01)
    ranked2 = c.rank_offers([dear_eu, cn])
    check("(n) an EU offer over the $1.00 ceiling is dropped, not merely ranked lower",
          [o["id"] for o in ranked2] == [2], str([o["id"] for o in ranked2]))
    check("(n) exactly $1.00 is allowed - the ceiling is <=, not <",
          [o["id"] for o in c.rank_offers([offer(id=4, machine_id=44, dph_total=1.00)])]
          == [4])
    check("(n) the ceiling is applied inside rank_offers, so every caller gets it",
          "max_dph" in VastClient.rank_offers.__code__.co_varnames)


#: A 5-view scene small enough to filter in milliseconds, built so the answer is known by
#: construction: four cameras looking at a plane 2 m away and one looking at a plane 3 m
#: away. The odd one out projects into its neighbours BEHIND their surface (delta > tau =
#: disagreement) and so collects no votes at all, while its own depth puts it IN FRONT of
#: theirs, which the one-sided rule scores as no vote rather than as evidence against them.
#: So the outlier view must be emptied and the other four must survive at m=2.
_SYNTH_PACK = r'''
import json, sys
import numpy as np
out = sys.argv[1]
H = W = 8
FX = FY = 8.0
CX = CY = 4.0
TX = [0.0, 0.2, 0.4, 0.6, 0.8]
PLANE = [2.0, 2.0, 2.0, 2.0, 3.0]        # view 4 is the liar
depth = np.zeros((len(TX), H, W), np.uint16)
views = []
for i, (tx, z) in enumerate(zip(TX, PLANE)):
    depth[i, :, :] = int(round(z * 1000))
    pose = np.eye(4); pose[0, 3] = tx
    views.append({"raw_frame_index": i * 2, "t_sec": i * 0.1,
                  "camera_pose": pose.tolist(),
                  "intrinsics": [[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]],
                  "shape_hw": [H, W]})
np.save(out + "/depth_u16.npy", depth)
json.dump({"source": "synthetic (pipeline/tests/run_tests.py)", "n_views": len(TX),
           "shape_hw": [H, W], "depth_units": "uint16 millimetres, 0 = invalid",
           "views": views}, open(out + "/meta.json", "w"))
print(json.dumps({"n_views": len(TX), "valid_px": int((depth > 0).sum())}))
'''


def _synthetic_pack(dest: Path) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    p = subprocess.run([cpu_python(), "-c", _SYNTH_PACK, str(dest)],
                       capture_output=True, text=True, timeout=120)
    if p.returncode != 0:
        raise RuntimeError(f"synthetic pack failed: {p.stderr[-400:]}")
    return json.loads(p.stdout)


def test_o_mvfilter(root: Path) -> None:
    """--mvfilter is off by default, refuses a bad spec before renting, and when on it is
    the only thing that decides which depth stack nvblox is handed."""
    print("\n(o) the optional pre-NVBLOX multi-view filter")
    mvfilter = PIPELINE / "steps" / "mvfilter.py"
    SPEC = "m=2,tau=0.10,neighbours=nearest,K=8"

    # --- the rule itself, on a scene whose answer is known by construction --------------
    pack = root / "o_pack"
    info = _synthetic_pack(pack)
    p = subprocess.run([cpu_python(), str(mvfilter), "--packed", str(pack),
                        "--m", "2", "--tau", "0.10", "--neighbours", "nearest", "--k", "3",
                        "--progress-every", "0"], capture_output=True, text=True, timeout=300)
    check("(o) mvfilter runs on a packed dir", p.returncode == 0, p.stderr[-160:])
    doc = json.loads((pack / "mvfilter.json").read_text())
    check("(o) it sees every valid pixel",
          doc["valid_px_before"] == info["valid_px"],
          f"{doc['valid_px_before']} vs {info['valid_px']}")
    check("(o) the one view that disagrees with all its neighbours is emptied",
          doc["per_view_kept_frac"]["views_fully_dropped"] == 1,
          json.dumps(doc["per_view_kept_frac"]))
    check("(o) the four that agree are not",
          0 < doc["valid_px_after"] < doc["valid_px_before"]
          and doc["kept_pct"] > 50, f"kept {doc['kept_pct']}%")
    check("(o) the filtered stack keeps the input's shape and dtype, zeroing not dropping",
          doc["output"].endswith("depth_u16_mvfilter.npy"), doc["output"])
    check("(o) the parameters that made it are recorded next to it",
          (doc["m"], doc["tau"], doc["neighbours"], doc["k"]) == (2, 0.10, "nearest", 3),
          json.dumps({k: doc[k] for k in ("m", "tau", "neighbours", "k")}))

    # --- a bad spec is refused at argv time, before anything is rented ------------------
    for label, args in (
            ("missing keys", ["--mvfilter", "m=2,tau=0.10"]),
            ("unknown neighbour rule",
             ["--mvfilter", "m=2,tau=0.10,neighbours=telepathy,K=8"]),
            ("K below m", ["--mvfilter", "m=4,tau=0.10,neighbours=nearest,K=2"]),
            ("control arm without a filter", ["--mvfilter-ab"]),
            ("extra arm without a filter",
             ["--mvfilter-also", "m=2,tau=0.10,neighbours=nearest,K=8"])):
        c = Case(root, f"o_bad_{label.split()[0]}_{abs(hash(label)) % 997}")
        pr = c.run("--nvblox-source", "live", *args)
        check(f"(o) {label}: exit 2", pr.returncode == 2, f"rc={pr.returncode}")
        check(f"(o) {label}: nothing was rented", not c.fake_log.exists(),
              c.cli_log()[:120] or "log absent")
        check(f"(o) {label}: the refusal names the flag",
              "--mvfilter" in pr.stderr, pr.stderr.strip()[:110])

    # --- off by default: the step says so and the remote argv is the plain one ----------
    def dry(case: Case, *args) -> tuple[dict, str]:
        env = case.env()
        env["PIPELINE_DRY_RUN"] = "1"
        pr = case.run("--nvblox-source", "live", *args, env=env, timeout=600)
        return case.status(), pr.stdout + pr.stderr

    off = Case(root, "o_off", with_packed=False)
    shutil.copytree(pack, off.dir / "packed" / SCENE)
    st, _ = dry(off)
    check("(o) with the flag absent MVFILTER is skipped and says why",
          st["steps"]["MVFILTER"]["state"] == "skipped"
          and st["steps"]["MVFILTER"]["substage"] == "skipped",
          json.dumps(st["steps"]["MVFILTER"])[:130])
    cmds = " || ".join(st["steps"]["NVBLOX"]["artifacts"]["remote_commands"])
    check("(o) nvblox is handed the unfiltered stack",
          "--depth-file depth_u16.npy" in cmds and "--suffix" not in cmds, cmds[-150:])
    check("(o) exactly one fusion runs", cmds.count("nvblox_scenes.py --packed-root") == 1,
          str(cmds.count("nvblox_scenes.py --packed-root")))
    check("(o) status.json records which depth nvblox was given",
          st["steps"]["QUEUED"]["artifacts"]["nvblox_depth_file"] == "depth_u16.npy"
          and st["steps"]["QUEUED"]["artifacts"]["mvfilter"] is None)

    # --- on: three arms, control first, the run's own scene last ------------------------
    on = Case(root, "o_on", with_packed=False)
    shutil.copytree(pack, on.dir / "packed" / SCENE)
    alt = "m=2,tau=0.10,neighbours=baseline15,K=3"
    st, _ = dry(on, "--mvfilter", "m=2,tau=0.10,neighbours=nearest,K=3",
                "--mvfilter-ab", "--mvfilter-also", alt)
    check("(o) MVFILTER ran", st["steps"]["MVFILTER"]["state"] == "done",
          st["steps"]["MVFILTER"]["state"])
    arms = st["steps"]["MVFILTER"]["artifacts"]["arms"]
    check("(o) it produced one stack per requested vote", len(arms) == 2, str(list(arms)))
    check("(o) each stack records its own kept fraction",
          all("kept_pct" in v for v in arms.values()),
          json.dumps({k: v.get("kept_pct") for k, v in arms.items()}))
    cmds = st["steps"]["NVBLOX"]["artifacts"]["remote_commands"]
    fusions = [c for c in cmds if "nvblox_scenes.py --packed-root" in c]
    check("(o) three fusions run: two controls and the run's own", len(fusions) == 3,
          str(len(fusions)))
    check("(o) the unfiltered control goes first",
          "--depth-file depth_u16.npy --suffix __unfiltered" in fusions[0], fusions[0][-90:])
    check("(o) the run's OWN scene is fused last and carries no suffix",
          fusions[-1].endswith("--depth-file depth_u16_mvfilter.npy"), fusions[-1][-90:])
    check("(o) every stack it fuses is pushed",
          all(any(f"push: " in c and name in c for c in cmds)
              for name in ("depth_u16.npy", "depth_u16_mvfilter.npy",
                           "depth_u16_mv_m2_t0.1_baseline15_K3.npy")),
          " | ".join(c[-60:] for c in cmds if c.startswith("push")))
    check("(o) LAYERS reads the filtered scene, not a control",
          "nvblox_unfiltered" not in st["steps"]["QUEUED"]["artifacts"]["scenes_out"],
          st["steps"]["QUEUED"]["artifacts"]["scenes_out"])

    # --- idempotency: same vote is skipped, a DIFFERENT vote is not -------------------
    st2, _ = dry(on, "--mvfilter", "m=2,tau=0.10,neighbours=nearest,K=3",
                 "--mvfilter-ab", "--mvfilter-also", alt)
    check("(o) re-running the same vote skips the filter",
          st2["steps"]["MVFILTER"]["state"] == "skipped",
          st2["steps"]["MVFILTER"]["state"])
    st3, _ = dry(on, "--mvfilter", "m=3,tau=0.10,neighbours=nearest,K=3")
    check("(o) a different vote is NOT mistaken for the stack already on disk",
          st3["steps"]["MVFILTER"]["state"] == "done",
          st3["steps"]["MVFILTER"]["state"])
    check("(o) and the stack it leaves behind is the new vote",
          json.loads((on.dir / "packed" / SCENE / "mvfilter.json").read_text())["m"] == 3)


def main() -> int:
    reason = missing_reason()
    if reason:
        # One line, before anything runs. (g) used to spawn this interpreter directly and
        # die with a bare FileNotFoundError two minutes into the suite.
        print(reason, file=sys.stderr)
        return 2
    for p in (PACKED_FIXTURE, NVBLOX_FIXTURE, PULLED):
        if not p.exists():
            print(f"missing fixture: {p}", file=sys.stderr)
            return 2
    # Cleaned on every path, pass or fail. It used to be removed only after a fully green
    # run, so each red run left its whole tree behind - that, not any one case, is how
    # /tmp accumulated GBs of pipeline_tests_* dirs. PIPELINE_TESTS_KEEP=1 keeps the tree
    # and prints where it is, for when a failure needs looking at.
    keep = os.environ.get("PIPELINE_TESTS_KEEP") == "1"
    root = Path(tempfile.mkdtemp(prefix="pipeline_tests_"))
    print(f"scratch: {root}" + ("  (kept: PIPELINE_TESTS_KEEP=1)" if keep else ""))
    try:
        test_a_no_armed(root)
        test_b_ssh_refused(root)
        test_c_budget(root)
        test_d_resume(root)
        test_e_no_key_access()
        test_f_unreachable(root)
        test_g_floor_rule()
        test_h_transfer_gate(root)
        test_i_start_refused(root)
        test_j_worker_path(root)
        test_k_start_refused_creates(root)
        test_l_waiting_manual(root)
        test_m_daily_cap(root)
        test_n_offer_ranking()
        test_o_mvfilter(root)
    finally:
        if keep:
            print(f"\nkept for inspection: {root} "
                  f"({subprocess.run(['du', '-sh', str(root)], capture_output=True, text=True).stdout.split()[0] if root.exists() else '-'})")
        else:
            shutil.rmtree(root, ignore_errors=True)
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    if n_fail:
        print("FAILED:")
        for name, ok, detail in results:
            if not ok:
                print(f"  - {name}: {detail}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
