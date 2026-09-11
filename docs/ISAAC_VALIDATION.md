# Isaac Sim validation

> **Status: UNVERIFIED.** Nothing on this page has been executed against a real Isaac
> Sim install. Isaac Sim cannot run on the CloudEye VPS (no NVIDIA GPU), and it is not
> yet installed on the rented `gpu` box either. Every command below is derived from
> NVIDIA's published Isaac Sim 6.0.1 documentation, not from a run. The "Verified runs"
> table at the bottom is empty; until it has a row, treat this document as a plan, not
> a report. See `tools/isaac_validate.py`'s own header for the same caveat applied to
> the script itself.

## What this validates, and what it doesn't

`tools/isaac_validate.py` opens a scene `.usd` exported by
`app/services/usd_export.py` inside a real Isaac Sim instance and checks:

1. The stage opens and its conventions match what `usd_export.py` promises (Z-up,
   `metersPerUnit=1.0`, `/World` as the default prim).
2. PhysX cooks the whole stage without errors.
3. Every `/World/Structure/*` and `/World/Objects/*` prim produces a **real, queryable
   PhysX collision shape** at its authored world location - not merely that the
   `UsdPhysics.CollisionAPI` schema is applied (a degenerate convex hull can fail
   PhysX's cook silently, leaving the schema applied with no actual shape).
4. A test sphere dropped from 2m onto `/World/Floor` comes to rest there.
5. No **static** prim (`/World/Floor`, `/World/Structure/*`) ends up below the floor.
   Dynamic-object (`/World/Objects/*`) displacement is measured and reported
   separately - see "Dynamics phase" below, it is diagnostic, not pass/fail by default.
6. Zero-physics structural sanity: zero/near-zero object mass (a real exporter defect
   path - a degenerate bbox fallback can compute `volume_m3 == 0`), non-positive cube
   scale, and (only under `--gpu-dynamics`) convex hull vertex/face counts against
   PhysX's GPU-dynamics convex-collider limits.

**What it does not validate**: robot navigation, articulated robots, friction/contact
materials, rendering/visual correctness, or the point cloud (`/World/PointCloud` is
`purpose=guide` and deliberately has no collider - see `usd_export.py`'s module
docstring for why).

## Prerequisites

Isaac Sim 6.0.1 (early developer preview - Isaac Lab requires 6.0.x, 6.0.1 is current)
needs: Python 3.12, Ubuntu 22.04+, an RTX GPU with RT cores, and >=16GB VRAM. The
rented `gpu` box (Ubuntu 24.04, RTX 4090 24GB, driver 580.126.09) qualifies. The
CloudEye VPS does not - it has no NVIDIA GPU at all.

Installed as the **standalone binary** (not pip), per
`deploy/install_isaacsim_gpu.sh`: no venv, the bundled `python.sh`/`isaac-sim.sh` are
used directly. Requires ~80GB free on the box's mounted persistent volume (13GB
download + extracted install + caches) - the script verifies this is a real mounted
volume (via `findmnt`), separate from the container's root disk, before writing
anything.

`usd-core==26.8` is pinned in this repo's `pyproject.toml` and is what
`usd_export.py` writes with (`Usd.Stage.Export`, binary crate by default). See "Crate
compatibility" below for what to do if Isaac's bundled USD rejects that crate version.

## Step 1 - produce the USD

```bash
curl -o scene.usd 'http://127.0.0.1:8000/api/scenes/<scene_id>/usd?include_pointcloud=false&include_fragments=false'
```

`include_pointcloud=false` is recommended for validation: the point cloud has no
collider by design, so it only adds file size and stage-open time here.

## Step 2 - copy the script and the USD to the GPU box

```bash
# tools/ is NOT synced by SshGpuClient.ensure_pipeline_code() - that rsyncs gpu/ to
# /workspace/pipeline/ WITH --delete before every job. Putting anything else there
# would get silently erased by the next pipeline run. Use a separate location.
ssh gpu 'mkdir -p /workspace/tools /workspace/validate'
rsync -avz tools/isaac_validate.py gpu:/workspace/tools/
rsync -avz scene.usd gpu:/workspace/validate/
```

## Step 3 - install Isaac Sim 6.0.1 on the GPU box

```bash
ssh gpu 'mkdir -p /workspace/deploy'
rsync -avz deploy/install_isaacsim_gpu.sh gpu:/workspace/deploy/
ssh gpu 'bash /workspace/deploy/install_isaacsim_gpu.sh'
```

Downloads the standalone binary zip directly onto the GPU box's persistent volume
(default `/workspace`, resolved via `findmnt` - aborts rather than ever writing to the
container's root disk), extracts it to `$VOLUME/isaacsim/`, runs its `post_install.sh`,
symlinks `~/.cache/ov`, `~/.local/share/ov`, and `~/.nvidia-omniverse` into
`$VOLUME/isaac-cache/` (so the shader/extension cache survives an instance recycle,
migrating any existing content rather than discarding it), and writes
`$VOLUME/env.sh` with `ISAACSIM_PATH`/`ISAACSIM_PYTHON_EXE`. Idempotent by default -
pass `--reinstall` to force a fresh install (the only path this script ever deletes is
`$VOLUME/isaacsim/`, and only with that flag). Follows the same "unverified, run it
yourself, report back what actually happened" standard as this page - it is meant to be
run manually on the GPU box when you're ready, not by an automated agent.

## Step 4 - run it

```bash
ssh gpu 'source /workspace/env.sh && cd /workspace && \
  "$ISAACSIM_PYTHON_EXE" tools/isaac_validate.py validate/scene.usd \
  --json-out validate/report.json --log-out validate/kit.log'
```

Headless launch of Isaac Sim itself (e.g. for the code-editor server, or manual
poking) uses the bundled launcher directly, no venv:

```bash
"$ISAACSIM_PATH/isaac-sim.sh" --no-window --enable isaacsim.code_editor.python_server
```

The one part runnable **today, on the VPS, with no Isaac Sim at all** - the pure-`pxr`
structural half:

```bash
uv run python tools/isaac_validate.py --usd-only scene.usd --json-out report.json
```

This is the only invocation actually exercised so far (against a fresh export of
`tests/fixtures/scene_horizontal` and a deliberately degenerate synthetic object) -
see "Verified runs" below.

## Step 5 - interpreting the JSON

Every check lands in `checks.<name>` as `{"status": "pass"|"fail"|"warn"|"skipped",
"required": bool, "detail": ...}`. `status` at the top level rolls all required checks
up into `pass`/`fail`/`error`. `failures`/`warnings` at the top level are the
human-readable one-liners.

### Triage table - keep these apart

| Symptom in the JSON | Class | What to do |
|---|---|---|
| `collider_coverage.missing` entries with `collision_source: "convex_hull"` | **exporter bug** | tighten `build_convex_hull`'s degeneracy handling / the bbox fallback in `_add_object_geometry` |
| `mass_sanity.zero_mass` non-empty | **exporter bug** | `_add_object_geometry` computed `volume_m3 == 0` for a flat/degenerate bbox |
| `scale_sanity.bad_scale` non-empty | **exporter bug** | a malformed bbox (min > max on some axis) produced a non-positive cube scale |
| `no_static_prim_below_floor` fails | **exporter bug** | floor/structure geometry itself is wrong |
| `floor_objects_footprint_overlap` fails | **exporter bug** | `/World/Floor`'s world bbox doesn't overlap `/World/Objects/*`'s in XY - check `_add_cube`'s xformOp order (a past regression: `AddScaleOp()` before `AddTranslateOp()` silently scaled every cube's world position by its own size) |
| `sphere_drop.verdict == "tunneled"` | **sim config**, not a missing collider | the floor is only 2cm thick and a 2m fall reaches ~6.3 m/s; lower `--physics-dt`, check `environment.feature_detection.PhysxSchema` (CCD) |
| `sphere_drop.verdict == "never_fell"` | **sim config** | no physics scene / zero gravity / body disabled - see `checks.physics_scene` |
| `collider_coverage` / `cooking` missing on *everything* | **API mismatch for this build** | check `environment.feature_detection` and `fatal_environment_error` - not a scene problem |
| `dynamics.classification_counts.ejected > 0`, `--fail-on-ejected` not set | **expected, over-constrained scene** | objects captured against walls overlap `/World/Structure` at t=0 by construction; PhysX depenetrating them is not an exporter defect |
| `physx_log_clean` status `skipped` | the `/log/file` carb setting didn't take effect | the functional `collider_coverage` check remains the real safety net regardless |

## Crate compatibility

If `stage_opened` fails on the Isaac side but `--usd-only` opened the same file fine,
Isaac's bundled USD likely rejects the crate version `usd-core==26.8` wrote (this
script sniffs and reports `input.format`/`input.crate_version` from the file's own
header specifically so this is named, not guessed from an opaque Tf error).

**Mitigation order** (cheapest, lowest-risk first):

1. **Convert to ASCII (`.usda`) and retry** - no exporter change needed, no risk to
   the rest of the backend (which depends on `usd-core==26.8` for the `/usd` endpoint
   itself):
   ```bash
   uv run python -c "from pxr import Usd; Usd.Stage.Open('scene.usd').GetRootLayer().Export('scene.usda')"
   ```
2. If ASCII works, decide whether to make it routine (an `--ascii` flag on the
   exporter, or `?format=usda` on the endpoint) versus staying ad-hoc for validation only.
3. If ASCII **also** fails, this is a schema/version incompatibility, not a crate
   version gate - only then consider whether `usd-core`'s pin needs to move (a bigger
   change, touching the whole backend, not just this script).
4. Last resort: round-trip the file through Isaac's own USD build on the GPU box to
   produce an Isaac-native crate.

## Known limitations of the validator itself

- No robot, no articulations, no friction/contact materials are modeled.
- Convex hulls are conservative (over-approximating) by design - see
  `usd_export.py`'s module docstring; this is the intended, safe-for-navigation
  direction of error, not something this script flags.
- `_add_hull_mesh` writes hull vertices in world space with **no xform op** - a hull
  object's rigid-body origin sits at `(0,0,0)`, not at its geometry's location. This is
  a real quirk of the current exporter (noted, not fixed, by this validation pass) -
  the dynamics phase's displacement measurement uses world-space Z directly rather than
  a local-origin delta for exactly this reason.
- The PhysX log capture is best-effort (see `LogWatch` in `tools/isaac_validate.py`)
  and degrades to `skipped` rather than a false pass if `/log/file` doesn't take effect
  on a given build.

## Verified runs

**This table is empty on purpose.** No run of `tools/isaac_validate.py` against a real
Isaac Sim install has ever happened. Add a row on the first real execution - including
failures, especially the failures - before deleting the UNVERIFIED banner at the top of
this file, and only then flip `pyproject.toml`'s `isaac_sim_status` from
`"targeted-untested"` to `"tested"`. Then append a decision entry to `docs/DECISIONS.md`.

| Date | Isaac Sim version | Install method | Scene | Exit code | Report | Notes |
|---|---|---|---|---|---|---|
| _(none yet)_ | | | | | | |
