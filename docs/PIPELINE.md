# PIPELINE — what the first live run needs

Written 2026-09-08 after the dry run recorded in `var/scratch/pipeline_dry/hero/`.
Every number below is measured unless the line says "estimate" or "assumption".

The dry run proved the state machine, the idempotency and the failure path. It did **not**
prove any GPU step: this VPS has no GPU (`nvidia-smi` absent, `import nvblox_torch` →
`ModuleNotFoundError`), so `NVBLOX` and `FLOOR` were served from the already-computed
`nvblox_v2/results/scenes_out/own_0901_173903__step2/`. Tomorrow is the first time those two
run under the driver.

---

## 0. The thing that decides the shape of tomorrow: it is two boxes, not one

`MapAnything` at 1148 views needs **81 716 MiB** of VRAM (measured; the fitted rule is
`allocated_MiB = 4704.7 + 67.0835 * n_views`, and `nvidia-smi` reads ~2853 MiB above it).
A 24 GB RTX 4090 tops out at **238 views**. So:

| stage | box | why |
|---|---|---|
| FRAMES, MAPANYTHING | a **B200** rented fresh (~$7.8/hr) | only box measured to hold 1148 views (capacity 2605) |
| NVBLOX, FLOOR | **50265868**, the existing RTX 4090 (~$0.369/hr) | `/workspace/venvs/nb` + `/workspace/nvblox_v2/` already installed there and survive a stop; nvblox needs 16.4 MiB of VRAM, so the GPU size is irrelevant |
| PACK, LAYERS, REACH, EXPORT | this VPS, `$PIPELINE_CPU_PYTHON` (default `~/venvs/o3d-cpu`) | CPU-only, measured today: PACK 29.1 s, LAYERS 2.6 s, REACH 9.9 s, EXPORT 14.7 s |

Running MapAnything on 50265868 instead is possible only by dropping to ≤238 views, which
changes the reconstruction — not a free substitution. Treat the two-box shape as the plan.

### 0.1 The CPU venv — where it lives and how to rebuild it

The four CPU stages do **not** run under this repo's `.venv`: they need `open3d`, and EXPORT
needs `pxr` (`usd-core`), neither of which the API/worker venv carries. They run under a
separate CPU-only venv, resolved in exactly one place — `pipeline/cpu_env.py`.

| | |
|---|---|
| env var | `PIPELINE_CPU_PYTHON` |
| default | `~/venvs/o3d-cpu/bin/python` |
| **not** | `/tmp/anything` |

It used to be hardcoded as `/tmp/o3d-venv/bin/python` in six places. `/tmp` is not a home
for a build dependency: `systemd-tmpfiles-clean` reclaims it on a timer, and on
**2026-09-09 it was deleted by hand** after `lsof` showed no open files — the wrong test
for "unused", since nothing holds a venv open between runs. Every CPU stage then died with
a bare `FileNotFoundError` naming a path, with nothing saying what the path was for. Both
`run_pipeline.py` and `pipeline/tests/run_tests.py` now check it **before anything starts**
and print one line naming the path, the env var and the rebuild command.

`run_pipeline.py` checks it before `QUEUED` specifically so that a missing venv cannot cost
a box: PACK is the first stage that needs it and it sits *after* `GPU_UP_NV` on the live
path, so discovering it there means renting, billing and discarding a box for a reason that
was knowable at argv-parse time.

Rebuild:

```bash
uv venv ~/venvs/o3d-cpu --python 3.12
uv pip install --python ~/venvs/o3d-cpu/bin/python \
    numpy==2.5.2 scipy==1.18.1 open3d==0.19.0 matplotlib usd-core trimesh
```

`usd-core` and `trimesh` are not optional: `isaac_export.py:30` imports `pxr`, and a rebuild
without it reaches EXPORT and dies there — measured on 2026-09-09, when a rebuild missing
`usd-core` failed tests (j), (k) and (l) at EXPORT with `ModuleNotFoundError: No module
named 'pxr'` while every earlier stage passed.

---

## 1. Instance 50265868 — by ID, always

**Every state-changing call names this one id. Never a filter, never a loop over
`vastai show instances`** — a parallel session's box can appear in a listing mid-cleanup, and
that rule exists because it nearly took one.

```bash
vastai show instance 50265868          # read
vastai start instance 50265868         # bring up
vastai stop instance 50265868          # put down — /workspace survives a stop
yes | vastai destroy instance 50265868 # NEVER for this box while nvblox lives on it
```

- `destroy` prompts interactively; without `yes |` it silently aborts.
- **50265868 is stop-only in practice.** Destroying it throws away the working nvblox install
  (`nvblox_torch 0.0.10 + torch 2.9.1+cu128`, wheel-pinned, no rebuild recipe in this repo).
  The rented B200 is the one that gets destroyed.
- The stub in `pipeline/vast_stub.py` can only express id-based calls — there is no method
  that takes a filter. Keep it that way when it becomes a real client.
- Connect with `nvblox_v2/nb` (proxy + `aes128-gcm`, ~13x faster than the default cipher).
  It deliberately does not touch `~/.ssh/config`, which a parallel session owns.

### Boxes in play

| id | label | role | note |
|---|---|---|---|
| **50381086** | **nvblox-v3** | **the box that completed a green run** | created 2026-09-09 from offer 30055208 (machine 49612, United Kingdom, $0.8222/hr, driver 590.44.01, cuda_max_good 13.1). Provisioned by `pipeline/box/onstart_nvblox.sh` in **4 min 12 s**; measured **4.38 MB/s up / 3.46 MB/s down**, ten times the old box. Ran `own_0901_161054` end to end in 265 s. Pass it with `--nvblox-instance-id 50381086`. |
| ~~50376970~~ | nvblox-v3 | destroyed | machine 109808, Taiwan. Container never started - CDI failure, see 4.4f. Destroyed by id the same day. |
| 50265868 | nvblox-v2 | **RETIRED, pending the owner's `destroy`** | its host refused `start` twice on 2026-09-09 with "Required resources are currently unavailable" - see 4.5. Listed in `pipeline/state/retired_box_ids`. Untouched: still stopped, still carrying the original `/workspace/venvs/nb`. **Nothing retires itself and nothing is destroyed automatically** - `destroy` is denied to the agent in `.claude/settings.json`. |

**Which box a run uses is no longer a constant.** `pipeline/state/nvblox_box_id` holds the
id to try first - currently **50381086** - and `VastClient.start_or_create()` reads it,
starts that box, and only if its host refuses (today's markers) searches, ranks
EU > US > other > CN, creates a replacement, writes the new id back to that file and appends
the old one to `pipeline/state/retired_box_ids`. Both paths log to `vast_calls.log`, and the
budget ledger covers both because the caller passes the create time as
`--billing-from-epoch`. `NVBLOX_BOX_ID` in `vast_client.py` remains only as the historical
default for a caller that passes nothing.

The `create` call also returns an `instance_api_key`. It is a credential, it is not needed by
anything here (this pipeline authenticates with the ssh key and the account key), and it
should be treated as exposed once it has appeared in a terminal - see the rotation note in
demo-take1's handoff for the precedent.

## 2. The ssh-key chmod — and why `start` cannot carry it

The `vastai/pytorch:cuda-12.8.1-auto` image writes `$HOME/.ssh/authorized_keys` with modes
sshd rejects (`Authentication refused: bad ownership or modes`). One box (50254121) was lost
to exactly this.

**Read from the CLI's own source (vastai 1.5.6), not assumed:**

| | |
|---|---|
| `create instance … --onstart-cmd "<script>"` | exists (`cli/commands/instances.py:129-130`) — this is where the chmod loop belongs |
| `start instance ID` | takes **no** `--onstart-cmd` (`instances.py:292-294`) |
| `update instance ID --onstart` | **recreates** the instance from a template (`:428`) — on 50265868 that would destroy the only working nvblox install. Rejected |
| `vastai execute` | runs only on stopped instances and does not whitelist chmod (this project's own note, `b200_run_20260908/setup.sh:29`) |

So there are two paths, and the pipeline implements both (`pipeline/box_ops.py`):

- **Renting a new box** (`create`): `ONSTART_CMD` — a 5-minute loop re-asserting
  `chown -R root:root $HOME/.ssh`, `chmod 700 $HOME/.ssh`, `chmod 600 authorized_keys`,
  because the image's own entrypoint rewrites the file after us. `--cancel-unavail` also
  matters on `create`: without it a failed schedule leaves a stopped instance still billing
  storage.
- **Starting an existing box** (this run): the same chmod is applied **over ssh, immediately
  after the first successful connect** — `SSH_FIX_CMD`, kept next to `ONSTART_CMD` so the two
  cannot drift. This works because the connection has just been proven; it does not help a
  box whose sshd refuses the key outright, which is what `--onstart-cmd` at create time is
  for.

## 3. Measure `inet_up` after the box is up — do not trust the listing

A listing claiming **736 Mbit/s** delivered **~10.5 MB/s** on real rsync. On the 4090 box the
link was ~1 MB/s up / ~8 MB/s down, and box 49860715 once gave 76 KB/s — 135x slower than the
B200 on the same day. Transfer, not compute, is the bottleneck on every stage.

### The minimum link this step needs

The gate in `produce_nvblox` computes this and refuses up front rather than dying mid-rsync
900 s later. For the hero-sized input the arithmetic is:

| direction | bytes | at `--step-timeout-s 900` the link must beat |
|---|---|---|
| **push** | `depth_u16.npy` + `meta.json` = **350.8 MB** | **0.39 MB/s** |
| **pull** | the five files of `NVBLOX_PULL_FILES` = **16.8 MB** | **0.019 MB/s** |

So the push sets the bar: **below ~0.39 MB/s up, a 350 MB scene cannot be pushed inside a
900 s stage** and the run is refused before it starts. Measured links on this box have been
0.42 MB/s (2026-09-09 run 2) and 1.44 MB/s (run 1) up - the slower of the two clears the bar
by 8%, which is uncomfortably close. Options when it does not clear: raise
`--step-timeout-s`, or push a smaller stack (fewer views), or rent a box with a better link.

Before the pull was restricted to those five files it moved 154.6 MB, needing 0.17 MB/s -
still under the push bar, so the push remains the binding constraint either way.

**The probe must be big enough to be true.** A 20 MB probe under-reads by about 2x: measured
2026-09-09 on the UK box, it reported 2.49 MB/s up while the real 209.6 MB push ran at
6.20 MB/s, and an external 50 MB probe read 4.38. The transfer is over before TCP reaches its
stride. The probe is now **at least 50 MB and at least 10 s**, whichever demands more - if the
first attempt finishes too quickly it is repeated once at a size scaled to the rate just seen,
capped at 200 MB so the probe never becomes the slow part of the bring-up.

Measure it first, then decide whether the run is worth starting:

```bash
./nb raw 'dd if=/dev/zero of=/workspace/_probe bs=1M count=200'   # then:
time rsync -a --info=progress2 -e "<ssh>" var/scratch/_probe200 root@<box>:/workspace/
./nb raw 'rm -f /workspace/_probe'
```

Record the measured MB/s in `status.json` before MAPANYTHING starts. What it decides:
uploading the 194 MB hero video at 1 MB/s is 3.2 min; pulling the 2.16 GB result at 8 MB/s is
4.5 min; pushing the 350 MB depth stack to the nvblox box at 1 MB/s is 5.8 min. At 76 KB/s
those become 43 min / 8 hr / 77 min — at which point the answer is to kill the box, not to wait.

## 4. The morning sequence, budget and timeouts

### 4.1 Two safeties, both required for a real vast call

Nothing in this pipeline can reach vast.ai unless **both** hold (`pipeline/vast_client.py`,
`_may_execute`):

1. `PIPELINE_DRY_RUN` is **absent** from the environment (absent, not "not 1"), and
2. `pipeline/ARMED` exists, reads `yes <YYYY-MM-DD>`, and **that date is today**.

A stale ARMED left from yesterday arms nothing. `pipeline/ARMED` is in `.gitignore` and is
never committed. This module never reads the API key: `vastai` resolves it itself from
`~/.config/vastai/vast_api_key`, and the guarantee here is that the CLI is not executed at
all unless both safeties pass.

### 4.1b The permission rule, and why ARMED still means something

`.claude/settings.json` in this worktree allows the agent to run `vastai` without a prompt:

```json
{
  "permissions": {
    "allow": [
      "Bash(vastai search offers:*)",
      "Bash(vastai show instance:*)",
      "Bash(vastai show instances:*)",
      "Bash(vastai create instance:*)",
      "Bash(vastai start instance:*)",
      "Bash(vastai stop instance:*)"
    ],
    "deny": [
      "Bash(vastai destroy:*)",
      "Bash(vastai destroy instances:*)"
    ]
  }
}
```

`destroy` stays denied: it is irreversible, and 50265868 carries the only other working
nvblox install. Destroying anything remains a human act.

**The rule alone would have removed the ARMED safety, so the safety moved.** ARMED is
enforced inside `VastClient._may_execute`; a bare `vastai create` from a shell never touches
it. `VastClient.create` now exists precisely so renting goes through that gate: without ARMED
it raises `VastNotArmed` before the CLI runs (verified - see the client's own log line
`REFUSED`). **Every create must go through `VastClient.create`.** Calling the CLI directly for
a create is the one thing that makes "nothing rents without ARMED" false, and the permission
rule makes that mistake easy, which is why it is written down here.

### 4.2 Exact sequence

```bash
cd <repo>

# 0. rehearse - prints every remote and vast command without running any of them
PIPELINE_DRY_RUN=1 python3 pipeline/run_pipeline.py \
  var/uploads/c6bdbc7e-3e33-49d6-92dd-5316a412ede1.mp4 \
  --scene-dir var/scratch/pipeline_live/hero --scene-name own_0901_173903__step2 \
  --from-pulled var/scratch/b200_run_20260908/pulled/own_0901_173903__step2 \
  --nvblox-source live --max-usd 2

# 1. offline tests - 43 checks, no network, fake vastai/ssh/scp/rsync on PATH
python3 pipeline/tests/run_tests.py

# 2. ARM (by hand, this is the deliberate act)
printf 'yes %s\n' "$(date +%F)" > pipeline/ARMED
cat pipeline/ARMED

# 3. LIVE - note PIPELINE_DRY_RUN must be UNSET, hence `env -u`
env -u PIPELINE_DRY_RUN python3 -u pipeline/run_pipeline.py \
  var/uploads/c6bdbc7e-3e33-49d6-92dd-5316a412ede1.mp4 \
  --scene-dir var/scratch/pipeline_live/hero --scene-name own_0901_173903__step2 \
  --from-pulled var/scratch/b200_run_20260908/pulled/own_0901_173903__step2 \
  --nvblox-source live --max-usd 2 --step-timeout-s 900 \
  2>&1 | tee var/scratch/pipeline_live/hero_run.log

# 4. DISARM the moment the run ends
rm -f pipeline/ARMED
```

**What runs where.** `--from-pulled` means the B200 is never rented: `GPU_UP_MA`,
`MAPANYTHING`, `PULLED`, `GPU_DOWN_MA` are served from disk and the machine that *would* be
needed is only recorded (`select_gpu`: 1148 views → 81 621 MiB → 79.7 GB → B200). The only
box touched is **50265868**, for `GPU_UP_NV` → `NVBLOX` → `FLOOR` → `GPU_DOWN_NV`.

**What to watch while it runs:**

| | |
|---|---|
| `<scene_dir>/heartbeat` | time, pid, state, running cost - rewritten every 30 s |
| `<scene_dir>/pipeline.pid` | the pid to check. **Never `pgrep -f run_pipeline.py`** - it matches the harness's own wrapper shell and has made a dead job look alive for over an hour |
| `<scene_dir>/status.json` | `state`, `steps.*`, `boxes.<id>` (ssh endpoint, measured `inet_up_mb_s`, GPU, `dph_total`, `cost_usd_measured`, `box_unreachable`) |
| `<scene_dir>/vast_calls.log` | every vast call: `RAN` if executed, `WOULD_RUN` in a dry run, `REFUSED` if the safeties blocked it |
| `<scene_dir>/logs/<STATE>.{argv,stdout,stderr}` | per-stage |

### 4.3 Stop everything, one command, id only

```bash
vastai stop instance 50265868; vastai stop instance <B200_ID>
```

Two explicit ids. **Never a filter, never a loop over a live listing** - a parallel session's
box can appear mid-cleanup. `<B200_ID>` is only relevant if a B200 was rented by hand; this
run does not rent one. `destroy` is not part of stopping: 50265868 carries the only working
nvblox install and there is no rebuild recipe in this repo. The driver itself stops every box
it brought up from that box's own `finally`, on success and on failure alike, and a failure
stopping one box does not prevent the other from being stopped (tested offline).

### 4.4 Budget and timeouts

- `--max-usd` (default 5, **use 2 for this run**), `--step-timeout-s` default **900**, the
  same per-stage timeout production uses (`gpu/run_pipeline.sh`).
- **Cost is measured, never estimated.** Every 30 s and at every state boundary the
  supervisor reads `dph_total` from `vastai show instance <id> --raw` and multiplies it by
  the measured time that box has been up. It is a *ledger*: a box that has been stopped keeps
  the cost it accrued, so the total cannot fall back to zero after `GPU_DOWN_NV`.
- **The clock runs from when the box began BILLING to a confirmed `exited`.** After a
  `create` that is minutes before this session's own `start` call - provisioning and link
  probes - and those minutes are real money this run caused. Measured on 2026-09-09: the
  ledger reported $0.0527 for its own window while the box had been billing 566 s
  (252 s provisioning + 81 s link probing + 233 s run) for **$0.1293** - a 2.43x undercount,
  in the "cheaper than it is" direction. Pass `--billing-from-epoch` with the moment `create`
  returned; without it the clock still starts at `start`, which is correct when the box was
  already up.
- **The clock ends at a CONFIRMED `exited`, not at the `stop` call.**
  Billing does not end the instant `stop` returns, so `stop()` polls `show instance` until the
  box reports `exited` (`--exit-confirm-s`, default 120 s) and bills to that moment. If the
  exit cannot be confirmed in time, the box is recorded as `exited_confirmed: false` and **the
  cost clock keeps running** — being wrong in the expensive direction is the safe way to be
  wrong here.
- **`uptime_mins` is NOT billed time — do not use it.** An earlier version of this section
  claimed vast counts about 20% more than our ledger, inferred from `uptime_mins` moving
  1237.3 → 1249.6 across the first live run. The second run disproves that reading: the same
  field went **1372.3 → 312.7**, i.e. it *decreased* across a run that took 11 minutes.
  Whatever `uptime_mins` measures, it is not this instance's billed time, and the 20% figure
  was never real. Our own ledger is self-consistent and is the number to use: run 2 measured
  $0.0704, which at $0.3689/hr is 11.45 min against a box that was up 09:10:10 → 09:21:39
  (start call to confirmed `exited`) — 11.5 min. For a figure of record, read the account's
  billing page; do not compute one from `uptime_mins`.
- **A failed `show` is not a budget breach.** It is logged, counted, and retried on the next
  tick. Three consecutive failures raise `boxes.<id>.box_unreachable` in `status.json` as a
  **signal only** - no stop is issued on it and the running stage is allowed to finish.
- Exceeding the budget sets an abort reason, kills the running stage's subprocess (so a 900 s
  stage cannot keep billing past the limit), marks the run `FAILED` naming the state it
  stopped at, and stops every box.

**Cost model for reference** (measured times; this run pays only the last two rows):

| item | time | $ |
|---|---|---|
| B200 provision (`setup.sh`, 4.6 GB weights, conf preflight) | 3.5 min | 0.45 |
| upload video 194 MB | ~3 min (estimate) | 0.39 |
| FRAMES (37.8 s) + extract (4.8 s) | 0.7 min | 0.09 |
| MAPANYTHING (468.5 s stage, 243.0 s pure inference) | 7.8 min | 1.01 |
| pull 2.16 GB | ~4.5 min (estimate) | 0.59 |
| *B200 subtotal - **not paid this run**, served from `--from-pulled`* | ~19.5 min | *~2.53* |
| **4090 (50265868): start + push 350 MB + NVBLOX (2.2 s) + pull** | ~10 min (estimate) | **~0.06** |
| **VPS CPU (PACK/LAYERS/REACH/EXPORT, 56.3 s measured)** | 1 min | **0** |

Per-stage timeouts: FRAMES 600 s (measured 42.6), MAPANYTHING 1800 s (468.5), PULLED 3600 s
(2.16 GB, the step that actually hangs on a bad link), PACK 900 s (29.1), NVBLOX+FLOOR 900 s
(compute 2.2 s - the timeout is for the upload), LAYERS/REACH/EXPORT 900 s (2.6 / 9.9 / 14.7).

### 4.4b Traps in the scripts this pipeline drives

**`isaac_export.py` masks a missing mesh as an empty one.** It does not fail when the file
named by `--mesh-name` is absent. Open3D's PLY reader prints `RPly: Unable to open file` to
stderr, returns an **empty mesh**, and the export carries on with zero vertices until it dies
much later inside `write_usd`:

```
ValueError: zero-size array to reduction operation minimum which has no identity
```

That traceback names numpy and a reduction, not the file that was never there - it cost real
diagnosis time on the 2026-09-09 live run. The driver now checks the input mesh exists and is
non-empty **before** invoking the script, and says which file is missing and which `.ply`
files the directory actually holds. `nvblox_v2` is read-only from here, so the script itself
is unchanged; the guard lives in `run_pipeline.py:produce_export`.

**`nvblox_scenes.py` writes `mesh.ply`, never `mesh_color.ply`.** The published Isaac export
was built from `mesh_color.ply`, which a different script (`nvblox_hero_color.py`) produced.
`--mesh-name` therefore defaults to `mesh.ply`; passing `mesh_color.ply` on a live run is the
exact mistake that failed the first live export.

**~~`mesh.ply` does carry vertex colours (`source_has_vertex_colors: True`), so nothing is
lost by using it.~~ — WRONG, corrected 2026-09-10.** That sentence, and the same claim in
commit `b2457e0`'s body, were read off `isaac_export.py`'s `has_color`, which tested whether
the mesh has an RGB *property* — not whether anything ever wrote to it. Depth-only runs still
call `update_color_mesh()`/`get_color_mesh()`, so every mesh has the property and every vertex
of it carries nvblox's unset colour voxel, **RGB(127,127,127)**. Measured: **1** unique colour
on all five live runs on disk, against **98,225** in `results/hero_color/mesh_color.ply`; the
exported USDs read `displayColor` = one value, `[0.498039]³` = 127/255. Seven of the eight
published USDs are flat grey for this reason.

The cause was upstream of the mesh: **PACK carries no colour at all.** MapAnything's
`per_view/*.npz` hold `pts3d`/`mask`/`conf`, FRAMES writes a frame index rather than JPEGs,
and `add_color_frame` appeared nowhere in this repo on any branch. `PACK_RGB`
(`steps/pack_rgb.py`) now re-derives RGB from the source video with the recipe
`extract_hero_rgb.sh` calibrated (NCC 0.9637 against real `img_no_norm`), writes
`rgb_u8.npy` beside the depth stack, and `nvblox_scenes.py --rgb-file` integrates it.
`has_color` now means variance, and `stats.json` carries `colour.unique_vertex_colors`.

**The 1-px pad is not optional.** nvblox's sphere tracer asserts
`image_width % raycast_subsampling_factor == 0` at `sphere_tracer.cu:398`. It is a glog
`CHECK` — a process **abort**, not an exception — the factor defaults to 4, and MapAnything's
images are 294 wide, so `294 % 4 == 2` and the colour path kills the process outright.
Depth-only runs never reach that code, which is why this was never hit before. Lowering the
factor through `ViewCalculatorParams` reads back correctly in Python and does **not** reach
the C++ tracer. Every image is therefore padded to 296x520 with the principal point shifted
to match; the border is invalid depth, so geometry is untouched. It costs 337 of 545,622
source triangles (-0.06%), which decimation absorbs entirely.

**The floor RANSAC is not reproducible unless it is both seeded and single-threaded.** See
`pipeline/steps/nvblox_scenes.py`'s header: `o3d.utility.random.seed()` alone does not do it,
and `segment_plane` in open3d 0.19 takes no seed argument of its own. The driver runs that
seeded copy from `/workspace/pipeline_scripts/`; the box's `/workspace/nvblox_v2/` is not
executed and not modified.

**`vastai start` can REFUSE while exiting 0.** On 2026-09-09 `vastai start instance
50265868` returned rc=0 with

```
Required resources are currently unavailable, state change queued.
```

The box never started. Because the return code said success, the driver went on to wait out
its whole 300 s ssh deadline and then failed with a message about ssh - blaming the symptom,
not the cause. **The return code is not the signal; the words are.** `VastClient._call` now
scans the stdout of `start` and `stop` (only those - `show` returns arbitrary JSON that may
legitimately contain such words) for `required resources are currently unavailable`,
`state change queued`, `not enough`, `failed`, `error`, case-insensitively, and raises
`VastStartUnavailable` carrying the text verbatim regardless of rc. The driver turns that
into `status.json`'s `fail_reason: "vast start refused: <verbatim text>"` within seconds, and
the box session still issues its `stop` - **a queued start is a pending request and must be
cancelled**, or it may come up later with nobody watching.

This is the third instance of the same shape in this pipeline: `isaac_export.py` returning an
empty mesh instead of failing on a missing file (above), `uptime_mins` looking like billed
time when it is not (4.4), and now a refusal delivered as a success. When a tool answers, read
what it said, not only whether it exited 0.

**`actual_status` from `show instance` lags reality.** Through the whole 2026-09-09 live run
the field read `exited` while ssh commands were executing on the box. Never gate on it - the
driver waits for ssh by connecting, and that is why.

### 4.4f A rented box can fail below our layer entirely (CDI), and reliability does not predict it

2026-09-09, third attempt at a live run. Instance **50376970** (`nvblox-v3`) was created from
offer 50113488 on **machine 109808** (Taiwan) - the host this runbook had recommended on its
metrics: `reliability2` 0.9960, `cuda_max_good` 13.0, driver 580.173.02, 524/865 Mbit.

It never started. ssh refused for four minutes, and `show instance` gave
`actual_status: created`, `cur_state: stopped` with:

```
Error response from daemon: failed to create task for container: failed to create shim task:
OCI runtime create failed: could not apply required modification to OCI specification:
error modifying OCI spec: failed to inject CDI devices: unresolvable CDI devices
D.b88275c.../gpu=8: unknown
Error: failed to start containers: C.50376970
```

The container could not be given a GPU through CDI. **Nothing of ours ran** - not the image,
not `--onstart`, not `onstart_nvblox.sh`; `/workspace/PROVISIONED` was never created. This is
a broken NVIDIA container toolkit on the host, below every layer this pipeline controls.

Three things worth carrying forward:

- **`reliability2` does not predict this.** 0.9960 is a historical uptime score; a host whose
  CDI mapping is broken can still score well. The only test that means anything is creating an
  instance and watching for the container.
- **`gpu=8` suggests an 8-GPU host** handing out one card, with the device mapping
  unresolvable. That is a hypothesis about the cause, not an established fact - it is not
  reproducible from here without renting the same host again.
- **Detect it by `cur_state`, not by ssh.** `actual_status: created` with `cur_state: stopped`
  and a `status_msg` is the signature, available seconds after create; waiting for ssh turns a
  10-second diagnosis into a 25-minute one. A future waiter should poll `status_msg` alongside
  the marker file.

The instance was stopped, then destroyed by id - it held nothing, its 80 GB was billing
storage, and destroying it also retires the `instance_api_key` that `create` printed. Host
109808 is recorded as known-bad and excluded from later searches.

Same day, same class of problem, different layer: 50265868's host refused `start` outright
for lack of resources (4.5). Two hosts, two failures, neither in our code.

### 4.4c Open, not fixed: two published floor planes are ceilings

Found on 2026-09-09 while validating the floor-selection rule, recorded here because
`nvblox_v2/results/` is read-only from this repo and nothing below was changed there.

**The check is one line.** Take the median camera up (OpenCV `-y` per view rotated to world
by `camera_pose`, median over all views) and measure the angle to a published
`floor_plane.json`'s `normal_up`. Candidate normals are oriented toward the cameras, so a
floor lands near 0 deg, a ceiling near 180, a wall near 90.

| scene | published angle | published `camera_height_above_plane_m` | what it is |
|---|---|---|---|
| own_0828_152850 | 3.53 | 1.143 | floor |
| own_0828_221137 | 3.90 | 1.263 | floor |
| **own_0829_000840** | **174.08** | 1.293 | **ceiling** |
| **own_0901_155452** | **172.88** | 1.175 | **ceiling** |
| own_0901_161054 | 5.08 | 1.279 | floor |
| own_0901_173903 (hero) | 1.40 | 1.304 | floor |
| own_0902_131247 | 4.15 | 1.808 | floor |
| own_0902_140657 | 2.18 | 1.427 | floor |

Both ceilings carry an entirely plausible camera height - 1.293 m and 1.175 m - which is why
this went unnoticed. `gpu/floor_ceiling.py`'s Motivation #1 describes the same failure in the
same words: *"every downstream number still looks internally consistent (geometry, scale,
tilt) - only the absolute reference is wrong"*. Everything in the published tree derived from
those two planes - `esdf_slice_0.3m.npy`, `frontend_layers/`, the reach outputs - inherits it.

**Consequence for this pipeline's own validation:** on those two scenes the camera-up rule
does not regress the published answer, it *contradicts* it, and the rule is the one that is
right (5.89 deg and 5.19 deg). Any statement of the form "our camera heights match the
published ones" is only meaningful for the six scenes whose published plane is a floor.

**`own_0901_155452` specifically has three near-parallel candidates**, which is what makes it
the awkward one. Relative to the published plane: our rule takes +0.147 m at 5.85 deg
(inlier 0.0729); there is another floor-parallel plane at **-0.393 m** at 5.64 deg (inlier
0.0404); and the published pick is the ceiling. Camera heights across the scene's candidates
span 0.78-1.32 m. **Which of the two floor-parallel planes is the real floor is not
established, and is deliberately not fixed here** - the 30 deg gate keeps a ceiling or a wall
out, but it cannot choose between two plausible floors 0.39 m apart.

### 4.4d Open, not closed: hero's floor fit still spreads 2.674 deg across seeds

The floor-plane selection was tightened twice on 2026-09-09 and measured each time over the
same 9 seeds (0, 1, 2, 3, 5, 7, 11, 42, 99) on two scenes:

| configuration | hero angle spread | hero camera height | own_0828_152850 angle | its height |
|---|---|---|---|---|
| height tiebreak, 3000 iters | 4.338 deg | 4.57 cm | 1.333 deg | 12.09 cm |
| **support** tiebreak, 3000 iters | 6.290 deg | 7.35 cm | **0.278 deg** | 3.45 cm |
| **support** tiebreak, **9000** iters (current) | **2.674 deg** | 4.11 cm | 0.343 deg | 3.57 cm |

**The 1 deg bar is met on own_0828_152850 and NOT met on hero (2.674 deg).** Recorded as
open rather than papered over.

Why the two scenes differ, measured rather than guessed: on own_0828_152850 the floor takes
`inlier_frac` 0.16 against 0.056 for the next candidate - a 2.9x margin - so picking by
support is decisive. On hero the floor and its nearest competitor are both ~0.05, level with
each other, so support picks a different fragment of the same floor on different draws
(inlier counts 8919-11531 at 9000 iterations). **The cause is a fragmented floor, not a bad
fit** - the least-squares refit was measured to change the plane by at most 0.001 deg (4.4c's
sibling finding), so nothing is gained by fitting harder.

`--ransac-iterations` is 9000, three times the original. It halved hero's spread
(6.290 -> 2.674 deg) for about 90 s of extra CPU per scene - measured 133-142 s per seed on
hero against 57-63 s on the smaller own_0828_152850. x10 was measured at ~9 min per seed and
extrapolates to ~1.5-2 deg, still short of 1 deg, so it was not taken.

**Retest after the union refit.** The next thing to try, deliberately not yet the default:
merge the inliers of every candidate within 30 deg of camera up AND within +/-5 cm of the
best one's height, then fit one least-squares plane to that union. That addresses
fragmentation directly, which is what the measurements point at. Until it is measured on
both scenes, hero's floor plane should be treated as reproducible to about 2.7 deg, not
better.

### 4.4g The union refit was tried and it is worse. Do not enable it.

4.4d named the union refit as the retest to try: pool the inliers of every candidate within
a few degrees of the winner and fit one plane to all of them, on the theory that the residual
spread comes from a fragmented floor. It is implemented behind `--floor-union-deg` and it is
**off by default (0.0)**, because measured over 9 seeds on two scenes at `--union-deg 3` it
makes the fit substantially worse:

| scene | angle spread, current rule | with union refit | camera height spread |
|---|---|---|---|
| own_0901_161054 | **0.644 deg** | 4.896 deg (7.6x worse) | 2.12 cm -> 17.49 cm |
| hero | **2.674 deg** | 5.293 deg (2.0x worse) | 4.11 cm -> 22.84 cm |

**Why, measured rather than guessed.** The residual scatter of the pooled inliers is
0.178-0.210 m on own_0901_161054 and 0.192-0.232 m on hero, against **0.013-0.017 m** for a
single candidate's inliers - **13x worse**. Candidates within 3 degrees of each other are
parallel, not coplanar: they are the floor and a rug, a step, a table top, or floor patches at
slightly different heights. Fitting one plane through a stack of parallel surfaces at
different heights tilts it, and how far it tilts depends on which patches that draw happened
to find - so pooling adds variance instead of removing it. The seeds where only one candidate
fell within 3 degrees (own_0901_161054 seed 11, hero seeds 11 and 99) are unchanged to the
digit, which is the control: with nothing to pool, nothing moves.

**The premise was wrong, not the implementation.** "Fragmented floor" was read as "one surface
found in pieces"; the data says the pieces are at different heights. A useful version would
have to pool only candidates that are both parallel AND coplanar - within a few centimetres in
offset, not only in angle - which is a different rule and is not written.

The flag stays for reproducibility of this result. `own_0901_161054` at 0.644 deg already
clears the 1 degree bar with the current rule; hero at 2.674 deg does not, and remains open.

### 4.4e Frame rotation: recorded at FRAMES, diagnosed at the floor gate, never applied

Phone captures carry a container rotation the pixels do not have. **Both scenes used so far
are tagged -90**: `var/uploads/c6bdbc7e-…mp4` (hero) and `…/3a7db716-…mp4`
(own_0901_161054). ffmpeg applies it by default when decoding, so the extracted frames are
upright - but that was an inherited default, not a stated one, and an unapplied rotation
moves the cameras' "up" by a quarter turn, which surfaces three stages later as a floor fit
that cannot find a floor.

- **FRAMES** now states it: `extract_frames.py` passes `-autorotate 1` explicitly, and
  `frames_by_fps.py:probe_rotation` records `container_rotation_deg`, `autorotate` and
  `frames_orientation_deg` into `frames.json`'s `video_meta`. Both spellings are read -
  modern ffprobe's `side_data_list[].rotation` and the older `rotate` tag.
- **The floor gate** now always writes `angle_by_frame_rotation_deg` into
  `floor_plane.json`: the angle between the chosen plane and the "up" each of the four frame
  rotations would imply. Measured on the published scenes, it separates cleanly:

  | scene | 0 deg | 90 | 180 | 270 |
  |---|---|---|---|---|
  | own_0901_173903 (hero) | **1.40** | 89.92 | 178.60 | 90.08 |
  | own_0828_152850 | **3.53** | 88.32 | 176.47 | 91.68 |
  | own_0829_000840 | 174.08 | 91.65 | **5.92** | 88.35 |

- **When the gate trips**, those four angles go into the failure message, and if 90 or 270
  would have passed it says so: *"that is the signature of video rotation metadata dropped at
  the FRAMES stage, not a bad floor. Nothing was applied; fix it at the source."*

**The 180 deg case is deliberately NOT reported as a rotation bug.** A 180 deg frame rotation
flips the camera's own "up" to "down", which is exactly what picking the CEILING instead of
the floor also looks like from here - the two are degenerate in this test. `own_0829_000840`
in the table above is precisely that: 4.4c reports it as a ceiling, and this test cannot tell
the two apart. The message says "ambiguous" and names both possibilities rather than picking
one.

Nothing here ever applies a rotation or corrects a plane. The run still fails.

### 4.5 Failure branch: 50265868 does not come up

`GPU_UP_NV` gives sshd **300 s** to answer (`--ssh-wait-s`, polling `show instance --raw`
and actually connecting, never sleeping). If it does not, the driver raises, the run ends
`FAILED` at `GPU_UP_NV`, and the box is stopped by id from its own `finally` — exercised
offline, test (b). Nothing is destroyed automatically.

From there the recovery is **a new 4090, not a fight with the old one**. The nvblox stage
needs 16.4 MiB of VRAM and 2.2 s of compute; what is expensive about 50265868 is only the
install sitting on its disk, and that install is reproducible from a script.

```bash
# 1. make sure the old box really is down (id only, never a filter)
vastai show instance 50265868 --raw | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["actual_status"])'
vastai stop instance 50265868

# 2. rent a replacement - Ubuntu 24.04 + CUDA 12.x, to match the wheel tag cu12ubuntu24
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 rentable=true disk_space>=80 inet_up>=200' -o dph
vastai create instance <offer_id> --image vastai/pytorch:cuda-12.8.1-auto \
  --disk 80 --ssh --direct --label nvblox-v3 --cancel-unavail \
  --onstart-cmd "$(python3 -c 'import sys;sys.path.insert(0,"pipeline");import box_ops;print(box_ops.ONSTART_CMD)')"

# 3. install nvblox (the recipe is HANDOFF_nvblox.md section 1 / nvblox_v2/setup_nvblox.sh)
rsync -a var/nvblox_v2/setup_nvblox.sh var/nvblox_v2/nvblox_scenes.py \
      root@<host>:/workspace/nvblox_v2/          # nvblox_v2 stays read-only on the VPS
ssh root@<host> 'bash /workspace/nvblox_v2/setup_nvblox.sh 2>&1 | tee /workspace/nvblox_v2/setup.log'

# 4. re-run, pointing at the new id
env -u PIPELINE_DRY_RUN python3 -u pipeline/run_pipeline.py ... \
  --nvblox-source live --nvblox-instance-id <new_id> --max-usd 2
```

**`--onstart-cmd` genuinely applies here** — this is a `create`, which is the only command
that accepts it (§2). The chmod loop is `box_ops.ONSTART_CMD`, the same string the started-box
path applies over ssh, so the two cannot drift.

**What the install actually costs** (measured on 50265868, 2026-09-08):

| step | time | note |
|---|---|---|
| create + boot to sshd | ~3 min (estimate) | the same shape as the B200 rental, which took 3 min |
| `setup_nvblox.sh` end to end | **~5 min** | inferred from timestamps: the script was written at 12:47:36 UTC and `results/setup.log` ends `=== setup done 2026-09-08T12:52:36+00:00 ===`. Not a timed run — treat as ±2 min |
| ↳ of which: pip step 3 downloads | **710 MB across 22 wheels** | counted from the log. torch 2.9.1+cu128 in step 2 installed under `pip -q`, so its own size is not in the log — do not quote a number for it |
| ↳ the nvblox wheel itself | **55.7 MB** | `nvblox_torch-0.0.10+cu12ubuntu24-py3-none-linux_x86_64.whl` |
| push depth stack 350 MB + run + pull | ~10 min (estimate, link-bound) | nvblox compute itself is 2.2 s |
| **total, create → results in hand** | **~20 min** | |
| **$ at the 4090 rate $0.369/hr** | **~$0.12** | plus whatever the failed 50265868 accrued before it was stopped |

That is well inside the `--max-usd 2` cap, and the cap keeps applying: the replacement box is
measured the same way, by `dph_total` × time up.

**Pins that must not drift** (`results/versions.json`, and the reason each exists):
`nvblox_torch` **0.0.10**, `torch` **2.9.1+cu128**, `numpy<1.27` (the wheel wants it),
Ubuntu **24.04** + CUDA **12.x** so the wheel tag `cu12ubuntu24` matches. The wheel pins
`torch<=2.9.1`; on the image's preinstalled torch 2.11 the import dies at dlopen with
`undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib` — torch changed
that symbol's 4th parameter from `int` to `unsigned int`. Hence the dedicated venv
`/workspace/venvs/nb`; never install nvblox into the image's own `/venv/main`.

**Offers live for minutes. Never hand an offer ID to a human to type.** On 2026-09-09,
between choosing a machine and creating it, offers expired repeatedly: Sweden, Taiwan, Hong
Kong and Beijing all vanished inside an hour, the same physical machine reappeared under
three different offer IDs in one afternoon (109808 as 50113494, 50113488, 50113494 again),
and two `create` attempts died with `no_such_ask` - one from a single-digit typo, one because
the offer had simply gone. Search and create belong in the same command, seconds apart, with
no round trip in between. That is what `.claude/settings.json` now allows and what
`VastClient.create` is for. If a search result must be shown to a person, show it as a
report, not as something to retype.

**The one message that means "go straight to a new box".** If `start` comes back with
`Required resources are currently unavailable, state change queued`, the host has no free
resources for this instance and no amount of waiting is indicated - `vastai`'s own guidance
says an instance stuck in `scheduling` for more than 30 s means exactly that. The run will
now fail in seconds with that text in `fail_reason`. Do not retry the same id in a loop; go
to step 2 above and rent a replacement, or come back later. This happened for real on
2026-09-09 (run 3, `own_0901_161054`): the run cost $0 because the box never ran, and the
only thing lost was the 316 s the driver spent waiting before the fix.

**Two things here are hypotheses, not measurements** — say so rather than acting as if they
are known:

1. **Why 50265868 would fail to come up is not established.** The one key-permission
   incident on record (50254121, lost to
   `Authentication refused: bad ownership or modes for file $HOME/.ssh/authorized_keys`)
   was on a **freshly created** box. Whether the same modes get rewritten on a
   *stopped-then-started* instance's preserved disk has **never been observed or tested
   here**. It is a plausible cause, not a diagnosis; a 300 s timeout is equally consistent
   with the machine simply having no free resources (the CLI's own note: an instance stuck
   in `scheduling` for more than 30 s usually means the host cannot currently satisfy it).
2. **Whether an instance's stored `onstart` re-runs on `start`** — and therefore whether a
   chmod loop set at create time would protect a later restart at all — is **not verified**.
   This is why the started-box path does not rely on it and applies the fix over ssh instead.
   Do not write "the onstart re-runs" into any runbook until someone has watched it happen.

If the box does come up but a later stage fails, the branch is different and cheaper: the
scene directory is idempotent, so fixing the cause and re-running the same command resumes
from the failed state without repeating completed work (tested offline, test (d)).

## 5. What stays manual tomorrow

1. **INGEST into the front end** — no script exists anywhere in the repo. hero-74 was done by
   hand (upload → production worker → `var/uploads/scenes/<uuid>/`). The driver writes
   `ingest_stub.json` naming what it would copy and stops there.
2. **Renting and destroying the B200** — the stub only logs. Wiring a real client is the next
   change, and it should keep the id-only shape.
3. **`pull_loop.sh` supervision** — it has no global stop marker on purpose (a global
   `ALL_DONE` killed it twice in one day and stranded results on a box). Starting it, noticing
   it died, and touching `PULL_STOP` at the end are manual.
4. **Verify-before-destroy** — compare `find . -type f -printf "%s\t%p\n" | sort -k2` on both
   sides and require an empty diff before any `destroy`.
5. **`inet_up` measurement** (§3) and the decision it feeds.
6. **`uv run alembic current` vs `alembic heads` before any `systemctl restart video-api` /
   `video-worker`.** A restart picks up on-disk ORM models immediately; a pending migration
   took prod down for ~12 min on 2026-09-06. Nothing in this pipeline restarts a service, so
   this only matters if the live run is combined with a deploy.

## 6. Risks, with an assessment

| # | risk | likelihood | impact | evidence / mitigation |
|---|---|---|---|---|
| 1 | **Transfer is the whole run.** A slow link turns a 31-min run into hours. | **high** — 3 of 3 boxes measured were transfer-bound | schedule blown, $ burned while idle | measured 76 KB/s … 10.5 MB/s across boxes. Mitigate: measure first (§3), abort under a floor (suggest 2 MB/s), ship depth (0.30 MB/view) not `pts3d` (2.4 MB/view) |
| 2 | **sshd rejects the key and the box is unreachable.** Cannot be fixed from outside. | medium — hit once, mitigated since | whole box lost, ~$0.5 wasted | `onstart.sh` loop, §2. Verify ssh works before uploading anything |
| 3 | **`rsync` killed without `--partial` deletes what it downloaded.** | medium | re-pull the full 2.16 GB | an interrupted 280 MB pull left 11 MB. `pull_loop.sh` already passes `--partial`; any hand-rolled pull must too |
| 4 | **NVBLOX and FLOOR are one script.** `fit_floor()` lives inside `nvblox_scenes.py`; there is no way to re-fit the floor without re-integrating. | certain (structural) | a floor-fit failure costs the whole GPU stage | today it is 2.2 s, so re-running is cheap — the risk is a *silent* bad fit, not cost. Guard: `floor_plane.json` `inlier_frac` and camera-height gate (0.8–2.0 m) |
| 5 | **The floor detector fails on some scenes for reasons not established.** | medium — 1 of 8 scenes | that scene's layers/reach/export are wrong, not absent | `own_0828_152850` failed at 59/300/1484 views and succeeded at 450, with `inlier_frac` 0.1619 — *higher* than four scenes that worked. Fallback exists: `align_external_floor.py`. **Cause is not established; do not claim it is** |
| 6 | **2 of 8 scenes reconstruct to an unusable map** and nothing gates it. | high for a new scene | a customer-visible garbage map | `own_0902_131010` raw coverage 1.94%, `own_0902_131247` 2.05%. There is no non-degeneracy gate yet — adding one is a real gap, not a nice-to-have |
| 7 | **open3d decimation determinism is unverified across machines.** | low | `triangles_out` drifts, comparisons stop being like-for-like | today it reproduced **exactly** (199 999 tri, USD md5 `0f568f4e…` identical) but on the *same* machine and library build. Not evidence about a different box |
| 8 | **Cost estimate understates idle time.** The model counts work, not the gap between steps. | medium | 2-3x the modelled $ on a bad day | the $15 cap is the actual protection, not the estimate |
| 9 | **Destroying 50265868 by reflex** (it is the one box with a working nvblox install). | low but irreversible | days of rebuild, no recipe in-repo | §1: stop-only. The stub logs `destroy` in `finally`; before the client goes live, that path must exclude this id |
| 10 | **The whole reachability layer rests on one room, one camera, one operator.** | certain | any cross-site claim is unsupported | stated in `demo-take1/HANDOFF.md` §8. This pipeline changes throughput, not evidence |
| 11 | **An armed `ARMED` file left on disk** after a run — the next invocation is live without anyone deciding so. | medium (it is a manual file) | an unintended live run | the date check makes it self-expiring: an ARMED dated any day but today arms nothing (tested offline, three variants). §4.2 step 4 removes it explicitly. Residual window: the rest of the same day |
| 12 | **nvblox determinism between runs on the same box is unverified.** | low | a silent change in the ESDF nobody notices | the NVBLOX step now reports `esdf_control_vs_published` — `max|diff|` of its own slice against the published one — into `status.json`. Reported, not gating: a difference is a finding, not a reason to fail. Exercised offline end to end (`max_abs_diff` 0.0, 0 NaN-pattern mismatches over 7672 finite cells) |

## 7. Running it

The live sequence is §4.2. Everything below is the offline half.

```bash
# offline tests: 43 checks, no network - fake vastai/ssh/scp/rsync on PATH
python3 pipeline/tests/run_tests.py

# dry run (refuses to start live without both safeties; see §4.1)
PIPELINE_DRY_RUN=1 python3 pipeline/run_pipeline.py <video.mp4> \
  --scene-dir var/scratch/pipeline_dry/<name> \
  --scene-name <run_id> \
  --from-pulled var/scratch/b200_run_20260908/pulled/<run_id>

# control checks against the published references
~/venvs/o3d-cpu/bin/python pipeline/compare_dry_run.py \
  --scene-dir var/scratch/pipeline_dry/<name> --scene-name <run_id>
```

Monitoring: `<scene_dir>/pipeline.pid` and `<scene_dir>/heartbeat` (rewritten every 30 s with
timestamp, pid and current state). **Never `pgrep -f run_pipeline.py`** — it matches the
harness's own wrapper shell and has made a dead job look alive for over an hour.

State of the last run is always `<scene_dir>/status.json`; per-step argv, stdout and stderr are
in `<scene_dir>/logs/<STATE>.{argv,stdout,stderr}`; every vast call the run would have made is
in `<scene_dir>/vast_calls.log`.

---

## 8. The production worker — what is known from `<repo>` (read-only)

Facts read out of the `main` checkout on 2026-09-08. Nothing here was modified. This is the
existing path that already writes `var/uploads/scenes/<uuid>`; the pipeline in this
directory does **not** replace it and does not write there.

### Process and entry point

| | |
|---|---|
| unit | `deploy/video-worker.service`, `ExecStart=~/.local/bin/uv run --project <repo> arq app.worker.WorkerSettings` |
| separate from | `deploy/video-api.service` (`uvicorn app.main:app --host 127.0.0.1 --port 8000`) — a run takes minutes and must survive an API restart (`app/worker.py` docstring) |
| job function | `app/worker.py:34 process_video(ctx, video_id, rerun=False)`, registered at `app/worker.py:93 functions = [process_video]` |
| calls | `app/services/pipeline_orchestrator.py:59 run_pipeline_for_scene()` → `:184 _run()` |
| job timeout | `settings.job_timeout_sec = 3600` (`app/config.py:25`) |
| idempotency | a scene already `done` short-circuits unless `rerun=True` (`app/worker.py:57-60`) |

### Where MapAnything actually runs — **not locally, and not on a box this pipeline rents**

`app/services/gpu_client.py:95` states it plainly: the only implementation talks to *"a
manually-provisioned, always-on Vast.ai instance over SSH, using the `gpu` host alias already
configured in `~/.ssh/config`"*. So:

- host comes from `settings.gpu_ssh_host`, default `"gpu"` (`app/config.py:21`); the `gpu`
  alias does exist in `~/.ssh/config`, and `GPU_SSH_HOST` is set in `<repo>/.env`
  (value not read or printed here).
- remote workspace `settings.gpu_workspace_dir = "/workspace"` (`app/config.py:24`); per-job
  directory `/workspace/data/jobs/<job_id>` (`gpu_client.py:116 _job_dir`).
- the box is **always-on and hand-provisioned** — `build_gpu_client()` (`gpu_client.py:375`)
  only opens an SSH connection. There is no create/destroy anywhere in it; its own docstring
  names a *"future Vast-SDK-automated client"* as the thing that would swap in.
- remote entrypoint `gpu/run_pipeline.sh <job_dir>`, started fire-and-forget
  (`nohup setsid bash run_pipeline.sh …`) and polled through `status.json`, never held open
  over a live SSH session.

**Seven stages, three separate venvs on the box** (`gpu/run_pipeline.sh`, `STAGE_TIMEOUT`
default 900 s *per stage*):

| # | stage | venv | script |
|---|---|---|---|
| 1 | keyframes | `envs/vidmap` | `gpu/stage_keyframes.py` |
| 2 | vocab | `envs/vidmap` | `gpu/stage_vocab.py` |
| 3 | **infer** | `envs/mapanything` | `gpu/stage_infer.py` |
| 4 | align | `envs/mapanything` | `gpu/stage_align.py` |
| 5 | objects | `envs/sam3` | `gpu/stage_objects.py` |
| 6 | occupancy | `envs/mapanything` | `gpu/stage_occupancy.py` |
| 7 | export_glb | `envs/mapanything` | `gpu/stage_export_glb.py` |

MapAnything itself: `gpu/stage_infer.py:69 DEFAULT_MODEL_NAME = "facebook/map-anything-apache"`,
and `ALLOWED_MODEL_NAMES` contains only that one (`:74`) — the same weights the 15 fps ladder
used. It is **one single `.infer()` call for the whole job** (`:110-113`,
`memory_efficient_inference=True, minibatch_size=1`), deliberately: the docstring says every
downstream artifact must come from that one call, because separate `.infer()` invocations do
not share a world frame.

### Frame count — 2 fps and at most 200, filtered; not the 15 fps ladder

| | value | source |
|---|---|---|
| `keyframe_fps` | **2.0** | `app/config.py:27` |
| `max_keyframes` | **200** | `app/config.py:28` |
| blur threshold | 15.0 (default; per-upload override `Video.blur_threshold`) | `gpu/stage_keyframes.py:38` |
| similarity threshold | 0.98 (default; per-upload override) | `gpu/stage_keyframes.py:39` |
| over the cap | uniform subsample to exactly `max_keyframes` | `gpu/stage_keyframes.py:61-74` |
| floor | fewer than 3 loaded views aborts the job | `gpu/stage_infer.py:107` |

Measured on real scenes: hero-74 kept **74 of 74** extracted, `subsampled: false`
(`var/scratch/hero-frozen/own_0901_173903/scene/keyframes.json`, with
`params.json` recording `keyframe_fps 2.0 / max_keyframes 200 / glb_max_points 400000`); a
production scene picked at random, `var/uploads/scenes/01d304bf-…/result.json`, reports
`keyframe_count: 21`.

**This is a different reconstruction from the one this pipeline drives**: 2 fps with
content-dependent blur/similarity filters and a 200-frame cap, versus the ladder's integer
decimation `step 2` = 15 fps = 1148 views on the same hero video. The two also live in
different world frames — the worker's `occupancy.npy` is in the pipeline's aligned frame
(floor at Y=0), ours is in the raw MapAnything frame rotated by its own fitted floor; the
measured relation is written per scene into `grid_meta.json` by `nvblox_v2/frontend_layers.py`,
which does not assume they match.

### How `var/uploads/scenes/<uuid>` is written

- path = `app/services/scene_service.py:39 scene_dir(scene_id)` →
  `Path(settings.upload_dir) / "scenes" / str(scene_id)`, with
  `upload_dir = "var/uploads"` (`app/config.py:12`). **`<uuid>` is the scene id, not the
  video id.** 142 such directories exist today.
- written by one `rsync -az` of the whole remote job dir
  (`app/services/gpu_client.py:315 fetch_results`), excluding `per_view/` and `per_view_png/`
  unless `settings.retain_per_view_artifacts` is true (default **False**, `app/config.py:44`),
  and always excluding `keyframes/`, `input.mp4` and `.env`.
- the remote job dir is then `rm -rf`'d by `cleanup()` (`gpu_client.py:366`) from the
  orchestrator's `finally`, so the scene dir is the only surviving copy.
- actual contents of a real scene dir (`var/uploads/scenes/01d304bf-0d43-428e-aefd-0aff80b014f6/`):
  `aligned_room.ply`, `alignment.json`, `alignment_transform.npz`,
  `camera_convention_report.json`, `cameras_aligned.json`, `infer_meta.json`, `keyframes.json`,
  `logs/`, `logs_run_pipeline.log`, `manifest.json`, `map_preview.png`, `occupancy.npy`,
  `occupancy_grid.npz`, `occupancy_meta.json`, `params.json`, `per_view/`, `per_view_png/`,
  `pipeline.pid`, `result.json`, `scene_objects/`, `scene_points.glb`, `status.json`.
- **after** the rsync the worker reads that directory (`app/services/scene_ingest.py`) and
  writes DB rows — objects, alignment, grid metadata, camera track, resolved robot start,
  cached vocabulary — via `scene_service.apply_scene_artifacts()` /
  `replace_scene_objects()` / `mark_done()`. A failed `validate_scene()` raises
  `PipelineError` and the scene is marked failed; the files stay on disk either way.

So an INGEST step that wanted to be real would either write into this directory alongside the
worker's own output, or go through the API — and neither exists yet. Nothing in this
pipeline writes to `var/uploads`.

---

## 9. MVFILTER — the optional pre-NVBLOX multi-view filter (2026-09-10)

Off by default. `--mvfilter` never changes a run that does not pass it: the state reports
`skipped` and NVBLOX fuses `depth_u16.npy` as before.

```bash
--mvfilter m=2,tau=0.10,neighbours=nearest,K=8    # every key required, no defaults
--mvfilter-ab                                     # ALSO fuse the unfiltered stack
--mvfilter-also m=2,tau=0.10,neighbours=baseline15,K=8   # ...and another vote (repeatable)
```

`MVFILTER` sits between `PACK` and `GPU_UP_NV` — CPU only, before any box is rented, so a
filter that fails costs nothing. It writes `depth_u16_mvfilter.npy` beside the pack and
`mvfilter.json` with the parameters and the kept fraction; `self.depth_file` then decides
which stack NVBLOX pushes and fuses. `nvblox_scenes.py` is untouched: `--depth-file` and
`--suffix` are its own existing flags.

### The two things to know before using it

**1. The neighbour rule is the parameter that matters, and it means different things on
different scenes.** Same m and tau, measured:

| scene | views | fps | `nearest` kept | `baseline15` kept |
|---|---|---|---|---|
| hero-15fps (`own_0901_173903__step2`) | 1148 | 15 | 74.0 % | 57.0 % |
| hero-74 (`7ccaa75d…`) | 74 | 2 | 47.2 % | 42.1 % |

At 15 fps a view's *nearest* cameras are its own temporal neighbours at near-zero baseline,
so they agree with it almost by construction — `REPORT.md` §1 measures residual mean 0.060 m
for `nearest` against 0.098 m for `baseline15` on that scene. At 2 fps the nearest cameras
already have real baselines. So `nearest` is a weak filter on a dense clip and a strong one
on a sparse one, and the same flag is not the same experiment twice. That is why the spec
takes no defaults and is recorded per run in `status.json`.

**2. `baseline15` is not safe on hero-74.** It kills `lamp_7` (0 surviving points of 462) —
one of the 17 non-fragment objects. `nearest` keeps it at 101. `REPORT.md` §5 warned exactly
this: the settings that filter hardest delete control objects.

### Measured, hero-74, m=2 tau=0.10 K=8 (CPU; `layers/backfill/`, band 0.1–1.5, min_points 3)

| arm | kept px | burger | go2 | husky | free comps | objects at 0 |
|---|---|---|---|---|---|---|
| shipped layer | — | 13/17 | 10/17 | 10/17 | 7 / 4 / 3 | — |
| unfiltered, same build | 100.0 % | 13/17 | 10/17 | 10/17 | 10 / 6 / 3 | 0 |
| **`nearest`** | 47.2 % | **15/17** | **13/17** | **12/17** | **1 / 1 / 1** | **0** (min 101 pts) |
| `baseline15` | 42.1 % | 15/17 | 13/17 | 12/17 | 1 / 1 / 1 | 1 (`lamp_7`) |

**Read the mechanism, not just the score.** Obstacle cells fall 3388 → 1924, but FREE cells
also fall 6043 → 4553 and UNKNOWN rises **40.2 % → 58.9 %** of the grid. The extra
reachability is obstacles becoming UNKNOWN, and `build_cost_grid` makes UNKNOWN traversable
at 1.5× — the robot is routed through cells nobody observed. Whether that is an improvement
is a product question, not a measurement one, which is why this is not merged.

### The port is exact

hero-15fps, m=2 tau=0.10 baseline15 K=8: **95,496,804 / 167,531,007 valid px = 57.002 %**
against `REPORT.md` §2's **57.00 %**, with the valid-pixel count matching §0's 167,531,007
point for point. 58.7 s for 1148 views, CPU, mmap'd. Reproduce with:

```bash
mkdir -p /tmp/probe && ln -sf var/nvblox_v2/packed/own_0901_173903__step2/depth_u16.npy /tmp/probe/
cp var/nvblox_v2/packed/own_0901_173903__step2/meta.json /tmp/probe/
~/venvs/o3d-cpu/bin/python pipeline/steps/mvfilter.py --packed /tmp/probe \
    --m 2 --tau 0.10 --neighbours baseline15 --k 8
```

### STILL UNMEASURED: the hero-15fps fusion arm

The point of a *pre*-NVBLOX filter is that nvblox re-meshes from what survives, and that
half has never run. On 2026-09-10 box **50381086 refused to start four times** — "Required
resources are currently unavailable", the same host failure that retired 50265868 the day
before (§4.5). The run FAILED at `GPU_UP_NV` having spent **$0.00**, the box was stopped from
its own `finally`, and renting a replacement was declined for that session.

So obstacle cells, band-free/largest per radius and NVBLOX compute-seconds for a filtered
*fusion* are unknown, and `REPORT.md` §7's self-declared non-like-for-like numbers are still
the only ones. `--mvfilter-ab` and `--mvfilter-also` exist and are covered by offline case
(o) precisely so the whole table comes out of ONE box session when a box is available:

```bash
env -u PIPELINE_DRY_RUN python3 -u pipeline/run_pipeline.py \
  var/uploads/c6bdbc7e-3e33-49d6-92dd-5316a412ede1.mp4 \
  --scene-dir var/scratch/mvfilter_prenvblox/live --scene-name own_0901_173903__step2 \
  --from-pulled var/scratch/b200_run_20260908/pulled/own_0901_173903__step2 \
  --nvblox-source live --nvblox-instance-id <A BOX THAT STARTS> --max-usd 1.0 \
  --mvfilter m=2,tau=0.10,neighbours=nearest,K=8 --mvfilter-ab \
  --mvfilter-also m=2,tau=0.10,neighbours=baseline15,K=8
```

The three filtered/unfiltered stacks are already computed and staged at
`var/scratch/mvfilter_prenvblox/live/packed/own_0901_173903__step2/`, so MVFILTER will
report `skipped` and the box time is fusion only (~3 × 64 s compute plus 1.05 GB of push).

### Why the controls are inside the run, not an older run

`680dbae` changed floor selection from a camera-height tiebreak at 3000 RANSAC iterations to
inlier support at 9000. The last unfiltered hero-15fps fusion (`var/scratch/pipeline_live/
hero2/`, 2026-09-09 09:22) therefore has a different floor plane, a different grid frame, and
incomparable obstacle and free-cell counts. Any A/B has to fuse both stacks in one session
with one script and one seed, which is what `--mvfilter-ab` is for.

### Scale check: the 0.85 m door does not appear on any layer

The owner's tape measurement is 0.85 m. Widest free passage anywhere, by max-min Dijkstra on
`EDT(~obstacle)` between open areas (clearance ≥ 0.30 m), width = (2·d − 1)·res:

| layer | obstacle | traversable (largest) | widest passage | ratio / 0.85 |
|---|---|---|---|---|
| hero-74 shipped | 6736 | 585 (1.46 m²) | fewer than two open areas | — |
| hero-74 unfiltered, same build | 7216 | 568 (1.42 m²) | fewer than two open areas | — |
| hero-74 filtered `nearest` | 4833 | 1132 (2.83 m²) | 0.050 m @ (66,49) | 0.059 |
| hero-15fps published | 4736 | 3936 (9.84 m²) | 0.450 m @ (37,112) | 0.529 |
| hero-15fps hero2 (stale floor rule) | 5076 | 3591 (8.98 m²) | 0.450 m @ (37,112) | 0.529 |

The 0.450 m gap sits at the same cell under both floor rules, so it is not an artefact of
`680dbae`. The §7 vertex-form filter deletes 64 % of the obstacle mask and moves it only to
0.516 m. Filtering hero-74 makes it *worse*, because 59 % unknown leaves almost nothing
"observed". Room extents: hero-74 5.55 × 7.10 m, hero-15fps 3.75 × 7.00 m; Umeyama scale
between the two solves 0.9726; camera height above the fitted floor 1.2703 m vs 1.3035 m.
The two reconstructions agree with each other to ~3 %, and neither resolves a doorway — the
openings are not sealed by a thin crust of drift that a point filter can lift off.
