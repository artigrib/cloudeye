# Renting the nvblox box on vast.ai

The `pipeline/` driver is a resumable state machine that turns a video into a scene
catalogue, renting a GPU box only for the stages that need one. It is separate from the
7-stage `gpu/` path the API's arq worker drives — see [MODELS.md](MODELS.md).

Nothing here rents anything unless you arm it. Read "Safeties" before the rest.

## Safeties: `ARMED` vs `PIPELINE_WORKER_ENABLED`

Both gate the same thing — any real vast.ai CLI call — for two different operators.

**`pipeline/ARMED`** is the interactive gate. `VastClient.armed_state()`
(`pipeline/vast_client.py:129-147`) requires *both*: `PIPELINE_DRY_RUN` absent from the
environment, and the file matching `^yes\s+YYYY-MM-DD$` with **today's** date. A stale file
arms nothing. The check sits inside `_call()` (`:194-197`), the single funnel every CLI call
goes through, so nothing can route around it. The driver exits 2 up front if neither a valid
ARMED file nor `PIPELINE_DRY_RUN` is present (`pipeline/run_pipeline.py:36-37`).

    echo "yes $(date +%F)" > pipeline/ARMED     # expires at midnight, by design

**`PIPELINE_WORKER_ENABLED`** replaces it for the unattended worker, which has no operator
to write a file each morning. It must be exactly `"1"`
(`pipeline/worker_driver.py:34`, `:69-81`); anything else, including unset, means dry run —
the driver still runs and still writes `status.json`, but makes no vast call and spends
nothing. When not live it also injects `PIPELINE_DRY_RUN=1` into the child's environment
(`:213-217`) so `VastClient` refuses regardless of intent. It is set in
`deploy/video-worker.service` (commented out by default) and is paired with
`PIPELINE_DAILY_USD` as its second bound. Both values are recorded into `status.json` under
`worker_guards` (`run_pipeline.py:1521-1529`), so "why did this rent / not rent" is
answerable from one file.

Renting is opt-in in both modes. There is no configuration in which the default is to spend.

## Budget cap

Three nested limits:

| Limit | Where | Default |
|---|---|---|
| Per run | `--max-usd` (`run_pipeline.py:1845`, enforced `:495-500`) | 5.00 |
| Per day | `PIPELINE_DAILY_USD` (`worker_driver.py:37`, checked `:141-160`) | unset = no daily cap |
| Ledger | `pipeline/state/daily_spend.json` (`worker_driver.py:116-138`) | per machine, gitignored |

The daily cap is checked **before the driver starts** — "a cap enforced after renting is an
invoice, not a cap" (`worker_driver.py:143-147`). Over the cap raises `JobRefused` and
nothing is rented. The per-run cap is then clamped to what is left of the day
(`worker_driver.py:202-203`), so a $3 job cannot push a $3 daily cap to $5.90.

Cost is a measured ledger, not an estimate: `Supervisor.poll_cost`
(`run_pipeline.py:444-500`) reads `dph_total` from `vastai show instance --raw` every 30 s
and multiplies by measured uptime, per box, summed over every box the run brought up —
stopped boxes keep their cost. The clock runs from `billing_from` (create or start) to a
**confirmed** `exited` (`box_ops.py:359-404`). On exceeding the cap the driver sets
`abort_reason` and kills the running stage's subprocess, so a 900 s stage cannot keep
billing; every box is stopped in the run's `finally`. A failed `show` is explicitly not
treated as a budget breach (`run_pipeline.py:462-478`).

## Which box gets used: `start_or_create`

`pipeline/state/nvblox_box_id` holds the instance id to try first. It is machine-local and
gitignored — a fresh clone has none, and the first live run must be given one with
`--nvblox-instance-id`. `VastClient.start_or_create()` (`vast_client.py:295-336`):

1. Read the cached id (or take `--nvblox-instance-id`). Raise if there is neither.
2. `start` it. On success, return — **no search, no create, no new charge for a new disk**.
3. Only if vast *refuses* the start does it search, rank offers, and create a replacement.
   The refusal is detected from stdout markers (`vast_client.py:94-100`) checked before the
   return code, because vast exits 0 while declining.
4. On create, the old id is retired into `pipeline/state/retired_box_ids` and the new one
   cached.

**Nothing is ever destroyed automatically.** `destroy` is a human act
(`vast_client.py:250-251`, `pipeline/state/README.md`). A retired box stays stopped with its
venv intact until someone deletes it.

### Criteria filter

The search query (`vast_client.py:62-63`):

    gpu_name=RTX_4090 num_gpus=1 inet_down>=200 inet_up>=200 disk_space>=60
    reliability>=0.98 verified=true rentable=true

and the hard filter re-applied to the results in `rank_offers` (`vast_client.py:272-293`):
`gpu_name == "RTX 4090"`, `num_gpus == 1`, `verification == "verified"`,
`cuda_max_good >= 12.8`, driver major `>= 580`, `inet_up >= 200`, `inet_down >= 200`,
`reliability2 >= 0.98`, `disk_space >= 60`, `machine_id` not in `KNOWN_BAD_MACHINES`, and
`dph_total <= MAX_DPH` where `MAX_DPH = 1.00` (`:70`) is a hard ceiling, not a preference.

Region is not a filter but the **primary sort key** (`:289-292`): EU `0`, US `1`, other `2`,
CN `3`, then price. Deliberately `(region, price)` and not `(price, region)` — the rationale
with measured numbers is at `:258-270`: a UK box moved 4.38 MB/s where a Taiwan box managed
0.42 MB/s, and this pipeline pushes hundreds of MB per job.

## Creating the box

`onstart_nvblox.sh` is passed as vast's `--onstart-cmd` at create time, and it re-runs on
every boot. It is idempotent: if the import check already passes it refreshes the marker and
exits. It does two things, in this order because the second is useless without the first:

1. **Keeps sshd able to accept the key.** The `vastai/pytorch` image rewrites
   `authorized_keys` with modes sshd rejects, sometimes *after* the onstart script runs, so
   the fix loops for the first five minutes. Instance 50254121 was lost to exactly that.
   `start instance` cannot carry this flag — it is create-only — which is why it lives here.
2. **Builds venv `nb` from scratch**: torch 2.9.1+cu128, the nvblox_torch **0.0.10** wheel,
   numpy 1.26.4, open3d 0.19.0. Each pin has a recorded reason in the script's header — on
   the image's own torch 2.11 the import dies at `dlopen` with an undefined symbol, because
   torch changed that symbol's fourth parameter from `int` to `unsigned`.

It finishes by writing `/workspace/PROVISIONED` with the output of
`import nvblox_torch, torch; print(torch.cuda.is_available())`, so a waiter polls one file:
absent means not ready, present but not ending in `True` means ready without CUDA, which is
a different problem.

    vastai create instance <offer-id> \
      --image vastai/pytorch:cuda-12.8.1-auto --disk 80 \
      --ssh --direct --label nvblox --cancel-unavail --raw \
      --onstart pipeline/box/onstart_nvblox.sh

The driver does this for you; run it by hand only to provision a box outside a job.

## MapAnything pass A is manual today

The driver does **not** run MapAnything. `run_pipeline.py:1557-1559` refuses the live path
outright — *"live MapAnything is not wired in this build: rent and drive the B200 by hand,
then pass `--from-pulled`"* — and the `MAPANYTHING` step parks in `waiting_manual` naming
the two paths it needs rather than failing with "output missing and no producer"
(`run_pipeline.py:712-726`). The job is resumable exactly as it stands: every step's check
runs before its producer, so re-running the driver over the same scene directory picks the
output up and carries on from `PULLED`.

**Where to put pass-A output.** Two conventions (`run_pipeline.py:529-533`):

- With `--from-pulled <dir>`: that directory must hold `poses.json` (keys `model`,
  `n_views`, `views`) and `per_view/*.npz`, with `n_views == len(views) == n_npz`
  (`check_mapanything`, `:680-692`).
- Without it: `<scene-dir>/pass_a/poses.json` and `<scene-dir>/pass_a/per_view/`.

In worker mode the bundle is resolved by the original upload's filename stem under
`PIPELINE_PASS_A_ROOT` (`worker_driver.py:88-113`). The match is exact — `<stem>` or
`<stem>__<variant>` — and more than one candidate is a refusal naming every candidate,
never a guess.

## Measured

From real runs, each with its source:

| Number | What | Source |
|---|---|---|
| **4 min 12 s** | `onstart_nvblox.sh` provisioning a fresh 4090 | `docs/PIPELINE.md:95` |
| **$0.1293** | one room, true cost of a 566 s billed window (252 s provisioning + 81 s link probing + 233 s run) | `docs/PIPELINE.md:306` |
| 265 s | `own_0901_161054` end to end on that box | `docs/PIPELINE.md:95` |
| 2.2 s / 16.4 MiB | nvblox compute and VRAM — the GPU size is irrelevant | `docs/PIPELINE.md:622` |
| 55.7 MB | the nvblox_torch 0.0.10 wheel | `docs/PIPELINE.md:657` |
| ~$0.12 | the nvblox install itself at $0.369/hr | `docs/PIPELINE.md:660` |
| 4.38 / 3.46 MB/s | up / down on the UK box, ten times the previous one | `docs/PIPELINE.md:95` |
| $0.8222/hr, $0.369/hr | the two box rates the figures above are quoted against | `docs/PIPELINE.md:95`, `:23` |

Call it **about $0.13 per room** plus whatever a fresh provision costs, and note that the
ledger undercounted by 2.43x before `--billing-from-epoch` existed — the $0.0527 figure in
older notes is wrong (`docs/PIPELINE.md:305-306`).

## Sample scene

The sample scene is **not in git** — it is a pipeline artifact, and this repository commits
none. It is published as an asset on the repository's
[Releases](https://github.com/artigrib/cloudeye/releases) page. Download and unpack it next
to the driver:

    mkdir -p var/sample-scene
    curl -L -o /tmp/sample-scene.tar.zst \
      https://github.com/artigrib/cloudeye/releases/latest/download/sample-scene.tar.zst
    tar --zstd -xf /tmp/sample-scene.tar.zst -C var/sample-scene

It contains a pass-A bundle (`poses.json` + `per_view/`) and the packed depth and RGB
stacks, which is enough to run `PACK` through `EXPORT` and to exercise
`pipeline/tests/run_tests.py` without renting anything. Point the driver at it with
`--from-pulled var/sample-scene/pass_a`, and the offline suite with
`PIPELINE_TEST_FIXTURES=var/sample-scene`.

If no release asset is published yet, produce your own by running a video through the
7-stage `gpu/` path and keeping its `per_view/` output — there is no other source, and this
repository will not grow one.
