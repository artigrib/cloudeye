# Job spec — what the wizard chooses, and how the progress screen reads it back

Two file formats meet here, and only one of them is ours:

- **the job spec** — written by `POST /api/videos/upload` from the New-workspace wizard,
  read by the worker. Schema: `docs/job_spec.schema.json`, model:
  `app/services/job_spec.py`. This document is its contract.
- **`status.json`** — written by **pipeline-v1**, which is READ-ONLY from this side. The
  shapes below were read out of
  `<repo>/pipeline/run_pipeline.py`, not agreed
  with anyone; if that file changes, this document is what is wrong.

Nothing here adds a DB column or a migration. The spec is a file next to the video.

---

## 1. The job spec

`<upload_dir>/<video_uuid>.job.json`, beside `<upload_dir>/<video_uuid>.mp4`. One file per
video, written once at upload, never mutated.

```json
{
  "spec_version": 1,
  "video_id": "b904b296-fd24-4ac7-b203-bc439102cfef",
  "frames_fps": 15,
  "mapping": "mapanything",
  "semantics": "openrouter-glm",
  "chat_llm": "openrouter-nemotron",
  "robots": [],
  "created_at": "2026-09-09T11:04:22+02:00"
}
```

| field | values | default | notes |
|---|---|---|---|
| `spec_version` | `1` | — | a reader that does not know the value must refuse the job, not guess |
| `video_id` | uuid | — | the `Video` row; also the file's own stem |
| `frames_fps` | > 0, ≤ 30 | **15** | the measured knee on the hero footage, HANDOFF §4 |
| `mapping` | `mapanything` | `mapanything` | one value today, and no UI choice — it exists so a second backend is a named schema change rather than an implicit default |
| `semantics` | `openrouter-glm`, `vertex-gemma` | **`openrouter-glm`** | the scene-vocabulary call |
| `chat_llm` | `openrouter-nemotron`, `vertex-gemma` | **`openrouter-nemotron`** | typed-command parsing |
| `robots` | ids from `app/robots.py` | `[]` | `[]` means every registered platform, which is what the viewer already shows |
| `created_at` | local ISO-8601, seconds | — | same shape as `status.json`'s timestamps |

### The catalog agrees with this — and one thing still does not

**`semantics` and the model catalog now share one default.** They did not: the catalog's
`"Scene vocabulary"` stage said `vertex` while the wizard defaulted to `openrouter-glm`,
so an operator who accepted the wizard's default got a different provider from one who
uploaded without a spec. Settled 2026-09-09 in favour of **OpenRouter**:
`model_catalog.DEFAULT_VOCAB_PROVIDER` is now `VOCAB_PROVIDER_OPENROUTER`, and
`tests/test_job_spec.py::test_catalog_default_matches_the_job_spec_default` (plus a second
test against what the endpoint actually serves) fails if the two drift apart again. This
does not touch the automatic OpenRouter fallback `gpu/stage_vocab.py`'s `dispatch_vocab()`
uses on a Vertex failure — that is a runtime recovery path, not a choice anyone makes.

**`chat_llm` is still recorded but not honoured by this app.** The catalog marks
`"Command parsing"` `editable: false`, and `app/services/vlm_client.py` has no Vertex
path — it is OpenRouter-only. The field is carried for the worker; the running app ignores
it. It is in the spec because the wizard was specified to ask, and it is left that way
deliberately.

### Provider ids

`semantics` maps onto the catalog's option ids, which is what the existing
`vocab_provider` upload field already takes:

| spec value | catalog id | model |
|---|---|---|
| `openrouter-glm` | `openrouter` | `z-ai/glm-5.3-flash` |
| `vertex-gemma` | `vertex` | `google/gemma-4-26b-a4b-it-maas` |

`chat_llm`'s `openrouter-nemotron` is the catalog's `nvidia/nemotron-3-nano-30b-a3b`.

---

## 2. `status.json` — pipeline-v1's format, as it actually is

`<scene_dir>/status.json`, rewritten atomically (tmp + `os.replace`) after every change,
so a reader never sees a half-written file and does not need a lock.

```json
{
  "scene": "<scene id>",
  "video": "<video path>",
  "pipeline_dry_run": true,
  "state": "MAPANYTHING",
  "started": "2026-09-09T11:04:25+02:00",
  "finished": null,
  "duration_s": null,
  "cost_estimate_usd": 0.0,
  "cost_basis": "measured: dph_total from `vastai show instance` x time up",
  "max_usd": null,
  "boxes": {},
  "error": null,
  "steps": {
    "FRAMES": {
      "state": "ok",
      "started": "2026-09-09T11:04:25+02:00",
      "finished": "2026-09-09T11:05:01+02:00",
      "duration_s": 36.2,
      "cost_estimate_usd": 0.0,
      "error": null,
      "artifacts": {}
    }
  }
}
```

**States**, in order (`run_pipeline.STATES`):

```
QUEUED  FRAMES  GPU_UP_MA  MAPANYTHING  PULLED  GPU_DOWN_MA  SEMANTICS  PACK  MVFILTER
GPU_UP_NV  NVBLOX  FLOOR  GPU_DOWN_NV  LAYERS  REACH  EXPORT  INGEST  DONE
```

`SEMANTICS` is a declared slot with no implementation in this build; it always reports
`skipped` / "not implemented". `MVFILTER` is optional and off unless `--mvfilter` is given,
in which case it reports `skipped` / "--mvfilter not given" — see
`pipeline/steps/mvfilter.py`.

Terminal is `DONE` or `FAILED`. `FAILED` is not in `STATES` — it is set by
`Status.terminal()` and carries `error`. A failing step also writes the full stderr to
`<scene_dir>/logs/<STATE>.stderr`, and `steps[<STATE>].error` holds it.

Timestamps are `datetime.now().astimezone().isoformat(timespec="seconds")` — **local
time with an offset**, not UTC, and not a Z suffix. Anything comparing them must parse the
offset rather than assume UTC.

### `provider_used` / `fallback_reason` — contract additions, not yet written

The progress screen shows "the provider you chose was unavailable" when the pipeline fell
back. **pipeline-v1 writes neither field today** (grepped: zero hits). They are defined
here for the worker to add, at the top level of `status.json`:

```json
{ "provider_used": "vertex-gemma", "fallback_reason": "openrouter 429, 3 retries" }
```

Both optional. A consumer must treat their absence as "no fallback happened" and must not
render the notice — which is exactly what the progress screen does today, so nothing
breaks while the worker does not write them. The notice fires only when `provider_used` is
present AND differs from the spec's requested provider.

## 3. `logs/worker.log` — the fallback source of stage timings

When `status.json` is absent, stages come from `<scene_dir>/logs/worker.log`, whose lines
look like:

```
2026-08-31 05:55:03,709 INFO [app.services.pipeline_orchestrator] scene=e1e21716 stage=keyframes (1/7) state=running message=starting keyframes
```

The parser takes `stage=<name>` and `state=<value>` and the leading
`%Y-%m-%d %H:%M:%S,%f` timestamp. `(1/7)` is informational. This is the same shape
`tests/test_nightly_batch_report.py`'s `parse_stage_timings` already consumes; the
progress endpoint reuses that reading rather than inventing a second one.

**`status.json` wins whenever it exists.** The log is a degraded source: it has no
per-step `error`, no artifacts and no terminal state, so a screen built on it can say what
ran and when, and nothing else.

## 4. What the progress screen may and may not do

- Every piece of state comes from the server on each poll. Nothing in `localStorage`,
  nothing in tab memory: two tabs on the same URL, and a reload at any moment, must show
  the same feed.
- **No fabricated entries.** If neither source has a record, the screen says so.
- If no record has appeared for **60 s**, the screen states *worker not connected* rather
  than spinning. A spinner claims progress that nothing has observed.
- `DONE` → redirect to the scene. `FAILED` → the failing state plus its stderr.
