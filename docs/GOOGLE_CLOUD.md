# Google Cloud (Vertex AI)

**OpenRouter is the default.** Vertex AI is an optional second provider for one stage.
Everything below is opt-in; with no Google configuration the project runs on OpenRouter
alone, and the upload form's "Scene vocabulary" selector pre-selects OpenRouter
(`app/services/model_catalog.py:38`, `app/services/job_spec.py:56`).

## What Vertex is used for

**Semantics naming — the object-vocabulary stage, and only that stage.** Five keyframes go
to the model, a list of object concept names comes back, and SAM 3 segments against those
names. The model is `google/gemma-4-26b-a4b-it-maas`
(`app/services/model_catalog.py:22`).

The job spec also has a `chat_llm: "vertex-gemma"` value (`app/services/job_spec.py:30`),
so a spec on disk can record that choice. **It is inert in the running app**: the model
catalog marks command parsing non-editable and `app/services/vlm_client.py` has no Vertex
path. Chat always runs on OpenRouter today.

## How to enable it

1. **Project id.** Set `GCP_PROJECT_ID` in `.env` to your Google Cloud project. This is the
   only Google value the app reads from configuration (`app/config.py:77`).
2. **Region.** There is no region variable. The endpoint is hard-coded to `locations/global`
   (`gpu/stage_vocab.py:63-65`). Changing region means editing that URL.
3. **Credentials — ADC, not a service account.** Authentication is Application Default
   Credentials:

       gcloud auth application-default login

   `app/services/vertex_auth.py` shells out to `gcloud auth application-default
   print-access-token` and caches the token in memory for 45 minutes (the cache exists
   because `gcloud`'s own CLI startup measured ~1.7 s, which would otherwise consume the
   health check's 3 s budget).

   **A service account is deliberately not supported.** There is no code path that reads a
   service account key file, and no `GOOGLE_APPLICATION_CREDENTIALS` handling: the comment
   at `app/config.py:70-74` records that org policy forbids service account keys, so this
   authenticates as a person. If you need a service account — and for anything
   multi-tenant you do — that is a change in `vertex_auth.py`, not a setting.
4. **Enable the API.** The project needs Vertex AI (`aiplatform.googleapis.com`) enabled and
   the signed-in identity needs permission to call it.
5. **Select it per video.** Choose "Vertex AI" in the upload form's Scene vocabulary
   selector, or send `semantics: "vertex-gemma"` in the job spec.

### Environment variables

| Variable | Read by | Meaning |
|---|---|---|
| `GCP_PROJECT_ID` | `app/config.py:77` | Your project id. Unset means Vertex is unavailable. |
| `VERTEX_ACCESS_TOKEN` | `gpu/stage_vocab.py:262` | A short-lived OAuth token. **Normally not set by hand** — the backend mints it per job. Set it only when running the stage standalone. |

## Where the credentials actually go — read this before enabling it

The command-parsing LLM (Nemotron) runs from the VPS. **The vocabulary stage does not.** It
is a GPU-side stage (`gpu/stage_vocab.py`), so for every job the backend mints an ADC
access token and pushes it to the rented GPU box as a per-job secret
(`app/services/pipeline_orchestrator.py:234-235`). The token is written over ssh stdin into
a mode-600 `<job_dir>/.env` on that box, never passed in argv and never logged, and
`gpu/run_pipeline.sh` deletes it when the job ends.

So: **a short-lived Google access token does leave this server**, to a machine rented from a
third party, on every Vertex job. It is scoped and expires on its own (~1 hour), and no
long-lived credential or service-account key ever leaves the VPS. If that is not acceptable
for your project, use OpenRouter — which is the default — or move the vocabulary call to the
VPS, which is a code change, not a setting.

## Spend caps

**There is no daily USD cap per provider, and no per-LLM spend cap of any kind.** Nothing in
`app/` or `gpu/` tracks or limits LLM spend. Use OpenRouter's own account-level limits, and
Google Cloud's budget alerts and quotas, to bound this.

The only USD caps in this repository are for renting GPU boxes on vast.ai, in the separate
`pipeline/` driver: `--max-usd` (default 5.0) per run and `PIPELINE_DAILY_USD` per day,
accumulated in `pipeline/state/daily_spend.json`. See [GPU_VAST.md](GPU_VAST.md). They do
not apply to LLM calls.

## What happens on failure

When a video's provider is left unset, Vertex is tried first and **falls back to OpenRouter
automatically** (`gpu/stage_vocab.py:283-318`, `dispatch_vocab`):

- **Vertex call fails** (`VocabProviderError`) → the stage logs
  `vocab provider: vertex (primary) failed (...) - falling back to openrouter` and runs the
  OpenRouter path. The job continues.
- **Token mint fails on the VPS** (no `gcloud`, no ADC, no project) → the backend logs
  `vertex token mint failed (...) - using openrouter for vocab` and sends a plain OpenRouter
  job (`app/services/pipeline_orchestrator.py:238-240`). Nothing is pushed to the GPU box.
- **OpenRouter then also fails** → the job fails. Neither provider ever substitutes a
  fabricated vocabulary; `VocabProviderError` propagates.
- **Explicitly choosing OpenRouter** opts out of Vertex entirely — it is the only path
  tried, and there is nothing to fall back from.

**The UI does not show a fallback reason.** The scene screen shows the provider that
actually produced the vocabulary — `OpenRouter · <model>` or `Vertex AI · <model>` — via
`frontend/src/lib/vocabSourceLabel.ts`. After a fallback the scene records
`vocab_source="openrouter"`, so the screen says OpenRouter and does not say that Vertex was
tried first. The reason exists only in the worker log and the GPU stage log. Surfacing it in
the UI would need a new field on the scene; it is not implemented.

The upload form does show live provider health — availability and latency, 3 s timeout,
30 s cache (`app/services/vocab_health.py`) — so a dead provider can be seen before upload
rather than after.

## Measured

- **Latency.** One clean Vertex sample: **6.52 s** for the vocabulary call. The other two
  samples in that trial hit `429 RESOURCE_EXHAUSTED` and are contaminated by retry and
  backoff, so the repository states there is not enough clean data to cite a Vertex latency
  (`docs/COMPARISON.md:164-168`). OpenRouter's own observed range across trials was
  1.3–20 s.
- **Completeness.** On the same video, Vertex Gemma named 7 concepts against GLM's 19
  (`docs/COMPARISON.md:17-25`). This is why OpenRouter/GLM remained the default.
- **Image cap.** Vertex accepts 4 keyframes per call against OpenRouter's 5, confirmed by
  testing (`gpu/stage_vocab.py:66-69`).
- **Cost.** **Not measured.** No dollar figure for any LLM call is recorded anywhere in this
  repository.
