# Where this export's checks differ from their brief

This repository was assembled against a written Definition of Done. Three of its items
described behaviour the code does not have, and one asked for something that could not be
done safely on the machine available. Each is recorded here with the file that settles it,
rather than being quietly satisfied or quietly dropped.

---

## 1. `docker compose up` → a native install, verified

**Brief:** "docker compose up brings postgres + api + vite; reproduce docs/INSTALL.md
yourself in a fresh directory, exit 0, GET /workspaces/ returns 200."

**What was done, and why.** The machine this export was built on is a live host running the
project's own API and worker behind `ufw`, with no container runtime installed. Installing
one would have added a daemon and its own iptables chains to a production firewall. The
owner's decision was to not do that.

So the criterion was rewritten: **[docs/INSTALL.md](INSTALL.md) is the native path** —
`uv sync`, Alembic, npm, an existing PostgreSQL and Redis — and it is reproduced in a fresh
clone against a throwaway database, exiting 0, with `GET /workspaces/` returning 200.
`docker-compose.yml`, `Dockerfile.api` and `Dockerfile.web` are still written and shipped,
labelled **written, not executed** in both `INSTALL.md` and `README.md`, and are reported as
untested rather than green.

## 2. `GET /workspaces/` is a UI route, not an API route

**There is no `GET /workspaces/` endpoint.** `app/routers/workspaces.py:22` mounts at
`/api/workspaces` and declares exactly one route, `GET /{workspace_id}/processing`
(`:25`). The workspace list is `GET /api/projects` (`app/routers/projects.py:42`) — a
workspace *is* a project, and the upload auto-creates one per video.

`/workspaces` is the SPA route (`frontend/src/App.tsx:43`), so the 200 comes from the dev
server. The install check asserts all four: `GET /workspaces/` 200 from vite, and `/health`,
`/api/projects` and the proxied `/api/projects` 200 from the API.

## 3. There is no daily USD cap per provider

**Brief:** "the daily USD cap per provider."

**There is none, for any LLM provider.** Nothing in `app/` or `gpu/` tracks or limits LLM
spend — no counter, no ledger, no check. The only USD caps in this repository rent GPU boxes
on vast.ai, in the separate `pipeline/` driver: `--max-usd` per run (default 5.00,
`pipeline/run_pipeline.py:1845`) and `PIPELINE_DAILY_USD` per day
(`pipeline/worker_driver.py:37`), accumulated in `pipeline/state/daily_spend.json`. They do
not apply to OpenRouter or Vertex calls.

[GOOGLE_CLOUD.md](GOOGLE_CLOUD.md) says this plainly and points at the provider-side
controls (OpenRouter account limits, Google Cloud budgets and quotas) that do exist.

## 4. The fallback is real; the UI does not show its reason

**Brief:** "the stage falls back to OpenRouter and the UI shows the fallback reason."

**The fallback is real.** When a video's provider is left unset, Vertex is primary and
`dispatch_vocab` falls back to OpenRouter on any `VocabProviderError`
(`gpu/stage_vocab.py:283-318`); a failure to mint the local ADC token degrades to OpenRouter
before the job is even pushed (`app/services/pipeline_orchestrator.py:238-240`). Both are
logged. Neither ever substitutes a fabricated vocabulary.

**The UI does not show the reason.** `frontend/src/lib/vocabSourceLabel.ts` renders the
provider that *succeeded* — after a fallback the scene records `vocab_source="openrouter"`,
so the screen says OpenRouter and never mentions that Vertex was tried first. The reason
exists only in the worker log and the GPU stage log. Surfacing it would need a new field on
the scene. Not implemented, and not invented for the sake of the doc.

## 5. Google credentials **do** leave the server

**Brief:** "LLM stages run on the VPS, not on the GPU box, so no Google credentials leave
the server."

**Not true as written, and it is the security-relevant one.** Only *command parsing*
(Nemotron, `app/services/vlm_client.py`) runs on the VPS. The **vocabulary stage is a
GPU-side stage** (`gpu/stage_vocab.py`), so for every Vertex job the backend mints an
Application Default Credentials access token and pushes it to the rented GPU box as a
per-job secret (`app/services/pipeline_orchestrator.py:234-235`). It is written over ssh
stdin into a mode-600 `<job_dir>/.env`, never passed in argv, never logged, and deleted by
`gpu/run_pipeline.sh` when the job ends.

A short-lived, scoped Google token therefore reaches a third-party rented machine on every
Vertex job. No long-lived credential and no service-account key ever does.
[GOOGLE_CLOUD.md](GOOGLE_CLOUD.md) states this in full, under its own heading, instead of
repeating the brief's claim. Making the claim true is a code change — moving the vocabulary
call to the VPS — not a setting.

## 6. Composition of this export

Files were copied, never merged from git history. `app/`, `frontend/` and most of `docs/`
come from one source branch; `pipeline/`, `gpu/` and `deploy/` from another. Three
consequences worth knowing:

- **`gpu/stage_vocab.py` was taken from the `app/` side**, not the `pipeline/` side. It is
  the one file that differs between the two, and its `dispatch_vocab` is what `app/` and the
  test suite call. The other side's copy lacks it and would not import.
- **`pipeline/` is a standalone driver here.** The worker wiring that connects it to
  `app/worker.py` lives on the other branch. In this repository the driver is run from the
  command line (`python3 pipeline/run_pipeline.py --scene-dir …`) and the arq worker drives
  the 7-stage `gpu/` path. Both are real; neither was hand-merged into the other as a side
  effect of publishing.
- **`demo/` was dropped**, except three acceptance instruments moved to `tests/acceptance/`
  (see below) and `record_isaac.py`, which shipped code imports and which therefore moved to
  `scripts/msa/record_isaac.py`. It carries no machine-specific paths and no browser
  dependency; leaving it out would have deleted `scripts/msa/export_presentation.py` and two
  passing test modules with it.

## 7. The acceptance instruments that back the README's numbers

| File | Lines | The number it backs |
|---|---:|---|
| `tests/acceptance/probe_hero_numbers.py` | 347 | hero reachability **13 / 10 / 10 of 17** (Burger / Go2 / Husky), **tightest gap 0.28 m**, route lengths 4.066 m and 2.216 m |
| `tests/acceptance/probe_route_obstacles.py` | 151 | **"14 cells through a wall → 0"** — obstacle cells on any planned route must be zero, across every registered platform |
| `tests/acceptance/probe_shot6_export.py` | 102 | the Isaac Sim export: **3 669 032 B, md5 `0f568f4e…`**, compared against the file on disk |

Their scene identifiers are named constants in `tests/acceptance/hero_scene.py`, overridable
by environment variable, instead of UUID literals. The visual probes that only take
1440×900 screenshots were not moved.

## 8. Two mechanical notes

- **`sk-` still matches prose.** The forbidden-string check looks for secret-shaped tokens
  (`sk-`, `hf_`, `ghp_` followed by 16+ characters) and finds none. A bare `sk-` also matches
  ordinary English and CSS — `mask-leak`, `disk-cost`, `mask-type`, `pre-Task-3b`. Those are
  reported separately rather than rewritten out of the prose.
- **The offline pipeline suite grew.** `docs/KNOWN_TEST_FAILURES.md` records 117 checks from
  an earlier run; the suite in this repository reports **153/153**. The count in the table is
  whatever the suite actually printed, not the one the older note remembers.
