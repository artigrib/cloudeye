# Models

One section per model, each with the same five fields. Every number in "Measured" comes
from a run recorded in this repository, with the file it is recorded in. Where a licence
could not be verified from inside this repository it is written **UNKNOWN**, not guessed.

This project ships no model weights. `gpu/*.py` downloads them by name from HuggingFace at
first run, or calls them by name through an HTTP API. Weights are downloaded under the
operator's own account and credentials — see [THIRD_PARTY.md](../THIRD_PARTY.md).

"4090 rented" and "B200 rented" are vast.ai instances the operator rents. "VPS CPU" is the
machine running `app/` and the arq worker. "API" means an HTTP call with no local weights;
the row says which machine originates the call, because that is where the credentials are.

---

## MapAnything

- **Role.** Monocular 3D reconstruction: keyframes in, per-view point maps and camera
  poses out. Everything downstream (floor alignment, occupancy, objects) is built on it.
- **Model id / version.** Weights `facebook/map-anything-apache`, no `revision=` pin
  (`gpu/stage_infer.py:69`, loaded at `:102`). Code `mapanything==1.1.4`, installed from a
  `git clone --depth 1` with no commit recorded. `gpu/stage_infer.py:74` allowlists exactly
  one repo id, enforced at `:94-99`, so a `params.json` override cannot substitute another.
- **Licence.** Apache-2.0 (HF model card tags `apache-2.0`). The un-suffixed
  `facebook/map-anything` is CC-BY-NC-4.0 and is deliberately not used
  (`gpu/stage_infer.py:71-73`).
- **Where it runs.** 4090 rented, `mapanything` venv, in the 7-stage `gpu/` path. Pass A of
  the `pipeline/` driver runs it on a B200 rented box, driven by hand.
- **Measured.** `infer` stage 46.5 s on the ikport scene (`docs/COMPARISON.md:192`); 132.0 s
  on the same video during the Gemma control run (`docs/COMPARISON.md:251`).

## NVIDIA nvblox

- **Role.** TSDF integration and ESDF over the packed depth+RGB stack; produces the band
  layer the planner uses, the floor plane, and the room mesh.
- **Model id / version.** Not a learned model: `nvblox_torch` **0.0.10**, the prebuilt wheel
  `nvblox_torch-0.0.10+cu12ubuntu24-py3-none-linux_x86_64.whl` from
  `github.com/nvidia-isaac/nvblox` releases, installed by
  `pipeline/box/onstart_nvblox.sh:37` (pin) and `:95` (install). Co-pinned there:
  torch 2.9.1+cu128, numpy 1.26.4, open3d 0.19.0.
- **Licence.** **UNKNOWN from this repository.** Nothing here states nvblox's licence; the
  wheel is downloaded from NVIDIA's release page and never redistributed. The vendored
  build patches in `patches/nvblox-cuda128/` carry an Apache-style header tail in their
  context lines, which is indicative and not verification. Read the upstream `LICENSE`.
- **Where it runs.** 4090 rented, venv `/workspace/venvs/nb`, driven over ssh by
  `pipeline/steps/nvblox_scenes.py`.
- **Measured.** Compute 2.2 s; VRAM 16.4 MiB (`docs/PIPELINE.md:622`). Wheel 55.7 MB
  (`docs/PIPELINE.md:657`). Provisioning the box from `onstart_nvblox.sh`: **4 min 12 s**
  (`docs/PIPELINE.md:95`). One room end to end: **$0.1293** for a 566 s billed window
  (`docs/PIPELINE.md:306`). The install itself: ~$0.12 at $0.369/hr
  (`docs/PIPELINE.md:660`).

## SAM3

- **Role.** Open-vocabulary segmentation: takes the vocabulary from the semantics stage and
  the per-view images, produces per-object masks that become 3D object point clouds and
  bounding boxes.
- **Model id / version.** Weights `facebook/sam3` (HF, gated), loaded through
  `build_sam3_image_model()` with `load_from_HF=True` and no `revision=`
  (`gpu/stage_objects.py:144`, `:176`). Code `sam3==0.1.0`, plain `git clone`, no commit
  recorded.
- **Licence.** "SAM License" — Meta's own terms, tagged `other` on HuggingFace, not an OSI
  licence. It adds a litigation-termination clause, an indemnification obligation, a
  publication-attribution requirement, and prohibited-use categories. Access is gated
  per HuggingFace account and must be requested before the stage can run at all.
- **Where it runs.** 4090 rented, `sam3` venv.
- **Measured.** `objects` stage 728.6 s, 75% of a 969.3 s end-to-end run — the stage that
  dominates the pipeline (`docs/COMPARISON.md:194`, `:197`, `:200`).

## Gemma — via OpenRouter

- **Role.** Alternative model for the semantics (object vocabulary) stage: five keyframes
  in, a list of object concept names out.
- **Model id / version.** OpenRouter slug `google/gemma-4-31b-it`. A routing slug, not a
  weight reference: OpenRouter selects the serving backend and exposes no revision pin.
- **Licence.** **UNKNOWN from this repository.** Google publishes Gemma under its own terms;
  nothing here records or verifies them.
- **Where it runs.** API. The call originates on the 4090 rented box, from
  `gpu/stage_vocab.py`, using the operator's OpenRouter key pushed as a per-job secret.
- **Measured.** 12 objects against GLM's 23 on the same video (`docs/COMPARISON.md:17`).
  Isolated 4-frame trial ~5 s for the vocab call; full-pipeline control run total 1144.7 s
  against GLM's 969.3 s, recorded as contaminated and not to be cited as a latency result
  (`docs/COMPARISON.md:246-249`). Not the default — see GLM below.

## Gemma — via Vertex AI

- **Role.** Same stage, second provider, selected per video by `vocab_provider="vertex"`.
- **Model id / version.** `google/gemma-4-26b-a4b-it-maas`
  (`app/services/model_catalog.py:22`, fallback default `gpu/stage_vocab.py:61`). Called
  through the Vertex OpenAPI chat-completions endpoint at `locations/global`.
- **Licence.** **UNKNOWN from this repository.** Governed by Google Cloud's Vertex AI terms
  plus Gemma's own; neither is recorded here.
- **Where it runs.** API. The call originates on the 4090 rented box, from
  `gpu/stage_vocab.py`, using a short-lived access token pushed as a per-job secret —
  see [GOOGLE_CLOUD.md](GOOGLE_CLOUD.md).
- **Measured.** 7 concepts against GLM's 19 on the same video (`docs/COMPARISON.md:17-25`).
  One clean latency sample 6.52 s; the other two samples hit `429 RESOURCE_EXHAUSTED` and
  are contaminated by retry/backoff, so the repository states there is not enough clean
  data to cite (`docs/COMPARISON.md:164-168`). Image cap 4 frames against OpenRouter's 5
  (`gpu/stage_vocab.py:69`).

## Nemotron

- **Role.** Command parsing: turns "go to the bed" into `{action, target, destination}`.
  No vision input.
- **Model id / version.** OpenRouter slug `nvidia/nemotron-3-nano-30b-a3b`
  (`app/config.py:65`). Underlying weights `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.
  Routing slug, no revision pin.
- **Licence.** NVIDIA Open Model License, per the underlying HuggingFace repository.
- **Where it runs.** API, called from the **VPS CPU** (`app/services/vlm_client.py`). This
  is the one LLM stage that does not originate on the GPU box.
- **Measured.** Selected over GLM for this stage on a 23-object scene; see
  `docs/COMPARISON.md:175` for the split. No per-call latency sample is recorded in this
  repository — not measured.

## GLM — the semantics default

- **Role.** Semantics (object vocabulary) stage. This is the default provider and model.
- **Model id / version.** OpenRouter slug `z-ai/glm-5.3-flash` (`app/config.py:66`,
  `gpu/stage_vocab.py:56`). Underlying weights `zai-org/GLM-5.3-Flash`. Routing slug, no
  revision pin. It is the default in four places: `app/config.py:66`,
  `gpu/stage_vocab.py:56`, `app/services/job_spec.py:61` (`semantics="openrouter-glm"`),
  and `docs/job_spec.schema.json:37`.
- **Licence.** MIT, per the underlying HuggingFace repository.
- **Where it runs.** API. The call originates on the 4090 rented box, from
  `gpu/stage_vocab.py`.
- **Measured.** `vocab` stage 93.0 s, 9.6% of a 969.3 s end-to-end run
  (`docs/COMPARISON.md:191`, `:200`). 23 objects on the ikport scene against Gemma's 12
  (`docs/COMPARISON.md:17`). Three repeat calls at `temperature: 0` with byte-identical
  input returned three different vocabularies (`docs/COMPARISON.md:32-38`).

---

## The pipeline

The 7-stage path in `gpu/run_pipeline.sh`, which is what the API's arq worker drives.

1. **keyframes** — in: `input.mp4`. out: `keyframes/`, `keyframes.json`. model: none
   (ffmpeg). runs on: 4090 rented.
2. **vocab** — in: 5 keyframes. out: `vocab.json`. model: GLM (default), Gemma via
   OpenRouter or Vertex. runs on: API call from the 4090 rented box.
3. **infer** — in: `keyframes/`. out: `per_view/*.npz`, `per_view_png/`, `infer_meta.json`.
   model: MapAnything. runs on: 4090 rented.
4. **align** — in: `per_view/`. out: `aligned_room.ply`, `alignment_transform.npz`,
   `cameras_aligned.json`. model: none (RANSAC floor fit). runs on: 4090 rented.
5. **objects** — in: `vocab.json`, `per_view/`, alignment transform. out:
   `scene_objects/scene_objects.json` and per-object `.ply`. model: SAM3. runs on: 4090 rented.
6. **occupancy** — in: aligned cloud. out: `occupancy_grid.npz`, `occupancy_meta.json`,
   `map_preview.png`. model: none. runs on: 4090 rented.
7. **export_glb** — in: aligned cloud. out: `scene_points.glb`. model: none. runs on:
   4090 rented.
8. **finalize + ingest** — in: all of the above. out: `result.json`, validated rows in
   PostgreSQL. model: none. runs on: 4090 rented, then VPS CPU.
9. **command** (on request, not part of a job) — in: user text plus the scene's objects.
   out: `{action, target, destination}` and an A* route. model: Nemotron. runs on: VPS CPU.

The `pipeline/` driver is a separate, resumable path that adds nvblox — see
[GPU_VAST.md](GPU_VAST.md) for its stage list and for why its MapAnything pass A is manual.
