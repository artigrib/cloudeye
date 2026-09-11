# Vast.ai vs. GCE comparison

The full side-by-side tables (Section A: performance, Section B: correctness) land
after the Phase 4 baseline runs on both sites, built from `runs/vast-*.json` and
`runs/gcp-*.json` by `scripts/build_run_json.py`. This document starts with
methodology notes recorded *before* any comparison data exists, because they
constrain what the eventual comparison can validly measure — written up now so they
don't get lost or rediscovered under time pressure once real numbers are on the
table.

## Vocabulary provider decision — two facts (2026-08-31)

Full-pipeline control runs, not isolated trials — see "Control run" sections below
for the complete write-up each. Recorded here as two standalone facts because the
decision they support is expensive enough not to live only in a terminal scrollback.

1. **Gemma rejected on completeness.** 12 solid objects against GLM's 23 on the same
   ikport video. Loses, by name: `air conditioner`, `blanket`, `bottle`, `clothes`,
   `door` (×2), `pillow`, `remote`. Zero new concepts gained. Geometry (footprint,
   ceiling height) is identical between the two runs — reconstruction does not
   depend on which vocabulary model ran.
2. **Vertex AI verified end-to-end.** Scene `2e45d2b6-4208-4a04-ba2d-ecb5a2f227b2`
   built completely; `vocab_source=vertex` confirmed in both the database and the
   API; `GET /scenes/{id}/usd` returned a working 12MB file. Vocabulary: 7 concepts
   against GLM's 19. The provider is supported and reproducible — it is not the
   default.

## Methodology notes and known confounds

### Object-vocabulary non-determinism — object count and reachability are not valid comparison metrics

`gpu/stage_vocab.py`'s OpenRouter call sets `"temperature": 0` explicitly
(`gpu/stage_vocab.py:77`) — this is not a missing setting. Despite that, three
repeat calls with byte-identical input (same 5 keyframes, same prompt, same model)
returned three different object vocabularies: `door` present in 2 of 3, `curtain`
present in 2 of 3, every other concept stable 3/3. Root cause is upstream of this
project (provider-side sampling/routing non-determinism at the LLM), not something
a prompt change or client-side fix resolves.

Reconstruction and segmentation themselves remain fully deterministic — confirmed
directly: when a concept name *is* present in the vocabulary, its point count
matches run-to-run to the last digit (e.g. `door`: 101,558 points in two independent
runs of the same source video, bit-identical). **The instability is entirely in
which concept names make it into the vocabulary JSON, not in what SAM3/MapAnything/
DBSCAN do once a concept is in the list.**

Confirmed effect on the headline reachability number: the same ikport video produced
"11 unreachable of 23 objects" in one run and "4 unreachable of 14" in another,
driven entirely by which concepts that run's vocabulary call happened to name — not
by GPU hardware, not by which LLM was used for the vocabulary call, not by the
room-bbox guard fix (see below).

**Consequence for this comparison**: object count and "N unreachable of M" must
not be used to compare the two GPU sites — a difference there would measure
vocabulary-call noise, not a real Vast-vs-GCE divergence. Valid correctness
comparison metrics are the ones computed *below* the vocab stage, i.e. independent
of which concepts happened to be named that run:

- per-stage wall-clock time and peak memory (Section A)
- room/scene dimensions (bbox extents, footprint)
- ceiling height
- total point-cloud size
- point counts for objects present in **both** runs' vocabularies, matched by
  concept name — confirmed near-bit-identical run-to-run when the concept is
  present at all

If Section B ends up including an object-count row for descriptive purposes, it
must be flagged inline as vocabulary-dependent and non-comparable, never presented
as a clean Δ.

### Room-bbox point guard (`fc07b4c`) — fixes leaked-tail geometry, not distortion in general

The per-point guard added to `gpu/stage_objects.py` (filters pooled points against
`room_bbox_min`/`room_bbox_max` before clustering, not just the pre-existing
centroid-only check) only catches points that leak **past the room boundary** —
e.g. through an open doorway, where MapAnything's monocular depth has no anchor and
produces runaway depth. Confirmed on `room.mp4`: 206 of 1,429,583 pooled `door`
points were outside the padded room bbox and got dropped — the pre-fix cluster
spanned z=[-3.69, +11.51]m (15.2m depth) against the room's own bbox depth of 7.76m
(z=[-4.01, +3.75]m); the fix removes exactly the points responsible for that
overshoot. Exact post-fix bbox depth for this cluster hasn't been re-measured and
recorded — only the dropped-point count is confirmed so far.

It does **not** address distortion that stays entirely inside the room. Same run,
same fix active: `mirror`'s pooled point count was unchanged (124,503 before and
after, zero points dropped), and its historical bbox depth of 1.47m (no real mirror
is that deep) remains unaddressed — whatever inflates it is a same-room artifact
(plausibly a reflection), structurally invisible to a room-bounds check. Any write-up
of this fix should say "corrects leaked-past-the-wall geometry," not "fixes object
dimension corruption" in general — the two failure modes are distinct and only one
is currently handled.

### Open question, not yet diagnosed: `door` concept pools far more points than comparable objects

In the same scene, `door`'s SAM3 mask pools ~1.43M points across views, vs. `sink`'s
211k and `mirror`'s 124k — roughly 5-7x more, despite door not plausibly having 5-7x
the visible surface area of a sink. Not yet distinguished whether this is a
size-class/DBSCAN-parameter interaction, a genuinely larger real detection area
across many views, or (the working suspicion) the SAM3 mask for `door`-type prompts
is systematically capturing the surrounding wall rather than just the door leaf.
Recorded here as an open question — no diagnosis performed, no fix implied.

### Vertex AI / Agent Platform — three limitations, confirmed by direct testing, not to be rediscovered

Recorded so a future attempt to add or extend the Vertex vocabulary provider doesn't
waste time re-finding these:

1. **The MaaS endpoint only works via the `global` location, not a region.** Every
   region tried (`us-central1`, `us-east5`, `europe-west4`) returns either
   `404 NOT_FOUND` or `400 FAILED_PRECONDITION` ("only available via global endpoint")
   for `google/gemma-4-26b-a4b-it-maas`. The working host is the bare
   `aiplatform.googleapis.com` (no region prefix) with `.../locations/global/...` in
   the path — not `{region}-aiplatform.googleapis.com`.
2. **Hard cap of 4 images per request.** A 5th image gets rejected outright
   ("Expected a maximum of 4 images (inclusive); found 5"). `gpu/stage_vocab.py`'s
   `N_SAMPLE_FRAMES` (5, OpenRouter) does not apply to Vertex —
   `VERTEX_MAX_FRAMES = 4` is a separate constant for this reason, not an oversight.
3. **`google/gemma-4-26b-a4b-it-maas` and `google/gemma-4-31b-it` are different
   models, not two names for the same weights.** Google's own Model Garden spec table
   lists them as separate Gemma 4 variants with different intended deployment targets
   (31B: "large servers or server clusters"; 26B-A4B: "higher-end desktop computers
   and servers") — 31B has no serverless MaaS offering at all, only self-deploy. Any
   comparison between the two is a cross-model comparison, not a same-model
   provider comparison.

### Vocabulary quality: OpenRouter `gemma-4-31b-it` recommended, `z-ai/glm-5.3-flash` is what's actually deployed

**Status as of 2026-08-31: the production default is `z-ai/glm-5.3-flash`, not
Gemma.** `app/config.py`'s `openrouter_model_vision` and `gpu/stage_vocab.py`'s
`DEFAULT_MODEL` have both named GLM since Aug 28/29 and neither has been changed.
The paragraph below records a quality recommendation from isolated trials, not a
code change that happened — see the findings note further down for the full
timeline of how those got conflated.

Measured (3 repeats each, same prompt, same 4 ikport frames, scored against the real
GLM baseline vocabulary — `cccbab37`'s `vocab.json`, which has **19** words, not the
"18" referenced in earlier discussion; using the real file):

| | Matched in all 3 trials | Recall |
|---|---|---|
| `gemma-4-31b-it` (OpenRouter) | bed, chair, door, mat, nightstand, rug, wardrobe (7) | 7-8/19 |
| `gemma-4-26b-a4b-it-maas` (Vertex MaaS) | bed, chair, door, nightstand, rug (5) | 5/19, every trial |

**Recommendation from this trial: prefer OpenRouter `gemma-4-31b-it` over Vertex's
`gemma-4-26b-a4b-it-maas`** if and when a switch away from GLM happens — the MaaS
variant was evaluated and found weaker on completeness, consistently missing `mat`
and `wardrobe` (both real, found by 31B every time) and never finding
`air conditioner` (found by 31B once). This is a same-provider-family comparison
between two Gemma variants, not a decision between Gemma and the GLM model actually
in use — no full-pipeline run on either Gemma variant has been done (see findings
note). Vertex remains available as the second option in the Models selector
(`app/services/model_catalog.py`), not as the default.

**Real finding, independent of which model "won": `gemma-4-26b-a4b-it-maas` was fully
deterministic across all 3 trials — bit-identical output, same 9 words, same order,
every time.** `gemma-4-31b-it` was not (trial 3 differed from trials 1-2, matching the
non-determinism already documented above). Since both were called at `temperature=0`
with identical input, **this shows the vocab-call non-determinism is a property of the
specific model and/or its serving/routing, not of LLM vocabulary calls in general.**
Worth remembering if a future model swap is being considered for determinism reasons
specifically, separately from quality.

**Speed: the comparison is not informative, and shouldn't be cited as one.** Vertex's
endpoint (Experimental launch stage) hit `429 RESOURCE_EXHAUSTED` ("the request queue
is full") repeatedly under back-to-back calls — 2 of the 3 timing measurements are
contaminated by retry/backoff delay, not real inference latency. The one clean
sample (6.52s) sits inside OpenRouter's own observed range (1.3-20s across trials) -
not enough clean data, on either side, to claim either provider is faster.

### Finding (2026-08-31): the Gemma default described above was never applied to code

A documentation/code split went unnoticed for most of a day. Recorded here plainly
so it isn't rediscovered the same way: **since 2026-08-28, every production run has
used `z-ai/glm-5.3-flash`** (`app/config.py`'s `openrouter_model_vision`, last
touched 2026-08-29 23:44 by the commit that split command-parsing off to Nemotron
and explicitly *kept* GLM for vocab; `gpu/stage_vocab.py`'s own `DEFAULT_MODEL`
fallback has said the same since 2026-08-28 20:10). The "stays the default" language
above describes a recommendation from isolated 4-frame trials, written into this
file on 2026-08-31 — it was never carried into either file that actually decides
the model at runtime. No full-pipeline run has ever been made on `gemma-4-31b-it`
or `gemma-4-26b-a4b-it-maas`; every full-pipeline number in this document, including
`cccbab37`'s baseline scene (35 objects, 23 solid, 19-word vocabulary), was produced
by GLM.

Real per-stage timing for that GLM baseline run (`cccbab37`, ikport video,
2026-08-28 23:42-23:58, from `video-worker.service` logs):

| stage | wall time |
|---|---:|
| keyframes | 15.5s |
| vocab (OpenRouter, GLM) | 93.0s |
| infer | 46.5s |
| align | 31.1s |
| objects | 728.6s |
| export_glb + finalize | 31.2s |
| result fetch (GPU → backend) | 15.9s |
| **total** (arq-measured, dequeue to done) | **969.3s** |

`vocab` is 9.6% of total wall time, not the ~20% a 1200s-total estimate would
suggest — actual total is 969s, and `objects` (SAM3 + MapAnything + DBSCAN) is the
stage that actually dominates at 75%. The isolated 4-frame Gemma trial above measured
5s for the vocab call alone, against this run's 93s for the same call under GLM -
if that gap holds in a full pipeline run (untested), switching saves at most ~88s,
i.e. **at most ~9% of total wall time**, not the ~15% an estimate built on the wrong
stage split would give. This is a floor, not a fixed saving: the 93s GLM figure is
one sample, and the pipeline's own documented vocab non-determinism (above) means
comparing single-sample latencies between providers is exactly the kind of noise
this document already warns against for object counts.

### Control run (2026-08-31): full pipeline on Gemma, same ikport video as the GLM baseline — Gemma loses real objects, GLM stays the default

Everything above this point was 4-frame isolated trials. This is the first (and so
far only) full-pipeline run on `gemma-4-31b-it`, done as a deliberate, temporary,
reverted config change (`openrouter_model_vision` → `google/gemma-4-31b-it`,
scene `3a42ab4c`, applied and reverted within a single restart cycle of
`video-worker.service`; production ran GLM before and after, with zero other scenes
processed during the window) — same source video as the `cccbab37` GLM baseline
(`VID_20260829_000840433.mp4`, byte-identical file, re-uploaded), so keyframes,
reconstruction, and room geometry are the same input on both sides; only the
vocabulary model differs.

**Vocabulary**: Gemma named 10 concepts (`bed, chair, dresser, nightstand, lamp,
mat, rug, radiator, wardrobe, curtain`) against GLM's 19. Every one of Gemma's 10 is
also in GLM's list — **Gemma found zero concepts GLM missed.**

**Objects**: 15 total / **12 solid** (Gemma) vs. 35 total / **23 solid** (GLM). At the
materialized-object level (not just proposed vocabulary), GLM's 17 distinct solid
object types are a strict superset of Gemma's 10. **7 object types present in the
GLM scene do not exist at all in the Gemma scene**: `air conditioner` (1), `blanket`
(1), `bottle` (1), `clothes` (2), **`door` (2)**, `pillow` (3), `remote` (1) — 11 solid
instances, exactly accounting for the 23→12 drop (23 − 11 = 12). Losing `door` is the
one that matters most for this product: doors are navigation-relevant obstacles/
waypoints, not decoration, and it's not a low-count outlier — 2 solid instances,
same as in the baseline count. Gemma introduced zero new objects in exchange.

**Room geometry, as expected unaffected by vocab**: footprint 3.50×4.30m (Gemma) vs.
3.55×4.30m (GLM) — one grid cell (5cm) of difference, ceiling 2.5838m vs 2.5865m
(2.7mm) — both within normal reconstruction noise for the same room, confirming
(again) that vocab choice doesn't touch geometry.

**Timing**: `video-worker.service`'s stage-transition log only records a stage when
two consecutive 15s status polls (`GPU_POLL_INTERVAL_SEC`) see different stage
names — Gemma's vocab call finished fast enough to fall entirely inside one polling
gap, so no distinct `vocab` line was ever logged for this run and no exact duration
can be quoted from it. Bound: `keyframes`+`vocab` combined took 16.1s here, against
GLM's `keyframes` alone at 15.5s — consistent with (not proof of) the isolated
benchmark's ~5s Gemma vs. this run's own 93.0s GLM figure. Total wall time was
1144.7s vs. GLM's 969.3s, but **this comparison is contaminated and should not be
read as "Gemma is slower"**: this run's video-push step alone took ~139s (vs. the
~8-13s `GPU_SETUP_LOG.md` documents as normal post-cipher-fix), and its `infer`
stage took 132.0s against GLM's 46.5s on the same video — both look like
this-instance-this-session variance, not a vocab-model effect. No clean total-time
comparison exists yet; none was the point of this run.

**Verdict: yes, we lose real objects on Gemma — GLM stays the default.** No vocabulary
was hand-picked to produce this answer; the concept and object lists above are the
complete, unedited output of both runs.

### Control run (2026-08-31): full pipeline on Vertex, same ikport video — Google Cloud confirmed working end-to-end in the production path

Separate from the Gemma run above, run after it and not mixed with it: same source
video (`VID_20260829_000840433.mp4`, re-uploaded again), `vocab_provider=vertex`,
`openrouter_model_vision` back on GLM the whole time (irrelevant to this path
anyway — Vertex never touches it). Purpose was narrower than the Gemma run: confirm
Vertex/Agent Platform is a real, working provider in this codebase's production
path, not just in the isolated 4-frame trials above.

- **Scene**: `2e45d2b6-4208-4a04-ba2d-ecb5a2f227b2`, status `done`.
- **`vocab_source`**: `vertex` — both in the database (`scenes.vocab_source`) and in
  the API (`GET /api/scenes/{id}` → `"vocab_source": "vertex"`). `vocab_model`:
  `google/gemma-4-26b-a4b-it-maas` in both places too.
- **Concepts**: 7 — `bed, chair, dresser, rug, wardrobe, radiator, curtain`. (11
  objects total, 8 solid — not compared against the GLM/Gemma baselines above since
  that wasn't this run's purpose; recorded for the record.)
- **The 4-image cap did not cause a failure.** `VERTEX_MAX_FRAMES = 4` already caps
  the request correctly (see the limitations section above) — the vocab call
  completed without incident. As instructed, no code was touched for this run
  regardless of outcome; there was simply nothing to report here since it worked.
- **Stage timing**: same polling-granularity limitation as the Gemma run —
  `keyframes`+`vocab` combined took 16.1s (`keyframes` start 16:29:18.266 →
  `infer` start 16:29:34.397), too fast for a 15s-interval poller to catch `vocab`
  as its own logged stage. Total wall time: **1044.6s** (arq-measured) — not
  compared to the other two runs' totals for the same instance-variance reasons
  noted above.
- **Reached USD: yes.** `GET /scenes/{id}/usd` returned `200` and generated
  `usd/scene_pc1_frag0.usd` (12,025,098 bytes) on first request (USD export is
  lazy/cached per `scenes.py:get_scene_usd`'s docstring, not part of the automated
  pipeline — `usd_path` is empty in the DB until the first request for a scene, this
  one included). This is the fact this run exists to establish: a scene built via
  Vertex/Google Cloud's vocabulary path reaches a fully valid, Isaac-Sim-ready
  OpenUSD export through the same code path as any OpenRouter-sourced scene.

### Provider-failure decision: fail the scene, no fallback of any kind

`gpu/stage_vocab.py` originally caught any vocabulary-call failure (missing
credentials, no keyframes, the call itself erroring) and silently substituted a
hardcoded 6-concept vocabulary, letting the job continue to `done`. **Decided
against, deliberately, for both possible fallback targets** - not just avoiding an
automatic switch to the *other* provider, but also removing the hardcoded-default
fallback that already existed before Vertex was added:

- A scene built on a substituted vocabulary looks exactly like a real one - the same
  class of silent corruption the project's existing "validate before writing to the
  DB" rule (`scene_validator.py`) was already built to prevent for other failure
  modes, not a new concern invented for this feature.
- Automatically substituting a *different provider* specifically would not be a
  neutral fallback: this document's own measurement above shows `gemma-4-26b-a4b-it-maas`
  finds fewer of the reference objects than `gemma-4-31b-it` (5/19 vs 7-8/19,
  missing `mat` and `wardrobe` entirely) - silently falling over to it would silently
  change a scene's results, not just its availability.

`gpu/stage_vocab.py` now raises `VocabProviderError` (naming the provider and the
reason) on any of these conditions instead of falling back to anything. This
propagates through the existing stage-failure machinery unchanged -
`run_pipeline.sh`'s `run_stage()` already treats any non-zero stage exit as a job
failure and captures a log tail into the error message, `pipeline_orchestrator.py`
already surfaces that as `scene.error_message` - no new failure-handling code was
needed in the backend, only the removal of the old fallback in the GPU-side stage.
Switching providers after a failure is a decision a person makes deliberately (the
Models selector), not something the pipeline does automatically.
