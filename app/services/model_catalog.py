"""The pipeline's model catalog - one place describing which model runs at each stage,
served to the frontend so it never hardcodes a name that could drift from docs/MODELS.md
or app/config.py. Names/repo ids are taken verbatim from docs/MODELS.md - do not invent
or paraphrase them here.

Only the "Scene vocabulary" stage is user-selectable (`editable=True`, multiple
`options`) - everything else is a single fixed option the frontend renders as plain
text, matching the product spec ("Фиксированные строки - обычный текст ... без
контрола"). VERTEX_VOCAB_MODEL is the only Vertex vocabulary model, per the earlier
Model-Garden check (docs/COMPARISON.md) - not user-overridable, unlike the OpenRouter
side which already reads settings.openrouter_model_vision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import settings

# The one Gemma variant actually offered as serverless MaaS on Vertex/Agent Platform -
# see docs/COMPARISON.md's "Vertex AI limitations" section for how this was confirmed.
VERTEX_VOCAB_MODEL = "google/gemma-4-26b-a4b-it-maas"

VOCAB_PROVIDER_OPENROUTER = "openrouter"
VOCAB_PROVIDER_VERTEX = "vertex"
VOCAB_PROVIDERS = {VOCAB_PROVIDER_OPENROUTER, VOCAB_PROVIDER_VERTEX}
# OpenRouter is the default as of 2026-09-09, changed from Vertex (which had been primary
# since 2026-09-07) so that this catalog and the New-workspace wizard agree. They did not:
# the wizard's `semantics` default is `openrouter-glm` (docs/JOB_SPEC.md), so an operator
# who accepted the wizard's default got a DIFFERENT provider from one who uploaded without
# a spec. One default for one decision - pinned by
# tests/test_job_spec.py::test_catalog_default_matches_the_job_spec_default, which fails if
# these two drift apart again.
#
# Unchanged by this: OpenRouter remains the automatic, logged fallback
# gpu/stage_vocab.py's dispatch_vocab() uses on any Vertex failure. That is a runtime
# recovery path, not a choice anyone makes, and it does not depend on which way this points.
DEFAULT_VOCAB_PROVIDER = VOCAB_PROVIDER_OPENROUTER


@dataclass(frozen=True)
class ModelOption:
    id: str | None  # None for a fixed (non-selectable) stage's single option
    provider: str
    model: str


@dataclass(frozen=True)
class ModelStage:
    stage: str
    editable: bool
    options: list[ModelOption] = field(default_factory=list)
    default: str | None = None  # matches an option's `id`, only meaningful when editable


def get_catalog() -> list[ModelStage]:
    return [
        ModelStage(
            stage="Reconstruction",
            editable=False,
            options=[ModelOption(id=None, provider="Hugging Face", model="MapAnything (facebook/map-anything-apache)")],
        ),
        ModelStage(
            stage="Segmentation",
            editable=False,
            options=[ModelOption(id=None, provider="Hugging Face", model="SAM3 (facebook/sam3)")],
        ),
        ModelStage(
            stage="Scene vocabulary",
            editable=True,
            options=[
                ModelOption(id=VOCAB_PROVIDER_OPENROUTER, provider="OpenRouter", model=settings.openrouter_model_vision),
                ModelOption(id=VOCAB_PROVIDER_VERTEX, provider="Vertex AI", model=VERTEX_VOCAB_MODEL),
            ],
            default=DEFAULT_VOCAB_PROVIDER,
        ),
        ModelStage(
            stage="Command parsing",
            editable=False,
            options=[ModelOption(id=None, provider="OpenRouter", model=settings.openrouter_model_command)],
        ),
    ]
