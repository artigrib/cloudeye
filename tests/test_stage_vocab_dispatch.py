"""run_openrouter()/run_vertex() must fail loudly (VocabProviderError), never fall back
to a *fabricated* substitute vocabulary - a scene built on a silently-substituted
vocabulary looks exactly like a real one (see docs/COMPARISON.md's provider-failure
decision and gpu/stage_vocab.py's module docstring). Covers the credential/no-frames
guard clauses for both providers. Network calls themselves are not mocked here for
these two - matches this project's existing test conventions (no network mocking infra
elsewhere either); these guard clauses are the actual logic under test.

dispatch_vocab() tests below cover the different thing this file's name promises but
didn't originally test: the actual dispatch/fallback decision (Vertex primary,
OpenRouter automatic fallback on a real Vertex failure - docs/DECISIONS.md,
2026-09-07). Those DO mock the provider (monkeypatching run_vertex/run_openrouter
themselves) rather than exercising real guard clauses, specifically so no network call
is ever reachable from a dispatch-logic test.
"""

import pytest

import gpu.stage_vocab as stage_vocab
from gpu.stage_vocab import VocabProviderError, dispatch_vocab, run_openrouter, run_vertex


@pytest.fixture
def paths(tmp_path):
    keyframes_dir = tmp_path / "keyframes"
    keyframes_dir.mkdir()
    return {"keyframes_dir": keyframes_dir}


def test_openrouter_no_api_key_raises(paths, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(VocabProviderError, match="openrouter.*OPENROUTER_API_KEY"):
        run_openrouter(paths, {})


def test_openrouter_no_keyframes_raises(paths, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    with pytest.raises(VocabProviderError, match="openrouter.*no keyframes"):
        run_openrouter(paths, {})


def test_vertex_no_credentials_raises(paths, monkeypatch):
    monkeypatch.delenv("VERTEX_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    with pytest.raises(VocabProviderError, match="vertex.*VERTEX_ACCESS_TOKEN"):
        run_vertex(paths, {})


def test_vertex_partial_credentials_still_raises(paths, monkeypatch):
    monkeypatch.setenv("VERTEX_ACCESS_TOKEN", "fake-token")
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    with pytest.raises(VocabProviderError, match="vertex.*VERTEX_ACCESS_TOKEN"):
        run_vertex(paths, {})


def test_vertex_no_keyframes_raises(paths, monkeypatch):
    monkeypatch.setenv("VERTEX_ACCESS_TOKEN", "fake-token")
    monkeypatch.setenv("GCP_PROJECT_ID", "fake-project")
    with pytest.raises(VocabProviderError, match="vertex.*no keyframes"):
        run_vertex(paths, {})


def test_default_vertex_model_matches_model_catalog():
    # These live in two separate processes (backend vs GPU host) with no shared import
    # path - only a text match keeps them from silently drifting apart.
    from app.services.model_catalog import VERTEX_VOCAB_MODEL
    from gpu.stage_vocab import DEFAULT_VERTEX_MODEL

    assert DEFAULT_VERTEX_MODEL == VERTEX_VOCAB_MODEL


# --- dispatch_vocab(): Vertex primary, OpenRouter automatic fallback ------------------
#
# These monkeypatch stage_vocab.run_vertex/run_openrouter themselves, so no network
# call is reachable from any of them regardless of what params/env look like.


def test_dispatch_vocab_override_skips_both_providers(paths, monkeypatch):
    calls = []
    monkeypatch.setattr(stage_vocab, "run_vertex", lambda *a, **k: calls.append("vertex") or (_ for _ in ()).throw(AssertionError("vertex should not be called")))
    monkeypatch.setattr(stage_vocab, "run_openrouter", lambda *a, **k: calls.append("openrouter") or (_ for _ in ()).throw(AssertionError("openrouter should not be called")))

    entries, source, model = dispatch_vocab(paths, {"vocab_override": [{"name": "chair", "size_class": "medium"}]})

    assert source == "override"
    assert model is None
    assert entries == [{"name": "chair", "size_class": "medium"}]
    assert calls == []


def test_dispatch_vocab_default_tries_vertex_first_and_uses_it_on_success(paths, monkeypatch):
    """Unset vocab_provider (the default) means Vertex primary - OpenRouter must not
    even be attempted when Vertex succeeds."""
    monkeypatch.setattr(stage_vocab, "run_vertex", lambda p, params: ([{"name": "bed", "size_class": "huge"}], "vertex", "google/gemma-4-26b-a4b-it-maas"))
    monkeypatch.setattr(stage_vocab, "run_openrouter", lambda p, params: (_ for _ in ()).throw(AssertionError("openrouter should not be called when vertex succeeds")))

    entries, source, model = dispatch_vocab(paths, {})

    assert source == "vertex"
    assert model == "google/gemma-4-26b-a4b-it-maas"
    assert entries == [{"name": "bed", "size_class": "huge"}]


def test_dispatch_vocab_explicit_vertex_tries_vertex_first_and_uses_it_on_success(paths, monkeypatch):
    monkeypatch.setattr(stage_vocab, "run_vertex", lambda p, params: ([{"name": "bed", "size_class": "huge"}], "vertex", "google/gemma-4-26b-a4b-it-maas"))
    monkeypatch.setattr(stage_vocab, "run_openrouter", lambda p, params: (_ for _ in ()).throw(AssertionError("openrouter should not be called when vertex succeeds")))

    entries, source, model = dispatch_vocab(paths, {"vocab_provider": "vertex"})

    assert source == "vertex"
    assert model == "google/gemma-4-26b-a4b-it-maas"


def test_dispatch_vocab_falls_back_to_openrouter_on_vertex_failure(paths, monkeypatch):
    """The actual fallback behavior this feature adds: a live Vertex failure
    (VocabProviderError) automatically falls back to a live OpenRouter call, logged."""
    calls = []

    def fake_vertex(p, params):
        calls.append("vertex")
        raise VocabProviderError("vertex: vocabulary call failed: fake network error")

    def fake_openrouter(p, params):
        calls.append("openrouter")
        return [{"name": "sofa", "size_class": "large"}], "openrouter", "z-ai/glm-5.3-flash"

    monkeypatch.setattr(stage_vocab, "run_vertex", fake_vertex)
    monkeypatch.setattr(stage_vocab, "run_openrouter", fake_openrouter)

    entries, source, model = dispatch_vocab(paths, {})

    assert calls == ["vertex", "openrouter"]  # vertex attempted first, then fallback
    assert source == "openrouter"
    assert model == "z-ai/glm-5.3-flash"
    assert entries == [{"name": "sofa", "size_class": "large"}]


def test_dispatch_vocab_explicit_openrouter_never_attempts_vertex(paths, monkeypatch):
    """An explicit "openrouter" choice (the Models selector opting out of Vertex) is
    honored exactly - no Vertex attempt, so nothing to fall back from either."""
    monkeypatch.setattr(stage_vocab, "run_vertex", lambda p, params: (_ for _ in ()).throw(AssertionError("vertex should not be called")))
    monkeypatch.setattr(stage_vocab, "run_openrouter", lambda p, params: ([{"name": "lamp", "size_class": "medium"}], "openrouter", "z-ai/glm-5.3-flash"))

    entries, source, model = dispatch_vocab(paths, {"vocab_provider": "openrouter"})

    assert source == "openrouter"
    assert model == "z-ai/glm-5.3-flash"


def test_dispatch_vocab_both_providers_failing_raises(paths, monkeypatch):
    """No fabricated substitute if both a live Vertex call and its OpenRouter fallback
    fail - still fails loudly, same as before this feature."""
    monkeypatch.setattr(stage_vocab, "run_vertex", lambda p, params: (_ for _ in ()).throw(VocabProviderError("vertex: down")))
    monkeypatch.setattr(stage_vocab, "run_openrouter", lambda p, params: (_ for _ in ()).throw(VocabProviderError("openrouter: down")))

    with pytest.raises(VocabProviderError, match="openrouter"):
        dispatch_vocab(paths, {})
