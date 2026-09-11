"""resolve_vocab_override(): the decision at the heart of the vocab cache - what (if
anything) goes into a job's `vocab_override` param, which is the one thing that
determines whether gpu/stage_vocab.py makes a fresh OpenRouter/Vertex call at all (see
its module docstring). Exercises the precedence documented on the function: an
explicit `Video.vocab_override` always wins; otherwise a `rerun` reuses
`Scene.vocab_entries` if there's anything cached; otherwise no override at all (a
normal fresh vocabulary call).

Pure-unit, no DB/GPU/network - matches this suite's convention (see
test_pipeline_orchestrator_retain_per_view.py: Video/Scene are plain SQLAlchemy models
instantiated standalone, never touching a session, purely for their attributes).
"""

from app.models import Scene, Video
from app.services.pipeline_orchestrator import resolve_vocab_override

CACHED_ENTRIES = [{"name": "chair", "size_class": "medium"}, {"name": "door", "size_class": "huge"}]


def _video(vocab_override=None) -> Video:
    return Video(
        filename="room.mp4", filepath="/tmp/room.mp4", size_bytes=1, format="mp4",
        vocab_override=vocab_override,
    )


def _scene(vocab_entries=None) -> Scene:
    return Scene(vocab_entries=vocab_entries)


def test_first_run_no_override_no_cache_returns_none():
    """Normal first run: no explicit override, nothing cached yet (fresh scene) ->
    fall through to a real vocabulary call."""
    assert resolve_vocab_override(_video(), _scene(), rerun=False) is None


def test_rerun_without_cache_falls_through_to_fresh_call():
    """A rerun on a scene that never got far enough to cache a vocabulary (e.g. its
    only prior attempt failed before vocab.json was written) has nothing to reuse -
    same as a first run, not an error."""
    assert resolve_vocab_override(_video(), _scene(vocab_entries=None), rerun=True) is None


def test_rerun_with_cache_reuses_it():
    """The actual cache-hit path: a rerun on a scene with a previously-resolved
    vocabulary reuses it verbatim, skipping the LLM call."""
    result = resolve_vocab_override(_video(), _scene(vocab_entries=CACHED_ENTRIES), rerun=True)
    assert result == CACHED_ENTRIES
    assert result is CACHED_ENTRIES  # reused as-is, not copied/rebuilt


def test_first_run_ignores_cache_even_if_present():
    """Cached vocab_entries only ever gets reused on an explicit rerun - a plain
    (non-rerun) processing pass never reads it, even if a stale value happens to be
    sitting on the scene row already."""
    assert resolve_vocab_override(_video(), _scene(vocab_entries=CACHED_ENTRIES), rerun=False) is None


def test_explicit_video_override_wins_over_cache_on_rerun():
    """An explicit vocab_override set at upload time is a deliberate user choice - a
    rerun must not silently replace it with whatever the scene resolved to on its own
    last run."""
    explicit = [{"name": "sofa", "size_class": "large"}]
    result = resolve_vocab_override(
        _video(vocab_override=explicit), _scene(vocab_entries=CACHED_ENTRIES), rerun=True
    )
    assert result == explicit


def test_explicit_video_override_wins_on_first_run_too():
    explicit = [{"name": "sofa", "size_class": "large"}]
    result = resolve_vocab_override(_video(vocab_override=explicit), _scene(), rerun=False)
    assert result == explicit


def test_empty_cached_entries_list_falls_through():
    """An empty (falsy) cached list is treated the same as no cache at all -
    scene_ingest.read_vocab_entries never actually persists an empty list (it returns
    None instead), but the function stays defensive about it regardless."""
    assert resolve_vocab_override(_video(), _scene(vocab_entries=[]), rerun=True) is None
