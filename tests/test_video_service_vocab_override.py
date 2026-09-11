"""parse_vocab_override() is the one validation gate between an upload's raw form
field and what gets stored on Video.vocab_override, then forwarded verbatim into
gpu/stage_vocab.py's params.json. It deliberately does NOT re-validate entry shape
(name/size_class) - that's build_vocab_entries()'s job on the GPU side - it only
answers "is this a JSON array at all", so a malformed multipart field fails loudly at
upload time instead of silently reaching the GPU as an unusable value.
"""

import pytest

from app.services.video_service import parse_vocab_override


def test_none_input_preserves_current_behavior():
    assert parse_vocab_override(None) is None


def test_empty_string_preserves_current_behavior():
    assert parse_vocab_override("") is None


def test_valid_json_array_of_strings():
    assert parse_vocab_override('["door", "curtain"]') == ["door", "curtain"]


def test_valid_json_array_of_dicts():
    raw = '[{"name": "door", "size_class": "huge"}]'
    assert parse_vocab_override(raw) == [{"name": "door", "size_class": "huge"}]


def test_malformed_json_raises_value_error():
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_vocab_override("[not json")


def test_non_array_json_raises_value_error():
    with pytest.raises(ValueError, match="must be a JSON array"):
        parse_vocab_override('{"name": "door"}')
