"""extract_json must handle the range of ways an LLM wraps its JSON response - fenced,
prose-wrapped, or outright malformed - per the spec's explicit requirement.
"""

import pytest

from app.services.vlm_client import VLMError, extract_json


def test_plain_json():
    assert extract_json('{"action": "goto", "target": "sink"}') == {
        "action": "goto",
        "target": "sink",
    }


def test_json_fenced_with_language_tag():
    text = '```json\n{"action": "goto", "target": "sink"}\n```'
    assert extract_json(text) == {"action": "goto", "target": "sink"}


def test_json_fenced_without_language_tag():
    text = '```\n{"action": "goto", "target": "sink"}\n```'
    assert extract_json(text) == {"action": "goto", "target": "sink"}


def test_leading_prose():
    text = 'Here is the JSON response:\n{"action": "goto", "target": "sink"}'
    assert extract_json(text) == {"action": "goto", "target": "sink"}


def test_trailing_prose():
    text = '{"action": "goto", "target": "sink"}\nLet me know if you need anything else!'
    assert extract_json(text) == {"action": "goto", "target": "sink"}


def test_leading_and_trailing_prose_with_fence():
    text = 'Sure, here you go:\n```json\n{"action": "goto", "target": "sink"}\n```\nHope that helps!'
    assert extract_json(text) == {"action": "goto", "target": "sink"}


def test_nested_object_finds_outer_boundary_not_inner():
    text = '{"action": "take", "extra": {"nested": true}, "target": "sink"}'
    result = extract_json(text)
    assert result["target"] == "sink"
    assert result["extra"] == {"nested": True}


def test_error_shape():
    assert extract_json('{"error": "no such object"}') == {"error": "no such object"}


def test_no_json_object_raises_vlm_error():
    with pytest.raises(VLMError):
        extract_json("I'm sorry, I don't understand the request.")


def test_malformed_json_raises_vlm_error():
    with pytest.raises(VLMError):
        extract_json('{"action": "goto", "target": }')


def test_unbalanced_braces_raises_vlm_error():
    with pytest.raises(VLMError):
        extract_json('{"action": "goto"')
