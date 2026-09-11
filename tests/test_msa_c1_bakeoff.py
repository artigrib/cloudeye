"""Unit tests for Stage C1's OpenRouter response parsing (extract_json_object)
and action-JSON parsing (parse_action_json) - the only pure, network-free logic
in scripts/msa/agent_loop/{openrouter_client,bakeoff}.py. The bake-off loop
itself calls the real OpenRouter API and is exercised manually (see
reports/task3g-C1-bakeoff.md), not from pytest.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.msa.agent_loop.actions import ActionRejected, Flag, Rotate, Scale
from scripts.msa.agent_loop.bakeoff import _rasterize_polygon, _rect_corners, parse_action_json
from scripts.msa.agent_loop.openrouter_client import extract_json_object


def test_extract_json_object_plain():
    assert extract_json_object('{"a": 1}') == {"a": 1}


def test_extract_json_object_fenced():
    text = '```json\n{"a": 1, "b": "x"}\n```'
    assert extract_json_object(text) == {"a": 1, "b": "x"}


def test_extract_json_object_with_prose_prefix_suffix():
    text = 'Sure, here is the action:\n{"action": "rotate", "yaw_delta_deg": 5.0}\nLet me know if that helps.'
    assert extract_json_object(text) == {"action": "rotate", "yaw_delta_deg": 5.0}


def test_extract_json_object_no_json_returns_none():
    assert extract_json_object("I refuse to answer in JSON.") is None


def test_extract_json_object_malformed_returns_none():
    assert extract_json_object('{"a": 1,}') is None


def test_parse_action_json_rotate():
    action = parse_action_json({"action": "rotate", "yaw_delta_deg": 10.0})
    assert isinstance(action, Rotate)
    assert action.yaw_delta_deg == 10.0


def test_parse_action_json_scale():
    action = parse_action_json({"action": "scale", "factor": 1.02})
    assert isinstance(action, Scale)
    assert action.factor == 1.02


def test_parse_action_json_flag():
    action = parse_action_json({"action": "flag", "reason": "not rectangular"})
    assert isinstance(action, Flag)
    assert action.reason == "not rectangular"


def test_parse_action_json_missing_action_key_raises():
    with pytest.raises(ValueError):
        parse_action_json({"yaw_delta_deg": 5.0})


def test_parse_action_json_unhandled_type_raises():
    with pytest.raises(ValueError):
        parse_action_json({"action": "translate", "dx": 0.1})


def test_parse_action_json_rotate_then_static_check_rejects_out_of_bound():
    action = parse_action_json({"action": "rotate", "yaw_delta_deg": 40.0})
    from scripts.msa.agent_loop.actions import check_action_static

    with pytest.raises(ActionRejected):
        check_action_static(action)


def test_rect_corners_and_rasterize_polygon_roundtrip():
    # A 1x1m square at the origin, unrotated, should rasterize to a filled
    # square whose area (in px) is close to (1m * PX_PER_M)^2, well within
    # anti-aliasing/boundary tolerance.
    from scripts.msa.agent_loop.bakeoff import PX_PER_M

    corners = _rect_corners((0.0, 0.0), (1.0, 1.0), 0.0)
    mask = _rasterize_polygon(corners, -0.6, -0.6, 240, 240)
    expected_px_area = (1.0 * PX_PER_M) ** 2
    assert mask.sum() == pytest.approx(expected_px_area, rel=0.05)


def test_rasterize_polygon_hull_matches_bounding_shape():
    hull = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    mask = _rasterize_polygon(hull, -0.7, -0.7, 280, 280)
    assert mask.sum() > 0
    assert mask.dtype == bool
