"""resolve_object's name matching and tie-break ordering - plain SQL/attribute
matching only, no vector search, per the explicit spec constraint.
"""

import uuid

import pytest

from app.models import SceneObject
from app.services.command_service import ObjectNotFoundError, resolve_object, resolve_object_by_id


def make_object(name: str, *, is_fragment=False, num_views=1, num_points=100) -> SceneObject:
    return SceneObject(
        id=uuid.uuid4(),
        scene_id=uuid.uuid4(),
        name=name,
        pos_x=0.0, pos_y=0.0, pos_z=0.0,
        bbox_min_x=0.0, bbox_min_y=0.0, bbox_min_z=0.0,
        bbox_max_x=1.0, bbox_max_y=1.0, bbox_max_z=1.0,
        num_views=num_views,
        num_points=num_points,
        is_fragment=is_fragment,
    )


def test_exact_match():
    objs = [make_object("sink"), make_object("mirror")]
    assert resolve_object(objs, "sink").name == "sink"


def test_case_insensitive_match():
    objs = [make_object("Sink")]
    assert resolve_object(objs, "sink").name == "Sink"


def test_substring_match_query_in_name():
    objs = [make_object("pedestal sink")]
    assert resolve_object(objs, "sink").name == "pedestal sink"


def test_substring_match_name_in_query():
    objs = [make_object("sink")]
    assert resolve_object(objs, "the sink over there").name == "sink"


def test_no_match_raises_object_not_found():
    objs = [make_object("sink")]
    with pytest.raises(ObjectNotFoundError):
        resolve_object(objs, "chair")


def test_tiebreak_prefers_not_fragment():
    objs = [
        make_object("sink", is_fragment=True, num_views=20, num_points=10_000),
        make_object("sink", is_fragment=False, num_views=1, num_points=10),
    ]
    result = resolve_object(objs, "sink")
    assert result.is_fragment is False


def test_tiebreak_then_prefers_more_views():
    objs = [
        make_object("sink", is_fragment=False, num_views=3, num_points=10_000),
        make_object("sink", is_fragment=False, num_views=15, num_points=10),
    ]
    result = resolve_object(objs, "sink")
    assert result.num_views == 15


def test_tiebreak_then_prefers_more_points():
    objs = [
        make_object("sink", is_fragment=False, num_views=5, num_points=100),
        make_object("sink", is_fragment=False, num_views=5, num_points=99_000),
    ]
    result = resolve_object(objs, "sink")
    assert result.num_points == 99_000


def test_resolve_by_id_finds_exact_object():
    a, b = make_object("sink"), make_object("sink")  # same name, different ids
    assert resolve_object_by_id([a, b], a.id) is a
    assert resolve_object_by_id([a, b], b.id) is b


def test_resolve_by_id_raises_when_missing():
    objs = [make_object("sink")]
    with pytest.raises(ObjectNotFoundError):
        resolve_object_by_id(objs, uuid.uuid4())
