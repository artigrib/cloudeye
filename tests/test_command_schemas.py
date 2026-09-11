"""CommandCreate validation: text-or-target_object_id requirement, the `from` alias
(world point to plan from instead of the scene's fixed robot_start), and radius bounds.
"""

import uuid

import pytest
from pydantic import ValidationError

from app.schemas import CommandCreate


def test_text_only_is_valid():
    body = CommandCreate(text="go to the sink")
    assert body.text == "go to the sink"
    assert body.target_object_id is None


def test_target_object_id_only_is_valid():
    oid = uuid.uuid4()
    body = CommandCreate(target_object_id=oid)
    assert body.target_object_id == oid
    assert body.text is None


def test_neither_text_nor_target_raises():
    with pytest.raises(ValidationError):
        CommandCreate()


def test_empty_text_and_no_target_raises():
    with pytest.raises(ValidationError):
        CommandCreate(text="")


def test_from_alias_populates_from_field():
    body = CommandCreate.model_validate({"target_object_id": str(uuid.uuid4()), "from": [1.5, -2.25]})
    assert body.from_ == [1.5, -2.25]


def test_from_defaults_to_none():
    body = CommandCreate(text="go to the sink")
    assert body.from_ is None


def test_from_requires_exactly_two_values():
    with pytest.raises(ValidationError):
        CommandCreate.model_validate({"text": "go to the sink", "from": [1.0]})


def test_radius_bounds_still_enforced():
    with pytest.raises(ValidationError):
        CommandCreate(text="go to the sink", radius=0.01)
    with pytest.raises(ValidationError):
        CommandCreate(text="go to the sink", radius=0.61)


def test_robot_id_defaults_to_none():
    body = CommandCreate(text="go to the sink")
    assert body.robot_id is None


def test_robot_id_and_radius_both_accepted_together():
    # The slider (radius) must keep working as an override on top of a selected
    # platform (robot_id) - the schema itself doesn't resolve precedence (that's
    # app.robots.resolve_radius_m's job), it just has to accept both at once.
    body = CommandCreate(text="go to the sink", robot_id="jackal", radius=0.2)
    assert body.robot_id == "jackal"
    assert body.radius == pytest.approx(0.2)
