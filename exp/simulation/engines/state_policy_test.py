"""Write boundaries exclude ancestor replacement and transient forbidden writes."""

import pytest
from jsonschema import Draft202012Validator

from exp.common.core.artifacts import JsonObject
from exp.simulation.engines.state_policy import StateEditPolicy
from exp.simulation.engines.transition import ApplyEdits, ProposedObservation, ProposedTransition


@pytest.mark.parametrize(
    "path,allowed",
    [
        (["domains", "tickets"], True),
        (["domains", "tickets", "rows", 0], True),
        (["domains", "tickets", True], False),
        (["domains", "archive"], False),
        (["domains"], False),
        (["environment", "logical_time"], False),
        ([], False),
    ],
)
def test_native_paths_match_admission(path: list[str | int], allowed: bool) -> None:
    """Every unsupported path is rejected regardless of its proposed value."""
    policy = StateEditPolicy(roots=(("domains", "tickets"),))
    edit: JsonObject = {"op": "set", "path": path, "value": 0}
    proposal = ProposedTransition(
        effect=ApplyEdits(edits=[edit]),
        observations=(ProposedObservation(call_id="c", reply={}, encoding="json"),),
    )
    assert Draft202012Validator(policy.edit_schema()).is_valid([edit]) is allowed
    if allowed:
        policy.require(proposal)
    else:
        with pytest.raises(ValueError, match="writable state paths"):
            policy.require(proposal)


def test_read_only_policy_allows_only_empty_edits() -> None:
    """A read cannot rewrite even a same-value business field."""
    validator = Draft202012Validator(StateEditPolicy().edit_schema())
    assert validator.is_valid([])
    assert not validator.is_valid([{"op": "set", "path": ["a"], "value": 1}])


def test_copy_requires_both_read_authority_and_write_authority() -> None:
    """Copying does not grant access to hidden domains or a writable ancestor."""
    policy = StateEditPolicy(roots=(("domains", "output"),), read_roots=(("domains", "input"),))
    validator = Draft202012Validator(policy.edit_schema())
    edit = {"op": "copy", "from": ["domains", "input"], "path": ["domains", "output"]}
    assert validator.is_valid([edit])
    assert not validator.is_valid([{**edit, "from": ["domains", "secret"]}])
    assert not validator.is_valid([{**edit, "from": ["domains"]}])
    assert not validator.is_valid([{**edit, "path": ["domains", "input"]}])
    assert not validator.is_valid([{**edit, "value": "invented"}])


def test_generation_schema_does_not_grant_native_write_authority() -> None:
    """Provider-compatible proposals stay untrusted at the native scope boundary."""
    policy = StateEditPolicy(roots=(("domains", "output"),), read_roots=(("call_arguments",),))
    validator = Draft202012Validator(policy.generation_schema())
    for path, allowed in (
        (["domains", "output", "file"], True),
        (["call_arguments", "content"], False),
    ):
        edit: JsonObject = {"op": "copy", "from": ["call_arguments", "content"], "path": path}
        assert validator.is_valid([edit])
        proposal = ProposedTransition(
            effect=ApplyEdits(edits=[edit]),
            observations=(ProposedObservation(call_id="c", reply={}, encoding="json"),),
        )
        if allowed:
            policy.require(proposal)
        else:
            with pytest.raises(ValueError, match="writable state paths"):
                policy.require(proposal)
