"""Bootstrap contracts are exercised by the real scripted flows in bootstrap_test.py."""


def test_a_long_source_trace_opens_clipped_with_an_inventory_count() -> None:
    """A 300-step session with large outputs opens small; the full trace stays readable."""
    import json

    from exp.optimize.evaluation.variance.bootstrap_agents import (
        OPENING_STEPS_HEAD,
        OPENING_STEPS_TAIL,
        _inventory_summary,
        _opening_view,
    )

    view = {
        "trace_id": "t",
        "steps": [
            {"step": i, "action": {"name": "bash", "arguments": {"cmd": "x" * 5000}}}
            for i in range(300)
        ],
    }
    opened = _opening_view(view)
    assert len(opened["steps"]) == OPENING_STEPS_HEAD + OPENING_STEPS_TAIL + 1
    assert opened["steps"][OPENING_STEPS_HEAD]["omitted_steps"] == 240
    assert len(json.dumps(opened)) < 200_000
    summary = _inventory_summary(frozenset({"trace:t", "trace:t:span:1", "rule:r1"}))
    assert summary["counts"] == {"rule": 1, "trace": 2}


def test_an_unescaped_slash_in_a_key_is_rewritten_only_when_one_location_is_meant() -> None:
    """``/Python/Node env/pip`` means ``/Python~1Node env/pip`` when no other key matches."""
    from exp.optimize.evaluation.variance.bootstrap_agents import (
        canonical_pointer,
        canonical_pointers,
        pointer_token,
    )

    state = {
        "Python/Node env": {"pip": "/usr/bin/pip", "interpreters": {"/opt/venv/bin/python": 1}},
        "Workspace": {"files": {"a.py": ""}},
    }
    assert pointer_token("Python/Node env") == "/Python~1Node env"
    assert canonical_pointer(state, "/Python/Node env/pip") == "/Python~1Node env/pip"
    assert (
        canonical_pointer(state, "/Python/Node env/interpreters//opt/venv/bin/python")
        == "/Python~1Node env/interpreters/~1opt~1venv~1bin~1python"
    )
    assert canonical_pointer(state, "/Workspace/files/a.py") == "/Workspace/files/a.py"
    assert canonical_pointer(state, "/Workspace/files/b.py") == "/Workspace/files/b.py"
    ambiguous = {"a/b": {"c": 1}, "a": {"b": {"c": 2}}}
    assert canonical_pointer(ambiguous, "/a/b/c") == "/a/b/c"
    assert canonical_pointer({"a/b": {"c": 1}}, "/a/b/c") == "/a~1b/c"
    answer = {
        "initial_state": state,
        "probe_paths": {"pip": "/Python/Node env/pip"},
        "facts": [{"fact_id": "f", "state_paths": ["/Python/Node env"]}],
    }
    fixed = canonical_pointers(answer)
    assert fixed["probe_paths"] == {"pip": "/Python~1Node env/pip"}
    assert fixed["facts"][0]["state_paths"] == ["/Python~1Node env"]
    assert answer["probe_paths"] == {"pip": "/Python/Node env/pip"}


def test_an_invalid_argument_policy_names_its_tool_and_the_schema_error() -> None:
    """The author is told which tool's policy is invalid and where, not only that one is."""
    import pytest
    from pydantic import ValidationError

    from exp.simulation.evaluation.tasks import TaskVerifier

    with pytest.raises(ValidationError, match=r"for tool 'bash': at properties/type"):
        TaskVerifier.model_validate(
            {
                "description": "d",
                "outcome_schema": {"type": "object"},
                "argument_schemas": {"bash": {"properties": {"type": "object"}}},
            }
        )
