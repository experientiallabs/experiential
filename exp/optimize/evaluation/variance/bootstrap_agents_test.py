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


def test_a_partly_escaped_pointer_resolves_a_slashed_key_at_any_depth() -> None:
    """``/a~1b/interpreters//opt/venv/bin/python/packages/0`` finds the nested slashed key."""
    from exp.optimize.evaluation.variance.bootstrap_agents import canonical_pointer, pointer_hint

    state = {"a/b": {"interpreters": {"/opt/venv/bin/python": {"packages": ["pytest"]}}}}
    pointer = "/a~1b/interpreters//opt/venv/bin/python/packages/0"
    assert canonical_pointer(state, pointer) == (
        "/a~1b/interpreters/~1opt~1venv~1bin~1python/packages/0"
    )
    assert "'~1opt~1venv~1bin~1python'" in pointer_hint(state, "/a~1b/interpreters/python")


def test_every_role_submits_through_its_typed_contract() -> None:
    """Review findings take ``issue``, never an author's fact fields; schemas are objects."""
    from exp.optimize.evaluation.variance.bootstrap_agents import (
        research_answer_schema,
        reveal_answer_schema,
        review_answer_schema,
    )

    finding = review_answer_schema().input_schema["properties"]["findings"]["items"]
    assert "issue" in finding["required"] and finding["additionalProperties"] is False
    assert "deciding" not in finding["properties"]
    research = research_answer_schema().input_schema["properties"]
    assert research["status"]["enum"] == ["resolved", "unresolved"]
    conflict = reveal_answer_schema().input_schema["properties"]["conflicts"]["items"]
    assert set(conflict["required"]) == {"probe", "explanation"}


def test_bare_types_are_wrapped_and_a_schema_written_as_properties_is_lifted() -> None:
    """The two live verifier-schema mistakes are repaired deterministically and recorded."""
    from exp.optimize.evaluation.variance.bootstrap_agents import normalise_schema
    from exp.simulation.reconstruction.models import check_schema

    notes: list[str] = []
    schema = {
        "type": "object",
        "properties": {
            "files": {
                "type": "object",
                "properties": {"test_myre.py": {"properties": {"type": "object"}}},
            },
            "lines": {"type": "array", "items": {"items": {"properties": {"type": "string"}}}},
            "name": "string",
        },
    }
    fixed = normalise_schema(schema, "verifier.outcome_schema", notes)
    check_schema(fixed)
    files = fixed["properties"]["files"]["properties"]["test_myre.py"]
    assert files == {"type": "object"}
    assert fixed["properties"]["lines"]["items"]["items"] == {"type": "string"}
    assert fixed["properties"]["name"] == {"type": "string"}
    assert len(notes) == 3
    kept = {"type": "object", "properties": {"type": "string", "path": "string"}}
    assert normalise_schema(kept, "", [])["properties"] == {
        "type": {"type": "string"},
        "path": {"type": "string"},
    }


def test_transient_and_size_failures_are_redrawn_and_others_are_not() -> None:
    """5xx, 520, unknown-status and allowance failures redraw; a 400 propagates."""
    from exp.optimize.evaluation.variance.bootstrap_agents import redraw_cause
    from exp.runtime.models.providers.errors import ProviderRetryableResponseError
    from exp.runtime.models.providers.transport import ProviderTransportError

    assert redraw_cause(ProviderTransportError("x", status_code=520)) == "transport_520"
    assert redraw_cause(ProviderTransportError("x")) == "transport_unknown"
    assert redraw_cause(ProviderTransportError("x", status_code=400)) is None
    allowance = ProviderRetryableResponseError("x", hit_length_limit=True)
    assert redraw_cause(allowance) == "output_allowance"
