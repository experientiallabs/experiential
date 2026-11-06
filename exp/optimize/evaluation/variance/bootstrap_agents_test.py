"""Starting-state evidence stays readable without duplicating full traces in each request."""

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import JsonValue

from exp.common.models import ModelRequest, ToolCall
from exp.optimize.evaluation.factory.authoring import author_request_bytes
from exp.optimize.evaluation.rules.agent_loop_test import (
    _model_reply,
    _native_loop,
    _script_loop,
)
from exp.optimize.evaluation.variance.bootstrap_agents import validate_draft
from exp.optimize.evaluation.variance.bootstrap_reads_test import material
from exp.optimize.evaluation.variance.bootstrap_test import _draft
from exp.optimize.evaluation.variance.evidence_tools_test import _ANSWER
from exp.simulation.rules.contracts_test import call, result, trace


def test_large_source_remains_readable_while_author_phases_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A complete source stays accessible without consuming every phase's working context."""
    evidence = trace(
        "source",
        tuple(
            span
            for index in range(6)
            for span in (
                call(f"c{index}", 2 * index, "read", {}, f"k{index}"),
                result(
                    f"r{index}",
                    2 * index + 1,
                    "read",
                    {"value": "x" * 18500},
                    f"k{index}",
                ),
            )
        ),
        declare=False,
    )
    source = replace(material(), sources={"source": evidence})
    loop = _native_loop(tmp_path, monkeypatch)
    outputs = [
        _model_reply(loop, [("record_plan", {"hypotheses": [], "planned_reads": []})]),
        _model_reply(loop, [("submit_draft", {"value": 2, "rationale": "y" * 20000})]),
        _model_reply(loop, [("record_self_check", {"issues": [], "revise": False})]),
        _model_reply(loop, [("submit", {"value": 2})]),
    ]
    captured: list[ModelRequest] = _script_loop(
        loop, monkeypatch, lambda request, index: outputs[index - 1]
    )
    transcript = loop.run(
        "Author from readable recorded evidence.",
        source.context(),
        (),
        lambda call: ("refused", "no reads requested", ()),
        _ANSWER,
        lambda draft, refs: None,
    )
    assert transcript.stop == "answered", transcript.failure
    assert len(captured) == 4
    assert all(author_request_bytes(loop.settings, request) <= 128000 for request in captured)
    status, body, refs = source.run(
        ToolCall(call_id="whole", name="read_trace", arguments={"trace_id": "source"}),
        256000,
    )
    assert status == "ok" and body == source.views["source"][0]
    assert "trace:source:span:r5" in refs
    status, selected, refs = source.run(
        ToolCall(
            call_id="part",
            name="read_trace_steps",
            arguments={"trace_id": "source", "start_index": 5, "count": 1},
        ),
        32000,
    )
    assert status == "ok" and isinstance(selected, dict)
    assert selected["coverage"]["selected_range_complete"] is True
    assert refs == ("trace:source:span:c5", "trace:source:span:r5")


def test_start_facts_require_delivered_evidence_not_an_inventory_listing() -> None:
    """A source locator becomes citable only after its complete evidence was delivered."""
    source = material()
    draft = _draft("auth", {"findings": []})
    facts = draft["facts"]
    assert isinstance(facts, list) and isinstance(facts[0], dict)
    facts[0]["references"] = ["trace:allowed:span:r1"]
    with pytest.raises(ValueError):
        validate_draft(draft, source, frozenset())
    assert validate_draft(draft, source, frozenset({"trace:allowed:span:r1"})).facts[
        0
    ].references == ("trace:allowed:span:r1",)


@pytest.mark.parametrize("projection", ["/tickets/note", "/tickets"])
def test_verifier_state_paths_need_a_container_projection(projection: str) -> None:
    """Future files are gradeable from an empty object, not from a scalar note."""
    draft = _draft("auth", {"findings": []})
    draft["facts"] = [
        {
            "fact_id": "unused",
            "statement": "Unverified tier.",
            "status": "unknown",
            "deciding": False,
        }
    ]
    draft["initial_state"] = {"tickets": {"note": "481-byte prompt file"}}
    draft["probe_paths"] = {"tickets": projection}
    draft["verifier"] = {
        "description": "The actor creates the result file.",
        "outcome_schema": {},
        "required_paths": ["/final_state/tickets/result.py"],
        "rules": [{"operator": "exists", "path": "/final_state/tickets/result.py"}],
    }
    if projection.endswith("/note"):
        with pytest.raises(ValueError, match="scalar"):
            validate_draft(draft, material(), frozenset())
    else:
        validate_draft(draft, material(), frozenset())


def test_verifier_cannot_require_an_unproduced_outcome_field() -> None:
    """Task-specific test results must come from the state projection."""
    draft = _draft("auth", {"findings": []})
    draft["facts"] = [
        {
            "fact_id": "unused",
            "statement": "Unverified tier.",
            "status": "unknown",
            "deciding": False,
        }
    ]
    verifier = draft["verifier"]
    assert isinstance(verifier, dict)
    verifier["required_paths"] = ["/pytest_status"]
    with pytest.raises(ValueError, match="workflow outcome"):
        validate_draft(draft, material(), frozenset())


def test_authored_start_cannot_pass_without_any_actor_action() -> None:
    """A precompleted state or trivial verifier fails the no-action control."""
    draft = _draft("auth", {"findings": []})
    draft["facts"] = [
        {
            "fact_id": "unused",
            "statement": "Unverified tier.",
            "status": "unknown",
            "deciding": False,
        }
    ]
    draft["initial_state"] = {"tickets": {"outcome": "right"}}
    with pytest.raises(ValueError, match="no-action"):
        validate_draft(draft, material(), frozenset())


def test_generated_probe_selects_a_full_domain_root() -> None:
    """An author cannot select a scalar note instead of its mutable domain."""
    from exp.optimize.evaluation.variance.bootstrap_agents import assemble_start_proposal
    from exp.optimize.evaluation.variance.bootstrap_test import _author_answer

    answer = _author_answer("auth", {"findings": []})
    answer["probe_paths"] = {"tickets": "/tickets/status"}
    with pytest.raises(ValueError, match="full domain roots"):
        assemble_start_proposal(answer)


@pytest.mark.parametrize(
    "broken",
    [
        "file contents",
        {"exit_code": 0, "passed": 4},
        {"files": "file contents", "pytest_last_run": None},
    ],
)
def test_generated_domain_rejects_clobbered_state(broken: JsonValue) -> None:
    """A file write or test result cannot replace the domain's state containers."""
    from exp.common.core.artifacts import JsonObject
    from exp.optimize.evaluation.variance.bootstrap_agents import assemble_start_proposal
    from exp.optimize.evaluation.variance.bootstrap_test import _author_answer
    from exp.runtime.environments.world_responses import validate_state_projection

    answer = _author_answer("auth", {"findings": []})
    answer["initial_state"] = {"tickets": {"files": {}, "pytest_last_run": None}}
    answer["probe_paths"] = {"tickets": "/tickets"}
    draft = assemble_start_proposal(answer)
    schemas: dict[str, JsonObject] = {
        row.name: row.state_schema for row in draft.manifest.domains if row.state_schema is not None
    }
    valid: JsonObject = {
        "tickets": {"files": {"myre.py": "code"}, "pytest_last_run": {"exit_code": 0}}
    }
    validate_state_projection(valid, schemas)
    with pytest.raises(ValueError, match="state violates schema"):
        validate_state_projection({"tickets": broken}, schemas)
