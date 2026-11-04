"""Tests for the read-only evidence tools, usage measurement and the bounded analysis loop."""

from __future__ import annotations

import copy
import json
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from exp.common.core.artifacts import ArtifactInput, JsonObject, sha256_json
from exp.common.models import (
    AssistantAction,
    ModelResponse,
    ModelSnapshot,
    NumericMeasurement,
    ToolCall,
    Usage,
    load_model_catalog,
    write_model_catalog,
)
from exp.common.project import ProjectStore
from exp.common.tasks import ToolSchema
from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.variance.evidence import (
    EstimationPool,
    EvidencePacket,
    execute_evidence_tool,
)
from exp.optimize.evaluation.variance.evidence_tools import (
    AGENTIC_PROFILE,
    TOOL_RESULT_TYPE,
    TRANSCRIPT_TYPE,
    AnalysisTranscript,
    EvidenceBudget,
    EvidenceToolProfile,
    MeasuredUsage,
    response_usage,
    run_analysis,
    total_usage,
)
from exp.optimize.evaluation.variance.role_prompts import PromptAssignment, assignment
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.runtime.models.spend_ceiling import (
    MAXIMUM_UNPRICED_ATTEMPTS,
    OperationSpendCeiling,
    UnpricedDispatchRefused,
    install_spend_ceiling,
    operation_spend_scope,
)

_REF = ArtifactInput(artifact_id="checkpoint-a", sha256="a" * 64)
_DOCUMENTS: JsonObject = {
    "group": {"success_rate": 0.5},
    "verifier": {"goal": "row deleted"},
    "attempt:0/step:1": {"action": {"name": "edit"}, "status": "valid"},
    "grade:0": {"grade": {"status": "pass"}},
    "rule:orders.delete": {"statement": "Deleting a row needs a lock."},
    "point:1": {"point": 1, "actor_calls_spent": 1, "public_prefix": [{"m": i} for i in range(5)]},
}
_PACKET = EvidencePacket(
    pool=EstimationPool(branch=0, points={0: _REF, 1: _REF}),
    payload={"full": "packet"},
    refs=frozenset(_DOCUMENTS),
    documents=dict(_DOCUMENTS),
    overview={"locators": sorted(_DOCUMENTS)},
)
_ANSWER = ToolSchema(
    name="submit",
    description="Typed answer",
    input_schema={"type": "object", "properties": {"value": {"type": "integer"}}},
)
_TOOLS = EvidenceToolProfile(
    mode="read_only_tools", budget=EvidenceBudget(model_calls=3, tool_calls=2)
)


def _call(name: str, arguments: JsonObject) -> ToolCall:
    """One tool call."""
    return ToolCall(call_id="c", name=name, arguments=arguments)


def test_profiles_are_coherent_and_the_forced_profile_is_the_default() -> None:
    """A forced response makes one call; tool profiles can read and still reach an answer."""
    assert EvidenceToolProfile().mode == "forced_response"
    assert AGENTIC_PROFILE.mode == "agentic" and AGENTIC_PROFILE.budget.model_calls >= 4
    with pytest.raises(ValueError, match="one model call"):
        EvidenceToolProfile(budget=EvidenceBudget(model_calls=2))
    with pytest.raises(ValueError, match="two model calls"):
        EvidenceToolProfile(mode="read_only_tools")
    with pytest.raises(ValueError, match="four model calls"):
        EvidenceToolProfile(mode="agentic", budget=EvidenceBudget(model_calls=3, tool_calls=1))


def test_tools_read_exact_evidence_and_refuse_everything_else() -> None:
    """Exact reads succeed; unknown tools, locators, bad arguments and big results are refused."""
    before = copy.deepcopy(_PACKET.documents)
    documents = _PACKET.documents

    assert execute_evidence_tool(
        documents, _call("read_trace_step", {"attempt": 0, "step": 1}), maximum_bytes=4096
    ) == ("ok", {"locator": "attempt:0/step:1", "content": _DOCUMENTS["attempt:0/step:1"]})
    status, prefix = execute_evidence_tool(
        documents,
        _call("read_checkpoint_prefix", {"point": 1, "start": 3, "count": 9}),
        maximum_bytes=4096,
    )
    assert status == "ok" and isinstance(prefix, dict)
    assert prefix["messages"] == [{"m": 3}, {"m": 4}] and prefix["messages_total"] == 5
    assert execute_evidence_tool(
        documents, _call("read_rule", {"rule_id": "orders.delete"}), maximum_bytes=4096
    )[0] == ("ok")
    assert execute_evidence_tool(
        documents, _call("read_grader_evidence", {"attempt": 0}), maximum_bytes=4096
    )[0] == ("ok")
    listed = execute_evidence_tool(
        documents, _call("list_evidence", {"prefix": "attempt:"}), maximum_bytes=4096
    )
    assert listed == ("ok", {"prefix": "attempt:", "locators": ["attempt:0/step:1"]})

    refusals = (
        (_call("execute_action", {"name": "delete"}), "not a read-only evidence tool"),
        (_call("read_evidence", {"locator": "reference:solution"}), "was not supplied"),
        (_call("read_trace_step", {"attempt": "0", "step": 1}), "malformed"),
        (_call("read_grader_evidence", {}), "malformed"),
    )
    for call, reason in refusals:
        status, detail = execute_evidence_tool(documents, call, maximum_bytes=4096)
        assert status == "refused" and reason in str(detail)
    status, detail = execute_evidence_tool(
        documents, _call("read_evidence", {"locator": "point:1"}), maximum_bytes=64
    )
    assert status == "refused" and "narrow it" in str(detail)
    assert documents == before


def test_usage_is_read_from_the_response_and_priced_without_assumptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Observed charges win, catalog rates price reported usage, and unknowns stay unpriced."""
    project, _, _ = job_fixture(tmp_path, monkeypatch)
    price_judge(project)
    catalog = load_model_catalog(project.model_catalog_path)
    model = RuntimeModelCatalog(catalog, environment={}).snapshot("judge")[0]

    priced = response_usage(
        catalog, "judge", _response(model, Usage(input_tokens=1000, output_tokens=500))
    )
    assert priced == MeasuredUsage(
        calls=1, measured_calls=1, input_tokens=1000, output_tokens=500, priced_usd=0.002
    )
    unreported = response_usage(catalog, "judge", _response(model, None))
    assert unreported.unpriced_calls == 1 and unreported.measured_calls == 0
    unpriced = response_usage(
        catalog, "world", _response(model, Usage(input_tokens=1, output_tokens=1))
    )
    assert unpriced.unpriced_calls == 1 and unpriced.priced_usd == 0.0
    observed = _response(model, Usage(input_tokens=1, output_tokens=1)).model_copy(deep=True)
    observed = observed.model_copy(
        update={
            "economics": observed.economics.model_copy(
                update={"cost_usd": NumericMeasurement(value=0.25, provenance="observed")}
            )
        }
    )
    assert response_usage(catalog, "world", observed).priced_usd == 0.25
    assert total_usage([priced, unreported]).calls == 2


def test_the_tool_loop_persists_every_step_answers_and_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requests, tool results, measured usage and the answer are saved; replay makes no call."""
    project, settings, transport = _loop_fixture(
        tmp_path,
        monkeypatch,
        [
            _reply([("read_trace_step", {"attempt": 0, "step": 1}), ("run_shell", {"cmd": "rm"})]),
            _reply([("submit", {"value": 3})]),
        ],
    )

    reference, transcript = _run(project, settings, _TOOLS)

    assert transcript.stop == "answered" and transcript.answer == {"value": 3}
    assert [call.forced_answer for call in transcript.calls] == [False, True]
    assert [(t.name, t.status) for t in transcript.tools] == [
        ("read_trace_step", "ok"),
        ("run_shell", "refused"),
    ]
    assert transcript.usage.calls == 2 and transcript.usage.input_tokens == 200
    assert transcript.usage.unpriced_calls == 0 and transcript.usage.priced_usd > 0
    store = project.artifacts
    assert store.read(reference.artifact_id).manifest.artifact_type == TRANSCRIPT_TYPE
    for record in transcript.tools:
        assert store.read(record.result.artifact_id).manifest.artifact_type == TOOL_RESULT_TYPE
    for call in transcript.calls:
        intent = store.read(call.response.artifact_id).manifest.inputs[0]
        assert store.read(intent.artifact_id).manifest.artifact_type == "author-intent"
    second = json.loads(json.dumps(transport.requests[1].payload))
    assert any(message.get("role") == "tool" for message in second["messages"])
    assert "edit" in json.dumps(second["messages"])

    assert _run(project, settings, _TOOLS) == (reference, transcript)
    assert len(transport.requests) == 2


def test_the_last_call_forces_the_answer_once_the_tool_budget_is_spent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With one tool call allowed, the next call forces the answer tool."""
    profile = EvidenceToolProfile(
        mode="read_only_tools", budget=EvidenceBudget(model_calls=4, tool_calls=1)
    )
    project, settings, transport = _loop_fixture(
        tmp_path,
        monkeypatch,
        [
            _reply(
                [
                    ("read_rule", {"rule_id": "orders.delete"}),
                    ("read_grader_evidence", {"attempt": 0}),
                ]
            ),
            _reply([("submit", {"value": 1})]),
        ],
    )

    _, transcript = _run(project, settings, profile)

    assert transcript.stop == "answered"
    assert [t.status for t in transcript.tools] == ["ok", "refused"]
    assert transcript.calls[-1].forced_answer
    assert transport.requests[1].payload["tool_choice"] == {
        "type": "function",
        "function": {"name": "submit"},
    }


def test_an_exhausted_budget_stops_the_draw_explicitly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A draw that cannot afford another call stops with the exhausted dimension named."""
    profile = EvidenceToolProfile(
        mode="read_only_tools",
        budget=EvidenceBudget(model_calls=3, tool_calls=4, input_tokens=50_000),
    )
    project, settings, transport = _loop_fixture(
        tmp_path,
        monkeypatch,
        [_reply([("read_evidence", {"locator": "group"})], prompt_tokens=60_000)],
    )

    _, transcript = _run(project, settings, profile)

    assert transcript.stop == "budget_exhausted" and transcript.answer is None
    assert "input_tokens" in transcript.exhausted
    assert len(transport.requests) == 1


def test_unpriced_usage_makes_the_dollar_budget_unverifiable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a price no further call is bought against a dollar budget."""
    project, settings, transport = _loop_fixture(
        tmp_path, monkeypatch, [_reply([("read_evidence", {"locator": "group"})])], priced=False
    )

    _, transcript = _run(project, settings, _TOOLS)

    assert transcript.stop == "budget_exhausted"
    assert transcript.exhausted == ("maximum_usd_unverifiable",)
    assert len(transport.requests) == 1


def price_judge(project: ProjectStore) -> str:
    """Give the fixture judge alias per-million rates and return the new catalog digest."""
    catalog = load_model_catalog(project.model_catalog_path)
    record = catalog.models["judge"]
    assert record.capabilities is not None
    priced = record.model_copy(
        update={
            "capabilities": record.capabilities.model_copy(
                update={
                    "input_cost_per_million_tokens_usd": 1.0,
                    "output_cost_per_million_tokens_usd": 2.0,
                }
            )
        }
    )
    catalog = catalog.model_copy(update={"models": {**catalog.models, "judge": priced}})
    write_model_catalog(project.model_catalog_path, catalog)
    return sha256_json(load_model_catalog(project.model_catalog_path).model_dump(mode="json"))


def _response(model: ModelSnapshot, usage: Usage | None) -> ModelResponse:
    """One completed response with the given usage."""
    return ModelResponse.completed(
        output=AssistantAction(content="x"),
        configured_model=model,
        served_model_id=None,
        usage=usage,
        latency_seconds=0.1,
    )


def _reply(calls: list[tuple[str, JsonObject]], *, prompt_tokens: int = 100) -> JsonHttpResponse:
    """One scripted role reply making the given tool calls, reporting usage."""
    return JsonHttpResponse(
        status_code=200,
        body={
            "id": "reply",
            "model": "judge-v1",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "type": "function",
                                "id": f"call-{i}",
                                "function": {"name": name, "arguments": json.dumps(arguments)},
                            }
                            for i, (name, arguments) in enumerate(calls)
                        ],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 50,
                "total_tokens": prompt_tokens + 50,
            },
        },
    )


def _loop_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replies: list[JsonHttpResponse],
    *,
    priced: bool = True,
) -> tuple[ProjectStore, AuthorSettings, ScriptedJsonTransport]:
    """Return a project, judge settings and the scripted role transport."""
    monkeypatch.setattr("socket.socket.connect", _forbid_connect)
    project, job, _ = job_fixture(tmp_path, monkeypatch)
    digest = price_judge(project) if priced else job.catalog_sha256
    catalog = RuntimeModelCatalog(load_model_catalog(project.model_catalog_path), environment={})
    settings = AuthorSettings(
        model_alias="judge",
        model=catalog.snapshot("judge")[0],
        catalog_sha256=digest,
        output_tokens=1024,
        deadline=job.deadline,
    )
    transport = ScriptedJsonTransport(tuple(replies))
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: transport
    )
    return project, settings, transport


def _positive(draft: JsonObject) -> str | None:
    """Accept a draft whose value is a positive integer."""
    value = draft.get("value")
    return None if isinstance(value, int) and value > 0 else "value must be positive"


def _message(transport: ScriptedJsonTransport, request: int, index: int) -> str:
    """Return the text content of one message of one sent request."""
    messages = transport.requests[request].payload["messages"]
    assert isinstance(messages, list) and isinstance(messages[index], dict)
    content = messages[index]["content"]
    assert isinstance(content, str)
    return content


def _run(
    project: ProjectStore,
    settings: AuthorSettings,
    profile: EvidenceToolProfile,
    *,
    prompt: PromptAssignment | None = None,
    packet: EvidencePacket = _PACKET,
) -> tuple[ArtifactInput, AnalysisTranscript]:
    """Run the loop over the fixture packet; a draft is valid when its value is positive."""
    return run_analysis(
        project,
        settings,
        profile,
        instruction="Read and answer.",
        packet=packet,
        content={"task": "estimate"},
        answer=_ANSWER,
        identity="draw-0",
        sources=(),
        code_revision="test",
        prompt=prompt,
        check=_positive,
    )


_RICH: JsonObject = {
    **_DOCUMENTS,
    "attempt:0": {"attempt": 0, "grade_status": "pass"},
    "attempt:1": {"attempt": 1, "grade_status": "fail"},
    "attempt:1/step:1": {"action": {"name": "edit"}, "status": "valid"},
    "attempt:0/step:2": {"action": {"name": "lock"}, "status": "valid"},
    "attempt:1/step:2": {"action": {"name": "delete"}, "status": "error"},
    "world:1": {"current": {"business_state": {"orders": {"7": {"locked": False}}}}},
    "environment": {"public_task": {"instruction": "Delete order 7."}},
    "tool:edit": {"name": "edit"},
    "contract:edit": {"effect": "writes a row"},
    "reference:policy": {"kind": "task_specification", "content": {"rules": ["lock first"]}},
}
_RICH_PACKET = _PACKET.model_copy(update={"documents": _RICH, "refs": frozenset(_RICH)})
_AGENTIC = EvidenceToolProfile(mode="agentic", budget=EvidenceBudget(model_calls=8, tool_calls=4))


def test_new_tools_read_world_state_contracts_comparisons_and_name_missing_material() -> None:
    """World state by pointer, contracts, comparisons and divergence read; absent tools refuse."""
    run = lambda name, arguments, size=4096: execute_evidence_tool(  # noqa: E731
        _RICH, _call(name, arguments), maximum_bytes=size
    )
    status, state = run("read_world_state", {"point": 1, "path": "/current/business_state/orders"})
    assert status == "ok" and isinstance(state, dict)
    assert state["content"] == {"7": {"locked": False}}
    status, detail = run("read_world_state", {"point": 1, "path": ""}, 64)
    assert status == "refused" and "children: ['current']" in str(detail)
    assert run("read_world_state", {"point": 1, "path": "/missing"})[0] == "refused"
    assert run("read_world_state", {"point": 4, "path": ""}) == (
        "refused",
        "locator world:4 was not supplied",
    )
    assert run("read_tool_contract", {"tool": "edit"}) == (
        "ok",
        {"tool": {"name": "edit"}, "contract": {"effect": "writes a row"}, "not_supplied": []},
    )
    assert run("read_reference", {"name": "policy", "path": "/content/rules/0"})[1] == {
        "locator": "reference:policy",
        "path": "/content/rules/0",
        "content": "lock first",
    }
    assert run("read_environment", {})[0] == "ok"
    status, compared = run("compare_attempts", {"attempts": [0, 1], "start": 1, "count": 2})
    assert status == "ok" and isinstance(compared, dict)
    assert [row["same_action"] for row in compared["steps"]] == [True, False]  # type: ignore[index]
    status, splits = run("list_divergence_points", {})
    assert status == "ok" and isinstance(splits, dict)
    (split,) = splits["divergence_points"]  # type: ignore[misc]
    assert split["step"] == 2 and split["shared_by"] == [0, 1]  # type: ignore[index]
    assert run("read_step_labels", {"attempt": 0, "step": 1})[1] == (
        "no traceml-labels reference was registered; labels not supplied"
    )
    status, detail = run(
        "search_world_traces", {"situation": "delete order", "message": "done", "top_k": 1}
    )
    assert status == "refused" and "no world-learning retrieval index" in str(detail)


def test_the_agentic_loop_persists_every_phase_and_replays_without_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan, reads, draft, self check with an extra read, and final are all saved and replay."""
    prompt = assignment("selector", "failure_forensics", 2)
    project, settings, transport = _loop_fixture(
        tmp_path,
        monkeypatch,
        [
            _reply(
                [("record_plan", {"hypotheses": ["lock missing"], "planned_reads": ["world:1"]})]
            ),
            _reply([("read_world_state", {"point": 1, "path": "/current"})]),
            _reply([("submit_draft", {"value": 0})]),
            _reply([("list_divergence_points", {})]),
            _reply([("record_self_check", {"issues": ["value must be positive"], "revise": True})]),
            _reply([("submit", {"value": 2})]),
        ],
    )

    reference, transcript = _run(project, settings, _AGENTIC, prompt=prompt, packet=_RICH_PACKET)

    assert transcript.stop == "answered" and transcript.answer == {"value": 2}
    assert [call.phase for call in transcript.calls] == [
        "plan",
        "investigate",
        "investigate",
        "self_check",
        "self_check",
        "final",
    ]
    assert [call.forced for call in transcript.calls] == [
        "record_plan",
        None,
        None,
        None,
        None,
        "submit",
    ]
    assert [(row.phase, row.call) for row in transcript.outputs] == [
        ("plan", 0),
        ("draft", 2),
        ("self_check", 4),
    ]
    assert transcript.outputs[1].host_validation == "value must be positive"
    assert [(t.name, t.call, t.status) for t in transcript.tools] == [
        ("read_world_state", 1, "ok"),
        ("list_divergence_points", 3, "ok"),
    ]
    assert transcript.calls[3].phase == "self_check" and transcript.prompt == prompt
    system = _message(transport, 0, 0)
    assert "STANCE (failure_forensics)" in system and prompt.stance in system
    opening = json.loads(_message(transport, 0, 1))
    assert opening["phase"] == "plan" and opening["world_learning_retrieval"]["available"] is False
    check_turn = json.dumps(transport.requests[3].payload)
    assert "value must be positive" in check_turn and "SELF CHECK" in check_turn

    assert _run(project, settings, _AGENTIC, prompt=prompt, packet=_RICH_PACKET) == (
        reference,
        transcript,
    )
    assert len(transport.requests) == 6
    with pytest.raises(ValueError, match="different profile or prompt"):
        _run(project, settings, _AGENTIC, prompt=assignment("selector", "pivot_and_return", 2))


def test_budget_exhaustion_mid_loop_skips_phases_or_stops_explicitly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without room for later phases the loop skips to the final answer, or stops without one."""
    profile = EvidenceToolProfile(
        mode="agentic",
        budget=EvidenceBudget(model_calls=8, tool_calls=4, input_tokens=50_000),
    )
    plan = _reply([("record_plan", {"hypotheses": [], "planned_reads": []})], prompt_tokens=45_000)
    project, settings, transport = _loop_fixture(
        tmp_path, monkeypatch, [plan, _reply([("submit", {"value": 1})])]
    )

    _, transcript = _run(project, settings, profile)

    assert transcript.stop == "answered" and transcript.skipped == ("investigate", "self_check")
    assert [call.phase for call in transcript.calls] == ["plan", "final"]
    assert "investigate" in _message(transport, 1, -1)

    plan = _reply([("record_plan", {"hypotheses": [], "planned_reads": []})], prompt_tokens=49_900)
    project, settings, transport = _loop_fixture(tmp_path / "second", monkeypatch, [plan])

    _, stopped = _run(project, settings, profile)

    assert stopped.stop == "budget_exhausted" and stopped.answer is None
    assert "input_tokens" in stopped.exhausted and stopped.outputs[0].phase == "plan"
    assert len(transport.requests) == 1


def test_an_agentic_role_that_ignores_a_forced_phase_tool_fails_explicitly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A forced plan answered with anything else is a protocol failure, never retried."""
    project, settings, transport = _loop_fixture(
        tmp_path, monkeypatch, [_reply([("submit", {"value": 1})])]
    )

    _, transcript = _run(project, settings, _AGENTIC)

    assert transcript.stop == "protocol_failure" and "record_plan" in str(transcript.failure)
    assert len(transport.requests) == 1


@pytest.mark.parametrize("allowance", [None, 0, MAXIMUM_UNPRICED_ATTEMPTS + 1])
def test_request_capped_draw_requires_finite_operation_authorization_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, allowance: int | None
) -> None:
    """No operation, zero allowance or summed grants over the cap cannot buy a call."""
    project, settings, transport = _loop_fixture(tmp_path, monkeypatch, [], priced=False)
    profile = EvidenceToolProfile(budget=EvidenceBudget(maximum_usd=None))
    with operation_spend_scope():
        if allowance is not None:
            ceiling = OperationSpendCeiling(tmp_path / "spend.sqlite", tariffs={})
            ceiling.authorize("first", ceiling_usd=1.0, unpriced_attempts=min(allowance, 100_000))
            if allowance > MAXIMUM_UNPRICED_ATTEMPTS:
                ceiling.authorize("second", ceiling_usd=1.0, unpriced_attempts=1)
            install_spend_ceiling(ceiling)
        with pytest.raises(UnpricedDispatchRefused, match="active finite"):
            _run(project, settings, profile)
    assert transport.requests == []
    assert not project.artifacts.exists("draw-0-request-capped-intent")


def test_approved_unpriced_draw_keeps_unknown_usage_and_replays_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bounded unpriced operation completes two calls and exact replay buys none."""
    project, settings, transport = _loop_fixture(
        tmp_path,
        monkeypatch,
        [_reply([("read_evidence", {"locator": "group"})]), _reply([("submit", {"value": 2})])],
        priced=False,
    )
    profile = EvidenceToolProfile(
        mode="read_only_tools", budget=EvidenceBudget(model_calls=2, tool_calls=1, maximum_usd=None)
    )
    ceiling = OperationSpendCeiling(tmp_path / "spend.sqlite", tariffs={})
    ceiling.authorize_policy(ceiling_usd=1.0, unpriced_attempts=2)
    with operation_spend_scope():
        install_spend_ceiling(ceiling)
        reference, transcript = _run(project, settings, profile)
    assert transcript.stop == "answered" and transcript.answer == {"value": 2}
    assert transcript.usage.unpriced_calls == 2 and transcript.usage.priced_usd == 0.0
    assert all(call.usage.unpriced_calls == 1 for call in transcript.calls)
    assert transcript.profile.budget.maximum_usd is None
    assert all("request-capped" in call.response.artifact_id for call in transcript.calls)
    assert ceiling.summary().unpriced_attempts == 2
    assert _run(project, settings, profile) == (reference, transcript)
    assert len(transport.requests) == 2
    assert transport.requests[-1].payload["tool_choice"] == {
        "type": "function",
        "function": {"name": "submit"},
    }
    with pytest.raises(ValueError, match="different profile"):
        _run(project, settings, _TOOLS)


@pytest.mark.parametrize("dimension", ["input_tokens", "output_tokens", "wall_clock_seconds"])
def test_request_capped_draw_still_enforces_token_and_time_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dimension: str
) -> None:
    """Removing a draw dollar cap preserves each finite token and elapsed-time bound."""
    project, settings, transport = _loop_fixture(
        tmp_path,
        monkeypatch,
        [_reply([("read_evidence", {"locator": "group"})], prompt_tokens=60_000)],
        priced=False,
    )
    budget = EvidenceBudget(model_calls=2, tool_calls=1, maximum_usd=None)
    if dimension == "input_tokens":
        budget = budget.model_copy(update={"input_tokens": 50_000})
    elif dimension == "output_tokens":
        budget = budget.model_copy(update={"output_tokens": 1})
    else:
        ticks = iter([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        monkeypatch.setattr(
            "exp.optimize.evaluation.variance.evidence_tools.time",
            SimpleNamespace(monotonic=lambda: next(ticks, 5.0)),
        )
        budget = budget.model_copy(update={"wall_clock_seconds": 0.1})
    profile = EvidenceToolProfile(mode="read_only_tools", budget=budget)
    ceiling = OperationSpendCeiling(tmp_path / "spend.sqlite", tariffs={})
    ceiling.authorize_policy(ceiling_usd=1.0, unpriced_attempts=2)
    with operation_spend_scope():
        install_spend_ceiling(ceiling)
        _, transcript = _run(project, settings, profile)
    assert transcript.stop == "budget_exhausted" and dimension in transcript.exhausted
    assert len(transport.requests) <= 1


def test_request_capped_draw_cannot_exceed_operation_attempt_allowance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Physical dispatch still refuses the next call once the unpriced grant is spent."""
    project, settings, transport = _loop_fixture(
        tmp_path,
        monkeypatch,
        [_reply([("read_evidence", {"locator": "group"})])],
        priced=False,
    )
    profile = EvidenceToolProfile(
        mode="read_only_tools", budget=EvidenceBudget(model_calls=2, tool_calls=1, maximum_usd=None)
    )
    ceiling = OperationSpendCeiling(tmp_path / "spend.sqlite", tariffs={})
    ceiling.authorize_policy(ceiling_usd=1.0, unpriced_attempts=1)
    with operation_spend_scope():
        install_spend_ceiling(ceiling)
        with pytest.raises(UnpricedDispatchRefused):
            _run(project, settings, profile)
    assert len(transport.requests) == 1 and ceiling.summary().unpriced_attempts == 1
    assert not project.artifacts.exists("draw-0-call-1-request-capped-intent")


def _forbid_connect(_connection: socket.socket, _address: str | tuple[str, int]) -> None:
    """Fail the offline role fixture if any real socket is opened."""
    raise AssertionError("offline role tests must not open provider sockets")


def test_interrupted_request_capped_draw_replays_saved_calls_without_spend_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash before transcript publication reuses its saved response without another call."""
    project, settings, transport = _loop_fixture(
        tmp_path, monkeypatch, [_reply([("submit", {"value": 2})])], priced=False
    )
    profile = EvidenceToolProfile(budget=EvidenceBudget(maximum_usd=None))
    ceiling = OperationSpendCeiling(tmp_path / "spend.sqlite", tariffs={})
    ceiling.authorize_policy(ceiling_usd=1.0, unpriced_attempts=1)
    with operation_spend_scope():
        install_spend_ceiling(ceiling)
        with (
            patch(
                "exp.optimize.evaluation.variance.evidence_tools._Loop._finish",
                side_effect=RuntimeError("interrupted before transcript publication"),
            ),
            pytest.raises(RuntimeError, match="interrupted"),
        ):
            _run(project, settings, profile)
    assert not project.artifacts.exists("draw-0-transcript")
    assert project.artifacts.exists("draw-0-request-capped-response")
    _, transcript = _run(project, settings, profile)
    assert transcript.stop == "answered" and transcript.answer == {"value": 2}
    assert transcript.usage.unpriced_calls == 1 and len(transport.requests) == 1
    assert ceiling.summary().unpriced_attempts == 1
