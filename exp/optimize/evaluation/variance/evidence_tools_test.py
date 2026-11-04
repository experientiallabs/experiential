"""Tests for the read-only evidence tools, usage measurement and the bounded analysis loop."""

from __future__ import annotations

import copy
import json
from pathlib import Path

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
from exp.optimize.evaluation.variance.evidence import EstimationPool, EvidencePacket
from exp.optimize.evaluation.variance.evidence_tools import (
    TOOL_RESULT_TYPE,
    TRANSCRIPT_TYPE,
    AnalysisTranscript,
    EvidenceBudget,
    EvidenceToolProfile,
    MeasuredUsage,
    execute_evidence_tool,
    response_usage,
    run_analysis,
    total_usage,
)
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog

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
    """A forced response makes one call; a tool profile can read and still answer."""
    assert EvidenceToolProfile().mode == "forced_response"
    with pytest.raises(ValueError, match="one model call"):
        EvidenceToolProfile(budget=EvidenceBudget(model_calls=2))
    with pytest.raises(ValueError, match="two model calls"):
        EvidenceToolProfile(mode="read_only_tools")


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


def _run(
    project: ProjectStore, settings: AuthorSettings, profile: EvidenceToolProfile
) -> tuple[ArtifactInput, AnalysisTranscript]:
    """Run the loop over the fixture packet."""
    return run_analysis(
        project,
        settings,
        profile,
        instruction="Read and answer.",
        packet=_PACKET,
        content={"task": "estimate"},
        answer=_ANSWER,
        identity="draw-0",
        sources=(),
        code_revision="test",
    )
