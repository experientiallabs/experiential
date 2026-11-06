"""Offline regressions for request-capped rule agents and exact persisted replay."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import JsonValue

from exp.common.core.artifacts import ArtifactInput, JsonObject
from exp.common.models import AssistantAction, ModelMessage, ModelRequest, ModelResponse, ToolCall
from exp.common.project import ProjectStore
from exp.common.tasks import ToolSchema
from exp.optimize.evaluation.factory.authoring import AuthorSettings, author_request_bytes
from exp.optimize.evaluation.rules.agent_loop import (
    AgentCall,
    AgentProfile,
    AgentToolRecord,
    AgentTranscript,
    Phase,
    ToolStatus,
    _Loop,
    run_agent,
)
from exp.optimize.evaluation.rules.evidence_pages import NativeEvidenceDelivery
from exp.optimize.evaluation.rules.locator_inventory import LocatorInventory, encode_inventory
from exp.optimize.evaluation.variance.evidence_tools import EvidenceBudget, MeasuredUsage
from exp.optimize.evaluation.variance.evidence_tools_test import _ANSWER, _loop_fixture, _reply
from exp.runtime.models.spend_ceiling import (
    OperationSpendCeiling,
    UnpricedDispatchRefused,
    install_spend_ceiling,
    operation_spend_scope,
)


def _run(
    project: ProjectStore, settings: AuthorSettings, profile: AgentProfile
) -> tuple[ArtifactInput, AgentTranscript]:
    """Run a rules role over a tiny opening without any evidence reads."""
    return run_agent(
        project,
        settings,
        profile,
        identity="rules-draw",
        role="research",
        variant="contract",
        instruction="Read and answer.",
        opening={"task": "rules"},
        tools=(),
        runner=lambda call: ("refused", "no evidence", ()),
        answer=_ANSWER,
        check=lambda draft, served: None,
        sources=(),
        code_revision="test",
    )


@pytest.mark.parametrize("request_capped", [False, True])
def test_rules_agent_unknown_prices_need_explicit_request_capped_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request_capped: bool
) -> None:
    """Only the explicit mode completes all four phases under the operation allowance."""
    project, settings, transport = _loop_fixture(
        tmp_path,
        monkeypatch,
        [
            _reply([("record_plan", {"hypotheses": [], "planned_reads": []})]),
            _reply([("submit_draft", {"value": 2})]),
            _reply([("record_self_check", {"issues": [], "revise": False})]),
            _reply([("submit", {"value": 2})]),
        ],
        priced=False,
    )
    profile = AgentProfile(
        budget=EvidenceBudget(
            model_calls=4, tool_calls=1, maximum_usd=None if request_capped else 5.0
        )
    )
    ceiling = OperationSpendCeiling(tmp_path / "spend.sqlite", tariffs={})
    ceiling.authorize_policy(ceiling_usd=1.0, unpriced_attempts=4)
    if request_capped:
        with pytest.raises(UnpricedDispatchRefused, match="active finite"):
            _run(project, settings, profile)
        assert transport.requests == []
    with operation_spend_scope():
        install_spend_ceiling(ceiling)
        reference, transcript = _run(project, settings, profile)
    if request_capped:
        assert transcript.stop == "answered" and transcript.answer == {"value": 2}
        assert [call.phase for call in transcript.calls] == [
            "plan",
            "investigate",
            "self_check",
            "final",
        ]
        assert transcript.usage.unpriced_calls == 4
        assert transcript.profile.budget.maximum_usd is None
        assert all("request-capped" in call.response.artifact_id for call in transcript.calls)
        assert len(transport.requests) == 4 and ceiling.summary().unpriced_attempts == 4
    else:
        assert transcript.stop == "budget_exhausted"
        assert transcript.exhausted == ("maximum_usd_unverifiable",)
        assert len(transport.requests) == 1 and ceiling.summary().unpriced_attempts == 1
    assert transcript.usage.priced_usd == 0.0
    assert _run(project, settings, profile) == (reference, transcript)
    assert len(transport.requests) == (4 if request_capped else 1)


def _model_reply(loop: _Loop, calls: list[tuple[str, JsonObject]]) -> ModelResponse:
    """Build a synthetic typed response without reading saved model replies."""

    return ModelResponse.completed(
        output=AssistantAction(
            tool_calls=tuple(
                ToolCall(call_id=f"call-{i}", name=name, arguments=args)
                for i, (name, args) in enumerate(calls)
            )
        ),
        configured_model=loop.settings.model,
        served_model_id=None,
        usage=None,
        latency_seconds=0.0,
    )


def _native_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, limit: int = 128000) -> _Loop:
    """Construct the real loop without any admitted operation or provider dispatch."""

    project, settings, _ = _loop_fixture(tmp_path, monkeypatch, [])
    return _Loop(
        project,
        settings.model_copy(update={"maximum_input_bytes": limit}),
        AgentProfile(),
        "offline",
        (),
        "test",
        "research",
        "contract",
    )


def _script_loop(
    loop: _Loop,
    monkeypatch: pytest.MonkeyPatch,
    replies: Callable[[ModelRequest, int], ModelResponse],
) -> list[ModelRequest]:
    """Capture actual assembled requests, returning only synthetic offline responses."""

    captured: list[ModelRequest] = []

    def respond(request: ModelRequest, *, phase: Phase, forced: str | None) -> ModelResponse:
        """Record a complete native assembly without author/cache/spend calls."""
        captured.append(request)
        output = replies(request, len(captured))
        loop.calls.append(
            AgentCall(
                index=len(loop.calls),
                response=ArtifactInput(artifact_id=f"synthetic-{len(captured)}", sha256="a" * 64),
                phase=phase,
                forced=forced,
                usage=MeasuredUsage(),
            )
        )
        return output

    monkeypatch.setattr(loop, "_call", respond)
    return captured


def test_native_phase_assemblies_preserve_history_and_all_opening_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real loop retains opening and mandatory messages while enforcing effective bytes."""

    loop = _native_loop(tmp_path, monkeypatch)
    refs = [f"trace:t{i // 32}:span:s{i}" for i in range(445)]
    original: JsonObject = {
        "inventory": sorted(refs),
        "source_traces": [{"quote": '雪"\\\n' * 5000}],
        "family": {"questions": ["all preserved"]},
    }
    opening: JsonObject = {**original, "inventory": encode_inventory(refs)}
    outputs = [
        _model_reply(loop, [("record_plan", {"hypotheses": [], "planned_reads": []})]),
        _model_reply(loop, [("submit_draft", {"value": 2})]),
        _model_reply(loop, [("record_self_check", {"issues": [], "revise": False})]),
        _model_reply(loop, [("submit", {"value": 2})]),
    ]
    captured = _script_loop(loop, monkeypatch, lambda request, index: outputs[index - 1])
    transcript = loop.run(
        "Evidence instruction",
        opening,
        (),
        lambda call: ("refused", "none", ()),
        _ANSWER,
        lambda draft, refs: None,
    )
    assert transcript.stop == "answered" and len(captured) == 4
    for previous, current in zip(captured, captured[1:], strict=False):
        assert current.messages[: len(previous.messages)] == previous.messages
    assert all(author_request_bytes(loop.settings, r) <= 128000 for r in captured)
    content = captured[0].messages[1].content
    assert isinstance(content, str)
    delivered = json.loads(content)
    assert LocatorInventory.model_validate(delivered["inventory"]).decode() == tuple(sorted(refs))
    assert (
        delivered["source_traces"] == original["source_traces"]
        and delivered["family"] == original["family"]
    )


def test_native_cumulative_assistant_growth_stops_before_second_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A large actual plan is retained; bytes exhaustion never skips it or dispatches oversized."""
    loop = _native_loop(tmp_path, monkeypatch)
    captured = _script_loop(
        loop,
        monkeypatch,
        lambda request, index: _model_reply(
            loop, [("record_plan", {"hypotheses": ["x" * 32000], "planned_reads": []})]
        ),
    )
    transcript = loop.run(
        "Evidence",
        {"source": "x" * 112000},
        (),
        lambda call: ("refused", "none", ()),
        _ANSWER,
        lambda draft, refs: None,
    )
    assert len(captured) == 1 and transcript.stop == "budget_exhausted"
    assert transcript.exhausted == ("serialized_request_bytes",) and transcript.outputs[
        0
    ].arguments["hypotheses"] == ["x" * 32000]


def test_native_delivery_accounts_for_escaping_schemas_and_refused_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raw32000-byte read can exceed retained capacity; no result or citation is granted."""

    loop = _native_loop(tmp_path, monkeypatch)
    fake = ArtifactInput(artifact_id="synthetic", sha256="a" * 64)
    loop.calls.append(AgentCall(index=0, response=fake, phase="investigate", usage=MeasuredUsage()))
    monkeypatch.setattr(
        loop,
        "_save_tool",
        lambda position, call, status, result, served, delivery=None: AgentToolRecord(
            call=0,
            name=call.name,
            arguments=call.arguments,
            status=status,
            served=served,
            result=fake,
        ),
    )
    read = ToolCall(call_id="c", name="read", arguments={})
    messages = loop._reads(
        [read],
        lambda call: ("ok", '"' * 16000, ("contract:x",)),
        history=[ModelMessage(role="user", content="x" * 113000)],
        schemas=((_ANSWER,),),
    )
    assert (
        loop.tools[0].status == "refused"
        and loop.tools[0].served == ()
        and loop._served() == frozenset()
    )
    assert "cumulative" in str(messages[0].content)


def test_native_loop_delivers_complete_values_without_automatic_paging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An admitted complete result stays in one message with its exact citation authority."""
    loop = _native_loop(tmp_path, monkeypatch)
    fake = ArtifactInput(artifact_id="synthetic", sha256="a" * 64)
    value = {"quote": '"\\雪' * 2200}
    seen: list[JsonValue] = []

    def save(
        position: int,
        call: ToolCall,
        status: ToolStatus,
        result: JsonValue,
        served: tuple[str, ...],
        delivery: NativeEvidenceDelivery | None = None,
    ) -> AgentToolRecord:
        """Record delivery metadata in memory without any artifact publication."""
        return AgentToolRecord(
            call=len(loop.calls) - 1,
            name=call.name,
            arguments=call.arguments,
            status=status,
            served=served,
            result=fake,
            delivery=delivery,
        )

    def respond(request: ModelRequest, index: int) -> ModelResponse:
        """Request exact sequential cursors, then finish the normal mandatory phases."""
        names = {schema.name for schema in request.tools}
        if index == 1:
            return _model_reply(loop, [("record_plan", {"hypotheses": [], "planned_reads": []})])
        if index == 2:
            return _model_reply(loop, [("read_region", {})])
        if request.messages[-1].role == "tool":
            content = request.messages[-1].content
            assert isinstance(content, str)
            envelope = json.loads(content)
            if envelope.get("status") == "ok" and "result" in envelope:
                seen.append(envelope["result"])
                assert loop._served() == frozenset({"trace:t:span:one"})
        if "submit_draft" in names:
            return _model_reply(loop, [("submit_draft", {"value": 2})])
        if "record_self_check" in names:
            return _model_reply(loop, [("record_self_check", {"issues": [], "revise": False})])
        return _model_reply(loop, [("submit", {"value": 2})])

    monkeypatch.setattr(loop, "_save_tool", save)
    requests = _script_loop(loop, monkeypatch, respond)
    transcript = loop.run(
        "Evidence",
        {},
        (ToolSchema(name="read_region", description="Read", input_schema={"type": "object"}),),
        lambda call: ("ok", value, ("trace:t:span:one",)),
        _ANSWER,
        lambda draft, refs: None,
    )
    assert transcript.stop == "answered" and seen == [value]
    assert len(transcript.tools) == 1
    assert transcript.tools[0].delivery is not None and transcript.tools[0].delivery.complete
    assert transcript.tools[-1].served == ("trace:t:span:one",)
    assert all(tool.served == () for tool in transcript.tools[:-1])
    assert all(author_request_bytes(loop.settings, request) <= 128000 for request in requests)
    assert all(
        requests[i].messages[: len(requests[i - 1].messages)] == requests[i - 1].messages
        for i in range(1, len(requests))
    )


def test_native_final_refuses_unfinished_evidence_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An abandoned partial read is never converted into a final answer or citable result."""
    loop = _native_loop(tmp_path, monkeypatch)
    status, _, refs = loop.pages.start(
        ToolCall(call_id="c", name="read", arguments={}),
        "x" * 10000,
        ("contract:x",),
        lambda value, refs: True,
    )
    assert status == "ok" and refs == ()
    final = loop._answered(_model_reply(loop, [("submit", {"value": 2})]), "submit")
    assert final.stop == "protocol_failure" and final.answer is None
    assert final.failure and "incomplete" in final.failure


def test_native_optional_reads_leave_capacity_for_mandatory_phase_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Complete evidence reads cannot consume the space needed to finish the draw."""
    loop = _native_loop(tmp_path, monkeypatch)
    loop.profile = AgentProfile(
        budget=EvidenceBudget(model_calls=7, tool_calls=16, maximum_usd=None)
    )
    fake = ArtifactInput(artifact_id="synthetic-reserved", sha256="a" * 64)
    answer = ToolSchema(
        name="submit",
        description="Submit the answer.",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "integer"}, "rationale": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    )

    def save(
        position: int,
        call: ToolCall,
        status: ToolStatus,
        result: JsonValue,
        served: tuple[str, ...],
        delivery: NativeEvidenceDelivery | None = None,
    ) -> AgentToolRecord:
        """Keep exact delivery metadata without publishing synthetic provider artifacts."""
        return AgentToolRecord(
            call=len(loop.calls) - 1,
            name=call.name,
            arguments=call.arguments,
            status=status,
            served=served,
            result=fake,
            delivery=delivery,
        )

    def respond(request: ModelRequest, index: int) -> ModelResponse:
        """Return bounded phase outputs after three complete evidence requests."""
        if index == 1:
            return _model_reply(loop, [("record_plan", {"hypotheses": [], "planned_reads": []})])
        if index <= 4:
            return _model_reply(loop, [("read_region", {})])
        if index == 5:
            return _model_reply(loop, [("submit_draft", {"value": 2, "rationale": "d" * 22000})])
        if index == 6:
            return _model_reply(
                loop, [("record_self_check", {"issues": ["s" * 14000], "revise": False})]
            )
        return _model_reply(loop, [("submit", {"value": 2})])

    monkeypatch.setattr(loop, "_save_tool", save)
    requests = _script_loop(loop, monkeypatch, respond)
    transcript = loop.run(
        "Read evidence and finish.",
        {"context": "c" * 5000},
        (ToolSchema(name="read_region", description="Read", input_schema={"type": "object"}),),
        lambda call: ("ok", {"text": "e" * 31000}, ("trace:source:span:evidence",)),
        answer,
        lambda draft, refs: None,
    )
    assert transcript.stop == "answered", transcript.failure
    assert transcript.answer == {"value": 2} and len(requests) == 7
    assert [call.phase for call in transcript.calls][-3:] == ["investigate", "self_check", "final"]
    assert any(tool.status == "refused" for tool in transcript.tools)
    assert all(
        tool.delivery is not None and tool.delivery.complete
        for tool in transcript.tools
        if tool.status == "ok"
    )
    assert all(author_request_bytes(loop.settings, request) <= 128000 for request in requests)
    assert all(
        current.messages[: len(previous.messages)] == previous.messages
        for previous, current in zip(requests, requests[1:], strict=False)
    )


def test_forced_phase_offers_only_its_required_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A forced draft cannot advertise another evidence read after that budget is spent."""
    loop = _native_loop(tmp_path, monkeypatch)
    read = ToolSchema(name="read_trace", description="Read", input_schema={"type": "object"})
    tools = (read, _ANSWER)
    messages = [ModelMessage(role="user", content="Finish the draft.")]
    forced = loop._request(messages, tools, _ANSWER.name)
    assert tuple(tool.name for tool in forced.tools) == (_ANSWER.name,)
    free = loop._request(messages, tools, None)
    assert free.tools == tools


@pytest.mark.parametrize("repair", [True, False])
def test_invalid_draft_returns_to_investigation_before_self_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repair: bool
) -> None:
    """Structural feedback is actionable while the bounded evidence phase is still open."""
    loop = _native_loop(tmp_path, monkeypatch)
    loop.profile = AgentProfile(budget=EvidenceBudget(model_calls=8, tool_calls=1))
    drafts = 0

    def respond(request: ModelRequest, index: int) -> ModelResponse:
        nonlocal drafts
        names = {tool.name for tool in request.tools}
        if "record_plan" in names:
            return _model_reply(loop, [("record_plan", {"hypotheses": [], "planned_reads": []})])
        if "submit_draft" in names:
            drafts += 1
            return _model_reply(
                loop, [("submit_draft", {"value": -1 if drafts == 1 or not repair else 2})]
            )
        if "record_self_check" in names:
            return _model_reply(loop, [("record_self_check", {"issues": [], "revise": False})])
        return _model_reply(loop, [("submit", {"value": 2})])

    _script_loop(loop, monkeypatch, respond)
    transcript = loop.run(
        "Evidence",
        {},
        (),
        lambda call: ("refused", "none", ()),
        _ANSWER,
        lambda answer, refs: "value must be positive" if answer.get("value") == -1 else None,
    )
    assert transcript.stop == "answered"
    recorded = [output for output in transcript.outputs if output.phase == "draft"]
    assert len(transcript.calls) <= loop.profile.budget.model_calls
    if not repair:
        assert "self_check" in transcript.skipped
        assert all(output.host_validation == "value must be positive" for output in recorded)
        return
    assert [output.host_validation for output in recorded] == ["value must be positive", "valid"]
    assert [output.phase for output in transcript.outputs] == [
        "plan",
        "draft",
        "draft",
        "self_check",
    ]
