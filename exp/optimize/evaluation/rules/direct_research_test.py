"""Offline native-owner proofs for one full-input research response per trajectory."""

from __future__ import annotations

import json
from contextvars import ContextVar
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.optimize.evaluation.rules.direct_research import (
    direct_research_in_parallel,
    direct_research_request,
)
from exp.optimize.evaluation.rules.research import RESEARCH_TOOL
from exp.optimize.evaluation.rules.research_test import (
    AgentScript,
    agent_setup,
    research_answer,
    trace_of,
    wire,
)
from exp.optimize.evaluation.variance.evidence_tools_test import _reply
from exp.runtime.models.providers.transport import JsonHttpResponse
from exp.simulation.rules.contracts import parse_tool_contracts, trajectory_view
from exp.simulation.rules.contracts_test import refund_trace

OWNER: ContextVar[str] = ContextVar("direct-research-test-owner", default="missing")


class DirectScript(AgentScript):
    """Answer the actual native author request once, recording inherited spend context."""

    def __init__(self) -> None:
        super().__init__(lambda role, opening: {}, delay=0.02)
        self.owners: list[str] = []

    def _reply(self, payload: JsonObject) -> JsonHttpResponse:
        self.owners.append(OWNER.get())
        messages = payload["messages"]
        assert isinstance(messages, list) and len(messages) == 2
        opening = json.loads(str(messages[1]["content"]))
        trace_id = trace_of(opening)
        answer = research_answer(trace_id)
        answer["facts"] = [
            {
                "kind": "state",
                "basis": "observed",
                "statement": "The recorded payment is forty.",
                "evidence": [f"trace:{trace_id}:span:r1"],
            }
        ]
        return _reply([(RESEARCH_TOOL, answer)])


def test_full_trace_request_preserves_every_step_contract_and_allowance() -> None:
    trace = refund_trace("alpha")
    contracts = parse_tool_contracts((trace,))
    request, supplied = direct_research_request(
        trace, contracts, variant="contract_first", output_tokens=4096
    )
    payload = json.loads(str(request.messages[1].content))
    expected, locators = trajectory_view(trace)
    assert payload["trajectory"] == expected
    assert supplied == locators == frozenset(payload["citable_locators"])
    assert payload["tool_contracts"] == [
        contracts[name].model_dump(mode="json", exclude={"examples"}) for name in sorted(contracts)
    ]
    assert request.maximum_output_tokens == 4096
    assert len(request.tools) == 1 and request.tools[0].name == RESEARCH_TOOL


def test_parallel_native_owner_one_call_each_precise_refs_history_and_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, author = agent_setup(tmp_path, monkeypatch)
    script = DirectScript()
    wire(monkeypatch, script)
    traces = (refund_trace("alpha"), refund_trace("beta"))
    token = OWNER.set("original-operation")
    try:
        result = direct_research_in_parallel(
            project,
            author,
            traces,
            parse_tool_contracts(traces),
            identity="direct-test",
            code_revision="new-direct-producer",
            concurrency=2,
            variants=("contract_first",),
        )
    finally:
        OWNER.reset(token)
    assert len(script.requests) == 2 and script.most_in_flight == 2
    assert script.owners == ["original-operation"] * 2
    for trace, history in zip(traces, result, strict=True):
        draw = history.draws[0]
        assert draw.answer is not None and draw.failure is None
        assert draw.response is not None and draw.transcript is None
        assert draw.answer.facts and draw.answer.hypotheses and draw.answer.questions_asked
        assert draw.evidence_read == tuple(sorted(trajectory_view(trace)[1]))
        assert history.history is not None
        stored = project.artifacts.read(history.history.artifact_id)
        assert draw.response in stored.manifest.inputs
    again = direct_research_in_parallel(
        project,
        author,
        traces,
        parse_tool_contracts(traces),
        identity="direct-test",
        code_revision="new-direct-producer",
        concurrency=2,
        variants=("contract_first",),
    )
    assert again == result and len(script.requests) == 2


def test_actual_native_byte_guard_refuses_full_request_without_call_or_citations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, author = agent_setup(tmp_path, monkeypatch)
    script = DirectScript()
    wire(monkeypatch, script)
    trace = refund_trace("alpha")
    result = direct_research_in_parallel(
        project,
        author.model_copy(update={"maximum_input_bytes": 1}),
        (trace,),
        parse_tool_contracts((trace,)),
        identity="oversize-direct",
        code_revision="new-direct-producer",
        concurrency=1,
        variants=("contract_first",),
    )
    draw = result[0].draws[0]
    assert draw.failure and draw.answer is None and not draw.evidence_read
    assert not script.requests
