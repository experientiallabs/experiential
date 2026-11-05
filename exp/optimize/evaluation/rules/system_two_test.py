"""Precise native citation delivery and strict topic/scope refusals."""

from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput
from exp.common.models import ToolCall
from exp.optimize.evaluation.rules.memory_edits import RuleDraft
from exp.optimize.evaluation.rules.research_test import agent_setup
from exp.optimize.evaluation.rules.system_one import Objective, ObjectiveFindings, ReturnedPassage
from exp.optimize.evaluation.rules.system_one_test import ScriptedJev, system_one
from exp.optimize.evaluation.rules.system_two import _Reader, validate_topic_answer
from exp.simulation.rules.contracts import parse_tool_contracts
from exp.simulation.rules.contracts_test import refund_trace


def test_native_reader_grants_exact_delivered_metadata_not_quote_strings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only native region IDs and successful known-contract results become citable."""
    project, _ = agent_setup(tmp_path, monkeypatch)
    one = system_one(project, ScriptedJev({}))
    contracts = parse_tool_contracts([refund_trace("t1")])
    reader = _Reader(one, one.corpus, contracts, "refund", "refund", 32000)
    status, value, refs = reader.run(
        ToolCall(
            call_id="r", name="read_region", arguments={"locator": "trace:t1:span:c2", "radius": 0}
        )
    )
    assert status == "ok" and set(refs) == {"trace:t1:span:c2", "trace:t1:span:r2"}
    assert isinstance(value, dict) and value["citations"] == list(refs)
    status, value, refs = reader.run(
        ToolCall(call_id="c", name="read_tool_contract", arguments={"tool": "refund"})
    )
    assert (
        status == "ok"
        and refs == ("contract:refund",)
        and isinstance(value, dict)
        and value["citations"] == ["contract:refund"]
    )
    for locator in ["trace:foreign:span:c2", "contract:not-a-step"]:
        assert (
            reader.run(ToolCall(call_id="f", name="read_region", arguments={"locator": locator}))[2]
            == ()
        )
    assert (
        reader.run(ToolCall(call_id="f", name="read_tool_contract", arguments={"tool": "foreign"}))[
            0
        ]
        == "refused"
    )
    assert reader._ok({"quote": "contract:invented", "padding": "x" * 32000}, ("contract:refund",))[
        0::2
    ] == ("refused", ())


def test_topic_and_unknown_reference_guards_remain_strict() -> None:
    """Adding trusted delivery metadata never expands topic or arbitrary citation authority."""

    rule = RuleDraft(
        kind="precondition",
        tool="refund",
        truism="Requires confirmation",
        reasoning="The record shows a refusal before confirmation.",
        evidence=("contract:refund",),
        status="supported",
        status_rationale="Observed",
        applicability="refund",
        open_questions=(),
    )
    answer = {
        "topic": "refund",
        "rules": [rule.model_dump(mode="json")],
        "limitations": ["Only recorded evidence"],
    }
    assert (
        validate_topic_answer(
            answer,
            topic="refund",
            tool="refund",
            served=frozenset({"contract:refund"}),
            tools=frozenset({"refund", "other"}),
        ).topic
        == "refund"
    )
    for tool, evidence, expected in [
        ("other", ["contract:refund"], "another topic"),
        ("refund", ["contract:invented"], "not served"),
        ("absent", ["contract:refund"], "no contract"),
    ]:
        changed = {
            **answer,
            "rules": [{**rule.model_dump(mode="json"), "tool": tool, "evidence": evidence}],
        }
        with pytest.raises(ValueError, match=expected):
            validate_topic_answer(
                changed,
                topic="refund",
                tool="refund",
                served=frozenset({"contract:refund"}),
                tools=frozenset({"refund", "other"}),
            )


def test_objective_results_require_exact_native_coordinates_and_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Matched quote text alone cannot authorize forged passage coordinates or objectives."""
    project, _ = agent_setup(tmp_path, monkeypatch)
    jev = ScriptedJev({})
    one = system_one(project, jev)
    contracts = parse_tool_contracts([refund_trace("t1")])
    reader = _Reader(one, one.corpus, contracts, "refund", "refund", 32000)
    region = one.corpus.by_locator["trace:t1:span:c2"]
    objective = Objective(topic="refund", tool="refund", question="What gates refund?")
    findings = ObjectiveFindings(
        objective=objective,
        objective_id="offline-objective",
        stop="enough",
        passages=(ReturnedPassage(passage=region.passage(), shows=1.0, relevance=2.0),),
    )
    reference = ArtifactInput(artifact_id="synthetic-findings", sha256="a" * 64)
    monkeypatch.setattr(one, "answer", lambda objective: (findings, reference))
    call = ToolCall(call_id="s", name="ask_system_one", arguments={"question": objective.question})
    status, _, refs = reader.run(call)
    assert status == "ok" and set(refs) == {"trace:t1:span:c2", "trace:t1:span:r2"}
    for changes in [{"trace_id": "foreign"}, {"step": 999}, {"locator": "trace:foreign:span:c2"}]:
        forged = findings.model_copy(
            update={
                "passages": (
                    findings.passages[0].model_copy(
                        update={"passage": region.passage().model_copy(update=changes)}
                    ),
                )
            }
        )
        monkeypatch.setattr(one, "answer", lambda objective, result=forged: (result, reference))
        status, _, refs = reader.run(call)
        assert status == "refused" and refs == ()
    forged = findings.model_copy(
        update={"objective": objective.model_copy(update={"topic": "another"})}
    )
    monkeypatch.setattr(one, "answer", lambda objective: (forged, reference))
    assert reader.run(call)[0::2] == ("refused", ())
    assert jev.calls == []
