"""Precise native citation delivery and strict topic/scope refusals."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, JsonObject
from exp.common.models import ToolCall
from exp.common.project import artifact_input
from exp.optimize.evaluation.rules.governance_test import SMALL, RoleScript
from exp.optimize.evaluation.rules.memory_edits import RuleDraft
from exp.optimize.evaluation.rules.memory_stage import lead_research_units
from exp.optimize.evaluation.rules.research_test import agent_setup, wire
from exp.optimize.evaluation.rules.system_one import Objective, ObjectiveFindings, ReturnedPassage
from exp.optimize.evaluation.rules.system_one_test import ScriptedJev, system_one
from exp.optimize.evaluation.rules.system_two import _Reader, research_topic, validate_topic_answer
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


@pytest.mark.parametrize(
    "delivery", ["refused", "complete", "wrong_objective", "partial", "orphan"]
)
def test_native_research_separates_attempts_and_spend_from_delivered_leads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, delivery: str
) -> None:
    """Actual research/loop delivery controls next-round seeds, including lead-only findings."""
    project, settings = agent_setup(tmp_path, monkeypatch)
    one = system_one(project, ScriptedJev({}))
    objective = Objective(topic="refund", tool="refund", question="What gates refund?")
    lead = Objective(
        topic="refund",
        tool="refund",
        question="What confirms refunds? " + "detail " * 60,
        origin="discovery",
    )
    findings = ObjectiveFindings(
        objective=objective
        if delivery != "wrong_objective"
        else objective.model_copy(update={"topic": "other"}),
        objective_id="lead-only",
        stop="enough",
        leads=(lead,) * (20 if delivery == "partial" else 1),
        bought=2,
        replayed=3,
        usd=0.125,
    )
    reference = artifact_input(
        project.artifacts.write_json(
            artifact_id="raw-findings",
            artifact_type="jev-objective-findings",
            envelope=ArtifactEnvelope(
                schema_version=1, created_at=datetime.now(UTC), code_revision="test"
            ),
            files={"findings.json": findings},
        )
    )
    monkeypatch.setattr(one, "answer", lambda requested: (findings, reference))

    def answer(role: str, opening: JsonObject) -> JsonObject:
        """Finish without inventing any rule citations when evidence was refused."""
        return {"topic": "refund", "rules": [], "limitations": ["Only recorded facts"]}

    transport = RoleScript(answer)
    wire(monkeypatch, transport)
    profile = SMALL.model_copy(
        update={"maximum_result_bytes": 256 if delivery == "refused" else 32000}
    )
    record = research_topic(
        project,
        settings,
        profile,
        one,
        parse_tool_contracts([refund_trace("t1")]),
        "refund",
        "refund",
        "failures_first",
        identity="delivery-draw",
        round_=0,
        code_revision="test",
    )
    # memory_stage's real follow-leads consumer iterates only record.leads, even on failed draws.
    if delivery in ("complete", "orphan"):
        assert record.findings == (reference,) and record.leads == (lead,)
        assert lead_research_units((record,), (("refund", "refund"),)) == [
            ("refund", "refund", (lead,), ())
        ]
    else:
        assert record.findings == () and record.leads == () and record.objectives == ()
        assert lead_research_units((record,), (("refund", "refund"),)) == []
    assert record.attempted_findings == (reference,)
    assert record.attempted_leads == findings.leads and record.attempted_objectives == (
        findings.objective,
    )
    assert (record.decisions_bought, record.decisions_replayed, record.decision_usd) == (
        2,
        3,
        0.125,
    )
    assert project.artifacts.exists(reference.artifact_id)
    if delivery == "orphan":
        native_exists = project.artifacts.exists
        monkeypatch.setattr(
            project.artifacts,
            "exists",
            lambda identity: (
                False if identity == "delivery-draw-topic" else native_exists(identity)
            ),
        )
        calls = len(transport.requests)
        with pytest.raises(ValueError, match="attempt summary"):
            research_topic(
                project,
                settings,
                profile,
                one,
                parse_tool_contracts([refund_trace("t1")]),
                "refund",
                "refund",
                "failures_first",
                identity="delivery-draw",
                round_=0,
                code_revision="test",
            )
        assert len(transport.requests) == calls and native_exists("delivery-draw-transcript")
        assert native_exists(reference.artifact_id)
    else:
        assert (
            research_topic(
                project,
                settings,
                profile,
                one,
                parse_tool_contracts([refund_trace("t1")]),
                "refund",
                "refund",
                "failures_first",
                identity="delivery-draw",
                round_=0,
                code_revision="test",
            )
            == record
        )
