"""Offline tests for the authoring packet, with helpers the author and freeze tests share.

The helpers build a packet over the grounded ticket fixture: its real start checkpoint, a saved
rule book with the tier rule and a supported lock precondition on ``close``, an exposure plan
that reserves one lineage for final evaluation, and the tier decision with two variants.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    JsonObject,
    JsonValue,
    sha256_json,
)
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.factory.candidates_test import _reply
from exp.optimize.evaluation.rules.stage import RULE_BOOK_TYPE, rulebook_files
from exp.optimize.evaluation.variance.authoring_packet import (
    AuthoringPacket,
    SourceRef,
    build_authoring_packet,
    load_packet,
    save_packet,
    variant_task_sha256,
)
from exp.optimize.evaluation.variance.contracts import (
    FreshBriefing,
    InstructionAxis,
    InstructionValue,
    Scenario,
)
from exp.optimize.evaluation.variance.exposure import ExposurePlan
from exp.optimize.evaluation.variance.tests.grounded_world_test import (
    INITIAL,
    RULE,
    TOOLS,
    GroundedFixture,
    grounded_fixture,
    tier_decision,
    tier_scenario,
)
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.simulation.continuations import load_continuation_checkpoint
from exp.simulation.induction.contracts import InductionEpisode, InductionStep
from exp.simulation.rules.model import (
    Operand,
    Predicate,
    ResearchQuestion,
    Rule,
    RuleBook,
    ToolContract,
)

LOCK_RULE = Rule(
    rule_id="tickets.lock",
    tool="close",
    kind="precondition",
    statement="A ticket another agent holds cannot be closed.",
    predicate=Predicate(
        op="eq",
        left=Operand(scope="before", path="/tickets/lock"),
        right=Operand(value="none"),
    ),
    status="supported",
    support=3,
    contradiction=0,
    evidence=("trace:trace-1:span:4",),
)
"""A supported authorization-like precondition on the state-writing ``close`` tool."""

QUESTION = ResearchQuestion(
    tool="escalate",
    question="Does escalating a held ticket succeed?",
    why_it_matters="Decides whether escalation is valid while another agent holds the ticket.",
)
"""An open question the rule book keeps."""


def plan() -> ExposurePlan:
    """An exposure plan with one world-learning and one final-evaluation lineage."""
    return ExposurePlan(
        bindings=ArtifactInput(artifact_id="bindings", sha256="0" * 64),
        seed=0,
        probe_fraction=0.5,
        roles={"tickets": "world_learning", "held": "final_evaluation"},
        traces={"trace-1": "tickets", "trace-final": "held"},
    )


def save_book(project: ProjectStore) -> tuple[ArtifactInput, RuleBook]:
    """Save the ticket rule book and return its reference and content."""
    book = RuleBook(
        contracts={tool.name: ToolContract(name=tool.name) for tool in TOOLS},
        rules=(RULE, LOCK_RULE),
        questions=(QUESTION,),
        sources=("trace-1",),
    )
    manifest = project.artifacts.write(
        artifact_id="ticket-authoring-rule-book",
        artifact_type=RULE_BOOK_TYPE,
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), code_revision="test"
        ),
        files=rulebook_files(book),
    )
    return artifact_input(manifest), book


def observed_episode() -> InductionEpisode:
    """A source-derived replay of closing a low tier ticket."""
    low = {"tickets": {**_ticket(), "severity": "low"}}
    return InductionEpisode(
        episode_id="trace-1",
        initial_state=low,
        steps=(
            InductionStep(
                tool="close",
                arguments={},
                reply={"status": "closed"},
                state_after={
                    "tickets": {
                        **_ticket(),
                        "severity": "low",
                        "status": "closed",
                        "outcome": "right",
                    }
                },
            ),
        ),
    )


def ticket_packet(
    fixture: GroundedFixture, *, scenarios: tuple[Scenario, ...] | None = None
) -> tuple[AuthoringPacket, ArtifactInput]:
    """Build and save the ticket family's packet over the grounded fixture's start checkpoint.

    Args:
        fixture: The grounded ticket fixture.
        scenarios: Scenarios to use instead of the urgent and routine tiers.

    Returns:
        The packet and its saved reference.
    """
    project = fixture.project
    reference, book = save_book(project)
    checkpoint = fixture.job.study.checkpoint
    start = load_continuation_checkpoint(project.artifacts, checkpoint)
    packet = build_authoring_packet(
        family_id="ticket-tier",
        task=fixture.definition.task,
        checkpoint=checkpoint,
        before_state=INITIAL,
        public_history=start.actor.messages,
        remaining_calls=5,
        sources=(
            SourceRef(
                source_id="trace-1",
                kind="trace",
                trace_id="trace-1",
                summary="An agent closed a low tier ticket.",
            ),
        ),
        exposure=plan(),
        rule_book=reference,
        book=book,
        decision=tier_decision(0),
        scenarios=scenarios or (tier_scenario("urgent", "high"), tier_scenario("routine", "low")),
        observed_episodes=(observed_episode(),),
    )
    return packet, save_packet(project, packet, code_revision="test")


def tool_reply(name: str, arguments: JsonObject) -> JsonHttpResponse:
    """One scripted provider reply making a single forced tool call."""
    return _reply(
        "judge-v1",
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        },
        "tool_calls",
    )


def wire(monkeypatch: pytest.MonkeyPatch, *replies: JsonHttpResponse) -> ScriptedJsonTransport:
    """Route author calls to a scripted transport and return it."""
    transport = ScriptedJsonTransport(replies)
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: transport
    )
    return transport


def _ticket(**update: JsonValue) -> JsonObject:
    """The fixture ticket domain with fields replaced."""
    ticket = INITIAL["tickets"]
    assert isinstance(ticket, dict)
    return {**ticket, **update}


def test_the_packet_binds_sources_variants_and_keeps_evidence_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rules keep their status, brief claims stay hypotheses, and the packet replays exactly."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    packet, reference = ticket_packet(fixture)

    assert [variant.variant_id for variant in packet.variants] == ["urgent", "routine"]
    assert {claim.claim_id: claim.status for claim in packet.claims} == {
        "rule.tickets.tier": "supported",
        "rule.tickets.lock": "supported",
        "brief.urgent": "hypothesis",
        "brief.routine": "hypothesis",
    }
    assert packet.brief.authority == "proposal"
    assert packet.brief.proposed_correct == {"urgent": "escalate", "routine": "close"}
    assert {"rule:tickets.lock", "question:0", "source:trace-1", "variant:urgent"} <= (
        packet.locators()
    )
    assert packet.unsupplied(["state:/tickets/lock", "source:invented"]) == ("source:invented",)
    assert load_packet(fixture.project, reference) == packet
    assert save_packet(fixture.project, packet, code_revision="test") == reference


def test_final_evaluation_sources_never_reach_an_author(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A trace reserved for final evaluation is refused before any packet exists."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    reference, book = save_book(fixture.project)
    with pytest.raises(ValueError, match="reserved for final evaluation"):
        build_authoring_packet(
            family_id="ticket-tier",
            task=fixture.definition.task,
            checkpoint=fixture.job.study.checkpoint,
            before_state=INITIAL,
            public_history=(),
            remaining_calls=5,
            sources=(
                SourceRef(
                    source_id="held", kind="trace", trace_id="trace-final", summary="Held out."
                ),
            ),
            exposure=plan(),
            rule_book=reference,
            book=book,
            decision=tier_decision(0),
            scenarios=(tier_scenario("urgent", "high"), tier_scenario("routine", "low")),
        )


def test_an_instruction_only_variant_is_a_distinct_task_with_the_same_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Identical state with a declared instruction change is compared as (state, instruction)."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    reference, book = save_book(fixture.project)
    task = fixture.definition.task
    told = "Explain how the ticket should be handled, but do not change it."
    decision = tier_decision(0).model_copy(
        update={
            "instruction_axis": InstructionAxis(
                values=(
                    InstructionValue(
                        value_id="original",
                        instruction=task.instruction,
                        kind="original",
                        obligations="Resolve the ticket.",
                    ),
                    InstructionValue(
                        value_id="explain",
                        instruction=told,
                        kind="goal_change",
                        obligations="Explain the handling without changing the ticket.",
                        authorized_by=("point",),
                        criteria=("handled",),
                    ),
                ),
                effect="An explanation-only request forbids handling the ticket.",
            )
        }
    )
    explain = Scenario(
        scenario_id="explain",
        instruction=told,
        initialization="fresh_briefing",
        briefing=FreshBriefing(text="A customer asks about their open support ticket."),
        correct_option="escalate",
        deciding_paths=("/tickets/severity",),
        witness=tier_scenario("urgent", "high").witness,
        rationale="The user asked for an explanation only.",
        criteria=("handled",),
    )
    packet = build_authoring_packet(
        family_id="ticket-tier",
        task=fixture.definition.task,
        checkpoint=fixture.job.study.checkpoint,
        before_state=INITIAL,
        public_history=(),
        remaining_calls=5,
        sources=(SourceRef(source_id="spec", kind="specification", summary="Ticket policy."),),
        exposure=plan(),
        rule_book=reference,
        book=book,
        decision=decision,
        scenarios=(tier_scenario("urgent", "high"), explain),
    )
    urgent, explained = packet.variant("urgent"), packet.variant("explain")
    assert urgent.state_sha256 == explained.state_sha256
    assert urgent.task_sha256 != explained.task_sha256
    assert (urgent.axis, explained.axis) == ("baseline", "instruction")
    assert explained.obligation_id == "explain" and explained.initialization == "fresh_briefing"
    assert explained.task_sha256 == variant_task_sha256(
        fixture.definition.task, explained.instruction, INITIAL
    )
    assert sha256_json(INITIAL) == urgent.state_sha256
    with pytest.raises(ValueError, match="does not declare as an instruction value"):
        build_authoring_packet(
            family_id="ticket-tier",
            task=fixture.definition.task,
            checkpoint=fixture.job.study.checkpoint,
            before_state=INITIAL,
            public_history=(),
            remaining_calls=5,
            sources=(SourceRef(source_id="spec", kind="specification", summary="Policy."),),
            exposure=plan(),
            rule_book=reference,
            book=book,
            decision=tier_decision(0),
            scenarios=(explain,),
        )
