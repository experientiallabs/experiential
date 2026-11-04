"""Offline tests for freezing an authored world and reviewing it against an authored rubric."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, sha256_json
from exp.common.models import ToolCall
from exp.common.project import artifact_input
from exp.optimize.evaluation.continuations.job import execute_continuation_job
from exp.optimize.evaluation.executable.grounded import continuation_runtime, simulated_image
from exp.optimize.evaluation.rules.challenges import ChallengeReport
from exp.optimize.evaluation.rules.stage import RULE_BOOK_TYPE, rulebook_files
from exp.optimize.evaluation.variance.authoring_packet import (
    AuthoringPacket,
    SourceRef,
    build_authoring_packet,
    save_packet,
)
from exp.optimize.evaluation.variance.authoring_packet_test import (
    LOCK_RULE,
    QUESTION,
    observed_episode,
    plan,
    ticket_packet,
    tool_reply,
    wire,
)
from exp.optimize.evaluation.variance.rubric_author import RUBRIC_TOOL, author_rubric
from exp.optimize.evaluation.variance.rubric_author_test import criterion, rubric_answer
from exp.optimize.evaluation.variance.tests.grounded_world_test import (
    INITIAL,
    RULE,
    TOOLS,
    GroundedFixture,
    grounded_fixture,
    tier_decision,
    tier_scenario,
)
from exp.optimize.evaluation.variance.world_author import (
    WORLD_TOOL,
    author_world,
    load_world_draft,
    world_draft_reference,
    world_runtime_definition,
)
from exp.optimize.evaluation.variance.world_author_test import CARELESS_CLOSE, CLOSE, world_answer
from exp.optimize.evaluation.variance.world_freeze import (
    ResumeTrajectory,
    freeze_world,
    review_compatibility,
)
from exp.runtime.agents.chat import ChatAgentRuntime
from exp.runtime.environments.world_contracts import ExecutableCheckpoint
from exp.runtime.environments.worlds import ExecutableWorldRuntime
from exp.simulation.continuations import load_continuation_checkpoint
from exp.simulation.induction.sandbox import ProgramSandbox
from exp.simulation.rules.model import Operand, Predicate, Rule, RuleBook, ToolContract


def _trajectory(fixture: GroundedFixture) -> ResumeTrajectory:
    """The fixture start as the one-boundary trajectory both tier variants fork from."""
    return ResumeTrajectory(
        branch_checkpoints=(fixture.job.study.checkpoint,),
        decision=tier_decision(0),
        scenarios=(tier_scenario("urgent", "high"), tier_scenario("routine", "low")),
    )


def _world(
    fixture: GroundedFixture,
    monkeypatch: pytest.MonkeyPatch,
    sandbox: ProgramSandbox,
    *,
    close: str = CLOSE,
    bindings: tuple[str, ...] = ("tickets.tier", "tickets.lock"),
) -> tuple[AuthoringPacket, ArtifactInput, ArtifactInput]:
    """Author a ready ticket world and return the packet, its reference and the world draft."""
    packet, reference = ticket_packet(fixture)
    wire(
        monkeypatch,
        tool_reply(WORLD_TOOL, world_answer(packet, close=close, bindings=bindings)),
    )
    draft = author_world(
        fixture.project,
        fixture.settings,
        reference,
        sandbox=sandbox,
        identity="freeze-world",
        code_revision="test",
    )
    assert draft.status == "ready", draft.failure
    return packet, reference, world_draft_reference(fixture.project, draft)


def test_a_coherent_world_freezes_and_every_variant_restores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gate probes resolve to refusals, and each variant start restores to its exact state."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    with ProgramSandbox() as sandbox:
        packet, reference, world = _world(fixture, monkeypatch, sandbox)
        freeze = freeze_world(
            fixture.project,
            reference,
            world,
            sandbox=sandbox,
            code_revision="test",
            base=fixture.definition,
            actor=ChatAgentRuntime(maximum_model_calls=6),
            trajectory=_trajectory(fixture),
        )
        replay = freeze_world(
            fixture.project,
            reference,
            world,
            sandbox=sandbox,
            code_revision="test",
            base=fixture.definition,
            actor=ChatAgentRuntime(maximum_model_calls=6),
            trajectory=_trajectory(fixture),
        )

    assert replay == freeze
    assert not freeze.problems and freeze.trainable
    assert [(row.variant_id, row.status, row.restored) for row in freeze.variants] == [
        ("urgent", "frozen", True),
        ("routine", "frozen", True),
    ]
    resolutions = {(probe.variant_id, probe.case): probe.resolution for probe in freeze.probes}
    assert resolutions == {
        ("urgent", "variant"): "consistent",
        ("urgent", "missing:/tickets/lock"): "refused",
        ("routine", "variant"): "consistent",
        ("routine", "missing:/tickets/lock"): "refused",
    }
    assert all(result.violated == 0 for result in freeze.properties.results)
    assert {variant.variant_id for variant in packet.variants} == {
        row.variant_id for row in freeze.variants
    }


def test_an_edited_resume_variant_needs_its_trajectory_once_history_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a trajectory, only the history-free start resumes in an edited state."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    study = execute_continuation_job(fixture.project, fixture.job, code_revision="test")
    packet, reference = _packet(fixture, checkpoint=study.branches[0].checkpoints[0], step=1)
    with ProgramSandbox() as sandbox:
        world_ref = _authored(fixture, monkeypatch, sandbox, packet, reference, "no-trajectory")
        freeze = freeze_world(
            fixture.project,
            reference,
            world_ref,
            sandbox=sandbox,
            code_revision="test",
            base=fixture.definition,
        )

    rows = {row.variant_id: row for row in freeze.variants}
    assert rows["urgent"].status == "frozen"
    assert rows["routine"].status == "rejected"
    assert any("pass its trajectory" in problem for problem in rows["routine"].problems)


def test_freeze_rejects_a_world_that_silently_allows_an_unauthorized_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closing a ticket whose lock state is missing is a silent sensitive write: rejected."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    with ProgramSandbox() as sandbox:
        _, reference, world = _world(fixture, monkeypatch, sandbox, close=CARELESS_CLOSE)
        freeze = freeze_world(
            fixture.project,
            reference,
            world,
            sandbox=sandbox,
            code_revision="test",
            base=fixture.definition,
        )

    assert not freeze.trainable
    assert {row.status for row in freeze.variants} == {"rejected"}
    assert all(row.checkpoint is None for row in freeze.variants)
    allowed = [probe for probe in freeze.probes if probe.resolution == "silently_allowed"]
    assert {probe.case for probe in allowed} == {"missing:/tickets/lock"}
    assert any("tickets.lock" in problem for problem in freeze.problems)


def test_without_a_runtime_the_world_stays_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restoration that was never proven cannot qualify a variant for training."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    with ProgramSandbox() as sandbox:
        _, reference, world = _world(fixture, monkeypatch, sandbox)
        freeze = freeze_world(
            fixture.project, reference, world, sandbox=sandbox, code_revision="test"
        )

    assert not freeze.trainable and not freeze.problems
    assert {row.status for row in freeze.variants} == {"diagnostic"}
    assert all(row.restored is None for row in freeze.variants)


def test_compatibility_review_records_disagreements_without_reconciling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A criterion needing an unrecorded receipt and an unbound rule are both kept."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    with ProgramSandbox() as sandbox:
        packet, reference, world_ref = _world(
            fixture, monkeypatch, sandbox, bindings=("tickets.tier",)
        )
    needy = criterion(
        "lock.respected",
        question="Did the agent leave a held ticket open?",
        supported_by=["rule:tickets.lock"],
        evidence_needed=[{"kind": "receipt", "detail": "lock receipt"}],
    )
    wire(
        monkeypatch,
        tool_reply(RUBRIC_TOOL, rubric_answer(criterion("handled"), needy)),
        tool_reply(RUBRIC_TOOL, rubric_answer()),
    )
    disagreeing = author_rubric(
        fixture.project, fixture.settings, reference, identity="rubric-needy", code_revision="t"
    )
    agreeing = author_rubric(
        fixture.project, fixture.settings, reference, identity="rubric-plain", code_revision="t"
    )
    world = load_world_draft(fixture.project, world_ref).world
    assert world is not None and disagreeing.rubric is not None and agreeing.rubric is not None

    review = review_compatibility(packet, world, disagreeing.rubric)
    clean = review_compatibility(packet, world, agreeing.rubric)

    assert not review.compatible
    assert {(row.kind, row.subject) for row in review.disagreements} == {
        ("unobservable_evidence", "lock.respected"),
        ("unbound_rule", "tickets.lock"),
    }
    assert review.world_sha256 == world.identity
    assert review.rubric_sha256 == disagreeing.rubric.identity
    assert clean.compatible and not clean.disagreements


HELD_ONLY = Rule(
    rule_id="tickets.held-only",
    tool="close",
    kind="precondition",
    statement="Only a ticket another agent holds can be closed.",
    predicate=Predicate(
        op="in",
        left=Operand(scope="before", path="/tickets/lock"),
        right=Operand(value=["held"]),
    ),
    status="unknown",
    support=1,
    contradiction=0,
)
"""An unwitnessed rule the served world contradicts on both sides of its boundary."""

FREE_LOCK = Rule(
    rule_id="tickets.free-lock",
    tool="close",
    kind="precondition",
    statement="A ticket nobody holds can be closed.",
    predicate=Predicate(
        op="in",
        left=Operand(scope="before", path="/tickets/lock"),
        right=Operand(value=["none"]),
    ),
    status="supported",
    support=2,
    contradiction=0,
)
"""A supported rule the served world confirms."""


def _packet(
    fixture: GroundedFixture,
    *,
    checkpoint: ArtifactInput,
    step: int,
    extra: tuple[Rule, ...] = (),
) -> tuple[AuthoringPacket, ArtifactInput]:
    """Build and save the ticket packet at one boundary with extra rules in its book.

    Args:
        fixture: The grounded ticket fixture.
        checkpoint: The decision checkpoint.
        step: Its boundary index.
        extra: Rules added to the ticket rule book.

    Returns:
        The packet and its saved reference.
    """
    project = fixture.project
    book = RuleBook(
        contracts={tool.name: ToolContract(name=tool.name) for tool in TOOLS},
        rules=(RULE, LOCK_RULE, *extra),
        questions=(QUESTION,),
        sources=("trace-1",),
    )
    rule_book = artifact_input(
        project.artifacts.write(
            artifact_id=f"ticket-freeze-book-{step}-{len(extra)}",
            artifact_type=RULE_BOOK_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1, created_at=datetime.now(UTC), code_revision="test"
            ),
            files=rulebook_files(book),
        )
    )
    start = load_continuation_checkpoint(project.artifacts, checkpoint)
    packet = build_authoring_packet(
        family_id="ticket-tier",
        task=fixture.definition.task,
        checkpoint=checkpoint,
        before_state=INITIAL,
        public_history=start.actor.messages,
        remaining_calls=4,
        sources=(
            SourceRef(
                source_id="trace-1",
                kind="trace",
                trace_id="trace-1",
                summary="An agent closed a low tier ticket.",
            ),
        ),
        exposure=plan(),
        rule_book=rule_book,
        book=book,
        decision=tier_decision(step),
        scenarios=(tier_scenario("urgent", "high"), tier_scenario("routine", "low")),
        observed_episodes=(observed_episode(),),
    )
    return packet, save_packet(project, packet, code_revision="test")


def _authored(
    fixture: GroundedFixture,
    monkeypatch: pytest.MonkeyPatch,
    sandbox: ProgramSandbox,
    packet: AuthoringPacket,
    reference: ArtifactInput,
    identity: str,
    close: str = CLOSE,
) -> ArtifactInput:
    """Author a ready world for a packet and return its saved draft."""
    wire(monkeypatch, tool_reply(WORLD_TOOL, world_answer(packet, close=close)))
    draft = author_world(
        fixture.project,
        fixture.settings,
        reference,
        sandbox=sandbox,
        identity=identity,
        code_revision="test",
    )
    assert draft.status == "ready", draft.failure
    return world_draft_reference(fixture.project, draft)


def test_resume_starts_carry_world_history_and_restore_in_the_composed_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later decision keeps its predictor history; the served world runs the programs."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    study = execute_continuation_job(fixture.project, fixture.job, code_revision="test")
    start, boundary = fixture.job.study.checkpoint, study.branches[0].checkpoints[0]
    source = load_continuation_checkpoint(fixture.project.artifacts, boundary).world
    assert isinstance(source, ExecutableCheckpoint)
    _, history = simulated_image(source)
    assert len(history) == 1
    packet, reference = _packet(fixture, checkpoint=boundary, step=1)
    with ProgramSandbox() as sandbox:
        world_ref = _authored(fixture, monkeypatch, sandbox, packet, reference, "late-world")
        freeze = freeze_world(
            fixture.project,
            reference,
            world_ref,
            sandbox=sandbox,
            code_revision="test",
            base=fixture.definition,
            trajectory=ResumeTrajectory(
                branch_checkpoints=(start, boundary),
                decision=tier_decision(1),
                scenarios=(tier_scenario("urgent", "high"), tier_scenario("routine", "low")),
            ),
        )
        careless = load_world_draft(
            fixture.project,
            _authored(
                fixture, monkeypatch, sandbox, packet, reference, "late-careless", CARELESS_CLOSE
            ),
        ).world
    world = load_world_draft(fixture.project, world_ref).world
    assert world is not None and careless is not None

    assert [(row.variant_id, row.status, row.restored) for row in freeze.variants] == [
        ("urgent", "frozen", True),
        ("routine", "frozen", True),
    ]
    for row in freeze.variants:
        assert row.checkpoint is not None
        saved = load_continuation_checkpoint(fixture.project.artifacts, row.checkpoint)
        assert isinstance(saved.world, ExecutableCheckpoint)
        state, carried = simulated_image(saved.world)
        assert carried == history and state == packet.variant(row.variant_id).state
        assert "induced.json" in saved.world.image.parts
        definition = world_runtime_definition(
            packet, world, row.variant_id, base=fixture.definition
        )
        assert definition.programs is not None
        assert row.definition_sha256 == sha256_json(definition)
        unbound = definition.model_copy(update={"programs": None})
        assert sha256_json(unbound) != row.definition_sha256
        other = world_runtime_definition(packet, careless, row.variant_id, base=fixture.definition)
        assert sha256_json(other) != row.definition_sha256

    urgent = freeze.variants[0]
    assert urgent.checkpoint is not None
    definition = world_runtime_definition(packet, world, "urgent", base=fixture.definition)
    saved = load_continuation_checkpoint(fixture.project.artifacts, urgent.checkpoint)
    assert isinstance(saved.world, ExecutableCheckpoint)
    before = len(fixture.predictor.requests)
    with continuation_runtime(fixture.project, definition.model_dump(mode="json")) as runtime:
        assert isinstance(runtime.world, ExecutableWorldRuntime)
        with runtime.world.restore(saved.world_task, saved.world) as session:
            reply = session.execute(ToolCall(name="close", call_id="close", arguments={}))
    assert reply.metadata["transition"] == "induced" and not reply.is_error
    assert len(fixture.predictor.requests) == before


def test_freeze_executes_rule_challenges_in_the_served_world_and_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A supported boundary is confirmed, an unwitnessed one contradicted, both replayed."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    packet, reference = _packet(
        fixture, checkpoint=fixture.job.study.checkpoint, step=0, extra=(HELD_ONLY, FREE_LOCK)
    )
    with ProgramSandbox() as sandbox:
        world_ref = _authored(fixture, monkeypatch, sandbox, packet, reference, "challenged")
        freeze, replay = (
            freeze_world(
                fixture.project,
                reference,
                world_ref,
                sandbox=sandbox,
                code_revision="test",
                base=fixture.definition,
                trajectory=_trajectory(fixture),
            )
            for _ in range(2)
        )

    assert replay == freeze
    assert freeze.challenges is not None
    report = ChallengeReport.model_validate_json(
        fixture.project.artifacts.read_bytes(freeze.challenges.artifact_id, "report.json")
    )
    assert report.rules == freeze.challenge_rules and report.runs == freeze.workloads
    statuses = {row.rule_id: (row.prior, row.status) for row in freeze.challenge_rules}
    assert statuses == {
        "tickets.held-only": ("unknown", "contradicted"),
        "tickets.free-lock": ("supported", "supported"),
    }
    assert {run.status for run in freeze.workloads} == {"executed"}
    assert {run.case.scope for run in freeze.workloads} == {"urgent", "routine"}
    assert report.diagnostic == ("tickets.held-only",) and not report.world_defects
    assert "tickets.held-only" in freeze.diagnostic_rules
    assert {row.status for row in freeze.variants} == {"diagnostic"}
    held = next(rule for rule in report.book.rules if rule.rule_id == "tickets.held-only")
    assert held.status == "contradicted" and held.counter_evidence
