"""Variant checkpoints fork before revealing turns over real saved grounded and native worlds."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, JsonObject, JsonValue
from exp.common.models import (
    AssistantAction,
    ModelCapabilities,
    ModelResponse,
    OperationEconomics,
    ToolCall,
)
from exp.common.project import ProjectStore, artifact_input
from exp.common.tasks import TaskCase
from exp.common.tasks.facts import FactProvenance, FactValidation, InstanceFact, InstanceFacts
from exp.optimize.evaluation.continuations.job import execute_continuation_job
from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.optimize.evaluation.variance.contracts import (
    Criterion,
    DecisionOption,
    DecisionState,
    Scenario,
    StateEdit,
    VariationAxis,
    WitnessStep,
)
from exp.optimize.evaluation.variance.variants import (
    VariantCheckpoint,
    build_variant_checkpoint,
    verify_restoration,
)
from exp.runtime.agents.chat import ChatAgentRuntime, ChatCheckpoint
from exp.runtime.agents.chat_test import _Model, _response
from exp.runtime.environments.tests.world_conformance_test import _task
from exp.runtime.environments.world_contracts import (
    ExecutableCheckpoint,
    StateDomain,
    ToolEffectPolicy,
    WorldManifest,
)
from exp.runtime.environments.worlds import ExecutableWorldRuntime
from exp.simulation.continuations import (
    ContinuationCheckpoint,
    load_continuation_checkpoint,
    persist_continuation_checkpoint,
)
from exp.simulation.world_model.executable import GroundedWorldFactory
from exp.simulation.world_model.grounded_session import GroundedTransitionEngine
from exp.simulation.world_model.runtime_test import _runtime

_INITIAL: JsonObject = {
    "tickets": {"status": "open", "priority": "low", "due_time": 90},
    "archive": {},
}
_REPLIES: tuple[JsonObject, ...] = (
    {"note": "queued"},
    {"status": "open"},
    {"note": "noted"},
)


def _grounded(task: TaskCase) -> ExecutableWorldRuntime:
    """A real grounded world whose scripted predictor replies with ``_REPLIES`` in order.

    Args:
        task: Public task the world serves.

    Returns:
        The executable runtime, able to open and restore the same world.
    """
    predictor, _, _ = _runtime("", capabilities=ModelCapabilities(supports_tools=True))
    engine = GroundedTransitionEngine(
        predictor, task, maximum_output_tokens=1024, output_protocol="json-tool"
    )
    manifest = WorldManifest(
        domains=tuple(StateDomain(name=name, owner="simulated") for name in _INITIAL),
        tools=(
            ToolEffectPolicy(name="escalate", reads=("tickets", "archive"), writes=("tickets",)),
        ),
    )
    calls: list[str] = []

    def complete(body: JsonObject, operation: str) -> JsonObject:
        """Return the next scripted proposal, leaving state unchanged.

        Args:
            body: Prediction request.
            operation: Operation identity.

        Returns:
            One raw provider response.
        """
        del body, operation
        calls.append("call")
        reply = _REPLIES[len(calls) - 1]
        return ModelResponse(
            model=predictor.artifact.model,
            economics=OperationEconomics(),
            output=AssistantAction(
                tool_calls=(
                    ToolCall(
                        name="submit_transition",
                        call_id="proposal",
                        arguments={
                            "observations": [
                                {
                                    "call_id": f"c-{'abc'[len(calls) - 1]}",
                                    "reply": reply,
                                    "encoding": "json",
                                }
                            ],
                            "effect": {"kind": "edits", "edits": []},
                        },
                    ),
                )
            ),
        ).model_dump(mode="json")

    factory = GroundedWorldFactory(
        engine, manifest, _INITIAL, complete=complete, record=lambda event: None
    )
    return ExecutableWorldRuntime(manifest, factory, owner="simulated")


def _trajectory(
    project: ProjectStore, task: TaskCase
) -> tuple[tuple[ArtifactInput, ...], ExecutableWorldRuntime]:
    """Run three real tool turns and save the paired checkpoint at every boundary.

    Args:
        project: Project receiving the checkpoints.
        task: Public task.

    Returns:
        Boundary references, start first, and the runtime that can restore them.
    """
    source = project.artifacts.write_json(
        artifact_id="authored-source",
        artifact_type="authored-task",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), code_revision="test"
        ),
        files={"task.json": {"origin": "authored variant fixture"}},
    )
    runtime = _grounded(task)
    actor = ChatAgentRuntime(maximum_model_calls=6)
    turns = [
        _response(
            AssistantAction(
                tool_calls=(ToolCall(name="escalate", call_id=f"c-{letter}", arguments={}),)
            )
        )
        for letter in "abc"
    ]
    saved: list[ArtifactInput] = []
    with runtime.open(task) as session:

        def capture(checkpoint: ChatCheckpoint) -> None:
            """Persist both halves at a completed actor turn."""
            paired = ContinuationCheckpoint(
                task=task,
                actor=checkpoint,
                world=session.checkpoint(),
                sources=(artifact_input(source),),
            )
            saved.append(
                persist_continuation_checkpoint(
                    project.artifacts, paired, created_at=datetime.now(UTC), code_revision="test"
                )
            )

        actor.run(
            task,
            model=_Model((*turns, _response(AssistantAction(content="done")))),
            environment=session,
            on_checkpoint=capture,
        )
    return tuple(saved), runtime


def _decision(step: int, path: str, values: tuple[JsonValue, ...]) -> DecisionState:
    """A decision with one axis at ``path``.

    Args:
        step: Decision boundary.
        path: Axis pointer.
        values: Permitted values.

    Returns:
        The decision state.
    """
    return DecisionState(
        step=step,
        decision="Escalate or close the ticket",
        options=(
            DecisionOption(option_id="escalate", action="Escalate the ticket"),
            DecisionOption(option_id="close", action="Close the ticket"),
        ),
        relevant_paths=(path,),
        axes=(VariationAxis(path=path, values=values, effect="Changes the right choice."),),
        criteria=(Criterion(criterion_id="right", text="The right option was taken."),),
        evidence_refs=("point:3",),
    )


def _scenario(path: str, value: JsonValue) -> Scenario:
    """One scenario editing ``path`` to ``value``."""
    return Scenario(
        scenario_id="variant",
        edits=(StateEdit(path=path, value=value),),
        correct_option="close",
        deciding_paths=(path,),
        witness=(WitnessStep(tool="escalate", arguments={}, reveals=(path,)),),
        rationale="The edited value makes closing correct.",
        criteria=("right",),
    )


def _status(check_name: str, variant: VariantCheckpoint) -> str:
    """Return one recorded check's status by name."""
    return next(check.status for check in variant.checks if check.name == check_name)


def _checkpoint_count(project: ProjectStore) -> int:
    """Count saved continuation checkpoints."""
    return len(project.artifacts.list_ids(artifact_type="continuation-checkpoint"))


def test_variant_forks_before_the_turn_that_revealed_the_edited_value_and_restores(
    tmp_path: Path,
) -> None:
    """Turn 2 shows the status, so the variant forks at boundary 1 and replays two turns."""
    project = ProjectStore(tmp_path, "case")
    task = _task("escalate")
    boundaries, runtime = _trajectory(project, task)
    assert len(boundaries) == 4

    variant = build_variant_checkpoint(
        project,
        base=boundaries[3],
        branch_checkpoints=boundaries,
        decision=_decision(3, "/tickets/status", ("open", "closed")),
        scenario=_scenario("/tickets/status", "closed"),
        facts=None,
        code_revision="test",
        maximum_model_calls=6,
    )

    assert variant.problems == ()
    assert (variant.revealing_turn, variant.fork_turn, variant.replay_turns) == (2, 1, 2)
    assert variant.fork == boundaries[1] and variant.reference is not None
    assert [check.status for check in variant.checks if check.status != "passed"] == []
    saved = load_continuation_checkpoint(project.artifacts, variant.reference)
    fork = load_continuation_checkpoint(project.artifacts, boundaries[1])
    assert saved.actor == fork.actor and saved.world_task == fork.world_task
    assert verify_restoration(project, variant.reference)
    assert verify_restoration(
        project,
        variant.reference,
        runtime=runtime,
        expected_state_sha256=variant.edited_state_sha256,
    )
    assert isinstance(saved.world, ExecutableCheckpoint)
    with runtime.restore(task, saved.world) as restored:
        assert restored.checkpoint().image == saved.world.image
        assert isinstance(fork.world, ExecutableCheckpoint)
        assert restored.domain_digests() != fork.world.image.domains
    assert not verify_restoration(project, variant.reference, expected_state_sha256="0" * 64)
    assert not verify_restoration(project, boundaries[1], expected_state_sha256="0" * 64)


def test_an_unrevealed_edit_is_applied_at_the_decision_itself(tmp_path: Path) -> None:
    """No turn mentions the priority, so the fork is the decision boundary with no replay."""
    project = ProjectStore(tmp_path, "case")
    boundaries, runtime = _trajectory(project, _task("escalate"))

    variant = build_variant_checkpoint(
        project,
        base=boundaries[3],
        branch_checkpoints=boundaries,
        decision=_decision(3, "/tickets/priority", ("low", "high")),
        scenario=_scenario("/tickets/priority", "high"),
        facts=None,
        code_revision="test",
    )

    assert (variant.revealing_turn, variant.fork_turn, variant.replay_turns) == (None, 3, 0)
    assert variant.reference is not None and variant.problems == ()
    assert _status("reachability", variant) == "review"
    assert verify_restoration(project, variant.reference, runtime=runtime)


def test_edits_stated_by_public_context_or_instance_facts_are_refused_unpersisted(
    tmp_path: Path,
) -> None:
    """Public context and facts condition the actor and the world, so they cannot vary."""
    project = ProjectStore(tmp_path, "case")
    task = _task("escalate").model_copy(
        update={"initial_context": {"tickets": {"priority": "low"}}}
    )
    boundaries, _ = _trajectory(project, task)
    before = _checkpoint_count(project)

    public = build_variant_checkpoint(
        project,
        base=boundaries[3],
        branch_checkpoints=boundaries,
        decision=_decision(3, "/tickets/priority", ("low", "high")),
        scenario=_scenario("/tickets/priority", "high"),
        facts=None,
        code_revision="test",
    )
    fact = InstanceFact(
        key="tickets/due_time",
        value=90,
        visibility="conditioning",
        provenance=FactProvenance(
            adapter="fixture",
            source="fixture",
            validation=FactValidation(status="validated", method="fixture"),
            state_version=0,
        ),
    )
    stated = build_variant_checkpoint(
        project,
        base=boundaries[3],
        branch_checkpoints=boundaries,
        decision=_decision(3, "/tickets/due_time", (90, 30)),
        scenario=_scenario("/tickets/due_time", 30),
        facts=InstanceFacts(facts=(fact,)),
        code_revision="test",
    )

    assert public.reference is None and _status("public_context", public) == "failed"
    assert public.revealing_turn == 0 and _status("fork_placement", public) == "failed"
    assert any("initial_context states /tickets/priority" in row for row in public.problems)
    assert stated.reference is None and _status("instance_facts", stated) == "failed"
    assert any("tickets/due_time" in row for row in stated.problems)
    assert _checkpoint_count(project) == before


def test_clock_dependent_edits_are_flagged_for_review(tmp_path: Path) -> None:
    """A due time is persisted but marked for an explicit clock decision."""
    project = ProjectStore(tmp_path, "case")
    boundaries, _ = _trajectory(project, _task("escalate"))

    variant = build_variant_checkpoint(
        project,
        base=boundaries[3],
        branch_checkpoints=boundaries,
        decision=_decision(3, "/tickets/due_time", (90, 30)),
        scenario=_scenario("/tickets/due_time", 30),
        facts=None,
        code_revision="test",
        maximum_model_calls=6,
    )

    assert variant.reference is not None
    assert [check.name for check in variant.needs_review] == ["clock_and_events"]
    assert "due_time" in variant.needs_review[0].detail


def test_a_base_outside_the_trajectory_or_an_undeclared_edit_is_refused(tmp_path: Path) -> None:
    """The base must be the decision's boundary and edits must be declared axes."""
    project = ProjectStore(tmp_path, "case")
    boundaries, _ = _trajectory(project, _task("escalate"))
    decision = _decision(3, "/tickets/priority", ("low", "high"))

    misplaced = build_variant_checkpoint(
        project,
        base=boundaries[2],
        branch_checkpoints=boundaries,
        decision=decision,
        scenario=_scenario("/tickets/priority", "high"),
        facts=None,
        code_revision="test",
    )
    undeclared = build_variant_checkpoint(
        project,
        base=boundaries[3],
        branch_checkpoints=boundaries,
        decision=decision,
        scenario=_scenario("/tickets/priority", "urgent"),
        facts=None,
        code_revision="test",
    )

    assert _status("trajectory", misplaced) == "failed" and misplaced.reference is None
    assert _status("edits", undeclared) == "failed" and undeclared.reference is None


def test_native_world_state_is_refused_for_a_new_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Workbook bytes from a real continuation job cannot be edited by a state pointer."""
    project, job, _ = job_fixture(tmp_path, monkeypatch)
    study = execute_continuation_job(project, job, code_revision="test")
    branch = next(row for row in study.branches if row.checkpoints)
    boundaries = (study.spec.checkpoint, *branch.checkpoints)

    variant = build_variant_checkpoint(
        project,
        base=boundaries[1],
        branch_checkpoints=boundaries,
        decision=_decision(1, "/orders/zqx_flag", ("zqx-one", "zqx-two")),
        scenario=_scenario("/orders/zqx_flag", "zqx-two"),
        facts=None,
        code_revision="test",
    )

    assert variant.reference is None and variant.fork_turn == 1
    assert _status("world_state", variant) == "failed"
    assert "new initialization" in variant.problems[0]
