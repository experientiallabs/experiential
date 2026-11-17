"""Development continuation experiments over paired, frozen actor/world checkpoints.

The application reuses the built-in chat actor and executable-world restoration. It records
probabilities before outcomes, retains every planned branch and never labels its own results
as independent evidence or admitted training targets. Provider admission remains with the
injected model client. A saved branch replays without opening an environment or calling a model;
an interrupted branch is retained as outcome-unknown and is never silently rerolled.
"""

from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from operator import not_
from typing import Literal, Protocol

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    FailureCode,
    JsonObject,
    Sha256,
    StructuredFailure,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.models import (
    ModelClient,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    ReasoningEffort,
    ToolCall,
)
from exp.common.project import ArtifactStore, artifact_input
from exp.common.rollouts import StopReason
from exp.common.tasks import TaskCase
from exp.optimize.evaluation.continuations.critic import require_prediction_binding
from exp.optimize.evaluation.continuations.deciders import RecordedDecider
from exp.optimize.evaluation.equivalence.policy import load_admission_policy
from exp.runtime.agents.cascade import (
    CascadeDecisionRecord,
    CascadeModelClient,
    CascadePolicy,
    NextActionDecider,
)
from exp.runtime.agents.chat import ChatAgentRuntime, ChatCheckpoint, InvisibleToolCallError
from exp.runtime.environments.interface import EnvironmentSession, Observation
from exp.runtime.environments.mixed import MixedCheckpoint, MixedRealityEnvironmentRuntime
from exp.runtime.environments.world_artifacts import persist_world_image
from exp.runtime.environments.world_contracts import ExecutableCheckpoint
from exp.runtime.environments.world_outcomes import (
    execute_with_recovery,
    execution_receipt,
    known_failed_effects,
)
from exp.runtime.environments.worlds import ExecutableWorldRuntime
from exp.runtime.models.call_evidence import call_consumer, canonical_call
from exp.runtime.models.providers.errors import (
    ProviderRetryableResponseError,
    ProviderTruncatedResponseError,
)
from exp.simulation.continuations import (
    ContinuationCheckpoint,
    WorldCheckpoint,
    load_continuation_checkpoint,
    persist_continuation_checkpoint,
)
from exp.simulation.development_scope import require_development_task
from exp.simulation.evaluation.tasks import TaskVerifier, VerifierGrade, grade

WorldRuntime = ExecutableWorldRuntime | MixedRealityEnvironmentRuntime

# An actor call that stopped at its output-token allowance with no complete tool call: the
# allowance was spent (typically on reasoning) before the actor acted. Grading is unchanged (a
# refused call stays unassessable; a returned length stop fails at a declared limit), and reports
# count these branches on their own instead of leaving them as unexplained unknowns.
ActorStop = Literal["actor_budget_exhausted"]


class TrainingEvidence(ContractModel):
    """Existing task evidence pinned before generating supervised candidates.

    Attributes:
        qualification: Challenged observable-contract qualification for the exact task.
        comparison: Optional saved paired actor/environment measurement, not a certificate.
        comparison_case: Exact candidate case covered by the comparison, required with it.
    """

    qualification: ArtifactInput
    comparison: ArtifactInput | None = None
    comparison_case: int | None = Field(default=None, ge=0, strict=True)

    @model_validator(mode="after")
    def paired_reference(self) -> "TrainingEvidence":
        """Refuse ambiguous report-wide applicability claims."""
        if (self.comparison is None) != (self.comparison_case is None):
            raise ValueError("comparison and comparison_case must be supplied together")
        return self

    def sources(self) -> tuple[ArtifactInput, ...]:
        """Return all immutable prerequisites without inferring their authority."""
        return (self.qualification,) + (() if self.comparison is None else (self.comparison,))


class ContinuationStudySpec(ContractModel):
    """Frozen registration for repeated development continuations from one boundary.

    Attributes:
        checkpoint: Exact paired actor/world checkpoint.
        actor_model: Expected model identity for every response.
        verifier: Declared assessment predicate; not independent label authority.
        probe_sha256: Identity of the evaluator-only state projection.
        repeats: Planned fresh continuations, all retained in the denominator.
        predicted_success: Optional critic estimate registered before any branch executes.
        critic_prediction: Exact prospective estimator evidence, when supplied.
        selection_plan: Registered matched-checkpoint plan, when this is one of its cells.
        selector: Identity and rationale of the checkpoint-selection policy.
        temperature: Explicit actor sampling temperature for every continuation call.
        maximum_output_tokens: Hard per-call output allowance, separate from remaining calls.
        reasoning_effort: Requested actor reasoning effort for every continuation call, or None
            for the model's catalog default; it must be one the actor deployment declares it
            can preserve. A lower effort leaves the output allowance for the tool call instead
            of spending it on reasoning.
        training_specification: Optional independent acceptance specification frozen before
            outcomes. Its absence keeps every branch ineligible for supervised export.
        training_evidence: Optional qualified-task and paired measurement handoff. Without it,
            admission checks observed outcomes only and carries no task qualification.
        cascade: Opt-in decision cascade over the actor model, off when absent. Its decider
            decides accepted steps; the actor model stays responsible for every other step.
        admission_policy: Saved admission policy pinned before any branch executes. The
            registration cites it, which is what lets that policy admit the study's branches
            later; a study without it cannot be admitted under any policy.
    """

    checkpoint: ArtifactInput
    critic_prediction: ArtifactInput | None = None
    selection_plan: ArtifactInput | None = None
    actor_model: ModelSnapshot
    verifier: TaskVerifier
    probe_sha256: Sha256
    repeats: int = Field(default=4, ge=1, le=100, strict=True)
    predicted_success: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    selector: str = Field(min_length=1, max_length=2000)
    temperature: float = Field(default=1.0, ge=0, le=2, allow_inf_nan=False)
    maximum_output_tokens: int = Field(default=1024, gt=0, strict=True)
    reasoning_effort: ReasoningEffort | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    training_specification: ArtifactInput | None = None
    training_evidence: TrainingEvidence | None = None
    cascade: CascadePolicy | None = Field(default=None, exclude_if=lambda value: value is None)
    admission_policy: ArtifactInput | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def cascade_measures_behavior(self) -> "ContinuationStudySpec":
        """Keep cascade episodes out of supervised training admission."""
        if self.cascade is not None and (
            self.training_specification is not None or self.training_evidence is not None
        ):
            raise ValueError("cascade episodes measure behavior; training admission is separate")
        return self


def study_identity(spec: ContinuationStudySpec, code_revision: str) -> str:
    """Name a study registration by its frozen spec and producing implementation.

    Args:
        spec: Pre-outcome registration.
        code_revision: Exact producing implementation; a job passes its ``implementation``.

    Returns:
        The registration identity; branches and the result extend it with suffixes.
    """
    return stable_id("continuation-study", {"spec": sha256_json(spec), "code": code_revision})


def actor_budget_exhausted(branches: Iterable["ContinuationBranch"]) -> int:
    """Count branches whose actor spent its output allowance without acting.

    The count is reported beside, not instead of, the assessed and unassessable counts.
    """
    return sum(branch.actor_stop == "actor_budget_exhausted" for branch in branches)


def known_outcomes(grades: Iterable[VerifierGrade | None]) -> tuple[int, ...]:
    """Binary labels of assessed outcomes: 1 for a pass, 0 for a known failure.

    Missing and unassessable grades have unknown outcomes and are left out; callers keep them in
    the planned denominator as missing-outcome bounds.
    """
    return tuple(
        int(grade.status == "pass")
        for grade in grades
        if grade is not None and grade.status != "unassessable"
    )


class ContinuationBranch(ContractModel):
    """One planned branch, including operationally incomplete measurements.

    Attributes:
        index: Zero-based planned branch position.
        status: Completed execution, known execution failure, or ambiguous outcome.
        outcome: Actor actions, observations and evaluator-only state projections.
        assessment: Goal/policy assessment when execution produced a saved outcome.
        failure: Sanitized exception type, never provider credentials or raw error text.
        failure_detail: Operator-only phase, stable reason and originating intent reference.
        final_world: Exact restored-world result for independent later inspection, when captured.
        checkpoints: Durable paired boundaries after fully observed tool turns, in call order.
        actor_calls: Durable final actor inputs and returned responses, including interrupted calls.
        cascade_decisions: With a cascade, which model decided each actor step and the decider's
            confidence, in step order. Empty without a cascade.
        actor_stop: ``actor_budget_exhausted`` when the actor's last call stopped at its output
            allowance with no complete tool call, whether the provider returned that response or
            refused it as empty after its bounded retries; None otherwise. It names the cause
            and never changes the assessment: a refused dispatch stays outcome-unknown and
            unassessable, and a returned response is graded by the verifier as before.
    """

    index: int
    status: Literal["completed", "failed", "outcome_unknown"]
    outcome: JsonObject
    assessment: VerifierGrade | None = None
    failure: str | None = None
    failure_detail: StructuredFailure | None = None
    final_world: ArtifactInput | None = None
    checkpoints: tuple[ArtifactInput, ...] = ()
    actor_calls: tuple[ArtifactInput, ...] = ()
    cascade_decisions: tuple[CascadeDecisionRecord, ...] = Field(default=(), exclude_if=not_)
    actor_stop: ActorStop | None = Field(default=None, exclude_if=lambda value: value is None)

    def grading_outcome(self) -> JsonObject:
        """Keep saved execution status authoritative over a promising partial observation."""
        outcome = dict(self.outcome)
        if self.status != "completed":
            outcome["stop_reason"] = self.status
        return outcome


class ContinuationStudy(ContractModel):
    """Development measurements, with unavailable outcomes retained explicitly.

    Attributes:
        spec: Frozen registration including the pre-outcome critic prediction.
        branches: Every planned branch in registration order.
        scope: Explicit development status; no independence or training admission claim.
    """

    spec: ContinuationStudySpec
    branches: tuple[ContinuationBranch, ...]
    scope: Literal["development"] = "development"

    def metrics(self) -> JsonObject:
        """Report known outcomes separately from all planned branches and critic coverage."""
        labels = known_outcomes(row.assessment for row in self.branches)
        probability = self.spec.predicted_success
        exhausted = actor_budget_exhausted(self.branches)
        return {
            "planned": len(self.branches),
            "assessable": len(labels),
            "unassessable": len(self.branches) - len(labels),
            "actor_budget_exhausted": exhausted,
            "actor_budget_exhausted_share": exhausted / len(self.branches)
            if self.branches
            else None,
            "successes": sum(labels),
            "resumed_success_rate": sum(labels) / len(labels) if labels else None,
            "brier": (
                sum((probability - label) ** 2 for label in labels) / len(labels)
                if probability is not None and labels
                else None
            ),
            "scope": "resumed checkpoint, development; branches share one task lineage",
            "training_admitted": False,
        }


class _ObservedEnvironment:
    """Execute-only actor capability; private state probes remain inside the application."""

    def __init__(
        self,
        session: EnvironmentSession,
        snapshot: Callable[[], WorldCheckpoint],
        probe: Callable[[WorldCheckpoint], JsonObject],
        outcome: JsonObject,
    ) -> None:
        """Bind the world and evaluator projection without exposing either to model messages."""
        self._session, self._snapshot, self._probe, self.outcome = session, snapshot, probe, outcome
        self.steps: list[JsonObject] = []

    def execute(self, action: ToolCall) -> Observation:
        """Retain attempted actions even when their execution or observation fails."""
        row: JsonObject = {"action": action.model_dump(mode="json"), "status": "failure"}
        self.steps.append(row)
        self.outcome["steps"] = list(self.steps)
        self.outcome["execution_phase"] = "tool_dispatch"
        try:
            observation = execute_with_recovery(self._session, action)
        except Exception:
            if known_failed_effects(execution_receipt(self._session, action)):
                self.outcome["execution_phase"] = "tool_dispatch_failed_known_effects"
            raise
        finally:
            receipt = execution_receipt(self._session, action)
            if receipt is not None:
                row["execution"] = receipt.model_dump(mode="json")
        self.outcome["execution_phase"] = "tool_observation"
        row.update(
            status="valid",
            reply=observation.model_dump(mode="json"),
            after=self._probe(self._snapshot()),
        )
        self.outcome["steps"] = list(self.steps)
        return observation


class _PinnedModel:
    """Check actual response identity against the registered actor policy."""

    def __init__(
        self,
        model: ModelClient,
        spec: ContinuationStudySpec,
        outcome: JsonObject,
        store: ArtifactStore,
        branch_id: str,
        revision: str,
    ) -> None:
        """Bind an already authorized model client and its expected identity."""
        self._model, self._spec, self._outcome = model, spec, outcome
        self._calls: list[JsonObject] = []
        self._store, self._branch_id, self._revision = store, branch_id, revision

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Reject identity drift before a returned action can affect the world."""
        request = request.model_copy(
            update={
                "temperature": self._spec.temperature,
                "maximum_output_tokens": self._spec.maximum_output_tokens,
                "reasoning_effort": self._spec.reasoning_effort,
            }
        )
        receipt: JsonObject = {"request": request.model_dump(mode="json")}
        self._outcome["execution_phase"] = "actor_request_persistence"
        identity = f"{self._branch_id}-call-{len(self._calls)}"
        intent = artifact_input(self._store.read(self._branch_id + "-intent").manifest)
        manifest = self._store.write_json(
            artifact_id=identity + "-intent",
            artifact_type="continuation-actor-request",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=self._revision,
                inputs=tuple(sorted((intent, self._spec.checkpoint), key=lambda r: r.artifact_id)),
            ),
            files={
                "request.json": request,
                "context.json": {
                    "projection": "chat actor after frozen sampling settings",
                    "initial_checkpoint": self._spec.checkpoint.model_dump(mode="json"),
                    "request_sha256": sha256_json(request),
                },
            },
        )
        reference = artifact_input(manifest)
        receipt["intent"] = reference.model_dump(mode="json")
        self._calls.append(receipt)
        self._outcome["model_calls"] = list(self._calls)
        self._outcome["execution_phase"] = "actor_model_dispatch"
        with canonical_call(reference), call_consumer("actor"):
            response = self._model.complete(request)
        self._outcome["execution_phase"] = "actor_response_persistence"
        self._store.write_json(
            artifact_id=identity + "-response",
            artifact_type="continuation-actor-response",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=self._revision,
                inputs=(reference,),
            ),
            files={"response.json": response},
        )
        receipt["response"] = response.model_dump(mode="json")
        self._outcome["model_calls"] = list(self._calls)
        if response.model != self._spec.actor_model:
            self._outcome["execution_phase"] = "actor_model_identity"
            raise ValueError("actor response identity differs from the frozen study")
        self._outcome["execution_phase"] = "actor_tool_authorization"
        return response


class CheckpointSession(EnvironmentSession, Protocol):
    """Shared restoration capability of the two supported executable runtime kinds."""

    def checkpoint(self) -> WorldCheckpoint:
        """Capture the current world without changing it."""
        ...


def restore_continuation_world(
    runtime: WorldRuntime, task: TaskCase, checkpoint: WorldCheckpoint
) -> AbstractContextManager[CheckpointSession]:
    """Restore only the declared world kind; never substitute a reset or replay."""
    if isinstance(runtime, ExecutableWorldRuntime) and isinstance(checkpoint, ExecutableCheckpoint):
        return runtime.restore(task, checkpoint)
    if isinstance(runtime, MixedRealityEnvironmentRuntime) and isinstance(
        checkpoint, MixedCheckpoint
    ):
        return runtime.restore(task, checkpoint)
    raise ValueError("checkpoint and runtime kinds differ; supply the captured world's runtime")


def run_continuation_study(
    store: ArtifactStore,
    spec: ContinuationStudySpec,
    *,
    actor: ChatAgentRuntime,
    model: ModelClient,
    world: WorldRuntime,
    probe: Callable[[WorldCheckpoint], JsonObject],
    probe_sha256: Sha256,
    code_revision: str,
    decider: NextActionDecider | None = None,
) -> ContinuationStudy:
    """Run or replay a frozen development study using existing actor and world execution.

    Args:
        store: Trusted durable artifact owner.
        spec: Pre-outcome registration; model and verifier cannot change on replay.
        actor: Stateless chat runtime matching the captured actor policy.
        model: Caller-admitted provider client with bounded spend and transport accounting.
        world: Exact executable or mixed runtime that can restore the checkpoint.
        probe: Evaluator-only projection of checkpoint state, never actor-visible.
        probe_sha256: Actual probe definition identity, required to match registration.
        code_revision: Exact producing implementation identity, part of study identity.
        decider: Caller-admitted cheap decider, required exactly when the spec has a cascade.

    Returns:
        Every planned branch. Unknown outcomes remain unknown; there is no automatic retry.
    """
    spec = ContinuationStudySpec.model_validate_json(spec.model_dump_json())
    if (spec.cascade is None) != (decider is None) or (
        spec.cascade is not None
        and decider is not None
        and decider.definition != spec.cascade.decider
    ):
        raise ValueError("a cascade study needs exactly its policy's frozen decider")
    checkpoint = load_continuation_checkpoint(store, spec.checkpoint)
    require_development_task(store, checkpoint.task)
    if spec.selection_plan is not None:
        from exp.optimize.evaluation.continuations.selection import require_selection_binding

        require_selection_binding(store, spec.selection_plan, spec)
    if spec.critic_prediction is not None:
        require_prediction_binding(
            store,
            spec.critic_prediction,
            checkpoint=spec.checkpoint,
            actor_model=spec.actor_model,
            temperature=spec.temperature,
            output_tokens=spec.maximum_output_tokens,
            reasoning_effort=spec.reasoning_effort,
            probability=spec.predicted_success,
            verifier_sha256=sha256_json(spec.verifier),
            probe_sha256=spec.probe_sha256,
        )
    if spec.admission_policy is not None:
        load_admission_policy(store, spec.admission_policy)
    if spec.training_specification is not None:
        manifest = store.read(spec.training_specification.artifact_id).manifest
        if (
            artifact_input(manifest) != spec.training_specification
            or manifest.artifact_type != "evaluation-specification"
        ):
            raise ValueError("training specification differs from its frozen evidence")
    if spec.training_evidence is not None:
        # Qualification consumes completed studies, so this execution-time import breaks
        # the evidence-reader/executor dependency cycle without loading providers.
        from exp.optimize.evaluation.continuations.evidence import verify_training_evidence

        verify_training_evidence(store, spec, execution_revision=code_revision)
    if checkpoint.task.partition != "fit":
        raise ValueError(
            "continuation development requires fit data; reserved tasks cannot be used"
        )
    if probe_sha256 != spec.probe_sha256 or actor.identity != checkpoint.actor.runtime_sha256:
        raise ValueError("continuation runtime or probe changed; use its frozen registration")
    if not actor.remaining_calls(checkpoint.actor):
        raise ValueError("continuation has no remaining actor calls; select an earlier turn")
    identity = study_identity(spec, code_revision)
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        result_id = identity + "-result"
        if store.exists(result_id):
            return ContinuationStudy.model_validate_json(store.read_bytes(result_id, "record.json"))
        _save(
            store,
            identity,
            spec,
            (spec.checkpoint,)
            + (() if spec.training_specification is None else (spec.training_specification,))
            + (() if spec.training_evidence is None else spec.training_evidence.sources())
            + (() if spec.critic_prediction is None else (spec.critic_prediction,))
            + (() if spec.selection_plan is None else (spec.selection_plan,))
            + (() if spec.admission_policy is None else (spec.admission_policy,)),
            code_revision,
        )
        registration = artifact_input(store.read(identity).manifest)
        branches = []
        for index in range(spec.repeats):
            branch_id = f"{identity}-branch-{index}"
            if store.exists(branch_id):
                branch = ContinuationBranch.model_validate_json(
                    store.read_bytes(branch_id, "record.json")
                )
            elif store.exists(branch_id + "-intent"):
                branch = ContinuationBranch(
                    index=index,
                    status="outcome_unknown",
                    outcome={},
                    failure="UnfinishedDispatchIntent",
                    failure_detail=StructuredFailure(
                        code=FailureCode.CANCELLED,
                        message=(
                            "Branch execution was interrupted; inspect saved actor calls "
                            "and world receipts before reconciliation. No automatic redraw."
                        ),
                        details={
                            "phase": "interrupted_branch",
                            "branch_intent": artifact_input(
                                store.read(branch_id + "-intent").manifest
                            ).model_dump(mode="json"),
                        },
                    ),
                    checkpoints=_saved_turns(store, branch_id),
                    actor_calls=saved_actor_call_evidence(store, branch_id),
                )
                _save(
                    store,
                    branch_id,
                    branch,
                    (
                        registration,
                        *branch.checkpoints,
                        *branch.actor_calls,
                        *(() if branch.final_world is None else (branch.final_world,)),
                    ),
                    code_revision,
                )
            else:
                _save(store, branch_id + "-intent", spec, (registration,), code_revision)
                branch = _run_branch(
                    index,
                    checkpoint,
                    actor,
                    model,
                    world,
                    probe,
                    spec,
                    store,
                    code_revision,
                    branch_id,
                    decider,
                )
                _save(
                    store,
                    branch_id,
                    branch,
                    (
                        registration,
                        *branch.checkpoints,
                        *branch.actor_calls,
                        *(() if branch.final_world is None else (branch.final_world,)),
                    ),
                    code_revision,
                )
            branches.append(branch)
        result = ContinuationStudy(spec=spec, branches=tuple(branches))
        inputs = tuple(
            artifact_input(store.read(f"{identity}-branch-{i}").manifest)
            for i in range(spec.repeats)
        )
        _save(store, result_id, result, (registration, *inputs), code_revision)
        return result


def _run_branch(
    index: int,
    checkpoint: ContinuationCheckpoint,
    actor: ChatAgentRuntime,
    model: ModelClient,
    world: WorldRuntime,
    probe: Callable[[WorldCheckpoint], JsonObject],
    spec: ContinuationStudySpec,
    store: ArtifactStore,
    code_revision: str,
    branch_id: str,
    decider: NextActionDecider | None,
) -> ContinuationBranch:
    """Restore one isolated world and retain failure as an explicit branch outcome."""
    outcome: JsonObject = {"steps": [], "stop_reason": "failure"}
    final_world: ArtifactInput | None = None
    points: list[ArtifactInput] = []
    intent = artifact_input(store.read(branch_id + "-intent").manifest)
    actor_model: ModelClient = _PinnedModel(model, spec, outcome, store, branch_id, code_revision)
    decisions: list[CascadeDecisionRecord] = []
    if spec.cascade is not None and decider is not None:
        actor_model = CascadeModelClient(
            main=actor_model,
            decider=RecordedDecider(
                decider,
                store=store,
                prefix=branch_id,
                inputs=(intent, spec.checkpoint),
                code_revision=code_revision,
            ),
            policy=spec.cascade,
            record=decisions.append,
        )
    outcome["execution_phase"] = "world_restore"
    try:
        with restore_continuation_world(world, checkpoint.world_task, checkpoint.world) as session:
            # Both supported runtime kinds expose checkpoint(), checked by the restore pairing.
            snapshot = session.checkpoint
            outcome["execution_phase"] = "initial_state_probe"
            outcome["initial_state"] = probe(snapshot())
            observed = _ObservedEnvironment(session, snapshot, probe, outcome)

            def capture_turn(point: ChatCheckpoint) -> None:
                """Persist paired visible context and state before another actor call is made."""
                outcome["execution_phase"] = "checkpoint_capture"
                reference = persist_continuation_checkpoint(
                    store,
                    ContinuationCheckpoint(
                        task=checkpoint.task,
                        environment_task=checkpoint.environment_task,
                        actor=point,
                        world=snapshot(),
                        sources=(spec.checkpoint, intent),
                    ),
                    created_at=datetime.now(UTC),
                    code_revision=code_revision,
                )
                store.write_json(
                    artifact_id=f"{branch_id}-turn-{point.calls}",
                    artifact_type="continuation-turn",
                    envelope=ArtifactEnvelope(
                        schema_version=1,
                        created_at=datetime.now(UTC),
                        code_revision=code_revision,
                        inputs=tuple(sorted((intent, reference), key=lambda ref: ref.artifact_id)),
                    ),
                    files={"checkpoint.json": reference},
                )
                points.append(reference)

            episode = actor.resume(
                checkpoint.task,
                checkpoint.actor,
                model=actor_model,
                environment=observed,
                on_checkpoint=capture_turn,
            )
            outcome["execution_phase"] = "final_state_capture"
            captured = snapshot()
            outcome["final_state"] = probe(captured)
            final_world = persist_world_image(
                store, captured, created_at=datetime.now(UTC), code_revision=code_revision
            )
            outcome["final_answer"] = (
                None if episode.final_action is None else episode.final_action.content
            )
            outcome["stop_reason"] = episode.stop_reason.value
            outcome["execution_phase"] = "world_cleanup"
        outcome["execution_phase"] = "verifier_grade"
        return ContinuationBranch(
            index=index,
            status="completed",
            outcome=outcome,
            assessment=grade(spec.verifier, outcome),
            final_world=final_world,
            checkpoints=tuple(points),
            actor_calls=saved_actor_call_evidence(store, branch_id),
            cascade_decisions=tuple(decisions),
            actor_stop="actor_budget_exhausted"
            if episode.stop_reason == StopReason.LENGTH
            and (episode.final_action is None or not episode.final_action.tool_calls)
            else None,
        )
    except (ValueError, OSError, RuntimeError) as error:
        # Runtime, environment and provider failures become retained branch outcomes. Any other
        # exception is a programming error: it propagates, and the saved intent keeps the branch
        # outcome-unknown on the next run instead of hiding the defect as a measurement.
        outcome["stop_reason"] = "failure"
        if isinstance(error, InvisibleToolCallError):
            steps = outcome.get("steps")
            if not isinstance(steps, list):
                raise TypeError("continuation outcome lost its step inventory") from error
            steps.append(
                {
                    "action": error.action.model_dump(mode="json"),
                    "status": "blocked",
                    "dispatch": "not_started",
                    "reason": "tool_not_visible",
                }
            )
            outcome["execution_phase"] = "actor_tool_authorization"
        return ContinuationBranch(
            index=index,
            status="failed"
            if outcome["execution_phase"] == "tool_dispatch_failed_known_effects"
            else "outcome_unknown",
            outcome=outcome,
            assessment=grade(spec.verifier, outcome),
            final_world=final_world,
            checkpoints=tuple(points),
            actor_calls=saved_actor_call_evidence(store, branch_id),
            cascade_decisions=tuple(decisions),
            actor_stop="actor_budget_exhausted"
            if outcome["execution_phase"] == "actor_model_dispatch"
            and (
                isinstance(error, ProviderTruncatedResponseError)
                or (isinstance(error, ProviderRetryableResponseError) and error.hit_length_limit)
            )
            else None,
            failure=type(error).__name__,
            failure_detail=StructuredFailure(
                code=FailureCode.TIMEOUT
                if isinstance(error, TimeoutError)
                else FailureCode.VALIDATION
                if isinstance(error, ValueError)
                else FailureCode.INTERNAL,
                message=(
                    "Continuation stopped at the recorded phase; inspect the linked branch "
                    "intent and actor-call receipts. A missing result is not proof that no "
                    "effect occurred."
                ),
                exception_type=type(error).__name__,
                details={
                    "phase": outcome["execution_phase"],
                    "branch_intent": intent.model_dump(mode="json"),
                },
            ),
        )


def saved_actor_call_evidence(store: ArtifactStore, branch_id: str) -> tuple[ArtifactInput, ...]:
    """Read the contiguous immutable call inventory, including an intent with unknown outcome."""
    branch_intent = artifact_input(store.read(branch_id + "-intent").manifest)
    spec = ContinuationStudySpec.model_validate_json(
        store.read_bytes(branch_id + "-intent", "record.json")
    )
    refs = []
    index = 0
    while True:
        identity = f"{branch_id}-call-{index}"
        if not store.exists(identity + "-intent"):
            if store.exists(identity + "-response"):
                raise ValueError("actor response is missing its original request intent")
            return tuple(refs)
        intent = store.read(identity + "-intent").manifest
        if intent.artifact_type != "continuation-actor-request" or set(intent.inputs) != {
            branch_intent,
            spec.checkpoint,
        }:
            raise ValueError("actor request lost its branch or checkpoint binding")
        request = ModelRequest.model_validate_json(
            store.read_bytes(intent.artifact_id, "request.json")
        )
        if (
            request.temperature != spec.temperature
            or request.maximum_output_tokens != spec.maximum_output_tokens
            or request.reasoning_effort != spec.reasoning_effort
        ):
            raise ValueError("actor request differs from the frozen sampling policy")
        reference = artifact_input(intent)
        refs.append(reference)
        if store.exists(identity + "-response"):
            response = store.read(identity + "-response").manifest
            if response.artifact_type != "continuation-actor-response" or response.inputs != (
                reference,
            ):
                raise ValueError("actor response lost its exact request binding")
            refs.append(artifact_input(response))
        index += 1


def _saved_turns(store: ArtifactStore, branch_id: str) -> tuple[ArtifactInput, ...]:
    """Recover committed boundaries after a process crash without resuming native effects."""
    intent = artifact_input(store.read(branch_id + "-intent").manifest)
    prefix = branch_id + "-turn-"
    points: dict[int, ArtifactInput] = {}
    for identity in store.list_ids(artifact_type="continuation-turn"):
        if not identity.startswith(prefix):
            continue
        call = int(identity[len(prefix) :])
        reference = ArtifactInput.model_validate_json(store.read_bytes(identity, "checkpoint.json"))
        point = load_continuation_checkpoint(store, reference)
        if (
            point.actor.calls != call
            or intent not in point.sources
            or set(store.read(identity).manifest.inputs) != {intent, reference}
        ):
            raise ValueError("captured turn lost its branch intent or call boundary")
        points[call] = reference
    return tuple(points[call] for call in sorted(points))


def _save(
    store: ArtifactStore,
    identity: str,
    value: ContractModel,
    inputs: tuple[ArtifactInput, ...],
    code_revision: str,
) -> None:
    """Publish immutable study records; the study lease serializes writers."""
    if not store.exists(identity):
        store.write_json(
            artifact_id=identity,
            artifact_type="continuation-study",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=tuple(sorted(inputs, key=lambda ref: ref.artifact_id)),
            ),
            files={"record.json": value},
        )
