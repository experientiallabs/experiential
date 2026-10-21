"""Deterministic matched-random controls for proposed development checkpoints.

Selection uses source identity, frozen environment semantics and remaining horizon only. It
never reads continuation outcomes or critic probabilities. Controls are sampled without
replacement; unmatched proposals remain in the plan with explicit reasons.
"""

import random
from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.project import ArtifactStore, ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job import (
    MAXIMUM_PHYSICAL_ATTEMPTS,
    ContinuationJob,
    preflight_continuation_job,
)
from exp.runtime.agents.chat import ChatAgentRuntime
from exp.runtime.environments.mixed import MixedCheckpoint
from exp.runtime.environments.world_contracts import ExecutableCheckpoint
from exp.simulation.continuations import ContinuationCheckpoint, load_continuation_checkpoint
from exp.simulation.development_scope import require_development_task

if TYPE_CHECKING:
    from exp.optimize.evaluation.continuations.study import ContinuationStudySpec


class CheckpointProposal(ContractModel):
    """Selector output retained independently of the outcomes later used to assess it.

    Attributes:
        checkpoint: Proposed point from the registered candidate pool.
        rationale: Selector explanation; never a reward or success label.
        evidence: Immutable evidence the selector used, including retrospective findings.
        probability: Selector sampling probability when known; unknown is explicitly None.
    """

    checkpoint: ArtifactInput
    rationale: str = Field(min_length=1, max_length=4000)
    evidence: tuple[ArtifactInput, ...] = Field(min_length=1, max_length=32)
    probability: float | None = Field(default=None, gt=0, le=1, allow_inf_nan=False)


class CheckpointSelectionSpec(ContractModel):
    """One task's proposed checkpoints and a finite, pre-registered control pool.

    Attributes:
        base: Actor/model/world policy and inference allowance shared by both arms.
        pool: Complete candidate checkpoint inventory before continuation outcomes.
        proposed: Ordered choices from a selector, with reasons and source references.
        selector_sha256: Frozen external selector policy identity; no training claim implied.
        seed: Registered random-control draw, never chosen after seeing its outcomes.
        horizon_tolerance: Maximum remaining-call difference; nearest candidates are preferred.
    """

    base: ContinuationJob
    pool: tuple[ArtifactInput, ...] = Field(min_length=2, max_length=256)
    proposed: tuple[CheckpointProposal, ...] = Field(min_length=1, max_length=32)
    selector_sha256: Sha256
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    horizon_tolerance: int = Field(default=0, ge=0, le=10)

    @model_validator(mode="after")
    def finite_unique(self) -> "CheckpointSelectionSpec":
        """Forbid repeated proposal IDs and bound the complete paired execution allowance."""
        if self.base.study.selection_plan is not None:
            raise ValueError("checkpoint selection plans cannot recursively inherit a prior plan")
        refs = tuple(row.checkpoint for row in self.proposed)
        if len(set(self.pool)) != len(self.pool) or len(set(refs)) != len(refs):
            raise ValueError("checkpoint pool and proposed identities must be unique")
        if not set(refs) <= set(self.pool):
            raise ValueError("every proposed checkpoint must belong to the registered pool")
        if len(self.proposed) * 2 * self.base.maximum_physical_attempts > (
            MAXIMUM_PHYSICAL_ATTEMPTS
        ):
            raise ValueError(
                f"checkpoint comparison exceeds the {MAXIMUM_PHYSICAL_ATTEMPTS} "
                "actor-attempt ceiling"
            )
        return self

    def identity(self, code_revision: str) -> str:
        """Bind proposals, explanations, population, random seed and every shared allowance."""
        return stable_id("checkpoint-selection", {"spec": sha256_json(self), "code": code_revision})


class CheckpointMatch(ContractModel):
    """A proposal and its random control, with conditional draw probability and horizon.

    Attributes:
        proposed: Original selector choice and rationale.
        control: Chosen random checkpoint, or None when matching is impossible.
        proposed_remaining: Calls remaining at the proposed point.
        control_remaining: Calls remaining at the matched point, when available.
        eligible_controls: Nearest eligible pool before this draw, after earlier draws.
        control_probability: Conditional uniform probability within that pool.
        refusal: Why a proposal could not receive a valid control.
    """

    proposed: CheckpointProposal
    control: ArtifactInput | None
    proposed_remaining: int
    control_remaining: int | None
    eligible_controls: int
    control_probability: float | None
    refusal: str | None = None


class CheckpointSelectionPlan(ContractModel):
    """Every proposed point and explicit control or exclusion, without outcome-based selection.

    Attributes:
        spec: Frozen input protocol.
        matches: Same order as the selector proposals, including unmatched points.
        scope: Limits on interpreting correlated branches of one source task.
    """

    spec: CheckpointSelectionSpec
    matches: tuple[CheckpointMatch, ...]
    scope: str = (
        "development; one source task; proposed versus horizon-matched random, not causal blame"
    )


def checkpoint_content_digest(point: ContinuationCheckpoint) -> Sha256:
    """Identify a checkpoint's task, actor context and world state, ignoring its episode label.

    Two checkpoints with the same digest are the same starting point, so neither adds a distinct
    control or pool entry.
    """
    world = point.world.model_dump(mode="json")
    world.pop("episode_id", None)
    return sha256_json(
        {
            "task": point.task.model_dump(mode="json"),
            "actor": point.actor.model_dump(mode="json"),
            "world": world,
        }
    )


def _world_policy(checkpoint: ExecutableCheckpoint | MixedCheckpoint) -> Sha256:
    """Compare frozen backend/ownership policies without equating different checkpoint states."""
    if isinstance(checkpoint, MixedCheckpoint):
        return sha256_json(
            {
                "kind": "mixed",
                "policy": checkpoint.policy_sha256,
                "native": None if checkpoint.native is None else _world_policy(checkpoint.native),
                "candidate": None
                if checkpoint.candidate is None
                else _world_policy(checkpoint.candidate),
            }
        )
    return sha256_json(
        {
            "kind": "executable",
            "seed": checkpoint.seed,
            "backend": checkpoint.backend_sha256,
            "manifest": checkpoint.manifest_sha256,
            "ownership": checkpoint.ownership,
            "owner": checkpoint.owner,
        }
    )


def match_checkpoints(
    project: ProjectStore, spec: CheckpointSelectionSpec
) -> CheckpointSelectionPlan:
    """Match task, world, actor policy and nearest horizon without provider calls."""
    spec = CheckpointSelectionSpec.model_validate_json(spec.model_dump_json())
    preflight_continuation_job(project, spec.base)
    return _match(project, spec)


def _match(project: ProjectStore, spec: CheckpointSelectionSpec) -> CheckpointSelectionPlan:
    """Reproduce the saved population match without importing runtime code or loading a catalog."""
    actor = ChatAgentRuntime(
        maximum_model_calls=spec.base.maximum_model_calls, system_prompt=spec.base.system_prompt
    )
    base = load_continuation_checkpoint(project.artifacts, spec.base.study.checkpoint)
    population = {}
    seen_points: set[str] = set()
    for reference in spec.pool:
        point = load_continuation_checkpoint(project.artifacts, reference)
        require_development_task(project.artifacts, point.task)
        if (
            point.task != base.task
            or point.world_task != base.world_task
            or _world_policy(point.world) != _world_policy(base.world)
        ):
            raise ValueError(
                "checkpoint controls must share the exact source task and world policy"
            )
        remaining = actor.remaining_calls(point.actor)
        digest = checkpoint_content_digest(point)
        if digest in seen_points:
            raise ValueError("duplicate checkpoint content does not add a random control")
        seen_points.add(digest)
        population[reference] = remaining
    for proposal in spec.proposed:
        for evidence in proposal.evidence:
            if artifact_input(project.artifacts.read(evidence.artifact_id).manifest) != evidence:
                raise ValueError("selector rationale evidence differs from its registered digest")
    available = set(spec.pool) - {row.checkpoint for row in spec.proposed}
    rng = random.Random(spec.seed)
    matches = []
    for proposal in spec.proposed:
        remaining = population[proposal.checkpoint]
        candidates = [
            ref
            for ref in available
            if population[ref] > 0
            and remaining > 0
            and abs(population[ref] - remaining) <= spec.horizon_tolerance
        ]
        if candidates:
            distance = min(abs(population[ref] - remaining) for ref in candidates)
            nearest = sorted(
                (ref for ref in candidates if abs(population[ref] - remaining) == distance),
                key=lambda ref: (ref.artifact_id, ref.sha256),
            )
            control = rng.choice(nearest)
            available.remove(control)
            matches.append(
                CheckpointMatch(
                    proposed=proposal,
                    control=control,
                    proposed_remaining=remaining,
                    control_remaining=population[control],
                    eligible_controls=len(nearest),
                    control_probability=1 / len(nearest),
                )
            )
        else:
            matches.append(
                CheckpointMatch(
                    proposed=proposal,
                    control=None,
                    proposed_remaining=remaining,
                    control_remaining=None,
                    eligible_controls=0,
                    control_probability=None,
                    refusal="terminal proposal"
                    if remaining <= 0
                    else "no unused matching-horizon control",
                )
            )
    return CheckpointSelectionPlan(spec=spec, matches=tuple(matches))


def persist_selection_plan(
    project: ProjectStore,
    spec: CheckpointSelectionSpec,
    *,
    created_at: datetime,
    code_revision: str,
) -> ArtifactInput:
    """Publish every match and selector rationale before any continuation calls are launched."""
    identity = spec.identity(code_revision)
    if project.artifacts.exists(identity):
        reference = artifact_input(project.artifacts.read(identity).manifest)
        saved = load_selection_plan(project, reference)
        if saved.spec != spec:
            raise ValueError("saved checkpoint selection specification differs")
        return reference
    # Publishing a data-only plan does not need credentials or a runtime file. The subsequent
    # continuation executor performs its own mandatory execution preflight.
    plan = _match(project, spec)
    inputs = {
        *spec.pool,
        spec.base.study.checkpoint,
        *(reference for proposal in spec.proposed for reference in proposal.evidence),
    }
    manifest = project.artifacts.write_json(
        artifact_id=identity,
        artifact_type="checkpoint-selection",
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=created_at,
            code_revision=code_revision,
            inputs=tuple(sorted(inputs, key=lambda row: row.artifact_id)),
        ),
        files={"plan.json": plan},
    )
    return artifact_input(manifest)


def selection_jobs(
    plan: CheckpointSelectionPlan, *, code_revision: str, plan_reference: ArtifactInput
) -> tuple[tuple[str, ContinuationJob], ...]:
    """Produce paired jobs with equal allowances and no inherited success estimates.

    Unmatched proposals remain in the plan and produce no unpaired comparison jobs. Subsequent
    inference is owned by the existing continuation executor and its per-job dispatch ledger.
    """
    jobs = []
    identity = plan.spec.identity(code_revision)
    for index, pair in enumerate(plan.matches):
        if pair.control is None:
            continue
        order = (("proposed", pair.proposed.checkpoint), ("random", pair.control))
        if index % 2:
            order = tuple(reversed(order))
        for arm, checkpoint in order:
            study = plan.spec.base.study.model_copy(
                update={
                    "checkpoint": checkpoint,
                    "predicted_success": None,
                    "critic_prediction": None,
                    "selection_plan": plan_reference,
                    "selector": f"{identity} pair {index} arm {arm}",
                }
            )
            jobs.append((arm, plan.spec.base.model_copy(update={"study": study})))
    return tuple(jobs)


def load_selection_plan(project: ProjectStore, reference: ArtifactInput) -> CheckpointSelectionPlan:
    """Reproduce every match from original immutable capsules, without runtime imports or calls."""
    store = project.artifacts
    manifest = store.read(reference.artifact_id).manifest
    if artifact_input(manifest) != reference or manifest.artifact_type != "checkpoint-selection":
        raise ValueError("checkpoint selection reference differs")
    plan = CheckpointSelectionPlan.model_validate_json(
        store.read_bytes(reference.artifact_id, "plan.json")
    )
    if plan.spec.identity(manifest.code_revision) != reference.artifact_id:
        raise ValueError("checkpoint selection identity differs")
    expected_sources = {
        *plan.spec.pool,
        plan.spec.base.study.checkpoint,
        *(ref for proposal in plan.spec.proposed for ref in proposal.evidence),
    }
    if set(manifest.inputs) != expected_sources:
        raise ValueError("checkpoint selection lost its source inventory")
    if _match(project, plan.spec) != plan:
        raise ValueError("checkpoint selection no longer reproduces its population match")
    return plan


def require_selection_binding(
    store: ArtifactStore, reference: ArtifactInput, study: "ContinuationStudySpec"
) -> None:
    """Verify an exact selected/control cell, allowing a prospective critic bound afterwards."""
    project = ProjectStore(store.paths.root, store.paths.project_id)
    plan = load_selection_plan(project, reference)
    manifest = store.read(reference.artifact_id).manifest
    actual = study.model_copy(update={"predicted_success": None, "critic_prediction": None})
    jobs = selection_jobs(plan, code_revision=manifest.code_revision, plan_reference=reference)
    if not any(job.study == actual for _, job in jobs):
        raise ValueError("continuation does not match its registered checkpoint comparison")
