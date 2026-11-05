"""Exact native prelude costs and mandatory remaining council work."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactInput,
    ContractModel,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job import (
    ContinuationJob,
    continuation_result_reference,
    preflight_continuation_job,
    saved_continuation_job,
)
from exp.optimize.evaluation.executable.costs import evaluation_costs, job_costs
from exp.optimize.evaluation.executable.model_costs import CostComponent, decision_cost
from exp.optimize.evaluation.variance.bootstrap_source import current_bootstrap_execution_source
from exp.optimize.evaluation.variance.council_prelude import (
    QualifiedCouncilPrelude,
    _job,
    _resolved,
    _rubric_proof,
    prelude_banks,
    prelude_sources,
)
from exp.optimize.evaluation.variance.council_prelude_hook import CouncilPreludeCheckpoint
from exp.optimize.evaluation.variance.council_preparation import (
    _pinned,
)
from exp.optimize.evaluation.variance.episode_evidence import run_task_sha256, run_world_sha256
from exp.optimize.evaluation.variance.judged import (
    JUDGED_LABELS_PROTOCOL,
    JUDGED_STUDY_TYPE,
    JudgedStudy,
)
from exp.optimize.evaluation.variance.pipeline_behaviors import behavior_costs
from exp.optimize.evaluation.variance.pipeline_errors import tier_error_bank
from exp.optimize.evaluation.variance.pipeline_inputs import rubric_draws
from exp.optimize.evaluation.variance.pipeline_plan import CostLine
from exp.optimize.evaluation.variance.pipeline_spec import (
    cost_lines,
)
from exp.optimize.evaluation.variance.pipeline_stages import (
    judge_costs,
    mining_profile,
)
from exp.runtime.models.providers.transport import RetryPolicy
from exp.simulation.continuations import load_continuation_checkpoint
from exp.simulation.mining.error_tags import bank_for_view, load_error_steps
from exp.simulation.mining.traceml import load_traceml_labels, traceml_bank
from exp.simulation.mining.traceml_moves import windows as move_windows


class CouncilLogicalFloor(ContractModel):
    """Mandatory remaining logical requests after the current prelude wave.

    Saved native group owners are deducted. Each pending actor attempt, judgment, selector and
    critic requires at least one logical model request. Suffixes retain at least one selected
    point per model. Tagging uses actual saved step counts; unknown future steps contribute zero
    to this lower bound while their finite full upper bounds remain in the phase inventory.
    Internal response-cache replay may reduce fresh HTTP work, never the requested logical work.

    Attributes:
        groups: Original actor attempts whose native group is not committed.
        group_judgments: Original full-episode labels not committed by their native owner.
        selectors: All fixed point-selector draws, never removed to fit headroom.
        critics: All fixed critic draws.
        suffixes: Minimum mandatory continuations across the five actor studies.
        suffix_judgments: Full-episode labels of those continuations.
        rubric: Required rubric protocol draws outside the current rubric wave.
        tagging: Required native tagging and move draws over already saved actor steps.
    """

    groups: int = Field(ge=0)
    group_judgments: int = Field(ge=0)
    selectors: int = Field(ge=0)
    critics: int = Field(ge=0)
    suffixes: int = Field(ge=0)
    suffix_judgments: int = Field(ge=0)
    rubric: int = Field(ge=0)
    tagging: int = Field(ge=0)

    @property
    def attempts(self) -> int:
        """Return the conservative mandatory remaining logical request floor."""
        return sum(
            (
                self.groups,
                self.group_judgments,
                self.selectors,
                self.critics,
                self.suffixes,
                self.suffix_judgments,
                self.rubric,
                self.tagging,
            )
        )


class CouncilPreludeInventory(ContractModel):
    """Native current-wave upper bound and mandatory future work, never a free callback.

    Attributes:
        protocol: Exact native inventory protocol.
        contract_sha256: Complete frozen prelude and future-study identity.
        checkpoint: Actual native job, group and judging inputs being admitted.
        phase: Actual native phase being admitted.
        actor_alias: Exact current tier, or None for shared rubric authoring.
        jobs: Actual jobs potentially dispatched in this current wave.
        costs: Native physical upper bounds including retries, world and embeddings.
        remaining: Mandatory logical future floor under the unchanged full study.
        replayed: Exact complete native owners omitted from current paid work.
        authority: A quote requires an original-ledger envelope before any dispatch.
    """

    protocol: Literal["council-native-prelude-inventory-v1"] = "council-native-prelude-inventory-v1"
    contract_sha256: Sha256
    checkpoint: CouncilPreludeCheckpoint
    phase: str
    actor_alias: str | None
    jobs: tuple[ContinuationJob, ...]
    costs: tuple[CostLine, ...]
    remaining: CouncilLogicalFloor
    replayed: tuple[ArtifactInput, ...] = ()
    authority: Literal["inventory_only"] = "inventory_only"

    @property
    def attempts(self) -> int:
        """Return the full current physical-request upper bound."""
        return sum(row.attempts for row in self.costs)

    @property
    def unpriced_attempts(self) -> int:
        """Retain every unknown-rate current physical liability without treating it as free."""
        return sum(row.attempts for row in self.costs if row.usd is None)

    @property
    def known_usd(self) -> float:
        """Keep known tariff liabilities separate from unpriced physical requests."""
        return sum(row.usd for row in self.costs if row.usd is not None)


def _judgments_pending(
    project: ProjectStore,
    job: ContinuationJob,
    checkpoint: CouncilPreludeCheckpoint,
    revision: str,
) -> tuple[int, ArtifactInput | None]:
    """Deduct only a complete native judged owner for the actual supplied frozen judge."""
    judge = checkpoint.judge
    if judge is None or saved_continuation_job(project, job, code_revision=revision) is None:
        return job.study.repeats, None
    group = continuation_result_reference(project, job, code_revision=revision)
    identity = stable_id(
        JUDGED_STUDY_TYPE,
        {
            "protocol": JUDGED_LABELS_PROTOCOL,
            "judge": judge.identity,
            "study": group.model_dump(mode="json"),
        },
    )
    if not project.artifacts.exists(identity):
        return job.study.repeats, None
    stored = project.artifacts.read(identity)
    result = JudgedStudy.model_validate_json(stored.payloads["judged.json"])
    if (
        stored.manifest.artifact_type != JUDGED_STUDY_TYPE
        or result.study != group
        or result.judge != judge.identity
        or len(result.branches) != job.study.repeats
    ):
        raise ValueError("prelude group judging differs from its exact complete native owner")
    return 0, artifact_input(stored.manifest)


def _completed_labels(
    project: ProjectStore,
    qualified: QualifiedCouncilPrelude,
    checkpoint: CouncilPreludeCheckpoint,
) -> set[str]:
    """Verify exact native label owners before deducting any model's remaining tagging work."""
    spec = qualified.contract.pipeline
    if not checkpoint.completed_labels:
        return set()
    if not spec.traceml.enabled:
        raise ValueError("unrequested TraceML owners cannot reduce native remaining work")
    bank = traceml_bank(spec.traceml.bank_root)
    groups = {
        continuation_result_reference(project, job, code_revision=qualified.producer_revision): job
        for job in qualified.jobs
        if saved_continuation_job(project, job, code_revision=qualified.producer_revision)
        is not None
    }
    completed: set[str] = set()
    for reference in checkpoint.completed_labels:
        labels = load_traceml_labels(project.artifacts, reference)
        manifest = project.artifacts.read(reference.artifact_id).manifest
        matches = tuple(group for group in groups if manifest.inputs == (group,))
        if len(matches) != 1 or manifest.code_revision != qualified.producer_revision:
            raise ValueError("completed TraceML labels lost their exact native group or producer")
        job = groups[matches[0]]
        if (
            labels.bank_id != bank.bank_id
            or labels.bank_version != bank.version
            or labels.question_ids != bank.question_ids()
            or len(labels.traces) != job.study.repeats
            or labels.move_window != spec.traceml.move_window
            or labels.move_variants != spec.traceml.move_variants
        ):
            raise ValueError("completed TraceML labels changed frozen bank or mandatory controls")
        completed.add(job.actor_alias)
    return completed


def _completed_errors(
    project: ProjectStore,
    qualified: QualifiedCouncilPrelude,
    checkpoint: CouncilPreludeCheckpoint,
) -> set[str]:
    """Deduct error tagging only from exact native step-error and point owners."""
    bank = tier_error_bank(qualified.contract.pipeline)
    if not checkpoint.completed_error_points:
        return set()
    if bank is None:
        raise ValueError("unrequested error-point owners cannot reduce native remaining work")
    groups: dict[ArtifactInput, ContinuationJob] = {}
    for job in qualified.jobs:
        if (
            saved_continuation_job(project, job, code_revision=qualified.producer_revision)
            is not None
        ):
            group = continuation_result_reference(
                project, job, code_revision=qualified.producer_revision
            )
            groups[group] = job
    done: set[str] = set()
    for reference in checkpoint.completed_error_points:
        _pinned(project, reference, "error-points")
        points = project.artifacts.read(reference.artifact_id)
        if len(points.manifest.inputs) != 1:
            raise ValueError("native error points lost their exact step-error source")
        source = points.manifest.inputs[0]
        steps = load_error_steps(project.artifacts, source)
        manifest = project.artifacts.read(source.artifact_id).manifest
        matches = tuple(group for group in groups if manifest.inputs == (group,))
        if (
            len(matches) != 1
            or manifest.code_revision != qualified.producer_revision
            or points.manifest.code_revision != qualified.producer_revision
            or steps.bank_sha256 != sha256_json(bank_for_view(bank, "step"))
        ):
            raise ValueError("native error points changed their group, producer or bank")
        job = groups[matches[0]]
        if len(steps.traces) != job.study.repeats:
            raise ValueError("native error points cannot drop a mandatory actor attempt")
        done.add(job.actor_alias)
    return done


def remaining_council_floor(
    project: ProjectStore,
    qualified: QualifiedCouncilPrelude,
    checkpoint: CouncilPreludeCheckpoint,
) -> CouncilLogicalFloor:
    """Retain every requested model's pending native attempts, grading and mandatory role draws."""
    spec = qualified.contract.pipeline
    mining = spec.mining
    completed_labels = _completed_labels(project, qualified, checkpoint)
    completed_errors = _completed_errors(project, qualified, checkpoint)
    if checkpoint.completed_rubric is not None:
        _rubric_proof(project, qualified, checkpoint.completed_rubric)
    bank = traceml_bank(spec.traceml.bank_root) if spec.traceml.enabled else None
    errors = tier_error_bank(spec) if spec.traceml.enabled else None
    groups = judgments = tagging = 0
    for job in qualified.jobs:
        current = job.actor_alias == checkpoint.actor_alias
        current_labels = current and checkpoint.phase == "traceml_labels"
        saved = saved_continuation_job(project, job, code_revision=qualified.producer_revision)
        if saved is None and not (
            current and checkpoint.phase in ("traceml_group", "behavior_reference")
        ):
            groups += job.study.repeats
        pending, _ = _judgments_pending(project, job, checkpoint, qualified.producer_revision)
        if not (current and checkpoint.phase == "behavior_reference"):
            judgments += pending
        counts: list[int] = []
        if saved is not None:
            for branch in saved.branches:
                rows = branch.outcome.get("steps")
                counts.append(len(rows) if isinstance(rows, list) else 0)
        if bank is not None and not current_labels and job.actor_alias not in completed_labels:
            tagging += sum(counts) * math.ceil(len(bank.questions) / 32)
            tagging += sum(
                len(move_windows(count, spec.traceml.move_window)) for count in counts
            ) * len(spec.traceml.move_variants)
        if errors is not None and not current_labels and job.actor_alias not in completed_errors:
            tagging += job.study.repeats * math.ceil(
                len(bank_for_view(errors, "observations").questions) / 32
            )
            tagging += sum(counts) * math.ceil(len(bank_for_view(errors, "step").questions) / 32)
    profile = spec.authoring.rubric_profile
    rubric = 0
    if (
        mining.rubric is None
        and checkpoint.phase != "mining_rubric"
        and checkpoint.judge is None
        and checkpoint.completed_rubric is None
    ):
        rubric = len(profile.variants) * (4 if profile.agentic else 1)
        reconciled = int(len(profile.variants) > 1)
        rubric += reconciled
        if profile.critic is not None:
            reviewed = len(profile.variants) * (2 if profile.agentic else 1) + reconciled
            rubric += reviewed * len(profile.critic.variants)
    suffixes = len(qualified.jobs) * mining.continuations
    return CouncilLogicalFloor(
        groups=groups,
        group_judgments=judgments,
        selectors=len(qualified.jobs) * mining.selector_draws,
        critics=len(qualified.jobs) * mining.critic_draws,
        suffixes=suffixes,
        suffix_judgments=suffixes,
        rubric=rubric,
        tagging=tagging,
    )


def prelude_inventory(
    project: ProjectStore,
    qualified: QualifiedCouncilPrelude,
    checkpoint: CouncilPreludeCheckpoint,
) -> CouncilPreludeInventory:
    """Price the actual current prelude and retain every mandatory future model's work."""
    if (
        qualified.contract.sources != prelude_sources()
        or qualified.contract.banks != prelude_banks(qualified.contract.pipeline)
        or qualified.contract.executor_source_sha256
        != current_bootstrap_execution_source(qualified.producer_revision).source_sha256
    ):
        raise ValueError("native prelude execution sources changed after qualification")
    preflight_continuation_job(project, checkpoint.job)
    spec = _resolved(project, qualified)
    phase, alias = checkpoint.phase, checkpoint.actor_alias
    costs: list[CostComponent] = []
    replayed: list[ArtifactInput] = []
    jobs: tuple[ContinuationJob, ...] = ()
    if phase != "behavior_reference" and checkpoint.judge is not None:
        raise ValueError("only native behavior grading may supply an episode judge")
    if phase.startswith("traceml_") and not spec.traceml.enabled:
        raise ValueError("TraceML was not requested in the frozen native pipeline")
    if phase == "mining_rubric":
        if alias is not None or checkpoint.job not in (None, spec.mining.registered):
            raise ValueError("shared mining rubric cannot substitute another native task")
        if spec.mining.rubric is not None:
            _pinned(project, spec.mining.rubric, "episode-rubric")
            replayed.append(spec.mining.rubric)
        else:
            costs.extend(
                row.scaled(rubric_draws(spec.authoring.rubric_profile))
                for row in evaluation_costs(project, spec.authoring.rubric_author)
            )
    else:
        job = _job(qualified, alias)
        if checkpoint.job != job:
            raise ValueError("prelude checkpoint changed its exact frozen native group job")
        one = spec.model_copy(
            update={"mining": spec.mining.model_copy(update={"actors": (alias,)})}
        )
        if phase in ("traceml_group", "behavior_reference"):
            if (
                saved_continuation_job(project, job, code_revision=qualified.producer_revision)
                is None
            ):
                jobs = (job,)
                costs.extend(job_costs(project, job))
            else:
                replayed.append(
                    continuation_result_reference(
                        project,
                        job,
                        code_revision=qualified.producer_revision,
                    )
                )
        if phase == "traceml_labels":
            if not spec.traceml.enabled:
                raise ValueError("TraceML labels were not requested in the frozen pipeline")
            expected = continuation_result_reference(
                project,
                job,
                code_revision=qualified.producer_revision,
            )
            if checkpoint.group != expected:
                raise ValueError("TraceML labels require the exact complete native group")
            saved = saved_continuation_job(project, job, code_revision=qualified.producer_revision)
            assert saved is not None
            if len(saved.branches) != job.study.repeats:
                raise ValueError("native group cannot drop any mandatory actor attempt")
            counts: list[int] = []
            for branch in saved.branches:
                rows = branch.outcome.get("steps")
                count = len(rows) if isinstance(rows, list) else 0
                if count > job.maximum_model_calls * spec.traceml.steps_per_call:
                    raise ValueError("native group exceeds its frozen step-per-call bound")
                counts.append(count)
            bank = traceml_bank(spec.traceml.bank_root)
            calls = sum(counts) * math.ceil(len(bank.questions) / 32)
            costs.append(
                decision_cost(
                    project,
                    spec.traceml.decision_alias,
                    calls * RetryPolicy().maximum_attempts,
                )
            )
            judge = spec.traceml.move_judge
            assert judge is not None
            windows = sum(len(move_windows(count, spec.traceml.move_window)) for count in counts)
            draws = windows * len(spec.traceml.move_variants) * (1 + spec.traceml.move_repairs)
            costs.extend(row.scaled(draws) for row in evaluation_costs(project, judge))
            errors = tier_error_bank(spec)
            if errors is not None:
                calls = len(counts) * math.ceil(
                    len(bank_for_view(errors, "observations").questions) / 32
                )
                calls += sum(counts) * math.ceil(len(bank_for_view(errors, "step").questions) / 32)
                costs.append(
                    decision_cost(
                        project,
                        spec.traceml.decision_alias,
                        calls * RetryPolicy().maximum_attempts,
                    )
                )
        elif phase == "behavior_reference":
            if spec.mining.behaviors is None or checkpoint.judge is None:
                raise ValueError("behavior prelude requires its real configured judge and controls")
            judge = checkpoint.judge
            if judge.settings != spec.mining.judge or judge.definition.profile != mining_profile(
                spec
            ):
                raise ValueError("behavior prelude changed the frozen native episode-judge profile")
            if spec.mining.rubric is not None and judge.definition.rubric != spec.mining.rubric:
                raise ValueError("behavior prelude substituted the registered mining rubric")
            _rubric_proof(project, qualified, judge.definition.rubric)
            start = load_continuation_checkpoint(project.artifacts, job.study.checkpoint)
            if judge.definition.task_sha256 != run_task_sha256(
                start
            ) or judge.definition.world_sha256 != run_world_sha256(job.configuration):
                raise ValueError("behavior prelude judge binds another native task or world")
            pending, owner = _judgments_pending(
                project,
                job,
                checkpoint,
                qualified.producer_revision,
            )
            if owner is not None:
                replayed.append(owner)
            costs.extend(judge_costs(project, spec, pending))
            costs.extend(behavior_costs(project, one))
        elif phase != "traceml_group":
            raise ValueError("unsupported native council prelude phase")
    return CouncilPreludeInventory(
        contract_sha256=sha256_json(qualified.contract),
        checkpoint=checkpoint,
        phase=phase,
        actor_alias=alias,
        jobs=jobs,
        costs=cost_lines(tuple(costs)),
        remaining=remaining_council_floor(project, qualified, checkpoint),
        replayed=tuple(sorted(set(replayed), key=lambda ref: ref.artifact_id)),
    )
