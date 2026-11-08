"""Governed preparation from a completed build to family-scoped registered mining starts.

Call ``plan_bootstrap_preparation`` without credentials or a starting job, review its exact
inventory, then pass ``BootstrapApproval`` to ``run_bootstrap_preparation``. Every admitted
request uses the existing operation spend ceiling. A completed preparation replays without
approval or calls. The resulting bindings are the six-stage pipeline's ordinary bootstrap input.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    Sha256,
    sha256_bytes,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.models import ModelSnapshot
from exp.common.project import ProjectStore, artifact_input
from exp.common.traces.store import load_trace_dataset
from exp.optimize.evaluation.executable import grounded
from exp.optimize.evaluation.executable.costs import (
    evaluation_costs,
    grounded_allowance_costs,
    runtime_costs,
)
from exp.optimize.evaluation.executable.grounded import GroundedRuntimeDefinition
from exp.optimize.evaluation.executable.model_costs import CostComponent, model_cost
from exp.optimize.evaluation.rules.system_one import DecisionBoundary
from exp.optimize.evaluation.spending import (
    operation_directory,
    operation_spend_ceiling,
    save_spend_receipt,
)
from exp.optimize.evaluation.variance.bootstrap import (
    freeze_batch,
    load_outcome,
    run_bootstrap,
    task_identity,
)
from exp.optimize.evaluation.variance.bootstrap_contracts import (
    MAX_REVEAL_PROBES,
    BootstrapBinding,
    BootstrapRegistration,
    BootstrapSettings,
)
from exp.optimize.evaluation.variance.bootstrap_selection import (
    BootstrapLimits,
    SelectedBootstrap,
    select_bootstrap_families,
)
from exp.optimize.evaluation.variance.build import (
    plan_variance_build,
    run_variance_build,
    variance_build_costs,
)
from exp.optimize.evaluation.variance.build_records import VarianceBuildReport, load_exposure_plan
from exp.optimize.evaluation.variance.family_simulators import save_family_simulator
from exp.optimize.evaluation.variance.pipeline_inputs import rules_costs
from exp.optimize.evaluation.variance.pipeline_review import family_reviewer, review_costs
from exp.optimize.evaluation.variance.pipeline_spec import BuildSettings, CostLine, cost_lines
from exp.runtime.models.providers.transport import RetryPolicy
from exp.runtime.models.spend_ceiling import (
    MAXIMUM_UNPRICED_ATTEMPTS,
    install_spend_ceiling,
    operation_spend_scope,
)
from exp.simulation.mining.decisions import DecisionsTraceTagger
from exp.simulation.mining.environment_review import CompatibilityReviewer
from exp.simulation.mining.tag_cluster_names import ClusterNamer

PREPARATION_TYPE = "bootstrap-preparation"


class BootstrapPreparationSpec(ContractModel):
    """Finite settings for automatic family selection and start preparation.

    Attributes:
        build: Variance build settings, including frozen compatibility reviewer.
        bootstrap: Role, runtime and future mining-job settings, with no required start.
        limits: Finite family and task bounds enforced before bootstrap calls.
        task_ids: Explicit selected probe IDs, or empty for ranked compatible probes up to the cap.
        simulators: Saved shared simulators whose exact scoped knowledge should be reused.
        learning_traces: Per family, the member traces its rules are learned from when the
            family has more members than ``limits.maximum_family_traces``; a larger family
            without an entry learns from an evenly spaced subset of that size.
    """

    build: BuildSettings
    bootstrap: BootstrapSettings
    limits: BootstrapLimits
    task_ids: tuple[str, ...] = ()
    simulators: tuple[ArtifactInput, ...] = ()
    learning_traces: dict[str, tuple[str, ...]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _preparable(self) -> Self:
        """Require a family build and refuse an unpriced constructor or impossible selection."""
        if self.build.spec.environment_families is None:
            raise ValueError("bootstrap preparation needs build.spec.environment_families")
        if self.build.spec.construction is not None:
            raise ValueError("bootstrap preparation reuses the completed build; omit construction")
        if self.build.saved_report is not None:
            raise ValueError(
                "build.saved_report is a pipeline reuse input; omit it from bootstrap preparation"
            )
        if (
            len(set(self.task_ids)) != len(self.task_ids)
            or len(self.task_ids) > self.limits.maximum_tasks
        ):
            raise ValueError("task_ids must be distinct and fit maximum_tasks")
        if self.bootstrap.rules is None and not self.simulators:
            raise ValueError("unlearned selected families need bootstrap.rules")
        if self.bootstrap.rules is not None and self.bootstrap.rules.profile == "agentic_research":
            raise ValueError(
                "bootstrap planning needs bounded jev_memory or single_draw rules; "
                "agentic_research has no finite merged-rule count"
            )
        return self


class BootstrapPlan(ContractModel):
    """Provider-free preparation inventory bound to exact source, settings and producer.

    Attributes:
        identity: Stable operation identity, retained across resumes.
        spec_sha256: Frozen preparation settings digest.
        code_revision: Producer identity.
        build_identity: Exact variance-build registration.
        build_saved: Whether family review and scoped indexes are already saved.
        completed_tasks: Exact selected tasks with saved registrations or refusals.
        remaining_tasks: Exact pending task IDs, empty while only the task cap is known.
        task_bound: Most pending tasks, exact once scopes exist and bounded before review.
        family_bound: Most selected families needing rules, exact once scopes exist.
        costs: Known and explicitly unpriced finite physical-request components.
        lines: Human-readable multiplicities and source/selection limitations.
        accounted_usd: Existing settled or retained charges counted against resumed consent.
        accounted_unpriced_attempts: Existing unpriced requests counted against resumed consent.
        complete: Every selected task is already saved.
    """

    identity: str
    spec_sha256: Sha256
    code_revision: str
    build_identity: str
    build_saved: bool
    completed_tasks: tuple[str, ...] = ()
    remaining_tasks: tuple[str, ...] = ()
    task_bound: int = Field(ge=0)
    family_bound: int = Field(ge=0)
    costs: tuple[CostLine, ...]
    lines: tuple[str, ...]
    accounted_usd: float = Field(default=0, ge=0)
    accounted_unpriced_attempts: int = Field(default=0, ge=0)
    complete: bool = False

    @property
    def digest(self) -> Sha256:
        """Return the digest approval must name exactly."""
        return sha256_json(self)

    @property
    def known_usd(self) -> float:
        """Sum existing accounted dollars and remaining priced components."""
        return self.accounted_usd + sum(row.usd for row in self.costs if row.usd is not None)

    @property
    def unpriced_attempts(self) -> int:
        """Return the finite allowance required by all unpriced components."""
        return self.accounted_unpriced_attempts + sum(
            row.attempts for row in self.costs if row.usd is None
        )


class BootstrapApproval(ContractModel):
    """Explicit consent to one preparation plan, enforced by the shared operation ceiling.

    Attributes:
        plan_sha256: Exact displayed plan digest.
        ceiling_usd: Whole-operation dollar ceiling for every priced role.
        unpriced_attempts: Explicit whole-operation allowance for unpriced requests.
    """

    plan_sha256: Sha256
    ceiling_usd: float = Field(ge=0)
    unpriced_attempts: int = Field(default=0, ge=0, le=MAXIMUM_UNPRICED_ATTEMPTS)


class BootstrapPreparationReport(ContractModel):
    """Prepared starts and reusable family definitions, including diagnostic task refusals.

    Attributes:
        identity: Preparation operation identity.
        plan: Exact plan authorized before this execution.
        build_report: Saved family-scoped variance build.
        batches: Frozen bootstrap batch manifests.
        bindings: Every successfully registered task's pipeline binding.
        simulators: Saved family definitions carrying the frozen rules and runtime.
        failed_tasks: Retryable failures; their preparation is not marked complete.
        refused_tasks: Saved diagnostic refusals, never silently treated as runnable jobs.
    """

    identity: str
    plan: BootstrapPlan
    build_report: ArtifactInput
    batches: tuple[ArtifactInput, ...]
    bindings: tuple[BootstrapBinding, ...]
    simulators: tuple[ArtifactInput, ...] = ()
    failed_tasks: tuple[str, ...] = ()
    refused_tasks: tuple[str, ...] = ()


@dataclass(frozen=True)
class BootstrapDependencies:
    """Optional offline or admitted collaborators passed to existing owners.

    Attributes:
        tagger: Variance build tagger, or None for the admitted default.
        reviewer: Compatibility reviewer matching the frozen settings, or None for the default.
        namer: Cluster naming client when naming is registered.
        decisions: Native rule-memory decision client, or None for its admitted default.
    """

    tagger: DecisionsTraceTagger | None = None
    reviewer: CompatibilityReviewer | None = None
    namer: ClusterNamer | None = None
    decisions: DecisionsTraceTagger | DecisionBoundary | None = None


def _identity(spec: BootstrapPreparationSpec, build_identity: str, revision: str) -> str:
    """Bind an operation to exact preparation inputs and the variance-build identity."""
    return stable_id(
        PREPARATION_TYPE, {"spec": sha256_json(spec), "build": build_identity, "code": revision}
    )


def _saved_build(project: ProjectStore, identity: str) -> VarianceBuildReport | None:
    """Read the build owner's immutable report, or None before family review."""
    key = identity + "-result"
    if not project.artifacts.exists(key):
        return None
    return VarianceBuildReport.model_validate_json(project.artifacts.read_bytes(key, "report.json"))


def _role_costs(
    project: ProjectStore, settings: BootstrapSettings, count: int
) -> tuple[CostComponent, ...]:
    """Price every bounded author, review, research and reveal-check draw for pending tasks."""
    rounds = settings.research_rounds + 1
    calls = (
        ("bootstrap author", settings.author, rounds * settings.author_profile.budget.model_calls),
        (
            "bootstrap reviewer",
            settings.role("reviewer"),
            rounds * len(settings.review_variants) * settings.review_profile.budget.model_calls,
        ),
        (
            "bootstrap researcher",
            settings.role("researcher"),
            settings.research_rounds
            * settings.research_items
            * settings.research_profile.budget.model_calls,
        ),
        (
            "bootstrap reveal review",
            settings.role("researcher"),
            len(settings.reveal_variants) * settings.research_profile.budget.model_calls,
        ),
    )
    return tuple(
        CostComponent(
            f"{name}: {row.name}",
            row.attempts * draws * count,
            None if row.usd is None else row.usd * draws * count,
        )
        for name, author, draws in calls
        for row in evaluation_costs(project, author)
    )


def _validation_costs(
    project: ProjectStore,
    settings: BootstrapSettings,
    count: int,
    selected: SelectedBootstrap | None = None,
) -> tuple[CostComponent, ...]:
    """Price the maximum fresh runtime lifetimes without inventing a task or checkpoint."""
    runtime = None if selected is None else selected.spec.families[0].family.runtime
    if runtime is not None:
        base = runtime_costs(
            project, settings.runtime.runtime_sha256, runtime.model_dump(mode="json")
        )
    else:
        template = settings.runtime if selected is None else selected.spec.runtime
        memory = None if selected is None else selected.spec.families[0].family.rule_memory
        recall = ()
        if memory is not None:
            recall = (memory.recall.jev_alias,)
        elif settings.rules is not None and settings.rules.memory is not None:
            recall = (template.jev_alias,)
        base = grounded_allowance_costs(
            project,
            template.world_model,
            attempts=template.maximum_physical_attempts,
            output_tokens=template.maximum_output_tokens,
            decision_aliases=recall,
        )
    # Capture may open a runtime; reconstructed validation opens one; authored validation opens
    # one per reveal. Counting all of them is conservative even though provenances are exclusive.
    lifetimes = (MAX_REVEAL_PROBES + 2) * count
    return tuple(
        CostComponent(
            f"bootstrap validation: {row.name}",
            row.attempts * lifetimes,
            None if row.usd is None else row.usd * lifetimes,
        )
        for row in base
    )


def plan_bootstrap_preparation(
    project: ProjectStore, spec: BootstrapPreparationSpec, *, code_revision: str
) -> BootstrapPlan:
    """Price selected completed-build preparation before credentials or a mining start exists."""
    template = spec.bootstrap.runtime
    maintained = sha256_bytes(Path(grounded.__file__).read_bytes())
    if (
        template.runtime_sha256 != maintained
        or sha256_bytes(template.runtime_file.read_bytes()) != maintained
    ):
        raise ValueError("bootstrap planning requires the maintained grounded.py runtime file")
    reviewer = spec.build.reviewer
    assert reviewer is not None
    build = plan_variance_build(
        project,
        spec.build.spec,
        code_revision=code_revision,
        reviewer_model=reviewer.model,
        reviewer_configuration_sha256=sha256_json(reviewer),
    )
    identity = _identity(spec, build.identity, code_revision)
    saved = _saved_build(project, build.identity)
    if saved is None and spec.simulators:
        raise ValueError("reusing frozen simulators requires their saved variance build first")
    components = list(variance_build_costs(project, build, spec.build.spec.decision_alias))
    components.extend(review_costs(project, spec.build, build))
    naming = spec.build.spec.cluster_naming
    if not build.saved and naming is not None:
        assert spec.build.naming_alias is not None
        components.append(
            model_cost(
                project,
                spec.build.naming_alias,
                build.naming_calls * RetryPolicy().maximum_attempts,
                naming.maximum_output_tokens,
            )
        )
    selected = (
        ()
        if saved is None
        else select_bootstrap_families(
            project,
            saved,
            spec.bootstrap,
            spec.limits,
            task_ids=spec.task_ids,
            simulators=spec.simulators,
            learning_traces=spec.learning_traces,
        )
    )
    complete: list[str] = []
    pending: list[str] = []
    task_count = min(
        spec.limits.maximum_tasks, len(spec.task_ids) or build.roles["mining_probes"].tasks
    )
    families = min(spec.limits.maximum_families, task_count) if saved is None else 0
    lines = [
        f"Variance build {build.identity}: {'saved' if build.saved else 'pending'}; "
        f"{build.decision_calls} tagging and {build.review_calls} compatibility-review calls."
    ]
    if saved is None:
        if spec.bootstrap.rules is not None:
            rows, explanation = rules_costs(
                project,
                spec.bootstrap.rules,
                spec.limits.maximum_family_traces,
                spec.limits.maximum_family_tools,
            )
            components.extend(row.scaled(families) for row in rows)
            lines.append(
                f"At most {families} selected families, each {explanation}; "
                "actual scopes must fit the registered bounds."
            )
        components.extend(_role_costs(project, spec.bootstrap, task_count))
        components.extend(_validation_costs(project, spec.bootstrap, task_count))
    else:
        task_count = 0
        for family in selected:
            batch = family.spec
            row = batch.families[0]
            missing = [
                task
                for task in row.tasks
                if load_outcome(project, task_identity(batch, row, task, code_revision)) is None
            ]
            pending.extend(task.task.task_id for task in missing)
            complete.extend(task.task.task_id for task in row.tasks if task not in missing)
            task_count += len(missing)
            if missing and not row.family.frozen:
                assert batch.rules is not None
                families += 1
                rows, explanation = rules_costs(
                    project,
                    batch.rules,
                    len(row.family.learning_traces),
                    len(row.family.tool_contracts),
                )
                components.extend(rows)
                lines.append(
                    f"Family {row.family.family_id}: {explanation}; learned once from "
                    f"{len(row.family.learning_traces)} of {len(family.scope.members)} member "
                    f"traces for {len(missing)} pending tasks."
                )
            components.extend(_role_costs(project, batch, len(missing)))
            components.extend(_validation_costs(project, batch, len(missing), family))
    settings = spec.bootstrap
    lines.append(
        f"At most {task_count} pending tasks: each {settings.research_rounds + 1} author rounds, "
        f"{len(settings.review_variants)} reviewers per round, "
        f"{settings.research_rounds * settings.research_items} research items, "
        f"{len(settings.reveal_variants)} reveal reviewers, and "
        f"{MAX_REVEAL_PROBES + 2} bounded runtime lifetimes. "
        "No actor inference runs during preparation."
    )
    accounted_usd, accounted_unpriced = 0.0, 0
    if (operation_directory(project, identity) / "ledger.sqlite").exists():
        spent = operation_spend_ceiling(project, identity).summary()
        accounted_usd, accounted_unpriced = spent.accounted_usd, spent.unpriced_attempts
    return BootstrapPlan(
        accounted_usd=accounted_usd,
        accounted_unpriced_attempts=accounted_unpriced,
        identity=identity,
        spec_sha256=sha256_json(spec),
        code_revision=code_revision,
        build_identity=build.identity,
        build_saved=build.saved,
        completed_tasks=tuple(complete),
        remaining_tasks=tuple(pending),
        task_bound=task_count,
        family_bound=families,
        costs=cost_lines(tuple(row for row in components if row.attempts)),
        lines=tuple(lines),
        complete=saved is not None and task_count == 0,
    )


def run_bootstrap_preparation(
    project: ProjectStore,
    spec: BootstrapPreparationSpec,
    *,
    code_revision: str,
    approval: BootstrapApproval | None = None,
    dependencies: BootstrapDependencies | None = None,
) -> BootstrapPreparationReport:
    """Prepare or replay exact family starts under one explicitly approved operation ceiling.

    Raises:
        ValueError: Approval is absent or mismatched, allowances are insufficient, or injected
            collaborators differ from the planned identities.
    """
    dependencies = dependencies or BootstrapDependencies()
    spec = BootstrapPreparationSpec.model_validate_json(spec.model_dump_json())
    plan = plan_bootstrap_preparation(project, spec, code_revision=code_revision)
    result_id = plan.identity + "-result"
    if project.artifacts.exists(result_id):
        return BootstrapPreparationReport.model_validate_json(
            project.artifacts.read_bytes(result_id, "report.json")
        )
    if not plan.complete:
        if approval is None or approval.plan_sha256 != plan.digest:
            raise ValueError(
                "bootstrap preparation needs explicit approval of its exact current plan"
            )
        if (
            approval.ceiling_usd < plan.known_usd
            or approval.unpriced_attempts < plan.unpriced_attempts
        ):
            raise ValueError("bootstrap approval does not cover its priced and unpriced inventory")
    if dependencies.tagger is not None:
        injected = ModelSnapshot.model_validate(dependencies.tagger.model_identity)
        expected = plan_variance_build(
            project,
            spec.build.spec,
            code_revision=code_revision,
            reviewer_model=spec.build.reviewer.model if spec.build.reviewer is not None else None,
            reviewer_configuration_sha256=(
                None if spec.build.reviewer is None else sha256_json(spec.build.reviewer)
            ),
        ).decision_model
        if injected != expected:
            raise ValueError("injected decision tagger differs from the approved model snapshot")
    reviewer = dependencies.reviewer or family_reviewer(
        project, spec.build, code_revision=code_revision
    )
    assert spec.build.reviewer is not None
    if reviewer.model != spec.build.reviewer.model or reviewer.configuration_sha256 != sha256_json(
        spec.build.reviewer
    ):
        raise ValueError("injected family reviewer differs from the approved model snapshot")
    if spec.build.spec.cluster_naming is not None and dependencies.namer is None:
        raise ValueError("the registered cluster naming needs an admitted namer before execution")
    with (
        CoordinationLock(
            project.artifacts.project_directory / (plan.identity + ".lock"), timeout=0
        ),
        operation_spend_scope(),
    ):
        if project.artifacts.exists(result_id):
            return BootstrapPreparationReport.model_validate_json(
                project.artifacts.read_bytes(result_id, "report.json")
            )
        current = plan_bootstrap_preparation(project, spec, code_revision=code_revision)
        if current.digest != plan.digest:
            raise ValueError(
                "bootstrap preparation changed during admission; review its current plan"
            )
        ceiling = operation_spend_ceiling(project, plan.identity)
        if approval is not None:
            ceiling.authorize(
                "preparation",
                ceiling_usd=approval.ceiling_usd,
                unpriced_attempts=approval.unpriced_attempts,
            )
        install_spend_ceiling(ceiling)
        try:
            result = _run(project, spec, plan, dependencies, reviewer, code_revision)
            if not result.failed_tasks:
                refs = (result.build_report, *result.batches, *result.simulators)
                project.artifacts.write_json(
                    artifact_id=result_id,
                    artifact_type=PREPARATION_TYPE,
                    envelope=ArtifactEnvelope(
                        schema_version=1,
                        created_at=datetime.now(UTC),
                        code_revision=code_revision,
                        inputs=tuple(sorted(refs, key=lambda ref: ref.artifact_id)),
                    ),
                    files={"report.json": result},
                )
            return result
        finally:
            save_spend_receipt(project, plan.identity, ceiling, "bootstrap-preparation")


def _run(
    project: ProjectStore,
    spec: BootstrapPreparationSpec,
    plan: BootstrapPlan,
    dependencies: BootstrapDependencies,
    reviewer: CompatibilityReviewer,
    revision: str,
) -> BootstrapPreparationReport:
    """Execute existing build and bootstrap owners inside the already installed ceiling."""
    report = run_variance_build(
        project,
        spec.build.spec,
        code_revision=revision,
        tagger=dependencies.tagger,
        reviewer=reviewer,
        namer=dependencies.namer,
    )
    if report.identity != plan.build_identity:
        raise ValueError("executed variance build differs from its approved identity")
    selected = select_bootstrap_families(
        project,
        report,
        spec.bootstrap,
        spec.limits,
        task_ids=spec.task_ids,
        simulators=spec.simulators,
        learning_traces=spec.learning_traces,
    )
    exposure = load_exposure_plan(project, report.exposure)
    traces = {
        row.trace_id: row
        for row in load_trace_dataset(
            project.artifacts, report.build.trace_dataset.artifact_id
        ).traces
    }
    batches: list[ArtifactInput] = []
    bindings: list[BootstrapBinding] = []
    simulators: list[ArtifactInput] = []
    failed: list[str] = []
    refused: list[str] = []
    for family in selected:
        manifest = run_bootstrap(
            project,
            family.spec,
            exposure=exposure,
            traces=traces,
            code_revision=revision,
            decisions=dependencies.decisions,
        )
        batch = freeze_batch(project, manifest)
        batches.append(batch)
        learned = None
        for entry in manifest.entries:
            if entry.status == "failed":
                failed.append(entry.task_id)
            elif entry.status == "refused":
                refused.append(entry.task_id)
            else:
                bindings.append(BootstrapBinding(batch=batch, task_id=entry.task_id))
                outcome = load_outcome(project, entry.identity)
                assert isinstance(outcome, BootstrapRegistration)
                learned = outcome
        if learned is not None:
            simulator = family.simulator.model_copy(
                update={
                    "rule_book": learned.rulebook,
                    "rule_memory": learned.rule_memory,
                    "rules_report": learned.rules_report,
                    "runtime": GroundedRuntimeDefinition.model_validate(learned.job.configuration),
                }
            )
            simulators.append(
                save_family_simulator(project.artifacts, simulator, code_revision=revision)
            )
    return BootstrapPreparationReport(
        identity=plan.identity,
        plan=plan,
        build_report=artifact_input(project.artifacts.read(report.identity + "-result").manifest),
        batches=tuple(batches),
        bindings=tuple(bindings),
        simulators=tuple(simulators),
        failed_tasks=tuple(failed),
        refused_tasks=tuple(refused),
    )
