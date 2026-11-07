"""The complete decision-variance pipeline, resumable stage by stage from immutable artifacts.

:func:`run_decision_variance` runs the six stages of
:mod:`~exp.optimize.evaluation.variance.pipeline_spec` in order:

1. ``variance-build``: the exposure plan before tagging, task and observation tags, the scoped
   world-learning index and world model, and the tag retrieval bindings.
2. ``rules``: the learned rule book over world-learning traces, its boundary workloads executed
   against the bound world, then the rule book frozen and bound into the mining job before any
   probe runs.
3. ``traceml``: each tier's group of mining-probe attempts labelled step by step and registered
   as a reference for that tier's selectors and critics (optional, on by default).
4. ``error-mining``: the mining-probe task's rubric (registered, or authored over a whole-task
   packet), then one actor-critic run per actor tier, never pooled, each labelled by an episode
   judge whose profile is fixed before any family rubric, its selectors and critics searching
   the scoped world-learning index; optionally driven by the training schedule.
5. ``families``: per screened point, the coalition evidence, one authoring packet with observed
   replay episodes, independent agentic world and rubric authors, the compatibility review, the
   world freeze with its challenges, the judge definitions, the family's contrasts from frozen
   starts (options forced by the decision-state author's calls unless the spec gives actions),
   judged confirmation, the rubric-revision loop when judging reports rubric problems, and
   admission.
6. ``records``: training records of admitted families, selector records, executable
   qualification and Harbor exports of admitted variants, and the optional cross-talk probe.

**Bootstrap.** ``mining.bootstrap`` may name a task's registered start in a frozen
start-bootstrap batch instead of a hand-registered ``mining.job``; the entry resolves it to the
registered job before planning, so every stage reads that job.

**Resume.** Every stage saves one ``decision-variance-stage`` record naming its outputs. A rerun
replays saved records with no call and continues the first unsaved stage, whose owners in turn
replay every draw, study and judgment they already saved, so an interrupted stage resumes where
it stopped. A stage's identity binds its spec slice and every earlier stage's identity, so a
changed field re-keys only that stage and the ones after it.

**Spend.** :func:`plan_decision_variance` prices every unsaved stage up front as an upper bound,
without credentials or calls. A run with unsaved stages needs one :class:`PipelineApproval` of
that exact plan; the caller obtains it once through its spend-consent path. Before each unsaved
stage the run checks that the spend measured so far plus the stage's bound fits the approved
ceiling and otherwise stops, saved work intact. Each admitted request is additionally held to
the operation's installed spend ceiling. Measured usage (provider attempts, and spend when a
ceiling is installed) is recorded per stage.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Protocol

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.variance.bootstrap import resolve_bootstrap
from exp.optimize.evaluation.variance.pipeline_build import BUILD_STAGE
from exp.optimize.evaluation.variance.pipeline_families import FAMILIES_STAGE
from exp.optimize.evaluation.variance.pipeline_records import RECORDS_STAGE
from exp.optimize.evaluation.variance.pipeline_spec import (
    PIPELINE_PROTOCOL,
    RUN_TYPE,
    STAGE_TYPE,
    STAGES,
    CostLine,
    DecisionVarianceSpec,
    PipelineApproval,
    PipelinePlan,
    PipelineReport,
    StageContext,
    StageDependencies,
    StageName,
    StagePlan,
    StageRecord,
    StageResult,
    StageRun,
    StageUsage,
    stage_identity,
)
from exp.optimize.evaluation.variance.pipeline_stages import (
    MINING_STAGE,
    RULES_STAGE,
    TRACEML_STAGE,
)
from exp.optimize.evaluation.variance.refined_prompts import STAGE_ROLES, refined_identity
from exp.runtime.models.spend_ceiling import current_spend_ceiling

logger = logging.getLogger(__name__)

PHYSICAL_ATTEMPT_TYPE = "provider-physical-attempt"
"""Artifact type the admitted transports record per provider attempt; counted as usage."""


class StageExecutor(Protocol):
    """Prices and runs one stage through its owners."""

    def keys(self, project: ProjectStore, spec: DecisionVarianceSpec) -> JsonObject:
        """Return content outside the spec that the stage reads, keyed into its identity."""
        ...

    def plan(
        self, project: ProjectStore, spec: DecisionVarianceSpec, code_revision: str
    ) -> tuple[tuple[CostLine, ...], tuple[str, ...]]:
        """Return the stage's upper-bound cost inventory and plan lines, credential-free."""
        ...

    def run(self, context: StageContext) -> StageRun:
        """Run, or resume, the stage and return its outputs."""
        ...


DEFAULT_EXECUTORS: dict[StageName, StageExecutor] = {
    "variance-build": BUILD_STAGE,
    "rules": RULES_STAGE,
    "traceml": TRACEML_STAGE,
    "error-mining": MINING_STAGE,
    "families": FAMILIES_STAGE,
    "records": RECORDS_STAGE,
}
"""The owner-bound executor of every stage."""


class SpendConsentRequired(ValueError):
    """Unsaved stages need explicit consent to this exact plan before any call."""


def plan_decision_variance(
    project: ProjectStore,
    spec: DecisionVarianceSpec,
    *,
    code_revision: str,
    executors: Mapping[StageName, StageExecutor] | None = None,
) -> PipelinePlan:
    """Name every stage, say which are saved and price the rest, with no credential or call.

    Args:
        project: Project holding the saved stages and the model catalog.
        spec: The registration.
        code_revision: Producing source identity.
        executors: Stage executors replacing the owner-bound defaults, by stage.

    Returns:
        The plan; a saved stage carries no cost.
    """
    spec = resolve_bootstrap(project, spec)
    bound = _executors(executors)
    rows: list[StagePlan] = []
    for stage, identity in _identities(project, spec, code_revision, bound):
        saved = project.artifacts.exists(identity + "-stage")
        costs, lines = ((), ("Saved; replays with no call.",))
        if not saved:
            costs, lines = bound[stage].plan(project, spec, code_revision)
        rows.append(
            StagePlan(stage=stage, identity=identity, saved=saved, costs=costs, lines=lines)
        )
    return PipelinePlan(
        spec_sha256=sha256_json(spec), code_revision=code_revision, stages=tuple(rows)
    )


def run_decision_variance(
    project: ProjectStore,
    spec: DecisionVarianceSpec,
    *,
    code_revision: str,
    approval: PipelineApproval | None = None,
    executors: Mapping[StageName, StageExecutor] | None = None,
    dependencies: StageDependencies | None = None,
) -> PipelineReport:
    """Run, resume or replay the whole decision-variance sequence.

    Args:
        project: Project holding every saved step.
        spec: The registration.
        code_revision: Producing source identity.
        approval: Consent to the exact plan; required whenever a stage is unsaved.
        executors: Stage executors replacing the owner-bound defaults, by stage.
        dependencies: Injected collaborators the stages use instead of their admitted defaults.

    Returns:
        The report; it stops early, saved work intact, when the next stage's bound does not fit
        the approved ceiling.

    Raises:
        SpendConsentRequired: A stage is unsaved and the approval is missing, names another
            plan, or does not accept the plan's unpriced components.
    """
    spec = resolve_bootstrap(
        project, DecisionVarianceSpec.model_validate_json(spec.model_dump_json())
    )
    bound = _executors(executors)
    plan = plan_decision_variance(project, spec, code_revision=code_revision, executors=bound)
    if not plan.complete:
        _require_approval(plan, approval)
    run = _register(project, spec, plan, code_revision)
    records: dict[StageName, StageRecord] = {}
    references: list[ArtifactInput] = []
    results: list[StageResult] = []
    spent = 0.0
    for row in plan.stages:
        saved = load_stage_record(project, row.identity)
        if saved is None:
            assert approval is not None
            if spent + row.known_usd > approval.ceiling_usd + 1e-12:
                reason = (
                    f"stage {row.stage} may cost up to ${row.known_usd:.6f}, more than the "
                    f"${approval.ceiling_usd - spent:.6f} left of the approved ceiling"
                )
                logger.info("decision-variance run stopped: %s", reason)
                return PipelineReport(run=run, plan=plan, stages=tuple(results), stopped=reason)
            saved = _execute(
                project, spec, row, bound[row.stage], records, references, code_revision,
                dependencies or StageDependencies(),
            )  # fmt: skip
            spent += saved.usage.accounted_usd or 0.0
            replayed = False
        else:
            replayed = True
        reference = artifact_input(project.artifacts.read(row.identity + "-stage").manifest)
        records[row.stage] = saved
        references.append(reference)
        results.append(
            StageResult(
                stage=row.stage,
                identity=row.identity,
                record=reference,
                replayed=replayed,
                usage=saved.usage,
                summary=saved.summary,
            )
        )
    return PipelineReport(run=run, plan=plan, stages=tuple(results))


def load_stage_record(project: ProjectStore, identity: str) -> StageRecord | None:
    """Read one saved stage record, or None when the stage has not finished.

    Args:
        project: Project holding the record.
        identity: The stage identity.

    Returns:
        The record, or None.
    """
    store = project.artifacts
    if not store.exists(identity + "-stage"):
        return None
    return StageRecord.model_validate_json(store.read_bytes(identity + "-stage", "record.json"))


def stage_outputs[Model: ContractModel](record: StageRecord, model: type[Model]) -> Model:
    """Validate one record's outputs as its stage's typed model.

    Args:
        record: The saved record.
        model: The stage's outputs model.

    Returns:
        The typed outputs.
    """
    return model.model_validate(record.outputs)


def _execute(
    project: ProjectStore,
    spec: DecisionVarianceSpec,
    row: StagePlan,
    executor: StageExecutor,
    records: Mapping[StageName, StageRecord],
    upstream: list[ArtifactInput],
    code_revision: str,
    dependencies: StageDependencies,
) -> StageRecord:
    """Run one unsaved stage under its lock and save its record with its measured usage.

    Args:
        project: Project holding every saved step.
        spec: The registration.
        row: The stage's plan.
        executor: Its executor.
        records: Records of every earlier stage.
        upstream: References of every earlier stage's record.
        code_revision: Producing source identity.
        dependencies: Injected collaborators.

    Returns:
        The saved record.
    """
    store = project.artifacts
    with CoordinationLock(store.project_directory / (row.identity + ".lock"), timeout=0):
        saved = load_stage_record(project, row.identity)
        if saved is not None:
            return saved
        before = _measure(project)
        context = StageContext(
            project=project,
            spec=spec,
            identity=row.identity,
            code_revision=code_revision,
            upstream=dict(records),
            dependencies=dependencies,
            plan=row,
        )
        result = executor.run(context)
        usage = _usage(before, _measure(project))
        record = StageRecord(
            stage=row.stage,
            identity=row.identity,
            upstream=tuple(upstream),
            outputs=result.outputs.model_dump(mode="json"),
            references=tuple(
                sorted({ref for ref in result.references if ref is not None}, key=_artifact_id)
            ),
            usage=usage,
            summary=result.summary,
        )
        store.write_json(
            artifact_id=row.identity + "-stage",
            artifact_type=STAGE_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=tuple(sorted({*upstream, *record.references}, key=_artifact_id)),
            ),
            files={"record.json": record},
        )
        logger.info("decision-variance stage %s saved as %s", row.stage, row.identity)
        return record


def _identities(
    project: ProjectStore,
    spec: DecisionVarianceSpec,
    code_revision: str,
    executors: Mapping[StageName, StageExecutor],
) -> tuple[tuple[StageName, str], ...]:
    """Chain every stage's identity onto the identities of the stages before it.

    Approved refined prompts of the roles a stage uses (:data:`~.refined_prompts.STAGE_ROLES`)
    join its identity, so approving one re-keys that stage and the ones after it; with none
    approved identities are unchanged.
    """
    chain: list[str] = []
    rows: list[tuple[StageName, str]] = []
    for stage in STAGES:
        refined = refined_identity(project, STAGE_ROLES.get(stage, frozenset()))
        extra = {**executors[stage].keys(project, spec), **refined}
        identity = stage_identity(stage, spec, tuple(chain), code_revision, extra)
        chain.append(identity)
        rows.append((stage, identity))
    return tuple(rows)


def _executors(
    executors: Mapping[StageName, StageExecutor] | None,
) -> dict[StageName, StageExecutor]:
    """Overlay caller executors on the owner-bound defaults."""
    return {**DEFAULT_EXECUTORS, **(executors or {})}


def _require_approval(plan: PipelinePlan, approval: PipelineApproval | None) -> None:
    """Refuse a run with unsaved stages that this exact plan was not consented for.

    Raises:
        SpendConsentRequired: The approval is missing, names another plan, or leaves unpriced
            components unaccepted.
    """
    if approval is None:
        raise SpendConsentRequired(
            f"{sum(not row.saved for row in plan.stages)} stages are unsaved; consent to the "
            f"plan (up to ${plan.known_usd:.6f} priced) before running"
        )
    if approval.plan_sha256 != plan.digest:
        raise SpendConsentRequired("the approval names another plan; review the current plan")
    if plan.unpriced and not approval.unpriced_allowed:
        names = sorted({row.name for row in plan.unpriced})
        raise SpendConsentRequired(f"unpriced components need explicit consent: {names}")


def _register(
    project: ProjectStore, spec: DecisionVarianceSpec, plan: PipelinePlan, code_revision: str
) -> ArtifactInput:
    """Save, or adopt, the run registration before any stage runs.

    The registration holds the spec and every stage identity, so the episode-judge profile in
    ``spec.mining`` is on record before any family rubric is authored.

    Returns:
        The registration's reference.
    """
    store = project.artifacts
    identity = stable_id(
        "decision-variance-run",
        {
            "protocol": PIPELINE_PROTOCOL,
            "spec": sha256_json(spec),
            "stages": [row.identity for row in plan.stages],
        },
    )
    artifact_id = identity + "-run"
    if not store.exists(artifact_id):
        store.write_json(
            artifact_id=artifact_id,
            artifact_type=RUN_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1, created_at=datetime.now(UTC), code_revision=code_revision
            ),
            files={
                "spec.json": spec,
                "stages.json": {row.stage: row.identity for row in plan.stages},
            },
        )
    return artifact_input(store.read(artifact_id).manifest)


def _measure(project: ProjectStore) -> tuple[int, float | None, float | None]:
    """Count recorded provider attempts and read the installed ceiling's accounting."""
    attempts = len(project.artifacts.list_ids(artifact_type=PHYSICAL_ATTEMPT_TYPE))
    ceiling = current_spend_ceiling()
    if ceiling is None:
        return attempts, None, None
    summary = ceiling.summary()
    return attempts, summary.accounted_usd, summary.settled_usd


def _usage(
    before: tuple[int, float | None, float | None], after: tuple[int, float | None, float | None]
) -> StageUsage:
    """Return the usage measured between two readings."""
    accounted = None if before[1] is None or after[1] is None else after[1] - before[1]
    settled = None if before[2] is None or after[2] is None else after[2] - before[2]
    return StageUsage(
        physical_attempts=max(0, after[0] - before[0]),
        accounted_usd=accounted,
        settled_usd=settled,
    )


def _artifact_id(reference: ArtifactInput) -> str:
    """Sort key of a reference."""
    return reference.artifact_id
