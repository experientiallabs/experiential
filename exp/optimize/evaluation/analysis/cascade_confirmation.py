"""Confirm a fitted decision cascade in episodes on held-out task lineages.

The fit estimates offline, on states the main model reached. Once the cascade decides, the actor
reaches states the fit never saw, so the frozen policy is confirmed by running it. Each held-out
case runs twice through the maintained actor panel, on the identical checkpoint, world, verifier
and actor allowance: once with the main model alone and once with the fitted cascade. Every
planned branch stays in the denominator.

Task success and policy compliance are compared per lineage, the independent unit, as the paired
difference cascade minus main with its two-sided 95 percent Wilson interval at the lineage count.
Unknown outcomes are imputed against the cascade (a failure in the cascade arm, a success in the
main arm), and the known-only difference is reported beside it. The cascade is confirmed only
when the lower interval end of both differences is within the fit's registered margins. The
report also measures delegation in the episodes, beside the fit's offline share, as the size of
the reached-state shift.

Continuation execution consumes only development (``fit``) partition tasks, so held-out here
means lineages disjoint from every fit source; the project's reserved held-out tasks stay closed.
"""

from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.evaluations.uncertainty import unit_difference_summary, unit_mean_summary
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.analysis.decision_cascade import (
    CASCADE_FIT_PROTOCOL,
    CascadeFitReport,
    load_cascade_fit,
)
from exp.optimize.evaluation.continuations.actors import (
    ActorPanelReport,
    ActorPanelSpec,
    PanelActor,
    preflight_actor_panel,
    run_actor_panel,
    saved_actor_panel,
)
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.continuations.study import ContinuationBranch
from exp.runtime.agents.cascade import CascadePolicy
from exp.simulation.continuations import load_continuation_checkpoint

CONFIRMATION_SCOPE = (
    "closed-loop episodes on lineages disjoint from the fit; development-partition tasks; "
    "intervals at the lineage count with unknowns imputed against the cascade; applies to "
    "the exact policy, main model, decider and environments run here"
)


class CascadeConfirmationSpec(ContractModel):
    """A saved fit and the held-out cases its frozen policy must be confirmed on.

    Attributes:
        fit: Saved fit result whose policy is confirmed.
        cases: Main-model jobs on lineages disjoint from the fit; each runs main-only and with
            the cascade.
    """

    fit: ArtifactInput
    cases: tuple[ContinuationJob, ...] = Field(min_length=1, max_length=100)

    def identity(self, code_revision: str) -> str:
        """Bind the fit, cases, protocol and implementation."""
        return stable_id(
            "decision-cascade-confirmation",
            {"spec": sha256_json(self), "protocol": CASCADE_FIT_PROTOCOL, "code": code_revision},
        )


class CascadeConfirmationReport(ContractModel):
    """Closed-loop non-inferiority of the cascade against the main model alone.

    Attributes:
        spec: Registration.
        policy: The confirmed policy, frozen by the fit.
        panel: Saved actor-panel result holding both arms' episodes.
        arms: Per-arm planned branches, verified outcomes and lineage-mean rates.
        success: Paired lineage difference of task success, unknowns imputed against the cascade.
        compliance: Paired lineage difference of policy compliance, imputed the same way.
        success_known: The task-success difference over known outcomes only.
        compliance_known: The policy-compliance difference over known outcomes only.
        delegation: Delegation in the cascade episodes beside the fit's offline estimate.
        confirmed: Whether both lower interval ends are within the registered margins.
        blockers: Every reason confirmation is withheld.
        scope: Interpretation limits.
    """

    spec: CascadeConfirmationSpec
    policy: CascadePolicy
    panel: ArtifactInput
    arms: JsonObject
    success: JsonObject
    compliance: JsonObject
    success_known: JsonObject
    compliance_known: JsonObject
    delegation: JsonObject
    confirmed: bool
    blockers: tuple[str, ...]
    scope: str = CONFIRMATION_SCOPE


def confirmation_panel(
    project: ProjectStore, spec: CascadeConfirmationSpec
) -> tuple[CascadeFitReport, ActorPanelSpec]:
    """Build the paired main-only and cascade panel after checking held-out separation.

    Raises:
        ValueError: The fit selected no policy, or a case shares a lineage or source trace with
            the fit, uses another main model or alias, or already runs a cascade.
    """
    fit = load_cascade_fit(project, spec.fit)
    if fit.policy is None:
        raise ValueError(f"cascade fit status is {fit.status}; there is no policy to confirm")
    used = set(fit.spec.development_lineages)
    traces: set[str] = set()
    for reference in fit.points.studies:
        study = load_study(project.artifacts, reference)
        checkpoint = load_continuation_checkpoint(project.artifacts, study.spec.checkpoint)
        traces.update(checkpoint.task.source_trace_ids)
    aliases = {job.actor_alias for job in spec.cases}
    for job in spec.cases:
        task = load_continuation_checkpoint(project.artifacts, job.study.checkpoint).task
        if task.lineage_group_id in used or traces.intersection(task.source_trace_ids):
            raise ValueError(
                f"confirmation lineage {task.lineage_group_id} was used to fit the cascade; "
                "confirm on lineages disjoint from every fit source"
            )
        if job.study.actor_model != fit.spec.main_model or job.study.cascade is not None:
            raise ValueError("confirmation cases must be main-model-only jobs of the fitted model")
    if len(aliases) != 1:
        raise ValueError("confirmation cases must share one main-model alias")
    alias = aliases.pop()
    panel = ActorPanelSpec(
        cases=spec.cases,
        actors=(
            PanelActor(alias=alias, model=fit.spec.main_model, label="main model only"),
            PanelActor(
                alias=alias,
                model=fit.spec.main_model,
                label="decision cascade",
                cascade=fit.policy,
            ),
        ),
        minimum_difference=fit.spec.objective.success_margin,
    )
    return fit, panel


def preflight_cascade_confirmation(project: ProjectStore, spec: CascadeConfirmationSpec) -> None:
    """Verify the fit, separation, both arms and every case without credentials or dispatch."""
    _, panel = confirmation_panel(project, spec)
    preflight_actor_panel(project, panel)


Outcomes = dict[str, dict[str, list[float]]]


def _arm(
    project: ProjectStore, report: ActorPanelReport, actor: int, *, unknown: float
) -> tuple[Outcomes, Outcomes, JsonObject]:
    """Collect one arm's per-lineage outcomes, imputed and known-only, and its counts."""
    imputed: Outcomes = {"success": {}, "compliance": {}}
    known: Outcomes = {"success": {}, "compliance": {}}
    planned = passes = compliant = missing = 0
    for cell in report.cells:
        if cell.actor != actor:
            continue
        lineage = load_continuation_checkpoint(
            project.artifacts, cell.job.study.checkpoint
        ).task.lineage_group_id
        branches = (
            () if cell.result is None else load_study(project.artifacts, cell.result).branches
        )
        grades = [branch.assessment for branch in branches]
        grades += [None] * (cell.job.study.repeats - len(grades))
        for grade in grades:
            planned += 1
            for name, verdict in (
                ("success", None if grade is None else grade.status),
                ("compliance", None if grade is None else grade.policy.status),
            ):
                value = None if verdict in (None, "unassessable") else float(verdict == "pass")
                imputed[name].setdefault(lineage, []).append(unknown if value is None else value)
                known[name].setdefault(lineage, [])
                if value is not None:
                    known[name][lineage].append(value)
            passes += grade is not None and grade.status == "pass"
            compliant += grade is not None and grade.policy.status == "pass"
            missing += grade is None or grade.status == "unassessable"
    counts: JsonObject = {
        "planned": planned,
        "passes": passes,
        "policy_compliant": compliant,
        "unknown": missing,
        "success": unit_mean_summary(known["success"]),
        "compliance": unit_mean_summary(known["compliance"]),
    }
    return imputed, known, counts


def _delegation(
    project: ProjectStore, report: ActorPanelReport, fit: CascadeFitReport
) -> JsonObject:
    """Measure delegation in the cascade episodes beside the fit's offline estimate."""
    branches: list[ContinuationBranch] = [
        branch
        for cell in report.cells
        if cell.actor == 1 and cell.result is not None
        for branch in load_study(project.artifacts, cell.result).branches
    ]
    records = [row for branch in branches for row in branch.cascade_decisions]
    delegated = [row for row in records if row.decided_by == "decider"]
    offline = next(row for row in fit.rows if row.threshold == fit.selected)
    return {
        "decisions": len(records),
        "delegated": len(delegated),
        "calls_avoided": sum(not row.main_called for row in delegated),
        "forced_not_followed": sum(row.followed is False for row in delegated),
        "delegated_share": len(delegated) / len(records) if records else None,
        "mean_delegated_confidence": (
            sum(row.confidence or 0 for row in delegated) / len(delegated) if delegated else None
        ),
        "fit_delegated_share": offline.delegated_share,
        "notes": {
            note: sum(row.note == note for row in records)
            for note in sorted({row.note for row in records if row.note is not None})
        },
    }


def _report(
    project: ProjectStore,
    spec: CascadeConfirmationSpec,
    fit: CascadeFitReport,
    panel: ActorPanelReport,
    panel_reference: ArtifactInput,
) -> CascadeConfirmationReport:
    """Derive both differences, the decision and its blockers from the saved panel."""
    assert fit.policy is not None
    main, main_known, main_counts = _arm(project, panel, 0, unknown=1.0)
    cascade, cascade_known, cascade_counts = _arm(project, panel, 1, unknown=0.0)
    objective = fit.spec.objective
    success = unit_difference_summary(cascade["success"], main["success"])
    compliance = unit_difference_summary(cascade["compliance"], main["compliance"])
    blockers = []
    for name, difference, margin in (
        ("task success", success, objective.success_margin),
        ("policy compliance", compliance, objective.compliance_margin),
    ):
        interval = difference["wilson_95"]
        units = difference["independent_units"]
        if not isinstance(units, int) or units < objective.minimum_lineages:
            blockers.append(f"{name}: fewer paired lineages than the registered minimum")
        if not isinstance(interval, list) or not isinstance(interval[0], float | int):
            blockers.append(f"{name}: no paired lineage outcome")
        elif interval[0] < -margin:
            blockers.append(f"{name}: lower interval end is below the registered margin")
    return CascadeConfirmationReport(
        spec=spec,
        policy=fit.policy,
        panel=panel_reference,
        arms={"main model only": main_counts, "decision cascade": cascade_counts},
        success=success,
        compliance=compliance,
        success_known=unit_difference_summary(cascade_known["success"], main_known["success"]),
        compliance_known=unit_difference_summary(
            cascade_known["compliance"], main_known["compliance"]
        ),
        delegation=_delegation(project, panel, fit),
        confirmed=not blockers,
        blockers=tuple(blockers),
    )


def saved_cascade_confirmation(
    project: ProjectStore, spec: CascadeConfirmationSpec, *, code_revision: str
) -> CascadeConfirmationReport | None:
    """Read a completed confirmation and re-derive it from the saved panel and fit."""
    identity = spec.identity(code_revision)
    if not project.artifacts.exists(identity + "-result"):
        return None
    report = CascadeConfirmationReport.model_validate_json(
        project.artifacts.read_bytes(identity + "-result", "report.json")
    )
    fit, panel_spec = confirmation_panel(project, spec)
    panel = saved_actor_panel(project, panel_spec, code_revision=code_revision)
    if (
        panel is None
        or report.spec != spec
        or _report(project, spec, fit, panel, report.panel) != report
    ):
        raise ValueError("saved cascade confirmation does not reproduce from its panel")
    return report


def run_cascade_confirmation(
    project: ProjectStore,
    spec: CascadeConfirmationSpec,
    *,
    code_revision: str,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> CascadeConfirmationReport:
    """Run or resume both arms through the actor panel, then judge non-inferiority.

    Returns:
        The saved confirmation; a completed one replays without runtime code or provider calls.
    """
    store, identity = project.artifacts, spec.identity(code_revision)
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        saved = saved_cascade_confirmation(project, spec, code_revision=code_revision)
        if saved is not None:
            return saved
        fit, panel_spec = confirmation_panel(project, spec)
        panel = run_actor_panel(project, panel_spec, code_revision=code_revision)
        panel_reference = artifact_input(
            store.read(panel_spec.identity(code_revision) + "-result").manifest
        )
        report = _report(project, spec, fit, panel, panel_reference)
        store.write_json(
            artifact_id=identity + "-result",
            artifact_type="decision-cascade-confirmation",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=clock(),
                code_revision=code_revision,
                inputs=tuple(sorted((spec.fit, panel_reference), key=lambda r: r.artifact_id)),
            ),
            files={"report.json": report},
        )
        return report
