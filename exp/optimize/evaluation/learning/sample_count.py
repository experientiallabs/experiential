"""How many trajectories the simulator needs: a registered select-then-confirm procedure.

Asking which memory checkpoint first exceeds an accuracy threshold is a search. Checking every
checkpoint on the same held-out traces and reporting the first one above the threshold turns
those traces into selection data, and an ordinary interval computed afterwards does not undo
the search. This module makes the question a registered procedure over the existing owners:

1. Register, before any measurement is read, the trajectory-count grid, the canonical fit-trace
   learning order, the metric, the threshold and its criterion (point estimate or Wilson lower
   bound), and the development lineages.
2. Measure each grid count on the development lineages with a held-out audit of the memory
   checkpoint that had learned exactly that many fit trajectories (``exp eval --audit``).
3. Select on development: walk the grid in registered order and take the first count whose
   development measurement meets the criterion.
4. Freeze the selected memory checkpoint, the simulator and serving configuration, retrieval
   and recall, role budgets, judge and code revision.
5. Confirm on untouched lineages: an independent audit receipt of the frozen checkpoint, opened
   after selection, over lineages never used for development.
6. Report the consumed trajectory count, the exact-reply metric, its uncertainty at the
   lineage count, coverage and every failure.

A development-selected checkpoint is never reported as confirmation: reusing a development
receipt or lineage, a re-audit, a changed configuration or an invalidated receipt is refused.
"""

from collections.abc import Sequence
from datetime import datetime
from typing import Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.evaluations.uncertainty import unit_mean_summary
from exp.common.project import ProjectStore, artifact_input
from exp.common.tasks.store import load_task_set
from exp.optimize.evaluation.audit.panel import PlannedPanel
from exp.optimize.evaluation.audit.receipts import Basis, ReceiptView
from exp.optimize.evaluation.fidelity.replay import ReplayResult
from exp.optimize.evaluation.learning.traces import fit_cases, learning_plan

# Identity of the code that decides this artifact's behavior, in place of its source text.
# Bump it when behavior changes. exp/tests/code_protocols_test.py fails when the covered
# source changes until it is bumped or, for a pure refactor, its digest is updated.
SAMPLE_COUNT_PROTOCOL = "sample-count-v1"
REGISTRATION_TYPE = "sample-count-registration"
ESTIMAND = "equal-weight mean over lineages of each lineage's exact-reply share"

# Every fidelity-spec field that varies by case rather than by frozen configuration.
_CASE_FIELDS: dict[str, bool | set[str]] = {
    "run_id": True,
    "deadline": True,
    "task": True,
    "workload": True,
    "references": True,
    "facts": True,
    "memory": {"checkpoint"},
}


class SampleCountRegistration(ContractModel):
    """The procedure registered before any development measurement is read.

    Attributes:
        project: Project whose fit-only memory learns the trajectories.
        memory_lineage: Fit-only memory lineage whose checkpoints are measured.
        trace_order_sha256: Identity of the canonical fit-trace learning order.
        grid: Whole-trajectory counts to evaluate, strictly increasing, in evaluation order.
        metric: Exact simulated replies over recorded replies.
        threshold: Accuracy the selected count must exceed.
        criterion: ``point_estimate`` compares the development estimate; ``wilson_lower_bound``
            compares its lower 95 percent Wilson bound at the lineage count.
        development_lineages: Held-out lineages used for selection, sorted.
        registered_at: When the procedure was registered; earlier audits never count.
    """

    project: str = Field(min_length=1)
    memory_lineage: str = Field(min_length=1)
    trace_order_sha256: Sha256
    grid: tuple[int, ...] = Field(min_length=1)
    metric: Literal["exact_replies"] = "exact_replies"
    threshold: float = Field(gt=0, lt=1, allow_inf_nan=False)
    criterion: Literal["point_estimate", "wilson_lower_bound"]
    development_lineages: tuple[str, ...] = Field(min_length=1)
    registered_at: AwareDatetime

    @model_validator(mode="after")
    def ordered(self) -> Self:
        """Require a strictly increasing positive grid and unique sorted lineages."""
        if self.grid[0] < 1 or any(b <= a for a, b in zip(self.grid, self.grid[1:], strict=False)):
            raise ValueError("the trajectory-count grid must be positive and strictly increasing")
        if self.development_lineages != tuple(sorted(set(self.development_lineages))):
            raise ValueError("development lineages must be unique and sorted")
        return self

    @property
    def identity(self) -> str:
        """Registration identity, bound to the procedure protocol."""
        return stable_id(
            REGISTRATION_TYPE,
            {"registration": self.model_dump(mode="json"), "protocol": SAMPLE_COUNT_PROTOCOL},
        )


class LineageMeasurement(ContractModel):
    """Exact replies over one audited lineage.

    Attributes:
        lineage: The build's leakage lineage.
        traces: Audited traces of the lineage.
        measured: Traces with a saved result.
        exact: Exactly reproduced replies over measured traces.
        steps: Planned measured replies over measured traces.
    """

    lineage: str
    traces: int = Field(ge=1)
    measured: int = Field(ge=0)
    exact: int = Field(ge=0)
    steps: int = Field(ge=0)

    @property
    def complete(self) -> bool:
        """Whether every audited trace of the lineage has a result with planned replies."""
        return self.measured == self.traces and self.steps > 0


class CheckpointMeasurement(ContractModel):
    """One held-out audit of one frozen memory checkpoint.

    Attributes:
        receipt_id: Audit receipt.
        opened_at: When the receipt and its reservation were committed.
        basis: ``independent`` or ``development-re-audit``, fixed at minting.
        headline: The receipt's derived headline, qualified by any re-attempt.
        memory_lineage: Memory lineage read.
        memory_version: Frozen memory version read.
        trajectories: Fit trajectories that memory had consumed.
        configuration_sha256: Everything frozen besides the memory version: simulator, serving
            and output protocols, retrieval and recall, memory policy, role budgets, judge and
            code revision.
        lineages: Exact replies per audited lineage.
        reattempted_traces: Traces bought again after a lost or replaced run.
        notices: Invalidation notices: exposure found that predated the reservation.
    """

    receipt_id: str
    opened_at: AwareDatetime
    basis: Basis
    headline: str
    memory_lineage: str
    memory_version: int = Field(ge=1)
    trajectories: int = Field(ge=0)
    configuration_sha256: Sha256
    lineages: tuple[LineageMeasurement, ...] = Field(min_length=1)
    reattempted_traces: int = Field(default=0, ge=0)
    notices: int = Field(default=0, ge=0)

    def summary(self) -> JsonObject:
        """The registered estimand with its lineage interval, coverage and pooled share."""
        complete = [row for row in self.lineages if row.complete]
        exact = sum(row.exact for row in self.lineages)
        steps = sum(row.steps for row in self.lineages)
        unmeasured = len(self.lineages) - len(complete)
        return {
            **unit_mean_summary({row.lineage: [row.exact / row.steps] for row in complete}),
            "unmeasured_lineages": unmeasured,
            "lineage_bounds": _lineage_bounds(complete, unmeasured),
            "pooled_exact_replies": {"exact": exact, "steps": steps},
            "coverage": {
                "traces": sum(row.traces for row in self.lineages),
                "measured": sum(row.measured for row in self.lineages),
            },
        }


class SampleCountSelection(ContractModel):
    """The development decision: one frozen count, or why no count was selected.

    Attributes:
        registration: Registration identity.
        status: ``selected``, ``threshold_not_met`` over the whole grid, or stopped at a grid
            count that was ``not_measured`` or ``incomplete``.
        selected_count: Selected trajectory count, when selected.
        stopped_at: Grid count where the walk stopped without a decision.
        memory_version: Frozen memory version of the selected count.
        configuration_sha256: Frozen configuration every measurement shared.
        development_receipts: Every development receipt consulted.
        last_development_at: Latest development receipt time; confirmation must follow it.
        evaluated: Each examined count with its development estimate and decision.
    """

    registration: str
    status: Literal["selected", "threshold_not_met", "not_measured", "incomplete"]
    selected_count: int | None = None
    stopped_at: int | None = None
    memory_version: int | None = None
    configuration_sha256: Sha256 | None = None
    development_receipts: tuple[str, ...] = ()
    last_development_at: AwareDatetime | None = None
    evaluated: tuple[JsonObject, ...] = ()


def register_sample_count_study(
    project: ProjectStore,
    *,
    grid: tuple[int, ...],
    threshold: float,
    criterion: Literal["point_estimate", "wilson_lower_bound"],
    development_lineages: Sequence[str],
    registered_at: datetime,
    code_revision: str,
) -> tuple[SampleCountRegistration, ArtifactInput]:
    """Register the procedure for the project's fit-only memory before reading any result.

    Args:
        project: Project with a completed build and a recorded fit-only learning plan.
        grid: Whole-trajectory counts to evaluate, strictly increasing.
        threshold: Accuracy the selected count must exceed.
        criterion: Whether the development estimate or its Wilson lower bound must exceed it.
        development_lineages: Held-out lineages reserved for selection.
        registered_at: Registration time.
        code_revision: Producing implementation identity.

    Returns:
        The registration and its immutable artifact; an identical registration replays.

    Raises:
        ValueError: A count exceeds the fit traces, a lineage is not held out, or a different
            registration already exists under the same identity.
    """
    order = fit_cases(project)
    registration = SampleCountRegistration(
        project=project.paths.project_id,
        memory_lineage=learning_plan(project, scope="fit").lineage,
        trace_order_sha256=sha256_json([case.trace_id for case in order]),
        grid=grid,
        threshold=threshold,
        criterion=criterion,
        development_lineages=tuple(sorted(set(development_lineages))),
        registered_at=registered_at,
    )
    if registration.grid[-1] > len(order):
        raise ValueError(
            f"the grid reaches {registration.grid[-1]} trajectories but the build has "
            f"{len(order)} fit traces; register counts the memory can learn"
        )
    build = project.load_project().build
    if build is None:
        raise ValueError("register after a completed build")
    held_out = {
        task.lineage_group_id
        for task in load_task_set(project.artifacts, build.task_set.artifact_id).tasks
        if task.partition == "held_out"
    }
    if set(registration.development_lineages) - held_out:
        raise ValueError("development lineages must be held-out lineages the memory never learns")
    if len(held_out) <= len(registration.development_lineages):
        raise ValueError("leave held-out lineages untouched for confirmation")
    identity = registration.identity
    if project.artifacts.exists(identity):
        saved = SampleCountRegistration.model_validate_json(
            project.artifacts.read_bytes(identity, "registration.json")
        )
        if saved != registration:
            raise ValueError("a different sample-count registration has this identity")
        return saved, artifact_input(project.artifacts.read(identity).manifest)
    manifest = project.artifacts.write_json(
        artifact_id=identity,
        artifact_type=REGISTRATION_TYPE,
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=registered_at,
            code_revision=code_revision,
            inputs=(build.task_set,),
        ),
        files={"registration.json": registration.model_dump(mode="json")},
    )
    return registration, artifact_input(manifest)


def audit_measurement(
    view: ReceiptView, panel: PlannedPanel, results: Sequence[ReplayResult]
) -> CheckpointMeasurement:
    """Measure one memory-profile audit receipt from its panel and saved results.

    Args:
        view: The receipt with every appended child.
        panel: The receipt's registered panel, whose specs carry the frozen configuration.
        results: Saved results of the panel's current runs; missing traces stay unmeasured.

    Returns:
        The measurement of the receipt's frozen checkpoint, per audited lineage.

    Raises:
        ValueError: The receipt read no memory checkpoint, the panel belongs to another
            receipt, or the panel's runs do not share one frozen configuration.
    """
    receipt = view.receipt
    if receipt.checkpoint is None:
        raise ValueError("trajectory counts are memory checkpoints; audit the memory profile")
    if panel.identity != receipt.panel:
        raise ValueError("the panel does not belong to this audit receipt")
    configurations = {
        sha256_json(
            {
                "spec": spec.model_dump(mode="json", exclude=_CASE_FIELDS),
                "code_revision": receipt.code_revision,
            }
        )
        for spec in panel.specs
    }
    if len(configurations) != 1:
        raise ValueError("audited runs do not share one frozen configuration")
    by_trace = {result.trace_id: result for result in results}
    lineages: dict[str, list[ReplayResult | None]] = {}
    for case in receipt.cases:
        lineages.setdefault(case.lineage, []).append(by_trace.get(case.trace_id))
    return CheckpointMeasurement(
        receipt_id=receipt.receipt_id,
        opened_at=receipt.opened_at,
        basis=receipt.basis,
        headline=view.headline,
        memory_lineage=receipt.checkpoint.lineage,
        memory_version=receipt.checkpoint.version,
        trajectories=receipt.checkpoint.fit_traces_learned,
        configuration_sha256=configurations.pop(),
        lineages=tuple(
            LineageMeasurement(
                lineage=lineage,
                traces=len(rows),
                measured=sum(row is not None for row in rows),
                exact=sum(row.exact for row in rows if row is not None),
                steps=sum(row.steps for row in rows if row is not None),
            )
            for lineage, rows in sorted(lineages.items())
        ),
        reattempted_traces=len(view.affected_traces),
        notices=len(view.notices),
    )


def select_sample_count(
    registration: SampleCountRegistration, development: Sequence[CheckpointMeasurement]
) -> SampleCountSelection:
    """Walk the registered grid on development measurements and freeze the first count that
    meets the registered criterion.

    Args:
        registration: The registered procedure.
        development: Audits of grid counts over exactly the development lineages.

    Returns:
        The selection, or the reason none was made. Counts examined after the selected one
        remain development evidence.

    Raises:
        ValueError: A measurement predates registration, reads another memory, misses or adds
            lineages, repeats or leaves the grid, or changes the frozen configuration.
    """
    by_count: dict[int, CheckpointMeasurement] = {}
    for row in development:
        if row.opened_at < registration.registered_at:
            raise ValueError("a development audit predates registration; register first")
        if row.memory_lineage != registration.memory_lineage:
            raise ValueError("a development audit read another memory lineage")
        if tuple(item.lineage for item in row.lineages) != registration.development_lineages:
            raise ValueError("development audits must cover exactly the development lineages")
        if row.trajectories not in registration.grid or row.trajectories in by_count:
            raise ValueError("each registered grid count is measured once, and only those")
        by_count[row.trajectories] = row
    if len({row.configuration_sha256 for row in development}) > 1:
        raise ValueError("development audits changed the frozen configuration between counts")
    base = SampleCountSelection(
        registration=registration.identity,
        status="threshold_not_met",
        configuration_sha256=development[0].configuration_sha256 if development else None,
        development_receipts=tuple(row.receipt_id for row in development),
        last_development_at=max((row.opened_at for row in development), default=None),
    )
    evaluated: list[JsonObject] = []
    for count in registration.grid:
        row = by_count.get(count)
        if row is None:
            return base.model_copy(
                update={
                    "status": "not_measured",
                    "stopped_at": count,
                    "evaluated": tuple(evaluated),
                }
            )
        summary = row.summary()
        met = _meets(registration, summary)
        evaluated.append(
            {"trajectories": count, "receipt": row.receipt_id, "development": summary, "met": met}
        )
        if summary["unmeasured_lineages"]:
            return base.model_copy(
                update={"status": "incomplete", "stopped_at": count, "evaluated": tuple(evaluated)}
            )
        if met:
            return base.model_copy(
                update={
                    "status": "selected",
                    "selected_count": count,
                    "memory_version": row.memory_version,
                    "evaluated": tuple(evaluated),
                }
            )
    return base.model_copy(update={"evaluated": tuple(evaluated)})


def confirm_sample_count(
    registration: SampleCountRegistration,
    selection: SampleCountSelection,
    confirmation: CheckpointMeasurement,
) -> JsonObject:
    """Report the frozen selected count measured on untouched lineages.

    Args:
        registration: The registered procedure.
        selection: The development decision being confirmed.
        confirmation: An independent audit of the frozen selected checkpoint.

    Returns:
        The consumed trajectory count, the metric with its interval at the lineage count,
        coverage, failures and whether the registered criterion holds on confirmation, beside
        the optimistic development estimate.

    Raises:
        ValueError: No count was selected, or the confirmation reuses development evidence,
            is not an independent audit, predates selection, was invalidated, or changed the
            frozen checkpoint or configuration.
    """
    if selection.registration != registration.identity:
        raise ValueError("the selection belongs to another registration")
    if selection.status != "selected" or selection.selected_count is None:
        raise ValueError(f"no development-selected count to confirm ({selection.status})")
    reused = set(registration.development_lineages) & {row.lineage for row in confirmation.lineages}
    if confirmation.receipt_id in selection.development_receipts or reused:
        raise ValueError(
            "a development-selected checkpoint cannot be reported as confirmation on the "
            "development evidence that selected it; audit untouched held-out lineages"
            + (f" (reused lineages: {', '.join(sorted(reused))})" if reused else "")
        )
    if confirmation.basis != "independent":
        raise ValueError("confirmation needs an independent audit receipt, not a re-audit")
    if confirmation.notices:
        raise ValueError("the confirmation receipt was invalidated by predating exposure")
    if (
        selection.last_development_at is not None
        and confirmation.opened_at <= selection.last_development_at
    ):
        raise ValueError("confirmation must be opened after every development measurement")
    if (
        confirmation.memory_lineage != registration.memory_lineage
        or confirmation.memory_version != selection.memory_version
        or confirmation.trajectories != selection.selected_count
        or confirmation.configuration_sha256 != selection.configuration_sha256
    ):
        raise ValueError(
            "confirmation must read the frozen selected checkpoint with the frozen simulator, "
            "retrieval, judge and code revision"
        )
    summary = confirmation.summary()
    development = next(
        row["development"]
        for row in selection.evaluated
        if row["trajectories"] == selection.selected_count
    )
    return {
        "protocol": SAMPLE_COUNT_PROTOCOL,
        "registration": registration.identity,
        "trajectories": selection.selected_count,
        "metric": registration.metric,
        "estimand": ESTIMAND,
        "threshold": registration.threshold,
        "criterion": registration.criterion,
        "confirmation": {
            "receipt": confirmation.receipt_id,
            "headline": confirmation.headline,
            "memory_version": confirmation.memory_version,
            **summary,
            "reattempted_traces": confirmation.reattempted_traces,
        },
        "criterion_met_on_confirmation": _meets(registration, summary),
        "development_at_selected_count": development,
        "scope": (
            "development estimates were used to choose the count and are optimistic; only the "
            "confirmation is an estimate for the frozen checkpoint, over the audited lineages"
        ),
    }


def _meets(registration: SampleCountRegistration, summary: JsonObject) -> bool:
    """Whether a measurement's estimate or lower bound exceeds the registered threshold."""
    estimate, interval = summary["estimate"], summary["wilson_95"]
    if summary["unmeasured_lineages"] or not isinstance(estimate, float | int):
        return False
    if registration.criterion == "point_estimate":
        return estimate > registration.threshold
    lower = interval[0] if isinstance(interval, list) else None
    return isinstance(lower, float | int) and lower > registration.threshold


def _lineage_bounds(complete: list[LineageMeasurement], unmeasured: int) -> list[float]:
    """Equal-weight lineage mean bounds with each unmeasured lineage at zero and at one."""
    shares = sum(row.exact / row.steps for row in complete)
    total = len(complete) + unmeasured
    return [shares / total, (shares + unmeasured) / total]
