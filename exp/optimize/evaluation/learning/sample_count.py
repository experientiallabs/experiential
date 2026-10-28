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
5. Confirm on untouched lineages: the first independent audit receipt of the frozen checkpoint
   opened after selection, over lineages never used for development. Later audits cannot
   replace it, so confirmation is never chosen by its score.
6. Report the consumed trajectory count, the exact-reply metric, its uncertainty at the
   lineage count, coverage and every failure.

A development-selected checkpoint is never reported as confirmation: reusing a development
receipt or lineage, a re-audit, a changed configuration or an invalidated receipt is refused.

Order is causal, never read from a clock. Audit receipts are numbered by one owner: each
receipt's ``sequence`` is assigned while the root exposure ledger's transaction is held. The
registration and the saved selection are each written while that same transaction is held, and
record the latest receipt sequence at that moment (``receipts_before``), so a receipt counts as
opened after them only when its sequence is larger. The selection cites its registration and
every development receipt, and the confirmation cites the selection and its receipt, each by
identity and manifest digest. Every ``registered_at``, ``opened_at`` and ``created_at`` is
informational: a receipt whose clock claims a later time than the selection, but whose sequence
shows it was minted first, is refused, and one minted after the selection is accepted whatever
its clock claims.
"""

from collections.abc import Sequence
from datetime import datetime
from typing import Annotated, Literal, Self

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
from exp.optimize.evaluation.audit.panel import PlannedPanel, saved_held_out_results
from exp.optimize.evaluation.audit.receipts import (
    Basis,
    ReceiptView,
    audit_receipts,
    find_receipt,
    receipt_panel,
    receipt_reference,
    receipt_view,
    receipts_held,
)
from exp.optimize.evaluation.fidelity.replay import ReplayResult
from exp.optimize.evaluation.learning.traces import fit_cases, learning_plan

# Identity of the code that decides this artifact's behavior, in place of its source text.
# Bump it when behavior changes. exp/tests/code_protocols_test.py fails when the covered
# source changes until it is bumped or, for a pure refactor, its digest is updated.
SAMPLE_COUNT_PROTOCOL = "sample-count-v4"
REGISTRATION_TYPE = "sample-count-registration"
SELECTION_TYPE = "sample-count-selection"
CONFIRMATION_TYPE = "sample-count-confirmation"
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
        registered_at: When the procedure was registered; informational only.
        receipts_before: Latest audit receipt sequence when the procedure was registered; only
            receipts with a larger sequence were minted after it and may count.
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
    receipts_before: int = Field(ge=0)

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
        sequence: The receipt's position among the project's receipts, from 1; orders it.
        opened_at: When the receipt and its reservation were committed; informational only.
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
    sequence: int = Field(ge=1)
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
        last_development_sequence: Largest development receipt sequence; confirmation must
            follow it.
        receipts_before: Latest audit receipt sequence when the selection was saved, or None
            before it is saved; confirmation must have a larger sequence. Not part of the
            selection's identity, so selecting again replays the saved selection.
        evaluated: Each examined count with its development estimate and decision.
    """

    registration: str
    status: Literal["selected", "threshold_not_met", "not_measured", "incomplete"]
    selected_count: int | None = None
    stopped_at: int | None = None
    memory_version: int | None = None
    configuration_sha256: Sha256 | None = None
    development_receipts: tuple[str, ...] = ()
    last_development_sequence: int | None = None
    receipts_before: int | None = None
    evaluated: tuple[JsonObject, ...] = ()

    @property
    def identity(self) -> str:
        """Selection identity over the decision, bound to the procedure protocol."""
        decision = self.model_dump(mode="json", exclude={"receipts_before"})
        return stable_id(
            SELECTION_TYPE, {"selection": sha256_json(decision), "protocol": SAMPLE_COUNT_PROTOCOL}
        )


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
        registered_at: Registration time, informational only.
        code_revision: Producing implementation identity.

    Returns:
        The registration and its immutable artifact; an identical registration replays.

    Raises:
        ValueError: A count exceeds the fit traces, a lineage is not held out, or a different
            registration already exists under the same identity.
    """
    with receipts_held(project) as receipts_before:
        registration = plan_sample_count_study(
            project,
            grid=grid,
            threshold=threshold,
            criterion=criterion,
            development_lineages=development_lineages,
            registered_at=registered_at,
            receipts_before=receipts_before,
        )
        identity = registration.identity
        if project.artifacts.exists(identity):
            saved = SampleCountRegistration.model_validate_json(
                project.artifacts.read_bytes(identity, "registration.json")
            )
            if saved != registration:
                raise ValueError("a different sample-count registration has this identity")
            return saved, artifact_input(project.artifacts.read(identity).manifest)
        build = project.load_project().build
        if build is None:
            raise ValueError("register after a completed build")
        # Written while receipt minting is held, so every later receipt has a larger sequence.
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


def plan_sample_count_study(
    project: ProjectStore,
    *,
    grid: tuple[int, ...],
    threshold: float,
    criterion: Literal["point_estimate", "wilson_lower_bound"],
    development_lineages: Sequence[str],
    registered_at: datetime,
    receipts_before: int,
) -> SampleCountRegistration:
    """Validate a registration against the project's build and fit order, writing nothing.

    Args:
        project: Project with a completed build and a recorded fit-only learning plan.
        grid: Whole-trajectory counts to evaluate, strictly increasing.
        threshold: Accuracy the selected count must exceed.
        criterion: Whether the development estimate or its Wilson lower bound must exceed it.
        development_lineages: Held-out lineages reserved for selection.
        registered_at: Registration time, informational only.
        receipts_before: Latest audit receipt sequence, read while receipt minting is held.

    Returns:
        The registration :func:`register_sample_count_study` would save.

    Raises:
        ValueError: A count exceeds the fit traces, a lineage is not held out, or no held-out
            lineage is left untouched for confirmation.
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
        receipts_before=receipts_before,
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
    return registration


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
        sequence=receipt.sequence,
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
        ValueError: A measurement's receipt was minted before registration (its sequence is
            not larger than the registration's ``receipts_before``), reads another memory,
            misses or adds lineages, repeats or leaves the grid, or changes the frozen
            configuration.
    """
    by_count: dict[int, CheckpointMeasurement] = {}
    for row in development:
        if row.sequence <= registration.receipts_before:
            raise ValueError(
                f"development audit {row.receipt_id} (receipt {row.sequence}) predates the "
                f"registration, which followed receipt {registration.receipts_before}; register "
                "first, then audit the grid counts"
            )
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
        last_development_sequence=max((row.sequence for row in development), default=None),
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
        ValueError: No count was selected or the selection was never saved, or the
            confirmation reuses development evidence, is not an independent audit, was minted
            before the saved selection or a development receipt (by receipt sequence), was
            invalidated, or changed the frozen checkpoint or configuration.
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
        selection.last_development_sequence is not None
        and confirmation.sequence <= selection.last_development_sequence
    ):
        raise ValueError(
            f"confirmation must be opened after every development measurement: receipt "
            f"{confirmation.sequence} is not after receipt {selection.last_development_sequence}"
        )
    if selection.receipts_before is None:
        raise ValueError("confirm a saved selection; record the selection first")
    if confirmation.sequence <= selection.receipts_before:
        raise ValueError(
            f"confirmation must be opened after the saved sample-count selection: receipt "
            f"{confirmation.receipt_id} (receipt {confirmation.sequence}) was minted before the "
            f"selection, which followed receipt {selection.receipts_before}; open a new "
            "independent audit after selecting"
        )
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


class SampleCountRegister(ContractModel):
    """Register the procedure before any development measurement is read.

    Attributes:
        resolution: ``register``.
        grid: Whole-trajectory counts to evaluate, strictly increasing.
        threshold: Accuracy the selected count must exceed.
        criterion: ``point_estimate`` or ``wilson_lower_bound``.
        development_lineages: Held-out lineages reserved for selection; the rest stay
            untouched for confirmation.
    """

    resolution: Literal["register"]
    grid: tuple[int, ...] = Field(min_length=1)
    threshold: float = Field(gt=0, lt=1, allow_inf_nan=False)
    criterion: Literal["point_estimate", "wilson_lower_bound"]
    development_lineages: tuple[str, ...] = Field(min_length=1)


class SampleCountSelect(ContractModel):
    """Select a count from development audits of the registered grid.

    Attributes:
        resolution: ``select``.
        registration: The saved registration.
        development: Audit receipt IDs, one per measured grid count, each over exactly the
            development lineages.
    """

    resolution: Literal["select"]
    registration: ArtifactInput
    development: tuple[str, ...] = Field(min_length=1)


class SampleCountConfirm(ContractModel):
    """Confirm the frozen selected checkpoint on one independent audit of untouched lineages.

    Attributes:
        resolution: ``confirm``.
        selection: The saved development selection.
        confirmation: Audit receipt ID of the independent confirmation audit.
    """

    resolution: Literal["confirm"]
    selection: ArtifactInput
    confirmation: str = Field(min_length=1)


SampleCountRequest = Annotated[
    SampleCountRegister | SampleCountSelect | SampleCountConfirm,
    Field(discriminator="resolution"),
]


def register_sample_count(
    project: ProjectStore,
    request: SampleCountRegister,
    *,
    registered_at: datetime,
    code_revision: str,
    dry_run: bool = False,
) -> tuple[SampleCountRegistration, ArtifactInput | None]:
    """Register the procedure once; the same procedure registered again replays the first.

    A registration's receipt sequence decides which audits may count, so registering an
    unchanged procedure again keeps the earlier registration instead of moving it past results.

    Args:
        project: Project with a completed build and a recorded fit-only learning plan.
        request: Grid, threshold, criterion and development lineages.
        registered_at: Registration time of a new registration, informational only.
        code_revision: Producing implementation identity.
        dry_run: Validate and return the registration without saving it.

    Returns:
        The registration and its artifact (None on a dry run of a new registration).
    """
    with receipts_held(project) as receipts_before:
        planned = plan_sample_count_study(
            project,
            grid=request.grid,
            threshold=request.threshold,
            criterion=request.criterion,
            development_lineages=request.development_lineages,
            registered_at=registered_at,
            receipts_before=receipts_before,
        )
    order_fields = {"registered_at", "receipts_before"}
    procedure = planned.model_dump(exclude=order_fields)
    for identity in project.artifacts.list_ids(artifact_type=REGISTRATION_TYPE):
        saved = SampleCountRegistration.model_validate_json(
            project.artifacts.read_bytes(identity, "registration.json")
        )
        if saved.model_dump(exclude=order_fields) == procedure:
            return saved, artifact_input(project.artifacts.read(identity).manifest)
    if dry_run:
        return planned, None
    return register_sample_count_study(
        project,
        grid=planned.grid,
        threshold=planned.threshold,
        criterion=planned.criterion,
        development_lineages=planned.development_lineages,
        registered_at=registered_at,
        code_revision=code_revision,
    )


def receipt_measurement(project: ProjectStore, receipt_id: str) -> CheckpointMeasurement:
    """Measure one saved memory-profile audit receipt from its saved results, without calls.

    Args:
        project: Project holding the receipt, its registrations and their saved reports.
        receipt_id: Audit receipt identity.

    Returns:
        The receipt's checkpoint measurement; traces without a saved result stay unmeasured.
    """
    receipt = find_receipt(project, receipt_id)
    panel = receipt_panel(project, receipt)
    return audit_measurement(
        receipt_view(project, receipt), panel, saved_held_out_results(project, panel)
    )


def record_sample_count_selection(
    project: ProjectStore,
    request: SampleCountSelect,
    *,
    created_at: datetime,
    code_revision: str,
    dry_run: bool = False,
) -> tuple[SampleCountSelection, ArtifactInput | None]:
    """Select on the named development audits and save the decision with its measurements.

    The selection is saved while receipt minting is held, with the latest receipt sequence at
    that moment, and cites its registration and every development receipt.

    Args:
        project: Project holding the registration and the audit receipts.
        request: Registration reference and development receipt IDs.
        created_at: Record time, informational only.
        code_revision: Producing implementation identity.
        dry_run: Select without saving.

    Returns:
        The selection and its artifact (None on a dry run of a new selection).

    Raises:
        ValueError: The registration is not the saved one, a receipt is not admissible
            development evidence, or a different count was already selected.
    """
    registration = _load(project, request.registration, REGISTRATION_TYPE, "registration.json")
    registered = SampleCountRegistration.model_validate_json(registration)
    measurements = tuple(receipt_measurement(project, row) for row in request.development)
    receipts = tuple(receipt_reference(project, row) for row in request.development)
    selected = select_sample_count(registered, measurements)
    identity = selected.identity
    with receipts_held(project) as receipts_before:
        for saved_id in project.artifacts.list_ids(artifact_type=SELECTION_TYPE):
            saved = SampleCountSelection.model_validate_json(
                project.artifacts.read_bytes(saved_id, "selection.json")
            )
            if (
                saved_id != identity
                and saved.registration == registered.identity
                and saved.status == "selected"
            ):
                raise ValueError(
                    f"registration {registered.identity} already selected "
                    f"{saved.selected_count} trajectories ({saved_id}); confirm that selection, "
                    "or register a new procedure"
                )
        if project.artifacts.exists(identity):
            saved = SampleCountSelection.model_validate_json(
                project.artifacts.read_bytes(identity, "selection.json")
            )
            return saved, artifact_input(project.artifacts.read(identity).manifest)
        selection = selected.model_copy(update={"receipts_before": receipts_before})
        if dry_run:
            return selection, None
        # Written while receipt minting is held, so every later receipt has a larger sequence.
        manifest = project.artifacts.write_json(
            artifact_id=identity,
            artifact_type=SELECTION_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=created_at,
                code_revision=code_revision,
                inputs=tuple(
                    sorted({request.registration, *receipts}, key=lambda item: item.artifact_id)
                ),
            ),
            files={
                "selection.json": selection.model_dump(mode="json"),
                "measurements.json": [row.model_dump(mode="json") for row in measurements],
            },
        )
    return selection, artifact_input(manifest)


def record_sample_count_confirmation(
    project: ProjectStore,
    request: SampleCountConfirm,
    *,
    created_at: datetime,
    code_revision: str,
    dry_run: bool = False,
) -> tuple[JsonObject, ArtifactInput | None]:
    """Confirm a saved selection on one independent audit and save the report.

    A selection is confirmed on one receipt only, fixed before any score is read: the first
    independent audit receipt of the frozen checkpoint minted after the saved selection.
    Trying further confirmation audits, or previewing several with dry runs, until one passes
    would be another search.

    Args:
        project: Project holding the selection and the confirmation receipt.
        request: Selection reference and confirmation receipt ID.
        created_at: Record time, informational only.
        code_revision: Producing implementation identity.
        dry_run: Report without saving.

    Returns:
        The confirmation report and its artifact (None on a dry run of a new confirmation).

    Raises:
        ValueError: The selection is not saved, the audit receipt was minted before the saved
            selection (by receipt sequence, whatever its clock claims), or another independent
            audit of the frozen checkpoint was minted after the selection and before it.
    """
    selection = SampleCountSelection.model_validate_json(
        _load(project, request.selection, SELECTION_TYPE, "selection.json")
    )
    selection_manifest = project.artifacts.read(request.selection.artifact_id).manifest
    registration_ref = next(
        (item for item in selection_manifest.inputs if item.artifact_id == selection.registration),
        None,
    )
    if registration_ref is None:
        raise ValueError(f"{request.selection.artifact_id} does not cite its registration")
    registration = SampleCountRegistration.model_validate_json(
        _load(project, registration_ref, REGISTRATION_TYPE, "registration.json")
    )
    if selection.receipts_before is None:
        raise ValueError("confirm a saved selection; record the selection first")
    # Bound by receipt order before any score is computed, so neither a dry run nor a further
    # audit can pick a passing confirmation among several.
    first = next(
        (
            row.receipt_id
            for row in audit_receipts(project)
            if row.sequence > selection.receipts_before
            and row.basis == "independent"
            and row.checkpoint is not None
            and row.checkpoint.lineage == registration.memory_lineage
            and row.checkpoint.version == selection.memory_version
        ),
        None,
    )
    if first is not None and first != request.confirmation:
        raise ValueError(
            f"selection {request.selection.artifact_id} is confirmed only on the first "
            f"independent audit of its frozen checkpoint opened after it ({first}); a selection "
            "is confirmed once, so register a new procedure instead of auditing again"
        )
    measurement = receipt_measurement(project, request.confirmation)
    receipt = receipt_reference(project, request.confirmation)
    report = confirm_sample_count(registration, selection, measurement)
    identity = stable_id(
        CONFIRMATION_TYPE, {"report": sha256_json(report), "protocol": SAMPLE_COUNT_PROTOCOL}
    )
    if project.artifacts.exists(identity):
        return report, artifact_input(project.artifacts.read(identity).manifest)
    if dry_run:
        return report, None
    manifest = project.artifacts.write_json(
        artifact_id=identity,
        artifact_type=CONFIRMATION_TYPE,
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=created_at,
            code_revision=code_revision,
            inputs=tuple(sorted((request.selection, receipt), key=lambda item: item.artifact_id)),
        ),
        files={"confirmation.json": report},
    )
    return report, artifact_input(manifest)


def _load(project: ProjectStore, reference: ArtifactInput, kind: str, path: str) -> bytes:
    """Read one saved file of an exact artifact of the expected type.

    Raises:
        ValueError: The artifact is missing, of another type, or differs from its reference.
    """
    if project.artifacts.exists(reference.artifact_id):
        manifest = project.artifacts.read(reference.artifact_id).manifest
        if manifest.artifact_type == kind and artifact_input(manifest) == reference:
            return project.artifacts.read_bytes(reference.artifact_id, path)
    raise ValueError(f"{reference.artifact_id} is not the saved {kind} it names")


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
