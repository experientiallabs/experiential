"""Fit a decision-cascade threshold on the main model's own development trajectories.

The procedure is registered before any decider answer is read:

1. Register the decider, the explicit threshold grid, the hard restrictions, the development
   lineages and the selection objective with its non-inferiority margins. There is no default
   threshold or grid.
2. Label the reached states: every main-model decision in the saved development studies, with
   its trajectory's verified outcome (``decision_points``).
3. Ask the decider once per distinct reached state (a shadow pass). Each answer is a durable
   receipt; an interrupted dispatch stays unknown and never redraws. As at runtime, where an
   unknown or failed decider dispatch stops the episode, a trajectory reaching a state without
   an answer stops there and counts as a task failure and a policy violation.
4. Estimate each grid threshold offline. A trajectory keeps its verified outcome unless the
   cascade would have accepted a decision differing from the main model's. From that reached
   state the cascade follows the decider, so the trajectories that took the decider's decision
   from the same state of the same checkpoint (continuation branches reach shared states) stand
   in for what would have followed. Without such an observed continuation the trajectory counts
   as a task failure and a policy violation, because what would have followed is not observed;
   an unknown donor outcome counts the same way, and an unknown main outcome of a trajectory the
   cascade changes counts as a main success.
5. Select by the registered objective: the largest delegated share whose task success and
   policy compliance stay within their margins of the main model alone, compared per lineage.
   Ties go to the higher threshold. The selected threshold becomes a deployable
   ``CascadePolicy``; none is produced when no threshold qualifies or none delegates anything.

The fit is development evidence only: the offline estimate cannot see states the cascade would
reach after a delegated decision. ``cascade_confirmation`` runs the frozen policy in episodes on
held-out lineages before anyone may claim it is non-inferior.
"""

from collections import defaultdict
from collections.abc import Collection, Mapping
from datetime import UTC, datetime
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
from exp.common.core.locks import CoordinationLock
from exp.common.evaluations.uncertainty import unit_difference_summary, unit_mean_summary
from exp.common.models import ModelRequest, ModelSnapshot, load_model_catalog
from exp.common.project import ArtifactStore, ProjectStore, artifact_input
from exp.optimize.evaluation.analysis.decision_points import (
    DecisionPointSet,
    extract_decision_points,
    state_request,
)
from exp.optimize.evaluation.continuations.deciders import (
    DecisionReceipt,
    RecordedDecider,
    admitted_decider,
    preflight_decider,
    saved_decision,
)
from exp.optimize.evaluation.continuations.job import MAXIMUM_PHYSICAL_ATTEMPTS
from exp.runtime.agents.cascade import (
    CascadePolicy,
    DeciderDefinition,
    NextActionDecision,
    action_projection,
    admitted_draft,
    decision_candidates,
    delegates,
)
from exp.runtime.agents.cascade_deciders import DECIDER_PROTOCOL
from exp.runtime.models.providers.admitted_transport import AdmittedJsonTransport
from exp.runtime.models.providers.transport import HttpxJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.runtime.sessions.dispatch_ledger import DispatchLedger

# Identity of the estimator and selection rule below, in place of their source text. Bump it
# when either changes; exp/tests/code_protocols_test.py records the covered source.
CASCADE_FIT_PROTOCOL = "decision-cascade-fit-v6"
FIT_SCOPE = (
    "development fit on the main model's own trajectories; offline estimate splicing observed "
    "continuations at accepted disagreements and failing a trajectory without one; states the "
    "cascade reaches that no main trajectory reached are unobserved, so held-out closed-loop "
    "confirmation is required"
)


class CascadeObjective(ContractModel):
    """The registered selection objective, fixed before any decider answer is read.

    Attributes:
        kind: Maximize the delegated share of decisions subject to non-inferior task success
            and policy compliance against the main model alone.
        success_margin: Largest tolerated drop in the lineage-mean task success rate.
        compliance_margin: Largest tolerated drop in the lineage-mean policy compliance rate.
        criterion: ``point_estimate`` compares the paired lineage-mean difference;
            ``wilson_lower_bound`` compares the lower end of its two-sided 95 percent Wilson
            interval at the lineage count.
        minimum_lineages: Paired development lineages required before any threshold qualifies.
    """

    kind: Literal["max_delegation_noninferior"] = "max_delegation_noninferior"
    success_margin: float = Field(gt=0, lt=1, allow_inf_nan=False)
    compliance_margin: float = Field(ge=0, lt=1, allow_inf_nan=False)
    criterion: Literal["point_estimate", "wilson_lower_bound"]
    minimum_lineages: int = Field(ge=1, strict=True)

    def non_inferior(self, difference: JsonObject, margin: float) -> bool:
        """Apply the registered criterion to one paired ``unit_difference_summary``."""
        units, estimate, interval = (
            difference["independent_units"],
            difference["estimate"],
            difference["wilson_95"],
        )
        if not isinstance(units, int) or units < self.minimum_lineages:
            return False
        if self.criterion == "point_estimate":
            return isinstance(estimate, float | int) and estimate >= -margin
        return (
            isinstance(interval, list)
            and isinstance(interval[0], float | int)
            and interval[0] >= -margin
        )


class CascadeFitSpec(ContractModel):
    """Everything the fit binds before the shadow pass.

    Attributes:
        sources: Complete saved continuation studies or actor-panel reports of the main model.
        main_model: The main model whose own decisions are labelled.
        development_lineages: Exactly the lineages the sources cover, sorted; confirmation must
            use none of them.
        decider: Exact cheap decider asked at every reached state.
        never_delegate: Hard restrictions carried into the policy, sorted.
        grid: Explicit candidate thresholds in increasing order; nothing else is considered.
        objective: Registered selection objective and margins.
        catalog_sha256: Frozen credential-free catalog configuration.
        maximum_physical_attempts: Decider attempt ceiling for the shadow pass.
        deadline: Aware deadline after which no new decider attempt starts.
    """

    sources: tuple[ArtifactInput, ...] = Field(min_length=1, max_length=1000)
    main_model: ModelSnapshot
    development_lineages: tuple[str, ...] = Field(min_length=1)
    decider: DeciderDefinition
    never_delegate: tuple[str, ...] = ()
    grid: tuple[float, ...] = Field(min_length=1, max_length=101)
    objective: CascadeObjective
    catalog_sha256: Sha256
    maximum_physical_attempts: int = Field(gt=0, le=MAXIMUM_PHYSICAL_ATTEMPTS)
    deadline: AwareDatetime

    @model_validator(mode="after")
    def registered(self) -> Self:
        """Require a canonical lineage set and a strictly increasing grid within [0, 1]."""
        if self.development_lineages != tuple(sorted(set(self.development_lineages))):
            raise ValueError("development lineages must be sorted and unique")
        if any(not 0 <= value <= 1 for value in self.grid) or any(
            right <= left for left, right in zip(self.grid, self.grid[1:], strict=False)
        ):
            raise ValueError("the threshold grid must increase strictly within [0, 1]")
        CascadePolicy(decider=self.decider, threshold=0, never_delegate=self.never_delegate)
        return self

    def identity(self, code_revision: str) -> str:
        """Bind the registration, both protocols and the implementation."""
        return stable_id(
            "decision-cascade-fit",
            {
                "spec": sha256_json(self),
                "protocol": CASCADE_FIT_PROTOCOL,
                "decider_protocol": DECIDER_PROTOCOL,
                "code": code_revision,
            },
        )


class ShadowPrediction(ContractModel):
    """The decider's answer at one distinct reached state.

    Attributes:
        state_sha256: Reached-state identity shared by every point at that state.
        decision: Decider answer, absent when unknown or failed.
        failure: Failure or unknown-outcome code when there is no answer.
    """

    state_sha256: Sha256
    decision: NextActionDecision | None = None
    failure: str | None = None


class ThresholdRow(ContractModel):
    """The conservative offline estimate for one grid threshold.

    Attributes:
        threshold: Grid value.
        decisions: Every decision point.
        delegated: Points the policy would delegate, excluding points after a stop.
        calls_avoided: Delegated points answered with a drafted action, so no main call.
        delegated_share: Delegated over every point.
        agreement: Delegated points matching the main model's decision (and complete action
            when drafted), over delegated points; None without delegation.
        verified_correct: Delegated decisions verified correct at their state.
        verified_error: Delegated decisions verified wrong at their state.
        affected_trajectories: Trajectories with at least one delegated disagreement.
        stopped_trajectories: Trajectories that reach a state without a decider answer before
            any delegated disagreement; the cascade would stop there, so each counts as a task
            failure and a policy violation.
        success: Paired lineage difference, cascade estimate minus main, of task success.
        compliance: Paired lineage difference, cascade minus main, of policy compliance.
        feasible: Whether both differences meet the registered objective.
    """

    threshold: float
    decisions: int
    delegated: int
    calls_avoided: int
    delegated_share: float
    agreement: float | None
    verified_correct: int
    verified_error: int
    affected_trajectories: int
    stopped_trajectories: int
    success: JsonObject
    compliance: JsonObject
    feasible: bool


class CascadeFitReport(ContractModel):
    """The complete development fit and, when one qualifies, its deployable policy.

    Attributes:
        spec: Registration.
        points: Every labelled decision point and trajectory read.
        predictions: The decider's answer at every distinct state, unknowns included.
        main_success: Lineage-mean task success of the main model alone.
        main_compliance: Lineage-mean policy compliance of the main model alone.
        rows: The estimate at every grid threshold, in grid order.
        status: ``selected``, ``no_feasible_threshold`` or ``no_delegation``.
        selected: Selected threshold, when one qualified and delegates.
        policy: The deployable cascade policy at that threshold.
        scope: Interpretation limits.
    """

    spec: CascadeFitSpec
    points: DecisionPointSet
    predictions: tuple[ShadowPrediction, ...]
    main_success: JsonObject
    main_compliance: JsonObject
    rows: tuple[ThresholdRow, ...]
    status: Literal["selected", "no_feasible_threshold", "no_delegation"]
    selected: float | None = None
    policy: CascadePolicy | None = None
    scope: str = FIT_SCOPE


def _known(verdict: str) -> float | None:
    """Return 1 for a pass, 0 for a failure and None for an unknown outcome."""
    return None if verdict == "unassessable" else float(verdict == "pass")


Trajectory = tuple[str, int]


def estimate_threshold(
    points: DecisionPointSet,
    predictions: Mapping[Sha256, NextActionDecision | None],
    threshold: float,
    spec: CascadeFitSpec,
    refused_drafts: Collection[Sha256],
) -> ThresholdRow:
    """Estimate one threshold by splicing observed continuations at accepted disagreements.

    A trajectory with no accepted disagreement keeps its verified outcome. At its first
    accepted disagreement the cascade would have taken the decider's decision from that reached
    state; the trajectories that took the same decision (or, for a draft, the same complete
    action) from the same state of the same checkpoint are its observed continuations, and their
    estimated outcomes are averaged in its place. Without any such continuation it counts as a
    task failure and a policy violation. Unknown outcomes stay bounds adverse to the cascade: an
    unknown donor counts as a failure, and an unknown main outcome of a trajectory the cascade
    would change counts as a main success. Splicing always moves to a later disagreement, so the
    recursion ends.

    The runtime is mirrored at every state. A state without a decider answer (an unknown or
    failed shadow dispatch) stops a trajectory still on the main path, as the deployed cascade
    stops the episode, and the stopped trajectory counts as a failure and contributes no
    delegation at its later points, which deployment never reaches; a content error at
    deployment would leave the main model responsible, so this bound is conservative. A drafted
    answer in ``refused_drafts``, which the runtime refuses as an invalid draft, is never
    delegated; callers compute them with ``refused_drafts``.

    Args:
        points: Labelled decision points and trajectories.
        predictions: The decider's answer at every distinct state, None without one.
        threshold: Grid value.
        spec: Registration, for the restrictions and the objective.
        refused_drafts: States whose drafted action ``admitted_draft`` refuses.

    Returns:
        The threshold's delegation counts and paired lineage differences.
    """
    decision_labels = points.labels(by="decision")
    action_labels = points.labels(by="action")
    delegated = avoided = agreed = correct = wrong = 0
    diverges: dict[Trajectory, tuple[str, Sha256, str]] = {}
    stopped: set[Trajectory] = set()
    takes: dict[tuple[str, Sha256, str], list[Trajectory]] = defaultdict(list)
    for point in points.points:
        trajectory = (point.study.artifact_id, point.branch)
        world = point.checkpoint.artifact_id
        takes[world, point.state_sha256, point.main_choice].append(trajectory)
        takes[world, point.state_sha256, point.main_action_sha256].append(trajectory)
        if trajectory in stopped:
            # The deployed episode ended at an earlier state, so this point is never reached.
            continue
        decision = predictions[point.state_sha256]
        if decision is None:
            if trajectory not in diverges:
                stopped.add(trajectory)
            continue
        allowed = tuple(name for name in point.candidates if name not in spec.never_delegate)
        if point.state_sha256 in refused_drafts or not delegates(
            decision.selected, decision.confidence, threshold=threshold, allowed=allowed
        ):
            continue
        delegated += 1
        avoided += decision.action is not None
        if decision.action is None:
            taken, same = decision.selected, decision.selected == point.main_choice
            label = decision_labels.get((world, point.state_sha256, taken), "unverified")
        else:
            taken = action_projection(decision.action)
            same = taken == point.main_action_sha256
            label = action_labels.get((world, point.state_sha256, taken), "unverified")
        agreed += same
        correct += label == "verified_correct"
        wrong += label == "verified_error"
        if not same:
            diverges.setdefault(trajectory, (world, point.state_sha256, taken))
    verdicts = {
        (row.study.artifact_id, row.branch): {"success": row.outcome, "compliance": row.policy}
        for row in points.trajectories
    }
    memo: dict[tuple[Trajectory, str], float] = {}

    def spliced(trajectory: Trajectory, name: str) -> float:
        """Return the cascade estimate of one trajectory, an unknown outcome counted as failed.

        Counting an unknown donor, or a stopped trajectory, as a failure is the bound adverse to
        the cascade.
        """
        if (trajectory, name) not in memo:
            if trajectory in stopped:
                memo[trajectory, name] = 0.0
            elif trajectory not in diverges:
                observed = _known(verdicts[trajectory][name])
                memo[trajectory, name] = 0.0 if observed is None else observed
            else:
                donors = [spliced(row, name) for row in takes.get(diverges[trajectory], [])]
                memo[trajectory, name] = sum(donors) / len(donors) if donors else 0.0
        return memo[trajectory, name]

    main: dict[str, dict[str, list[float]]] = {"success": {}, "compliance": {}}
    cascade: dict[str, dict[str, list[float]]] = {"success": {}, "compliance": {}}
    for row in points.trajectories:
        trajectory = (row.study.artifact_id, row.branch)
        for name in ("success", "compliance"):
            observed = _known(verdicts[trajectory][name])
            main[name].setdefault(row.lineage, [])
            cascade[name].setdefault(row.lineage, [])
            if trajectory in diverges or trajectory in stopped:
                # An unknown main outcome is bounded against the cascade, as in confirmation.
                main[name][row.lineage].append(1.0 if observed is None else observed)
                cascade[name][row.lineage].append(spliced(trajectory, name))
            elif observed is not None:
                # Without a delegated disagreement both arms share the observed outcome; an
                # unknown one carries no paired evidence and is left out of both.
                main[name][row.lineage].append(observed)
                cascade[name][row.lineage].append(observed)
    success = unit_difference_summary(cascade["success"], main["success"])
    compliance = unit_difference_summary(cascade["compliance"], main["compliance"])
    count = len(points.points)
    return ThresholdRow(
        threshold=threshold,
        decisions=count,
        delegated=delegated,
        calls_avoided=avoided,
        delegated_share=delegated / count if count else 0.0,
        agreement=agreed / delegated if delegated else None,
        verified_correct=correct,
        verified_error=wrong,
        affected_trajectories=len(diverges),
        stopped_trajectories=len(stopped),
        success=success,
        compliance=compliance,
        feasible=spec.objective.non_inferior(success, spec.objective.success_margin)
        and spec.objective.non_inferior(compliance, spec.objective.compliance_margin),
    )


def select_threshold(
    rows: tuple[ThresholdRow, ...],
) -> tuple[Literal["selected", "no_feasible_threshold", "no_delegation"], float | None]:
    """Take the largest feasible delegated share, preferring the higher threshold on ties."""
    feasible = [row for row in rows if row.feasible]
    if not feasible:
        return "no_feasible_threshold", None
    best = max(feasible, key=lambda row: (row.delegated_share, row.threshold))
    if best.delegated == 0:
        return "no_delegation", None
    return "selected", best.threshold


def preflight_cascade_fit(project: ProjectStore, spec: CascadeFitSpec) -> DecisionPointSet:
    """Label every reached state and check the decider without credentials or dispatch.

    Raises:
        ValueError: Sources, lineages, catalog, decider or attempt allowance are inconsistent.
    """
    points = extract_decision_points(project.artifacts, spec.sources, main_model=spec.main_model)
    if points.lineages != spec.development_lineages:
        raise ValueError(
            "development_lineages must equal the lineages the fit sources cover: "
            + ", ".join(points.lineages)
        )
    if not points.points:
        raise ValueError("fit sources contain no main-model decision to label")
    catalog = load_model_catalog(project.model_catalog_path)
    if sha256_json(catalog.model_dump(mode="json")) != spec.catalog_sha256:
        raise ValueError("model catalog changed; register the fit with its current digest")
    preflight_decider(catalog, spec.decider)
    if len(points.states()) * spec.decider.samples > spec.maximum_physical_attempts:
        raise ValueError(
            f"{len(points.states())} reached states need more decider attempts than "
            f"maximum_physical_attempts={spec.maximum_physical_attempts}"
        )
    return points


def _state_keys(spec: CascadeFitSpec, points: DecisionPointSet, revision: str) -> list[str]:
    """Return the receipt key of every distinct reached state, in first-seen order."""
    identity = spec.identity(revision)
    return [f"{identity}-state-{index}" for index in range(len(points.states()))]


def cascade_fit_needs_dispatch(
    project: ProjectStore, spec: CascadeFitSpec, points: DecisionPointSet, *, code_revision: str
) -> bool:
    """Whether any distinct state still lacks a decider receipt or intent."""
    if datetime.now(UTC) >= spec.deadline:
        return False
    return any(
        saved_decision(project.artifacts, key) is None
        for key in _state_keys(spec, points, code_revision)
    )


def saved_cascade_fit(
    project: ProjectStore, spec: CascadeFitSpec, *, code_revision: str
) -> CascadeFitReport | None:
    """Read and re-derive a completed fit without credentials or provider calls."""
    identity = spec.identity(code_revision)
    if not project.artifacts.exists(identity + "-result"):
        return None
    report = CascadeFitReport.model_validate_json(
        project.artifacts.read_bytes(identity + "-result", "report.json")
    )
    if report.spec != spec or (
        _derive(project.artifacts, spec, report.points, report.predictions) != report
    ):
        raise ValueError("saved cascade fit does not reproduce from its registration")
    return report


def load_cascade_fit(project: ProjectStore, reference: ArtifactInput) -> CascadeFitReport:
    """Read a fit result by reference and verify it reproduces from its own registration."""
    manifest = project.artifacts.read(reference.artifact_id).manifest
    if artifact_input(manifest) != reference or manifest.artifact_type != "decision-cascade-fit":
        raise ValueError("cascade fit reference differs from its immutable result")
    report = CascadeFitReport.model_validate_json(
        project.artifacts.read_bytes(reference.artifact_id, "report.json")
    )
    if report.spec.identity(manifest.code_revision) + "-result" != reference.artifact_id or (
        _derive(project.artifacts, report.spec, report.points, report.predictions) != report
    ):
        raise ValueError("cascade fit does not reproduce from its registration")
    return report


def refused_drafts(
    store: ArtifactStore,
    spec: CascadeFitSpec,
    points: DecisionPointSet,
    predictions: tuple[ShadowPrediction, ...],
) -> frozenset[Sha256]:
    """Return the states whose drafted answer the runtime would refuse as an invalid draft.

    Each draft is checked by the runtime's own ``admitted_draft`` against the saved request that
    first reached its state, so visibility, argument schemas and ``never_delegate`` on every
    call decide delegation exactly as in deployment.
    """
    calls = points.states()
    refused = set()
    for row in predictions:
        if row.decision is None or row.decision.action is None:
            continue
        request = state_request(
            ModelRequest.model_validate_json(
                store.read_bytes(calls[row.state_sha256].artifact_id, "request.json")
            )
        )
        if not admitted_draft(request, row.decision.action, spec.never_delegate):
            refused.add(row.state_sha256)
    return frozenset(refused)


def _derive(
    store: ArtifactStore,
    spec: CascadeFitSpec,
    points: DecisionPointSet,
    predictions: tuple[ShadowPrediction, ...],
) -> CascadeFitReport:
    """Compute every row, the selection and the policy from labels and saved answers."""
    answers = {row.state_sha256: row.decision for row in predictions}
    refused = refused_drafts(store, spec, points, predictions)
    rows = tuple(estimate_threshold(points, answers, value, spec, refused) for value in spec.grid)
    status, selected = select_threshold(rows)
    main: dict[str, dict[str, list[float]]] = defaultdict(dict)
    for row in points.trajectories:
        for name, verdict in (("success", row.outcome), ("compliance", row.policy)):
            values = main[name].setdefault(row.lineage, [])
            value = _known(verdict)
            if value is not None:
                values.append(value)
    return CascadeFitReport(
        spec=spec,
        points=points,
        predictions=predictions,
        main_success=unit_mean_summary(main["success"]),
        main_compliance=unit_mean_summary(main["compliance"]),
        rows=rows,
        status=status,
        selected=selected,
        policy=None
        if selected is None
        else CascadePolicy(
            decider=spec.decider, threshold=selected, never_delegate=spec.never_delegate
        ),
    )


def run_cascade_fit(
    project: ProjectStore, spec: CascadeFitSpec, *, code_revision: str
) -> CascadeFitReport:
    """Register, ask the decider once per reached state, then estimate and select.

    Returns:
        The saved fit; a completed fit replays without credentials or provider calls.
    """
    spec = CascadeFitSpec.model_validate_json(spec.model_dump_json())
    store, identity = project.artifacts, spec.identity(code_revision)
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        saved = saved_cascade_fit(project, spec, code_revision=code_revision)
        if saved is not None:
            return saved
        points = preflight_cascade_fit(project, spec)
        envelope = ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            inputs=tuple(sorted(points.studies, key=lambda row: row.artifact_id)),
        )
        if not store.exists(identity):
            store.write_json(
                artifact_id=identity,
                artifact_type="decision-cascade-fit-registration",
                envelope=envelope,
                files={"spec.json": spec},
            )
        elif CascadeFitSpec.model_validate_json(store.read_bytes(identity, "spec.json")) != spec:
            raise ValueError("cascade fit registration changed; inspect its immutable evidence")
        registration = artifact_input(store.read(identity).manifest)
        decider: RecordedDecider | None = None
        predictions = []
        for key, (state, call) in zip(
            _state_keys(spec, points, code_revision), points.states().items(), strict=True
        ):
            receipt = saved_decision(store, key)
            if receipt is None and datetime.now(UTC) >= spec.deadline:
                receipt = DecisionReceipt(failure="DeadlineBeforeDispatch")
            if receipt is None:
                decider = decider or _decider(project, spec, registration, code_revision)
                request = state_request(
                    ModelRequest.model_validate_json(
                        store.read_bytes(call.artifact_id, "request.json")
                    )
                )
                try:
                    receipt = DecisionReceipt(
                        decision=decider.decide_at(key, request, decision_candidates(request))
                    )
                except (ValueError, RuntimeError, OSError) as error:
                    receipt = DecisionReceipt(failure=type(error).__name__)
            predictions.append(
                ShadowPrediction(
                    state_sha256=state, decision=receipt.decision, failure=receipt.failure
                )
            )
        report = _derive(store, spec, points, tuple(predictions))
        store.write_json(
            artifact_id=identity + "-result",
            artifact_type="decision-cascade-fit",
            envelope=envelope.model_copy(
                update={
                    "inputs": tuple(
                        sorted({registration, *points.studies}, key=lambda row: row.artifact_id)
                    )
                }
            ),
            files={"report.json": report},
        )
        return report


def _decider(
    project: ProjectStore, spec: CascadeFitSpec, registration: ArtifactInput, revision: str
) -> RecordedDecider:
    """Build the admitted decider for the shadow pass, with one physical attempt per slot."""
    identity = spec.identity(revision)
    transport = AdmittedJsonTransport(
        HttpxJsonTransport(),
        ledger=DispatchLedger(
            project.artifacts.project_directory / (identity + "-physical.sqlite"),
            spec.maximum_physical_attempts,
            sha256_json(identity),
        ),
        artifacts=project.artifacts,
        code_revision=revision,
        deadline=spec.deadline,
    )
    catalog = load_model_catalog(project.model_catalog_path)
    if sha256_json(catalog.model_dump(mode="json")) != spec.catalog_sha256:
        raise ValueError("model catalog changed before the shadow pass")
    decider = admitted_decider(
        RuntimeModelCatalog(catalog, transport_factory=lambda: transport),
        spec.decider,
        transport,
        spec.deadline,
    )
    return RecordedDecider(
        decider,
        store=project.artifacts,
        prefix=identity,
        inputs=(registration,),
        code_revision=revision,
    )
