"""Immutable held-out text-versus-sandbox comparison without provider calls."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ArtifactInput,
    ContractModel,
    FailureCode,
    Sha256,
    StructuredFailure,
    canonical_json_bytes,
    sha256_json,
    sorted_unique_inputs,
    stable_id,
)
from exp.common.evaluations import EvaluationCell, EvaluationPlan
from exp.common.models import ModelSnapshot
from exp.common.project import (
    ArtifactCorruptionError,
    ArtifactStore,
    StoredArtifact,
    artifact_input,
)
from exp.common.rollouts import (
    RolloutArtifact,
    SandboxSimulatorSnapshot,
    SimulationArtifactSet,
    SimulationMode,
    WorldModelSimulatorSnapshot,
)
from exp.common.tasks import LoadedTaskSet, TaskCase, load_task_set
from exp.simulation.specs import SimulationSpec

_COMPARISON_SPEC_FILE = "simulation-comparison-spec.json"
_COMPARISON_REPORT_FILE = "simulation-comparison-report.json"
_PLAN_FILE = "evaluation-plan.json"
_SIMULATION_SPEC_FILE = "simulation-spec.json"
_ARTIFACT_SET_FILE = "artifact-set.json"
_ROLLOUT_FILE = "rollout.json"


class PairedSimulationCell(ContractModel):
    """One exact held-out task coordinate and its two expected simulator artifacts."""

    pair_id: ArtifactId
    task_id: ArtifactId
    task_lineage_group_id: ArtifactId
    task_sha256: Sha256
    candidate_alias: ArtifactId
    candidate: ModelSnapshot
    agent_id: str = Field(min_length=1, max_length=256)
    repeat: int = Field(ge=0)
    text_cell_id: ArtifactId
    sandbox_cell_id: ArtifactId
    text_rollout_id: ArtifactId
    text_rollout_sha256: Sha256 | None = None
    sandbox_rollout_id: ArtifactId
    sandbox_rollout_sha256: Sha256 | None = None
    text_simulator: WorldModelSimulatorSnapshot
    sandbox_simulator: SandboxSimulatorSnapshot


class SimulationComparisonSpec(ArtifactEnvelope):
    """Frozen post-lock comparison protocol over exact plans, specs, and artifact sets."""

    comparison_id: ArtifactId
    policy_lock_input: ArtifactInput
    task_set_input: ArtifactInput
    text_evaluation_plan_input: ArtifactInput
    sandbox_evaluation_plan_input: ArtifactInput
    text_simulation_spec_input: ArtifactInput
    sandbox_simulation_spec_input: ArtifactInput
    text_artifact_set_input: ArtifactInput
    sandbox_artifact_set_input: ArtifactInput
    pairs: tuple[PairedSimulationCell, ...] = Field(min_length=1)

    @field_validator("pairs")
    @classmethod
    def _require_unique_pair_coordinates(
        cls,
        value: tuple[PairedSimulationCell, ...],
    ) -> tuple[PairedSimulationCell, ...]:
        """Reject any repeated pair, text cell, or sandbox cell denominator."""
        for label, values in (
            ("pair IDs", tuple(pair.pair_id for pair in value)),
            ("text cell IDs", tuple(pair.text_cell_id for pair in value)),
            ("sandbox cell IDs", tuple(pair.sandbox_cell_id for pair in value)),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"simulation comparison must have unique {label}")
        return value

    @model_validator(mode="after")
    def _require_exact_inputs(self) -> SimulationComparisonSpec:
        """Make every named immutable dependency a hash-bound envelope input."""
        expected = sorted_unique_inputs(
            self.policy_lock_input,
            self.task_set_input,
            self.text_evaluation_plan_input,
            self.sandbox_evaluation_plan_input,
            self.text_simulation_spec_input,
            self.sandbox_simulation_spec_input,
            self.text_artifact_set_input,
            self.sandbox_artifact_set_input,
        )
        if self.inputs != expected:
            raise ValueError("comparison spec inputs must exactly match every named artifact")
        if self.text_simulation_spec_input == self.sandbox_simulation_spec_input:
            raise ValueError("text and sandbox comparison specs must be distinct")
        if self.text_artifact_set_input == self.sandbox_artifact_set_input:
            raise ValueError("text and sandbox artifact sets must be distinct")
        return self


class PairedSimulationOutcome(ContractModel):
    """One preserved denominator row with complete identity and both failure states."""

    pair_id: ArtifactId
    task_id: ArtifactId
    task_lineage_group_id: ArtifactId
    candidate_alias: ArtifactId
    candidate: ModelSnapshot
    agent_id: str = Field(min_length=1, max_length=256)
    repeat: int = Field(ge=0)
    text_cell_id: ArtifactId
    sandbox_cell_id: ArtifactId
    expected_text_rollout_id: ArtifactId
    expected_sandbox_rollout_id: ArtifactId
    text_rollout_id: ArtifactId | None = None
    sandbox_rollout_id: ArtifactId | None = None
    text_simulator: WorldModelSimulatorSnapshot
    sandbox_simulator: SandboxSimulatorSnapshot
    text_failure: StructuredFailure | None = None
    sandbox_failure: StructuredFailure | None = None
    terminal_match: bool | None = None

    @model_validator(mode="after")
    def _require_complete_pair_state(self) -> PairedSimulationOutcome:
        """Keep missing and failed sides explicit without ambiguous null values."""
        if self.text_rollout_id is None and self.text_failure is None:
            raise ValueError("missing text rollout requires a structured failure")
        if self.sandbox_rollout_id is None and self.sandbox_failure is None:
            raise ValueError("missing sandbox rollout requires a structured failure")
        if self.terminal_match is not None and (
            self.text_rollout_id is None
            or self.sandbox_rollout_id is None
            or self.text_failure is not None
            or self.sandbox_failure is not None
        ):
            raise ValueError("terminal_match requires two usable rollouts")
        return self


class SimulationComparisonReport(ArtifactEnvelope):
    """Structural agreement report with exact missing and failure denominators."""

    report_id: ArtifactId
    comparison_id: ArtifactId
    comparison_spec_sha256: Sha256
    expected_pairs: int = Field(gt=0)
    paired_rollouts: int = Field(ge=0)
    usable_pairs: int = Field(ge=0)
    missing_text_rollouts: int = Field(ge=0)
    missing_sandbox_rollouts: int = Field(ge=0)
    failed_text_rollouts: int = Field(ge=0)
    failed_sandbox_rollouts: int = Field(ge=0)
    terminal_matches: int = Field(ge=0)
    outcomes: tuple[PairedSimulationOutcome, ...]

    @model_validator(mode="after")
    def _require_exact_report_counts(self) -> SimulationComparisonReport:
        """Derive every report count from the preserved denominator rows."""
        if len(self.outcomes) != self.expected_pairs:
            raise ValueError("comparison outcomes must equal expected_pairs")
        if len({outcome.pair_id for outcome in self.outcomes}) != len(self.outcomes):
            raise ValueError("comparison outcomes must have unique pair IDs")
        counts = _outcome_counts(self.outcomes)
        rendered = (
            self.paired_rollouts,
            self.usable_pairs,
            self.missing_text_rollouts,
            self.missing_sandbox_rollouts,
            self.failed_text_rollouts,
            self.failed_sandbox_rollouts,
            self.terminal_matches,
        )
        if rendered != counts:
            raise ValueError("comparison report counts do not match raw outcomes")
        return self


class SimulationComparisonError(ValueError):
    """An immutable input is not one exact post-lock held-out comparison protocol."""


@dataclass(frozen=True)
class _ResolvedInputs:
    """Typed, digest-verified comparison dependencies loaded from the artifact store."""

    comparison_spec_input: ArtifactInput
    lock_created_at: datetime
    task_set_input: ArtifactInput
    text_plan_input: ArtifactInput
    sandbox_plan_input: ArtifactInput
    text_spec_input: ArtifactInput
    sandbox_spec_input: ArtifactInput
    tasks: Mapping[str, TaskCase]
    task_set: LoadedTaskSet
    text_plan: EvaluationPlan
    sandbox_plan: EvaluationPlan
    text_spec: SimulationSpec
    sandbox_spec: SimulationSpec
    text_set: SimulationArtifactSet
    sandbox_set: SimulationArtifactSet


def persist_comparison_spec(
    spec: SimulationComparisonSpec,
    store: ArtifactStore,
) -> ArtifactInput:
    """Persist the frozen protocol before resolving any rollout evidence.

    Args:
        spec: Complete hash-bound post-lock comparison protocol.
        store: Project-local immutable artifact store.

    Returns:
        Exact manifest identity required by the report.
    """
    manifest = store.write_json(
        artifact_id=spec.comparison_id,
        artifact_type="simulation-comparison-spec",
        envelope=spec,
        files={_COMPARISON_SPEC_FILE: spec},
    )
    return artifact_input(manifest)


def compare_text_and_sandbox(
    spec: SimulationComparisonSpec,
    *,
    store: ArtifactStore,
    created_at: datetime,
    code_revision: str,
) -> SimulationComparisonReport:
    """Resolve a frozen comparison using only immutable local artifacts.

    The function accepts no model client, provider, environment, judge, or mutable task mapping.
    It can therefore report structural agreement only after the named policy lock.

    Args:
        spec: Already-persisted comparison protocol.
        store: Store containing every exact input and available rollout.
        created_at: Aware report completion time after the comparison protocol was frozen.
        code_revision: Exact source revision producing the report.

    Returns:
        Exact-denominator report over available and missing rollout evidence.

    Raises:
        SimulationComparisonError: Any hash, mode, lineage, lock, plan, or rollout binding drifts.
    """
    resolved = _resolve_inputs(spec, store)
    if created_at < spec.created_at:
        raise SimulationComparisonError("comparison report cannot predate its frozen protocol")
    text_ids = set(resolved.text_set.artifact_ids)
    sandbox_ids = set(resolved.sandbox_set.artifact_ids)
    outcomes = tuple(
        _pair_outcome(pair, resolved, store, text_ids, sandbox_ids) for pair in spec.pairs
    )
    report_id = stable_id(
        "simulation-comparison-report",
        {
            "comparison_spec_input": resolved.comparison_spec_input.model_dump(mode="json"),
            "outcomes": [outcome.model_dump(mode="json") for outcome in outcomes],
        },
    )
    counts = _outcome_counts(outcomes)
    return SimulationComparisonReport(
        schema_version=1,
        created_at=created_at,
        inputs=sorted_unique_inputs(resolved.comparison_spec_input, *spec.inputs),
        code_revision=code_revision,
        source=spec.source,
        report_id=report_id,
        comparison_id=spec.comparison_id,
        comparison_spec_sha256=sha256_json(spec),
        expected_pairs=len(spec.pairs),
        paired_rollouts=counts[0],
        usable_pairs=counts[1],
        missing_text_rollouts=counts[2],
        missing_sandbox_rollouts=counts[3],
        failed_text_rollouts=counts[4],
        failed_sandbox_rollouts=counts[5],
        terminal_matches=counts[6],
        outcomes=outcomes,
    )


def persist_comparison(report: SimulationComparisonReport, store: ArtifactStore) -> None:
    """Write one immutable report without expanding raw task text.

    Args:
        report: Fully resolved structural comparison output.
        store: Project-local immutable artifact store.
    """
    store.write(
        artifact_id=report.report_id,
        artifact_type="simulation-comparison-report",
        envelope=report,
        files={_COMPARISON_REPORT_FILE: canonical_json_bytes(report)},
    )


def _resolve_inputs(spec: SimulationComparisonSpec, store: ArtifactStore) -> _ResolvedInputs:
    """Load and cross-check every exact named protocol dependency."""
    comparison_stored = _read_exact(
        store,
        ArtifactInput(
            artifact_id=spec.comparison_id, sha256=_manifest_digest(store, spec.comparison_id)
        ),
        "simulation-comparison-spec",
    )
    try:
        persisted = SimulationComparisonSpec.model_validate_json(
            store.read_bytes(spec.comparison_id, _COMPARISON_SPEC_FILE)
        )
    except (ArtifactCorruptionError, ValueError) as exc:
        raise SimulationComparisonError("persisted comparison protocol is invalid") from exc
    if persisted != spec:
        raise SimulationComparisonError("supplied comparison protocol differs from stored bytes")
    comparison_input = artifact_input(comparison_stored.manifest)
    lock = _read_exact(store, spec.policy_lock_input)
    task_set_stored = _read_exact(store, spec.task_set_input, "task-set")
    del task_set_stored
    try:
        task_set = load_task_set(store, spec.task_set_input.artifact_id)
    except ArtifactCorruptionError as exc:
        raise SimulationComparisonError("comparison task set is invalid") from exc
    text_plan = _load_plan(store, spec.text_evaluation_plan_input)
    sandbox_plan = _load_plan(store, spec.sandbox_evaluation_plan_input)
    text_spec = _load_simulation_spec(store, spec.text_simulation_spec_input)
    sandbox_spec = _load_simulation_spec(store, spec.sandbox_simulation_spec_input)
    text_set = _load_artifact_set(store, spec.text_artifact_set_input)
    sandbox_set = _load_artifact_set(store, spec.sandbox_artifact_set_input)
    if spec.created_at < lock.manifest.created_at:
        raise SimulationComparisonError("comparison protocol must be frozen after the policy lock")
    _validate_root_bindings(
        spec,
        task_set,
        text_plan,
        sandbox_plan,
        text_spec,
        sandbox_spec,
        text_set,
        sandbox_set,
        lock.manifest.created_at,
    )
    tasks = {task.task_id: task for task in task_set.tasks}
    return _ResolvedInputs(
        comparison_spec_input=comparison_input,
        lock_created_at=lock.manifest.created_at,
        task_set_input=spec.task_set_input,
        text_plan_input=spec.text_evaluation_plan_input,
        sandbox_plan_input=spec.sandbox_evaluation_plan_input,
        text_spec_input=spec.text_simulation_spec_input,
        sandbox_spec_input=spec.sandbox_simulation_spec_input,
        tasks=tasks,
        task_set=task_set,
        text_plan=text_plan,
        sandbox_plan=sandbox_plan,
        text_spec=text_spec,
        sandbox_spec=sandbox_spec,
        text_set=text_set,
        sandbox_set=sandbox_set,
    )


def _validate_root_bindings(
    comparison: SimulationComparisonSpec,
    task_set: LoadedTaskSet,
    text_plan: EvaluationPlan,
    sandbox_plan: EvaluationPlan,
    text_spec: SimulationSpec,
    sandbox_spec: SimulationSpec,
    text_set: SimulationArtifactSet,
    sandbox_set: SimulationArtifactSet,
    locked_at: datetime,
) -> None:
    """Verify modes, task set, plans, specs, sets, and post-lock timestamps."""
    if text_plan.task_set_id != task_set.task_set.task_set_id:
        raise SimulationComparisonError("text plan names a different task set")
    if sandbox_plan.task_set_id != task_set.task_set.task_set_id:
        raise SimulationComparisonError("sandbox plan names a different task set")
    for label, plan, plan_input, simulation, simulation_input, artifact_set in (
        (
            "text",
            text_plan,
            comparison.text_evaluation_plan_input,
            text_spec,
            comparison.text_simulation_spec_input,
            text_set,
        ),
        (
            "sandbox",
            sandbox_plan,
            comparison.sandbox_evaluation_plan_input,
            sandbox_spec,
            comparison.sandbox_simulation_spec_input,
            sandbox_set,
        ),
    ):
        expected_mode = SimulationMode.WORLD_MODEL if label == "text" else SimulationMode.SANDBOX
        if simulation.mode is not expected_mode:
            raise SimulationComparisonError(f"{label} simulation spec has the wrong mode")
        if simulation.evaluation_plan_id != plan.plan_id:
            raise SimulationComparisonError(f"{label} simulation spec names a different plan")
        if (
            plan_input not in simulation.inputs
            or comparison.task_set_input not in simulation.inputs
        ):
            raise SimulationComparisonError(f"{label} simulation spec omits an exact input")
        if artifact_set.simulation_id != simulation.simulation_id:
            raise SimulationComparisonError(f"{label} artifact set names a different simulation")
        required_set_inputs = {plan_input, comparison.task_set_input, simulation_input}
        if not required_set_inputs.issubset(set(artifact_set.inputs)):
            raise SimulationComparisonError(f"{label} artifact set omits an exact input")
        if simulation.created_at < locked_at or artifact_set.created_at < locked_at:
            raise SimulationComparisonError(f"{label} evidence predates the policy lock")


def _pair_outcome(
    pair: PairedSimulationCell,
    resolved: _ResolvedInputs,
    store: ArtifactStore,
    text_set_ids: set[str],
    sandbox_set_ids: set[str],
) -> PairedSimulationOutcome:
    """Resolve one exact task coordinate and preserve both sides of its denominator."""
    task = resolved.tasks.get(pair.task_id)
    if task is None or task.partition != "held_out":
        raise SimulationComparisonError("text-versus-sandbox comparison uses held-out tasks only")
    if task.lineage_group_id != pair.task_lineage_group_id or sha256_json(task) != pair.task_sha256:
        raise SimulationComparisonError("comparison task lineage or digest drifted")
    text_cell = _cell(resolved.text_plan, pair.text_cell_id, "text")
    sandbox_cell = _cell(resolved.sandbox_plan, pair.sandbox_cell_id, "sandbox")
    _validate_plan_candidate(pair, resolved.text_plan, "text")
    _validate_plan_candidate(pair, resolved.sandbox_plan, "sandbox")
    _validate_cell(pair, text_cell, resolved.text_spec, "text")
    _validate_cell(pair, sandbox_cell, resolved.sandbox_spec, "sandbox")
    text = _optional_rollout(
        store,
        pair.text_rollout_id,
        pair.text_rollout_sha256,
        text_set_ids,
        "text",
    )
    sandbox = _optional_rollout(
        store,
        pair.sandbox_rollout_id,
        pair.sandbox_rollout_sha256,
        sandbox_set_ids,
        "sandbox",
    )
    if text is not None:
        _validate_rollout(pair, text, resolved, SimulationMode.WORLD_MODEL)
    if sandbox is not None:
        _validate_rollout(pair, sandbox, resolved, SimulationMode.SANDBOX)
    text_failure = (
        text.failure if text is not None else _missing_failure("text", pair.text_rollout_id)
    )
    sandbox_failure = (
        sandbox.failure
        if sandbox is not None
        else _missing_failure("sandbox", pair.sandbox_rollout_id)
    )
    terminal_match = None
    if (
        text is not None
        and sandbox is not None
        and text_failure is None
        and sandbox_failure is None
    ):
        terminal_match = (
            text.stop_reason == sandbox.stop_reason and text.final_output == sandbox.final_output
        )
    return PairedSimulationOutcome(
        pair_id=pair.pair_id,
        task_id=pair.task_id,
        task_lineage_group_id=pair.task_lineage_group_id,
        candidate_alias=pair.candidate_alias,
        candidate=pair.candidate,
        agent_id=pair.agent_id,
        repeat=pair.repeat,
        text_cell_id=pair.text_cell_id,
        sandbox_cell_id=pair.sandbox_cell_id,
        expected_text_rollout_id=pair.text_rollout_id,
        expected_sandbox_rollout_id=pair.sandbox_rollout_id,
        text_rollout_id=text.rollout_id if text is not None else None,
        sandbox_rollout_id=sandbox.rollout_id if sandbox is not None else None,
        text_simulator=pair.text_simulator,
        sandbox_simulator=pair.sandbox_simulator,
        text_failure=text_failure,
        sandbox_failure=sandbox_failure,
        terminal_match=terminal_match,
    )


def _validate_cell(
    pair: PairedSimulationCell,
    cell: EvaluationCell,
    simulation: SimulationSpec,
    label: str,
) -> None:
    """Require one plan cell to be the exact simulated held-out pair coordinate."""
    if cell.purpose != "held_out" or cell.execution != "simulate":
        raise SimulationComparisonError(f"{label} comparison cell is not simulated held-out data")
    if (
        cell.task_id,
        cell.candidate_alias,
        cell.repeat,
    ) != (pair.task_id, pair.candidate_alias, pair.repeat):
        raise SimulationComparisonError(f"{label} plan cell does not match its pair coordinate")
    if cell.cell_id not in simulation.cell_ids or simulation.agent_id != pair.agent_id:
        raise SimulationComparisonError(f"{label} simulation spec does not bind the pair")


def _validate_rollout(
    pair: PairedSimulationCell,
    rollout: RolloutArtifact,
    resolved: _ResolvedInputs,
    mode: SimulationMode,
) -> None:
    """Verify task, model, agent, simulator, spec, and per-cell binding identity."""
    is_text = mode is SimulationMode.WORLD_MODEL
    cell_id = pair.text_cell_id if is_text else pair.sandbox_cell_id
    simulation = resolved.text_spec if is_text else resolved.sandbox_spec
    expected_simulator = pair.text_simulator if is_text else pair.sandbox_simulator
    if (
        rollout.mode is not mode
        or rollout.evidence_source != ("world_model" if is_text else "sandbox")
        or rollout.cell_id != cell_id
        or rollout.task_id != pair.task_id
        or rollout.repeat != pair.repeat
        or rollout.candidate != pair.candidate
        or rollout.agent_id != pair.agent_id
        or rollout.simulator != expected_simulator
        or rollout.simulation_id != simulation.simulation_id
        or rollout.simulation_spec_sha256 != sha256_json(simulation)
    ):
        raise SimulationComparisonError(f"{mode.value} rollout identity does not match its pair")
    if rollout.created_at < resolved.lock_created_at:
        raise SimulationComparisonError(f"{mode.value} rollout predates the policy lock")
    if is_text:
        binding = rollout.simulation_binding
        settings = resolved.text_spec.world_model
        if binding is None or (
            settings is None
            or binding.task_sha256 != pair.task_sha256
            or binding.candidate_alias != pair.candidate_alias
            or binding.world_model_alias != settings.world_model_alias
            or binding.prompt_version != settings.prompt_version
            or binding.evaluation_plan_input != resolved.text_plan_input
            or binding.task_set_input != resolved.task_set_input
            or binding.simulation_spec_input != resolved.text_spec_input
        ):
            raise SimulationComparisonError("text rollout binding drifted")
    else:
        binding = rollout.sandbox_binding
        settings = resolved.sandbox_spec.sandbox
        if binding is None or (
            settings is None
            or binding.cell_id != cell_id
            or binding.task_id != pair.task_id
            or binding.purpose != "held_out"
            or binding.task_sha256 != pair.task_sha256
            or binding.task_lineage_group_id != pair.task_lineage_group_id
            or binding.candidate_alias != pair.candidate_alias
            or binding.environment_id != settings.environment_id
            or binding.environment_sha256 != settings.environment_sha256
            or binding.evaluation_plan_input != resolved.sandbox_plan_input
            or binding.task_set_input != resolved.task_set_input
            or binding.simulation_spec_input != resolved.sandbox_spec_input
        ):
            raise SimulationComparisonError("sandbox rollout binding drifted")


def _validate_plan_candidate(
    pair: PairedSimulationCell,
    plan: EvaluationPlan,
    label: str,
) -> None:
    """Require the pair's candidate alias to resolve to the plan's exact model snapshot."""
    candidates = {candidate.alias: candidate.model for candidate in plan.candidate_snapshots}
    if candidates.get(pair.candidate_alias) != pair.candidate:
        raise SimulationComparisonError(f"{label} plan candidate identity drifted")


def _optional_rollout(
    store: ArtifactStore,
    rollout_id: str,
    expected_sha256: str | None,
    artifact_set_ids: set[str],
    label: str,
) -> RolloutArtifact | None:
    """Resolve an exact rollout or preserve an explicitly absent denominator side."""
    listed = rollout_id in artifact_set_ids
    exists = rollout_id in set(store.list_ids())
    if expected_sha256 is None:
        if listed and exists:
            raise SimulationComparisonError(
                f"available {label} rollout requires its exact manifest digest"
            )
        return None
    if not listed:
        raise SimulationComparisonError(f"{label} rollout digest is not named by its artifact set")
    stored = _read_exact(
        store,
        ArtifactInput(artifact_id=rollout_id, sha256=expected_sha256),
        "rollout",
    )
    try:
        rollout = RolloutArtifact.model_validate_json(store.read_bytes(rollout_id, _ROLLOUT_FILE))
    except (ArtifactCorruptionError, ValueError) as exc:
        raise SimulationComparisonError(f"{label} rollout is invalid") from exc
    if rollout.rollout_id != rollout_id or stored.manifest.artifact_id != rollout_id:
        raise SimulationComparisonError(f"{label} rollout ID drifted")
    return rollout


def _load_plan(store: ArtifactStore, reference: ArtifactInput) -> EvaluationPlan:
    """Load one exact evaluation plan."""
    _read_exact(store, reference, "evaluation-plan")
    return _parse_model(store, reference.artifact_id, _PLAN_FILE, EvaluationPlan, "plan")


def _load_simulation_spec(store: ArtifactStore, reference: ArtifactInput) -> SimulationSpec:
    """Load one exact simulation specification."""
    _read_exact(store, reference, "simulation-spec")
    return _parse_model(
        store,
        reference.artifact_id,
        _SIMULATION_SPEC_FILE,
        SimulationSpec,
        "simulation spec",
    )


def _load_artifact_set(store: ArtifactStore, reference: ArtifactInput) -> SimulationArtifactSet:
    """Load one exact artifact set and verify its canonical ID index bytes."""
    _read_exact(store, reference, "simulation-artifact-set")
    artifact_set = _parse_model(
        store,
        reference.artifact_id,
        _ARTIFACT_SET_FILE,
        SimulationArtifactSet,
        "artifact set",
    )
    try:
        payload = store.read_bytes(reference.artifact_id, artifact_set.artifacts_path)
    except ArtifactCorruptionError as exc:
        raise SimulationComparisonError("artifact set index cannot be read") from exc
    if hashlib.sha256(payload).hexdigest() != artifact_set.artifacts_sha256:
        raise SimulationComparisonError("artifact set index digest disagrees with its envelope")
    parsed: list[str] = []
    try:
        for raw_line in payload.splitlines():
            record = json.loads(raw_line)
            if (
                not isinstance(record, dict)
                or set(record) != {"artifact_id"}
                or not isinstance(record["artifact_id"], str)
                or canonical_json_bytes(record) != raw_line
            ):
                raise ValueError("noncanonical artifact index row")
            parsed.append(record["artifact_id"])
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SimulationComparisonError("artifact set index is malformed") from exc
    if tuple(parsed) != artifact_set.artifact_ids:
        raise SimulationComparisonError("artifact set index disagrees with artifact_ids")
    return artifact_set


def _parse_model[T: BaseModel](
    store: ArtifactStore,
    artifact_id: str,
    path: str,
    model: type[T],
    label: str,
) -> T:
    """Parse one verified JSON file into its frozen Pydantic contract."""
    try:
        return model.model_validate_json(store.read_bytes(artifact_id, path))
    except (ArtifactCorruptionError, ValueError) as exc:
        raise SimulationComparisonError(f"comparison {label} is invalid") from exc


def _read_exact(
    store: ArtifactStore,
    reference: ArtifactInput,
    artifact_type: str | None = None,
) -> StoredArtifact:
    """Read one immutable artifact and require its exact manifest digest and optional type."""
    try:
        stored = store.read(reference.artifact_id)
    except ArtifactCorruptionError as exc:
        raise SimulationComparisonError(
            f"comparison input {reference.artifact_id!r} is missing or corrupt"
        ) from exc
    if artifact_input(stored.manifest) != reference:
        raise SimulationComparisonError(
            f"comparison input {reference.artifact_id!r} manifest digest drifted"
        )
    if artifact_type is not None and stored.manifest.artifact_type != artifact_type:
        raise SimulationComparisonError(
            f"comparison input {reference.artifact_id!r} has the wrong artifact type"
        )
    return stored


def _manifest_digest(store: ArtifactStore, artifact_id: str) -> Sha256:
    """Return the verified manifest digest for an already-required stored artifact."""
    try:
        return artifact_input(store.read(artifact_id).manifest).sha256
    except ArtifactCorruptionError as exc:
        raise SimulationComparisonError("comparison protocol must be persisted first") from exc


def _cell(plan: EvaluationPlan, cell_id: str, label: str) -> EvaluationCell:
    """Resolve one exact plan cell without inferring a broader grid."""
    cells = {cell.cell_id: cell for cell in plan.cells}
    try:
        return cells[cell_id]
    except KeyError as exc:
        raise SimulationComparisonError(f"{label} comparison cell is absent from its plan") from exc


def _outcome_counts(
    outcomes: tuple[PairedSimulationOutcome, ...],
) -> tuple[int, int, int, int, int, int, int]:
    """Count paired, usable, missing, failed, and matching denominator rows."""
    paired = sum(
        item.text_rollout_id is not None and item.sandbox_rollout_id is not None
        for item in outcomes
    )
    usable = sum(
        item.text_rollout_id is not None
        and item.sandbox_rollout_id is not None
        and item.text_failure is None
        and item.sandbox_failure is None
        for item in outcomes
    )
    missing_text = sum(item.text_rollout_id is None for item in outcomes)
    missing_sandbox = sum(item.sandbox_rollout_id is None for item in outcomes)
    failed_text = sum(
        item.text_rollout_id is not None and item.text_failure is not None for item in outcomes
    )
    failed_sandbox = sum(
        item.sandbox_rollout_id is not None and item.sandbox_failure is not None
        for item in outcomes
    )
    matches = sum(item.terminal_match is True for item in outcomes)
    return paired, usable, missing_text, missing_sandbox, failed_text, failed_sandbox, matches


def _missing_failure(mode: str, rollout_id: str) -> StructuredFailure:
    """Describe one absent expected artifact without silently shrinking the denominator."""
    return StructuredFailure(
        code=FailureCode.VALIDATION,
        message=f"required {mode} rollout {rollout_id} is absent from its frozen artifact set",
        details={"mode": mode, "rollout_id": rollout_id, "phase": "artifact_resolution"},
    )
