"""Machine-only provisional judge setup for the noninteractive hosted router workflow."""

from __future__ import annotations

from datetime import datetime

from wmo.common.core.artifacts import ArtifactInput, canonical_json_bytes, stable_id
from wmo.common.judging import (
    HumanLabelSet,
    HumanScoreHistory,
    JudgeCalibrationService,
    RouterLineageAssignment,
    RouterLineageSplit,
    Rubric,
    verify_persisted_calibration,
    write_router_lineage_split,
)
from wmo.common.models import (
    CompletionCostReservation,
    ModelCapabilities,
    ModelCatalog,
    ModelSnapshot,
    completion_cost_reservation,
)
from wmo.common.project import (
    ProjectHostedJudgeEvidence,
    ProjectStore,
    artifact_input,
)
from wmo.common.tasks import TaskCase, load_task_set
from wmo.common.traces import Trace, load_trace_dataset
from wmo.optimize.router.automatic.preflight import HostedAutomaticJudgeEvidence
from wmo.optimize.router.judging.artifacts import write_production_rollout
from wmo.optimize.router.judging.contracts import ProvisionalJudgeSetupArtifact
from wmo.optimize.router.judging.selection import representative_pairs, trace_preview
from wmo.optimize.router.judging.template_bind import (
    default_judge_dimensions,
    default_judge_template,
)
from wmo.runtime.models import RuntimeModelCatalog
from wmo.simulation.mining.bindings import load_task_set_lineage_bindings


def prepare_hosted_provisional_judge(
    project: ProjectStore,
    catalog: ModelCatalog,
    *,
    maximum_input_tokens: int,
    maximum_output_tokens: int,
    maximum_attempts: int,
    created_at: datetime,
    code_revision: str,
) -> HostedAutomaticJudgeEvidence:
    """Persist deterministic machine-only setup and zero-label provisional calibration.

    Args:
        project: Project with a selected completed grounded build.
        catalog: Transient secret-free catalog used only for static identity and pricing.
        maximum_input_tokens: Per-judge-request input ceiling.
        maximum_output_tokens: Per-judge-request output ceiling.
        maximum_attempts: Retry ceiling reserved for every judge request.
        created_at: Artifact completion time.
        code_revision: Exact producer revision.

    Returns:
        Verified setup, provisional calibration, and bounded request reservation.

    Raises:
        ValueError: Project evidence, role identity, capacity, or pricing is incomplete.
    """
    config = project.load_project()
    completed = config.build
    models = config.models
    if completed is None or models is None:
        raise ValueError("hosted provisional judge setup requires a completed configured build")
    resolver = RuntimeModelCatalog(catalog, environment={})
    judge_model, capabilities = resolver.snapshot(models.judge)
    if capabilities.supports_completions is not True:
        raise ValueError("hosted judge must declare supports_completions=true")
    existing = config.hosted_judge
    if existing is not None:
        setup_stored = project.artifacts.read(existing.setup.artifact_id)
        if (
            setup_stored.manifest.artifact_type != "provisional-judge-setup"
            or artifact_input(setup_stored.manifest) != existing.setup
        ):
            raise ValueError("selected hosted judge setup pointer is stale or wrongly typed")
        setup = ProvisionalJudgeSetupArtifact.model_validate_json(
            project.artifacts.read_bytes(existing.setup.artifact_id, "setup.json")
        )
        calibration, calibration_input = verify_persisted_calibration(
            project,
            existing.calibration.artifact_id,
        )
        if (
            setup.setup_id != existing.setup.artifact_id
            or setup.judge_alias != models.judge
            or setup.judge_model != judge_model
            or calibration_input != existing.calibration
            or calibration.status != "provisional"
            or calibration.label_count != 0
        ):
            raise ValueError("selected hosted judge evidence differs from active provisional setup")
        return HostedAutomaticJudgeEvidence(
            setup=setup,
            setup_input=existing.setup,
            calibration_id=calibration.calibration_id,
            calibration_input=calibration_input,
            request_reservation=_judge_request_reservation(
                capabilities,
                judge_model=judge_model,
                maximum_input_tokens=maximum_input_tokens,
                maximum_output_tokens=maximum_output_tokens,
                maximum_attempts=maximum_attempts,
            ),
        )
    tasks = load_task_set(project.artifacts, completed.task_set.artifact_id).tasks
    traces = load_trace_dataset(project.artifacts, completed.trace_dataset.artifact_id).traces
    selected = tuple(representative_pairs(tasks, traces, 1))
    if not selected:
        raise ValueError("hosted provisional judge setup needs one representative real trace")
    rubric, rubric_input = _persist_provisional_rubric(
        project,
        completed.task_set,
        created_at=created_at,
        code_revision=code_revision,
    )
    setup, setup_input = _persist_provisional_setup(
        project,
        project_id=config.project_id,
        judge_alias=models.judge,
        judge_model=judge_model,
        trace_dataset=completed.trace_dataset,
        task_set=completed.task_set,
        rubric=rubric_input,
        selected=selected,
        created_at=created_at,
        code_revision=code_revision,
    )
    task, trace = selected[0]
    rollout_input = write_production_rollout(
        project,
        setup,
        task,
        trace,
        created_at,
        code_revision,
        allow_provider_free_source=True,
    )
    split = _persist_lineage_split(
        project,
        completed.task_set,
        rollout_id=rollout_input.artifact_id,
        rollout_lineage_id=task.lineage_group_id,
        created_at=created_at,
        code_revision=code_revision,
    )
    labels = _persist_empty_label_set(
        project,
        rubric,
        rubric_input,
        created_at=created_at,
        code_revision=code_revision,
    )
    calibration = JudgeCalibrationService().bootstrap_provisional(
        project,
        rubric_id=rubric.rubric_id,
        label_set_id=labels.label_set_id,
        router_lineage_split_id=split.split_id,
        judge_model=judge_model,
        judge_prompt=setup.prompt_template.prompt,
        created_at=created_at,
        code_revision=code_revision,
    )
    calibration_input = artifact_input(project.artifacts.read(calibration.calibration_id).manifest)
    request = _judge_request_reservation(
        capabilities,
        judge_model=judge_model,
        maximum_input_tokens=maximum_input_tokens,
        maximum_output_tokens=maximum_output_tokens,
        maximum_attempts=maximum_attempts,
    )
    project.bind_hosted_judge_evidence(
        ProjectHostedJudgeEvidence(
            setup=setup_input,
            calibration=calibration_input,
        )
    )
    return HostedAutomaticJudgeEvidence(
        setup=setup,
        setup_input=setup_input,
        calibration_id=calibration.calibration_id,
        calibration_input=calibration_input,
        request_reservation=request,
    )


def _persist_provisional_rubric(
    project: ProjectStore,
    task_set: ArtifactInput,
    *,
    created_at: datetime,
    code_revision: str,
) -> tuple[Rubric, ArtifactInput]:
    """Persist the built-in provisional task-success rubric.

    Args:
        project: Project-local artifact store.
        task_set: Exact task-set manifest pointer.
        created_at: Artifact completion time.
        code_revision: Exact producer revision.

    Returns:
        Stored rubric and exact manifest pointer.
    """
    dimensions = default_judge_dimensions()
    rubric_id = stable_id(
        "provisional-rubric",
        {
            "version": "hosted-provisional-rubric-v1",
            "task_set": task_set.model_dump(mode="json"),
            "dimensions": [item.model_dump(mode="json") for item in dimensions],
        },
    )
    rubric = Rubric(
        schema_version=1,
        created_at=created_at,
        inputs=(task_set,),
        code_revision=code_revision,
        rubric_id=rubric_id,
        dimensions=dimensions,
        source_task_set_id=task_set.artifact_id,
        status="provisional",
    )
    stored, manifest = project.artifacts.write_or_replay(
        artifact_id=rubric_id,
        artifact_type="rubric",
        envelope=rubric,
        envelope_path="rubric.json",
        envelope_type=Rubric,
        files={"rubric.json": canonical_json_bytes(rubric)},
    )
    return stored, artifact_input(manifest)


def _persist_provisional_setup(
    project: ProjectStore,
    *,
    project_id: str,
    judge_alias: str,
    judge_model: ModelSnapshot,
    trace_dataset: ArtifactInput,
    task_set: ArtifactInput,
    rubric: ArtifactInput,
    selected: tuple[tuple[TaskCase, Trace], ...],
    created_at: datetime,
    code_revision: str,
) -> tuple[ProvisionalJudgeSetupArtifact, ArtifactInput]:
    """Persist one machine-only executable setup bound to real trace evidence.

    Args:
        project: Project-local artifact store.
        project_id: Project identity.
        judge_alias: Selected judge alias.
        judge_model: Exact static judge snapshot.
        trace_dataset: Exact trace-dataset pointer.
        task_set: Exact task-set pointer.
        rubric: Exact provisional rubric pointer.
        selected: One representative task/trace pair.
        created_at: Artifact completion time.
        code_revision: Exact producer revision.

    Returns:
        Stored setup and exact manifest pointer.
    """
    if len(selected) != 1:
        raise ValueError("provisional setup must contain one representative pair")
    task, trace = selected[0]
    inputs = tuple(sorted((trace_dataset, task_set, rubric), key=lambda item: item.artifact_id))
    template = default_judge_template()
    setup_id = stable_id(
        "provisional-judge-setup",
        {
            "version": "hosted-provisional-judge-setup-v1",
            "project_id": project_id,
            "judge_alias": judge_alias,
            "judge_model": judge_model.model_dump(mode="json"),
            "prompt_template": template.model_dump(mode="json"),
            "inputs": [item.model_dump(mode="json") for item in inputs],
        },
    )
    setup = ProvisionalJudgeSetupArtifact(
        schema_version=1,
        created_at=created_at,
        inputs=inputs,
        code_revision=code_revision,
        setup_id=setup_id,
        project_id=project_id,
        judge_alias=judge_alias,
        judge_model=judge_model,
        prompt_template=template,
        trace_dataset=trace_dataset,
        task_set=task_set,
        rubric=rubric,
        previews=(trace_preview(task, trace),),
    )
    stored, manifest = project.artifacts.write_or_replay(
        artifact_id=setup_id,
        artifact_type="provisional-judge-setup",
        envelope=setup,
        envelope_path="setup.json",
        envelope_type=ProvisionalJudgeSetupArtifact,
        files={"setup.json": canonical_json_bytes(setup)},
    )
    return stored, artifact_input(manifest)


def _persist_lineage_split(
    project: ProjectStore,
    task_set: ArtifactInput,
    *,
    rollout_id: str,
    rollout_lineage_id: str,
    created_at: datetime,
    code_revision: str,
) -> RouterLineageSplit:
    """Persist the task-set partitions and one bootstrap rollout assignment.

    Args:
        project: Project-local artifact store.
        task_set: Exact task-set pointer.
        rollout_id: Provider-free production rollout identity.
        rollout_lineage_id: Frozen lineage for that rollout.
        created_at: Artifact completion time.
        code_revision: Exact producer revision.

    Returns:
        Stored router-lineage split.
    """
    bindings = load_task_set_lineage_bindings(project.artifacts, task_set.artifact_id)
    fit = tuple(sorted({item.lineage_id for item in bindings.bindings if item.partition == "fit"}))
    held = tuple(
        sorted({item.lineage_id for item in bindings.bindings if item.partition == "held_out"})
    )
    split_id = stable_id(
        "router-lineage-split",
        {
            "task_set": task_set.model_dump(mode="json"),
            "fit": list(fit),
            "held_out": list(held),
            "assignments": [{"rollout_id": rollout_id, "lineage_id": rollout_lineage_id}],
        },
    )
    split = RouterLineageSplit(
        schema_version=1,
        created_at=created_at,
        inputs=(task_set,),
        code_revision=code_revision,
        split_id=split_id,
        source_task_set_id=task_set.artifact_id,
        fit_lineage_ids=fit,
        held_out_lineage_ids=held,
        assignments=(
            RouterLineageAssignment(
                rollout_id=rollout_id,
                lineage_id=rollout_lineage_id,
            ),
        ),
    )
    return write_router_lineage_split(project, split)


def _persist_empty_label_set(
    project: ProjectStore,
    rubric: Rubric,
    rubric_input: ArtifactInput,
    *,
    created_at: datetime,
    code_revision: str,
) -> HumanLabelSet:
    """Persist an explicit empty human-label set proving the evidence is machine only.

    Args:
        project: Project-local artifact store.
        rubric: Provisional rubric receiving no labels.
        rubric_input: Exact rubric manifest pointer.
        created_at: Artifact completion time.
        code_revision: Exact producer revision.

    Returns:
        Stored zero-label set.
    """
    history = HumanScoreHistory()
    label_set = HumanLabelSet(
        schema_version=1,
        created_at=created_at,
        inputs=(rubric_input,),
        code_revision=code_revision,
        label_set_id=stable_id(
            "human-label-set",
            {
                "rubric_id": rubric.rubric_id,
                "history": history.model_dump(mode="json"),
                "inputs": [rubric_input.model_dump(mode="json")],
            },
        ),
        rubric_id=rubric.rubric_id,
        history=history,
        active_label_ids=(),
    )
    stored, _manifest = project.artifacts.write_or_replay(
        artifact_id=label_set.label_set_id,
        artifact_type="human-label-set",
        envelope=label_set,
        envelope_path="labels.json",
        envelope_type=HumanLabelSet,
        files={"labels.json": canonical_json_bytes(label_set)},
    )
    return stored


def _judge_request_reservation(
    capabilities: ModelCapabilities,
    *,
    judge_model: ModelSnapshot,
    maximum_input_tokens: int,
    maximum_output_tokens: int,
    maximum_attempts: int,
) -> CompletionCostReservation:
    """Build one complete judge request reservation from static catalog declarations.

    Args:
        capabilities: Explicit judge model capabilities and token prices.
        judge_model: Exact judge snapshot.
        maximum_input_tokens: Per-call input ceiling.
        maximum_output_tokens: Per-call output ceiling.
        maximum_attempts: Per-call retry ceiling.

    Returns:
        Conservative retry-inclusive completion reservation.
    """
    prices = (
        capabilities.input_cost_per_million_tokens_usd,
        capabilities.output_cost_per_million_tokens_usd,
        capabilities.cached_input_cost_per_million_tokens_usd,
        capabilities.cache_write_cost_per_million_tokens_usd,
    )
    if any(value is None for value in prices):
        raise ValueError("hosted judge requires complete input, output, and cache pricing")
    context = capabilities.context_window_tokens
    output_capacity = capabilities.maximum_output_tokens
    if (
        context is None
        or output_capacity is None
        or maximum_output_tokens > output_capacity
        or maximum_input_tokens + maximum_output_tokens > context
    ):
        raise ValueError("hosted judge request ceilings exceed explicit model capacity")
    input_price, output_price, cached_price, cache_write_price = prices
    assert input_price is not None and output_price is not None
    assert cached_price is not None and cache_write_price is not None
    return completion_cost_reservation(
        model=judge_model,
        input_usd_per_million_tokens=input_price,
        output_usd_per_million_tokens=output_price,
        cached_input_usd_per_million_tokens=cached_price,
        cache_write_usd_per_million_tokens=cache_write_price,
        maximum_attempts=maximum_attempts,
        maximum_input_tokens=maximum_input_tokens,
        maximum_output_tokens=maximum_output_tokens,
    )
