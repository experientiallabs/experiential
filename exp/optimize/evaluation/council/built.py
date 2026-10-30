"""Compose a fresh build's held-out calibration and actor council through existing owners."""

from collections.abc import Callable
from datetime import UTC, datetime

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.evaluations.actor_rubric import actor_judge
from exp.common.models import load_model_catalog
from exp.common.progress import ProgressHook, report
from exp.common.project import ProjectStore, artifact_input
from exp.common.tasks.store import load_task_set
from exp.optimize.evaluation.audit.panel import source_rag_inputs
from exp.optimize.evaluation.council.runner import (
    CouncilActor,
    TraceCouncilSpec,
    execute_trace_council,
    prepare_trace_council,
    validate_council_models,
)
from exp.optimize.evaluation.fidelity.replay import DEADLINE, replay_spec
from exp.optimize.evaluation.fidelity.runner import (
    FidelityRetrieval,
    PreparedFidelity,
    execute_fidelity_evaluation,
    prepare_fidelity_evaluation,
)
from exp.runtime.models.providers.transport import RetryPolicy
from exp.simulation.evaluation.counterfactuals import RecordedEpisode, RecordedStep
from exp.simulation.retrieval.jev import JEVPolicy
from exp.simulation.world_model.artifact import persist_grounded_world_model
from exp.simulation.world_model.runtime import load_grounded_world_model_artifact

DEFAULT_ACTOR_MODELS = (
    "actor-qwen27",
    "actor-glmflash",
    "actor-glmfull",
    "actor-mimo",
    "actor-kimi",
)


class BuiltCouncil(ContractModel):
    """Saved credential-free composition of two lineage-disjoint held-out workloads.

    Attributes:
        id: Immutable build, model and source-bound registration.
        calibration: Existing prepared fidelity run, whose report analysts receive later.
        council: Existing council specification, prepared only after calibration completes.
        code_revision: Exact source identity for both maintained execution owners.
    """

    id: str
    calibration: PreparedFidelity
    council: TraceCouncilSpec
    code_revision: str


def prepare_built_council(
    project: ProjectStore,
    models: tuple[str, ...] | None = None,
    *,
    code_revision: str,
    clock: Callable[[], datetime] | None = None,
    maximum_concurrency: int = 1,
) -> BuiltCouncil:
    """Bind calibration and actors to the same fit-only simulator from the current build.

    Args:
        project: A project completed by the normal ``exp build`` entry point.
        models: Distinct configured actor aliases; omission selects the five-model council.
        code_revision: Producing source identity.
        clock: Aware clock used only when creating a new registration.
        maximum_concurrency: Maximum concurrently executing independent actor sessions.

    Returns:
        Saved plan preserving deadlines on identical replay, without provider calls.

    Raises:
        ValueError: Build, models, tools or disjoint held-out lineages are unavailable.
    """
    if maximum_concurrency < 1:
        raise ValueError("maximum concurrency must be positive")
    if models is None:
        models = DEFAULT_ACTOR_MODELS
    if len(models) < 2 or len(set(models)) != len(models):
        raise ValueError("built council needs at least two distinct actor model aliases")
    config = project.load_project()
    build = config.build
    if build is None:
        raise ValueError("built council needs a completed exp build")
    catalog = load_model_catalog(project.model_catalog_path)
    actor_limits = [4096]
    for alias in models:
        entry = catalog.models.get(alias)
        if (
            entry is None
            or entry.capabilities is None
            or entry.capabilities.supports_tools is not True
            or entry.capabilities.context_window_tokens is None
            or catalog.connections[entry.connection].provider != "openai-compatible"
        ):
            raise ValueError(f"actor {alias!r} needs configured tool and context capabilities")
        actor_limits.append(entry.capabilities.maximum_output_tokens or 4096)
    identity = stable_id(
        "built-council",
        {
            "profile": "built-council-actor-judge-v3",
            "build": build.model_dump(mode="json"),
            "models": list(models),
            "catalog_sha256": sha256_json(catalog),
            "code_revision": code_revision,
            **({"maximum_concurrency": maximum_concurrency} if maximum_concurrency != 1 else {}),
        },
    )
    if project.artifacts.exists(identity):
        return BuiltCouncil.model_validate_json(project.artifacts.read_bytes(identity, "plan.json"))
    task_count = len(load_task_set(project.artifacts, build.task_set.artifact_id).tasks)
    world_model, cases = source_rag_inputs(project, task_count)
    calibration_case = cases[0]
    actor_case = next(
        (
            case
            for case in cases
            if case.task.lineage_group_id != calibration_case.task.lineage_group_id
        ),
        None,
    )
    if actor_case is None:
        raise ValueError("calibration and actors require two distinct held-out task lineages")
    calibration_case = calibration_case.model_copy(
        update={"task": calibration_case.task.model_copy(update={"reference_answer": None})}
    )
    actor_case = actor_case.model_copy(
        update={"task": actor_case.task.model_copy(update={"reference_answer": None})}
    )
    simulator = load_grounded_world_model_artifact(project.artifacts, world_model)
    retrieval = None
    decision_aliases = [
        alias
        for alias, entry in catalog.models.items()
        if catalog.connections[entry.connection].provider == "typesafe"
    ]
    if len(decision_aliases) == 1:
        policy = JEVPolicy(candidate_limit=8)
        retrieval = FidelityRetrieval(jev_alias=decision_aliases[0], jev=policy, memory_top_k=4)
        artifact = load_grounded_world_model_artifact(project.artifacts, world_model)
        rebound = persist_grounded_world_model(
            project.artifacts,
            artifact.serving_rag,
            model_alias=artifact.model_alias,
            model=artifact.model,
            created_at=artifact.created_at,
            code_revision=code_revision,
            top_k=artifact.top_k,
            knowledge=artifact.knowledge,
            selection_policy_sha256=policy.identity,
        )
        world_model = artifact_input(rebound.manifest)
    now = (clock or (lambda: datetime.now(UTC)))()
    deadline = now + DEADLINE
    calibration_spec = replay_spec(
        project,
        world_model,
        calibration_case,
        deadline=deadline,
        label="built-calibration",
        retrieval=retrieval,
    )
    calibration_spec = calibration_spec.model_copy(
        update={
            "run_id": identity + "-calibration",
            "judge_output_protocol": "json",
        }
    )
    calibration = prepare_fidelity_evaluation(
        project, calibration_spec, code_revision=code_revision
    )
    native_id = stable_id(
        "built-native",
        {
            "trace": actor_case.trace_id,
            "lineage": actor_case.task.lineage_group_id,
            "initial_context": actor_case.task.initial_context,
            "steps": [row.model_dump(mode="json") for row in actor_case.steps],
        },
    )
    native = RecordedEpisode(
        episode_id=native_id,
        task_lineage=actor_case.task.lineage_group_id,
        initial_context=actor_case.task.initial_context,
        steps=tuple(
            RecordedStep(action={"name": row.name, "arguments": row.arguments}, reply=row.reply)
            for row in actor_case.steps
        ),
    )
    if not project.artifacts.exists(native_id):
        project.artifacts.write_json(
            artifact_id=native_id,
            artifact_type="archived-native-episode",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=now,
                inputs=(build.trace_dataset,),
                code_revision=code_revision,
            ),
            files={"episode.json": native.model_dump(mode="json")},
        )
    native_ref = artifact_input(project.artifacts.read(native_id).manifest)
    actor_output = min(actor_limits)
    # Each tool operation can purchase one embedding, one JEV decision and one proposal.
    per_tool = 1 + (simulator.original_evidence != "none") + (retrieval is not None)
    physical_attempts = RetryPolicy().maximum_attempts * (len(models) * (3 + 2 * per_tool + 1) + 1)
    council = TraceCouncilSpec(
        run_id=identity + "-actors",
        world_model=world_model,
        tasks=(actor_case.task,),
        actors=tuple(CouncilActor(alias=alias, tier="operator-selected") for alias in models),
        analysts=(calibration_spec.judge_alias,),
        judge_alias=calibration_spec.judge_alias,
        # Actors are judged on task behavior, not on the calibration's reconstruction fidelity.
        judge=actor_judge(
            calibration_spec.judge.model_identity, per_turn=calibration_spec.judge.per_turn
        ),
        retrieval=retrieval,
        memory=None,
        analyst_fidelity_plans=(artifact_input(project.artifacts.read(calibration.id).manifest),),
        analyst_native_episodes=(native_ref,),
        deadline=deadline,
        maximum_physical_attempts=physical_attempts,
        maximum_concurrency=maximum_concurrency,
        maximum_actor_turns=3,
        maximum_tool_operations=2,
        actor_output_tokens=actor_output,
        simulator_output_tokens=calibration_spec.budget("prediction").output_tokens,
        judge_output_tokens=calibration_spec.budget("judging").output_tokens,
        analyst_output_tokens=calibration_spec.budget("judging").output_tokens,
        output_protocol=calibration_spec.output_protocol,
        judge_output_protocol=calibration_spec.judge_output_protocol,
    )
    validate_council_models(catalog, council, simulator_alias=simulator.model_alias)
    prepared = BuiltCouncil(
        id=identity, calibration=calibration, council=council, code_revision=code_revision
    )
    inputs: tuple[ArtifactInput, ...] = tuple(
        sorted(
            (world_model, council.analyst_fidelity_plans[0], native_ref),
            key=lambda value: value.artifact_id,
        )
    )
    project.artifacts.write_json(
        artifact_id=identity,
        artifact_type="built-trace-council-plan",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=now, inputs=inputs, code_revision=code_revision
        ),
        files={"plan.json": prepared.model_dump(mode="json")},
    )
    return prepared


def execute_built_council(
    project: ProjectStore,
    prepared: BuiltCouncil,
    *,
    progress: ProgressHook | None = None,
    clock: Callable[[], datetime] | None = None,
) -> JsonObject:
    """Execute existing fidelity and council owners with one saved composition.

    Args:
        project: Owner of the completed build and immutable registration.
        prepared: Credential-free saved plan returned by ``prepare_built_council``.
        progress: Maintained progress callback for both execution phases.
        clock: Optional shared aware clock.

    Returns:
        Existing trace-council report with all actor and analyst outcomes retained.
    """
    saved = BuiltCouncil.model_validate_json(project.artifacts.read_bytes(prepared.id, "plan.json"))
    if saved != prepared:
        raise ValueError("built council execution differs from its saved registration")
    report(progress, "Measuring simulator fidelity on a separate held-out lineage")
    execute_fidelity_evaluation(project, prepared.calibration, progress=progress, clock=clock)
    report(progress, "Running actors and simulator-aware trace analysis")
    council = prepare_trace_council(project, prepared.council, code_revision=prepared.code_revision)
    return execute_trace_council(project, council, progress=progress, clock=clock)
