"""Application-owned semantic verification for hosted Project bundles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal

from pydantic import BaseModel

from exp.common.core.artifacts import (
    ArtifactInput,
    sha256_json,
    sorted_unique_inputs,
)
from exp.common.evaluations import load_evaluation_dataset
from exp.common.evaluations.evidence import read_evaluation_plan, read_judgment, read_rollout
from exp.common.judging import verify_persisted_calibration
from exp.common.project import (
    ProjectBuildArtifacts,
    ProjectConfig,
    ProjectHostedJudgeEvidence,
    ProjectRouterPolicyArtifacts,
    ProjectRouterReportArtifacts,
    ProjectStage,
    ProjectStore,
    artifact_input,
)
from exp.common.project.catalog import load_project_model_catalog
from exp.common.routing import (
    KnnRouterPolicy,
    ReservedFrozenEmbeddingSet,
    load_frozen_embedding_set,
)
from exp.common.routing.bank import KnnBankManifest
from exp.common.routing.decision import policy_content_sha256
from exp.optimize.router.automatic.execution_contract import (
    load_router_execution_contract,
)
from exp.optimize.router.composition import RouterPolicyLock
from exp.optimize.router.fit.report import HeldOutRouterReport
from exp.optimize.router.hosted_spend import provider_spend_source_pairs
from exp.optimize.router.judging.contracts import ProvisionalJudgeSetupArtifact
from exp.optimize.router.spend import (
    ProviderSpendComponent,
    ProviderSpendLedger,
    ProviderSpendStatus,
)
from exp.runtime.agents import agent_factory_sha256
from exp.simulation.retrieval import (
    RAGLineageBinding,
    RAGTransition,
    load_completed_build_rag_lineage_bindings,
    load_rag_index,
)
from exp.simulation.world_model import verify_grounded_world_model_artifact


def verify_hosted_project(project: ProjectStore) -> None:
    """Verify the complete semantic graph of any selected hosted Project stage.

    Args:
        project: Materialized Project whose hosted pointers must cross-bind before use.

    Raises:
        ValueError: Hosted setup, build, judge, policy, report, or spend evidence differs.
    """
    config = project.load_project()
    hosted = any(
        value is not None
        for value in (
            config.system,
            config.build_spend_ledger,
            config.hosted_judge,
            config.router_policy,
            config.router_report,
        )
    )
    if not hosted:
        return
    if (
        config.system is None
        or config.models is None
        or config.model_catalog is None
        or config.retrieval is None
        or config.budgets is None
        or config.budgets.maximum_provider_cost_usd is None
    ):
        raise ValueError("hosted Project has an incomplete late setup")
    catalog = load_project_model_catalog(project.artifacts, config.model_catalog)
    aliases = {item.alias for item in catalog.models}
    required = {
        config.models.world_model,
        config.models.judge,
        config.models.embedder,
        *config.models.candidates,
    }
    if config.models.incumbent is not None:
        required.add(config.models.incumbent)
    if catalog.project_id != config.project_id or not required.issubset(aliases):
        raise ValueError("hosted Project catalog differs from its selected roles")
    if config.build is None:
        if any(
            value is not None
            for value in (
                config.build_spend_ledger,
                config.hosted_judge,
                config.router_policy,
                config.router_report,
            )
        ):
            raise ValueError("hosted Project selects downstream evidence before its build")
        return
    if config.build_spend_ledger is None:
        raise ValueError("hosted Project build omits its provider-spend ledger")
    _verify_grounded_build(project, config, config.build)
    _verify_stage_ledger(
        project,
        config,
        config.build_spend_ledger,
        stage=ProjectStage.BUILDING_WORLD_MODEL,
        stage_outputs=_build_outputs(config.build),
    )
    if config.hosted_judge is None:
        if config.router_policy is not None or config.router_report is not None:
            raise ValueError("hosted Project selects router evidence before judge setup")
        return
    _verify_judge_evidence(project, config, config.hosted_judge)
    if config.router_policy is None:
        if config.router_report is not None:
            raise ValueError("hosted Project selects a report before its frozen policy")
        return
    _verify_policy_selection(project, config, config.router_policy)
    if config.router_report is not None:
        _verify_report_selection(project, config, config.router_report)


def _verify_grounded_build(
    project: ProjectStore,
    config: ProjectConfig,
    build: ProjectBuildArtifacts,
) -> None:
    """Bind verified RAG/world payloads and task lineages to the selected setup."""
    if config.models is None or config.model_catalog is None or config.retrieval is None:
        raise ValueError("hosted build requires selected roles, catalog, and retrieval settings")
    expected_types = {
        "trace_dataset": "trace-dataset",
        "task_set": "task-set",
        "serving_rag": "trace-rag-index",
        "fit_rag": "trace-rag-index",
        "world_model": "grounded-world-model",
    }
    for field_name, artifact_type in expected_types.items():
        _verify_artifact_pointer(
            project,
            getattr(build, field_name),
            artifact_type=artifact_type,
        )
    expected_inputs = {
        "task_set": (build.trace_dataset,),
        "serving_rag": (build.trace_dataset,),
        "fit_rag": (build.trace_dataset,),
        "world_model": (build.serving_rag,),
    }
    for field_name, inputs in expected_inputs.items():
        manifest = project.artifacts.read(getattr(build, field_name).artifact_id).manifest
        if manifest.inputs != inputs:
            raise ValueError(f"{field_name} artifact does not bind the hosted build graph")
    catalog = load_project_model_catalog(project.artifacts, config.model_catalog)
    snapshots = {item.alias: item.model for item in catalog.models}
    serving = load_rag_index(project.artifacts, build.serving_rag.artifact_id)
    fit = load_rag_index(project.artifacts, build.fit_rag.artifact_id)
    world = verify_grounded_world_model_artifact(project.artifacts, build.world_model)
    bindings = load_completed_build_rag_lineage_bindings(project.artifacts, build)
    binding_by_trace = {item.trace_id: item for item in bindings}
    expected_fit_lineages = tuple(
        sorted({item.lineage_id for item in bindings if item.partition == "fit"})
    )
    expected_all_lineages = tuple(sorted({item.lineage_id for item in bindings}))
    if (
        artifact_input(serving.manifest) != build.serving_rag
        or artifact_input(fit.manifest) != build.fit_rag
        or serving.index.embedder != snapshots.get(config.models.embedder)
        or fit.index.embedder != snapshots.get(config.models.embedder)
        or serving.index.default_top_k != config.retrieval.top_k
        or fit.index.default_top_k != config.retrieval.top_k
        or serving.index.included_partitions != ("fit", "held_out")
        or fit.index.included_partitions != ("fit",)
        or serving.index.sources != fit.index.sources
        or serving.index.fit_lineage_ids != expected_fit_lineages
        or fit.index.fit_lineage_ids != expected_fit_lineages
        or serving.index.included_lineage_ids != expected_all_lineages
        or fit.index.included_lineage_ids != expected_fit_lineages
        or serving.index.embedding_dimension != fit.index.embedding_dimension
        or not _transitions_match_bindings(serving.transitions, binding_by_trace, fit_only=False)
        or not _transitions_match_bindings(fit.transitions, binding_by_trace, fit_only=True)
        or world.world_model_id != build.world_model.artifact_id
        or world.serving_rag != build.serving_rag
        or world.model_alias != config.models.world_model
        or world.model != snapshots.get(config.models.world_model)
        or world.top_k != config.retrieval.top_k
    ):
        raise ValueError("grounded build or RAG lineage differs from the hosted Project setup")


def _transitions_match_bindings(
    transitions: Sequence[RAGTransition],
    binding_by_trace: Mapping[str, RAGLineageBinding],
    *,
    fit_only: bool,
) -> bool:
    """Return whether every RAG transition preserves its task-set lineage assignment."""
    for transition in transitions:
        binding = binding_by_trace.get(transition.trace_id)
        if binding is None or binding.lineage_id != transition.lineage_id:
            return False
        if fit_only and binding.partition != "fit":
            return False
    return True


def _verify_judge_evidence(
    project: ProjectStore,
    config: ProjectConfig,
    evidence: ProjectHostedJudgeEvidence,
) -> None:
    """Bind provisional setup and canonical calibration to build and judge snapshot."""
    if config.build is None or config.models is None or config.model_catalog is None:
        raise ValueError("hosted judge evidence requires build, roles, and Project catalog")
    _verify_artifact_pointer(project, evidence.setup, artifact_type="provisional-judge-setup")
    _verify_artifact_pointer(project, evidence.calibration, artifact_type="judge-calibration")
    setup = _read_payload(
        project,
        evidence.setup,
        relative_path="setup.json",
        model_type=ProvisionalJudgeSetupArtifact,
    )
    calibration, calibration_input = verify_persisted_calibration(
        project,
        evidence.calibration.artifact_id,
    )
    catalog = load_project_model_catalog(project.artifacts, config.model_catalog)
    models = {item.alias: item.model for item in catalog.models}
    prompt = setup.prompt_template.prompt
    if (
        setup.setup_id != evidence.setup.artifact_id
        or setup.project_id != config.project_id
        or setup.trace_dataset != config.build.trace_dataset
        or setup.task_set != config.build.task_set
        or setup.judge_alias != config.models.judge
        or setup.judge_model != models.get(config.models.judge)
        or setup.status != "provisional"
    ):
        raise ValueError("hosted provisional judge setup differs from the selected build")
    if (
        calibration.calibration_id != evidence.calibration.artifact_id
        or calibration_input != evidence.calibration
        or calibration.rubric_id != setup.rubric.artifact_id
        or calibration.judge_model != setup.judge_model
        or calibration.judge_prompt_id != prompt.prompt_id
        or calibration.judge_prompt_sha256 != prompt.sha256
        or setup.rubric not in calibration.inputs
        or calibration.status != "provisional"
        or calibration.label_count != 0
        or calibration.approved_at is not None
        or calibration.risk_acceptance is not None
    ):
        raise ValueError("hosted provisional calibration differs from its selected setup")


def _verify_policy_selection(
    project: ProjectStore,
    config: ProjectConfig,
    selection: ProjectRouterPolicyArtifacts,
) -> None:
    """Bind fit lock, frozen policy, Project roles, execution, and stage spend."""
    if (
        config.build is None
        or config.models is None
        or config.model_catalog is None
        or config.hosted_judge is None
    ):
        raise ValueError("router policy requires selected build, judge, roles, and catalog")
    _verify_artifact_pointer(project, selection.policy_lock, artifact_type="router-policy-lock")
    _verify_artifact_pointer(project, selection.policy, artifact_type="router-policy")
    _verify_artifact_pointer(
        project,
        selection.spend_ledger,
        artifact_type="provider-spend-ledger",
    )
    lock = _read_payload(
        project,
        selection.policy_lock,
        relative_path="lock.json",
        model_type=RouterPolicyLock,
    )
    policy = _read_payload(
        project,
        selection.policy,
        relative_path="policy.json",
        model_type=KnnRouterPolicy,
    )
    bank = _read_payload(
        project,
        lock.bank,
        relative_path="bank.json",
        model_type=KnnBankManifest,
    )
    fit_evaluation = load_evaluation_dataset(
        project.artifacts,
        lock.fit_evaluation.artifact_id,
    )
    catalog = load_project_model_catalog(project.artifacts, config.model_catalog)
    snapshots = {item.alias: item.model for item in catalog.models}
    expected_lock_inputs = sorted_unique_inputs(
        lock.plan,
        lock.fit_evaluation,
        lock.bank,
        lock.policy,
    )
    if (
        lock.lock_id != selection.policy_lock.artifact_id
        or lock.policy != selection.policy
        or lock.inputs != expected_lock_inputs
        or policy.policy_id != selection.policy.artifact_id
        or policy.inputs != sorted_unique_inputs(lock.fit_evaluation, lock.bank)
        or policy.evaluation_plan_id != lock.plan.artifact_id
        or policy.evaluation_plan_sha256 != lock.plan.sha256
        or policy.fit_evaluation_id != lock.fit_evaluation.artifact_id
        or policy.bank_artifact_id != lock.bank.artifact_id
        or bank.bank_artifact_id != lock.bank.artifact_id
        or policy.bank_sha256 != bank.bank_sha256
        or bank.fit_evaluation_id != lock.fit_evaluation.artifact_id
        or bank.evaluation_plan_id != lock.plan.artifact_id
        or bank.evaluation_plan_sha256 != lock.plan.sha256
        or policy.task_set_id != config.build.task_set.artifact_id
        or policy.task_set_sha256 != config.build.task_set.sha256
        or bank.task_set_id != config.build.task_set.artifact_id
        or bank.task_set_sha256 != config.build.task_set.sha256
        or artifact_input(project.artifacts.read(lock.fit_evaluation.artifact_id).manifest)
        != lock.fit_evaluation
        or fit_evaluation.manifest.evaluation_plan_id != lock.plan.artifact_id
        or fit_evaluation.manifest.evaluation_plan_sha256 != lock.plan.sha256
        or fit_evaluation.manifest.task_set_id != config.build.task_set.artifact_id
        or lock.plan not in fit_evaluation.manifest.inputs
        or config.build.task_set not in fit_evaluation.manifest.inputs
        or fit_evaluation.manifest.held_out_task_ids
        or any(row.purpose != "fit" for row in fit_evaluation.rows)
        or tuple(item.alias for item in policy.candidates) != config.models.candidates
        or policy.baseline_alias != config.models.incumbent
        or policy.embedder_alias != config.models.embedder
        or policy.embedder != snapshots.get(config.models.embedder)
        or any(item.model != snapshots.get(item.alias) for item in policy.candidates)
        or policy.judgment_status != "provisional"
    ):
        raise ValueError("router policy differs from its lock or hosted Project setup")
    _verify_policy_execution(project, config, lock)
    _verify_stage_ledger(
        project,
        config,
        selection.spend_ledger,
        stage=ProjectStage.OPTIMIZING_ROUTER,
        stage_outputs=(selection.policy_lock, selection.policy),
    )


def _verify_policy_execution(
    project: ProjectStore,
    config: ProjectConfig,
    lock: RouterPolicyLock,
) -> None:
    """Bind the policy plan's unique execution contract to the hosted setup."""
    if (
        config.build is None
        or config.build_spend_ledger is None
        or config.hosted_judge is None
        or config.system is None
        or config.models is None
        or config.model_catalog is None
        or config.budgets is None
        or config.budgets.maximum_provider_cost_usd is None
    ):
        raise ValueError("router execution verification requires complete hosted Project state")
    plan, plan_input = read_evaluation_plan(project.artifacts, lock.plan.artifact_id)
    if plan_input != lock.plan:
        raise ValueError("router policy lock plan manifest changed")
    execution_inputs = _inputs_of_type(project, plan.inputs, "router-execution-contract")
    if len(execution_inputs) != 1:
        raise ValueError("router policy plan must bind one exact execution contract")
    execution_input = execution_inputs[0]
    execution = load_router_execution_contract(
        project.artifacts,
        execution_input.artifact_id,
    )
    _verify_artifact_pointer(
        project,
        execution_input,
        artifact_type="router-execution-contract",
    )
    catalog = load_project_model_catalog(project.artifacts, config.model_catalog)
    snapshots = {item.alias: item.model for item in catalog.models}
    expected_agent = agent_factory_sha256(
        config.agent,
        maximum_model_calls=config.system.maximum_model_calls,
        system_prompt=config.system.system_prompt,
    )
    expected_simulation = sha256_json(
        {
            "version": "automatic-router-simulation-configuration-v1",
            "agent_factory_sha256": expected_agent,
            "redacted_field_names": list(config.redacted_field_names),
        }
    )
    build_ledger = _verify_stage_ledger(
        project,
        config,
        config.build_spend_ledger,
        stage=ProjectStage.BUILDING_WORLD_MODEL,
        stage_outputs=_build_outputs(config.build),
    )
    remaining = config.budgets.maximum_provider_cost_usd - build_ledger.total_usd
    execution_ceiling = Decimal.from_float(execution.maximum_provider_cost_usd)
    required_inputs = (
        config.build.trace_dataset,
        config.build.task_set,
        config.build.fit_rag,
        config.build.world_model,
        config.hosted_judge.setup,
        config.hosted_judge.calibration,
    )
    if (
        plan.task_set_id != config.build.task_set.artifact_id
        or tuple(item.alias for item in plan.candidate_snapshots) != config.models.candidates
        or any(item.model != snapshots.get(item.alias) for item in plan.candidate_snapshots)
        or tuple(item.candidate_alias for item in execution.candidates) != config.models.candidates
        or execution.incumbent_alias != config.models.incumbent
        or any(item.model != snapshots.get(item.candidate_alias) for item in execution.candidates)
        or execution.router_embedding_reservation.model != snapshots.get(config.models.embedder)
        or execution.world_model_alias != config.models.world_model
        or execution.world_model != snapshots.get(config.models.world_model)
        or execution.judge_alias != config.models.judge
        or execution.judge_model != snapshots.get(config.models.judge)
        or execution.agent_factory_sha256 != expected_agent
        or execution.simulation_configuration_sha256 != expected_simulation
        or execution_ceiling > remaining
        or any(item not in execution.inputs for item in required_inputs)
    ):
        raise ValueError("router execution contract differs from the hosted Project setup")


def _verify_report_selection(
    project: ProjectStore,
    config: ProjectConfig,
    selection: ProjectRouterReportArtifacts,
) -> None:
    """Bind report to exact policy, held-out evaluation, plan, task set, and ledger."""
    if config.router_policy is None or config.build is None:
        raise ValueError("router report requires a selected frozen policy and build")
    _verify_artifact_pointer(project, selection.report, artifact_type="router-report")
    _verify_artifact_pointer(
        project,
        selection.spend_ledger,
        artifact_type="provider-spend-ledger",
    )
    policy = _read_payload(
        project,
        config.router_policy.policy,
        relative_path="policy.json",
        model_type=KnnRouterPolicy,
    )
    lock = _read_payload(
        project,
        config.router_policy.policy_lock,
        relative_path="lock.json",
        model_type=RouterPolicyLock,
    )
    report = _read_payload(
        project,
        selection.report,
        relative_path="report.json",
        model_type=HeldOutRouterReport,
    )
    evaluation_inputs = _inputs_of_type(project, report.inputs, "evaluation")
    if len(evaluation_inputs) != 1:
        raise ValueError("held-out report must bind one exact evaluation")
    evaluation_input = evaluation_inputs[0]
    evaluation = load_evaluation_dataset(project.artifacts, evaluation_input.artifact_id)
    if (
        artifact_input(project.artifacts.read(evaluation_input.artifact_id).manifest)
        != evaluation_input
        or report.report_id != selection.report.artifact_id
        or report.policy_id != policy.policy_id
        or report.policy_sha256 != policy_content_sha256(policy)
        or report.evaluation_id != evaluation.manifest.evaluation_id
        or report.inputs != sorted_unique_inputs(config.router_policy.policy, evaluation_input)
        or evaluation.manifest.evaluation_plan_id != lock.plan.artifact_id
        or evaluation.manifest.evaluation_plan_sha256 != lock.plan.sha256
        or evaluation.manifest.task_set_id != config.build.task_set.artifact_id
        or config.build.task_set not in evaluation.manifest.inputs
        or lock.plan not in evaluation.manifest.inputs
        or evaluation.manifest.fit_task_ids
        or not evaluation.manifest.held_out_task_ids
        or any(row.purpose != "held_out" for row in evaluation.rows)
        or report.held_out_task_ids != evaluation.manifest.held_out_task_ids
    ):
        raise ValueError("held-out report differs from its selected policy evaluation")
    _verify_stage_ledger(
        project,
        config,
        selection.spend_ledger,
        stage=ProjectStage.COMPLETING_REPORT,
        stage_outputs=(selection.report,),
    )


def _verify_stage_ledger(
    project: ProjectStore,
    config: ProjectConfig,
    pointer: ArtifactInput,
    *,
    stage: ProjectStage,
    stage_outputs: tuple[ArtifactInput, ...],
) -> ProviderSpendLedger:
    """Bind one ledger to its stage outputs and exact cumulative attempt spend."""
    _verify_artifact_pointer(project, pointer, artifact_type="provider-spend-ledger")
    ledger = _read_payload(
        project,
        pointer,
        relative_path="spend-ledger.json",
        model_type=ProviderSpendLedger,
    )
    ceiling = config.budgets.maximum_provider_cost_usd if config.budgets is not None else None
    expected_outputs = tuple(sorted(stage_outputs, key=lambda item: item.artifact_id))
    if (
        ledger.ledger_id != pointer.artifact_id
        or ledger.project_id != config.project_id
        or ledger.stage != stage
        or ledger.ceiling_usd != ceiling
        or ledger.stage_outputs != expected_outputs
        or ledger.outcome != "completed"
        or ledger.restart != "completed_stage_bundle"
    ):
        raise ValueError("provider spend ledger differs from its selected hosted stage")
    _verify_ledger_billing_sources(project, config, ledger)
    prior_pointer = (
        config.router_policy.spend_ledger
        if stage == ProjectStage.COMPLETING_REPORT and config.router_policy is not None
        else config.build_spend_ledger
        if stage == ProjectStage.OPTIMIZING_ROUTER
        else None
    )
    if prior_pointer is not None:
        prior = _read_payload(
            project,
            prior_pointer,
            relative_path="spend-ledger.json",
            model_type=ProviderSpendLedger,
        )
        if (
            ledger.attempt_id != prior.attempt_id
            or ledger.attempt_authority_sha256 != prior.attempt_authority_sha256
            or ledger.ceiling_usd != prior.ceiling_usd
        ):
            raise ValueError("provider spend ledger changes the selected hosted attempt")
        current_entries = {item.operation_id: item for item in ledger.entries}
        prior_incurred = tuple(
            item for item in prior.entries if item.status != ProviderSpendStatus.NOT_INCURRED
        )
        if any(current_entries.get(item.operation_id) != item for item in prior_incurred):
            raise ValueError("provider spend ledger drops or changes prior incurred spend")
    return ledger


def _verify_ledger_billing_sources(
    project: ProjectStore,
    config: ProjectConfig,
    ledger: ProviderSpendLedger,
) -> None:
    """Bind every alias-free ledger source to exact catalog and operation evidence.

    Args:
        project: Hosted Project owning the selected ledger.
        config: Verified Project setup and role selection.
        ledger: Completed stage ledger being restored.

    Raises:
        ValueError: A component source is absent, extra, or differs from its exact evidence.
    """
    if config.model_catalog is None or config.models is None:
        raise ValueError("provider spend ledger requires a frozen Project model catalog")
    catalog = load_project_model_catalog(project.artifacts, config.model_catalog)
    models = {item.alias: item.model for item in catalog.models}
    try:
        expected_pairs = set(
            provider_spend_source_pairs(
                candidates=tuple(models[alias] for alias in config.models.candidates),
                world_model=models[config.models.world_model],
                judge=models[config.models.judge],
                embedder=models[config.models.embedder],
            )
        )
    except KeyError as exc:
        raise ValueError("provider spend ledger role is absent from the Project catalog") from exc
    actual_pairs = {(item.component, item.billing_source) for item in ledger.entries}
    if actual_pairs != expected_pairs:
        raise ValueError("provider spend ledger differs from the frozen billing-source plan")
    embedder_source = models[config.models.embedder].billing_source
    build_entries = set()
    if ledger.stage != ProjectStage.BUILDING_WORLD_MODEL and config.build_spend_ledger is not None:
        build_entries = set(
            _read_payload(
                project,
                config.build_spend_ledger,
                relative_path="spend-ledger.json",
                model_type=ProviderSpendLedger,
            ).entries
        )
    for entry in ledger.entries:
        if entry.status == ProviderSpendStatus.NOT_INCURRED:
            continue
        if entry.evidence is None:
            if (
                entry.component != ProviderSpendComponent.RETRIEVAL_EMBEDDING
                or entry.billing_source != embedder_source
                or (
                    ledger.stage != ProjectStage.BUILDING_WORLD_MODEL and entry not in build_entries
                )
            ):
                raise ValueError("provider spend entry omits its source-bearing evidence")
            continue
        stored = project.artifacts.read(entry.evidence.artifact_id)
        if artifact_input(stored.manifest) != entry.evidence:
            raise ValueError("provider spend evidence manifest digest changed")
        if stored.manifest.artifact_type == "rollout":
            rollout, _pointer = read_rollout(project.artifacts, entry.evidence.artifact_id)
            if rollout.candidate is None or rollout.world_model is None:
                raise ValueError("provider spend rollout omits its model billing sources")
            if entry.component == ProviderSpendComponent.CANDIDATE:
                expected_source = rollout.candidate.billing_source
            elif entry.component == ProviderSpendComponent.WORLD_MODEL:
                expected_source = rollout.world_model.billing_source
            elif entry.component == ProviderSpendComponent.RETRIEVAL_EMBEDDING:
                if rollout.simulation_binding is None:
                    raise ValueError("provider spend rollout omits its embedding reservation")
                expected_source = rollout.simulation_binding.query_embedding.model.billing_source
            else:
                raise ValueError("provider spend rollout names an unrelated component")
        elif stored.manifest.artifact_type == "judgment":
            judgment, _pointer = read_judgment(project.artifacts, entry.evidence.artifact_id)
            if entry.component != ProviderSpendComponent.JUDGE:
                raise ValueError("provider spend judgment names an unrelated component")
            expected_source = judgment.judge_model.billing_source
        elif stored.manifest.artifact_type == "router-embeddings":
            embeddings = load_frozen_embedding_set(
                project.artifacts,
                entry.evidence.artifact_id,
            )
            if entry.component != ProviderSpendComponent.ROUTER_EMBEDDING or not isinstance(
                embeddings, ReservedFrozenEmbeddingSet
            ):
                raise ValueError("provider spend embeddings omit their exact reservation")
            expected_source = embeddings.reservation.model.billing_source
        else:
            raise ValueError("provider spend entry names unsupported operation evidence")
        if entry.billing_source != expected_source:
            raise ValueError("provider spend entry billing source differs from its evidence")


def _verify_artifact_pointer(
    project: ProjectStore,
    pointer: ArtifactInput,
    *,
    artifact_type: str,
) -> None:
    """Require one exact immutable pointer and artifact type."""
    stored = project.artifacts.read(pointer.artifact_id)
    if stored.manifest.artifact_type != artifact_type:
        raise ValueError(
            f"artifact {pointer.artifact_id} is {stored.manifest.artifact_type!r}, "
            f"not {artifact_type!r}"
        )
    if artifact_input(stored.manifest) != pointer:
        raise ValueError(f"artifact {pointer.artifact_id} manifest digest changed")


def _inputs_of_type(
    project: ProjectStore,
    inputs: tuple[ArtifactInput, ...],
    artifact_type: str,
) -> tuple[ArtifactInput, ...]:
    """Return exact inputs whose verified manifests have one artifact type."""
    return tuple(
        item
        for item in inputs
        if project.artifacts.read(item.artifact_id).manifest.artifact_type == artifact_type
    )


def _build_outputs(build: ProjectBuildArtifacts) -> tuple[ArtifactInput, ...]:
    """Return the canonical complete hosted-build output pointer set."""
    return (
        build.trace_dataset,
        build.task_set,
        build.serving_rag,
        build.fit_rag,
        build.world_model,
    )


def _read_payload[ModelT: BaseModel](
    project: ProjectStore,
    pointer: ArtifactInput,
    *,
    relative_path: str,
    model_type: type[ModelT],
) -> ModelT:
    """Parse one typed payload after its artifact pointer has been verified."""
    return model_type.model_validate_json(
        project.artifacts.read_bytes(pointer.artifact_id, relative_path)
    )
