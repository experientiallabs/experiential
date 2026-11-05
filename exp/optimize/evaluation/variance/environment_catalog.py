"""Provider-free named task views and independent environment-base planning.

Views reuse one canonical task set and saved tag matrices. Bases reuse the build's scoped
learning artifacts. A task/base row is diagnostic planning data: schema coverage, native family
membership and native start validation are independent evidence, never execution authority.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import ArtifactInput, ContractModel, sha256_json
from exp.common.models import ModelSnapshot
from exp.common.project import ProjectStore, artifact_input
from exp.common.tasks import TaskCase, load_task_set
from exp.optimize.evaluation.executable.grounded import GroundedRuntimeDefinition
from exp.optimize.evaluation.variance.bootstrap_contracts import BootstrapBinding, Provenance
from exp.optimize.evaluation.variance.bootstrap_stage import bootstrapped_registration
from exp.optimize.evaluation.variance.build_records import (
    MATRIX_TYPE,
    REPORT_TYPE,
    FamilyScope,
    TagMatrixRecord,
    VarianceBuildReport,
    load_exposure_plan,
    load_tag_matrix,
    save_artifact,
)
from exp.optimize.evaluation.variance.exposure import ExposurePlan
from exp.optimize.evaluation.variance.family_embedding import load_family_embedding_completion
from exp.optimize.evaluation.variance.family_simulators import (
    FamilySimulator,
    load_family_simulator,
)
from exp.simulation.mining.environment_families import load_environment_families
from exp.simulation.mining.questions import QuestionBank
from exp.simulation.mining.tag_selection import TagRule, TagSelection, select_traces
from exp.simulation.mining.tagging import TagMatrix
from exp.simulation.retrieval.contracts import RAG_ARTIFACT_TYPE, RAG_INDEX_PATH
from exp.simulation.retrieval.store import load_rag_index
from exp.simulation.retrieval.tags import TagView
from exp.simulation.rules.model import ToolContract
from exp.simulation.world_model.runtime import load_grounded_world_model_artifact

ENVIRONMENT_CATALOG_PROTOCOL = "environment-catalog-v2"
"""Interpretation of saved views, evidence coverage and unique task/base plan rows."""

CATALOG_TYPE = "environment-catalog"
"""Artifact type of a provider-free plan, stored as ``catalog.json``."""

DevelopmentRole = Literal["world_learning", "mining_probes"]
"""Roles eligible for catalog views; final evaluation remains globally held out."""


class TagTaskSetSpec(ContractModel):
    """One independently named view over the canonical tasks, selected by source tags.

    Attributes:
        view: Saved rendered view whose matrix is reused.
        rule: Native predicate; its name identifies this task-set view.
        role: Exact global role retained by this view.
        source_match: Whether any or every source trace must match the predicate.
    """

    view: TagView = "task"
    rule: TagRule
    role: DevelopmentRole = "mining_probes"
    source_match: Literal["any", "all"] = "any"


class EnvironmentBaseSpec(ContractModel):
    """One named reusable base, independently chosen from the build's scopes.

    Attributes:
        name: Stable name, independent of tag-set names.
        family_id: Native family whose scoped learning inputs are reused.
        embedding_completion: Optional verified completed native exact-family index overlay.
        simulator: Optional pinned learned simulator of precisely the resulting scope.
        starts: Existing frozen native start bindings, never requests to create starts.
    """

    name: str = Field(min_length=1, max_length=64)
    family_id: str = Field(min_length=1)
    embedding_completion: ArtifactInput | None = None
    simulator: ArtifactInput | None = None
    starts: tuple[BootstrapBinding, ...] = ()


class EnvironmentCatalogSpec(ContractModel):
    """Inputs to provider-free catalog construction.

    Attributes:
        task_sets: Independently named views; their task membership may overlap.
        bases: Independent scope/simulator references, reusable by every view.
        banks: Exact question text/version used by each saved matrix.
    """

    task_sets: tuple[TagTaskSetSpec, ...] = Field(min_length=1)
    bases: tuple[EnvironmentBaseSpec, ...] = Field(min_length=1)
    banks: dict[TagView, QuestionBank]


class TagTaskSetView(ContractModel):
    """A view references tasks without copying or changing their lineage or partition.

    Attributes:
        spec: Predicate and retained global role.
        matrix: Reused saved analysis artifact.
        selection: Native trace selection and its uncertainty diagnostics.
        task_ids: Ordered canonical tasks matching the source predicate.
        missing_tag_tasks: Eligible tasks with at least one untagged source trace.
        unmatched_trace_ids: Selected traces with no selected canonical task.
    """

    spec: TagTaskSetSpec
    matrix: ArtifactInput
    selection: TagSelection
    task_ids: tuple[str, ...]
    missing_tag_tasks: tuple[str, ...]
    unmatched_trace_ids: tuple[str, ...]


class EnvironmentBase(ContractModel):
    """Independent learning inputs with their native evidence and optional runtime.

    Attributes:
        spec: Named native inputs, including optional frozen start bindings.
        families: Native reviewed family record.
        scope: Separate evidence, retrieval, world and rule-memory inputs.
        tool_contracts: Saved declarations used solely for schema diagnostics.
        runtime_bound: Whether a pinned simulator has a frozen runtime template.
        index_embedder: Actual model snapshot of this base index, independent of other bases.
        producer_revision: Actual scoped world producer, preserved through native overlays.
    """

    spec: EnvironmentBaseSpec
    families: ArtifactInput
    scope: FamilyScope
    tool_contracts: dict[str, ToolContract]
    runtime_bound: bool
    index_embedder: ModelSnapshot
    producer_revision: str


class NativeStartProof(ContractModel):
    """A passed native validation of this task/base start, with explicit provenance.

    Attributes:
        binding: Verified frozen bootstrap batch and task.
        checkpoint: Captured native start.
        provenance: Observed, reconstructed or authored; inferred starts stay explicit.
        method: Native validation method, not an invented catalog QA.
        executable_task_sha256: Exact authored public task in the validated runtime.
        initial_state_sha256: Explicit state input of that native runtime.
    """

    binding: BootstrapBinding
    checkpoint: ArtifactInput
    provenance: Provenance
    method: Literal["exact_restore", "prefix_replay", "coherence_and_first_reveal"]
    executable_task_sha256: str
    initial_state_sha256: str


class EnvironmentInstancePlan(ContractModel):
    """One unique task/base pair, shared by every matching named task-set view.

    Attributes:
        task_id: Canonical task reference.
        base_name: Independent base name.
        task_sets: All named views requesting this same pair.
        lineage_id: Original global leakage lineage.
        partition: Original task partition, unchanged by views.
        role: Original exposure-plan role, unchanged by views.
        tool_coverage: Exact schema coverage, distinct from semantic or state validity.
        tool_issues: Missing or unequal declarations by tool name.
        native_family_member: Whether the native family placed this exact task.
        state_evidence: Recorded context only, no family state, or passed native validation.
        task_start_qa: Passed only with a verified frozen native start; otherwise pending.
        start: Native proof when passed. Planning grants no execution authority.
    """

    task_id: str
    base_name: str
    task_sets: tuple[str, ...]
    lineage_id: str
    partition: Literal["fit"]
    role: DevelopmentRole
    tool_coverage: Literal["schema_matches", "schema_unknown", "schema_mismatch", "missing_tools"]
    tool_issues: dict[str, str]
    native_family_member: bool
    state_evidence: Literal["no_family_state", "recorded_context_only", "native_start_validated"]
    task_start_qa: Literal["pending", "passed"]
    start: NativeStartProof | None = None


class EnvironmentCatalog(ContractModel):
    """A reusable planning artifact; one physical task/base row despite overlapping views.

    Attributes:
        protocol: Planning semantics version.
        report: Pinned source build report.
        task_set: Canonical physical task set reused by every view.
        exposure: Exact immutable global lineage plan reused by every view and base.
        spec: Inputs needed to reproduce the catalog without provider calls.
        task_sets: Named tag-filtered views and missing-analysis diagnostics.
        bases: Independent evidence/tool/runtime inputs.
        instances: Unique many-to-many task/base plans, never a choice of one winner.
        omitted_tasks: Canonical tasks not selected, with global holdout or view diagnosis.
    """

    protocol: str = ENVIRONMENT_CATALOG_PROTOCOL
    report: ArtifactInput
    task_set: ArtifactInput
    exposure: ArtifactInput
    spec: EnvironmentCatalogSpec
    task_sets: tuple[TagTaskSetView, ...]
    bases: tuple[EnvironmentBase, ...]
    instances: tuple[EnvironmentInstancePlan, ...]
    omitted_tasks: dict[str, str]


def _payload(project: ProjectStore, ref: ArtifactInput, kind: str, name: str) -> bytes:
    """Read a pinned, typed input before interpreting its payload."""
    stored = project.artifacts.read(ref.artifact_id)
    if artifact_input(stored.manifest) != ref or stored.manifest.artifact_type != kind:
        raise ValueError(f"catalog input {ref.artifact_id} is not the pinned {kind}")
    return stored.payloads[name]


def _tasks(project: ProjectStore, ref: ArtifactInput, plan: ExposurePlan) -> tuple[TaskCase, ...]:
    """Verify the canonical task pin, source lineages and global held-out partition."""
    _payload(project, ref, "task-set", "task-set.json")
    if plan.bindings != ref:
        raise ValueError("catalog exposure plan names another canonical task set")
    tasks = load_task_set(project.artifacts, ref.artifact_id).tasks
    for task in tasks:
        role = plan.roles.get(task.lineage_group_id)
        if role is None or (task.partition == "held_out") != (role == "final_evaluation"):
            raise ValueError(f"task {task.task_id} partition differs from global exposure role")
        if any(plan.traces.get(trace) != task.lineage_group_id for trace in task.source_trace_ids):
            raise ValueError(f"task {task.task_id} source trace differs from its global lineage")
    return tasks


def _matrix(
    project: ProjectStore,
    report: VarianceBuildReport,
    view: TagView,
    bank: QuestionBank,
    plan: ExposurePlan,
) -> tuple[ArtifactInput, TagMatrix]:
    """Reuse one saved matrix only when its exact bank text and content digests agree."""
    matches = [tag for tag in report.tagging if tag.plan.view == view]
    if len(matches) != 1:
        raise ValueError(f"catalog needs exactly one saved {view} tag matrix")
    ref = matches[0].matrix
    record = TagMatrixRecord.model_validate_json(_payload(project, ref, MATRIX_TYPE, "record.json"))
    matrix = load_tag_matrix(project, ref)
    if (
        record.bank_id != bank.bank_id
        or record.bank_version != bank.version
        or matrix.bank_id != bank.bank_id
        or matrix.bank_version != bank.version
        or record.traces != len(matrix.rows)
        or record.bank_sha256 != sha256_json(bank)
        or record.matrix_sha256 != sha256_json(matrix)
        or record.view != view
        or matrix.view != view
    ):
        raise ValueError("catalog matrix or exact question-bank digest differs")
    if any(plan.role_of(row.trace_id) not in record.roles for row in matrix.rows):
        raise ValueError("catalog matrix rows differ from their recorded exposure roles")
    if any(plan.role_of(row.trace_id) == "final_evaluation" for row in matrix.rows):
        raise ValueError("catalog tag matrix contains globally held-out evidence")
    return ref, matrix


def _view(
    spec: TagTaskSetSpec,
    ref: ArtifactInput,
    matrix: TagMatrix,
    bank: QuestionBank,
    tasks: tuple[TaskCase, ...],
    plan: ExposurePlan,
) -> TagTaskSetView:
    """Apply the native tag predicate inside one fixed role, retaining missing coverage."""
    eligible = matrix.model_copy(
        update={
            "rows": tuple(row for row in matrix.rows if plan.role_of(row.trace_id) == spec.role)
        }
    )
    selection = select_traces(eligible, spec.rule, bank=bank)
    selected, tagged = set(selection.trace_ids), {row.trace_id for row in eligible.rows}
    ids: list[str] = []
    missing: list[str] = []
    covered: set[str] = set()
    for task in tasks:
        if plan.roles[task.lineage_group_id] != spec.role:
            continue
        sources = set(task.source_trace_ids)
        if not sources or not sources <= tagged:
            missing.append(task.task_id)
        matches = (
            bool(sources & selected)
            if spec.source_match == "any"
            else bool(sources) and sources <= selected
        )
        if matches:
            ids.append(task.task_id)
            covered.update(sources & selected)
    return TagTaskSetView(
        spec=spec,
        matrix=ref,
        selection=selection,
        task_ids=tuple(ids),
        missing_tag_tasks=tuple(missing),
        unmatched_trace_ids=tuple(trace for trace in selection.trace_ids if trace not in covered),
    )


def _base(
    project: ProjectStore,
    report_ref: ArtifactInput,
    report: VarianceBuildReport,
    spec: EnvironmentBaseSpec,
    plan: ExposurePlan,
) -> tuple[EnvironmentBase, FamilySimulator | None]:
    """Verify independently referenced native base evidence remains world-learning only."""
    scoping = report.environment_families
    if scoping is None:
        raise ValueError("catalog source build has no environment-family scopes")
    scope = scoping.scope(spec.family_id)
    if spec.embedding_completion is not None:
        intent, completion = load_family_embedding_completion(
            project, spec.embedding_completion, historical=True
        )
        if intent.source_build != report_ref or intent.source_scope != scope:
            raise ValueError("catalog embedding completion names another source build or base")
        scope = completion.scope
    plan.require((*scope.members, *scope.rules_traces), "world_learning")
    if not set(scope.rules_traces) <= set(scope.members):
        raise ValueError("base rules evidence lies outside its native family members")
    if scope.retrieval_scope.source_plan != sha256_json(plan):
        raise ValueError("base retrieval scope names another global exposure plan")
    if scope.retrieval_scope.kind != "trace" or not set(scope.retrieval_scope.ids) <= set(
        scope.members
    ):
        raise ValueError("base retrieval evidence differs from native family members")
    plan.require(scope.retrieval_scope.ids, "world_learning")
    _payload(project, scope.index, RAG_ARTIFACT_TYPE, RAG_INDEX_PATH)
    index = load_rag_index(project.artifacts, scope.index.artifact_id)
    world = load_grounded_world_model_artifact(project.artifacts, scope.world_model)
    if index.index.corpus_scope != scope.retrieval_scope or world.serving_rag != scope.index:
        raise ValueError("base index/world inputs differ from its native scope")
    record = load_environment_families(project.artifacts, scoping.families)
    family = next((row for row in record.families if row.family_id == scope.family_id), None)
    if family is not None:
        for task in family.tasks:
            plan.require_lineage(task.lineage_id, task.role)
            plan.require(task.source_trace_ids, task.role)
    contracts = {} if family is None else family.tool_contracts
    simulator = None
    if spec.simulator is not None:
        simulator = load_family_simulator(project.artifacts, spec.simulator)
        if (
            simulator.family_id != scope.family_id
            or simulator.members != scope.members
            or simulator.corpus != scope.retrieval_scope
            or simulator.index != scope.index
            or simulator.world_model != scope.world_model
            or simulator.families != scoping.families
            or simulator.rules_traces != scope.rules_traces
            or simulator.tag_retrieval != scope.tag_retrieval
            or set(task.task_id for task in simulator.tasks) != set(scope.tasks)
        ):
            raise ValueError("catalog simulator differs from native base scope")
        if family is not None and (
            simulator.tool_contracts != family.tool_contracts
            or simulator.state_domains != family.state_domains
            or simulator.world_semantics != family.world_semantics
            or simulator.tasks != family.tasks
            or simulator.status != family.status
        ):
            raise ValueError("catalog simulator differs from native reviewed family evidence")
        contracts = simulator.tool_contracts
        if simulator.runtime is not None and (
            simulator.runtime.world_model != scope.world_model
            or simulator.runtime.rule_book != simulator.rule_book
            or simulator.runtime.rule_memory != simulator.rule_memory
            or simulator.runtime.tag_retrieval != scope.tag_retrieval
        ):
            raise ValueError("catalog simulator runtime differs from its frozen learning inputs")
    if family is not None and (
        family.members != scope.members or set(family.task_ids()) != set(scope.tasks)
    ):
        raise ValueError("catalog family membership differs from native scope")
    return EnvironmentBase(
        spec=spec,
        families=scoping.families,
        scope=scope,
        tool_contracts=contracts,
        runtime_bound=simulator is not None and simulator.runtime is not None,
        index_embedder=index.index.embedder,
        producer_revision=world.code_revision,
    ), simulator


def _tool_coverage(
    task: TaskCase,
    contracts: dict[str, ToolContract],
) -> tuple[
    Literal["schema_matches", "schema_unknown", "schema_mismatch", "missing_tools"], dict[str, str]
]:
    """Compare declarations only; equal tool schemas do not establish semantic compatibility."""
    issues: dict[str, str] = {}
    for tool in task.tools:
        contract = contracts.get(tool.name)
        if contract is None:
            issues[tool.name] = "missing tool"
        elif contract.declared_schema is None:
            issues[tool.name] = "no saved declaration"
        elif contract.declared_schema != tool.input_schema:
            issues[tool.name] = "declaration differs"
    if "missing tool" in issues.values():
        return "missing_tools", issues
    if "declaration differs" in issues.values():
        return "schema_mismatch", issues
    return ("schema_unknown" if issues else "schema_matches"), issues


def _starts(
    project: ProjectStore,
    base: EnvironmentBase,
    simulator: FamilySimulator | None,
    tasks: dict[str, TaskCase],
) -> dict[str, NativeStartProof]:
    """Attach passed QA only from verified frozen native task/base registrations."""
    proofs: dict[str, NativeStartProof] = {}
    for binding in base.spec.starts:
        if simulator is None or simulator.runtime is None:
            raise ValueError("catalog start proof requires a pinned runtime-bound simulator")
        registration = bootstrapped_registration(project, binding)
        task = tasks.get(binding.task_id)
        if task is None:
            raise ValueError("catalog start names a task outside all selected views")
        definition = GroundedRuntimeDefinition.model_validate(registration.job.configuration)
        shared_fields = {"task", "manifest", "initial_state", "probe_paths", "facts"}
        if definition.model_dump(exclude=shared_fields) != simulator.runtime.model_dump(
            exclude=shared_fields
        ):
            raise ValueError("catalog start changed the frozen base runtime beyond its task/state")
        if (
            registration.family_id != base.scope.family_id
            or registration.task_id != task.task_id
            or task.task_id not in base.scope.tasks
            or not registration.validation.passed
            or registration.validation.provenance != registration.provenance
            or registration.job.study.checkpoint != registration.checkpoint
            or definition.initial_state != registration.draft.initial_state
            or definition.manifest != registration.draft.manifest
            or definition.probe_paths != registration.draft.probe_paths
            or definition.task
            != task.model_copy(
                update={
                    "instruction": registration.draft.instruction,
                    "tools": registration.draft.tools,
                }
            )
            or definition.world_model != base.scope.world_model
            or definition.rule_book != simulator.rule_book
            or definition.rule_memory != simulator.rule_memory
            or definition.tag_retrieval != simulator.tag_retrieval
        ):
            raise ValueError("catalog start proof differs from the task/base or failed native QA")
        if binding.task_id in proofs:
            raise ValueError("catalog base repeats native task-start proof")
        proofs[binding.task_id] = NativeStartProof(
            binding=binding,
            checkpoint=registration.checkpoint,
            provenance=registration.provenance,
            method=registration.validation.method,
            executable_task_sha256=sha256_json(definition.task),
            initial_state_sha256=sha256_json(definition.initial_state),
        )
    return proofs


def plan_environment_catalog(
    project: ProjectStore,
    report_ref: ArtifactInput,
    spec: EnvironmentCatalogSpec,
) -> EnvironmentCatalog:
    """Compose many named task views with independent bases, reading saved inputs only.

    Raises:
        ValueError: Names repeat, pins/banks differ, or any view/base violates global exposure.
    """
    report = VarianceBuildReport.model_validate_json(
        _payload(project, report_ref, REPORT_TYPE, "report.json")
    )
    plan = load_exposure_plan(project, report.exposure)
    tasks = _tasks(project, report.build.task_set, plan)
    if len({row.rule.name for row in spec.task_sets}) != len(spec.task_sets):
        raise ValueError("catalog task-set names must be unique")
    if len({row.name for row in spec.bases}) != len(spec.bases):
        raise ValueError("catalog base names must be unique")
    if len({row.family_id for row in spec.bases}) != len(spec.bases):
        raise ValueError("catalog base aliases duplicate the same physical learning inputs")
    if not {row.view for row in spec.task_sets} <= spec.banks.keys():
        raise ValueError("catalog needs the exact saved question bank for every selected view")
    matrices = {
        view: _matrix(project, report, view, spec.banks[view], plan)
        for view in {row.view for row in spec.task_sets}
    }
    views = tuple(
        _view(row, *matrices[row.view], spec.banks[row.view], tasks, plan) for row in spec.task_sets
    )
    bases_and_simulators = tuple(
        _base(project, report_ref, report, row, plan) for row in spec.bases
    )
    memberships: dict[str, list[str]] = {}
    for view in views:
        for task_id in view.task_ids:
            memberships.setdefault(task_id, []).append(view.spec.rule.name)
    selected = {task.task_id: task for task in tasks if task.task_id in memberships}
    rows: list[EnvironmentInstancePlan] = []
    for base, simulator in bases_and_simulators:
        proofs = _starts(project, base, simulator, selected)
        for task_id, task in selected.items():
            role = plan.roles[task.lineage_group_id]
            if role == "final_evaluation" or task.partition != "fit":
                raise ValueError("catalog views selected a globally held-out task")
            coverage, issues = _tool_coverage(task, base.tool_contracts)
            member, proof = task_id in base.scope.tasks, proofs.get(task_id)
            rows.append(
                EnvironmentInstancePlan(
                    task_id=task_id,
                    base_name=base.spec.name,
                    task_sets=tuple(memberships[task_id]),
                    lineage_id=task.lineage_group_id,
                    partition=task.partition,
                    role=role,
                    tool_coverage=coverage,
                    tool_issues=issues,
                    native_family_member=member,
                    state_evidence="native_start_validated"
                    if proof
                    else "recorded_context_only"
                    if member
                    else "no_family_state",
                    task_start_qa="passed" if proof else "pending",
                    start=proof,
                )
            )
    return EnvironmentCatalog(
        report=report_ref,
        task_set=report.build.task_set,
        exposure=report.exposure,
        spec=spec,
        task_sets=views,
        bases=tuple(base for base, _ in bases_and_simulators),
        instances=tuple(rows),
        omitted_tasks={
            task.task_id: "global final evaluation"
            if task.partition == "held_out"
            else "no named view selected task"
            for task in tasks
            if task.task_id not in selected
        },
    )


def save_environment_catalog(
    project: ProjectStore,
    report: ArtifactInput,
    spec: EnvironmentCatalogSpec,
    *,
    code_revision: str,
) -> ArtifactInput:
    """Persist a reproducible provider-free plan without changing execution or spend authority."""
    catalog = plan_environment_catalog(project, report, spec)
    refs = [report, catalog.task_set, catalog.exposure, *(row.matrix for row in catalog.task_sets)]
    for base in catalog.bases:
        refs.extend((base.families, base.scope.index, base.scope.world_model))
        if base.scope.tag_retrieval is not None:
            refs.append(base.scope.tag_retrieval)
        if base.spec.embedding_completion is not None:
            refs.append(base.spec.embedding_completion)
        if base.spec.simulator is not None:
            refs.append(base.spec.simulator)
        refs.extend(binding.batch for binding in base.spec.starts)
    return save_artifact(
        project.artifacts, CATALOG_TYPE, {"catalog.json": catalog}, tuple(refs), code_revision
    )


def load_environment_catalog(project: ProjectStore, ref: ArtifactInput) -> EnvironmentCatalog:
    """Verify a pinned catalog by reproducing its views and diagnostics without model calls."""
    catalog = EnvironmentCatalog.model_validate_json(
        _payload(project, ref, CATALOG_TYPE, "catalog.json")
    )
    if catalog.protocol != ENVIRONMENT_CATALOG_PROTOCOL:
        raise ValueError("catalog protocol differs from this implementation")
    if plan_environment_catalog(project, catalog.report, catalog.spec) != catalog:
        raise ValueError("saved catalog differs from native source evidence")
    return catalog
