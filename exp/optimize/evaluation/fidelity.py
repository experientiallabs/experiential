"""Prepare and execute simulation fidelity through existing durable runtime owners."""

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Literal

from filelock import FileLock
from pydantic import AwareDatetime, Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.evaluations.reconstruction import (
    DualReport,
    EvaluationReferences,
    JudgeConfig,
    JudgeNotRun,
)
from exp.common.models import ModelCapabilities, ModelCatalog, load_model_catalog
from exp.common.models.model import ReasoningEffort
from exp.common.progress import ProgressHook, report
from exp.common.project import ProjectStore, artifact_input
from exp.common.tasks import TaskCase
from exp.optimize.evaluation.decision_access import decision_caller
from exp.optimize.evaluation.fidelity_memory import (
    check_partition,
    consume_trajectory,
    maintain_memory,
    trajectory_record,
)
from exp.optimize.evaluation.knowledge_exposure import require_unexposed_task
from exp.optimize.evaluation.retrieval_binding import (
    bind_index_embedder,
    index_embedder_alias,
    require_unindexed_held_out_task,
)
from exp.runtime.gateway.decisions_contracts import DecisionRequest
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.models.budget import RequestBudget
from exp.runtime.models.providers.admitted_transport import AdmittedJsonTransport
from exp.runtime.models.providers.transport import HttpxJsonTransport
from exp.runtime.models.registry import ResolvedModel
from exp.runtime.models.structured_completion import (
    CompletionPlan,
    ModelCompletion,
    StructuredCompletion,
)
from exp.runtime.sessions.dispatch_ledger import DispatchLedger
from exp.runtime.sessions.store import SessionStore
from exp.simulation.evaluation import FixedWorkflow, evaluate_workflow
from exp.simulation.retrieval.jev import JEVCueSelector, JEVPolicy, JEVSelector
from exp.simulation.retrieval.selection import EvidencePolicy
from exp.simulation.world_model.artifact import GroundedWorldModelArtifact
from exp.simulation.world_model.durable import SessionPolicy
from exp.simulation.world_model.experience_memory import (
    Eligibility,
    MemoryVersion,
    load_source,
    materialize_log,
    memory_head,
    memory_view,
    provenance_gaps,
    public_trajectory_identity,
    read_memory_version,
    require_memory_scope,
    reserve_task_lineage,
    reserved_task_lineages,
    summary_generations,
)
from exp.simulation.world_model.grounded_session import open_grounded_session
from exp.simulation.world_model.memory_tree import (
    overview,
    overview_sources,
)
from exp.simulation.world_model.overview_pin import pinned_summaries
from exp.simulation.world_model.runtime import (
    load_grounded_world_model,
    load_grounded_world_model_artifact,
)
from exp.simulation.world_model.session_research import OnlineResearchPolicy


class FidelityRetrieval(ContractModel):
    """Native JEV relevance decisions through a configured decision-model alias.

    Attributes:
        jev_alias: Catalog alias served by the native decision wire.
        jev: Frozen rubric; it must equal any reranker bound in the simulator artifact.
        memory_top_k: Maximum research records delivered to one prediction.
    """

    jev_alias: str = Field(min_length=1)
    jev: JEVPolicy
    memory_top_k: int = Field(default=4, gt=0, strict=True)


class RoleBudget(ContractModel):
    """Output allowance and optional reasoning configuration for one model role.

    Attributes:
        output_tokens: Output reservation for every call in this role.
        reasoning_effort: Effort for this role; accepted only for models whose capabilities
            pin a reasoning effort, the same rule catalog role efforts follow.
    """

    output_tokens: int = Field(gt=0, strict=True)
    reasoning_effort: ReasoningEffort | None = None


class RoleBudgets(ContractModel):
    """Separate allowances for each model role; attempts and deadline remain shared.

    Attributes:
        prediction: Simulator transition proposals.
        investigation: Post-feedback investigations and memory compression.
        judging: The required LLM judge.
    """

    prediction: RoleBudget
    investigation: RoleBudget
    judging: RoleBudget


class FidelityMemory(ContractModel):
    """Cross-trajectory memory: learn from this trajectory, or evaluate a frozen checkpoint.

    Attributes:
        lineage: Project-local versioned memory sequence.
        mode: ``learn`` pins the current head and publishes the next version after a
            completed trajectory; ``frozen`` reads an exact checkpoint and never publishes.
        checkpoint: Exact frozen version; absent in learning mode.
        reserved: Frozen evaluation that declares this task lineage reserved: it is refused if
            the lineage already consumed the task lineage, and later learning of it is refused.
            Without it a frozen run is a diagnostic and relies on same-trajectory withholding.
        hide_task_lineage: Learning only: memory from earlier trajectories of this task lineage
            is hidden from the run, so every prediction is made on a task memory has not seen
            (test-then-train), the same exclusion recorded-evidence retrieval applies.
        withhold_own: False only for an unreserved frozen retest that reads what memory learned
            from this very trajectory: a retention check, never a generalization measure.
        overview_lines: Line budget of the memory overview delivered with every prediction.
        maximum_compressions: Compression calls a learning run may buy after publication.
    """

    lineage: str = Field(pattern=r"^[a-z][a-z0-9-]{0,63}$")
    mode: Literal["learn", "frozen"]
    checkpoint: int | None = Field(default=None, ge=1, strict=True)
    reserved: bool = Field(default=False, strict=True, exclude_if=lambda value: not value)
    hide_task_lineage: bool = Field(default=False, strict=True, exclude_if=lambda value: not value)
    withhold_own: bool = Field(default=True, strict=True, exclude_if=lambda value: value)
    overview_lines: int = Field(default=32, ge=1, strict=True)
    maximum_compressions: int = Field(default=16, ge=0, strict=True)

    @model_validator(mode="after")
    def checkpoint_mode(self) -> "FidelityMemory":
        """Reserved evaluation names its checkpoint; learning always follows the head."""
        if (self.mode == "frozen") != (self.checkpoint is not None):
            raise ValueError("frozen memory requires a checkpoint; learning uses the head")
        if self.reserved and self.mode != "frozen":
            raise ValueError("only frozen evaluation can declare a reserved task lineage")
        if self.hide_task_lineage and self.mode != "learn":
            raise ValueError("hiding the task lineage applies only while learning")
        if not self.withhold_own and (self.mode != "frozen" or self.reserved):
            raise ValueError("only an unreserved frozen retest may read its own trajectory")
        return self


class FidelitySpec(ContractModel):
    """Fixed actions and independent references for an existing simulation.

    Attributes:
        run_id: New logical run identity; repeated completed runs replay saved results.
        world_model: Exact existing simulation manifest, including rules and retrieval policy.
        task: Public instruction and current observations, without a reference answer.
        workload: Fixed supplied actions, not a live actor-equivalence experiment.
        references: Evaluator-only outcomes; never passed into prediction or retrieval.
        judge: Frozen five-axis rubric and prompt.
        judge_alias: Configured model used only for independent judging.
        deadline: Absolute dispatch and commit cutoff.
        maximum_physical_attempts: Global ceiling including provider retries and judging.
        maximum_output_tokens: Explicit per-generation output allowance.
        json_object_output: Declared JSON-mode treatment for supported prediction routes.
        output_protocol: Plain text framing or typed native-value proposals.
        request_reservation_usd: Accounting estimate, not an asserted tariff upper bound.
        online_research: Optional declared per-step reply feedback; absent freezes knowledge.
        retrieval: Native JEV selection for originals and memory, resolved from the catalog.
        memory: Cross-trajectory memory lineage and its learning or frozen mode.
        roles: Per-role output and reasoning budgets; absent, every role uses
            ``maximum_output_tokens`` and the catalog's reasoning configuration.
        proposal_reasks: Times a refused simulator proposal is asked again with the refusal
            reason, within the same turn; refused output is never repaired.
        judged: False skips the judge; the report records it as not run. Learning uses it:
            its score is exact replies, so a judge call per trace would buy nothing.
    """

    run_id: str = Field(pattern=r"^[a-z][a-z0-9-]{0,79}$")
    world_model: ArtifactInput
    task: TaskCase
    workload: FixedWorkflow
    references: EvaluationReferences
    judge: JudgeConfig
    judge_alias: str = Field(min_length=1)
    deadline: AwareDatetime
    maximum_physical_attempts: int = Field(gt=0, strict=True)
    maximum_output_tokens: int = Field(gt=0, strict=True)
    json_object_output: bool = Field(default=False, strict=True)
    output_protocol: Literal["text", "json-tool"] = "text"
    request_reservation_usd: float = Field(gt=0, allow_inf_nan=False)
    online_research: OnlineResearchPolicy | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    retrieval: FidelityRetrieval | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    memory: FidelityMemory | None = Field(default=None, exclude_if=lambda value: value is None)
    roles: RoleBudgets | None = Field(default=None, exclude_if=lambda value: value is None)
    proposal_reasks: int = Field(default=0, ge=0, le=3, strict=True, exclude_if=lambda v: not v)
    judged: bool = Field(default=True, strict=True, exclude_if=lambda value: value)

    @model_validator(mode="after")
    def separate_references(self) -> "FidelitySpec":
        """Bind action inventories before any model or credential is consulted."""
        if self.output_protocol == "json-tool" and self.json_object_output:
            raise ValueError("Typed proposal and JSON text modes are distinct protocols")
        if self.task.reference_answer is not None or self.task.instruction != self.workload.task:
            raise ValueError("public task must match workload and exclude reference answers")
        if self.references.steps and (
            len(self.references.steps) != len(self.workload.actions)
            or any(
                reference.action != {"name": action.name, "arguments": action.arguments}
                for reference, action in zip(
                    self.references.steps, self.workload.actions, strict=True
                )
            )
        ):
            raise ValueError("references must bind exactly the supplied public actions")
        if self.online_research is not None and (
            len(self.references.steps) != len(self.workload.actions)
            or any(not row.reply_available for row in self.references.steps)
        ):
            raise ValueError("online research requires explicitly disclosed post-step replies")
        if self.memory is not None and self.retrieval is None:
            raise ValueError("memory delivery requires a registered JEV retrieval policy")
        if self.memory is not None and (self.memory.mode == "learn") != (
            self.online_research is not None
        ):
            raise ValueError("memory learning needs feedback; frozen checkpoints exclude it")
        if (
            self.retrieval is not None
            and self.online_research is not None
            and (self.online_research.selection_policy_sha256 != self.memory_selection_identity)
        ):
            raise ValueError("online research must register the spec's JEV memory selector")
        return self

    def budget(self, role: Literal["prediction", "investigation", "judging"]) -> RoleBudget:
        """The registered budget for one role, or the shared legacy allowance."""
        if self.roles is None:
            return RoleBudget(output_tokens=self.maximum_output_tokens)
        return getattr(self.roles, role)

    @property
    def memory_selection_identity(self) -> str | None:
        """Identity of the JEV memory selector this spec constructs, without a provider."""
        if self.retrieval is None:
            return None
        return JEVCueSelector(
            policy=self.retrieval.jev,
            top_k=self.retrieval.memory_top_k,
            decide=_never,
            record=_discard,
        ).identity


def _never(request: DecisionRequest) -> JsonObject:
    """Identity-only selector seam; never dispatches."""
    raise AssertionError("identity selector cannot decide")


def _discard(receipt: JsonObject) -> None:
    """Identity-only selector seam; records nothing."""


class PreparedFidelity(ContractModel):
    """Immutable evaluation registration, without credentials or another execution loop.

    Attributes:
        id: Content-derived plan identity.
        spec: Complete frozen evaluation inputs and limits.
        catalog_sha256: Model configuration identity; credential references stay local.
        code_revision: Installed producer identity.
        embedder_alias: Configured alias embedding retrieval queries for a semantic index;
            absent for the local hashing index.
    """

    id: str
    spec: FidelitySpec
    catalog_sha256: str
    code_revision: str
    embedder_alias: str | None = Field(default=None, exclude_if=lambda value: value is None)


def prepare_fidelity_evaluation(
    project: ProjectStore, spec: FidelitySpec, *, code_revision: str
) -> PreparedFidelity:
    """Verify and freeze a workload without provider calls or credential resolution.

    Args:
        project: Existing project owning the simulation and model catalog.
        spec: Public workload plus separately typed evaluator references.
        code_revision: Exact producing implementation identity.

    Returns:
        Saved registration; reusing a run ID with different inputs is rejected.
    """
    spec = FidelitySpec.model_validate_json(spec.model_dump_json())
    plan_id = stable_id("fidelity", {"run_id": spec.run_id})
    if project.artifacts.exists(plan_id):
        saved = PreparedFidelity.model_validate_json(
            project.artifacts.read_bytes(plan_id, "plan.json")
        )
        if saved.spec != spec:
            raise ValueError("fidelity run inputs changed; use the saved spec or a new run ID")
        artifact = load_grounded_world_model_artifact(project.artifacts, spec.world_model)
        # A registration may predate a guard: recheck what the predictor could see.
        _require_unexposed(project, artifact, spec)
        if spec.memory is not None and spec.memory.reserved:
            _validate_memory(project, spec, artifact)
            _establish_reservation(project, spec, code_revision)
        return saved
    manifest = project.artifacts.read(spec.world_model.artifact_id).manifest
    if artifact_input(manifest) != spec.world_model:
        raise ValueError("simulation manifest differs from the supplied frozen reference")
    artifact = load_grounded_world_model_artifact(project.artifacts, spec.world_model)
    catalog = load_model_catalog(project.model_catalog_path)
    registry = RuntimeModelCatalog(catalog)
    predictor, predictor_caps = registry.snapshot(artifact.model_alias)
    judge_model, judge_caps = registry.snapshot(spec.judge_alias)
    for role, caps in (
        ("prediction", predictor_caps),
        ("investigation", predictor_caps),
        ("judging", judge_caps),
    ):
        _check_budget(role, spec.budget(role), caps)
    if predictor != artifact.model:
        raise ValueError("configured predictor differs from the simulation; bind a new artifact")
    if spec.judge.model_identity != judge_model.model_id or judge_caps.supports_tools is not True:
        raise ValueError("judge must match the rubric model and support named structured output")
    for alias in (artifact.model_alias, spec.judge_alias):
        connection = catalog.connections[catalog.models[alias].connection]
        if connection.provider != "openai-compatible":
            raise ValueError("this fidelity execution profile requires openai-compatible HTTP")
    bound = artifact.selection_policy_sha256
    if bound is not None and (spec.retrieval is None or spec.retrieval.jev.identity != bound):
        raise ValueError("simulator binds a JEV reranker; register the same policy in retrieval")
    if spec.retrieval is not None:
        record = catalog.models.get(spec.retrieval.jev_alias)
        if record is None or catalog.connections[record.connection].provider != "typesafe":
            raise ValueError("JEV retrieval requires a configured native decision-model alias")
    _validate_memory(project, spec, artifact)
    _require_unexposed(project, artifact, spec)
    prepared = PreparedFidelity(
        id=plan_id,
        spec=spec,
        catalog_sha256=sha256_json(catalog.model_dump(mode="json")),
        code_revision=code_revision,
        embedder_alias=index_embedder_alias(project, catalog, artifact),
    )
    project.artifacts.write_json(
        artifact_id=plan_id,
        artifact_type="fidelity-plan",
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            inputs=(spec.world_model,),
            code_revision=code_revision,
        ),
        files={"plan.json": prepared.model_dump(mode="json")},
    )
    _establish_reservation(project, spec, code_revision)
    return prepared


def _validate_memory(
    project: ProjectStore, spec: FidelitySpec, artifact: GroundedWorldModelArtifact
) -> None:
    """Check partition, checkpoint and knowledge before anything is written."""
    if spec.memory is None:
        return
    check_partition(project, spec)
    pinned = (
        memory_head(project.artifacts, spec.memory.lineage)
        if spec.memory.checkpoint is None
        else read_memory_version(project.artifacts, spec.memory.lineage, spec.memory.checkpoint)
    )
    require_memory_scope(
        pinned,
        world_model=spec.world_model,
        artifact=artifact,
        selection_policy_sha256=str(spec.memory_selection_identity),
    )


def _establish_reservation(project: ProjectStore, spec: FidelitySpec, code_revision: str) -> None:
    """Register a declared held-out panel only after validation, under the lineage lock."""
    if spec.memory is None or not spec.memory.reserved:
        return
    reserve_task_lineage(
        project.artifacts,
        spec.memory.lineage,
        spec.task.lineage_group_id,
        created_at=datetime.now(UTC),
        code_revision=code_revision,
    )


def saved_fidelity_report(project: ProjectStore, prepared: PreparedFidelity) -> DualReport | None:
    """Read completed evidence without credentials, provider calls, or a live deadline."""
    identity = prepared.id + "-report"
    if not project.artifacts.exists(identity):
        return None
    return DualReport.model_validate_json(project.artifacts.read_bytes(identity, "report.json"))


def execute_fidelity_evaluation(
    project: ProjectStore,
    prepared: PreparedFidelity,
    *,
    progress: ProgressHook | None = None,
    research_selection: EvidencePolicy | None = None,
    clock: Callable[[], datetime] | None = None,
) -> DualReport:
    """Run after explicit spend consent, or replay a completed evaluation exactly.

    Args:
        project: Owner of the verified registration and durable execution artifacts.
        prepared: Exact saved plan. The caller authorizes unpriced spend explicitly;
            physical-attempt, per-output, and deadline bounds remain enforced.
        progress: Existing observational progress hook shared with the CLI.
        research_selection: Injected session-only selector for specs without retrieval.
        clock: Optional aware clock for qualification.

    Returns:
        Separate mechanical and mandatory judge results. Incomplete earlier dispatches
        stop visibly rather than authorizing an automatic repurchase.
    """
    now = clock or (lambda: datetime.now(UTC))
    saved = PreparedFidelity.model_validate_json(
        project.artifacts.read_bytes(prepared.id, "plan.json")
    )
    if saved != prepared:
        raise ValueError("execution differs from the saved fidelity registration")
    directory = project.artifacts.project_directory
    with FileLock(directory / (prepared.id + ".lock"), timeout=0):
        previous = saved_fidelity_report(project, prepared)
        intent = prepared.id + "-execution"
        if previous is not None:
            report(progress, "Saved fidelity result")
            consume_trajectory(project, prepared, now())
            return previous
        if project.artifacts.exists(intent):
            raise ValueError(
                "incomplete fidelity run; inspect saved operations before new dispatch "
                "(a derived held-out or learning run is bought again only with --retry-incomplete)"
            )
        spec = prepared.spec
        _require_unexposed(
            project, load_grounded_world_model_artifact(project.artifacts, spec.world_model), spec
        )
        if spec.retrieval is not None and research_selection is not None:
            raise ValueError("registered JEV retrieval cannot be replaced by an injected selector")
        if (
            spec.online_research is not None
            and spec.retrieval is None
            and research_selection is None
        ):
            raise ValueError("online research needs its registered memory selector")
        if (
            spec.online_research is not None
            and research_selection is not None
            and (research_selection.identity != spec.online_research.selection_policy_sha256)
        ):
            raise ValueError("memory selector differs from the registered fidelity spec")
        if now() >= spec.deadline:
            raise ValueError("fidelity deadline expired before dispatch")
        identity = sha256_json(prepared.model_dump(mode="json"))
        ledger = DispatchLedger(
            directory / (prepared.id + "-physical.sqlite"),
            spec.maximum_physical_attempts,
            identity,
        )
        transport = AdmittedJsonTransport(
            HttpxJsonTransport(),
            ledger=ledger,
            artifacts=project.artifacts,
            code_revision=prepared.code_revision,
            deadline=spec.deadline,
            clock=now,
        )
        catalog = load_model_catalog(project.model_catalog_path)
        if sha256_json(catalog.model_dump(mode="json")) != prepared.catalog_sha256:
            raise ValueError("model configuration changed; prepare a new fidelity run")
        registry = RuntimeModelCatalog(catalog, transport_factory=lambda: transport)
        artifact = load_grounded_world_model_artifact(project.artifacts, spec.world_model)
        predictor = role_model(registry, catalog, artifact.model_alias, spec, "prediction")
        budget = RequestBudget(project, identity=identity, maximum_cost_usd=None)
        originals: JEVSelector | None = None
        prior: MemoryVersion | None = None
        prior_identity: JsonObject | None = None
        prior_records: dict[str, JsonObject] = {}
        prior_overview: list[JsonObject] = []
        overview_refs: tuple[ArtifactInput, ...] = ()
        prior_sources: dict[str, tuple[ArtifactInput, ...]] = {}
        overview_view: Callable[[frozenset[str]], list[JsonObject]] | None = None
        overview_sources_of: Callable[[Sequence[JsonObject]], Sequence[ArtifactInput]] | None = None
        if spec.retrieval is not None:
            decide = decision_caller(
                registry.resolve(spec.retrieval.jev_alias), transport, spec.deadline, now
            )
            record = _receipt_recorder(project, prepared, now)
            if artifact.selection_policy_sha256 is not None:
                originals = JEVSelector(policy=spec.retrieval.jev, decide=decide, record=record)
            if spec.memory is not None or spec.online_research is not None:
                research_selection = JEVCueSelector(
                    policy=spec.retrieval.jev,
                    top_k=spec.retrieval.memory_top_k,
                    decide=decide,
                    record=record,
                )
        if spec.memory is not None:
            prior = (
                memory_head(project.artifacts, spec.memory.lineage)
                if spec.memory.checkpoint is None
                else read_memory_version(
                    project.artifacts, spec.memory.lineage, spec.memory.checkpoint
                )
            )
            require_memory_scope(
                prior,
                world_model=spec.world_model,
                artifact=artifact,
                selection_policy_sha256=str(spec.memory_selection_identity),
            )
            materialize_log(project.artifacts, prior)
            check_partition(project, spec)
            if spec.memory.reserved:
                reserve_task_lineage(
                    project.artifacts,
                    spec.memory.lineage,
                    spec.task.lineage_group_id,
                    created_at=now(),
                    code_revision=prepared.code_revision,
                )
            rule = Eligibility(
                exclude_public=public_trajectory_identity(trajectory_record(spec))
                if spec.memory.withhold_own
                else None,
                reserved_task_lineages=reserved_task_lineages(
                    project.artifacts, spec.memory.lineage
                ),
                hidden_task_lineage=spec.task.lineage_group_id
                if spec.memory.hide_task_lineage
                else None,
                strict=spec.memory.reserved,
            )
            view = memory_view(project.artifacts, prior, rule)
            prior_records, withheld = view.records, view.withheld
            total = 0 if prior is None else prior.log_length or 0
            sources: dict[ArtifactInput, bool] = {}

            def admits(reference: ArtifactInput) -> bool:
                """One cached eligibility decision per typed source reference."""
                if reference not in sources:
                    sources[reference] = rule.admits(load_source(project.artifacts, reference))
                return sources[reference]

            lineage, rows = spec.memory.lineage, spec.memory.overview_lines
            generations = summary_generations(prior)
            pinned = pinned_summaries(
                project.artifacts,
                prior,
                created_at=now(),
                code_revision=prepared.code_revision,
            )

            def render(extra_retired: frozenset[str]) -> list[JsonObject]:
                """Serving overview: admissible lines, minus every retired hypothesis."""
                retired = set(view.retired) | extra_retired
                return overview(
                    project.artifacts,
                    lineage,
                    total,
                    rows,
                    admits=admits,
                    admits_line=lambda line: line.origin not in retired,
                    generations=generations,
                    pinned=pinned,
                )

            prior_overview = render(frozenset())
            overview_refs = overview_sources(
                project.artifacts, lineage, prior_overview, generations
            )
            prior_sources, overview_view = view.sources, render

            def sources_of(delivered: Sequence[JsonObject]) -> Sequence[ArtifactInput]:
                """Sources behind exactly the overview rows a prediction received."""
                return overview_sources(project.artifacts, lineage, delivered, generations)

            overview_sources_of = sources_of
            prior_identity = {
                "lineage": spec.memory.lineage,
                "mode": spec.memory.mode,
                "reserved": spec.memory.reserved,
                # Recorded only when set, so earlier memory identities stay unchanged.
                **(
                    {"hidden_task_lineage": rule.hidden_task_lineage}
                    if rule.hidden_task_lineage is not None
                    else {}
                ),
                **({} if spec.memory.withhold_own else {"withhold_own": False}),
                "version": 0 if prior is None else prior.version,
                "consumed_trajectories": 0 if prior is None else len(prior.consumed),
                "library": None if prior is None else prior.library.model_dump(mode="json"),
                "withheld_same_trajectory": withheld,
                "log_length": total,
                "excluded_sources": sorted(r.artifact_id for r, ok in sources.items() if not ok),
                "unknown_provenance": provenance_gaps(project.artifacts, prior),
                "retired": view.retired,
                "overview_sha256": sha256_json(prior_overview),
            }

        def completion(
            resolved: ResolvedModel,
            *,
            structured: bool,
            role: Literal["prediction", "investigation", "judging"],
        ) -> StructuredCompletion:
            """Share provider resolution, physical admission and logical accounting."""
            return StructuredCompletion(
                client=resolved.client,
                plan=CompletionPlan(
                    model=resolved.snapshot,
                    served_model_id=resolved.served_model_id,
                    capabilities=resolved.capabilities,
                    deadline=spec.deadline,
                    output_tokens=spec.budget(role).output_tokens,
                    maximum_cost_per_call=spec.request_reservation_usd,
                    code_revision=prepared.code_revision,
                    output_protocol="tool" if structured else "json",
                ),
                budget=budget,
                artifacts=project.artifacts,
                clock=now,
            )

        prediction = completion(predictor, structured=False, role="prediction")
        investigating = role_model(registry, catalog, artifact.model_alias, spec, "investigation")
        judging = completion(
            role_model(registry, catalog, spec.judge_alias, spec, "judging"),
            structured=True,
            role="judging",
        )
        runtime = load_grounded_world_model(
            project.artifacts,
            spec.world_model.artifact_id,
            client=predictor.client,
            capabilities=predictor.capabilities,
            embedder=bind_index_embedder(
                registry, prepared.embedder_alias, artifact, project, budget
            ),
            evidence_selector=originals,
        )
        project.artifacts.write_json(
            artifact_id=intent,
            artifact_type="fidelity-execution",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=now(),
                code_revision=prepared.code_revision,
                inputs=(artifact_input(project.artifacts.read(prepared.id).manifest),),
            ),
            files={
                "execution.json": {
                    "plan_sha256": identity,
                    **({"memory": prior_identity} if prior_identity is not None else {}),
                }
            },
        )
        report(progress, "Loading simulator and memory")
        session = open_grounded_session(
            runtime=runtime,
            task=spec.task,
            store=SessionStore(directory / (prepared.id + "-sessions.sqlite"), clock=now),
            artifacts=project.artifacts,
            session_id=prepared.id,
            policy=SessionPolicy(
                deadline=spec.deadline,
                maximum_operations=len(spec.workload.actions),
                maximum_reasks=spec.proposal_reasks,
                inference_identity=identity,
                code_revision=prepared.code_revision,
            ),
            complete=ModelCompletion(prediction),
            lock_path=directory / (prepared.id + "-session.lock"),
            maximum_output_tokens=spec.budget("prediction").output_tokens,
            json_object_output=spec.json_object_output,
            output_protocol=spec.output_protocol,
            served_model_id=predictor.served_model_id,
            research_policy=spec.online_research,
            research_selection=research_selection,
            research_complete=completion(investigating, structured=True, role="investigation")
            if spec.online_research is not None
            else None,
            research_prior=prior_records if prior_identity is not None else None,
            research_overview=prior_overview,
            research_prior_sources=prior_sources,
            research_overview_sources=overview_refs,
            research_overview_view=overview_view,
            research_overview_sources_of=overview_sources_of,
            research_prior_identity=prior_identity,
            clock=now,
        )

        def judge_call(body: JsonObject, logical_id: str) -> JsonObject:
            """Expose the real judging stage without adding a retry or scoring loop."""
            if not spec.judged:
                raise JudgeNotRun("judging is skipped for this run; it is scored by exact replies")
            report(progress, "Judging")
            return judging(body, logical_id)

        result = evaluate_workflow(
            session,
            spec.workload,
            references=spec.references,
            judge=spec.judge,
            judge_complete=judge_call,
            artifacts=project.artifacts,
            clock=now,
            code_revision=prepared.code_revision,
            evaluation_id=prepared.id + "-report",
            reply_feedback_source=spec.online_research.feedback_source
            if spec.online_research is not None
            else None,
            progress=progress,
            feedback_after_predictions=spec.online_research is not None
            and not spec.online_research.notes_reach_predictions,
        )
        learning = spec.memory is not None and spec.memory.mode == "learn"
        if learning:
            report(progress, "Publishing memory")
        consume_trajectory(project, prepared, now())
        if learning:
            report(progress, "Compressing memory")
            maintain_memory(
                project,
                prepared,
                completion(investigating, structured=True, role="investigation"),
                now(),
            )
        return result


def _require_unexposed(
    project: ProjectStore, artifact: GroundedWorldModelArtifact, spec: FidelitySpec
) -> None:
    """Refuse a task whose own evidence the simulator could retrieve or knows.

    Runs at registration, on reuse of a saved registration, and before any new dispatch, so
    a registration created before a guard existed is checked too.
    """
    require_unindexed_held_out_task(project, artifact, spec.task)
    require_unexposed_task(project, artifact, spec.task)


def role_model(
    registry: RuntimeModelCatalog,
    catalog: ModelCatalog,
    alias: str,
    spec: FidelitySpec,
    role: Literal["prediction", "investigation", "judging"],
) -> ResolvedModel:
    """Resolve an alias with this role's registered reasoning effort, if any.

    The effort replaces the catalog capability pin only; capability identity excludes
    reasoning effort, so model and artifact bindings are unchanged.
    """
    effort = spec.budget(role).reasoning_effort
    if effort is None:
        return registry.resolve(alias)
    record = catalog.models[alias]
    if record.capabilities is None:
        raise ValueError(f"{role} model {alias!r} has no capability record for reasoning")
    adjusted = record.capabilities.model_copy(update={"reasoning_effort": effort})
    patched = catalog.model_copy(
        update={
            "models": {
                **catalog.models,
                alias: record.model_copy(update={"capabilities": adjusted}),
            }
        }
    )
    return registry.with_catalog(patched).resolve(alias)


def _check_budget(role: str, budget: RoleBudget, caps: ModelCapabilities) -> None:
    """Refuse allowances or reasoning settings the role's model cannot honor."""
    if caps.maximum_output_tokens is not None and budget.output_tokens > caps.maximum_output_tokens:
        raise ValueError(f"{role} output allowance exceeds the model's maximum output")
    if budget.reasoning_effort is not None and (
        caps.supports_reasoning is not True or caps.reasoning_effort is None
    ):
        raise ValueError(f"{role} model does not accept a configured reasoning effort")


def _receipt_recorder(
    project: ProjectStore, prepared: PreparedFidelity, clock: Callable[[], datetime]
) -> Callable[[JsonObject], None]:
    """Retain exact JEV requests, raw answers and selected IDs as immutable receipts."""

    def record(receipt: JsonObject) -> None:
        """Persist one content-addressed receipt beside the run's other artifacts."""
        identity = stable_id("jev-receipt", {"run": prepared.id, "receipt": receipt})
        if not project.artifacts.exists(identity):
            project.artifacts.write_json(
                artifact_id=identity,
                artifact_type="jev-receipt",
                envelope=ArtifactEnvelope(
                    schema_version=1, created_at=clock(), code_revision=prepared.code_revision
                ),
                files={"receipt.json": {"run": prepared.id, **receipt}},
            )

    return record
