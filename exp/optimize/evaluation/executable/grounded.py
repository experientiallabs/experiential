"""Built-in executable factory for a frozen learned simulator with admitted provider calls.

This factory supplies simulated state only. Native implementations can compose its world with
MixedRealityEnvironmentRuntime; domain ownership forbids pretending simulated effects changed
native files. Private probes here describe simulated state, never native truth.

Instance facts are declared explicitly in the definition, with the fidelity-owned
``ReplayFacts`` model and its validation and reservation rules. They condition every
prediction exactly as in fidelity replay and are never derived from native state.

A rule book is bound the same way: the definition pins one immutable artifact holding the
book as ``rules.json``, and the runtime verifies that exact artifact before every proposal is
judged against it.

A tag retrieval binding is pinned likewise: the definition names one saved
``tag-retrieval-binding`` artifact, the runtime verifies its manifest and type, and the grounded
world ranks the task's retrieval by its policy when the binding lists the task.

Validated transition programs compose the same way: the definition may carry one
:class:`~exp.simulation.induction.contracts.InducedCoverage`. Covered tools then run as exact code
in a :class:`~exp.simulation.induction.sandbox.ProgramSandbox` inside their validated domain;
every other call, and every call outside a program's domain, is predicted by the grounded model
under the bound rule book. The coverage is part of the definition, so every identity derived from
it, including the world digest a judge binds, covers the programs.

A rule memory set is bound likewise: the definition pins one saved ``rule-memory-set`` artifact
and a recall policy. The runtime verifies every file against its digest, and before each
prediction a native Jev recall selects only the rule files that govern the current call, within
a byte budget. Recall decisions go through the project decision cache, so a replayed step makes
no call, and every step's receipt (scores, selection, cost) is retained. The binding is part of
the definition, so the runtime identity and the world digest cover the exact memory set.

A state memory policy is bound the same way: before each prediction an admitted open-weight
author describes the task's current visible state, once per distinct state, and the description
is attached beside the recalled rules within the policy's byte budget. Its completions are bought
once through the project decision cache, so a replayed step makes no call. The policy is part of
the definition, so the runtime identity and the world digest cover it.
"""

import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Self

from pydantic import AwareDatetime, Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    JsonValue,
    Sha256,
    canonical_json_bytes,
    sha256_json,
)
from exp.common.models import ModelCatalog, ModelRequest, ReasoningEffort, load_model_catalog
from exp.common.project import ArtifactStore, ProjectStore
from exp.common.project.manifests import artifact_input
from exp.common.release_revision import installed_execution_revision
from exp.common.tasks import TaskCase
from exp.optimize.evaluation.executable.loader import ContinuationRuntime
from exp.optimize.evaluation.executable.provider_grant import ProviderGrant
from exp.optimize.evaluation.executable.state_recall import (
    StateRecallBinding,
    bind_state_recall,
    check_state_recall,
)
from exp.optimize.evaluation.fidelity.decision_access import decision_caller, receipt_recorder
from exp.optimize.evaluation.fidelity.instance_facts import ReplayFacts, require_fact_scope
from exp.optimize.evaluation.fidelity.retrieval_binding import index_embedder_alias
from exp.optimize.evaluation.learning.profile import EvaluationMemory, bind_evaluation_memory
from exp.optimize.evaluation.rules.memory_admission import load_guard_table
from exp.optimize.evaluation.variance.build_records import load_tag_retrieval_binding
from exp.runtime.environments.world_contracts import (
    ExecutableCheckpoint,
    NativeImage,
    WorldManifest,
)
from exp.runtime.environments.worlds import ExecutableWorldRuntime, WorldBackendFactory
from exp.runtime.gateway.decisions_contracts import DecisionRequest
from exp.runtime.models.budget import RequestBudget
from exp.runtime.models.call_evidence import call_consumer
from exp.runtime.models.deadlines import dispatch_deadline
from exp.runtime.models.providers.admitted_transport import (
    AdmittedJsonTransport,
    require_admitted_transport,
)
from exp.runtime.models.providers.transport import HttpxJsonTransport
from exp.runtime.models.reasoning_view import reasoning_catalog
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.runtime.models.structured_completion import CompletionPlan, StructuredCompletion
from exp.runtime.sessions.dispatch_ledger import DispatchLedger
from exp.simulation.continuations import WorldCheckpoint, load_continuation_checkpoint
from exp.simulation.development_scope import require_development_task
from exp.simulation.induction.backend import InducedWorldFactory
from exp.simulation.induction.contracts import InducedCoverage
from exp.simulation.induction.sandbox import ProgramSandbox
from exp.simulation.mining.tagging import DecisionCache
from exp.simulation.reconstruction.models import resolve
from exp.simulation.retrieval import RAGEmbedderBinding
from exp.simulation.retrieval.evidence_reading import EvidenceReader
from exp.simulation.retrieval.tags import TagRetrievalBinding
from exp.simulation.rules.admission import GuardTable, guard_rulebook
from exp.simulation.rules.memory_set import MemorySet, load_memory_set
from exp.simulation.rules.model import RuleBook
from exp.simulation.rules.recall import RECALL_PROTOCOL, RecallPolicy, RuleRecall
from exp.simulation.rules.rulebook import RULES_JSON
from exp.simulation.world_model.artifact import GroundedWorldModelArtifact
from exp.simulation.world_model.executable import GroundedWorldFactory, PredictorEvent
from exp.simulation.world_model.grounded_session import GroundedTransitionEngine
from exp.simulation.world_model.prediction_reading import PredictionReading, PredictionResolver
from exp.simulation.world_model.runtime import (
    load_grounded_world_model,
    load_grounded_world_model_artifact,
)
from exp.simulation.world_model.state_memory import StateMemory, StateMemoryPolicy

_PREDICTOR_PART = "predictor.json"


class RuleMemoryBinding(ContractModel):
    """A pinned rule memory set and the per-step recall policy over it.

    Attributes:
        memory_set: Saved ``rule-memory-set`` artifact; its manifest digests every file.
        recall: Frozen recall policy, naming the admitted native decision alias.
        guards: The family's saved ``rule-guard-table`` from tag admission. Its checkable
            entries join the rule book every proposal must satisfy before it commits (through
            ``guard_rulebook`` and ``check_proposed_transition``). Absent, the binding and
            every derived identity are unchanged.
    """

    memory_set: ArtifactInput
    recall: RecallPolicy
    guards: ArtifactInput | None = Field(default=None, exclude_if=lambda value: value is None)


class GroundedRuntimeDefinition(ContractModel):
    """Frozen simulator, owned domains, private projection and finite provider allowance.

    Attributes:
        task: Public development task, without evaluator answers.
        world_model: Exact build-produced grounded artifact, including its retrieval index.
        catalog_sha256: Frozen connection/model definitions, excluding credential values.
        manifest: Domain ownership and per-tool effects; this factory owns simulated domains.
        initial_state: Exact initial JSON value of each simulated domain.
        facts: Optional validated instance facts describing the episode's starting world, with
            ``first_version`` the state version of that start. Every prediction receives the
            conditioning facts in their own channel; they are not state and cannot be edited.
            Absent facts leave the definition, and so every derived identity, unchanged.
        research_history: False ablates current research delivery without mutating the bank.
        prediction_reading: Optional bounded reading of eligible originals before prediction.
        memory: Explicit version pinned before capture; None means no learned memory is used.
        probe_paths: Evaluator field names mapped to JSON pointers into simulated domain state.
            Initial pointers must resolve. Missing later values are omitted as unavailable evidence.
        seed: World scheduler seed; does not make future model responses deterministic.
        maximum_output_tokens: Per-prediction output ceiling.
        reasoning_effort: Optional explicit predictor-only profile; None retains catalog behavior.
        maximum_reasks: Explicit extra proposals after known contract refusals, within the
            same physical budget. Never retries unknown outcomes or a valid scored result.
        maximum_physical_attempts: Cumulative durable ceiling, including query/recall retries.
        deadline: Shared deadline for every provider call made by this definition.
        rule_book: Optional immutable artifact holding a learned rule book as ``rules.json``.
            Every proposal must satisfy its supported rules before it commits; a rejected
            proposal is re-asked with the broken rules within ``maximum_reasks``. Absent, the
            definition and every derived identity are unchanged.
        tag_retrieval: Optional saved ``tag-retrieval-binding`` artifact. When it lists the
            task, retrieval ranks by its tag policy and records the policy as evidence; a task it
            does not list retrieves by cosine similarity. Absent, the definition and every
            derived identity are unchanged.
        programs: Optional validated transition programs, validated under exactly this
            manifest. Covered tools run as exact sandboxed code inside their validated domain;
            everything else goes to the grounded model and its rule-book check. Absent, the
            definition and every derived identity are unchanged.
        rule_memory: Optional pinned rule memory set with its recall policy. Each prediction
            receives only the rule files recalled for its call, alongside the task memory.
            Absent, the definition and every derived identity are unchanged.
        state_memory: Optional per-task state description policy, naming an admitted
            completion alias. Each prediction receives the description of the current visible
            state, refreshed when it changes. Absent, the definition and every derived identity
            are unchanged.
        state_recall: Optional pinned state-tag index and policy. Each prediction receives the
            recorded transitions whose pre-turn state agrees best with the live one, tagged by
            the native decision alias through the decision cache. Absent, the definition and
            every derived identity are unchanged.
    """

    task: TaskCase
    world_model: ArtifactInput
    catalog_sha256: Sha256
    manifest: WorldManifest
    initial_state: JsonObject
    facts: ReplayFacts | None = Field(default=None, exclude_if=lambda value: value is None)
    memory: EvaluationMemory | None = None
    research_history: bool = Field(default=True, strict=True, exclude_if=lambda value: value)
    prediction_reading: PredictionReading | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    probe_paths: dict[str, str] = Field(min_length=1, max_length=128)
    seed: int = 0
    maximum_output_tokens: int = Field(default=4096, ge=1, le=65536)
    reasoning_effort: ReasoningEffort | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    maximum_reasks: int = Field(default=0, ge=0, le=3, strict=True)
    maximum_physical_attempts: int = Field(ge=1, le=100_000)
    deadline: AwareDatetime
    rule_book: ArtifactInput | None = Field(default=None, exclude_if=lambda value: value is None)
    tag_retrieval: ArtifactInput | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    programs: InducedCoverage | None = Field(default=None, exclude_if=lambda value: value is None)
    rule_memory: RuleMemoryBinding | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    state_memory: StateMemoryPolicy | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    state_recall: StateRecallBinding | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def _starting_facts_only(self) -> Self:
        """Refuse facts an executable episode cannot honor or that shadow native receipts."""
        if self.programs is not None and self.programs.manifest != self.manifest:
            raise ValueError(
                "transition programs were validated under another world manifest; re-validate "
                "them under this runtime's manifest"
            )
        if self.facts is None:
            return self
        for fact in self.facts.facts.facts:
            if fact.provenance.state_version > self.facts.first_version:
                raise ValueError(
                    f"instance fact {fact.key!r} describes state version "
                    f"{fact.provenance.state_version}, after the episode's starting version "
                    f"{self.facts.first_version}; an executable actor chooses its own actions, "
                    "so declare only facts about the starting world"
                )
            if fact.key.startswith("native-read/"):
                raise ValueError(
                    f"instance fact {fact.key!r} uses the reserved native-read/ namespace of "
                    "native read receipts; rename the fact key"
                )
        return self

    @property
    def probe_sha256(self) -> Sha256:
        """Bind a pure projection whose authority is explicitly simulated."""
        return sha256_json({"kind": "simulated-domain-projection", "paths": self.probe_paths})


def simulated_probe(state: JsonObject, paths: Mapping[str, str]) -> JsonObject:
    """Project registered simulated state, omitting fields the state no longer has.

    Initial pointers are validated before execution. A later absence is unknown evidence, not
    a null value, a successful effect, or grounds to discard a valid actor observation.

    Args:
        state: Simulated domain state of one checkpoint.
        paths: Evaluator field names mapped to JSON pointers into that state.

    Returns:
        Every resolvable field by name; unresolvable fields are omitted.
    """
    values: JsonObject = {}
    for name, path in paths.items():
        try:
            values[name] = resolve(state, path)
        except ValueError:
            continue
    return values


def simulated_image(checkpoint: ExecutableCheckpoint) -> tuple[JsonObject, list[JsonValue]]:
    """Decode the simulated state and world-model history of one grounded checkpoint.

    Args:
        checkpoint: A checkpoint of the grounded world, with or without transition programs.

    Returns:
        ``(state, history)`` held by the grounded predictor's image.

    Raises:
        ValueError: The image holds no well-formed grounded predictor part.
    """
    payload = checkpoint.image.parts.get(_PREDICTOR_PART)
    wire = None if payload is None else json.loads(payload)
    if (
        not isinstance(wire, dict)
        or not isinstance(wire.get("state"), dict)
        or not isinstance(wire.get("history"), list)
    ):
        raise ValueError("the checkpoint holds no grounded simulator state and history")
    return wire["state"], wire["history"]


def with_simulated_image(
    checkpoint: ExecutableCheckpoint, state: JsonObject, history: list[JsonValue]
) -> ExecutableCheckpoint:
    """Return a checkpoint of the same runtime holding another state and world-model history.

    Every other image part (such as the transition programs' partition) and every kernel field
    is kept, and the domain digests are recomputed, so the runtime's own restore verifies it.

    Args:
        checkpoint: A checkpoint taken by the runtime that will restore the result.
        state: Complete simulated state.
        history: Predictor history the restored world conditions on.

    Returns:
        The rewritten checkpoint.
    """
    parts = {
        **checkpoint.image.parts,
        _PREDICTOR_PART: canonical_json_bytes({"state": state, "history": history}),
    }
    domains = {key: sha256_json(value) for key, value in state.items()}
    return checkpoint.model_copy(update={"image": NativeImage(parts=parts, domains=domains)})


def restores_exactly(
    project: ProjectStore,
    definition: GroundedRuntimeDefinition,
    reference: ArtifactInput,
    state: JsonObject,
) -> bool:
    """Prove a saved start holds one state and restores exactly in its runtime.

    The checkpoint's simulated state must equal ``state`` and its domain digests; the runtime of
    ``definition`` (its programs included) must restore it through its own restore path and
    recapture the identical image, world-model history included, and clock.

    Args:
        project: Project holding the checkpoint.
        definition: The runtime the start was taken in.
        reference: The saved continuation checkpoint.
        state: The exact simulated state it must hold.

    Returns:
        Whether every comparison held.
    """
    try:
        checkpoint = load_continuation_checkpoint(project.artifacts, reference)
        world = checkpoint.world
        if not isinstance(world, ExecutableCheckpoint):
            return False
        held, _ = simulated_image(world)
        if held != state or world.image.domains != {k: sha256_json(v) for k, v in held.items()}:
            return False
        with continuation_runtime(project, definition.model_dump(mode="json")) as runtime:
            if not isinstance(runtime.world, ExecutableWorldRuntime):
                return False
            with runtime.world.restore(checkpoint.world_task, world) as restored:
                recaptured = restored.checkpoint()
        return recaptured.image == world.image and recaptured.clock == world.clock
    except ValueError:
        return False


def continuation_runtime(project: ProjectStore, configuration: JsonObject) -> ContinuationRuntime:
    """Compose the built simulator for capture, continuation, or an executable Harbor trial.

    The CLI obtains spend consent before executing this explicitly selected factory. SDKs that
    bypass the shared HTTP transport are refused before credential resolution. Predictor, query
    embedding and memory readers share one physical-attempt ledger across all restored forks.
    """
    return _runtime(project, GroundedRuntimeDefinition.model_validate(configuration))


def trial_runtime(
    project: ProjectStore, configuration: JsonObject, provider_grant: ProviderGrant
) -> ContinuationRuntime:
    """Use explicitly renewed trial authority while keeping the frozen world semantics exact."""
    provider_grant.require_configuration(project, configuration)
    spec = GroundedRuntimeDefinition.model_validate(configuration)
    if provider_grant.limits.maximum_physical_attempts > spec.maximum_physical_attempts:
        raise ValueError("trial provider allowance exceeds the frozen simulator ceiling")
    return _runtime(project, spec, provider_grant)


def load_rule_book(store: ArtifactStore, reference: ArtifactInput) -> RuleBook:
    """Read the rule book of one exact immutable artifact.

    Args:
        store: Project artifact store.
        reference: The artifact and its manifest digest; the manifest digests ``rules.json``.

    Returns:
        The parsed rule book.

    Raises:
        ValueError: The artifact's manifest digest differs from the reference, or it holds no
            valid ``rules.json``.
    """
    stored = store.read(reference.artifact_id)
    if artifact_input(stored.manifest) != reference:
        raise ValueError(f"rule book artifact {reference.artifact_id} manifest digest changed")
    payload = stored.payloads.get(RULES_JSON)
    if payload is None:
        raise ValueError(f"rule book artifact {reference.artifact_id} holds no {RULES_JSON}")
    return RuleBook.model_validate_json(payload)


def preflight_grounded_runtime(project: ProjectStore, spec: GroundedRuntimeDefinition) -> None:
    """Check a definition against the project without credentials, runtime state or calls.

    Args:
        project: Project whose build, catalog and artifacts the definition pins.
        spec: Frozen simulator definition.

    Raises:
        ValueError: The task is not development data, the state or probe disagrees with the
            manifest, the catalog changed, a provider wire is not admitted, the rule book or
            tag retrieval binding artifact differs from its reference, a memory-first
            simulator is given a tag binding, or the rule memory set or its recall alias is not
            the pinned, admitted one.
    """
    _checked(project, spec)


def _checked(
    project: ProjectStore, spec: GroundedRuntimeDefinition
) -> tuple[
    ModelCatalog,
    GroundedWorldModelArtifact,
    str | None,
    RuleBook | None,
    TagRetrievalBinding | None,
    MemorySet | None,
]:
    """Run every credential-free check; return catalog, simulator, embedder, rules, tags, memory."""
    require_development_task(project.artifacts, spec.task)
    if spec.task.reference_answer is not None:
        raise ValueError("grounded executable tasks must not contain private expected answers")
    if set(spec.initial_state) != {d.name for d in spec.manifest.domains if d.owner == "simulated"}:
        raise ValueError("initial state must name exactly the simulated domains")
    for pointer in spec.probe_paths.values():
        resolve(spec.initial_state, pointer)
    require_fact_scope(
        project, spec.task, spec.facts, None if spec.memory is None else spec.memory.lineage
    )
    catalog = load_model_catalog(project.model_catalog_path)
    if sha256_json(catalog.model_dump(mode="json")) != spec.catalog_sha256:
        raise ValueError("executable simulator catalog differs from its frozen definition")
    artifact = load_grounded_world_model_artifact(project.artifacts, spec.world_model)
    reasoning_catalog(catalog, artifact.model_alias, spec.reasoning_effort)
    embedder_alias = index_embedder_alias(project, catalog, artifact)
    aliases = [artifact.model_alias]
    if embedder_alias is not None:
        aliases.append(embedder_alias)
    for alias in aliases:
        require_admitted_transport(catalog, alias, consumer="executable simulator")
    if spec.memory is not None:
        recall = catalog.models[spec.memory.recall.jev_alias]
        if catalog.connections[recall.connection].provider != "typesafe":
            raise ValueError("executable memory recall requires the admitted native decision wire")
        # _runtime binds this memory with bind_evaluation_memory over its decision transport;
        # TypeSafeClient supplies endpoint/auth facts and never performs a completion.
    if spec.prediction_reading is not None:
        alias = catalog.models[spec.prediction_reading.reader_alias]
        if catalog.connections[alias.connection].provider != "typesafe":
            raise ValueError("prediction reader requires the admitted native decision wire")
        if spec.prediction_reading.policy.rounds > 2:
            raise ValueError("prediction reading permits at most two rounds")
    rule_book = (
        None if spec.rule_book is None else load_rule_book(project.artifacts, spec.rule_book)
    )
    if spec.rule_memory is not None and spec.rule_memory.guards is not None:
        rule_book = guarded_rule_book(rule_book, load_guard_table(project, spec.rule_memory.guards))
    tag_retrieval = None
    if spec.tag_retrieval is not None:
        tag_retrieval = load_tag_retrieval_binding(project, spec.tag_retrieval)
        if artifact.original_evidence == "none":
            raise ValueError("a memory-first simulator retrieves nothing to rank by tags")
    rule_memory = None
    if spec.rule_memory is not None:
        alias = catalog.models.get(spec.rule_memory.recall.jev_alias)
        if alias is None or catalog.connections[alias.connection].provider != "typesafe":
            raise ValueError("rule memory recall requires the admitted native decision wire")
        rule_memory = load_memory_set(project.artifacts, spec.rule_memory.memory_set)
    if spec.state_memory is not None:
        alias = catalog.models.get(spec.state_memory.author_alias)
        if alias is None or catalog.connections[alias.connection].provider == "typesafe":
            raise ValueError(
                "state memory needs an admitted completion alias, not a decision-only model; "
                "name an open-weight chat model in state_memory.author_alias"
            )
        require_admitted_transport(
            catalog, spec.state_memory.author_alias, consumer="executable simulator"
        )
    if spec.state_recall is not None:
        check_state_recall(catalog, spec.state_recall)
    return catalog, artifact, embedder_alias, rule_book, tag_retrieval, rule_memory


def guarded_rule_book(rule_book: RuleBook | None, table: GuardTable) -> RuleBook:
    """Return the rule book with the guard table's checkable entries enforced alongside it.

    Args:
        rule_book: The learned rule book, or None.
        table: The family's admitted action guards, postconditions and invariants.

    Returns:
        One book the runtime checker enforces; a guard whose rule id the book already holds
        keeps the book's rule.
    """
    guards = guard_rulebook(table)
    if rule_book is None:
        return guards
    held = {rule.rule_id for rule in rule_book.rules}
    extra = tuple(rule for rule in guards.rules if rule.rule_id not in held)
    return rule_book.model_copy(update={"rules": (*rule_book.rules, *extra)})


def recall_decider(
    project: ProjectStore,
    decide: Callable[[DecisionRequest], JsonObject],
    model: JsonObject,
) -> Callable[[DecisionRequest], JsonObject]:
    """Buy each recall decision once through the project decision cache, replaying repeats.

    Args:
        project: Project owning the cache.
        decide: Admitted native decision call.
        model: Decision model identity, part of every cache key.

    Returns:
        A decision call that replays a saved answer without dispatching.
    """
    cache = DecisionCache(project.paths)

    def cached(request: DecisionRequest) -> JsonObject:
        """Replay or buy one recall decision."""
        key = sha256_json(
            {
                "protocol": RECALL_PROTOCOL,
                "request": sha256_json(request.model_dump(mode="json")),
                "model": model,
            }
        )
        return cache.purchase(key, request, lambda: decide(request))

    return cached


def _runtime(
    project: ProjectStore,
    spec: GroundedRuntimeDefinition,
    provider_grant: ProviderGrant | None = None,
) -> ContinuationRuntime:
    """Compose identical state/predictor definitions under one explicitly selected call ledger."""
    catalog, artifact, embedder_alias, rule_book, tag_retrieval, memory_set = _checked(
        project, spec
    )
    revision = installed_execution_revision()
    identity_fields: JsonObject = {"definition": spec.model_dump(mode="json"), "code": revision}
    if provider_grant is not None:
        identity_fields = {
            "provider_grant": provider_grant.model_dump(mode="json"),
            "code": revision,
        }
    identity = sha256_json(identity_fields)
    transport = AdmittedJsonTransport(
        HttpxJsonTransport(),
        artifacts=project.artifacts,
        code_revision=revision,
        deadline=dispatch_deadline(
            spec.deadline if provider_grant is None else provider_grant.limits.deadline
        ),
        ledger=DispatchLedger(
            project.artifacts.project_directory / (identity + "-world.sqlite"),
            spec.maximum_physical_attempts
            if provider_grant is None
            else provider_grant.limits.maximum_physical_attempts,
            identity,
        ),
    )
    registry = RuntimeModelCatalog(catalog, transport_factory=lambda: transport)
    predictor_catalog = reasoning_catalog(catalog, artifact.model_alias, spec.reasoning_effort)
    predictor_registry = (
        registry
        if predictor_catalog is catalog
        else RuntimeModelCatalog(predictor_catalog, transport_factory=lambda: transport)
    )
    predictor = predictor_registry.resolve(artifact.model_alias)
    if predictor.snapshot != artifact.model:
        raise ValueError("executable predictor differs from the built simulator")
    embedding = None
    if embedder_alias is not None:
        resolved = registry.resolve(embedder_alias)
        if resolved.embedding_client is None:
            raise ValueError("retrieval alias cannot embed queries")
        embedding = RAGEmbedderBinding(
            client=resolved.embedding_client,
            snapshot=resolved.snapshot,
            maximum_input_tokens=resolved.capabilities.context_window_tokens,
        )
    runtime = load_grounded_world_model(
        project.artifacts,
        spec.world_model.artifact_id,
        client=predictor.client,
        capabilities=predictor.capabilities,
        embedder=embedding,
        tag_retrieval=tag_retrieval,
    )
    memory = (
        None
        if spec.memory is None
        else bind_evaluation_memory(
            project,
            spec.memory,
            catalog=registry,
            artifact=artifact,
            world_model=spec.world_model,
            created_at=datetime.now(UTC),
            code_revision=revision,
            decision_transport=transport,
            research_history=spec.research_history,
        ).for_task(spec.task)
    )

    deadline = dispatch_deadline(
        spec.deadline if provider_grant is None else provider_grant.limits.deadline
    )
    completion = StructuredCompletion(
        client=predictor.client,
        plan=CompletionPlan(
            model=predictor.snapshot,
            capabilities=predictor.capabilities,
            deadline=deadline,
            output_tokens=spec.maximum_output_tokens,
            maximum_cost_per_call=1.0,
            code_revision=revision,
            output_protocol="tool",
        ),
        budget=RequestBudget(project, identity=identity),
        artifacts=project.artifacts,
    )

    def complete(body: JsonObject, operation: str) -> JsonObject:
        """Use the actual provider client through the shared admitted physical transport."""
        request = ModelRequest.model_validate(body)
        if spec.reasoning_effort is not None:
            if request.reasoning_effort not in (None, spec.reasoning_effort):
                raise ValueError("simulator request differs from its frozen reasoning profile")
            request = request.model_copy(update={"reasoning_effort": spec.reasoning_effort})
        with call_consumer("simulator"):
            return completion.complete_request(request, operation).model_dump(mode="json")

    reader = None
    if spec.prediction_reading is not None:
        native_reader = registry.resolve(spec.prediction_reading.reader_alias)

        def plan(body: JsonObject, operation: str) -> JsonObject:
            """Meter and retain the planner independently from the simulator's final request."""
            with call_consumer("selector"):
                return completion(body, operation)

        reader = PredictionResolver(
            EvidenceReader(
                spec.prediction_reading.policy,
                decision_caller(native_reader, transport, deadline, lambda: datetime.now(UTC)),
                receipt_recorder(project, identity, revision, lambda: datetime.now(UTC)),
            ),
            plan,
            {}
            if memory is None
            else {
                key: row
                for key, row in memory.records.items()
                if row.get("kind") == "original-observation"
            },
            sha256_json(
                {
                    "planner": completion.completion_profile_sha256,
                    "reader": native_reader.snapshot.model_dump(mode="json"),
                }
            ),
        )

    def record(event: PredictorEvent) -> None:
        """Retain original prediction requests and raw responses before parsing or committing."""
        project.artifacts.write_json(
            artifact_id=event.operation_id + "-" + event.stage,
            artifact_type="grounded-world-prediction",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=revision,
                inputs=tuple(
                    sorted(
                        (spec.world_model,)
                        + (() if provider_grant is None else (provider_grant.registration,))
                        + (() if spec.rule_book is None else (spec.rule_book,))
                        + (() if spec.tag_retrieval is None else (spec.tag_retrieval,))
                        + (() if spec.rule_memory is None else (spec.rule_memory.memory_set,))
                        + (
                            ()
                            if spec.rule_memory is None or spec.rule_memory.guards is None
                            else (spec.rule_memory.guards,)
                        )
                        + (() if spec.state_recall is None else (spec.state_recall.index,)),
                        key=lambda reference: reference.artifact_id,
                    )
                ),
            ),
            files={"event.json": event},
        )

    recall = None
    if spec.rule_memory is not None and memory_set is not None:
        native_recall = registry.resolve(spec.rule_memory.recall.jev_alias)
        recall = RuleRecall(
            memory_set,
            spec.rule_memory.recall,
            decide=recall_decider(
                project,
                decision_caller(native_recall, transport, deadline, lambda: datetime.now(UTC)),
                native_recall.snapshot.model_dump(mode="json"),
            ),
            record=receipt_recorder(project, identity, revision, lambda: datetime.now(UTC)),
        )

    state_memory = None
    if spec.state_memory is not None:
        author = registry.resolve(spec.state_memory.author_alias)
        descriptions = DecisionCache(project.paths)

        def author_text(request: ModelRequest) -> str:
            """Write one state description draft through the shared admitted transport."""
            with call_consumer("analyst"):
                return author.client.complete(request).output.content or ""

        state_memory = StateMemory(
            spec.state_memory,
            complete=author_text,
            model=author.snapshot.model_dump(mode="json"),
            purchase=descriptions.purchase,
            record=receipt_recorder(project, identity, revision, lambda: datetime.now(UTC)),
        )

    state_recall = None
    if spec.state_recall is not None:
        state_recall = bind_state_recall(
            project,
            spec.state_recall,
            world_model=spec.world_model,
            registry=registry,
            transport=transport,
            deadline=deadline,
            clock=lambda: datetime.now(UTC),
            record=receipt_recorder(project, identity, revision, lambda: datetime.now(UTC)),
            embedding=embedding,
        )

    grounded_factory = GroundedWorldFactory(
        GroundedTransitionEngine(
            runtime,
            spec.task,
            maximum_output_tokens=spec.maximum_output_tokens,
            output_protocol="json-tool",
            facts=None if spec.facts is None else spec.facts.facts,
            facts_version=0 if spec.facts is None else spec.facts.first_version,
            research_history=spec.research_history,
            prediction_reader=reader,
        ),
        spec.manifest,
        spec.initial_state,
        complete=complete,
        record=record,
        memory=memory,
        maximum_reasks=spec.maximum_reasks,
        rule_book=rule_book,
        rule_memory=recall,
        state_memory=state_memory,
        state_recall=state_recall,
    )
    factory: WorldBackendFactory = grounded_factory
    sandbox = None
    if spec.programs is not None:
        sandbox = ProgramSandbox(spec.programs.limits)
        starting = (
            None if spec.facts is None else spec.facts.facts.conditioning(spec.facts.first_version)
        )
        factory = InducedWorldFactory(
            spec.programs,
            grounded_factory,
            sandbox,
            facts={} if starting is None else {fact.key: fact.value for fact in starting.facts},
        )

    def probe(checkpoint: WorldCheckpoint) -> JsonObject:
        """Omit missing private probe fields without discarding valid actor observations."""
        if not isinstance(checkpoint, ExecutableCheckpoint):
            raise ValueError("simulated projection requires its executable checkpoint")
        if checkpoint.backend_sha256 != sha256_json(factory.definition):
            raise ValueError("probe checkpoint differs from the frozen simulator")
        return simulated_probe(simulated_image(checkpoint)[0], spec.probe_paths)

    world = ExecutableWorldRuntime(spec.manifest, factory, owner="simulated", seed=spec.seed)
    if sandbox is None:
        return ContinuationRuntime(world, probe, spec.probe_sha256)
    return ContinuationRuntime(world, probe, spec.probe_sha256, sandbox.close)
