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
"""

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Self

from pydantic import AwareDatetime, Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
    sha256_json,
)
from exp.common.models import ModelCatalog, ModelRequest, ReasoningEffort, load_model_catalog
from exp.common.project import ArtifactStore, ProjectStore
from exp.common.project.manifests import artifact_input
from exp.common.release_revision import installed_execution_revision
from exp.common.tasks import TaskCase
from exp.optimize.evaluation.executable.loader import ContinuationRuntime
from exp.optimize.evaluation.executable.provider_grant import ProviderGrant
from exp.optimize.evaluation.fidelity.decision_access import decision_caller, receipt_recorder
from exp.optimize.evaluation.fidelity.instance_facts import ReplayFacts, require_fact_scope
from exp.optimize.evaluation.fidelity.retrieval_binding import index_embedder_alias
from exp.optimize.evaluation.learning.profile import EvaluationMemory, bind_evaluation_memory
from exp.optimize.evaluation.variance.build_records import load_tag_retrieval_binding
from exp.runtime.environments.world_contracts import ExecutableCheckpoint, WorldManifest
from exp.runtime.environments.worlds import ExecutableWorldRuntime
from exp.runtime.models.budget import RequestBudget
from exp.runtime.models.call_evidence import call_consumer
from exp.runtime.models.providers.admitted_transport import (
    AdmittedJsonTransport,
    require_admitted_transport,
)
from exp.runtime.models.providers.transport import HttpxJsonTransport
from exp.runtime.models.reasoning_view import reasoning_catalog
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.runtime.models.structured_completion import CompletionPlan, StructuredCompletion
from exp.runtime.sessions.dispatch_ledger import DispatchLedger
from exp.simulation.continuations import WorldCheckpoint
from exp.simulation.development_scope import require_development_task
from exp.simulation.reconstruction.models import resolve
from exp.simulation.retrieval import RAGEmbedderBinding
from exp.simulation.retrieval.evidence_reading import EvidenceReader
from exp.simulation.retrieval.tags import TagRetrievalBinding
from exp.simulation.rules.model import RuleBook
from exp.simulation.rules.rulebook import RULES_JSON
from exp.simulation.world_model.artifact import GroundedWorldModelArtifact
from exp.simulation.world_model.executable import GroundedWorldFactory, PredictorEvent
from exp.simulation.world_model.grounded_session import GroundedTransitionEngine
from exp.simulation.world_model.prediction_reading import PredictionReading, PredictionResolver
from exp.simulation.world_model.runtime import (
    load_grounded_world_model,
    load_grounded_world_model_artifact,
)


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

    @model_validator(mode="after")
    def _starting_facts_only(self) -> Self:
        """Refuse facts an executable episode cannot honor or that shadow native receipts."""
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
            tag retrieval binding artifact differs from its reference, or a memory-first
            simulator is given a tag binding.
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
]:
    """Run every credential-free check; return catalog, simulator, embedder, rules and tags."""
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
    tag_retrieval = None
    if spec.tag_retrieval is not None:
        tag_retrieval = load_tag_retrieval_binding(project, spec.tag_retrieval)
        if artifact.original_evidence == "none":
            raise ValueError("a memory-first simulator retrieves nothing to rank by tags")
    return catalog, artifact, embedder_alias, rule_book, tag_retrieval


def _runtime(
    project: ProjectStore,
    spec: GroundedRuntimeDefinition,
    provider_grant: ProviderGrant | None = None,
) -> ContinuationRuntime:
    """Compose identical state/predictor definitions under one explicitly selected call ledger."""
    catalog, artifact, embedder_alias, rule_book, tag_retrieval = _checked(project, spec)
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
        deadline=spec.deadline if provider_grant is None else provider_grant.limits.deadline,
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

    deadline = spec.deadline if provider_grant is None else provider_grant.limits.deadline
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
                        + (() if spec.tag_retrieval is None else (spec.tag_retrieval,)),
                        key=lambda reference: reference.artifact_id,
                    )
                ),
            ),
            files={"event.json": event},
        )

    factory = GroundedWorldFactory(
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
    )

    def probe(checkpoint: WorldCheckpoint) -> JsonObject:
        """Omit missing private probe fields without discarding valid actor observations."""
        if not isinstance(checkpoint, ExecutableCheckpoint):
            raise ValueError("simulated projection requires its executable checkpoint")
        if checkpoint.backend_sha256 != sha256_json(factory.definition):
            raise ValueError("probe checkpoint differs from the frozen simulator")
        wire = json.loads(checkpoint.image.parts["predictor.json"])
        return simulated_probe(wire["state"], spec.probe_paths)

    return ContinuationRuntime(
        ExecutableWorldRuntime(spec.manifest, factory, owner="simulated", seed=spec.seed),
        probe,
        spec.probe_sha256,
    )
