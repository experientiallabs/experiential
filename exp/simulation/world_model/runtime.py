"""Executable RAG-grounded text world model from completed build artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

from exp.common.core.artifacts import (
    ArtifactInput,
    JsonObject,
    envelope_matches_manifest,
    stable_id,
)
from exp.common.models import (
    AssistantAction,
    Embedding,
    ModelCapabilities,
    ModelClient,
    ModelMessage,
    ModelRequest,
    ModelResponse,
)
from exp.common.project import ArtifactStore, artifact_input
from exp.common.tasks import TaskCase, ToolSchema
from exp.common.tasks.facts import ConditioningFacts
from exp.simulation.engines.text.packing import pack_world_model_request
from exp.simulation.engines.text.prompt import (
    RECONSTRUCTED_TEXT_PROMPT_VERSION,
    MemoryContext,
    TextWorldModelTransition,
    build_world_model_request,
    candidate_rag_actions,
    parse_world_model_transition,
    validate_transition_action,
)
from exp.simulation.engines.text.tokens import (
    TokenCounter,
    Utf8UpperBoundTokenCounter,
    WorldModelCapacityError,
    bound_unpublished_output,
)
from exp.simulation.retrieval import (
    CorpusScope,
    LoadedRAGIndex,
    RAGIndex,
    RAGMatch,
    RAGQuery,
    TraceRAGRetriever,
    load_rag_index,
)
from exp.simulation.retrieval.embedding import RAGEmbedderBinding
from exp.simulation.retrieval.jev import JEVSelector
from exp.simulation.retrieval.tags import TagRetrievalBinding, TagRetrievalEvidence
from exp.simulation.world_model.artifact import (
    GROUNDED_WORLD_MODEL_PROMPT_VERSION,
    WORLD_MODEL_ARTIFACT_PATH,
    GroundedWorldModelArtifact,
    grounded_world_model_content,
    grounded_world_model_prompt_sha256,
    persist_grounded_world_model,
)


@dataclass(frozen=True)
class PreparedGroundedWorldModelCall:
    """One retrieved and framed grounded request before provider dispatch.

    Attributes:
        request: Fully framed world-model request ready for dispatch.
        matches: Ordered grounding examples selected for the action.
        action: Exact candidate action whose call IDs constrain generated results.
        tag_retrieval: Policy and task tags that ranked the matches, when tags were applied.
    """

    request: ModelRequest
    matches: tuple[RAGMatch, ...]
    action: AssistantAction
    tag_retrieval: TagRetrievalEvidence | None = None

    def fit_context(
        self,
        capabilities: ModelCapabilities,
        token_counter: TokenCounter,
        maximum_input_tokens: int | None = None,
    ) -> PreparedGroundedWorldModelCall:
        """Pack optional examples and bind provenance to the unchanged request allowance.

        Args:
            capabilities: Frozen world-model context and output capacity.
            token_counter: Full-request input counter used by dispatch admission.
            maximum_input_tokens: Optional stricter frozen input reservation.

        Returns:
            Request with only whole included examples in its matching provenance.
        """
        context = capabilities.context_window_tokens
        output = self.request.maximum_output_tokens
        if context is None or output is None:
            return self
        ceiling = context - output
        if maximum_input_tokens is not None:
            ceiling = min(ceiling, maximum_input_tokens)
        request, identifiers = pack_world_model_request(
            self.request, maximum_input_tokens=ceiling, token_counter=token_counter
        )
        # Optional evidence yields to the requested output first. An unpublished output limit
        # can then use remaining context around the required prompt, never around dropped examples.
        request = bound_unpublished_output(request, capabilities, token_counter)
        matches = tuple(
            match for match in self.matches if match.transition.transition_id in identifiers
        )
        if tuple(match.transition.transition_id for match in matches) != identifiers:
            raise ValueError("packed world-model examples differ from retrieved provenance")
        return replace(self, request=request, matches=matches)


@dataclass(frozen=True)
class DispatchedGroundedWorldModelCall:
    """One artifact-bound response paired with its exact request and retrieval evidence.

    Attributes:
        request: Exact dispatched world-model request.
        response: Provider response awaiting protocol validation.
        matches: Grounding evidence included in the request.
        action: Exact candidate action used when validating generated result identities.
    """

    request: ModelRequest
    response: ModelResponse
    matches: tuple[RAGMatch, ...]
    action: AssistantAction


@dataclass(frozen=True)
class GroundedWorldModelCall:
    """One artifact-bound grounded request, response, retrieval set, and parsed transition."""

    request: ModelRequest
    response: ModelResponse
    matches: tuple[RAGMatch, ...]
    transition: TextWorldModelTransition


@dataclass(frozen=True)
class GroundedWorldModel:
    """Call one configured model with nearest observed transitions as immutable evidence.

    Attributes:
        artifact_input: Exact verified grounded-model manifest reference.
        artifact: Frozen model identity and grounding protocol.
        retriever: Fit or serving retriever bound to the artifact's immutable corpus.
        client: Explicit completion provider for world predictions.
        capabilities: Required resolved metadata matching the artifact's frozen identity.
            Unknown capacity fields remain explicit rather than disabling identity verification.
        token_counter: Complete-request counter used for capacity admission.
        evidence_selector: Optional artifact-bound JEV policy over eligible candidates.
        tag_retrieval: Optional session tag policy and per-task task-view tags. A task with
            tags retrieves by the policy's combined score; others retrieve by cosine alone.
    """

    artifact_input: ArtifactInput
    artifact: GroundedWorldModelArtifact
    retriever: TraceRAGRetriever
    client: ModelClient
    capabilities: ModelCapabilities
    token_counter: TokenCounter = field(default_factory=Utf8UpperBoundTokenCounter)
    evidence_selector: JEVSelector | None = None
    tag_retrieval: TagRetrievalBinding | None = None

    def __post_init__(self) -> None:
        """Require the current prompt, exact capacity, and a tag binding the index can serve."""
        _require_current_prompt(self.artifact)
        binding = self.tag_retrieval
        if binding is not None:
            if self.artifact.original_evidence == "none":
                raise ValueError("a memory-first simulator retrieves nothing to rank by tags")
            for tags in binding.tasks.values():
                self.retriever.tag_ranking(binding.policy, tags)
        selector_id = (
            None if self.evidence_selector is None else self.evidence_selector.policy.identity
        )
        if selector_id != self.artifact.selection_policy_sha256:
            raise ValueError("evidence selection policy differs from the simulator artifact")
        if self.capabilities.identity_sha256() != self.artifact.model.capabilities_sha256:
            raise ValueError(
                "world-model capabilities differ from the frozen build artifact; "
                "use the artifact's resolved model or rebuild with the intended model"
            )

    def prepare_turn(
        self,
        *,
        task: TaskCase,
        visible_messages: Sequence[ModelMessage],
        candidate_response: AssistantAction,
        excluded_lineage_ids: tuple[str, ...],
        maximum_output_tokens: int,
        state: JsonObject | None = None,
        json_object_output: bool = False,
        memory: MemoryContext | None = None,
        output_protocol: Literal["text", "json-tool"] = "text",
        facts: ConditioningFacts | None = None,
    ) -> PreparedGroundedWorldModelCall:
        """Retrieve and frame one fit- or serving-bound grounded text transition.

        Args:
            task: Current canonical task and safe initial context.
            visible_messages: Candidate-visible conversation through the latest request.
            candidate_response: Latest visible candidate action.
            excluded_lineage_ids: Source lineages forbidden from retrieval.
            maximum_output_tokens: Explicit provider output ceiling.
            state: Private environment state retained between simulated turns.
            json_object_output: Provider JSON control included in required framing.
            memory: Typed cross-trajectory memory rendered into its authority channels.
            output_protocol: Text transition or typed-proposal system instructions.
            facts: Instance facts admissible for this prediction, in their own channel.

        Returns:
            Exact request and retrieved evidence before provider dispatch.
        """
        self.preflight_turn(
            task=task,
            visible_messages=visible_messages,
            candidate_response=candidate_response,
            maximum_output_tokens=maximum_output_tokens,
            state=state,
            json_object_output=json_object_output,
            memory=memory,
            output_protocol=output_protocol,
            facts=facts,
        )
        if self.artifact.original_evidence == "none":
            # Memory-first profile: no recorded transition is retrieved for any prediction.
            request = build_world_model_request(
                task,
                visible_messages=visible_messages,
                candidate_response=candidate_response,
                grounded_examples=(),
                knowledge=self.artifact.knowledge,
                maximum_output_tokens=maximum_output_tokens,
                state=state,
                memory=memory,
                output_protocol=output_protocol,
                facts=facts,
            )
            request = request.model_copy(update={"json_object_output": json_object_output})
            return PreparedGroundedWorldModelCall(
                request=request, matches=(), action=candidate_response
            )
        binding = self.tag_retrieval
        tags = None if binding is None else binding.for_task(task.task_id)
        policy = None if binding is None or tags is None else binding.policy
        queries = tuple(
            RAGQuery(
                task=task.instruction,
                initial_context=task.initial_context,
                action=action,
                excluded_lineage_ids=excluded_lineage_ids,
                top_k=(
                    self.evidence_selector.policy.candidate_limit
                    if self.evidence_selector
                    else self.artifact.top_k
                ),
            )
            for action in candidate_rag_actions(candidate_response)
        )
        batches = tuple(
            self.retriever.retrieve(query)
            if policy is None
            else self.retriever.retrieve(query, tag_policy=policy, query_tags=tags)
            for query in queries
        )
        if self.evidence_selector is not None:
            batches = tuple(
                self.evidence_selector.select(
                    query,
                    candidates,
                    state=state or {},
                    history=visible_messages,
                    top_k=self.artifact.top_k,
                )
                for query, candidates in zip(queries, batches, strict=True)
            )
        # Round-robin over each call's nearest examples, retaining the pinned total context bound.
        selected: dict[str, RAGMatch] = {}
        for rank in range(self.artifact.top_k):
            for batch in batches:
                if rank < len(batch) and len(selected) < self.artifact.top_k:
                    match = batch[rank]
                    selected.setdefault(match.transition.transition_id, match)
        matches = tuple(selected.values())
        request = build_world_model_request(
            task,
            visible_messages=visible_messages,
            candidate_response=candidate_response,
            grounded_examples=matches,
            knowledge=self.artifact.knowledge,
            maximum_output_tokens=maximum_output_tokens,
            state=state,
            memory=memory,
            output_protocol=output_protocol,
            facts=facts,
        )
        request = request.model_copy(update={"json_object_output": json_object_output})
        return PreparedGroundedWorldModelCall(
            request=request,
            matches=matches,
            action=candidate_response,
            tag_retrieval=None
            if policy is None or tags is None
            else self.retriever.tag_evidence(policy, tags),
        )

    def tag_retrieval_identity(self, task: TaskCase) -> str | None:
        """Return the tag-ranking identity a session for one task binds, or None without tags.

        Args:
            task: The task a session serves.

        Returns:
            Digest of the policy and that task's tags, or None when no binding is configured.
        """
        binding = self.tag_retrieval
        return None if binding is None else binding.identity_for(task.task_id)

    def preflight_turn(
        self,
        *,
        task: TaskCase,
        visible_messages: Sequence[ModelMessage],
        candidate_response: AssistantAction,
        maximum_output_tokens: int,
        state: JsonObject | None = None,
        json_object_output: bool = False,
        maximum_input_tokens: int | None = None,
        memory: MemoryContext | None = None,
        output_protocol: Literal["text", "json-tool"] = "text",
        facts: ConditioningFacts | None = None,
    ) -> None:
        """Admit required framing before retrieval or its paid-work accounting window.

        Args:
            task: Complete canonical task, including tool schemas and initial context.
            visible_messages: Candidate-visible transcript retained in the world prompt.
            candidate_response: Exact assistant action and original tool-call identities.
            maximum_output_tokens: Original requested output allowance.
            state: Complete private environment state retained between simulated turns.
            json_object_output: Provider JSON control included in token accounting.
            maximum_input_tokens: Optional stricter frozen request reservation.
            memory: Typed cross-trajectory memory counted as required content.
            output_protocol: Text transition or typed-proposal system instructions.
            facts: Instance facts counted as required content.

        Raises:
            ValueError: The output allowance is not positive.
            WorldModelCapacityError: Required framing cannot fit the declared capacities.
        """
        request = build_world_model_request(
            task,
            visible_messages=visible_messages,
            candidate_response=candidate_response,
            grounded_examples=(),
            knowledge=self.artifact.knowledge,
            maximum_output_tokens=maximum_output_tokens,
            state=state,
            memory=memory,
            output_protocol=output_protocol,
            facts=facts,
        ).model_copy(update={"json_object_output": json_object_output})
        # Admission may bind unpublished output around required content. Retrieval still uses
        # the original allowance so optional examples yield before actual output is bound.
        request = bound_unpublished_output(request, self.capabilities, self.token_counter)
        self._require_capacity(request, maximum_input_tokens=maximum_input_tokens)

    def _require_capacity(
        self, request: ModelRequest, *, maximum_input_tokens: int | None = None
    ) -> None:
        """Validate one fully rendered request without retrieval, dispatch, or accounting."""
        output = request.maximum_output_tokens
        if output is None or output <= 0:
            raise WorldModelCapacityError("world-model maximum_output_tokens must be positive")
        published_output = self.capabilities.maximum_output_tokens
        if published_output is not None and output > published_output:
            raise WorldModelCapacityError(
                "world-model output exceeds the published model capacity; lower "
                "maximum_output_tokens or choose a model with a larger output capacity"
            )
        required = self.token_counter.count(request)
        context = self.capabilities.context_window_tokens
        if required < 0 or (context is not None and required + output > context):
            raise WorldModelCapacityError(
                "required world-model input and output exceed the context capacity; "
                "choose a larger-context world model"
            )
        if maximum_input_tokens is not None and required > maximum_input_tokens:
            raise WorldModelCapacityError(
                "required world-model input exceeds the frozen input reservation; "
                "prepare a larger input reservation before retrying"
            )

    def complete_turn(
        self,
        prepared: PreparedGroundedWorldModelCall,
    ) -> DispatchedGroundedWorldModelCall:
        """Dispatch one prepared artifact-bound request.

        Args:
            prepared: Exact retrieved evidence and request produced by ``prepare_turn``.

        Returns:
            Exact request, response, and retrieved evidence.

        Raises:
            ValueError: Required evidence or output cannot fit the bound model capacity.
        """
        prepared = prepared.fit_context(self.capabilities, self.token_counter)
        self._require_capacity(prepared.request)
        response = self.client.complete(prepared.request)
        return DispatchedGroundedWorldModelCall(
            request=prepared.request,
            response=response,
            matches=prepared.matches,
            action=prepared.action,
        )

    def parse_turn(
        self,
        dispatched: DispatchedGroundedWorldModelCall,
    ) -> GroundedWorldModelCall:
        """Parse one dispatched response through the artifact's transition protocol.

        Args:
            dispatched: Exact artifact-bound provider result.

        Returns:
            Completed grounded call with its parsed visible transition.
        """
        transition = parse_world_model_transition(dispatched.response.output)
        validate_transition_action(transition, dispatched.action)
        return GroundedWorldModelCall(
            request=dispatched.request,
            response=dispatched.response,
            matches=dispatched.matches,
            transition=transition,
        )

    def step(
        self,
        *,
        task: str,
        action: AssistantAction,
        tools: tuple[ToolSchema, ...] = (),
        initial_context: JsonObject | None = None,
        excluded_lineage_ids: tuple[str, ...] = (),
        maximum_output_tokens: int = 1_024,
    ) -> TextWorldModelTransition:
        """Predict one next visible observation grounded on nearest real transitions.

        Args:
            task: Current request-visible task.
            action: Latest assistant text or tool calls retaining their original call IDs.
            tools: Declared schemas for every tool the assistant may invoke.
            initial_context: Safe request-visible starting context.
            excluded_lineage_ids: Source lineages forbidden for this query.
            maximum_output_tokens: Explicit provider output ceiling.

        Returns:
            Parsed next visible message and terminal state from the text world-model protocol.

        Raises:
            TypeError: The action is not a canonical assistant action.
            ValueError: Tool names, call IDs, response identity, or generated results are invalid.
        """
        if not isinstance(action, AssistantAction):
            raise TypeError("world-model actions must use AssistantAction with original tool IDs")
        names = {tool.name for tool in tools}
        if any(call.name not in names for call in action.tool_calls):
            raise ValueError("declare every assistant tool in the tools argument")
        if len({call.call_id for call in action.tool_calls}) != len(action.tool_calls):
            raise ValueError("assistant tool calls must have unique call IDs")
        context = {} if initial_context is None else initial_context
        task_id = stable_id("world-model-step-task", {"task": task, "initial_context": context})
        task_case = TaskCase(
            task_id=task_id,
            lineage_group_id=task_id,
            partition="held_out",
            instruction=task,
            initial_context=context,
            tools=tools,
            workload_weight=1.0,
            source_trace_ids=(task_id,),
        )
        prepared = self.prepare_turn(
            task=task_case,
            visible_messages=(),
            candidate_response=action,
            excluded_lineage_ids=excluded_lineage_ids,
            maximum_output_tokens=maximum_output_tokens,
        )
        dispatched = self.complete_turn(prepared)
        if dispatched.response.model != self.artifact.model:
            raise ValueError("world-model response identity differs from its build artifact")
        return self.parse_turn(dispatched).transition


def load_grounded_world_model(
    store: ArtifactStore,
    artifact_id: str,
    *,
    client: ModelClient,
    capabilities: ModelCapabilities,
    embedder: RAGEmbedderBinding | None = None,
    token_counter: TokenCounter | None = None,
    evidence_selector: JEVSelector | None = None,
    tag_retrieval: TagRetrievalBinding | None = None,
) -> GroundedWorldModel:
    """Load and verify one executable grounded world-model artifact.

    Args:
        store: Project-local immutable artifact store.
        artifact_id: Completed grounded world-model artifact ID.
        client: Runtime client. Every returned response must match the artifact's exact model
            identity before its output is accepted.
        embedder: Exact explicit semantic embedding binding used to build the serving RAG.
        capabilities: Required resolved metadata for pre-dispatch packing. Its identity must
            match the artifact's frozen model snapshot, including unknown capacity fields.
        token_counter: Optional exact counter; otherwise uses a conservative UTF-8 bound.
        evidence_selector: Reranker whose identity must match the persisted policy.
        tag_retrieval: Optional tag policy and per-task task-view tags for ranking.

    Returns:
        Executable grounded world model.

    Raises:
        ValueError: Supplied capabilities differ from the artifact's frozen model identity.
    """
    stored = store.read(artifact_id)
    world_model_input = artifact_input(stored.manifest)
    artifact = _load_verified_artifact(store, world_model_input)
    loaded_rag = load_rag_index(store, artifact.serving_rag.artifact_id)
    if artifact.original_evidence == "none":
        if embedder is not None:
            raise ValueError("a memory-first simulator embeds no retrieval queries")
        embedder = RAGEmbedderBinding(
            client=_NoRetrievalQueries(), snapshot=loaded_rag.index.embedder
        )
    return GroundedWorldModel(
        artifact_input=world_model_input,
        artifact=artifact,
        retriever=TraceRAGRetriever(loaded_rag, embedder=embedder),
        client=client,
        capabilities=capabilities,
        token_counter=token_counter or Utf8UpperBoundTokenCounter(),
        evidence_selector=evidence_selector,
        tag_retrieval=tag_retrieval,
    )


def load_grounded_world_model_artifact(
    store: ArtifactStore,
    world_model_input: ArtifactInput,
) -> GroundedWorldModelArtifact:
    """Load one verified grounded world-model envelope without binding a provider client.

    Args:
        store: Project-local immutable artifact store.
        world_model_input: Exact completed grounded world-model manifest pointer.

    Returns:
        Fully verified persisted grounded world-model envelope.

    Raises:
        ValueError: Manifest, envelope, prompt, model, or content identity differs.
    """
    return _load_verified_artifact(store, world_model_input)


def verify_grounded_world_model_artifact(
    store: ArtifactStore,
    world_model_input: ArtifactInput,
) -> GroundedWorldModelArtifact:
    """Load and verify one persisted grounded world-model artifact without a client.

    Args:
        store: Project-local immutable artifact store.
        world_model_input: Exact selected grounded world-model manifest pointer.

    Returns:
        Fully verified, content-addressed grounded world-model envelope.

    Raises:
        ValueError: The manifest, envelope, model, prompt, or content identity differs.
    """
    return load_grounded_world_model_artifact(store, world_model_input)


class _NoRetrievalQueries:
    """Embedding seam of a memory-first simulator, which never retrieves recorded evidence."""

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Refuse every query; a memory-first prediction never reaches retrieval.

        Raises:
            ValueError: Always.
        """
        raise ValueError("a memory-first simulator never embeds retrieval queries")


def persist_memory_first_world_model(
    store: ArtifactStore, world_model_input: ArtifactInput
) -> ArtifactInput:
    """Persist the same world model in the memory-first profile, as its own simulator.

    The result keeps the built model, prompt, knowledge and serving-index identity (the stored
    evidence corpus) but retrieves no recorded transition: predictions read only the task,
    the visible history and learned memory. No embedder is needed to serve it. Its identity is
    content-derived, so repeated calls return the same artifact.

    Args:
        store: Project-local immutable artifact store.
        world_model_input: Exact completed grounded world-model manifest pointer.

    Returns:
        Manifest pointer of the memory-first grounded world model.
    """
    artifact = _load_verified_artifact(store, world_model_input)
    persisted = persist_grounded_world_model(
        store,
        artifact.serving_rag,
        model_alias=artifact.model_alias,
        model=artifact.model,
        created_at=artifact.created_at,
        code_revision=artifact.code_revision,
        top_k=artifact.top_k,
        knowledge=artifact.knowledge,
        original_evidence="none",
        construction_scope=artifact.construction_scope,
    )
    return artifact_input(persisted.manifest)


def bind_fit_grounded_world_model(
    store: ArtifactStore,
    world_model_input: ArtifactInput,
    *,
    client: ModelClient,
    fit_retriever: TraceRAGRetriever,
    capabilities: ModelCapabilities,
    token_counter: TokenCounter | None = None,
) -> GroundedWorldModel:
    """Bind a persisted world-model protocol to the exact fit-only simulation index.

    Args:
        store: Project-local immutable artifact store.
        world_model_input: Exact completed grounded world-model manifest pointer.
        client: Resolved world-model provider client.
        fit_retriever: Exact fit-only retriever used by optimization simulation.
        capabilities: Required resolved metadata matching the artifact's frozen identity.
        token_counter: Optional exact counter; otherwise uses a conservative UTF-8 bound.

    Returns:
        Artifact-bound executor that can retrieve only fit evidence.

    Raises:
        ValueError: Artifact, capability, source, schema, embedder, lineage, or top-k
            identity differs.
    """
    artifact = _load_verified_artifact(store, world_model_input)
    _require_fit_subset(store, artifact, fit_retriever.index)
    return GroundedWorldModel(
        artifact_input=world_model_input,
        artifact=artifact,
        retriever=fit_retriever,
        client=client,
        capabilities=capabilities,
        token_counter=token_counter or Utf8UpperBoundTokenCounter(),
    )


def persist_fit_only_world_model(
    store: ArtifactStore,
    world_model_input: ArtifactInput,
    fit_rag_input: ArtifactInput,
) -> ArtifactInput:
    """Persist the same world model over the build's fit-only index, as its own simulator.

    The build's serving index also holds held-out traces, so a simulator measured or used on
    held-out tasks must retrieve from the fit-only index instead. The result keeps the exact
    model, prompt, knowledge, reranker and top-k of the built world model; only its index
    changes. Its identity is content-derived, so repeated calls return the same artifact.

    Args:
        store: Project-local immutable artifact store.
        world_model_input: Exact completed grounded world-model manifest pointer.
        fit_rag_input: The same build's fit-only RAG index.

    Returns:
        Manifest pointer of the fit-only grounded world model.

    Raises:
        ValueError: The index is not the fit-only subset of this world model's serving index.
    """
    artifact = _load_verified_artifact(store, world_model_input)
    fit = load_rag_index(store, fit_rag_input.artifact_id)
    if artifact_input(fit.manifest) != fit_rag_input:
        raise ValueError("fit RAG manifest differs from its selected input")
    _require_fit_subset(store, artifact, fit.index)
    return _persist_over_index(store, artifact, fit_rag_input, artifact.construction_scope)


def persist_scoped_world_model(
    store: ArtifactStore,
    world_model_input: ArtifactInput,
    scoped_rag_input: ArtifactInput,
    *,
    retrieval_scope: CorpusScope,
    construction_scope: CorpusScope | None = None,
) -> ArtifactInput:
    """Persist the same world model over a scoped fit-only index, as its own simulator.

    The scoped index must hold exactly the fit transitions of this world model's serving index
    that ``retrieval_scope`` admits, and nothing held out or out of scope. When
    ``construction_scope`` is given, every trace the world model's learned knowledge was
    constructed from must lie inside it and inside the build's fit evidence. Construction and
    retrieval scopes are independent and may overlap. The retrieval scope is recorded in the
    scoped index, which becomes the new world model's index. The construction scope, given or
    already recorded on the world model, is verified against the serving index and recorded on
    the result; a world model that already records one accepts only that same scope. The result
    keeps the exact model, prompt, knowledge, reranker and top-k, and its identity is
    content-derived.

    Args:
        store: Project-local immutable artifact store.
        world_model_input: Exact completed grounded world-model manifest pointer.
        scoped_rag_input: Fit-only index built with ``corpus_scope=retrieval_scope``.
        retrieval_scope: The scope the caller requires the index to declare.
        construction_scope: Optional scope the world model's construction corpus must lie in;
            omitted, the world model's recorded construction scope, if any, is used.

    Returns:
        Manifest pointer of the scoped grounded world model.

    Raises:
        ValueError: The index declares another scope or differs from the in-scope fit subset,
            the construction scope differs from the recorded one, or the construction corpus
            is missing, held out, or outside its scope.
    """
    artifact = _load_verified_artifact(store, world_model_input)
    scoped = load_rag_index(store, scoped_rag_input.artifact_id)
    if artifact_input(scoped.manifest) != scoped_rag_input:
        raise ValueError("scoped RAG manifest differs from its selected input")
    recorded = artifact.construction_scope
    if recorded is not None and construction_scope not in (None, recorded):
        raise ValueError(
            f"world model records construction scope {recorded.name!r}; pass that exact scope "
            "or omit construction_scope"
        )
    serving = _require_fit_subset(store, artifact, scoped.index, scope=retrieval_scope)
    construction = recorded if construction_scope is None else construction_scope
    if construction is not None:
        _require_construction_scope(artifact, serving, construction)
    return _persist_over_index(store, artifact, scoped_rag_input, construction)


def _persist_over_index(
    store: ArtifactStore,
    artifact: GroundedWorldModelArtifact,
    rag_input: ArtifactInput,
    construction_scope: CorpusScope | None,
) -> ArtifactInput:
    """Persist a verified world model's exact protocol over another verified index.

    Args:
        store: Project-local immutable artifact store.
        artifact: Verified grounded world model whose protocol is kept.
        rag_input: Index already checked against the world model's serving index.
        construction_scope: Verified construction scope to record, or None for none.

    Returns:
        Manifest pointer of the persisted world model.
    """
    persisted = persist_grounded_world_model(
        store,
        rag_input,
        model_alias=artifact.model_alias,
        model=artifact.model,
        created_at=artifact.created_at,
        code_revision=artifact.code_revision,
        top_k=artifact.top_k,
        knowledge=artifact.knowledge,
        selection_policy_sha256=artifact.selection_policy_sha256,
        original_evidence=artifact.original_evidence,
        construction_scope=construction_scope,
    )
    return artifact_input(persisted.manifest)


def _require_fit_subset(
    store: ArtifactStore,
    artifact: GroundedWorldModelArtifact,
    fit: RAGIndex,
    *,
    scope: CorpusScope | None = None,
) -> LoadedRAGIndex:
    """Require a fit-only index built from the same graph as the artifact's serving index.

    Without a scope the index must be the serving index's complete fit subset. With one, it
    must declare exactly that scope, name only the serving index's fit lineages, and hold
    exactly the serving fit transitions the scope admits.

    Args:
        store: Project-local immutable artifact store.
        artifact: Verified grounded world model whose serving index covers fit and held out.
        fit: Candidate fit-only index envelope.
        scope: Corpus scope the index must declare, or None for an unscoped index.

    Returns:
        The verified serving index.

    Raises:
        ValueError: Partition scope, corpus scope, source, schema, embedder, lineage, top-k, or
            transition identity differs.
    """
    serving = load_rag_index(store, artifact.serving_rag.artifact_id)
    if artifact_input(serving.manifest) != artifact.serving_rag:
        raise ValueError("grounded world-model serving RAG manifest digest changed")
    if serving.index.included_partitions != ("fit", "held_out"):
        raise ValueError("grounded world-model serving RAG has an unsupported partition scope")
    if fit.included_partitions != ("fit",):
        raise ValueError("grounded simulation requires a fit-only retrieval index")
    if fit.corpus_scope != scope:
        raise ValueError(
            "fit RAG corpus scope differs from the declared scope; bind a scoped index only "
            "through persist_scoped_world_model with its exact scope"
        )
    serving_fit = set(serving.index.fit_lineage_ids)
    lineages_match = (
        serving.index.fit_lineage_ids == fit.fit_lineage_ids
        if scope is None
        else set(fit.fit_lineage_ids) <= serving_fit
    )
    if (
        serving.index.sources != fit.sources
        or serving.index.key_schema_version != fit.key_schema_version
        or serving.index.embedder != fit.embedder
        or serving.index.embedding_dimension != fit.embedding_dimension
        or serving.index.embedding_chunk_bytes != fit.embedding_chunk_bytes
        or not lineages_match
        or serving.index.default_top_k != fit.default_top_k
    ):
        raise ValueError("fit RAG identity differs from the grounded world-model build graph")
    if scope is not None and scope.kind == "trace":
        known = {trace for source in serving.index.sources for trace in source.trace_ids}
        outside = sorted(set(scope.ids) - known)
        if outside:
            raise ValueError(
                f"corpus scope {scope.name!r} names trace {outside[0]} outside the build's sources"
            )
    expected = tuple(
        sorted(
            transition.transition_id
            for transition in serving.transitions
            if transition.lineage_id in serving_fit
            and (
                scope is None
                or scope.admits(trace_id=transition.trace_id, lineage_id=transition.lineage_id)
            )
        )
    )
    if fit.transition_ids != expected:
        raise ValueError(
            "fit RAG transitions differ from the serving index fit subset"
            if scope is None
            else "scoped RAG transitions differ from the serving index's in-scope fit subset"
        )
    return serving


def _require_construction_scope(
    artifact: GroundedWorldModelArtifact, serving: LoadedRAGIndex, scope: CorpusScope
) -> None:
    """Require a world model's construction corpus to lie inside one scope and fit evidence.

    A trace's lineage is read from its serving transitions, so a lineage scope can be checked
    only for construction traces that produced at least one transition; declare a trace scope
    otherwise.

    Args:
        artifact: Verified grounded world model carrying its construction knowledge.
        serving: Verified serving index of the same build.
        scope: Scope the construction corpus must lie in.

    Raises:
        ValueError: The world model has no recorded construction corpus, or a construction trace
            is unknown, held out, or outside the scope.
    """
    if artifact.knowledge is None:
        raise ValueError(
            "a construction scope needs a world model with recorded construction knowledge"
        )
    fit_lineages = set(serving.index.fit_lineage_ids)
    lineage_of = {item.trace_id: item.lineage_id for item in serving.transitions}
    for trace_id in sorted(artifact.knowledge.source_trace_ids):
        lineage = lineage_of.get(trace_id)
        if lineage is not None and lineage not in fit_lineages:
            raise ValueError(f"world model was constructed from held-out trace {trace_id}")
        if scope.kind == "trace":
            admitted = trace_id in scope.ids
        elif lineage is None:
            raise ValueError(
                f"construction trace {trace_id} has no serving transition to resolve its "
                "lineage; declare a trace construction scope"
            )
        else:
            admitted = scope.admits(trace_id=trace_id, lineage_id=lineage)
        if not admitted:
            raise ValueError(
                f"world model was constructed from trace {trace_id} outside corpus scope "
                f"{scope.name!r}"
            )


def _load_verified_artifact(
    store: ArtifactStore,
    world_model_input: ArtifactInput,
) -> GroundedWorldModelArtifact:
    """Load one exact content-addressed grounded world-model envelope.

    Args:
        store: Project-local immutable artifact store.
        world_model_input: Exact manifest pointer selected by the caller.

    Returns:
        Fully verified persisted grounded world-model envelope.

    Raises:
        ValueError: Manifest, envelope, prompt, model, or content identity differs.
    """
    artifact_id = world_model_input.artifact_id
    stored = store.read(artifact_id)
    if stored.manifest.artifact_type != "grounded-world-model":
        raise ValueError(f"artifact {artifact_id!r} is not a grounded world model")
    if artifact_input(stored.manifest) != world_model_input:
        raise ValueError("grounded world-model manifest differs from its selected input")
    artifact = GroundedWorldModelArtifact.model_validate_json(
        store.read_bytes(artifact_id, WORLD_MODEL_ARTIFACT_PATH)
    )
    if not envelope_matches_manifest(artifact, stored.manifest):
        raise ValueError("grounded world-model envelope differs from its artifact manifest")
    if artifact.world_model_id != artifact_id:
        raise ValueError("grounded world-model artifact ID differs from its directory")
    content = grounded_world_model_content(
        serving_rag=artifact.serving_rag,
        model_alias=artifact.model_alias,
        model=artifact.model,
        prompt_version=artifact.prompt_version,
        prompt_sha256=artifact.prompt_sha256,
        top_k=artifact.top_k,
        code_revision=artifact.code_revision,
        knowledge=artifact.knowledge,
        selection_policy_sha256=artifact.selection_policy_sha256,
        original_evidence=artifact.original_evidence,
        construction_scope=artifact.construction_scope,
    )
    if stable_id("grounded-world-model", content) != artifact_id:
        raise ValueError("grounded world-model artifact ID differs from its complete content")
    _require_current_prompt(artifact)
    return artifact


def _require_current_prompt(artifact: GroundedWorldModelArtifact) -> None:
    """Require the frozen prompt identity before loading or directly constructing a runtime."""
    reconstructed = artifact.knowledge is not None
    version = (
        RECONSTRUCTED_TEXT_PROMPT_VERSION if reconstructed else GROUNDED_WORLD_MODEL_PROMPT_VERSION
    )
    if artifact.prompt_version != version:
        raise ValueError(
            "grounded world-model prompt version is not supported by this runtime; "
            "persist a new world model over the unchanged RAG index and prepare a new evaluation"
        )
    if artifact.prompt_sha256 != grounded_world_model_prompt_sha256(reconstructed=reconstructed):
        raise ValueError(
            "grounded world-model prompt digest differs from this runtime; "
            "persist a new world model over the unchanged RAG index and prepare a new evaluation"
        )
