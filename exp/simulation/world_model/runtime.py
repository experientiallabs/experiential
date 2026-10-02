"""Executable RAG-grounded text world model from completed build artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from exp.common.core.artifacts import (
    ArtifactInput,
    JsonObject,
    envelope_matches_manifest,
    stable_id,
)
from exp.common.models import (
    AssistantAction,
    ModelCapabilities,
    ModelClient,
    ModelMessage,
    ModelRequest,
    ModelResponse,
)
from exp.common.project import ArtifactStore, artifact_input
from exp.common.tasks import TaskCase, ToolSchema
from exp.simulation.engines.text.packing import pack_world_model_request
from exp.simulation.engines.text.prompt import (
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
    RAGMatch,
    RAGQuery,
    TraceRAGRetriever,
    load_rag_index,
)
from exp.simulation.retrieval.embedding import RAGEmbedderBinding
from exp.simulation.world_model.artifact import (
    GROUNDED_WORLD_MODEL_PROMPT_VERSION,
    WORLD_MODEL_ARTIFACT_PATH,
    GroundedWorldModelArtifact,
    grounded_world_model_content,
    grounded_world_model_prompt_sha256,
)


@dataclass(frozen=True)
class PreparedGroundedWorldModelCall:
    """One retrieved and framed grounded request before provider dispatch.

    Attributes:
        request: Fully framed world-model request ready for dispatch.
        matches: Ordered grounding examples selected for the action.
        action: Exact candidate action whose call IDs constrain generated results.
    """

    request: ModelRequest
    matches: tuple[RAGMatch, ...]
    action: AssistantAction

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
    """

    artifact_input: ArtifactInput
    artifact: GroundedWorldModelArtifact
    retriever: TraceRAGRetriever
    client: ModelClient
    capabilities: ModelCapabilities
    token_counter: TokenCounter = field(default_factory=Utf8UpperBoundTokenCounter)

    def __post_init__(self) -> None:
        """Reject a capacity binding that differs from the immutable build artifact."""
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
        )
        queries = tuple(
            RAGQuery(
                task=task.instruction,
                initial_context=task.initial_context,
                action=action,
                excluded_lineage_ids=excluded_lineage_ids,
                top_k=self.artifact.top_k,
            )
            for action in candidate_rag_actions(candidate_response)
        )
        batches = tuple(self.retriever.retrieve(query) for query in queries)
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
            maximum_output_tokens=maximum_output_tokens,
            state=state,
        )
        request = request.model_copy(update={"json_object_output": json_object_output})
        return PreparedGroundedWorldModelCall(
            request=request, matches=matches, action=candidate_response
        )

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

        Raises:
            ValueError: The output allowance is not positive.
            WorldModelCapacityError: Required framing cannot fit the declared capacities.
        """
        request = build_world_model_request(
            task,
            visible_messages=visible_messages,
            candidate_response=candidate_response,
            grounded_examples=(),
            maximum_output_tokens=maximum_output_tokens,
            state=state,
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

    Returns:
        Executable grounded world model.

    Raises:
        ValueError: Supplied capabilities differ from the artifact's frozen model identity.
    """
    stored = store.read(artifact_id)
    world_model_input = artifact_input(stored.manifest)
    artifact = _load_verified_artifact(store, world_model_input)
    loaded_rag = load_rag_index(store, artifact.serving_rag.artifact_id)
    return GroundedWorldModel(
        artifact_input=world_model_input,
        artifact=artifact,
        retriever=TraceRAGRetriever(loaded_rag, embedder=embedder),
        client=client,
        capabilities=capabilities,
        token_counter=token_counter or Utf8UpperBoundTokenCounter(),
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
    serving = load_rag_index(store, artifact.serving_rag.artifact_id)
    fit = fit_retriever.index
    if artifact_input(serving.manifest) != artifact.serving_rag:
        raise ValueError("grounded world-model serving RAG manifest digest changed")
    if serving.index.included_partitions != ("fit", "held_out"):
        raise ValueError("grounded world-model serving RAG has an unsupported partition scope")
    if fit.included_partitions != ("fit",):
        raise ValueError("grounded simulation requires a fit-only retrieval index")
    if (
        serving.index.sources != fit.sources
        or serving.index.key_schema_version != fit.key_schema_version
        or serving.index.embedder != fit.embedder
        or serving.index.embedding_dimension != fit.embedding_dimension
        or serving.index.embedding_chunk_bytes != fit.embedding_chunk_bytes
        or serving.index.fit_lineage_ids != fit.fit_lineage_ids
        or serving.index.default_top_k != fit.default_top_k
        or artifact.top_k != fit.default_top_k
    ):
        raise ValueError("fit RAG identity differs from the grounded world-model build graph")
    fit_lineages = set(fit.fit_lineage_ids)
    serving_fit_ids = tuple(
        sorted(
            transition.transition_id
            for transition in serving.transitions
            if transition.lineage_id in fit_lineages
        )
    )
    if fit.transition_ids != serving_fit_ids:
        raise ValueError("fit RAG transitions differ from the serving index fit subset")
    return GroundedWorldModel(
        artifact_input=world_model_input,
        artifact=artifact,
        retriever=fit_retriever,
        client=client,
        capabilities=capabilities,
        token_counter=token_counter or Utf8UpperBoundTokenCounter(),
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
    )
    if stable_id("grounded-world-model", content) != artifact_id:
        raise ValueError("grounded world-model artifact ID differs from its complete content")
    if artifact.prompt_version != GROUNDED_WORLD_MODEL_PROMPT_VERSION:
        raise ValueError("grounded world-model prompt version is not supported by this runtime")
    if artifact.prompt_sha256 != grounded_world_model_prompt_sha256():
        raise ValueError("grounded world-model prompt digest differs from this runtime")
    return artifact
