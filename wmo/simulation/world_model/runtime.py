"""Executable RAG-grounded text world model from completed build artifacts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from wmo.common.core.artifacts import (
    ArtifactInput,
    JsonObject,
    envelope_matches_manifest,
    stable_id,
)
from wmo.common.models import (
    AssistantAction,
    ModelClient,
    ModelMessage,
    ModelRequest,
    ModelResponse,
)
from wmo.common.project import ArtifactStore, artifact_input
from wmo.common.tasks import TaskCase
from wmo.simulation.engines.text.prompt import (
    TextWorldModelTransition,
    build_world_model_request,
    parse_world_model_transition,
)
from wmo.simulation.retrieval import (
    RAGAction,
    RAGMatch,
    RAGQuery,
    TraceRAGRetriever,
    load_rag_index,
)
from wmo.simulation.retrieval.embedding import RAGEmbedderBinding
from wmo.simulation.world_model.artifact import (
    GROUNDED_WORLD_MODEL_PROMPT_VERSION,
    GROUNDED_WORLD_MODEL_SYSTEM_PROMPT,
    WORLD_MODEL_ARTIFACT_PATH,
    GroundedWorldModelArtifact,
    grounded_world_model_content,
    grounded_world_model_prompt_sha256,
)


@dataclass(frozen=True)
class PreparedGroundedWorldModelCall:
    """One retrieved and framed grounded request before provider dispatch."""

    request: ModelRequest
    matches: tuple[RAGMatch, ...]


@dataclass(frozen=True)
class DispatchedGroundedWorldModelCall:
    """One artifact-bound response paired with its exact request and retrieval evidence."""

    request: ModelRequest
    response: ModelResponse
    matches: tuple[RAGMatch, ...]


@dataclass(frozen=True)
class GroundedWorldModelCall:
    """One artifact-bound grounded request, response, retrieval set, and parsed transition."""

    request: ModelRequest
    response: ModelResponse
    matches: tuple[RAGMatch, ...]
    transition: TextWorldModelTransition


@dataclass(frozen=True)
class GroundedWorldModel:
    """Call one configured model with nearest observed transitions as immutable evidence."""

    artifact_input: ArtifactInput
    artifact: GroundedWorldModelArtifact
    retriever: TraceRAGRetriever
    client: ModelClient

    def prepare_turn(
        self,
        *,
        task: TaskCase,
        visible_messages: Sequence[ModelMessage],
        candidate_response: AssistantAction,
        excluded_lineage_ids: tuple[str, ...],
        maximum_output_tokens: int,
    ) -> PreparedGroundedWorldModelCall:
        """Retrieve and frame one fit- or serving-bound grounded text transition.

        Args:
            task: Current canonical task and safe initial context.
            visible_messages: Candidate-visible conversation through the latest request.
            candidate_response: Latest visible candidate action.
            excluded_lineage_ids: Source lineages forbidden from retrieval.
            maximum_output_tokens: Explicit provider output ceiling.

        Returns:
            Exact request and retrieved evidence before provider dispatch.
        """
        if candidate_response.content is None:
            raise ValueError("grounded text simulation requires a visible candidate message")
        query = RAGQuery(
            task=task.instruction,
            initial_context=task.initial_context,
            action=RAGAction(kind="message", content=candidate_response.content),
            excluded_lineage_ids=excluded_lineage_ids,
            top_k=self.artifact.top_k,
        )
        matches = self.retriever.retrieve(query)
        request = build_world_model_request(
            task,
            visible_messages=visible_messages,
            candidate_response=candidate_response,
            grounded_examples=matches,
            maximum_output_tokens=maximum_output_tokens,
        )
        return PreparedGroundedWorldModelCall(request=request, matches=matches)

    def complete_turn(
        self,
        prepared: PreparedGroundedWorldModelCall,
    ) -> DispatchedGroundedWorldModelCall:
        """Dispatch one prepared artifact-bound request.

        Args:
            prepared: Exact retrieved evidence and request produced by ``prepare_turn``.

        Returns:
            Exact request, response, and retrieved evidence.
        """
        response = self.client.complete(prepared.request)
        return DispatchedGroundedWorldModelCall(
            request=prepared.request,
            response=response,
            matches=prepared.matches,
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
        return GroundedWorldModelCall(
            request=dispatched.request,
            response=dispatched.response,
            matches=dispatched.matches,
            transition=parse_world_model_transition(dispatched.response.output),
        )

    def step(
        self,
        *,
        task: str,
        action: RAGAction,
        initial_context: JsonObject | None = None,
        excluded_lineage_ids: tuple[str, ...] = (),
        maximum_output_tokens: int = 1_024,
    ) -> TextWorldModelTransition:
        """Predict one next visible observation grounded on nearest real transitions.

        Args:
            task: Current request-visible task.
            action: Latest assistant message or tool call.
            initial_context: Safe request-visible starting context.
            excluded_lineage_ids: Source lineages forbidden for this query.
            maximum_output_tokens: Explicit provider output ceiling.

        Returns:
            Parsed next visible message and terminal state from the text world-model protocol.
        """
        context = {} if initial_context is None else initial_context
        query = RAGQuery(
            task=task,
            initial_context=context,
            action=action,
            excluded_lineage_ids=excluded_lineage_ids,
            top_k=self.artifact.top_k,
        )
        matches = self.retriever.retrieve(query)
        evidence = [
            {
                "task": match.transition.task,
                "action": match.transition.action.model_dump(mode="json", exclude_none=True),
                "observation": match.transition.observation.model_dump(mode="json"),
            }
            for match in matches
        ]
        request = ModelRequest(
            messages=(
                ModelMessage(
                    role="system",
                    content=GROUNDED_WORLD_MODEL_SYSTEM_PROMPT,
                ),
                ModelMessage(
                    role="user",
                    content=json.dumps(
                        {
                            "task": task,
                            "initial_context": context,
                            "action": action.model_dump(mode="json", exclude_none=True),
                            "grounded_examples": evidence,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                ),
            ),
            tool_choice="none",
            maximum_output_tokens=maximum_output_tokens,
        )
        response = self.client.complete(request)
        if response.model != self.artifact.model:
            raise ValueError("world-model response identity differs from its build artifact")
        return parse_world_model_transition(response.output)


def load_grounded_world_model(
    store: ArtifactStore,
    artifact_id: str,
    *,
    client: ModelClient,
    embedder: RAGEmbedderBinding | None = None,
) -> GroundedWorldModel:
    """Load and verify one executable grounded world-model artifact.

    Args:
        store: Project-local immutable artifact store.
        artifact_id: Completed grounded world-model artifact ID.
        client: Runtime client. Every returned response must match the artifact's exact model
            identity before its output is accepted.
        embedder: Exact explicit semantic embedding binding used to build the serving RAG.

    Returns:
        Executable grounded world model.
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
    )


def bind_fit_grounded_world_model(
    store: ArtifactStore,
    world_model_input: ArtifactInput,
    *,
    client: ModelClient,
    fit_retriever: TraceRAGRetriever,
) -> GroundedWorldModel:
    """Bind a persisted world-model protocol to the exact fit-only simulation index.

    Args:
        store: Project-local immutable artifact store.
        world_model_input: Exact completed grounded world-model manifest pointer.
        client: Resolved world-model provider client.
        fit_retriever: Exact fit-only retriever used by optimization simulation.

    Returns:
        Artifact-bound executor that can retrieve only fit evidence.

    Raises:
        ValueError: Artifact, source, schema, embedder, lineage, or top-k identity differs.
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
