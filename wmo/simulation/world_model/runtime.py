"""Executable RAG-grounded text world model from completed build artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass

from wmo.common.core.artifacts import JsonObject, stable_id
from wmo.common.models import ModelClient, ModelMessage, ModelRequest
from wmo.common.project import ArtifactStore, artifact_input
from wmo.simulation.engines.text import TextWorldModelTransition, parse_world_model_transition
from wmo.simulation.retrieval import RAGAction, RAGQuery, TraceRAGRetriever, load_rag_index
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
class GroundedWorldModel:
    """Call one configured model with nearest observed transitions as immutable evidence."""

    artifact: GroundedWorldModelArtifact
    retriever: TraceRAGRetriever
    client: ModelClient

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
    if stored.manifest.artifact_type != "grounded-world-model":
        raise ValueError(f"artifact {artifact_id!r} is not a grounded world model")
    artifact = GroundedWorldModelArtifact.model_validate_json(
        store.read_bytes(artifact_id, WORLD_MODEL_ARTIFACT_PATH)
    )
    manifest_identity = (
        stored.manifest.schema_version,
        stored.manifest.created_at,
        stored.manifest.inputs,
        stored.manifest.code_revision,
        stored.manifest.source,
    )
    envelope_identity = (
        artifact.schema_version,
        artifact.created_at,
        artifact.inputs,
        artifact.code_revision,
        artifact.source,
    )
    if manifest_identity != envelope_identity:
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
    loaded_rag = load_rag_index(store, artifact.serving_rag.artifact_id)
    if artifact_input(loaded_rag.manifest) != artifact.serving_rag:
        raise ValueError("grounded world-model serving RAG manifest digest changed")
    return GroundedWorldModel(
        artifact=artifact,
        retriever=TraceRAGRetriever(loaded_rag, embedder=embedder),
        client=client,
    )
