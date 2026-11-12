"""Served-runtime binding of pre-turn state retrieval (state RAG) over a family's tagged fit steps.

A grounded runtime that names a :class:`StateRecallBinding` retrieves, before every predicted
tool response, the recorded transitions whose pre-turn state agrees best with the live one
(:class:`~exp.simulation.retrieval.state_rag.StateRAG`). The live state is tagged by the
admitted native decision alias (Clef by default) through the project decision cache, so a
replayed episode makes no call; the tie-break embedder is the world model's own retrieval
embedder and is bound only when the index stored vectors with that exact model.

A binding that also names a ``transition_bank`` serves transition-tag retrieval instead
(:class:`~exp.simulation.retrieval.transition_tags.TransitionRAG`): ``index`` is then a
``transition-tag-index`` and the live (state, action) is tagged in the predicted framing. Without
it the binding, its identity and its behavior are exactly today's state-tag retrieval.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from pydantic import Field

from exp.common.core.artifacts import ArtifactInput, ContractModel, JsonObject
from exp.common.models import ModelCatalog
from exp.common.project import ProjectStore
from exp.optimize.evaluation.fidelity.decision_access import decision_caller
from exp.runtime.models.providers.admitted_transport import AdmittedJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.mining.decisions import DEFAULT_DECISION_ALIAS, DecisionCache
from exp.simulation.retrieval import RAGEmbedderBinding
from exp.simulation.retrieval.embedding import embed_rag_texts
from exp.simulation.retrieval.state_rag import (
    StateEmbedder,
    StateRAG,
    StateRetrievalPolicy,
    load_state_tag_index,
)
from exp.simulation.retrieval.state_tags import cached_state_decider
from exp.simulation.retrieval.store import load_rag_index
from exp.simulation.retrieval.transition_tags import (
    TransitionRAG,
    TransitionRetrievalPolicy,
    load_transition_bank,
    load_transition_index,
)
from exp.simulation.world_model.runtime import load_grounded_world_model_artifact


class StateRecallBinding(ContractModel):
    """A pinned state-tag index and the frozen policy a served runtime retrieves with.

    Attributes:
        index: Saved ``state-tag-index`` artifact built over the world model's fit steps.
        policy: Frozen agreement scoring.
        decision_alias: Admitted native decision alias tagging each live state.
        transition_bank: Saved ``transition-tag-bank`` version; when set, ``index`` is its
            ``transition-tag-index`` and retrieval matches transitions, not states.
        transition_policy: Transition scoring, used only with ``transition_bank``.
    """

    index: ArtifactInput
    policy: StateRetrievalPolicy = Field(default_factory=StateRetrievalPolicy)
    decision_alias: str = Field(default=DEFAULT_DECISION_ALIAS, min_length=1)
    transition_bank: ArtifactInput | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    transition_policy: TransitionRetrievalPolicy | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


def check_state_recall(catalog: ModelCatalog, binding: StateRecallBinding) -> None:
    """Refuse a binding whose decision alias is not on the native decision wire.

    Raises:
        ValueError: The alias is unknown or not a typed-decision connection.
    """
    alias = catalog.models.get(binding.decision_alias)
    if alias is None or catalog.connections[alias.connection].provider != "typesafe":
        raise ValueError("state recall requires the admitted native decision wire")


def bind_state_recall(
    project: ProjectStore,
    binding: StateRecallBinding,
    *,
    world_model: ArtifactInput,
    registry: RuntimeModelCatalog,
    transport: AdmittedJsonTransport,
    deadline: datetime,
    clock: Callable[[], datetime],
    record: Callable[[JsonObject], None],
    embedding: RAGEmbedderBinding | None,
) -> StateRAG | TransitionRAG:
    """Return the served state retrieval over the world model's own fit transitions.

    Args:
        project: Project owning the index and the decision cache.
        binding: The pinned index and policy.
        world_model: The runtime's grounded world model; its serving RAG supplies transitions.
        registry: Runtime catalog over the admitted transport.
        transport: The runtime's shared admitted physical transport.
        deadline: Shared provider deadline.
        clock: Clock for decision accounting.
        record: Durable receipt sink.
        embedding: The world model's retrieval embedder, used for ties when the index matches.

    Returns:
        The bound retrieval.

    Raises:
        ValueError: The index was built over other transitions or another embedder.
    """
    artifact = load_grounded_world_model_artifact(project.artifacts, world_model)
    loaded = load_rag_index(project.artifacts, artifact.serving_rag.artifact_id)
    native = registry.resolve(binding.decision_alias)
    model = native.snapshot.model_dump(mode="json")
    decide = cached_state_decider(
        DecisionCache(project.paths),
        decision_caller(native, transport, deadline, clock),
        model,
    )
    if binding.transition_bank is not None:
        transition_index = load_transition_index(project.artifacts, binding.index)
        if transition_index.embedder is not None:
            raise ValueError("served transition retrieval binds only an unembedded index")
        return TransitionRAG(
            load_transition_bank(project.artifacts, binding.transition_bank),
            transition_index,
            binding.transition_policy or TransitionRetrievalPolicy(),
            {transition.transition_id: transition for transition in loaded.transitions},
            decide=decide,
            record=record,
        )
    index = load_state_tag_index(project.artifacts, binding.index)
    embedder = None
    if index.embedder is not None:
        if embedding is None or embedding.snapshot.model_dump(mode="json") != index.embedder:
            raise ValueError("the state-tag index was embedded with another model")
        bound = embedding
        embedder = StateEmbedder(
            model=index.embedder, embed=lambda texts: embed_rag_texts(bound, texts)
        )
    return StateRAG(
        index,
        binding.policy,
        {transition.transition_id: transition for transition in loaded.transitions},
        decide=decide,
        record=record,
        embedder=embedder,
    )
