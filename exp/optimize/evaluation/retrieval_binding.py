"""Retrieval bindings shared by simulator evaluations: build-time embedder and leakage guard.

A built simulator retrieves recorded transitions from a semantic index whose query vectors must
come from the exact embedder that built it. Evaluations that dispatch inside their own request
budget and physical-attempt ledger (fidelity, trace analysis) resolve that embedder here, so
query embeddings are metered by the same owners as every completion. The local hashing index
needs no provider and binds nothing.

The same index decides what a simulator can see. A task declared held out must not be
answerable from its own recorded transitions, so evaluation refuses a simulator whose index
contains the task's source traces or lineage.
"""

from collections.abc import Sequence

from exp.common.core.artifacts import sha256_json
from exp.common.models import Embedding, EmbeddingClient, EmbeddingCostReservation, ModelCatalog
from exp.common.project import ProjectStore
from exp.common.tasks import TaskCase
from exp.optimize.evaluation.spending import BudgetedEmbedding
from exp.runtime.models import CapabilityRequirement, RuntimeModelCatalog
from exp.runtime.models.budget import RequestBudget
from exp.runtime.models.providers.transport import RetryPolicy
from exp.simulation.retrieval import RAGEmbedderBinding, load_rag_index
from exp.simulation.retrieval.embedding import default_rag_embedder
from exp.simulation.retrieval.embedding_inputs import embedding_chunk_bytes
from exp.simulation.world_model.artifact import GroundedWorldModelArtifact


def index_embedder_alias(
    project: ProjectStore, catalog: ModelCatalog, artifact: GroundedWorldModelArtifact
) -> str | None:
    """Find the configured alias that embeds queries for a simulator's retrieval index.

    Credential-free: compares static catalog identities only, so evaluation registration can
    refuse a mismatch before any provider client exists.

    Args:
        project: Project whose configured embedder role built the index.
        catalog: Frozen model catalog of the evaluation.
        artifact: Verified simulator whose serving index is bound.

    Returns:
        The project's embedder alias for a semantic index, or None for the local hashing index
        and for a memory-first simulator, which never retrieves.

    Raises:
        ValueError: No embedder is configured, or its identity differs from the index's.
    """
    if artifact.original_evidence == "none":
        return None
    index = load_rag_index(project.artifacts, artifact.serving_rag.artifact_id).index
    if default_rag_embedder(index.embedding_dimension).snapshot == index.embedder:
        return None
    config = project.load_project()
    alias = config.models.embedder if config.models is not None else catalog.roles.embedder
    if alias is None or alias not in catalog.models:
        raise ValueError(
            f"the simulator's retrieval index was embedded with {index.embedder.model_id}; "
            "configure that model as the project's embedder alias and run again"
        )
    configured, _ = RuntimeModelCatalog(catalog).snapshot(alias)
    if configured != index.embedder:
        raise ValueError(
            f"embedder alias {alias!r} serves {configured.model_id}, but the simulator's "
            f"retrieval index was embedded with {index.embedder.model_id}; restore the build's "
            "embedder configuration or rebuild the project"
        )
    return alias


def bind_index_embedder(
    registry: RuntimeModelCatalog,
    alias: str | None,
    artifact: GroundedWorldModelArtifact,
    project: ProjectStore,
    budget: RequestBudget,
) -> RAGEmbedderBinding | None:
    """Resolve the registered embedder and meter its query calls in the evaluation budget.

    Args:
        registry: Evaluation catalog whose transport admits every physical attempt through
            the evaluation's dispatch ledger.
        alias: Alias returned by :func:`index_embedder_alias` at registration.
        artifact: Simulator whose serving index the queries search.
        project: Project holding the index.
        budget: The evaluation's request budget; each query batch is one replayable request.

    Returns:
        Binding for a semantic index, or None for the local hashing index.

    Raises:
        ValueError: The resolved embedder differs from the index or cannot embed.
    """
    if alias is None:
        return None
    index = load_rag_index(project.artifacts, artifact.serving_rag.artifact_id).index
    resolved = registry.preflight(alias, CapabilityRequirement(requires_embeddings=True))
    if resolved.snapshot != index.embedder or resolved.embedding_client is None:
        raise ValueError("configured embedder changed since registration; register a new run")
    price = resolved.capabilities.input_cost_per_million_tokens_usd
    attempts = RetryPolicy().maximum_attempts
    limit = resolved.capabilities.context_window_tokens
    reservation = EmbeddingCostReservation(
        model=resolved.snapshot,
        input_usd_per_million_tokens=0.0 if price is None else price,
        maximum_attempts=attempts,
        maximum_input_tokens=embedding_chunk_bytes(limit),
    )
    return RAGEmbedderBinding(
        client=ScopedQueryEmbedding(
            BudgetedEmbedding(resolved.embedding_client, budget, reservation), budget
        ),
        snapshot=resolved.snapshot,
        maximum_attempts=attempts,
        input_usd_per_million_tokens=reservation.input_usd_per_million_tokens,
        maximum_input_tokens=limit,
    )


class ScopedQueryEmbedding:
    """Give each retrieval query batch its own content-addressed request-budget scope.

    Query embeddings run before the prediction they serve, outside any completion's logical
    scope. Scoping each batch by its exact texts makes the request coordinate deterministic:
    a resumed or repeated identical query replays its saved vectors without a provider call.
    """

    def __init__(self, inner: EmbeddingClient, budget: RequestBudget) -> None:
        """Wrap a budgeted embedding client.

        Args:
            inner: Client that records each call in ``budget``.
            budget: Request budget whose scope is set around each call.
        """
        self._inner = inner
        self._budget = budget

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Embed one query batch inside its content-addressed scope.

        Args:
            texts: Retrieval query chunks.

        Returns:
            New or replayed embeddings in input order.
        """
        with self._budget.scope("retrieval-query:" + sha256_json(list(texts))):
            return self._inner.embed(texts)


def require_unindexed_held_out_task(
    project: ProjectStore, artifact: GroundedWorldModelArtifact, task: TaskCase
) -> None:
    """Refuse a held-out task whose own evidence the simulator could retrieve.

    Args:
        project: Project holding the simulator's index.
        artifact: Simulator to be evaluated on the task.
        task: Public task; only held-out tasks are checked, and only for a simulator that
            retrieves recorded evidence.

    Raises:
        ValueError: The index contains a transition from the task's source traces or lineage.
    """
    if task.partition != "held_out" or artifact.original_evidence == "none":
        return
    sources = set(task.source_trace_ids)
    transitions = load_rag_index(project.artifacts, artifact.serving_rag.artifact_id).transitions
    if any(t.trace_id in sources or t.lineage_id == task.lineage_group_id for t in transitions):
        raise ValueError(
            f"the simulator's retrieval index contains held-out task {task.task_id}'s own "
            "recorded traces; bind a simulator that cannot retrieve them: the build's "
            "fit-only simulator, or the memory-first one that exp eval PROJECT --audit N "
            "uses"
        )
