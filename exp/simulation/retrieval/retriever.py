"""Read-only cosine retrieval from a loaded immutable observed-transition index.

The corpus and vectors are frozen before load and expose no mutation surface. A query may also
carry the current task's task-view tags and a declared :class:`TagRetrievalPolicy`; ranking then
adds the policy's weighted tag overlap to cosine similarity (see
:mod:`exp.simulation.retrieval.tags`). Without tags, ranking is cosine similarity alone.

Scoring is two-stage so large indexes stay fast without changing a single returned bit. One
float64 matrix-vector product scores every eligible transition. Its error against the exact
per-transition Python sum is bounded (see :func:`_score_error_bound`), so every candidate that
could reach the top ``k`` under exact scoring lies within twice that bound of the ``k``-th
approximate total. Only that shortlist is rescored with the exact Python sum, then sorted by
descending score and transition ID exactly as a full pure-Python scan would.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from exp.common.core.artifacts import ArtifactInput
from exp.common.models import EmbeddingCostReservation, NumericMeasurement, OperationEconomics
from exp.common.project import ArtifactStore, artifact_input
from exp.simulation.retrieval.contracts import RAGIndex, RAGMatch, RAGQuery
from exp.simulation.retrieval.embedding import (
    RAGEmbedderBinding,
    default_rag_embedder,
    embed_rag_texts,
)
from exp.simulation.retrieval.embedding_checkpoint import RAGEmbeddingCheckpoint
from exp.simulation.retrieval.embedding_inputs import plan_rag_embedding_inputs
from exp.simulation.retrieval.store import LoadedRAGIndex, load_rag_index
from exp.simulation.retrieval.tags import (
    TagRanking,
    TagRetrievalEvidence,
    TagRetrievalPolicy,
    TaskQueryTags,
)
from exp.simulation.retrieval.transitions import render_rag_key


class RAGQueryInputLimitError(ValueError):
    """An eligible query needs more embedding input than its frozen reservation."""


class TraceRAGRetriever:
    """Serve stable nearest real transitions without mutating the persisted corpus."""

    def __init__(
        self,
        loaded: LoadedRAGIndex,
        *,
        embedder: RAGEmbedderBinding | None = None,
        query_cache: RAGEmbeddingCheckpoint | None = None,
    ) -> None:
        """Bind one loaded index to the exact embedder identity used at build time.

        Args:
            loaded: Fully verified immutable RAG index.
            embedder: Explicit semantic embedder binding. The built-in hashing client is
                reconstructed automatically only when its complete snapshot matches the index.
            query_cache: Optional durable query-embedding store. Query chunks are then looked up
                by text digest under the index's model and chunk namespace before any provider
                call, and newly embedded chunks are committed, so a repeated identical query
                does not embed again. Omitted, every query embeds through the provider.

        Raises:
            ValueError: A semantic index lacks an explicit matching client, snapshots differ, or
                the query cache namespace differs from the index's model or chunk size.
        """
        if embedder is None:
            default = default_rag_embedder(loaded.index.embedding_dimension)
            if default.snapshot != loaded.index.embedder:
                raise ValueError(
                    "semantic RAG indexes require the exact explicit embedder used at build time"
                )
            embedder = default
        if embedder.snapshot != loaded.index.embedder:
            raise ValueError("RAG query embedder snapshot differs from the persisted index")
        if query_cache is not None and (
            query_cache.snapshot != loaded.index.embedder
            or query_cache.maximum_chunk_bytes != loaded.index.embedding_chunk_bytes
        ):
            raise ValueError("RAG query cache differs from the index's model or chunk identity")
        self._rag_input = artifact_input(loaded.manifest)
        self._index = loaded.index
        self._transitions = loaded.transitions
        self._matrix = loaded.matrix
        self._embedder = embedder
        self._query_cache = query_cache
        self._lineage_ids, self._lineage_codes = _codes(
            tuple(item.lineage_id for item in loaded.transitions)
        )
        self._transition_ids = tuple(item.transition_id for item in loaded.transitions)
        self._trace_ids, self._trace_codes = _codes(
            tuple(item.trace_id for item in loaded.transitions)
        )

    @property
    def rag_input(self) -> ArtifactInput:
        """Return the exact immutable RAG manifest bound to this retriever."""
        return self._rag_input

    @property
    def index(self) -> RAGIndex:
        """Return the verified immutable index envelope used for every query."""
        return self._index

    @property
    def maximum_attempts(self) -> int:
        """Return the maximum provider attempts made by each query-embedding batch."""
        return self._embedder.maximum_attempts

    @property
    def input_usd_per_million_tokens(self) -> float:
        """Return the active catalog price for query-embedding input."""
        return self._embedder.input_usd_per_million_tokens

    def estimate_query_economics(
        self,
        query: RAGQuery,
        reservation: EmbeddingCostReservation,
    ) -> OperationEconomics:
        """Price one exact query conservatively before its embedding call.

        Args:
            query: Complete query whose canonical key will be embedded.
            reservation: Immutable embedder identity, input price, and retry bound.

        Returns:
            Retry-inclusive estimated cost for the exact canonical query text.

        Raises:
            ValueError: The reservation differs from the active embedder or retry policy.
        """
        if reservation.model != self._index.embedder:
            raise ValueError("query-embedding reservation model differs from the fit RAG index")
        if reservation.maximum_attempts != self._embedder.maximum_attempts:
            raise ValueError("query-embedding reservation retry bound differs from the client")
        if reservation.input_usd_per_million_tokens != self.input_usd_per_million_tokens:
            raise ValueError("query-embedding reservation price differs from the active catalog")
        if not self._has_eligible_transition(query):
            return OperationEconomics(cost_usd=NumericMeasurement(value=0, provenance="estimated"))
        key_text = render_rag_key(
            task=query.task,
            initial_context=query.initial_context,
            action=query.action,
        )
        input_tokens = plan_rag_embedding_inputs(
            (key_text,), maximum_chunk_bytes=self._index.embedding_chunk_bytes
        ).maximum_input_tokens
        if input_tokens > reservation.maximum_input_tokens:
            raise RAGQueryInputLimitError(
                "canonical RAG query exceeds its reserved input-token ceiling"
            )
        maximum_input_tokens = input_tokens * reservation.maximum_attempts
        cost = maximum_input_tokens * reservation.input_usd_per_million_tokens / 1_000_000
        return OperationEconomics(cost_usd=NumericMeasurement(value=cost, provenance="estimated"))

    def retrieve(
        self,
        query: RAGQuery,
        *,
        tag_policy: TagRetrievalPolicy | None = None,
        query_tags: TaskQueryTags | None = None,
    ) -> tuple[RAGMatch, ...]:
        """Return top matches after excluding query-related lineages.

        Args:
            query: Request-visible task, initial context, action, lineage exclusions, and optional
                limit. An omitted limit uses the immutable index default.
            tag_policy: Optional declared combination of cosine similarity and task-tag overlap.
                Requires ``query_tags`` and an index recording task-view tags.
            query_tags: The current task's task-view tags, computed with the policy's bank and
                threshold. Requires ``tag_policy``.

        Returns:
            Up to ``top_k`` eligible real transitions. Without tags the score is cosine
            similarity. With tags it is cosine plus ``weight`` times the Jaccard overlap of fired
            task tags, and a ``require_shared_family`` policy first drops candidates sharing no
            filter-family tag with the task. Equal scores use transition ID order.

        Raises:
            ValueError: Query embedding dimensions differ from the frozen index, only one of the
                policy and query tags is given, or they disagree with the index's tag record.
        """
        ranking = self.tag_ranking(tag_policy, query_tags)
        eligible = np.flatnonzero(self._eligible_mask(query, ranking))
        if eligible.size == 0:
            return ()
        key_text = render_rag_key(
            task=query.task,
            initial_context=query.initial_context,
            action=query.action,
        )
        query_vector = embed_rag_texts(
            self._embedder,
            (key_text,),
            maximum_chunk_bytes=self._index.embedding_chunk_bytes,
            checkpoint=self._query_cache,
        )[0]
        if len(query_vector) != self._index.embedding_dimension:
            raise ValueError(
                f"RAG query embedding has dimension {len(query_vector)}, expected "
                f"{self._index.embedding_dimension}"
            )
        limit = self._index.default_top_k if query.top_k is None else query.top_k
        tag_term = None
        if ranking is not None:
            tag_term = TagTerm(
                weight=ranking.policy.weight,
                overlaps=tuple(ranking.overlap(trace_id) for trace_id in self._trace_ids),
                trace_codes=self._trace_codes,
            )
        ranked = rank_rows(
            self._matrix,
            query_vector,
            eligible,
            transition_ids=self._transition_ids,
            limit=limit,
            tag_term=tag_term,
        )
        return tuple(
            RAGMatch(transition=self._transitions[row], score=score)
            if overlap is None
            else RAGMatch(transition=self._transitions[row], score=score, tag_overlap=overlap)
            for row, score, overlap in ranked
        )

    def tag_ranking(
        self,
        tag_policy: TagRetrievalPolicy | None,
        query_tags: TaskQueryTags | None,
    ) -> TagRanking | None:
        """Validate one tag policy and query tags against this index's tag record.

        Args:
            tag_policy: Optional declared ranking policy.
            query_tags: Optional current-task task-view tags.

        Returns:
            The validated ranking, or None when neither is given.

        Raises:
            ValueError: Only one of the two is given, or they disagree with the index.
        """
        if tag_policy is None and query_tags is None:
            return None
        if tag_policy is None or query_tags is None:
            raise ValueError(
                "tag-aware retrieval needs both a tag policy and the task's query tags"
            )
        return TagRanking(tag_policy, query_tags, self._index.trace_tags)

    def tag_evidence(
        self, tag_policy: TagRetrievalPolicy, query_tags: TaskQueryTags
    ) -> TagRetrievalEvidence:
        """Return the evidence record of one validated tag-aware retrieval.

        Args:
            tag_policy: Declared ranking policy.
            query_tags: Current-task task-view tags.

        Returns:
            Policy, its identity, the query tags and the index matrix digest.

        Raises:
            ValueError: The policy or tags disagree with the index's tag record.
        """
        return TagRanking(tag_policy, query_tags, self._index.trace_tags).evidence

    def _has_eligible_transition(self, query: RAGQuery, ranking: TagRanking | None = None) -> bool:
        """Check lineage and tag-filter eligibility before estimating or embedding.

        Args:
            query: Query whose lineage exclusions apply.
            ranking: Optional validated tag ranking whose hard filter also applies.

        Returns:
            Whether any transition could be returned.
        """
        return bool(self._eligible_mask(query, ranking).any())

    def _eligible_mask(self, query: RAGQuery, ranking: TagRanking | None) -> npt.NDArray[np.bool_]:
        """Mark transitions outside excluded lineages that pass the tag filter.

        Args:
            query: Query whose lineage exclusions apply.
            ranking: Optional validated tag ranking whose hard filter also applies.

        Returns:
            One flag per transition, in transition order.
        """
        excluded = set(query.excluded_lineage_ids)
        lineage_admitted = np.fromiter(
            (lineage_id not in excluded for lineage_id in self._lineage_ids),
            dtype=np.bool_,
            count=len(self._lineage_ids),
        )
        mask = lineage_admitted[self._lineage_codes]
        if ranking is not None:
            trace_admitted = np.fromiter(
                (ranking.admits(trace_id) for trace_id in self._trace_ids),
                dtype=np.bool_,
                count=len(self._trace_ids),
            )
            mask &= trace_admitted[self._trace_codes]
        return mask


@dataclass(frozen=True)
class TagTerm:
    """Tag-overlap addend of tag-aware ranking, resolved for one query.

    Attributes:
        weight: Policy weight multiplying the overlap.
        overlaps: Jaccard overlap per distinct trace code.
        trace_codes: Distinct-trace code of every matrix row.
    """

    weight: float
    overlaps: tuple[float, ...]
    trace_codes: npt.NDArray[np.intp]


def rank_rows(
    matrix: npt.NDArray[np.float64],
    query_vector: tuple[float, ...],
    eligible: npt.NDArray[np.intp],
    *,
    transition_ids: tuple[str, ...],
    limit: int,
    tag_term: TagTerm | None = None,
) -> tuple[tuple[int, float, float | None], ...]:
    """Rank eligible rows exactly as a full pure-Python cosine scan would.

    The exact score of a row is ``sum(q * v for q, v in zip(query_vector, row))`` over Python
    floats, plus ``weight * overlap`` under tag ranking. Approximate totals from one
    matrix-vector product select a shortlist: a row whose exact total reaches the exact
    ``limit``-th total has an approximate total at least the approximate ``limit``-th total minus
    twice :func:`_score_error_bound`, so the shortlist holds every exact top-``limit`` row and
    every row tied with one. Only shortlisted rows are scored exactly. Non-finite approximate
    totals disable the shortlist so the exact path sees every row.

    Args:
        matrix: Read-only float64 row vectors.
        query_vector: Query embedding with the matrix width.
        eligible: Ascending row indexes that pass lineage exclusion and the tag filter.
        transition_ids: Transition ID per row, the tie-break key.
        limit: Positive number of rows to return.
        tag_term: Optional tag-overlap addend.

    Returns:
        Up to ``limit`` ``(row, score, overlap)`` triples by descending score, then transition
        ID. ``overlap`` is None without tag ranking.

    Raises:
        ValueError: An exact cosine score is not finite.
    """
    shortlist = eligible
    if eligible.size > limit:
        query = np.asarray(query_vector, dtype=np.float64)
        totals = (matrix @ query)[eligible]
        weight = 0.0
        if tag_term is not None:
            weight = tag_term.weight
            per_trace = np.asarray(tag_term.overlaps, dtype=np.float64)
            totals = totals + weight * per_trace[tag_term.trace_codes[eligible]]
        if bool(np.isfinite(totals).all()):
            threshold = float(np.partition(totals, totals.size - limit)[totals.size - limit])
            bound = _score_error_bound(
                dimension=query.size,
                query_norm=math.sqrt(math.fsum(value * value for value in query_vector)),
                weight=weight,
            )
            shortlist = eligible[totals >= threshold - 2 * bound]
    ranked: list[tuple[int, float, float | None]] = []
    for row in shortlist.tolist():
        values = matrix[row].tolist()
        score = sum(left * right for left, right in zip(query_vector, values, strict=True))
        if not math.isfinite(score):
            raise ValueError("RAG match scores must be finite")
        if tag_term is None:
            ranked.append((row, score, None))
            continue
        overlap = tag_term.overlaps[int(tag_term.trace_codes[row])]
        ranked.append((row, score + tag_term.weight * overlap, overlap))
    ranked.sort(key=lambda item: (-item[1], transition_ids[item[0]]))
    return tuple(ranked[:limit])


def _codes(values: tuple[str, ...]) -> tuple[tuple[str, ...], npt.NDArray[np.intp]]:
    """Encode repeated identities as distinct values and one code per position.

    Args:
        values: Identities in transition order.

    Returns:
        Distinct identities in first-occurrence order and each position's index into them.
    """
    positions: dict[str, int] = {}
    codes = np.fromiter(
        (positions.setdefault(value, len(positions)) for value in values),
        dtype=np.intp,
        count=len(values),
    )
    return tuple(positions), codes


def _score_error_bound(*, dimension: int, query_norm: float, weight: float) -> float:
    """Bound the gap between an approximate and an exact ranking total for one row.

    A float64 dot product of ``n`` terms, summed in any order with or without fused
    multiply-add, differs from the real value by at most ``gamma(n) * sum(|q_i v_i|)`` with
    ``gamma(n) = n u / (1 - n u)`` and ``u = 2**-53``. Persisted rows have unit norm within
    ``1e-6``, so by Cauchy-Schwarz ``sum(|q_i v_i|) <= 1.001 * |q|``. The matrix product and the
    exact Python sum each obey that bound, and adding the tag term rounds each total once more
    by at most ``u`` times its magnitude. The result doubles the sum of those terms as margin.

    Args:
        dimension: Vector width ``n``.
        query_norm: Euclidean norm of the query vector.
        weight: Tag-overlap weight, or 0.0 without tag ranking.

    Returns:
        A strict upper bound on ``|approximate total - exact total|``.
    """
    unit = 2.0**-53
    gamma = dimension * unit / (1 - dimension * unit)
    magnitude = 1.001 * query_norm
    return 2 * (2 * gamma * magnitude + 2 * unit * (magnitude + abs(weight) + 1))


def load_fit_rag_retriever(
    store: ArtifactStore,
    fit_rag_input: ArtifactInput,
    *,
    embedder: RAGEmbedderBinding | None = None,
    query_cache: RAGEmbeddingCheckpoint | None = None,
) -> TraceRAGRetriever:
    """Load the exact immutable fit-only index used by router simulation.

    Args:
        store: Project artifact store that owns the completed build.
        fit_rag_input: Exact manifest pointer recorded by the completed project build.
        embedder: Exact semantic embedder binding used to create the frozen vectors. Local
            hashing indexes reconstruct their deterministic embedder when this is omitted.
        query_cache: Optional durable query-embedding store; see :class:`TraceRAGRetriever`.

    Returns:
        A read-only retriever limited to the frozen fit lineage set.

    Raises:
        ValueError: The manifest pointer, partition scope, lineage scope, or embedder differs from
            the completed fit index.
    """
    loaded = load_rag_index(store, fit_rag_input.artifact_id)
    if artifact_input(loaded.manifest) != fit_rag_input:
        raise ValueError("fit RAG manifest digest differs from the completed project build")
    index = loaded.index
    if index.included_partitions != ("fit",):
        raise ValueError("router simulation requires a fit-only RAG index")
    if index.included_lineage_ids != index.fit_lineage_ids:
        raise ValueError("fit RAG included lineages differ from its frozen fit lineage set")
    return TraceRAGRetriever(loaded, embedder=embedder, query_cache=query_cache)
