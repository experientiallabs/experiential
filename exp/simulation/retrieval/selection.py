"""Runtime evidence selection over a caller-admitted immutable source catalog."""

import json
from collections.abc import Mapping
from typing import Protocol

from exp.common.core.artifacts import ContractModel, JsonObject, JsonValue, Sha256, sha256_json
from exp.common.models import ToolCall
from exp.simulation.retrieval.contracts import RAGAction, RAGQuery
from exp.simulation.retrieval.embedding import HashingRAGEmbedder
from exp.simulation.retrieval.retriever import TraceRAGRetriever


class SelectionQuery(ContractModel):
    """Permitted current-world selection input, without reference outcomes.

    Attributes:
        action: Current public tool invocation.
        state: Candidate-owned prestate.
        history: Prior public observations from this instance.
    """

    action: ToolCall
    state: JsonObject
    history: tuple[JsonObject, ...]


class EvidencePolicy(Protocol):
    """Rank already admissible source IDs without modifying their authority."""

    @property
    def identity(self) -> Sha256:
        """Return the frozen policy identity, including index and lineage restrictions."""
        ...

    def select(self, query: SelectionQuery, catalog: Mapping[str, JsonValue]) -> tuple[str, ...]:
        """Return ordered, unique IDs from this exact admitted catalog."""
        ...


class AllOriginals:
    """Deliver every admitted original in stable catalog order."""

    @property
    def identity(self) -> Sha256:
        """Identify an all-originals control, not a selective retrieval experiment."""
        return sha256_json({"policy": "all-admitted-originals"})

    def select(self, query: SelectionQuery, catalog: Mapping[str, JsonValue]) -> tuple[str, ...]:
        """Keep original source order; query content does not change this control."""
        del query
        return tuple(catalog)


class HashingOriginals:
    """Rank caller-admitted raw originals with the maintained local hashing embedder.

    This policy accepts the same raw-source catalog as reconstruction. It neither admits
    records into the production trace index nor certifies native evidence authority.
    It is a deterministic lexical control, not learned semantic retrieval or JEV.
    """

    def __init__(self, *, top_k: int, dimensions: int = 256) -> None:
        """Bind a positive result count and the exact existing hashing implementation."""
        if type(top_k) is not int or top_k < 1:
            raise ValueError("selection top_k must be a positive integer")
        self._top_k = top_k
        self._embedder = HashingRAGEmbedder(dimensions)

    @property
    def identity(self) -> Sha256:
        """Identify this local control without a remote encoder or source-authority claim."""
        return sha256_json(
            {
                "policy": "raw-originals-signed-token-hashing-v1",
                "dimensions": self._embedder.dimensions,
                "top_k": self._top_k,
            }
        )

    def select(self, query: SelectionQuery, catalog: Mapping[str, JsonValue]) -> tuple[str, ...]:
        """Recompute over the current admitted packet; no stale per-action evidence cache."""
        identities = tuple(catalog)
        texts = [
            query.model_dump_json(),
            *(
                json.dumps(catalog[key], ensure_ascii=False, sort_keys=True, allow_nan=False)
                for key in identities
            ),
        ]
        vectors = self._embedder.embed(texts)
        query_vector = vectors[0].values
        scores = [
            sum(a * b for a, b in zip(query_vector, value.values, strict=True))
            for value in vectors[1:]
        ]
        ranked = sorted(range(len(identities)), key=lambda index: (-scores[index], index))
        return tuple(identities[index] for index in ranked[: self._top_k])


class RAGOriginals:
    """Bridge the maintained retriever to explicit original-record identities."""

    def __init__(
        self,
        retriever: TraceRAGRetriever,
        *,
        source_by_transition: Mapping[str, str],
        excluded_lineages: tuple[str, ...],
        top_k: int,
    ) -> None:
        """Bind exact transition-to-original mapping and prospective lineage exclusions."""
        if type(top_k) is not int or top_k < 1:
            raise ValueError("selection top_k must be a positive integer")
        if retriever.index.embedder.provider != "local":
            raise ValueError("hosted embeddings require a budget-qualified retrieval adapter")
        self._retriever = retriever
        self._mapping = dict(source_by_transition)
        self._excluded, self._top_k = excluded_lineages, top_k

    @property
    def identity(self) -> Sha256:
        """Bind the real index, mapping, exclusions and retrieval count."""
        return sha256_json(
            {
                "policy": "trace-rag-originals",
                "index": self._retriever.rag_input.model_dump(mode="json"),
                "mapping": self._mapping,
                "excluded": list(self._excluded),
                "top_k": self._top_k,
            }
        )

    def select(self, query: SelectionQuery, catalog: Mapping[str, JsonValue]) -> tuple[str, ...]:
        """Retrieve without evaluator inputs; reject an unadmitted returned source."""
        matches = self._retriever.retrieve(
            RAGQuery(
                task=query.action.name,
                initial_context={"state": query.state, "history": list(query.history)},
                action=RAGAction(
                    kind="tool_call",
                    tool_name=query.action.name,
                    tool_arguments=query.action.arguments,
                ),
                excluded_lineage_ids=self._excluded,
                top_k=self._top_k,
            )
        )
        selected = []
        for match in matches:
            identity = self._mapping.get(match.transition.transition_id)
            if identity is None or identity not in catalog:
                raise ValueError("retriever returned a transition without an admitted original")
            if identity not in selected:
                selected.append(identity)
        return tuple(selected)


def selected_evidence(
    policy: EvidencePolicy, query: SelectionQuery, catalog: Mapping[str, JsonValue]
) -> tuple[dict[str, JsonValue], JsonObject]:
    """Resolve admissible content and distinguish identity, order and policy hashes."""
    identities = policy.select(query, catalog) if catalog else ()
    if len(set(identities)) != len(identities) or not set(identities) <= catalog.keys():
        raise ValueError("selection must return unique IDs from the admitted catalog")
    records = {identity: catalog[identity] for identity in identities}
    receipt: JsonObject = {
        "policy_sha256": policy.identity,
        "selected_ids": list(identities),
        "source_identities_sha256": sha256_json(list(identities)),
        "ordered_content_sha256": sha256_json([records[key] for key in identities]),
        "catalog_sha256": sha256_json(dict(catalog)),
    }
    return records, receipt
