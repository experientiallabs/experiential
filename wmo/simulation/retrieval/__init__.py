"""Immutable retrieval grounded exclusively in verified real trace transitions."""

from wmo.simulation.retrieval.build import PersistedRAGIndex, persist_trace_rag
from wmo.simulation.retrieval.contracts import (
    RAGAction,
    RAGIndex,
    RAGLineageBinding,
    RAGMatch,
    RAGObservation,
    RAGQuery,
    RAGSourceRef,
    RAGTransition,
    RAGVector,
)
from wmo.simulation.retrieval.embedding import (
    HashingRAGEmbedder,
    RAGEmbedderBinding,
    default_rag_embedder,
)
from wmo.simulation.retrieval.retriever import TraceRAGRetriever
from wmo.simulation.retrieval.store import LoadedRAGIndex, load_rag_index

__all__ = [
    "HashingRAGEmbedder",
    "LoadedRAGIndex",
    "PersistedRAGIndex",
    "RAGAction",
    "RAGEmbedderBinding",
    "RAGIndex",
    "RAGLineageBinding",
    "RAGMatch",
    "RAGObservation",
    "RAGQuery",
    "RAGSourceRef",
    "RAGTransition",
    "RAGVector",
    "TraceRAGRetriever",
    "default_rag_embedder",
    "load_rag_index",
    "persist_trace_rag",
]
