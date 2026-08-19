"""Immutable retrieval grounded exclusively in verified real trace transitions."""

from __future__ import annotations

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
    RealTraceSourceIdentity,
)
from wmo.simulation.retrieval.embedding import (
    HashingRAGEmbedder,
    RAGEmbedderBinding,
    default_rag_embedder,
)
from wmo.simulation.retrieval.retriever import TraceRAGRetriever, load_fit_rag_retriever
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
    "RealTraceSourceIdentity",
    "TraceRAGRetriever",
    "default_rag_embedder",
    "load_rag_index",
    "load_fit_rag_retriever",
    "persist_trace_rag",
]
