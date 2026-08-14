"""Immutable retrieval grounded exclusively in verified real trace transitions."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wmo.simulation.retrieval.refresh import (
        PersistedRuntimeRAGRefresh as PersistedRuntimeRAGRefresh,
    )
    from wmo.simulation.retrieval.refresh import RuntimeRAGRefresh as RuntimeRAGRefresh
    from wmo.simulation.retrieval.refresh import (
        RuntimeRAGRefreshError as RuntimeRAGRefreshError,
    )
    from wmo.simulation.retrieval.refresh import (
        load_runtime_rag_refresh as load_runtime_rag_refresh,
    )
    from wmo.simulation.retrieval.refresh import (
        refresh_runtime_trace_rag as refresh_runtime_trace_rag,
    )
    from wmo.simulation.retrieval.refresh_dataset import (
        PersistedRuntimeRAGDataset as PersistedRuntimeRAGDataset,
    )
    from wmo.simulation.retrieval.refresh_dataset import (
        RuntimeRAGDatasetError as RuntimeRAGDatasetError,
    )
    from wmo.simulation.retrieval.runtime_stitching import (
        RuntimeTraceStitchingError as RuntimeTraceStitchingError,
    )

from wmo.simulation.retrieval.build import PersistedRAGIndex, persist_trace_rag
from wmo.simulation.retrieval.build_inputs import load_completed_build_rag_lineage_bindings
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

_REFRESH_EXPORT_MODULES = {
    "PersistedRuntimeRAGDataset": "wmo.simulation.retrieval.refresh_dataset",
    "PersistedRuntimeRAGRefresh": "wmo.simulation.retrieval.refresh",
    "RuntimeRAGDatasetError": "wmo.simulation.retrieval.refresh_dataset",
    "RuntimeRAGRefresh": "wmo.simulation.retrieval.refresh",
    "RuntimeRAGRefreshError": "wmo.simulation.retrieval.refresh",
    "RuntimeTraceStitchingError": "wmo.simulation.retrieval.runtime_stitching",
    "load_runtime_rag_refresh": "wmo.simulation.retrieval.refresh",
    "refresh_runtime_trace_rag": "wmo.simulation.retrieval.refresh",
}

__all__ = [
    "HashingRAGEmbedder",
    "LoadedRAGIndex",
    "PersistedRAGIndex",
    "PersistedRuntimeRAGDataset",
    "PersistedRuntimeRAGRefresh",
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
    "RuntimeRAGDatasetError",
    "RuntimeRAGRefresh",
    "RuntimeRAGRefreshError",
    "RuntimeTraceStitchingError",
    "TraceRAGRetriever",
    "default_rag_embedder",
    "load_runtime_rag_refresh",
    "load_rag_index",
    "load_fit_rag_retriever",
    "load_completed_build_rag_lineage_bindings",
    "persist_trace_rag",
    "refresh_runtime_trace_rag",
]


def __getattr__(name: str) -> object:
    """Resolve a runtime-refresh export without loading router services eagerly.

    Args:
        name: Package attribute requested by Python.

    Returns:
        The supported public refresh object loaded from its owning module.

    Raises:
        AttributeError: The name is not a public runtime-refresh export.
    """
    module_name = _REFRESH_EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return module globals plus supported runtime-refresh exports."""
    return sorted(set(globals()) | set(_REFRESH_EXPORT_MODULES))
