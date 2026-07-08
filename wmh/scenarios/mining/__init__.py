"""Mining: reduce raw traces to facets, cluster them, and select representative source traces."""

from wmh.scenarios.mining.clustering import TraceCluster, cluster_facets, name_clusters
from wmh.scenarios.mining.facets import (
    FacetExtractor,
    Outcome,
    TraceFacet,
    tool_signature,
    trace_digest,
    trace_domain,
)
from wmh.scenarios.mining.selection import SelectedTrace, hybrid_select, semdedup_keep

__all__ = [
    "FacetExtractor",
    "Outcome",
    "SelectedTrace",
    "TraceCluster",
    "TraceFacet",
    "cluster_facets",
    "hybrid_select",
    "name_clusters",
    "semdedup_keep",
    "tool_signature",
    "trace_digest",
    "trace_domain",
]
