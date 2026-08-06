"""Mining: reduce raw traces to facets, cluster them, and select representative source traces."""

from wmo.simulation.scenarios.mining.clustering import TraceCluster, cluster_facets, name_clusters
from wmo.simulation.scenarios.mining.facets import (
    FacetExtractor,
    Outcome,
    TraceFacet,
    tool_signature,
    trace_digest,
    trace_domain,
)
from wmo.simulation.scenarios.mining.selection import SelectedTrace, hybrid_select, semdedup_keep

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
