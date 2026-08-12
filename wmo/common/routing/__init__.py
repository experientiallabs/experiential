"""Canonical router-policy and routing-decision contracts."""

from wmo.common.routing.embeddings import (
    FrozenEmbedding,
    FrozenEmbeddingClient,
    FrozenEmbeddingSet,
    load_frozen_embedding_set,
)
from wmo.common.routing.features import (
    ROUTER_FEATURE_EXTRACTOR_ID,
    ROUTER_FEATURE_SCHEMA_SHA256,
    RouterFeatureExtractor,
    RouterFeatureRecord,
)
from wmo.common.routing.policy import KnnGuard, KnnRouterPolicy, RouterPolicy, RoutingDecision

__all__ = [
    "ROUTER_FEATURE_EXTRACTOR_ID",
    "ROUTER_FEATURE_SCHEMA_SHA256",
    "KnnGuard",
    "KnnRouterPolicy",
    "RouterFeatureExtractor",
    "RouterFeatureRecord",
    "RouterPolicy",
    "RoutingDecision",
    "FrozenEmbedding",
    "FrozenEmbeddingClient",
    "FrozenEmbeddingSet",
    "load_frozen_embedding_set",
]
