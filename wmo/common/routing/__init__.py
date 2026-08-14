"""Canonical router-policy and routing-decision contracts."""

from wmo.common.routing.embeddings import (
    FrozenEmbedding,
    FrozenEmbeddingClient,
    FrozenEmbeddingSet,
    ReservedFrozenEmbeddingSet,
    RouterEmbeddingReservation,
    load_frozen_embedding_set,
    persist_router_embeddings,
    router_embedding_reservation,
    router_feature_token_upper_bound,
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
    "RouterEmbeddingReservation",
    "RouterPolicy",
    "RoutingDecision",
    "FrozenEmbedding",
    "FrozenEmbeddingClient",
    "FrozenEmbeddingSet",
    "ReservedFrozenEmbeddingSet",
    "load_frozen_embedding_set",
    "persist_router_embeddings",
    "router_embedding_reservation",
    "router_feature_token_upper_bound",
]
