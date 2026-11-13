"""Canonical router-policy and routing-decision contracts."""

from exp.common.routing.embeddings import (
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
from exp.common.routing.features import (
    ROUTER_FEATURE_EXTRACTOR_ID,
    ROUTER_FEATURE_SCHEMA_SHA256,
    RouterFeatureExtractor,
    RouterFeatureRecord,
)
from exp.common.routing.policy import (
    CacheSwitchGuard,
    KnnGuard,
    KnnRouterPolicy,
    RouterPolicy,
    RoutingDecision,
    SwitchOutcome,
)

__all__ = [
    "ROUTER_FEATURE_EXTRACTOR_ID",
    "ROUTER_FEATURE_SCHEMA_SHA256",
    "CacheSwitchGuard",
    "KnnGuard",
    "KnnRouterPolicy",
    "RouterFeatureExtractor",
    "RouterFeatureRecord",
    "RouterEmbeddingReservation",
    "RouterPolicy",
    "RoutingDecision",
    "SwitchOutcome",
    "FrozenEmbedding",
    "FrozenEmbeddingClient",
    "FrozenEmbeddingSet",
    "ReservedFrozenEmbeddingSet",
    "load_frozen_embedding_set",
    "persist_router_embeddings",
    "router_embedding_reservation",
    "router_feature_token_upper_bound",
]
