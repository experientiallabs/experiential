"""Deterministic cache-affinity placement over a pool's certified rungs.

Under a pool's ``maximize_cache_affinity`` policy the initial dispatch order is
a per-request weighted rendezvous (highest-random-weight) permutation: every
worker computes the same order from the same request fingerprint and catalog
weights, so fleet-wide conversation-to-rung affinity needs no per-worker memory
and no shared-state reads. Removing a rung (death, shed, flex-start cycling)
moves only that rung's fingerprints to their deterministic next choice; every
other placement is untouched, so a spilled conversation lands on the SAME
alternate every time and builds warm cache there instead of scattering.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from urllib.parse import urlsplit

from exp.common.core.artifacts import canonical_json_bytes
from exp.common.models.catalog import BillingSource
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.models.providers.base import GatewayWireProfile


def affinity_seed_material(
    request: GatewayRequest,
    *,
    continuation_episode_key: str | None,
    request_id: str,
) -> str:
    """Choose the request's stable conversation identity for affinity hashing.

    Reuses the identities the lineage machinery already carries rather than
    recomputing a prefix digest: the caller's explicit ``prompt_cache_key``
    (OpenAI's own same-prefix cache-routing hint) wins, then a Responses
    continuation's ORIGINAL episode key (so a continued conversation keeps its
    placement), then the session-scoped ``X-Client-Request-Id`` that agent CLIs
    send on every request of a session, then the per-operation idempotency
    key. A request carrying none of these degrades to its request id, which
    rendezvous turns into a weighted spread instead of a fixed rung.

    Args:
        request: Canonical decoded request.
        continuation_episode_key: The original turn's episode key when this
            request continues a stored response, else ``None``.
        request_id: The accepted request's id, the last-resort material.

    Returns:
        The strongest available stable conversation identity.
    """
    return (
        request.prompt_cache_key
        or continuation_episode_key
        or request.client_request_id
        or request.idempotency_key
        or request_id
    )


def affinity_fingerprint(
    *,
    organization_id: str,
    identity_id: str,
    material: str,
) -> bytes:
    """Derive the tenant-isolated affinity fingerprint for one request.

    The fingerprint deliberately EXCLUDES the alias revision: a catalog write
    (the daily price sync, one admin edit) must not scatter every warm
    conversation across rungs. Tenant fields keep one organization's chosen
    material from colliding with another's.

    Args:
        organization_id: Authorized organization.
        identity_id: Authorized identity within the organization.
        material: Stable conversation identity from
            :func:`affinity_seed_material`.

    Returns:
        A 32-byte digest suitable as a rendezvous key.
    """
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "organization_id": organization_id,
                "identity_id": identity_id,
                "material": material,
            }
        )
    ).digest()


def rendezvous_order(
    fingerprint: bytes,
    weighted_rungs: Sequence[tuple[str, float]],
) -> tuple[int, ...]:
    """Order rung indexes by descending weighted rendezvous score.

    Standard highest-random-weight hashing with the weighted score
    ``-weight / ln(u)`` where ``u`` is a uniform draw in (0, 1) keyed on the
    fingerprint and the rung's deployment id alone (never the connection
    digest, so a credential rotation does not move placement). Pure integer
    and float arithmetic over already-loaded catalog data: identical on every
    worker, no state.

    Args:
        fingerprint: 32-byte affinity fingerprint.
        weighted_rungs: One ``(deployment_id, weight)`` per rung, in route
            order; weights must be positive and finite.

    Returns:
        A permutation of ``range(len(weighted_rungs))``, best rung first,
        with the route index breaking (practically impossible) score ties.

    Raises:
        ValueError: A weight is not positive and finite.
    """
    scores: list[float] = []
    for deployment_id, weight in weighted_rungs:
        if not (weight > 0) or math.isinf(weight):
            raise ValueError("rendezvous weights must be positive and finite")
        digest = hashlib.blake2b(
            deployment_id.encode("utf-8"),
            digest_size=8,
            key=fingerprint,
        ).digest()
        # Center the draw inside (0, 1) so ln(u) is finite and nonzero.
        uniform = (int.from_bytes(digest, "big") + 0.5) / 2.0**64
        scores.append(-weight / math.log(uniform))
    return tuple(sorted(range(len(scores)), key=lambda index: (-scores[index], index)))


def affinity_weighted_rungs(
    deployments: Sequence[ExactModelDeployment],
) -> tuple[tuple[str, float], ...]:
    """Return each rung's ``(deployment_id, weight)`` for :func:`rendezvous_order`.

    The weight is the deployment's authored
    ``GatewayRungDispatchPolicy.affinity_weight``, default 1.0.
    """
    return tuple(
        (
            deployment.deployment_id,
            (
                1.0
                if deployment.gateway.dispatch is None
                or deployment.gateway.dispatch.affinity_weight is None
                else deployment.gateway.dispatch.affinity_weight
            ),
        )
        for deployment in deployments
    )


def house_aggregator_last(
    order: Sequence[int],
    deployments: Sequence[ExactModelDeployment],
    profiles: Sequence[GatewayWireProfile],
) -> tuple[int, ...]:
    """Move host-funded OpenRouter rungs behind every other rung of an affinity order.

    OpenRouter is never the first lane (owner rule): it is an aggregator the
    house buys through, so a direct lane of the same model always leads it.
    Rendezvous order is blind to that and put a house OpenRouter rung first
    for its share of fingerprints. The demotion is a stable partition, so the
    rendezvous order of the other rungs (and of the OpenRouter rungs among
    themselves) is kept and placement stays deterministic on every worker. A
    customer-managed OpenRouter rung is the customer's own key and keeps its
    place, and an all-OpenRouter route is left as it is.

    Args:
        order: Route indexes in affinity order.
        deployments: The route's deployments, in route order.
        profiles: The route's wire profiles, aligned with ``deployments``.

    Returns:
        ``order`` with every house OpenRouter rung moved to the end.
    """
    aggregator = frozenset(
        index
        for index, (deployment, profile) in enumerate(zip(deployments, profiles, strict=True))
        if deployment.billing_source == BillingSource.HOST_MANAGED
        and (
            deployment.provider == "openrouter"
            # A Responses-wire rung dials OpenRouter through an ``openai``
            # connection on OpenRouter's own origin.
            or (urlsplit(profile.url).hostname or "").lower() == "openrouter.ai"
        )
    )
    return (
        *(index for index in order if index not in aggregator),
        *(index for index in order if index in aggregator),
    )
