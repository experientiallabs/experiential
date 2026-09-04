"""Bounded single-flight publication for retained router selections."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from exp.common.core.artifacts import Sha256
from exp.common.models import ModelRequest, OperationEconomics
from exp.common.routing import RoutingDecision
from exp.runtime.router.economics import RoutedSpendDisposition, zero_operation_economics
from exp.runtime.router.runtime_support import sticky_decision

if TYPE_CHECKING:
    from exp.runtime.router.runtime import RouterRuntime


@dataclass(frozen=True)
class PreparedSelection:
    """One decision and the exact physical evidence incurred while selecting it."""

    decision: RoutingDecision
    request_key: tuple[Sha256, Sha256]
    economics: OperationEconomics
    disposition: RoutedSpendDisposition


def retained_selection(
    runtime: RouterRuntime,
    request: ModelRequest,
    *,
    episode_id: str,
) -> PreparedSelection:
    """Return one retained decision with detached exact embedding evidence.

    Args:
        runtime: Active router selector whose bounded state owns publication.
        request: Provider-neutral request visible to selection.
        episode_id: Exact non-empty sticky identity for this selection.

    Returns:
        Retained decision plus the physical evidence incurred by its selection.

    Raises:
        RouterRuntimeIntegrityError: Selection admission capacity is exhausted.
    """
    identity_sha256 = hashlib.sha256(episode_id.encode("utf-8")).hexdigest()
    feature = runtime._extractor.from_request(request)  # noqa: SLF001
    request_sha256 = hashlib.sha256(feature.encode("utf-8")).hexdigest()
    request_key = (identity_sha256, request_sha256)
    while True:
        with runtime._episode_lock:  # noqa: SLF001
            runtime._expire_decisions()  # noqa: SLF001
            existing = runtime._request_decisions.get(request_key)  # noqa: SLF001
            if existing is not None:
                runtime._request_decisions.move_to_end(request_key)  # noqa: SLF001
                return PreparedSelection(
                    decision=existing,
                    request_key=request_key,
                    economics=runtime._request_embedding_economics[request_key],  # noqa: SLF001
                    disposition=runtime._request_embedding_dispositions[request_key],  # noqa: SLF001
                )
            episode_decision = runtime._episode_decisions.get(identity_sha256)  # noqa: SLF001
            if episode_decision is not None:
                decision = sticky_decision(episode_decision, request_sha256)
                economics = zero_operation_economics()
                disposition = RoutedSpendDisposition.DEFINITELY_NOT_INCURRED
                published = runtime._publish_decision(  # noqa: SLF001
                    request=request,
                    decision=decision,
                    request_key=request_key,
                    identity=episode_id,
                    embedding_economics=economics,
                    embedding_disposition=disposition,
                )
                return PreparedSelection(
                    decision=published,
                    request_key=request_key,
                    economics=economics,
                    disposition=disposition,
                )
            waiter = runtime._selection_inflight.get(request_key)  # noqa: SLF001
            if waiter is None:
                if len(runtime._selection_inflight) >= runtime._decision_capacity:  # noqa: SLF001
                    from exp.runtime.router.runtime import RouterRuntimeIntegrityError

                    raise RouterRuntimeIntegrityError(
                        "router selection admission capacity is exhausted"
                    )
                waiter = threading.Event()
                runtime._selection_inflight[request_key] = waiter  # noqa: SLF001
                break
        waiter.wait()
    try:
        prepared = runtime._select_unretained(request, episode_id=episode_id)  # noqa: SLF001
        proposed = prepared.decision
        with runtime._episode_lock:  # noqa: SLF001
            runtime._expire_decisions()  # noqa: SLF001
            existing = runtime._request_decisions.get(request_key)  # noqa: SLF001
            if existing is not None:
                return PreparedSelection(
                    decision=existing,
                    request_key=request_key,
                    economics=runtime._request_embedding_economics[request_key],  # noqa: SLF001
                    disposition=runtime._request_embedding_dispositions[request_key],  # noqa: SLF001
                )
            episode_decision = runtime._episode_decisions.get(identity_sha256)  # noqa: SLF001
            decision = runtime._episode_selection(  # noqa: SLF001
                episode_decision,
                proposed,
                request_sha256=request_sha256,
                feature=feature,
            )
            published = runtime._publish_decision(  # noqa: SLF001
                request=request,
                decision=decision,
                request_key=request_key,
                identity=episode_id,
                embedding_economics=prepared.economics,
                embedding_disposition=prepared.disposition,
            )
            return PreparedSelection(
                decision=published,
                request_key=request_key,
                economics=prepared.economics,
                disposition=prepared.disposition,
            )
    finally:
        with runtime._episode_lock:  # noqa: SLF001
            completed = runtime._selection_inflight.pop(request_key, None)  # noqa: SLF001
            if completed is not None:
                completed.set()
