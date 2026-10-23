"""Count and retain physical attempts at the existing provider transport seam."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from copy import copy, deepcopy
from datetime import UTC, datetime
from urllib.parse import urlsplit

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, JsonObject, sha256_json
from exp.common.models import ModelCatalog
from exp.common.project import ArtifactStore, ArtifactStoreError, artifact_input
from exp.common.project.request_budget import validate_provider_response
from exp.runtime.models.call_evidence import current_call_intent, current_consumer
from exp.runtime.models.providers.attempt_timeouts import ATTEMPT_TIMEOUT_PROTOCOL
from exp.runtime.models.providers.backpressure import (
    AdmissionSlot,
    HostProviderAdmission,
    ProviderRoute,
)
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    JsonHttpTransport,
    ProviderTransportError,
    public_error_code,
)
from exp.runtime.sessions.dispatch_ledger import DispatchLedger

ADMITTED_TRANSPORT_PROVIDERS = frozenset(
    {"anthropic", "azure", "gemini", "openai", "openai-compatible", "openrouter"}
)
"""Providers whose every completion and embedding request goes through the injected transport.

The registry builds these clients on its ``transport_factory``, so wrapping that factory in
:class:`AdmittedJsonTransport` admits, bounds and retains each physical request. The other
registry providers fall outside it: ``bedrock`` and ``tinker`` dispatch through their own SDK
runtimes, ``typesafe`` and ``elevenlabs`` serve no completion (the native gateway owns their
wire), and ``vertex`` mints its OAuth bearer token over Google's own HTTP client.
"""


def require_admitted_transport(catalog: ModelCatalog, alias: str, *, consumer: str) -> None:
    """Refuse a model alias whose provider sends requests outside the admitted transport.

    Args:
        catalog: Credential-free model catalog that declares ``alias``.
        alias: Catalog model alias the consumer will call.
        consumer: Short name of the calling role, used in the refusal message.

    Raises:
        ValueError: The alias's provider is not in :data:`ADMITTED_TRANSPORT_PROVIDERS`.
    """
    provider = catalog.connections[catalog.models[alias].connection].provider
    if provider not in ADMITTED_TRANSPORT_PROVIDERS:
        raise ValueError(
            f"{consumer} alias {alias!r} uses provider {provider!r}, which sends requests "
            "outside the admitted HTTP transport; point the alias at a connection whose "
            f"provider is one of {', '.join(sorted(ADMITTED_TRANSPORT_PROVIDERS))}"
        )


class AdmittedJsonTransport(JsonHttpTransport):
    """Admit, retain and pace physical HTTP requests, without another retry owner.

    Before every ledger claim the request passes the host-wide provider admission
    (:mod:`exp.runtime.models.providers.backpressure`): the provider's circuit breaker must
    admit it and an in-flight slot of its provider and alias must be free. The slot is held
    only while the request is in flight, and every definite status updates the breaker.
    """

    def __init__(
        self,
        inner: JsonHttpTransport,
        *,
        ledger: DispatchLedger,
        artifacts: ArtifactStore,
        code_revision: str,
        deadline: datetime,
        clock: Callable[[], datetime] | None = None,
        before_claim: Callable[[], None] | None = None,
        admission: HostProviderAdmission | None = None,
        route: ProviderRoute | None = None,
    ) -> None:
        """Bind physical capacity, immutable receipts and an aware global deadline.

        ``before_claim`` runs before every slot claim and refuses by raising, so a caller can
        recheck that it is still allowed to dispatch; nothing is claimed or sent after a refusal.
        ``admission`` defaults to the host-wide database and registered limits. ``route`` names
        the provider and alias scopes; the registry binds it per client with :meth:`for_route`,
        and an unbound transport is scoped by its endpoint host.
        """
        if deadline.tzinfo is None:
            raise ValueError("transport deadline must be timezone-aware")
        self._inner, self._ledger, self._deadline = inner, ledger, deadline
        self._admission = admission or HostProviderAdmission.for_host()
        self._route = route
        self._before_claim = before_claim
        self._artifacts, self._revision = artifacts, code_revision
        self._clock = clock or _utc_now

    def for_route(self, route: ProviderRoute) -> AdmittedJsonTransport:
        """Return a view sharing this ledger, deadline and admission, scoped to ``route``."""
        view = copy(self)
        view._route = route
        return view

    def _route_for(self, url: str) -> ProviderRoute:
        """Return the bound route, or an endpoint-host scope for an unbound transport."""
        if self._route is not None:
            return self._route
        return ProviderRoute(provider="endpoint:" + (urlsplit(url).hostname or "unknown"))

    def _record(
        self, identity: str, value: JsonObject, inputs: tuple[ArtifactInput, ...] = ()
    ) -> ArtifactInput:
        """Publish credential-free attempt evidence without request headers or URL secrets."""
        manifest = self._artifacts.write_json(
            artifact_id=identity,
            artifact_type="provider-physical-attempt",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=self._clock(),
                inputs=inputs,
                code_revision=self._revision,
            ),
            files={"receipt.json": value},
        )
        return artifact_input(manifest)

    def _request(
        self,
        method: str,
        url: str,
        payload: JsonObject | None,
        timeout_seconds: float,
        send: Callable[[float], JsonHttpResponse],
        idempotency_key: str | None = None,
    ) -> JsonHttpResponse:
        """Admit once, record intent, send once, then retain the returned body or failure.

        Raises:
            ProviderAdmissionRefused: The host refused before any claim or network I/O.
        """
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("physical request timeout must be finite and positive")
        try:
            validate_provider_response(json.dumps(payload, ensure_ascii=False, allow_nan=False))
        except ArtifactStoreError as error:
            raise ArtifactStoreError(
                "provider request cannot be retained at the credential boundary; "
                "remove credential-bearing content before dispatch"
            ) from error
        now = self._clock()
        remaining = (self._deadline - now).total_seconds()
        if remaining <= 0:
            raise TimeoutError("transport deadline reached before physical dispatch")
        if self._before_claim is not None:
            self._before_claim()
        route = self._route_for(url)
        with self._admission.admit(route, wait_seconds=min(remaining, timeout_seconds)) as slot:
            return self._dispatch(
                method, url, payload, timeout_seconds, send, slot, idempotency_key
            )

    def _dispatch(
        self,
        method: str,
        url: str,
        payload: JsonObject | None,
        timeout_seconds: float,
        send: Callable[[float], JsonHttpResponse],
        slot: AdmissionSlot,
        idempotency_key: str | None,
    ) -> JsonHttpResponse:
        """Claim a ledger slot, record intent, send once and report the outcome to the slot."""
        timeout_seconds -= slot.waited_seconds
        now = self._clock()
        remaining = (self._deadline - now).total_seconds()
        if remaining <= 0 or timeout_seconds <= 0:
            raise TimeoutError("transport deadline reached before physical dispatch")
        sequence, attempt = self._ledger.claim(now.isoformat())
        base = "provider-attempt-" + attempt
        canonical = current_call_intent()
        intent = self._record(
            base + "-intent",
            {
                "sequence": sequence,
                "method": method,
                "endpoint_sha256": sha256_json({"url": url}),
                "request_sha256": sha256_json(payload),
                "request_body": payload,
                "consumer": current_consumer(),
                "timeout_seconds": min(remaining, timeout_seconds),
                "timeout_protocol": ATTEMPT_TIMEOUT_PROTOCOL,
                **({"idempotency_key": idempotency_key} if idempotency_key else {}),
                **({"canonical_request": canonical.model_dump(mode="json")} if canonical else {}),
            },
            () if canonical is None else (canonical,),
        )
        remaining = (self._deadline - self._clock()).total_seconds()
        if remaining <= 0:
            self._record(
                base + "-not-dispatched",
                {
                    "sequence": sequence,
                    "outcome": "deadline_before_io",
                    "dispatch": "not_started",
                },
                (intent,),
            )
            raise TimeoutError("transport deadline reached while persisting intent")
        try:
            response = send(min(remaining, timeout_seconds))
        except Exception as error:  # noqa: BLE001 - preserve ambiguity without retrying.
            failure = error if isinstance(error, ProviderTransportError) else None
            slot.record(
                status_code=None if failure is None else failure.status_code,
                retry_after_seconds=None if failure is None else failure.retry_after_seconds,
                error_code=None if failure is None else failure.error_code,
            )
            self._record(
                base + "-failure",
                {
                    "sequence": sequence,
                    "error_type": type(error).__name__,
                    "outcome": "unknown",
                    "billing": "unknown",
                },
                (intent,),
            )
            raise
        slot.record(
            status_code=response.status_code,
            retry_after_seconds=response.retry_after_seconds,
            error_code=public_error_code(response.body),
        )
        encoded = json.dumps(response.body, ensure_ascii=False, allow_nan=False)
        try:
            validate_provider_response(encoded)
        except ArtifactStoreError:
            self._record(
                base + "-rejected",
                {
                    "sequence": sequence,
                    "http_status": response.status_code,
                    "outcome": "response_not_retained_credential_boundary",
                    "response_sha256": sha256_json(response.body),
                    "billing": "unknown",
                },
                (intent,),
            )
            raise
        self._record(
            base + "-response",
            {
                "sequence": sequence,
                "http_status": response.status_code,
                "response_body": response.body,
                "known_unbilled": response.known_unbilled,
                "returned_after_deadline": self._clock() >= self._deadline,
            },
            (intent,),
        )
        if self._clock() >= self._deadline:
            self._record(
                base + "-delivery-refused",
                {
                    "sequence": sequence,
                    "outcome": "saved_response_after_deadline",
                    "phase": "response_delivery",
                    "dispatch": "completed",
                },
                (intent,),
            )
            raise TimeoutError(
                "provider response retained after execution deadline; no action is authorized"
            )
        return response

    def get(
        self, url: str, *, headers: Mapping[str, str], timeout_seconds: float
    ) -> JsonHttpResponse:
        """Count a metadata request and retain its exact decoded response object."""
        return self._request(
            "GET",
            url,
            None,
            timeout_seconds,
            lambda timeout: self._inner.get(url, headers=headers, timeout_seconds=timeout),
        )

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Record each provider-owned retry separately before model-protocol parsing."""
        payload = deepcopy(payload)
        return self._request(
            "POST",
            url,
            payload,
            timeout_seconds,
            lambda timeout: self._inner.post(
                url,
                headers=headers,
                payload=payload,
                timeout_seconds=timeout,
            ),
            next(
                (value for name, value in headers.items() if name.lower() == "idempotency-key"),
                None,
            ),
        )


def _utc_now() -> datetime:
    """Return the aware UTC clock for physical request authorization."""
    return datetime.now(UTC)
