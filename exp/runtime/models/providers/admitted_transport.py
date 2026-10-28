"""Count and retain physical attempts at the existing provider transport seam."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from copy import copy, deepcopy
from datetime import UTC, datetime
from urllib.parse import urlsplit

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    JsonObject,
    JsonValue,
    sha256_bytes,
    sha256_json,
)
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
from exp.runtime.models.spend_ceiling import (
    OperationSpendCeiling,
    Reservation,
    current_spend_ceiling,
    is_embedding_endpoint,
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

_EMBEDDED_TEXT_KEYS = frozenset({"input", "text"})
"""Embedding request fields that carry text: OpenAI-wire ``input`` and Gemini part ``text``."""


def _retained_request(url: str, payload: JsonObject | None) -> JsonObject | None:
    """Return the request body an intent receipt keeps for one physical request.

    Embedding inputs are chunks of the caller's own stored evidence, so the receipt cites each
    text by its digest and length instead of copying it. Agent traces routinely quote placeholder
    credentials such as ``Authorization: token ${GITHUB_TOKEN}``, which the credential boundary
    would otherwise refuse to retain. Every other field, and every non-embedding body, is kept
    whole and still passes the boundary unchanged.

    Args:
        url: Absolute provider endpoint, which distinguishes embedding requests.
        payload: Exact JSON request body, or None for a bodiless request.

    Returns:
        The body to retain; ``request_sha256`` still digests the exact sent body.
    """
    if payload is None or not is_embedding_endpoint(url):
        return payload
    return {name: _cite_texts(value, name) for name, value in payload.items()}


def _cite_texts(value: JsonValue, key: str | None) -> JsonValue:
    """Replace embedded text under :data:`_EMBEDDED_TEXT_KEYS` with its digest and length."""
    if isinstance(value, str) and key in _EMBEDDED_TEXT_KEYS:
        return {"text_sha256": sha256_bytes(value.encode("utf-8")), "characters": len(value)}
    if isinstance(value, list):
        return [_cite_texts(item, key) for item in value]
    if isinstance(value, dict):
        return {name: _cite_texts(item, name) for name, item in value.items()}
    return value


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

    Under an operation spend ceiling every POST reserves its worst-case cost, priced as its
    route's alias (or the alias named by :meth:`priced_as`), before the ledger claim and the
    intent receipt; the response settles it, an unknown outcome keeps it, and a request that
    provably never left releases it.
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
        spend_ceiling: OperationSpendCeiling | None = None,
    ) -> None:
        """Bind physical capacity, immutable receipts and an aware global deadline.

        ``before_claim`` runs before every slot claim and refuses by raising, so a caller can
        recheck that it is still allowed to dispatch; nothing is claimed or sent after a refusal.
        ``admission`` defaults to the host-wide database and registered limits. ``route`` names
        the provider and alias scopes; the registry binds it per client with :meth:`for_route`,
        and an unbound transport is scoped by its endpoint host. ``spend_ceiling`` defaults to
        the ceiling installed in the current operation scope when the transport is built.
        """
        if deadline.tzinfo is None:
            raise ValueError("transport deadline must be timezone-aware")
        self._inner, self._ledger, self._deadline = inner, ledger, deadline
        self._admission = admission or HostProviderAdmission.for_host()
        self._route = route
        self._before_claim = before_claim
        self._artifacts, self._revision = artifacts, code_revision
        self._clock = clock or _utc_now
        self._ceiling = spend_ceiling or current_spend_ceiling()
        self._priced_alias: str | None = None

    @property
    def spend_ceiling(self) -> OperationSpendCeiling | None:
        """Return the operation ceiling this transport reserves against, if any."""
        return self._ceiling

    def priced_as(self, alias: str) -> AdmittedJsonTransport:
        """Return a view whose requests are priced as ``alias`` without changing admission.

        A wire posted outside the registry's clients, such as the native decision wire, names
        the catalog alias whose prices bound it; pacing keeps the transport's own route.
        """
        view = copy(self)
        view._priced_alias = alias
        return view

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
        retained = _retained_request(url, payload)
        try:
            validate_provider_response(json.dumps(retained, ensure_ascii=False, allow_nan=False))
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
                method, url, payload, retained, timeout_seconds, send, slot, idempotency_key
            )

    def _dispatch(
        self,
        method: str,
        url: str,
        payload: JsonObject | None,
        retained: JsonObject | None,
        timeout_seconds: float,
        send: Callable[[float], JsonHttpResponse],
        slot: AdmissionSlot,
        idempotency_key: str | None,
    ) -> JsonHttpResponse:
        """Reserve spend, claim a ledger slot, record intent, send once and settle the outcome.

        A reservation is released only while the request provably has not left: before the
        dispatch mark commits, nothing has been sent. The intent keeps ``retained``, the
        receipt form of the body, beside the digest of the exact ``payload`` that is sent.
        """
        timeout_seconds -= slot.waited_seconds
        now = self._clock()
        remaining = (self._deadline - now).total_seconds()
        if remaining <= 0 or timeout_seconds <= 0:
            raise TimeoutError("transport deadline reached before physical dispatch")
        ceiling = self._ceiling
        reservation: Reservation | None = None
        if ceiling is not None and payload is not None:
            alias = self._priced_alias or (None if self._route is None else self._route.alias)
            reservation = ceiling.reserve(alias=alias, url=url, payload=payload)
        try:
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
                    "request_body": retained,
                    "consumer": current_consumer(),
                    "timeout_seconds": min(remaining, timeout_seconds),
                    "timeout_protocol": ATTEMPT_TIMEOUT_PROTOCOL,
                    **({"idempotency_key": idempotency_key} if idempotency_key else {}),
                    **(
                        {"canonical_request": canonical.model_dump(mode="json")}
                        if canonical
                        else {}
                    ),
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
            if reservation is not None and ceiling is not None:
                ceiling.mark_dispatched(reservation, attempt)
        except BaseException:
            if reservation is not None and ceiling is not None:
                ceiling.release(reservation)
            raise
        try:
            response = send(min(remaining, timeout_seconds))
        except BaseException as error:
            if reservation is not None and ceiling is not None:
                ceiling.retain_unknown(reservation)
            if not isinstance(error, Exception):
                raise
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
        if reservation is not None and ceiling is not None:
            ceiling.settle(reservation, response)
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
