"""Count and retain physical attempts at the existing provider transport seam."""

import json
import math
from collections.abc import Callable, Mapping
from copy import deepcopy
from datetime import UTC, datetime

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, JsonObject, sha256_json
from exp.common.project import ArtifactStore, ArtifactStoreError, artifact_input
from exp.common.project.request_budget import validate_provider_response
from exp.runtime.models.call_evidence import current_call_intent, current_consumer
from exp.runtime.models.providers.transport import JsonHttpResponse, JsonHttpTransport
from exp.runtime.sessions.dispatch_ledger import DispatchLedger


class AdmittedJsonTransport(JsonHttpTransport):
    """Retain decoded HTTP responses before model parsing, without another retry owner."""

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
    ) -> None:
        """Bind physical capacity, immutable receipts and an aware global deadline.

        ``before_claim`` runs before every slot claim and refuses by raising, so a caller can
        recheck that it is still allowed to dispatch; nothing is claimed or sent after a refusal.
        """
        if deadline.tzinfo is None:
            raise ValueError("transport deadline must be timezone-aware")
        self._inner, self._ledger, self._deadline = inner, ledger, deadline
        self._before_claim = before_claim
        self._artifacts, self._revision = artifacts, code_revision
        self._clock = clock or _utc_now

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
    ) -> JsonHttpResponse:
        """Admit once, record intent, send once, then retain the returned body or failure."""
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
        )


def _utc_now() -> datetime:
    """Return the aware UTC clock for physical request authorization."""
    return datetime.now(UTC)
