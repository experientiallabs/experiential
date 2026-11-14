"""Shared admitted native-decision access for evaluation orchestration."""

import random
import time
from collections.abc import Callable
from datetime import datetime

from exp.common.core.artifacts import ArtifactEnvelope, JsonObject, stable_id
from exp.common.project import ProjectStore
from exp.common.project.errors import ArtifactAlreadyExistsError
from exp.runtime.gateway.decisions_contracts import DecisionRequest
from exp.runtime.models.deadlines import dispatch_deadline
from exp.runtime.models.providers.admitted_transport import AdmittedJsonTransport
from exp.runtime.models.providers.transport import ProviderTransportError
from exp.runtime.models.registry import ResolvedModel

RETRIED_STATUS_CODES = frozenset({429, 502, 503, 504})
"""Gateway statuses a decision retries: rate limited (never processed) or a bad hop."""

MAXIMUM_DECISION_RETRIES = 8
"""Most retries of one decision after its first attempt, all inside its deadline."""

INITIAL_BACKOFF_SECONDS = 1.0
MAXIMUM_BACKOFF_SECONDS = 30.0

_sleep: Callable[[float], None] = time.sleep
_sample: Callable[[], float] = random.random


def refusal_reason(body: object) -> str:
    """Return the provider's stated error type, code and message, bounded, never the request."""
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        return "no error object in the response"
    parts = [
        f"{key}={str(error[key])[:300]}"
        for key in ("type", "code", "message")
        if isinstance(error.get(key), str)
    ]
    return "; ".join(parts) or "an error object with no type, code or message"


def decision_caller(
    resolved: ResolvedModel,
    transport: AdmittedJsonTransport,
    deadline: datetime,
    clock: Callable[[], datetime],
) -> Callable[[DecisionRequest], JsonObject]:
    """Send JEV decisions on the catalog's native wire through the admitted transport."""
    profile_of = getattr(resolved.client, "gateway_wire_profile", None)
    if profile_of is None:
        raise ValueError("JEV alias does not resolve to a native decision client")
    profile = profile_of()
    if profile.dialect != "typesafe_systemone" or not profile.decisions_url:
        raise ValueError("JEV alias does not expose the native decision wire")
    transport = transport.priced_as(resolved.alias)
    deadline = dispatch_deadline(deadline)

    def decide(request: DecisionRequest) -> JsonObject:
        """Dispatch one decision; physical attempts share the run's ledger and deadline.

        A rate-limited (429) or bad-hop (502, 503, 504) answer is retried with full-jitter
        exponential backoff, never sooner than the server's Retry-After, while the wait fits
        the deadline; a 429 was rejected before any processing, so its retry is safe. Under
        sustained gateway concurrency one 429 must not abort a phase of thousands of calls.
        """
        payload = request.provider_body(profile.model_id)
        backoff = INITIAL_BACKOFF_SECONDS
        retries = 0
        while True:
            remaining = (deadline - clock()).total_seconds()
            response = transport.post(
                profile.decisions_url,
                headers=profile.headers,
                payload=payload,
                timeout_seconds=max(min(remaining, 120.0), 0.001),
            )
            if response.status_code == 200 and isinstance(response.body, dict):
                return response.body
            failure = ProviderTransportError(
                f"memory-recall decision returned HTTP {response.status_code}: "
                f"{refusal_reason(response.body)}",
                status_code=response.status_code,
                retry_after_seconds=response.retry_after_seconds,
            )
            if response.status_code not in RETRIED_STATUS_CODES or (
                retries >= MAXIMUM_DECISION_RETRIES
            ):
                raise failure
            floor = response.retry_after_seconds or 0.0
            wait = max(floor, backoff * _sample())
            if wait >= (deadline - clock()).total_seconds():
                raise failure
            _sleep(wait)
            retries += 1
            backoff = min(backoff * 2, MAXIMUM_BACKOFF_SECONDS)

    return decide


def receipt_recorder(
    project: ProjectStore, run_id: str, code_revision: str, clock: Callable[[], datetime]
) -> Callable[[JsonObject], None]:
    """Retain exact JEV requests, raw answers and selected IDs as immutable receipts."""

    def record(receipt: JsonObject) -> None:
        """Persist one content-addressed receipt beside the run's other artifacts."""
        identity = stable_id("jev-receipt", {"run": run_id, "receipt": receipt})
        if project.artifacts.exists(identity):
            return
        try:
            project.artifacts.write_json(
                artifact_id=identity,
                artifact_type="jev-receipt",
                envelope=ArtifactEnvelope(
                    schema_version=1, created_at=clock(), code_revision=code_revision
                ),
                files={"receipt.json": {"run": run_id, **receipt}},
            )
        except ArtifactAlreadyExistsError:
            # Parallel branches asking the same decision record the same content-addressed
            # receipt; whichever wrote first holds it, and the receipt is the same.
            return

    return record
