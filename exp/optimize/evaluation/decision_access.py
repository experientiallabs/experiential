"""Shared admitted native-decision access for evaluation orchestration."""

from collections.abc import Callable
from datetime import datetime

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.decisions_contracts import DecisionRequest
from exp.runtime.models.providers.admitted_transport import AdmittedJsonTransport
from exp.runtime.models.providers.transport import ProviderTransportError
from exp.runtime.models.registry import ResolvedModel


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

    def decide(request: DecisionRequest) -> JsonObject:
        """Dispatch one decision; physical attempts share the run's ledger and deadline."""
        remaining = (deadline - clock()).total_seconds()
        response = transport.post(
            profile.decisions_url,
            headers=profile.headers,
            payload=request.provider_body(profile.model_id),
            timeout_seconds=max(min(remaining, 120.0), 0.001),
        )
        if response.status_code != 200 or not isinstance(response.body, dict):
            raise ProviderTransportError(
                f"memory-recall decision returned HTTP {response.status_code}",
                status_code=response.status_code,
            )
        return response.body

    return decide
