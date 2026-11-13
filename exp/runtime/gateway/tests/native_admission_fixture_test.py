"""Native policy state for tests that stop before financial admission."""

from dataclasses import dataclass, field

from exp_gateway_native import NativeCallbacks

from exp.runtime.gateway.contracts import AuthorizationSnapshot
from exp.runtime.gateway.native_accounting import NativeAttemptAccounting
from exp.runtime.gateway.native_execution import InflightRequest
from exp.runtime.gateway.recovery import RecoveryHost


class AdmissionOnlyFactory:
    """Reject accidental financial dispatch in an initial route-selection test."""

    def __call__(self, entry: InflightRequest, /) -> NativeCallbacks:
        """Require serving tests to supply a real native ledger."""
        raise AssertionError(f"Unexpected serving admission: {entry.authorization.request_id}")

    def admission(self, authorization: AuthorizationSnapshot, /) -> NativeCallbacks:
        """Require cleanup tests to supply a real native ledger."""
        raise AssertionError(f"Unexpected terminal admission: {authorization.request_id}")


def admission_accounting(
    *, recovery_host: RecoveryHost | None = None, default_lane_bound: int | None = None
) -> NativeAttemptAccounting:
    """Expose actual native policy views without inventing a financial callback."""
    return NativeAttemptAccounting(
        native_request_factory=AdmissionOnlyFactory(),
        recovery_host=recovery_host,
        default_lane_bound=default_lane_bound,
        background_recovery=False,
    )


@dataclass
class AdmissionRecords:
    """Observe handoff inputs without implementing or replacing any native callback."""

    entries: dict[NativeAttemptAccounting, dict[str, InflightRequest]] = field(default_factory=dict)

    def remember(self, accounting: NativeAttemptAccounting, entry: InflightRequest) -> None:
        """Keep the resolved input after the actual native registration succeeds."""
        self.entries.setdefault(accounting, {})[str(entry.authorization.request_id)] = entry

    def get(self, accounting: NativeAttemptAccounting, request_id: str) -> InflightRequest:
        """Return only the initial handoff, never a live financial or lifecycle view."""
        return self.entries[accounting][request_id]
