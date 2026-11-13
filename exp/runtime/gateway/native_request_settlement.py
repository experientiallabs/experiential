"""Native retention of failed initial admissions, including paid prework."""

from __future__ import annotations

from exp.runtime.gateway.contracts import AuthorizationSnapshot, GatewayFailure
from exp.runtime.gateway.native_accounting import NativeAttemptAccounting
from exp.runtime.gateway.native_handoff import NativeRequestFactory


class RequestSettlements:
    """Lifecycle view of the sole Rust financial owner for long-running job admission."""

    def __init__(self, factory: NativeRequestFactory) -> None:
        """Bind interpreter-independent financial authority and native recovery."""
        self._accounting = NativeAttemptAccounting(native_request_factory=factory)

    @property
    def pending(self) -> bool:
        """Report retained failures independently of ordinary writes still in progress."""
        return not self._accounting.accounting_healthy

    @property
    def outstanding(self) -> bool:
        """Keep host transports alive until every accepted native write has completed."""
        return not self._accounting.native.drained()

    def fenced_organizations(self) -> frozenset[str]:
        """Read the native owner's exact set of unresolved financial authorities."""
        return self._accounting.fenced_organizations()

    def fenced(self, organization_id: str) -> bool:
        """Read one authority fence without scanning Python request objects."""
        return not self._accounting.native.is_clear(organization_id)

    def finish(
        self,
        authorization: AuthorizationSnapshot,
        failure: GatewayFailure,
        *,
        certify_no_effects: bool = False,
        web_search_requests: int = 0,
    ) -> bool:
        """Transfer failed admission to Rust before attempting its durable terminal write."""
        return self._accounting.finish_request_quietly(
            authorization,
            failure,
            certify_no_effects=certify_no_effects,
            web_search_requests=web_search_requests,
        )

    def retry(self) -> None:
        """Visit a bounded native recovery batch without reconstructing terminal evidence."""
        self._accounting.sweep_expired()

    def close(self, timeout_seconds: float = 10) -> bool:
        """Drain the native financial owner while retaining unacknowledged work."""
        return self._accounting.close(timeout_seconds)
