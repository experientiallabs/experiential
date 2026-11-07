"""Serving callback facade for compositions using synchronous transaction owners."""

from __future__ import annotations

from exp_gateway_native import NativeCallbacks

from exp.runtime.gateway.native_accounting import NativeAttemptAccounting
from exp.runtime.gateway.native_handoff import accounting_callbacks


class NativeTransactionsMixin:
    """Expose the composition's explicitly selected accounting callback targets."""

    _accounting: NativeAttemptAccounting

    @property
    def native_callbacks(self) -> NativeCallbacks | None:
        """Retain this control plane's native request registry in its serving bridge."""
        return accounting_callbacks(self._accounting)

    def start_attempt(self, argument: str) -> str:
        """Reserve one physical dispatch through the synchronous ledger contract."""
        return self._accounting.start_attempt(argument)

    def settle(self, argument: str) -> str:
        """Durably settle an attempt selected for the synchronous ledger contract."""
        return self._accounting.settle(argument)

    def price(self, argument: str) -> str:
        """Price one decided terminal before its response bytes flush."""
        return self._accounting.price(argument)

    def abandon(self, argument: str) -> str:
        """Terminalize an accepted request selected for the synchronous ledger."""
        return self._accounting.abandon(argument)
