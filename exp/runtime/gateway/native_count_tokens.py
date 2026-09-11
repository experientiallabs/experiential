"""The ``count_tokens`` callback for the native bridge.

:class:`NativeCountTokensMixin` answers ``POST /v1/messages/count_tokens``
for :class:`~exp.runtime.gateway.native_bridge.NativeControlPlane`: the
decoded Messages request is counted with the gateway's own reservation
tokenizer (the estimator the credit reservation uses, without its headroom)
and answered in Anthropic's ``{"input_tokens": N}`` shape. No request is
accepted, no attempt is reserved, nothing is charged: the count is a read
against the same grants ``/v1/messages`` admits under.

The gateway has no tokenizer authority for any rung (the Anthropic provider
client does not forward ``count_tokens``), so every answer is an estimate and
says so through the shared ``x-experiential-ignored-parameters`` disclosure
the Messages surface already carries on its message bodies.
"""

from __future__ import annotations

import json

from exp.runtime.gateway.attempt_tokens import counted_input_tokens
from exp.runtime.gateway.discovery import require_granted_authority
from exp.runtime.gateway.native_accounting import NativeBridgeError
from exp.runtime.gateway.native_accounting import (
    authority_error as _authority_error,
)
from exp.runtime.gateway.native_components import NativeGatewayComponents
from exp.runtime.gateway.native_decode import NativeDecodeError, decode_native_body
from exp.runtime.gateway.native_settlement import optional_text

COUNT_TOKENS_ESTIMATE_DISCLOSURE = "input_tokens->estimated(gateway_tokenizer)"
"""Body disclosure telling the caller the count is the gateway's estimate."""


class NativeCountTokensMixin:
    """Token counting for the Messages surface, without a ledger row."""

    _components: NativeGatewayComponents

    def count_tokens(self, argument: str) -> str:
        """Count one Messages request's input tokens for an authenticated key.

        Args:
            argument: JSON object with ``raw_key``, ``body`` (raw request body
                text), optional ``surface`` (defaulting to ``"messages"``),
                and the optional ``anthropic_beta`` header list, decoded with
                the same shared decoder ``/v1/messages`` admits through.

        Returns:
            JSON object with ``input_tokens`` (the counted prompt before any
            reservation headroom) and the ``x-experiential-ignored-parameters``
            disclosure naming the count as the gateway's estimate.

        Raises:
            NativeBridgeError: The body failed protocol validation (400), or
                the alias is unknown or not granted to this key (the shared
                no-oracle 404).
        """
        data = json.loads(argument)
        try:
            decoded = decode_native_body(
                data["body"],
                surface=str(data.get("surface", "messages")),
                anthropic_beta=optional_text(data.get("anthropic_beta")),
            )
        except NativeDecodeError as exc:
            raise NativeBridgeError(exc.error) from exc
        try:
            authorities = self._components.store.granted_alias_authorities(raw_key=data["raw_key"])
            require_granted_authority(authorities, decoded.alias)
        except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
            raise _authority_error(exc) from exc
        disclosures = dict.fromkeys(
            (*decoded.request.ignored_parameters, COUNT_TOKENS_ESTIMATE_DISCLOSURE)
        )
        body = {
            "input_tokens": counted_input_tokens(decoded.request),
            "x-experiential-ignored-parameters": list(disclosures),
        }
        return json.dumps(body, separators=(",", ":"))
