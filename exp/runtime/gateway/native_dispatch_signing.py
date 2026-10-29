"""Dispatch-body signing callback for the native data plane."""

from __future__ import annotations

import json
from typing import Protocol

from exp.common.core.artifacts import sha256_bytes
from exp.runtime.gateway.contracts import GatewayFailure, GatewayFailureClass
from exp.runtime.gateway.native_accounting import NativeAttemptAccounting, NativeBridgeError
from exp.runtime.gateway.native_dispatch import dispatch_signature_headers
from exp.runtime.openai_protocol.errors import OpenAIProtocolError, public_failure_error


class _Plane(Protocol):
    _accounting: NativeAttemptAccounting


class NativeDispatchSigningMixin:
    """Sign one frozen dispatch body immediately before the provider POST."""

    def sign_dispatch(self: _Plane, argument: str) -> str:
        """Sign one frozen dispatch body immediately before the provider POST.

        The data plane calls this after it acquires its bounded dispatch
        permit and immediately before the open attempt reserved by
        ``start_attempt``, so queue time can never age a signature toward
        AWS's short clock window; a same-deployment redial or a failover
        advance is a fresh physical attempt through ``start_attempt``, so it
        always signs afresh too.

        Args:
            argument: JSON object with ``request_id``, the exact ``url``, and
                the exact frozen ``body`` string the data plane will send.

        Returns:
            JSON object with the ``headers`` to send verbatim.

        Raises:
            NativeBridgeError: The attempt is unknown, its route depth
                carries no signer, or credential resolution failed.
        """
        data = json.loads(argument)
        entry = self._accounting.entry(str(data.get("request_id") or ""))
        signer = None
        binding = None
        if entry is not None and entry.active_attempt_id is not None:
            depth = entry.attempt_depths.get(entry.active_attempt_id)
            if depth is not None and depth < len(entry.signers):
                signer = entry.signers[depth]
            if depth is not None and depth < len(entry.dispatch_bindings):
                binding = entry.dispatch_bindings[depth]
        try:
            url = str(data["url"])
            body = str(data["body"])
            if (
                binding is None
                or url != binding.url
                or sha256_bytes(body.encode("utf-8")) != binding.body_sha256
            ):
                raise public_failure_error(
                    GatewayFailure(
                        failure_class=GatewayFailureClass.INTERNAL,
                        safe_message=(
                            "gateway dispatch differs from the admitted destination or frozen body"
                        ),
                    )
                )
            headers = dispatch_signature_headers(
                signer,
                url=url,
                body=body,
            )
        except OpenAIProtocolError as exc:
            raise NativeBridgeError(exc) from exc
        return json.dumps({"headers": headers}, separators=(",", ":"))
