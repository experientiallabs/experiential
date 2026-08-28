"""Run one ordered input or output guardrail chain under request deadlines."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence

from exp.runtime.gateway.contracts import GatewayMessage, GatewayRequest
from exp.runtime.gateway.guardrails.bounded import BoundedInspect, ClassifierTimeoutError
from exp.runtime.gateway.guardrails.client import InternalClassifierClient
from exp.runtime.gateway.guardrails.contracts import (
    ClassifierVerdict,
    GuardrailAction,
    GuardrailCheck,
    GuardrailCompletion,
    GuardrailPolicy,
    GuardrailRejected,
    guardrail_failure,
    request_content_bytes,
)
from exp.runtime.gateway.guardrails.store import GuardrailPolicyStore

_logger = logging.getLogger(__name__)


def _restored_provider_authority(
    original: Sequence[GatewayMessage],
    replacement: Sequence[GatewayMessage],
) -> tuple[GatewayMessage, ...] | None:
    """Validate visible edits and restore hidden provider replay authority.

    Hosted classifiers receive only the normal serialized message projection,
    because replay-only reasoning, raw arguments, provider identity, status,
    and phase are excluded from that contract. A valid replacement must keep
    the classifier-visible authenticated prefix exact. The gateway then uses
    the original prefix objects, reattaching every hidden field without asking
    the classifier to receive or echo it.
    """

    def has_authority(message: GatewayMessage) -> bool:
        """Identify fields that must replay byte-exact on a provider continuation."""
        return bool(
            message.provider_reasoning
            or message.provider_item_id is not None
            or message.provider_output_index is not None
            or message.provider_status is not None
            or message.provider_phase is not None
            or message.tool_is_error
            or any(
                call.raw_arguments is not None
                or call.provider_item_id is not None
                or call.provider_output_index is not None
                or call.provider_status is not None
                for call in message.tool_calls
            )
        )

    original_carrier_indexes = tuple(
        index for index, message in enumerate(original) if has_authority(message)
    )
    if not original_carrier_indexes:
        return (
            None if any(has_authority(message) for message in replacement) else tuple(replacement)
        )
    bound = original_carrier_indexes[-1]
    if len(replacement) <= bound:
        return None
    original_visible = tuple(message.model_dump(mode="json") for message in original[: bound + 1])
    replacement_visible = tuple(
        message.model_dump(mode="json") for message in replacement[: bound + 1]
    )
    if replacement_visible != original_visible:
        return None
    if any(has_authority(message) for message in replacement[bound + 1 :]):
        return None
    return (*original[: bound + 1], *replacement[bound + 1 :])


class GuardrailEngine:
    """Look up identity policies and run classifier chains once per stage.

    The engine never logs request text, completions, detector payloads, or
    replacements. Decision metadata is limited to identity, policy, check,
    capability, action, and latency.
    """

    def __init__(
        self,
        *,
        store: GuardrailPolicyStore,
        client: InternalClassifierClient,
        monotonic: Callable[[], float],
        inspects: BoundedInspect | None = None,
    ) -> None:
        """Bind lookup, the internal client, and the deadline clock.

        Args:
            store: Identity-keyed policy lookup.
            client: Injected adapter seam that cannot use the public route.
            monotonic: Process-local clock in seconds.
            inspects: Optional async inflight limiter. ``None`` uses the
                default shared cap.
        """
        self._store = store
        self._client = client
        self._monotonic = monotonic
        self._inspects = inspects or BoundedInspect()
        self.input_invocations = 0
        self.output_invocations = 0
        self.classifier_calls = 0

    def policy_for(self, organization_id: str, identity_id: str) -> GuardrailPolicy | None:
        """Return the assigned policy, or ``None`` for unguarded traffic."""
        return self._store.policy_for(organization_id, identity_id)

    async def enforce_input(
        self,
        *,
        policy: GuardrailPolicy,
        request: GatewayRequest,
        deadline_monotonic: float,
    ) -> GatewayRequest:
        """Run the input chain once and return the validated or transformed request.

        Args:
            policy: Assigned identity policy.
            request: Canonical request after continuation expansion.
            deadline_monotonic: Remaining request-wide deadline.

        Returns:
            The original request, or the last successful modification.

        Raises:
            GuardrailRejected: A check blocked, errored, or fail-closed.
        """
        self.input_invocations += 1
        if request_content_bytes(request) > policy.max_request_bytes:
            self._record(policy, None, GuardrailAction.ERROR, 0.0)
            raise GuardrailRejected(guardrail_failure(action=GuardrailAction.ERROR))
        current = request
        for check in policy.input_checks:
            verdict = await self._run_check(
                policy=policy,
                check=check,
                inspect=lambda bound=check, payload=current: self._client.inspect_input(
                    request=payload,
                    check=bound,
                ),
                deadline_monotonic=deadline_monotonic,
            )
            if verdict is None:
                continue
            current = self._apply_input(policy, check, current, verdict)
        return current

    async def enforce_output(
        self,
        *,
        policy: GuardrailPolicy,
        completion: GuardrailCompletion,
        deadline_monotonic: float,
    ) -> GuardrailCompletion:
        """Run the output chain once on the winning normalized completion.

        Args:
            policy: Assigned identity policy.
            completion: Buffered winning text, refusal, and tool calls.
            deadline_monotonic: Remaining request-wide deadline.

        Returns:
            The original completion, or a text-only modification.

        Raises:
            GuardrailRejected: A check blocked, errored, or fail-closed.
        """
        self.output_invocations += 1
        if completion.content_bytes() > policy.max_response_bytes:
            self._record(policy, None, GuardrailAction.ERROR, 0.0)
            raise GuardrailRejected(guardrail_failure(action=GuardrailAction.ERROR))
        current = completion
        for check in policy.output_checks:
            verdict = await self._run_check(
                policy=policy,
                check=check,
                inspect=lambda bound=check, payload=current: self._client.inspect_output(
                    completion=payload,
                    check=bound,
                ),
                deadline_monotonic=deadline_monotonic,
            )
            if verdict is None:
                continue
            current = self._apply_output(policy, check, current, verdict)
        return current

    async def _run_check(
        self,
        *,
        policy: GuardrailPolicy,
        check: GuardrailCheck,
        inspect: Callable[[], Awaitable[ClassifierVerdict]],
        deadline_monotonic: float,
    ) -> ClassifierVerdict | None:
        """Invoke one adapter under the tighter of check timeout and request deadline.

        The inspect itself runs on an isolation worker. This caller only waits
        until the remaining budget elapses.

        Returns:
            The verdict, or ``None`` when a non-protected check is skipped.

        Raises:
            GuardrailRejected: Protected identities fail closed. Error actions
                and expired deadlines are always terminal.
        """
        remaining = deadline_monotonic - self._monotonic()
        timeout = min(check.timeout_ms / 1000.0, remaining)
        if timeout <= 0:
            return self._uncertain(policy, check, GuardrailAction.ERROR)
        started = self._monotonic()
        try:
            self.classifier_calls += 1
            verdict = await self._inspects.run(
                inspect,
                timeout,
                adapter_id=check.adapter_id,
            )
        except ClassifierTimeoutError:
            return self._uncertain(policy, check, GuardrailAction.ERROR)
        except Exception:  # noqa: BLE001 - classifier failures are fail-closed or skipped
            return self._uncertain(policy, check, GuardrailAction.ERROR)
        elapsed = self._monotonic() - started
        if not verdict.flagged:
            self._record(policy, check, GuardrailAction.ALLOW, elapsed)
            return None
        self._record(policy, check, check.action, elapsed)
        return verdict

    def _uncertain(
        self,
        policy: GuardrailPolicy,
        check: GuardrailCheck,
        action: GuardrailAction,
    ) -> ClassifierVerdict | None:
        """Apply fail-closed or skip-and-continue for an uncertain check."""
        self._record(policy, check, action, 0.0)
        if policy.protected:
            raise GuardrailRejected(
                guardrail_failure(action=GuardrailAction.ERROR, check_id=check.check_id)
            )
        return None

    def _apply_input(
        self,
        policy: GuardrailPolicy,
        check: GuardrailCheck,
        request: GatewayRequest,
        verdict: ClassifierVerdict,
    ) -> GatewayRequest:
        """Apply one flagged input action."""
        del policy
        if check.action is GuardrailAction.ALLOW:
            return request
        if check.action is GuardrailAction.MODIFY:
            if verdict.replacement_messages is None:
                raise GuardrailRejected(
                    guardrail_failure(action=GuardrailAction.ERROR, check_id=check.check_id)
                )
            restored = _restored_provider_authority(
                request.messages,
                verdict.replacement_messages,
            )
            if restored is None:
                raise GuardrailRejected(
                    guardrail_failure(action=GuardrailAction.ERROR, check_id=check.check_id)
                )
            return request.model_copy(update={"messages": restored})
        raise GuardrailRejected(guardrail_failure(action=check.action, check_id=check.check_id))

    def _apply_output(
        self,
        policy: GuardrailPolicy,
        check: GuardrailCheck,
        completion: GuardrailCompletion,
        verdict: ClassifierVerdict,
    ) -> GuardrailCompletion:
        """Apply one flagged output action. Tool-call arguments are never rewritten."""
        del policy
        if check.action is GuardrailAction.ALLOW:
            return completion
        if check.action is GuardrailAction.MODIFY:
            if completion.tool_calls:
                raise GuardrailRejected(
                    guardrail_failure(action=GuardrailAction.BLOCK, check_id=check.check_id)
                )
            if verdict.replacement_text is None:
                raise GuardrailRejected(
                    guardrail_failure(action=GuardrailAction.ERROR, check_id=check.check_id)
                )
            return completion.model_copy(update={"text": verdict.replacement_text})
        raise GuardrailRejected(guardrail_failure(action=check.action, check_id=check.check_id))

    def _record(
        self,
        policy: GuardrailPolicy,
        check: GuardrailCheck | None,
        action: GuardrailAction,
        latency_seconds: float,
    ) -> None:
        """Emit content-free decision metadata."""
        _logger.info(
            "guardrail decision policy_id=%s organization_id=%s identity_id=%s "
            "check_id=%s capability=%s action=%s latency_ms=%.1f",
            policy.policy_id,
            policy.organization_id,
            policy.identity_id,
            None if check is None else check.check_id,
            None if check is None else check.capability.value,
            action.value,
            latency_seconds * 1000.0,
        )
