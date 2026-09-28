# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""Admission and route assembly for the native gateway control plane."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import time
from typing import TYPE_CHECKING

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.attempt_tokens import counted_input_tokens
from exp.runtime.gateway.contracts import (
    DirectTarget,
    GatewayApiSurface,
    GatewayFailure,
    GatewayFailureClass,
)
from exp.runtime.gateway.group_commit import SyncGroupCommitLedger
from exp.runtime.gateway.guardrails import deterministic
from exp.runtime.gateway.guardrails.client import assert_not_internal_classification
from exp.runtime.gateway.guardrails.contracts import GuardrailRejected
from exp.runtime.gateway.guardrails.native import (
    enforce_native_input,
    native_output_mode,
)
from exp.runtime.gateway.model_chain_authority import authorize_serving_model_chains
from exp.runtime.gateway.native_accounting import (
    NativeBridgeError,
)
from exp.runtime.gateway.native_accounting import (
    authority_error as _authority_error,
)
from exp.runtime.gateway.native_admission import (
    admitted_route_requests,
    fold_parallel_tool_call_disclosures,
    log_reasoning_continuation_rejection,
    record_dead_admission_rungs,
    select_single_route_before_search,
)
from exp.runtime.gateway.native_bridge_errors import (
    ledger_capability_message,
)
from exp.runtime.gateway.native_bridge_errors import (
    public_capability_error as _public_capability_error,
)
from exp.runtime.gateway.native_capture import (
    begin_capture,
    select_capture_model,
)
from exp.runtime.gateway.native_continuation import (
    continuation_binding_error as _continuation_binding_error,
)
from exp.runtime.gateway.native_continuation import (
    require_bound_wire_authority as _require_bound_wire_authority,
)
from exp.runtime.gateway.native_continuation import (
    select_bound_continuation_route as _select_bound_continuation_route,
)
from exp.runtime.gateway.native_execution import (
    FrozenDispatchBinding,
    InflightRequest,
    NativeDialectUnavailableError,
    dispatchable_route_profiles,
    select_route_deployments,
)
from exp.runtime.gateway.native_explicit_cache import (
    bind_explicit_cache,
)
from exp.runtime.gateway.native_reasoning import (
    authenticate_reasoning_history,
    has_active_reasoning_content,
    rung_provider_request,
    strip_stale_reasoning_history,
    unseal_reasoning_history,
)
from exp.runtime.gateway.native_request_policy import require_route_authority
from exp.runtime.gateway.native_responses import (
    ContinuationContext,
    continuation_route_binding,
    continued_request,
    responses_envelope,
)
from exp.runtime.gateway.native_rung_policy import throttle_redial_budgets
from exp.runtime.gateway.native_rungs import build_rung_dispatch
from exp.runtime.gateway.native_settlement import (
    gateway_updating_failure,
    optional_text,
)
from exp.runtime.gateway.reasoning_carrier import (
    ReasoningCarrierAuthority,
)
from exp.runtime.gateway.recovery_binding import validated_recovery_binding
from exp.runtime.gateway.request_policy import attempt_policy
from exp.runtime.gateway.routing import GatewayRoute, GatewayRoutingError
from exp.runtime.gateway.tool_search.plan import plan_tool_search
from exp.runtime.gateway.web_search.plan import plan_web_search
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.errors import (
    ProviderCapabilityError,
    ProviderParameterError,
    normalized_provider_failure,
)
from exp.runtime.models.providers.logprobs import require_unmodified_probability_output
from exp.runtime.models.providers.protocol import GatewayDispatchSigner, NativeWireClient
from exp.runtime.openai_protocol.errors import (
    OpenAIProtocolError,
    invalid_field,
    public_failure_error,
)

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from exp.runtime.gateway.native_bridge import NativeControlPlane


class NativeAdmissionMixin:
    """Implement request admission on the composed native control plane."""

    def admit(self: NativeControlPlane, argument: str) -> str:
        """Decode, authorize, inspect, route, and durably accept one request.

        Args:
            argument: JSON object with ``raw_key``, ``body`` (raw request
                body text), optional ``surface`` (``"chat"`` or
                ``"responses"``, defaulting to chat), and optional
                ``app_referer``/``app_title`` caller app identity.

        Returns:
            The ordered certified ``route`` with dispatch and retry configuration,
            or ``{"escalate": reason}`` after finalizing an accepted request that
            the native plane cannot serve without writing an attempt row.

        Raises:
            NativeBridgeError: Decoding, authorization, routing, or
                capability admission failed.
        """
        assert_not_internal_classification()
        sweep_started = time.monotonic()
        self._accounting.sweep_expired()
        self._control_plane_timing.record("expired_sweep_ms", sweep_started)
        decode_started = time.monotonic()
        data = json.loads(argument)
        preauthenticated_key = self._take_chat_admission_preflight(str(data.get("raw_key", "")))
        surface = str(data.get("surface", "chat"))
        decoded = self._decode_body(
            data["body"],
            surface=surface,
            idempotency_key=optional_text(data.get("idempotency_key")),
            client_request_id=optional_text(data.get("client_request_id")),
            anthropic_beta=optional_text(data.get("anthropic_beta")),
        )
        request = decoded.request
        deadline = time.monotonic() + self._request_timeout_seconds
        self._control_plane_timing.record("decode_and_body_ms", decode_started)
        authorization_started = time.monotonic()
        try:
            # Freeze native app attribution and the trusted client IP onto caller authority.
            sqlite_authority_started = time.monotonic()
            try:
                if preauthenticated_key is None:
                    authorization = self._components.store.authorize_request(
                        raw_key=data["raw_key"],
                        alias=decoded.alias,
                        request=request,
                        deadline_monotonic=deadline,
                        app_referer=optional_text(data.get("app_referer")),
                        app_title=optional_text(data.get("app_title")),
                        client_ip=optional_text(data.get("client_ip")),
                    )
                else:
                    authorization = self._components.store.authorize_request(
                        raw_key=data["raw_key"],
                        alias=decoded.alias,
                        request=request,
                        deadline_monotonic=deadline,
                        preauthenticated_key=preauthenticated_key,
                        app_referer=optional_text(data.get("app_referer")),
                        app_title=optional_text(data.get("app_title")),
                        client_ip=optional_text(data.get("client_ip")),
                    )
            finally:
                self._control_plane_timing.record(
                    "sqlite_request_authority_ms", sqlite_authority_started
                )
            chain_authority_started = time.monotonic()
            try:
                authorization = authorize_serving_model_chains(self._components, authorization)
            finally:
                self._control_plane_timing.record(
                    "serving_chain_authorization_ms", chain_authority_started
                )
        except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
            self._control_plane_timing.record("alias_authorization_ms", authorization_started)
            mapped = _authority_error(exc)
            pointer = self._batch_pointer_error(alias=decoded.alias, mapped=mapped)
            if pointer is not None:
                raise pointer from exc
            raise mapped from exc
        self._control_plane_timing.record("alias_authorization_ms", authorization_started)

        # Resolve continuation after authorization and before any durable acceptance.
        pre_accept_started = time.monotonic()
        continuation_context: ContinuationContext | None = None
        if request.surface == GatewayApiSurface.RESPONSES:
            try:
                request, continuation_context = continued_request(
                    self._continuations,
                    authorization=authorization,
                    request=request,
                )
            except OpenAIProtocolError as exc:
                raise NativeBridgeError(exc) from exc

        pinned_reasoning_route: GatewayRoute | None = None
        try:
            request, pinned_reasoning_route = authenticate_reasoning_history(
                self._components,
                authorization,
                request,
            )
        except ProviderParameterError as exc:
            raise NativeBridgeError(invalid_field(exc.param, str(exc))) from exc
        except Exception as exc:  # noqa: BLE001 - one public shape prevents an oracle.
            log_reasoning_continuation_rejection(authorization, "authenticate", exc)
            error = invalid_field(
                "messages.reasoning_content",
                "'messages.reasoning_content' must be an authentic continuation for this route.",
            )
            raise NativeBridgeError(error) from exc

        policy = None
        try:
            request, policy = enforce_native_input(
                self._guardrails,
                authorization=authorization,
                request=request,
                deadline_monotonic=deadline,
                detectors=self._guardrail_detectors,
            )
        except GuardrailRejected as exc:
            raise NativeBridgeError(public_failure_error(exc.failure)) from exc
        captured_request = request
        retention_request = strip_stale_reasoning_history(request)
        try:
            request, verified_reasoning_route = unseal_reasoning_history(
                self._components,
                authorization,
                request,
            )
        except ProviderParameterError as exc:
            raise NativeBridgeError(invalid_field(exc.param, str(exc))) from exc
        except Exception as exc:  # noqa: BLE001 - one public shape prevents an oracle.
            log_reasoning_continuation_rejection(authorization, "unseal", exc)
            error = invalid_field(
                "messages.reasoning_content",
                "'messages.reasoning_content' must be an authentic continuation for this route.",
            )
            raise NativeBridgeError(error) from exc
        if (
            pinned_reasoning_route is not None
            and verified_reasoning_route is not None
            and pinned_reasoning_route.deployment != verified_reasoning_route.deployment
        ):
            log_reasoning_continuation_rejection(
                authorization, "route_pin", "authenticate and unseal resolved different deployments"
            )
            raise NativeBridgeError(
                invalid_field(
                    "messages.reasoning_content",
                    "'messages.reasoning_content' must be an authentic continuation "
                    "for this route.",
                )
            )
        pinned_reasoning_route = verified_reasoning_route
        request = strip_stale_reasoning_history(request)
        if pinned_reasoning_route is not None and not has_active_reasoning_content(request):
            pinned_reasoning_route = None
        if continuation_context is not None:
            # Execution receives authenticated plaintext, but the bounded
            # continuation store keeps the post-guardrail history sealed.
            continuation_context.messages = retention_request.messages
        self._control_plane_timing.record("pre_accept_policy_ms", pre_accept_started)

        # Keyed operations must be accepted before route selection so an existing
        # durable terminal or reused key fails closed before learned selection can
        # run request-time embedding or other provider-touching work. A narrow
        # unkeyed direct Chat path has no learned selector or admission-side effects;
        # queue its durable acceptance and overlap the writer with route assembly.
        # The commit is still required before this admission returns.
        ledger_accept_started = time.monotonic()
        pending_acceptance: concurrent.futures.Future[object] | None = None
        overlap_acceptance = (
            authorization.caller_operation_sha256 is None
            and authorization.model_chain_authority is None
            and authorization.surface is GatewayApiSurface.CHAT_COMPLETIONS
            and isinstance(authorization.target, DirectTarget)
            and continuation_context is None
            and pinned_reasoning_route is None
            and not has_active_reasoning_content(request)
            and self._capture is None
            and self._guardrails is None
            and self._native_route_eligible is None
            and not request.tools
            and request.tool_choice is None
            and request.parallel_tool_calls is None
            and not request.provider_native_tools
            and not request.provider_server_tools
            and request.web_search is None
            and request.tool_search is None
            and request.gateway is None
            and request.previous_response_id is None
            and request.provider_preferences is None
            and not request.zdr_requested
            and request.provider_output_config is None
            and request.provider_prompt_cache_key is None
            and request.native_tool_translation is None
            and all(
                message.role != "tool"
                and message.content is not None
                and not message.content_parts
                and message.tool_call_id is None
                and not message.tool_calls
                and not message.capture_only_reasoning
                and message.provider_specific_fields is None
                and not message.provider_reasoning
                and message.provider_item_id is None
                and message.provider_output_index is None
                and message.provider_status is None
                and message.provider_phase is None
                and message.provider_tool_name is None
                and message.provider_tool_namespace is None
                and message.provider_tool_caller is None
                and message.provider_native_item is None
                and message.provider_anthropic_blocks is None
                and message.provider_anthropic_block is None
                and not message.provider_text_blocks
                and not message.tool_is_error
                for message in request.messages
            )
        )
        try:
            if overlap_acceptance and isinstance(self._write_ledger, SyncGroupCommitLedger):
                pending_acceptance = self._write_ledger.enqueue_accept_request(
                    authorization=authorization
                )
            else:
                self._write_ledger.accept_request(authorization=authorization)
        except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
            self._control_plane_timing.record("ledger_accept_ms", ledger_accept_started)
            raise _authority_error(exc) from exc
        self._control_plane_timing.record("ledger_accept_ms", ledger_accept_started)

        acceptance_observed = pending_acceptance is None

        def wait_for_pending_acceptance() -> None:
            """Observe deferred acceptance before any route exit or success reply."""
            nonlocal acceptance_observed
            if acceptance_observed or pending_acceptance is None:
                return
            accept_wait_started = time.monotonic()
            try:
                pending_acceptance.result()
            except Exception as exc:  # noqa: BLE001 - keep the ledger boundary sanitized.
                raise _authority_error(exc) from exc
            finally:
                self._control_plane_timing.record("ledger_accept_wait_ms", accept_wait_started)
            acceptance_observed = True

        if not begin_capture(
            self._capture,
            authorization,
            captured_request,
            session_id=optional_text(data.get("capture_session_id")),
        ):
            message = "Traffic capture is unavailable or at capacity. Restore capacity and retry."
            self._accounting.finish_request_quietly(
                authorization,
                GatewayFailure(failure_class=GatewayFailureClass.UNAVAILABLE, safe_message=message),
            )
            raise NativeBridgeError(
                OpenAIProtocolError(status_code=503, code="capture_unavailable", message=message)
            )
        route_started = time.monotonic()
        # Escalation finishes the accepted request quietly before returning, so it is
        # accounted content-free and never billed. Routing failures found by
        # the probe are raised against the accepted request below.
        probe_failure: Exception | None = None
        web_search_admission: JsonObject | None = None
        tool_search_admission: JsonObject | None = None
        tool_search_state = None
        route: GatewayRoute | None = None
        resolved_wires: tuple[tuple[GatewayWireProfile, NativeWireClient], ...] | None = None
        try:
            route = pinned_reasoning_route or self._resolve_route(
                authorization,
                request,
                continuation=continuation_context,
            )
            require_route_authority(authorization, request, route)
            route = _select_bound_continuation_route(
                route,
                None
                if continuation_context is None
                else continuation_context.required_route_binding,
            )
            # A rung that is dead at admission (a lost credential, a drifted
            # connection) is skipped so a live fallback still serves the
            # request instead of the whole request failing on a dead lead.
            dispatchable = dispatchable_route_profiles(self._components.runtime_catalogs, route)
            record_dead_admission_rungs(
                self._accounting,
                authorization,
                dispatchable.dead,
                fallback_available=bool(dispatchable.indexes),
            )
            if not dispatchable.indexes:
                if (
                    continuation_context is not None
                    and continuation_context.required_route_binding is not None
                ):
                    raise _continuation_binding_error()
                # No dispatchable rung remains; finalize the accepted request closed.
                wait_for_pending_acceptance()
                return self._escalate_accepted(
                    authorization,
                    "every certified deployment was unavailable at admission",
                )
            route = select_route_deployments(route, dispatchable.indexes)
            resolved_wires = dispatchable.resolved_wires
            route, resolved_wires, selected_placement = select_single_route_before_search(
                route,
                resolved_wires,
                request,
                accounting=self._accounting,
                authorization=authorization,
                continuation=continuation_context,
            )
            searched = plan_web_search(
                request,
                [profile.dialect for profile, _client in resolved_wires],
                self._web_search,
                deadline_monotonic=deadline,
            )
            request, web_search_admission = searched.request, searched.admission
            # Caller-declared tool search on a route with no native one (tool_search.plan).
            planned = plan_tool_search(request, [p.dialect for p, _c in resolved_wires])
            request, tool_search_state = planned.request, planned.state
            tool_search_admission = planned.admission
            _require_bound_wire_authority(
                None
                if continuation_context is None
                else continuation_context.required_route_binding,
                route,
                resolved_wires,
            )
        except NativeDialectUnavailableError as exc:
            wait_for_pending_acceptance()
            return self._escalate_accepted(authorization, str(exc))
        except OpenAIProtocolError as exc:
            # A continuation whose bound provider authority is no longer
            # available is a CLIENT error (400 previous_response_not_found: resend
            # the full conversation), not a gateway-internal fault. Class the
            # durable failure by the public status so usage and health read it
            # as a client failure and it never pages as internal; the caller
            # still receives the exact public error unchanged.
            wait_for_pending_acceptance()
            self._accounting.finish_request_quietly(
                authorization,
                GatewayFailure(
                    failure_class=(
                        GatewayFailureClass.INTERNAL
                        if exc.status_code >= 500
                        else GatewayFailureClass.INVALID_REQUEST
                    ),
                    safe_message=exc.detail.message,
                ),
            )
            raise NativeBridgeError(exc) from exc
        except Exception as exc:  # noqa: BLE001 - raised after route packaging below.
            probe_failure = exc
        if route is not None and self._native_route_eligible is not None:
            try:
                native_route_eligible = self._native_route_eligible(route, request)
            except Exception:  # noqa: BLE001 - hosted policy fails closed.
                native_route_eligible = False
            if not native_route_eligible:
                wait_for_pending_acceptance()
                return self._escalate_accepted(
                    authorization,
                    "host policy does not permit native execution of this route",
                )

        # Admission returns the full ordered route; no attempt row exists
        # until the data plane's first `start_attempt`.
        public_request = request
        provider_request = request.model_copy(update={"stream": True, "include_usage": True})
        try:
            if probe_failure is not None or route is None or resolved_wires is None:
                raise probe_failure or GatewayRoutingError("authorized route did not resolve")
            route, resolved_wires, public_request, provider_request, placement = (
                admitted_route_requests(
                    route,
                    resolved_wires,
                    request,
                    accounting=self._accounting,
                    authorization=authorization,
                    continuation=continuation_context,
                )
            )
            placement = selected_placement or placement
            require_route_authority(authorization, request, route)
            require_unmodified_probability_output(request, bool(policy and policy.output_checks))
            wire_route: list[JsonObject] = []
            parallel_disclosures: set[str] = set()
            output_bounds: list[int] = []
            signers: list[GatewayDispatchSigner | None] = []
            dispatch_bindings: list[FrozenDispatchBinding | None] = []
            carrier_authorities: list[ReasoningCarrierAuthority | None] = []
            # How long a throttle is worth waiting on per rung for THIS
            # request (the pool's schedule scaled by the cache at stake),
            # decided here so the data plane never waits on a rung whose
            # throttle should fail over cold at once. `route` is the admitted
            # route (dead and incompatible rungs already removed), so its last
            # rung is the one with no cold alternative; the sticky binding is
            # the one placement already read.
            redial_budgets = throttle_redial_budgets(
                self._accounting.loads,
                route,
                authorization.organization_id,
                sticky_deployment_id=placement.sticky_deployment_id,
            )
            for deployment, (profile, client), budget in zip(
                route.deployments, resolved_wires, redial_budgets, strict=True
            ):
                # A reasoning-pinned route's fallback rung is frozen WITHOUT
                # the pinned provider's sealed reasoning (it cannot unseal
                # it), so a failover past the issuing rung dispatches the
                # conversation minus that turn's thinking, never a foreign
                # sealed block.
                dispatch = build_rung_dispatch(
                    route,
                    deployment,
                    profile,
                    client,
                    provider_request=rung_provider_request(route, deployment, provider_request),
                    public_request=public_request,
                    authorization=authorization,
                    throttle_redial_budget=budget,
                )
                if dispatch.parallel_disclosure is not None:
                    parallel_disclosures.add(dispatch.parallel_disclosure)
                if dispatch.output_disclosure is not None:
                    parallel_disclosures.add(dispatch.output_disclosure)
                output_bounds.append(dispatch.reserved_output_tokens)
                wire_route.append(dispatch.wire_entry)
                signers.append(dispatch.signer)
                dispatch_bindings.append(dispatch.binding)
                carrier_authorities.append(dispatch.carrier_authority)
            public_request = fold_parallel_tool_call_disclosures(
                public_request,
                parallel_disclosures,
                accounting=self._accounting,
                authorization=authorization,
            )
            if continuation_context is not None:
                continuation_context.route_bindings = tuple(
                    continuation_route_binding(deployment, profile)
                    for deployment, (profile, _client) in zip(
                        route.deployments,
                        resolved_wires,
                        strict=True,
                    )
                )
            cache_state, public_request = bind_explicit_cache(
                self._explicit_cache,
                authorization,
                route.deployments,
                resolved_wires,
                provider_request,
                public_request,
                wire_route,
            )
        except NativeBridgeError:
            # The enriched fail-closed capability rejection above already
            # finished the accepted request; let it cross the boundary as-is.
            wait_for_pending_acceptance()
            raise
        except (ProviderParameterError, ProviderCapabilityError) as exc:
            # One shared normalizer keeps both pre-dispatch rejections
            # field-specific: the parameter path names the parameter and the
            # capability path names the capability, so a triager sees which
            # request feature the route cannot preserve.
            failure = normalized_provider_failure(exc)
            if isinstance(exc, ProviderCapabilityError):
                public_error = _public_capability_error(
                    exc,
                    provider_request.surface,
                    public_stream=public_request.stream,
                    public_tools=bool(public_request.tools),
                    developer_messages_param=decoded.developer_messages_param,
                )
                # The ledger keeps the capability-free generic sentence, but a
                # bare "cannot preserve a requested capability" is untriageable
                # from an alert. Append the PUBLIC field the caller was told
                # about (never the internal literal), so operators read
                # "(field: stop)" without opening the request.
                failure = failure.model_copy(
                    update={
                        "safe_message": ledger_capability_message(
                            failure.safe_message, public_error.detail.param
                        )
                    }
                )
            else:
                public_error = public_failure_error(failure, param=exc.param)
            wait_for_pending_acceptance()
            self._accounting.finish_request_quietly(authorization, failure)
            raise NativeBridgeError(public_error) from exc
        except GatewayRoutingError as exc:
            # A route/catalog that cannot be built during a rolling deploy is a
            # transient control-plane condition, not a bug: record it retryable
            # so it never pages as INTERNAL. The public error is already a 503.
            failure = gateway_updating_failure()
            wait_for_pending_acceptance()
            self._accounting.finish_request_quietly(authorization, failure)
            raise _authority_error(exc) from exc
        except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
            error = _authority_error(exc)
            failure = GatewayFailure(
                failure_class=GatewayFailureClass.INTERNAL,
                safe_message="gateway admission failed before provider dispatch",
            )
            # The public error and the ledger row carry only the sanitized
            # text, so this record is the ONLY place the real exception
            # survives: an unlogged INTERNAL here left a granted alias failing
            # 500 for hours with nothing to diagnose (platform staging,
            # 2026-09-03). The message names the request and alias; the
            # traceback rides exc_info. Nothing here carries a credential.
            _logger.exception(
                "gateway admission failed before provider dispatch",
                extra={
                    "operation": "native_admit",
                    "request_id": authorization.request_id,
                    "alias": authorization.alias,
                    "alias_revision_id": authorization.alias_revision_id,
                    "exception_type": type(exc).__name__,
                },
            )
            wait_for_pending_acceptance()
            self._accounting.finish_request_quietly(authorization, failure)
            raise error from exc

        plan = deterministic.native_output_plan(policy, self._guardrail_detectors)
        wait_for_pending_acceptance()
        self._accounting.register(
            InflightRequest(
                authorization=authorization,
                route=route,
                request=provider_request,
                deadline_monotonic=deadline,
                continuation=continuation_context,
                policy=policy,
                signers=tuple(signers),
                dispatch_bindings=tuple(dispatch_bindings),
                reasoning_carrier_authorities=tuple(carrier_authorities),
                tier_forwarded_by_depth=tuple(
                    profile.forwards_tier(provider_request.service_tier)
                    for profile, _client in resolved_wires
                ),
                reserved_output_tokens_by_depth=tuple(output_bounds),
                affinity_fingerprint=placement.fingerprint,
                verified_warm_deployment_id=placement.verified_warm_deployment_id,
                verified_warm_until_monotonic=placement.verified_warm_until_monotonic,
                recovery_scoped=placement.recovery_scoped,
                sticky_preferred=placement.sticky_preferred,
                throttle_redial_budgets=redial_budgets,
                recovery_reason=placement.recovery_reason,
                recovery_bindings={
                    deployment.deployment_id: binding
                    for deployment, (profile, _) in zip(
                        route.deployments, resolved_wires, strict=True
                    )
                    if (
                        binding := validated_recovery_binding(
                            deployment, profile, authorization.organization_id
                        )
                    )
                    is not None
                },
                resolved_wires=None if tool_search_state is None else tuple(resolved_wires),
                public_request=None if tool_search_state is None else public_request,
                tool_search=tool_search_state,
                explicit_cache_state=cache_state,
            )
        )
        select_capture_model(self._capture, authorization.request_id, route.snapshot.exact_model_id)
        response: JsonObject = {
            "request_id": authorization.request_id,
            "alias": authorization.alias,
            "alias_revision_id": authorization.alias_revision_id,
            "stream": request.stream,
            "include_usage": request.include_usage,
            "exact_model_id": route.snapshot.exact_model_id,
            "route_reason": route.route_reason,
            "route": wire_route,
            "ignored_parameters": list(public_request.ignored_parameters),
            **attempt_policy(request.gateway).model_dump(mode="json"),
            "refusal_failover": authorization.refusal_failover,
            "output_guardrail": native_output_mode(
                self._guardrails,
                policy,
                public_request,
                image_output=any(wire.get("image_output") is True for wire in wire_route),
            ).value,
            "caller_scope": f"{authorization.organization_id}:{authorization.identity_id}",
        }
        if route.snapshot.throttle_redial is not None:
            # The pool's frozen backoff-and-redial schedule; absent (not null) on
            # pools that keep throttles failover-only: their admission is byte-identical.
            response["throttle_redial"] = route.snapshot.throttle_redial.model_dump(mode="json")
        if plan is not None:
            response["guardrail_output_plan"] = plan
        if web_search_admission is not None:
            # The gateway searched: the data plane cites the sources and counts it.
            response["web_search"] = web_search_admission
        if tool_search_admission is not None:
            response["tool_search"] = tool_search_admission
        if request.surface == GatewayApiSurface.MESSAGES:
            # Display-only: what `message_start` shows as input when the
            # upstream reports nothing before its final chunk. The ledger
            # never reads it; settlement keeps the provider's meters.
            response["input_token_estimate"] = counted_input_tokens(public_request)
        if public_request.maximum_output_tokens is not None:
            # The caller's cap: classifies an output-less, usage-less `stop` (capped -> length).
            response["maximum_output_tokens"] = public_request.maximum_output_tokens
        if request.surface == GatewayApiSurface.RESPONSES:
            response["surface"] = "responses"
            response["envelope"] = responses_envelope(public_request)
        self._control_plane_timing.record("route_and_register_ms", route_started)
        response_started = time.monotonic()
        encoded_response = json.dumps(response, separators=(",", ":"))
        self._control_plane_timing.record("response_encode_ms", response_started)
        return encoded_response
