"""Run one ordered input or output guardrail chain under request deadlines."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from hashlib import sha256

from exp.common.core.artifacts import canonical_json_bytes
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    GatewayFailure,
    GatewayFailureClass,
    GatewayRequest,
)
from exp.runtime.gateway.guardrails.bounded import BoundedInspect, ClassifierTimeoutError
from exp.runtime.gateway.guardrails.client import InternalClassifierClient
from exp.runtime.gateway.guardrails.contracts import (
    ClassifierCoverageError,
    ClassifierUncertainError,
    ClassifierVerdict,
    GuardrailAction,
    GuardrailCheck,
    GuardrailCompletion,
    GuardrailOutcome,
    GuardrailPolicy,
    GuardrailRejected,
    OutputGuardrailMode,
    coverage_failure,
    guardrail_failure,
    request_exceeds_inspection_limit,
)
from exp.runtime.gateway.guardrails.observation import (
    ObservationAdmission,
    ObservationLease,
    ObservationOwner,
)
from exp.runtime.gateway.guardrails.recording import ObservationRecord, ObservationRecorder
from exp.runtime.gateway.guardrails.redaction import restored_provider_authority
from exp.runtime.gateway.guardrails.session import GuardrailSession
from exp.runtime.gateway.guardrails.store import GuardrailPolicyStore
from exp.runtime.gateway.guardrails.streaming import (
    StreamingRedactor,
    StreamSegment,
    release_segment,
)
from exp.runtime.gateway.guardrails.subjects import (
    ObservedSubjects,
    observation_subject_bytes,
    observation_subject_exceeds,
)

_logger = logging.getLogger(__name__)


class GuardrailEngine:
    """Resolve scoped policies and execute their checks through request-owned sessions.

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
        deterministic_specifications: Mapping[str, str] | None = None,
        max_observations: int = 8,
        max_observation_bytes: int = 8 * 1024 * 1024,
        max_observation_records: int = 128,
    ) -> None:
        """Bind lookup, the internal client, and the deadline clock.

        Args:
            store: Applicable platform and identity policy lookup.
            client: Injected adapter seam that cannot use the public route.
            monotonic: Process-local clock in seconds.
            inspects: Optional async inflight limiter. ``None`` uses the
                default shared cap.
            deterministic_specifications: Content-free native rules for the
                registered deterministic adapters, keyed by adapter. A host
                that runs the Rust data plane compiles these once and lets
                matching chains run in plane. Omitting them keeps every
                chain on this engine.
            max_observations: Maximum retained asynchronous observation jobs, default 8.
            max_observation_bytes: Maximum complete subject bytes retained, default 8 MiB.
            max_observation_records: Maximum queued plus active metadata records, default 128.
        """
        self.deterministic_specifications: Mapping[str, str] = dict(
            deterministic_specifications or {}
        )
        self._store = store
        self._client = client
        self._monotonic = monotonic
        self._owns_inspects = inspects is None
        self._inspects = inspects or BoundedInspect()
        self._observation_inspects = BoundedInspect(max_inflight=max_observations)
        self._observations = ObservationOwner(
            max_jobs=max_observations, max_bytes=max_observation_bytes
        )
        self._observation_records = ObservationRecorder(
            self._deliver_observation, max_records=max_observation_records
        )
        self.input_invocations = 0
        self.output_invocations = 0
        self.classifier_calls = 0

    def policies_for(self, organization_id: str, identity_id: str) -> tuple[GuardrailPolicy, ...]:
        """Resolve all applicable scopes through the single policy store contract."""
        return self._store.policies_for(organization_id, identity_id)

    def open_request(
        self, authorization: AuthorizationSnapshot, *, deadline_monotonic: float
    ) -> GuardrailSession | None:
        """Freeze all applicable policies into one request-owned execution session."""
        policies = self.policies_for(authorization.organization_id, authorization.identity_id)
        if not policies:
            return None
        return GuardrailSession(
            engine=self, policies=policies, deadline_monotonic=deadline_monotonic
        )

    def revision_for(self, authorization: AuthorizationSnapshot) -> str | None:
        """Bind replay to enforcing policy, adapter, and execution configuration."""
        return self.policy_revision(
            self.policies_for(authorization.organization_id, authorization.identity_id)
        )

    def policy_revision(self, policies: tuple[GuardrailPolicy, ...]) -> str | None:
        """Hash the enforcing snapshot; observation cannot alter response replay identity."""
        policies = tuple(policy for policy in policies if policy.mode == "enforce")
        adapters = {check.adapter_id for policy in policies for check in policy.checks}
        return (
            None
            if not policies
            else sha256(
                canonical_json_bytes(
                    {
                        "policies": [p.model_dump(mode="json") for p in policies],
                        "detectors": {
                            key: value
                            for key, value in self.deterministic_specifications.items()
                            if key in adapters
                        },
                    }
                )
            ).hexdigest()
        )

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
        if policy.mode == "observe":
            self.observe_input(
                policy=policy, request=request, deadline_monotonic=deadline_monotonic
            )
            return request
        return await self._inspect_input(
            policy=policy, request=request, deadline_monotonic=deadline_monotonic
        )

    async def _inspect_input(
        self,
        *,
        policy: GuardrailPolicy,
        request: GatewayRequest,
        deadline_monotonic: float,
        retention: ObservationLease | None = None,
    ) -> GatewayRequest:
        """Evaluate the same bounded checks while granting actions only to enforcing policies."""
        self.input_invocations += 1
        if request_exceeds_inspection_limit(request, policy.max_request_bytes):
            self._decision(policy, None, GuardrailAction.ERROR, GuardrailOutcome.UNSUPPORTED, 0.0)
            if policy.mode == "observe":
                return request
            raise GuardrailRejected(coverage_failure())
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
                retention=retention,
            )
            if verdict is None or policy.mode == "observe":
                continue
            current = self._apply_input(policy, check, current, verdict)
        return current

    def observe_input(
        self, *, policy: GuardrailPolicy, request: GatewayRequest, deadline_monotonic: float
    ) -> None:
        """Submit one ephemeral input to the shared executor without gating its request.

        Admission is bounded by jobs, bytes, and the original request deadline.
        Normal request completion does not cancel these engine-owned jobs. Hosts
        call ``close`` before shutting down their observation recorder.
        """
        if policy.mode != "observe":
            raise ValueError("observation requires an observe policy")
        self.observe_inputs(
            policies=(policy,), request=request, deadline_monotonic=deadline_monotonic
        )

    def observe_inputs(
        self,
        *,
        policies: tuple[GuardrailPolicy, ...],
        request: GatewayRequest,
        deadline_monotonic: float,
        observed: ObservedSubjects | None = None,
    ) -> None:
        """Prepare one bounded complete subject under the shared nonblocking permit.

        A free permit preserves exact deduplication for known session subjects even
        when inspection capacity is full. Preparation contention ends this session's
        optional coverage without claiming that an unknown subject was distinct.
        """
        if any(policy.mode != "observe" for policy in policies):
            raise ValueError("observation requires observe policies")
        if not policies or (observed is not None and observed.closed):
            return
        deduplicating = observed is not None and bool(observed.fingerprints)
        if not deduplicating and deadline_monotonic <= self._monotonic():
            self._preparation_unavailable(policies, observed)
            return
        with self._observations.preparation(deduplicating=deduplicating) as admitted:
            if not admitted:
                self._preparation_unavailable(policies, observed)
                return
            try:
                self._prepare_observations(
                    policies=policies,
                    request=request,
                    deadline_monotonic=deadline_monotonic,
                    observed=observed,
                )
            except Exception:  # noqa: BLE001 - optional preparation cannot reject serving.
                pass
            else:
                return
            self._preparation_unavailable(policies, observed, outcome=GuardrailOutcome.UNAVAILABLE)

    def _preparation_unavailable(
        self,
        policies: tuple[GuardrailPolicy, ...],
        observed: ObservedSubjects | None,
        *,
        outcome: GuardrailOutcome = GuardrailOutcome.SKIPPED,
    ) -> None:
        """End incomplete admission once without declaring a distinct missed subject."""
        if observed is not None and not observed.close("preparation_unavailable"):
            return
        for policy in policies:
            self._emit_observation(policy, None, outcome, 0.0)

    def _prepare_observations(
        self,
        *,
        policies: tuple[GuardrailPolicy, ...],
        request: GatewayRequest,
        deadline_monotonic: float,
        observed: ObservedSubjects | None,
    ) -> None:
        """Release every projection/hash temporary before returning the preparation permit."""
        limit = min(
            self._observations.max_subject_bytes,
            max(policy.max_request_bytes for policy in policies),
        )
        if observation_subject_exceeds(request, limit):
            outcomes = tuple(
                GuardrailOutcome.UNSUPPORTED
                if observation_subject_exceeds(request, policy.max_request_bytes)
                else GuardrailOutcome.SKIPPED
                for policy in policies
            )
            if observed is not None and not observed.close("subject_oversized"):
                return
            for policy, outcome in zip(policies, outcomes, strict=True):
                self._emit_observation(policy, None, outcome, 0.0)
            return
        # Frozen contracts can contain mutable JSON containers owned by the caller.
        # Detach once so the hashed, budgeted and asynchronously inspected subject agrees.
        request = request.model_copy(deep=True)
        subject: bytes | None = None
        try:
            subject = observation_subject_bytes(request)
            size = len(subject)
            fingerprint = sha256(subject).digest()
        finally:
            del subject
        if observed is not None:
            if fingerprint in observed.fingerprints:
                return
            observed.fingerprints.add(fingerprint)
        if (
            deadline_monotonic <= self._monotonic()
            or self._observations.available_subject_bytes == 0
        ):
            for policy in policies:
                self._emit_observation(policy, None, GuardrailOutcome.SKIPPED, 0.0)
            return
        for policy in policies:
            self._submit_observation(
                policy=policy,
                request=request,
                deadline_monotonic=deadline_monotonic,
                size=size,
            )

    def _submit_observation(
        self,
        *,
        policy: GuardrailPolicy,
        request: GatewayRequest,
        deadline_monotonic: float,
        size: int,
    ) -> None:
        """Reserve one policy's subject using the exact shared encoded size."""
        if size > policy.max_request_bytes:
            self._emit_observation(policy, None, GuardrailOutcome.UNSUPPORTED, 0.0)
            return
        if deadline_monotonic <= self._monotonic():
            self._emit_observation(policy, None, GuardrailOutcome.SKIPPED, 0.0)
            return

        async def inspect(lease: ObservationLease) -> None:
            """Evaluate this frozen subject with the ordinary shared classifier executor."""
            if deadline_monotonic <= self._monotonic():
                self._emit_observation(policy, None, GuardrailOutcome.SKIPPED, 0.0)
                return
            await self._inspect_input(
                policy=policy,
                request=request,
                deadline_monotonic=deadline_monotonic,
                retention=lease,
            )

        def interrupted(cancelled: bool) -> None:
            """Record work that did not produce an inspection decision without request content."""
            self._emit_observation(
                policy,
                None,
                GuardrailOutcome.SKIPPED if cancelled else GuardrailOutcome.UNAVAILABLE,
                0.0,
            )

        admission = self._observations.submit(
            inspect, subject_bytes=size, on_interrupted=interrupted
        )
        if admission is not ObservationAdmission.ACCEPTED:
            outcome = (
                GuardrailOutcome.UNAVAILABLE
                if admission is ObservationAdmission.UNAVAILABLE
                else GuardrailOutcome.SKIPPED
            )
            self._emit_observation(policy, None, outcome, 0.0)

    def record_unsupported_observations(self, policies: tuple[GuardrailPolicy, ...]) -> None:
        """Record incomplete observation coverage on a surface with no inspection boundary."""
        for policy in policies:
            if policy.mode == "observe":
                self._emit_observation(policy, None, GuardrailOutcome.UNSUPPORTED, 0.0)

    def close(self, *, timeout_seconds: float = 1.0) -> None:
        """Stop observations and drain within a bounded host shutdown budget."""
        deadline = time.monotonic() + timeout_seconds
        self._observations.close(timeout_seconds=timeout_seconds)
        self._observation_inspects.close(timeout_seconds=max(0.0, deadline - time.monotonic()))
        if self._owns_inspects:
            self._inspects.close(timeout_seconds=max(0.0, deadline - time.monotonic()))
        self._observation_records.close(timeout_seconds=max(0.0, deadline - time.monotonic()))

    @property
    def observation_recording_dropped(self) -> int:
        """Count metadata records lost to bounded capacity, shutdown, or worker startup."""
        return self._observation_records.dropped_count

    @property
    def observation_recording_failed(self) -> int:
        """Count sink exceptions without exposing recorder diagnostics."""
        return self._observation_records.failed_count

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
        if policy.mode == "observe":
            return completion
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

    def output_mode(
        self,
        policy: GuardrailPolicy | None,
        *,
        streaming: bool,
        tools_offered: bool,
        reasoning_text_requested: bool,
    ) -> OutputGuardrailMode:
        """Decide how one admission's output chain must be enforced.

        Incremental enforcement releases bytes the caller can never take
        back, so it is offered only when every decision is final at the
        moment it is made. That needs one output check whose action is
        ``modify`` (a later ``block`` could not suppress bytes already sent)
        and whose adapter offers a deterministic redactor (a detector that
        needs the whole completion cannot decide about a prefix). A chain of
        several checks would have to compose redactions over partially
        released text, so it stays buffered.

        The request shape decides the rest. A buffered rewrite protects the
        caller from alternate channels by dropping them once it has the whole
        completion: tool calls, reasoning text, server-tool activity, and
        citations never survive a redaction. Incremental release cannot drop
        what it has already sent, so a request that can produce one of those
        channels stays buffered: any offered tool (which is also what admits
        server tools and their citations) and any request for thinking or a
        reasoning summary.

        Args:
            policy: Assigned identity policy, or ``None`` when unguarded.
            streaming: Whether the caller asked for a streamed response.
            tools_offered: Whether the request exposes any tool to the model.
            reasoning_text_requested: Whether the caller asked for thinking
                or a reasoning summary in its own output.

        Returns:
            The mode the data plane must apply for this admission.
        """
        if policy is None or not policy.output_checks:
            return OutputGuardrailMode.OFF
        if not streaming or tools_offered or reasoning_text_requested:
            return OutputGuardrailMode.BUFFER
        if len(policy.output_checks) != 1:
            return OutputGuardrailMode.BUFFER
        check = policy.output_checks[0]
        if check.action is not GuardrailAction.MODIFY:
            return OutputGuardrailMode.BUFFER
        if self._stream_redactor(check) is None:
            return OutputGuardrailMode.BUFFER
        return OutputGuardrailMode.STREAM

    def release_output_segment(
        self,
        *,
        policy: GuardrailPolicy,
        pending: str,
        final: bool,
        settled_bytes: int,
        deadline_monotonic: float,
    ) -> StreamSegment:
        """Redact and release the settled part of one buffered stream tail.

        The call is synchronous and keeps no per-request state: a
        deterministic redactor is pure bounded CPU work, and keeping it off
        the isolation worker is what preserves the caller's time to first
        byte.

        Every failure is terminal, for protected and unprotected identities
        alike. The skip-and-continue path an unprotected buffered chain uses
        would have to emit the unredacted tail, and released bytes cannot be
        recalled, so the incremental path always fails closed.

        Args:
            policy: Assigned identity policy.
            pending: Buffered completion tail, oldest character first.
            final: Whether the provider stream has ended.
            settled_bytes: Provider completion bytes already released from
                the buffer, counted before redaction so a short replacement
                cannot shrink the completion against its bound.
            deadline_monotonic: Request-wide deadline that also bounds this segment.

        Returns:
            The redacted release, the tail to keep buffered, and the flag.

        Raises:
            GuardrailRejected: The chain is not stream eligible, a bound was
                breached, or the adapter refused the subject.
        """
        self.output_invocations += 1
        check = policy.output_checks[0] if len(policy.output_checks) == 1 else None
        redactor = None if check is None else self._stream_redactor(check)
        if check is None or check.action is not GuardrailAction.MODIFY or redactor is None:
            self._record(policy, check, GuardrailAction.ERROR, 0.0)
            raise GuardrailRejected(guardrail_failure(action=GuardrailAction.ERROR))
        if settled_bytes + len(pending.encode("utf-8")) > policy.max_response_bytes:
            self._record(policy, check, GuardrailAction.ERROR, 0.0)
            raise GuardrailRejected(guardrail_failure(action=GuardrailAction.ERROR))
        started = self._monotonic()
        budget = min(check.timeout_ms / 1000.0, deadline_monotonic - started)
        if budget <= 0:
            self._record(policy, check, GuardrailAction.ERROR, 0.0)
            raise GuardrailRejected(guardrail_failure(action=GuardrailAction.ERROR))
        self.classifier_calls += 1
        try:
            segment = release_segment(redactor=redactor, pending=pending, final=final)
        except Exception:  # noqa: BLE001 - an adapter failure releases nothing.
            self._record(policy, check, GuardrailAction.ERROR, self._monotonic() - started)
            raise GuardrailRejected(
                guardrail_failure(action=GuardrailAction.ERROR, check_id=check.check_id)
            ) from None
        elapsed = self._monotonic() - started
        if elapsed > budget:
            self._record(policy, check, GuardrailAction.ERROR, elapsed)
            raise GuardrailRejected(guardrail_failure(action=GuardrailAction.ERROR))
        if segment.flagged:
            self._record(policy, check, check.action, elapsed)
        return segment

    def _stream_redactor(self, check: GuardrailCheck) -> StreamingRedactor | None:
        """Return the check adapter's deterministic redactor, or ``None``."""
        try:
            return self._client.stream_redactor(check=check)
        except Exception:  # noqa: BLE001 - an unresolvable adapter is not streamable.
            return None

    async def _run_check(
        self,
        *,
        policy: GuardrailPolicy,
        check: GuardrailCheck,
        inspect: Callable[[], Awaitable[ClassifierVerdict]],
        deadline_monotonic: float,
        retention: ObservationLease | None = None,
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
            return self._uncertain(policy, check, GuardrailOutcome.TIMEOUT)
        started = self._monotonic()
        inspects = self._observation_inspects if policy.mode == "observe" else self._inspects
        try:
            self.classifier_calls += 1
            verdict = await inspects.run(
                inspect,
                timeout,
                adapter_id=check.adapter_id,
                retention=retention,
            )
        except ClassifierCoverageError:
            self._decision(
                policy,
                check,
                GuardrailAction.ERROR,
                GuardrailOutcome.UNSUPPORTED,
                self._monotonic() - started,
            )
            if policy.mode == "enforce":
                raise GuardrailRejected(coverage_failure(check_id=check.check_id)) from None
            return None
        except ClassifierUncertainError:
            return self._uncertain(
                policy, check, GuardrailOutcome.UNCERTAIN, self._monotonic() - started
            )
        except ClassifierTimeoutError:
            return self._uncertain(
                policy, check, GuardrailOutcome.TIMEOUT, self._monotonic() - started
            )
        except Exception:  # noqa: BLE001 - classifier failures are fail-closed or skipped
            return self._uncertain(
                policy, check, GuardrailOutcome.UNAVAILABLE, self._monotonic() - started
            )
        elapsed = self._monotonic() - started
        if not verdict.flagged:
            self._decision(policy, check, GuardrailAction.ALLOW, GuardrailOutcome.ALLOW, elapsed)
            return None
        self._decision(policy, check, check.action, GuardrailOutcome.FLAGGED, elapsed)
        return verdict

    def _uncertain(
        self,
        policy: GuardrailPolicy,
        check: GuardrailCheck,
        outcome: GuardrailOutcome,
        latency_seconds: float = 0.0,
    ) -> ClassifierVerdict | None:
        """Apply fail-closed or skip-and-continue for an uncertain check."""
        self._decision(policy, check, GuardrailAction.ERROR, outcome, latency_seconds)
        if policy.protected and policy.mode == "enforce":
            raise GuardrailRejected(
                GatewayFailure(
                    failure_class=GatewayFailureClass.UNAVAILABLE,
                    safe_message="Content inspection is unavailable. Retry later.",
                    safe_details={"action": "error", "check_id": check.check_id},
                )
            )
        return None

    def _decision(
        self,
        policy: GuardrailPolicy,
        check: GuardrailCheck | None,
        action: GuardrailAction,
        outcome: GuardrailOutcome,
        latency_seconds: float,
    ) -> None:
        """Separate observed outcomes from actions that actually affect customer requests."""
        if policy.mode == "observe":
            self._emit_observation(policy, check, outcome, latency_seconds)
        else:
            self._record(policy, check, action, latency_seconds)

    def _emit_observation(
        self,
        policy: GuardrailPolicy,
        check: GuardrailCheck | None,
        outcome: GuardrailOutcome,
        latency_seconds: float,
    ) -> None:
        """Queue only immutable metadata; recorder latency never enters serving or inspection."""
        self._observation_records.submit(ObservationRecord(policy, check, outcome, latency_seconds))

    def _deliver_observation(self, record: ObservationRecord) -> None:
        """Invoke the host sink only on the separately bounded metadata worker."""
        self._record_observation(
            record.policy, record.check, record.outcome, record.latency_seconds
        )

    def _record_observation(
        self,
        policy: GuardrailPolicy,
        check: GuardrailCheck | None,
        outcome: GuardrailOutcome,
        latency_seconds: float,
    ) -> None:
        """Emit content-free observation metadata separately from enforced decisions."""
        _logger.info(
            "guardrail observation policy_id=%s revision=%s organization_id=%s identity_id=%s "
            "check_id=%s capability=%s would_action=%s outcome=%s latency_ms=%.1f",
            policy.policy_id,
            policy.revision,
            policy.organization_id,
            policy.identity_id,
            None if check is None else check.check_id,
            None if check is None else check.capability.value,
            None if check is None else check.action.value,
            outcome.value,
            latency_seconds * 1000,
        )

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
            restored = restored_provider_authority(
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
            return completion.model_copy(
                update={"text": verdict.replacement_text, "context": (), "refusal": False}
            )
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
