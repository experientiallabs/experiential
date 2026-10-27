"""The video job engine: admission, failover, durable jobs, polling, and settlement.

Create runs the synchronous surfaces' admission unchanged (authority, alias
grant, route resolution, ledger acceptance) and then walks the route: each
rung reserves ``seconds x rate`` on its own ledger attempt, a job row is
persisted, and only then does the provider see the submit. A rung whose
provider answers without creating a job settles at zero and the next rung
serves; once a provider accepts, the job is bound to that rung for good.

The accepted attempt stays dispatched while the provider renders, holding its
reservation. Whichever process first observes the terminal state (the owner's
poll or the server-side poller) settles it exactly once: a completed video
bills its rendered seconds, never more than requested, at the frozen rate for
its resolution; a provider-reported failed or filtered render bills zero. A
local settlement timeout finalizes delivery as failed and releases the customer
reservation while preserving unknown provider usage for later reconciliation.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.client_apps import with_client_identity
from exp.runtime.gateway.contracts import AuthorizationSnapshot, DirectTarget
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.native import require_unguarded_surface
from exp.runtime.gateway.ledger_errors import (
    AccountingRecoverySupersededError,
    AccountingRecoveryUnresolvedTerminalError,
)
from exp.runtime.gateway.model_chain_authority import authorize_serving_model_chains
from exp.runtime.gateway.native_accounting_errors import NativeBridgeError, authority_error
from exp.runtime.gateway.native_accounting_recovery import RecoveryRotation
from exp.runtime.gateway.native_components import NativeGatewayComponents, SyncWriteLedger
from exp.runtime.gateway.native_execution import (
    _resolve_deployment_profile,  # The identity-checked resolver every admission uses.
    dispatchable_route_profiles,
    select_route_deployments,
)
from exp.runtime.gateway.native_request_settlement import RequestSettlements
from exp.runtime.gateway.native_settlement import gateway_updating_failure
from exp.runtime.gateway.stream_contracts import (
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
)
from exp.runtime.gateway.video.contracts import (
    VIDEO_JOB_SETTLEMENT_SECONDS,
    VIDEO_LEDGER_DEADLINE_SLACK_SECONDS,
    VideoJob,
    VideoRequest,
    VideoStatus,
    video_ceiling_nano_usd,
    video_unit_rate,
)
from exp.runtime.gateway.video.output import VideoOutputStore
from exp.runtime.gateway.video.providers import (
    VideoContentFetch,
    VideoProviderClient,
    VideoProviderSnapshot,
    VideoSubmission,
    VideoSubmitRejected,
    default_video_clients,
    video_dialect,
)
from exp.runtime.gateway.video.settlement import (
    JobGate,
    PendingSettlement,
    closed_job,
    require_same_job,
    terminal_settlement,
    zero_units,
)
from exp.runtime.gateway.video.store import ListOrder, VideoJobStore
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.openai_protocol.errors import OpenAIProtocolError, public_failure_error

_LOGGER = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 5.0
"""Minimum spacing between provider polls of one job, across owner reads and the poller."""

DISPATCH_GRACE_SECONDS = 5 * 60
"""How long a job may sit without a provider handle before its dispatch counts as lost."""

_POLLER_BATCH = 100
_LIST_LIMIT_MAXIMUM = 100


def _now() -> datetime:
    """Return the current timezone-aware wall-clock time."""
    return datetime.now(UTC)


def not_a_video_model_error(alias: str) -> OpenAIProtocolError:
    """Build the public 400 for an alias none of whose rungs generate video."""
    return OpenAIProtocolError(
        status_code=400,
        code="unsupported_capability",
        message=(
            f"The model {alias!r} does not generate video. Choose a video model from "
            "GET /v1/models and resend the request to /v1/videos."
        ),
        param="model",
    )


def video_not_found_error(video_id: str) -> OpenAIProtocolError:
    """Build the public 404 shared by absent, deleted, and foreign video ids."""
    return OpenAIProtocolError(
        status_code=404,
        code="not_found",
        message=f"No video found with id {video_id!r}.",
        param="video_id",
    )


class VideoEngine:
    """Video job lifecycle over the native control plane's shared seams.

    Every public read is scoped to the creating organization and identity.
    Settlement is safe to race: the ledger finishes an attempt exactly once
    and the job store's compare-and-set keeps the first settled record.
    """

    def __init__(
        self,
        *,
        components: NativeGatewayComponents,
        ledger: SyncWriteLedger,
        jobs: VideoJobStore,
        guardrails: GuardrailEngine | None = None,
        clients: Mapping[str, VideoProviderClient] | None = None,
        clock: Callable[[], datetime] = _now,
        poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
        outputs: VideoOutputStore | None = None,
    ) -> None:
        """Bind the shared seams and the provider client registry.

        Args:
            components: Authority store, routes, and runtime catalogs.
            ledger: The synchronous write ledger every surface settles through.
            jobs: Durable job persistence.
            guardrails: Policy engine; a guarded identity is refused this surface.
            clients: Provider clients keyed by video dialect.
            clock: Injectable wall clock.
            poll_interval_seconds: Minimum spacing between polls of one job.
            outputs: Host storage rendered videos upload into; when set, every
                job on a dialect that ``stores_output`` uploads there.
        """
        self._components = components
        self._ledger = ledger
        self._request_settlements = RequestSettlements(ledger)
        self._jobs = jobs
        self._guardrails = guardrails
        self._clients = dict(clients) if clients is not None else default_video_clients()
        self._clock = clock
        self._poll_interval = timedelta(seconds=max(0.0, poll_interval_seconds))
        self._outputs = outputs
        self._pending_lock = threading.Lock()
        self._active_creates = 0
        self._pending: dict[str, PendingSettlement] = {}
        self._job_gates: dict[str, JobGate] = {}
        self._recovery = RecoveryRotation[PendingSettlement](lambda item: item.job.video_id)
        self._poll_stop = threading.Event()
        self._poller_lock = threading.Lock()
        self._poller: threading.Thread | None = None

    # -- create -----------------------------------------------------------------

    def create(
        self,
        *,
        raw_key: str,
        alias: str,
        request: VideoRequest,
        reference_image: bytes | None,
        client_headers: Mapping[str, object],
    ) -> VideoJob:
        """Admit, reserve, and submit one video job.

        Args:
            raw_key: The caller's bearer key.
            alias: The public model alias.
            request: The decoded canonical request.
            reference_image: First-frame image bytes matching ``request.input_reference``.
            client_headers: Forwarded calling-application header values.

        Returns:
            The accepted job.

        Raises:
            NativeBridgeError: Any refusal, with its public error.
        """
        with self._pending_lock:
            if self._poll_stop.is_set():
                raise NativeBridgeError(_unavailable_error())
            self._active_creates += 1
        try:
            deadline = (
                time.monotonic()
                + VIDEO_JOB_SETTLEMENT_SECONDS
                + VIDEO_LEDGER_DEADLINE_SLACK_SECONDS
            )
            try:
                authorization = self._components.store.authorize_request(
                    raw_key=raw_key, alias=alias, request=request, deadline_monotonic=deadline
                )
                authorization = authorize_serving_model_chains(self._components, authorization)
                authorization = with_client_identity(authorization, client_headers)
            except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
                raise authority_error(exc) from exc
            require_unguarded_surface(self._guardrails, authorization, "video")
            try:
                self._ledger.accept_request(authorization=authorization)
            except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
                raise authority_error(exc) from exc
            return self._dispatch(authorization, request, reference_image)
        finally:
            with self._pending_lock:
                self._active_creates -= 1

    def _finish_request(
        self, authorization: AuthorizationSnapshot, failure: GatewayFailure
    ) -> None:
        """Retain a refused request's exact terminal write before attempting delivery."""
        try:
            self._request_settlements.finish(authorization, failure, certify_no_effects=True)
        except Exception as error:  # noqa: BLE001 - the caller already gets the refusal.
            _LOGGER.warning("video request settlement retained: %s", type(error).__name__)

    def _dispatch(
        self,
        authorization: AuthorizationSnapshot,
        request: VideoRequest,
        reference_image: bytes | None,
    ) -> VideoJob:
        """Route one accepted request and walk its video rungs until one accepts."""
        if not isinstance(authorization.target, DirectTarget):
            self._finish_request(authorization, _unsupported_failure())
            raise NativeBridgeError(not_a_video_model_error(authorization.alias))
        try:
            route = self._components.routes.resolve_direct(authorization)
            dispatchable = dispatchable_route_profiles(self._components.runtime_catalogs, route)
        except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
            self._finish_request(authorization, gateway_updating_failure())
            raise authority_error(exc) from exc
        candidates: list[tuple[int, GatewayWireProfile, VideoProviderClient, str, int]] = []
        claimed = False
        refusal: tuple[str, str] | None = None
        for index, (profile, _client) in zip(
            dispatchable.indexes, dispatchable.resolved_wires, strict=True
        ):
            deployment = route.deployments[index]
            dialect = video_dialect(profile)
            client = None if dialect is None else self._clients.get(dialect)
            if not deployment.gateway.capabilities.supports_video_generation or client is None:
                continue
            claimed = True
            rate = video_unit_rate(deployment.gateway.prices, request.resolution)
            reason = client.unsupported(request, profile.model_id)
            if reason is None and rate is None:
                reason = ("size", f"This model has no price for {request.resolution} video.")
            if reason is not None:
                refusal = refusal or reason
                continue
            assert dialect is not None and rate is not None
            candidates.append((index, profile, client, dialect, rate))
        if not claimed:
            if not dispatchable.indexes and dispatchable.dead:
                failure = dispatchable.dead[0].failure
                self._finish_request(authorization, failure)
                raise NativeBridgeError(public_failure_error(failure))
            self._finish_request(authorization, _unsupported_failure())
            raise NativeBridgeError(not_a_video_model_error(authorization.alias))
        if not candidates:
            param, message = refusal or ("model", "No rung of this model can render the request.")
            self._finish_request(
                authorization,
                GatewayFailure(
                    failure_class=GatewayFailureClass.INVALID_REQUEST,
                    safe_message="no video rung renders the requested parameters",
                ),
            )
            raise NativeBridgeError(
                OpenAIProtocolError(
                    status_code=400, code="unsupported_value", message=message, param=param
                )
            )
        route = select_route_deployments(route, tuple(entry[0] for entry in candidates))
        last = len(candidates) - 1
        for depth, (deployment, (_index, profile, client, dialect, rate)) in enumerate(
            zip(route.deployments, candidates, strict=True)
        ):
            reservation = video_ceiling_nano_usd(request, rate)
            try:
                attempt_id = self._ledger.start_attempt(
                    snapshot=route.snapshot,
                    deployment=deployment,
                    attempt_ordinal=depth,
                    route_depth=depth,
                    maximum_cost_nano_usd=reservation,
                    reserved_input_tokens=0,
                    reserved_output_tokens=0,
                )
            except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
                self._finish_request(
                    authorization,
                    GatewayFailure(
                        failure_class=GatewayFailureClass.QUOTA_EXCEEDED,
                        safe_message="the video reservation was refused",
                    ),
                )
                raise authority_error(exc) from exc
            stored = client.stores_output and self._outputs is not None
            job = self._new_job(
                authorization, request, deployment, attempt_id, dialect, rate, stored=stored
            )
            try:
                self._jobs.create(job=job)
            except Exception as exc:  # noqa: BLE001 - nothing reached the provider.
                self._settle_unsent(job, _internal_failure(), finalize=True)
                raise authority_error(exc) from exc
            upload_url: str | None = None
            if stored:
                assert self._outputs is not None
                try:
                    upload_url = self._outputs.upload_target(
                        video_id=job.video_id, organization_id=job.organization_id
                    ).url
                except Exception as exc:  # noqa: BLE001 - nothing reached the provider.
                    _LOGGER.exception("video %s upload target unavailable", job.video_id)
                    failover = depth < last
                    self._settle_unsent(job, _storage_failure(), finalize=not failover)
                    if failover:
                        continue
                    raise NativeBridgeError(public_failure_error(_storage_failure())) from exc
            try:
                provider_job_id = client.submit(
                    profile,
                    VideoSubmission(
                        request=request,
                        model_id=profile.model_id,
                        reference_image=reference_image,
                        upload_url=upload_url,
                    ),
                )
            except VideoSubmitRejected as rejected:
                failover = rejected.failure.failover_eligible and depth < last
                self._settle_unsent(job, rejected.failure, finalize=not failover)
                if failover:
                    continue
                raise NativeBridgeError(public_failure_error(rejected.failure)) from rejected
            except Exception as ambiguous:  # noqa: BLE001 - any other outcome may have landed.
                # VideoSubmitAmbiguous, or an unexpected client error: either
                # way the provider may hold the job, so it is never re-sent.
                failure = GatewayFailure(
                    failure_class=GatewayFailureClass.TRANSPORT,
                    safe_message="the video provider did not confirm the job; it was not retried",
                )
                self._settle_unknown(job, failure)
                raise NativeBridgeError(public_failure_error(failure)) from ambiguous
            accepted = job.model_copy(update={"provider_job_id": provider_job_id})
            if not self._jobs.save_open(job=accepted):  # pragma: no cover - fresh rows are open.
                _LOGGER.error("video %s was settled before its acceptance landed", job.video_id)
            return accepted
        raise AssertionError("the rung walk returns or raises")  # pragma: no cover

    def _new_job(
        self,
        authorization: AuthorizationSnapshot,
        request: VideoRequest,
        deployment: ExactModelDeployment,
        attempt_id: str,
        dialect: str,
        rate: int,
        *,
        stored: bool = False,
    ) -> VideoJob:
        """Build the durable intent row for one reserved rung.

        ``stored`` marks a job whose provider uploads to host storage.
        """
        now = self._clock()
        return VideoJob(
            video_id=f"video_{secrets.token_hex(16)}",
            organization_id=authorization.organization_id,
            identity_id=authorization.identity_id,
            key_id=authorization.virtual_key_id,
            alias=authorization.alias,
            request_id=authorization.request_id,
            attempt_id=attempt_id,
            deployment=deployment,
            alias_revision_id=authorization.alias_revision_id,
            catalog_sha256=authorization.catalog_sha256,
            dialect=dialect,
            seconds=request.seconds,
            size=request.size,
            variant=request.resolution,
            rate_nano_usd=rate,
            reserved_nano_usd=video_ceiling_nano_usd(request, rate) or 0,
            output_stored=stored,
            created_at=now,
            settle_by=now + timedelta(seconds=VIDEO_JOB_SETTLEMENT_SECONDS),
        )

    def _settle_unsent(self, job: VideoJob, failure: GatewayFailure, *, finalize: bool) -> None:
        """Retain an observed zero-cost rejection, including its exact failover boundary."""
        event = GatewayEvent(
            kind=GatewayEventKind.FAILED,
            sequence_number=0,
            failure=failure,
            usage=zero_units(job.variant),
        )
        self._settle_created(job, event, failure, finalize=finalize, settled_milli=0)

    def _settle_unknown(self, job: VideoJob, failure: GatewayFailure) -> None:
        """Retain a submit the provider may have accepted without inventing zero usage."""
        self._settle_created(
            job,
            GatewayEvent.for_accounting_recovery(failure=failure),
            failure,
            finalize=True,
            settled_milli=None,
        )

    def _settle_created(
        self,
        job: VideoJob,
        event: GatewayEvent | None,
        failure: GatewayFailure,
        *,
        finalize: bool,
        settled_milli: int | None,
    ) -> None:
        """Own a create-path terminal observation before either durable write starts."""
        closed = closed_job(job, failure, settled_milli=settled_milli, now=self._clock())
        closed = closed.model_copy(update={"deleted": True})
        with self._job_gate(job.video_id):
            with self._pending_lock:
                pending = self._pending.setdefault(
                    job.video_id, PendingSettlement(job, closed, event, failure, finalize)
                )
            self._deliver_settlement(pending)

    # -- reads ------------------------------------------------------------------

    def owned(self, *, organization_id: str, identity_id: str, video_id: str) -> VideoJob:
        """Return one owned job brought up to date with its provider.

        Raises:
            NativeBridgeError: 404 when absent, deleted, or owned by anyone else.
        """
        job = self._jobs.load(
            video_id=video_id, organization_id=organization_id, identity_id=identity_id
        )
        if job is None:
            raise NativeBridgeError(video_not_found_error(video_id))
        return self.refresh(job)

    def list_jobs(
        self,
        *,
        organization_id: str,
        identity_id: str,
        limit: int,
        after: str | None,
        order: ListOrder,
    ) -> tuple[list[VideoJob], bool]:
        """Return one page of owned jobs and whether another page exists."""
        bounded = max(1, min(limit, _LIST_LIMIT_MAXIMUM))
        page = self._jobs.list_jobs(
            organization_id=organization_id,
            identity_id=identity_id,
            limit=bounded + 1,
            after=after,
            order=order,
        )
        return page[:bounded], len(page) > bounded

    def delete(self, *, organization_id: str, identity_id: str, video_id: str) -> None:
        """Hide one owned, finished job.

        Raises:
            NativeBridgeError: 404 when not owned; 400 while the job still runs.
        """
        job = self.owned(
            organization_id=organization_id, identity_id=identity_id, video_id=video_id
        )
        if not job.settled:
            raise NativeBridgeError(
                OpenAIProtocolError(
                    status_code=400,
                    code="video_in_progress",
                    message="This video is still being generated; delete it once it finishes.",
                    param="video_id",
                )
            )
        if not self._jobs.mark_deleted(
            video_id=video_id, organization_id=organization_id, identity_id=identity_id
        ):
            raise NativeBridgeError(video_not_found_error(video_id))
        if job.output_stored and self._outputs is not None:
            try:
                self._outputs.delete(video_id=video_id, organization_id=organization_id)
            except Exception:  # noqa: BLE001 - the job is hidden; storage cleanup is best effort.
                _LOGGER.exception("video %s stored output could not be deleted", video_id)

    def content(
        self, *, organization_id: str, identity_id: str, video_id: str, variant: str
    ) -> VideoContentFetch:
        """Describe how the data plane downloads one owned, completed video.

        Raises:
            NativeBridgeError: 404 when not owned or expired, 400 when not ready
                or for an asset variant the provider does not render.
        """
        if variant != "video":
            raise NativeBridgeError(
                OpenAIProtocolError(
                    status_code=400,
                    code="unsupported_value",
                    message="Only variant=video is available for videos on this gateway.",
                    param="variant",
                )
            )
        job = self.owned(
            organization_id=organization_id, identity_id=identity_id, video_id=video_id
        )
        if job.status is not VideoStatus.COMPLETED or (
            job.content_url is None and not job.output_stored
        ):
            raise NativeBridgeError(
                OpenAIProtocolError(
                    status_code=400,
                    code="video_not_ready",
                    message=f"Video {video_id!r} is {job.status.value}, not yet downloadable.",
                    param="video_id",
                )
            )
        if job.expires_at is not None and self._clock() >= job.expires_at:
            raise NativeBridgeError(
                OpenAIProtocolError(
                    status_code=404,
                    code="video_expired",
                    message=f"Video {video_id!r} expired and is no longer downloadable.",
                    param="video_id",
                )
            )
        if job.output_stored:
            return self._stored_content(job)
        resolved = self._resolve(job)
        if resolved is None:
            raise NativeBridgeError(_unavailable_error())
        profile, client = resolved
        assert job.content_url is not None
        return client.content(profile, job.content_url)

    def _stored_content(self, job: VideoJob) -> VideoContentFetch:
        """Describe the download of one job's video from host storage.

        Raises:
            NativeBridgeError: 503 when the store is not installed or fails,
                404 when the stored object is gone.
        """
        if self._outputs is None:
            raise NativeBridgeError(_unavailable_error())
        try:
            fetch = self._outputs.download(
                video_id=job.video_id, organization_id=job.organization_id
            )
        except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
            _LOGGER.exception("video %s stored output unreadable", job.video_id)
            raise NativeBridgeError(_unavailable_error()) from exc
        if fetch is None:
            raise NativeBridgeError(
                OpenAIProtocolError(
                    status_code=404,
                    code="video_expired",
                    message=f"Video {job.video_id!r} is no longer downloadable.",
                    param="video_id",
                )
            )
        return fetch

    # -- polling and settlement -------------------------------------------------

    def poll_open(self) -> int:
        """Advance every open job one step; returns how many were examined."""
        self._retry_settlements()
        jobs = self._jobs.open_jobs(limit=_POLLER_BATCH)
        examined = 0
        for job in jobs:
            if self._poll_stop.is_set():
                break
            try:
                with self._job_gate(job.video_id, blocking=False) as acquired:
                    if acquired:
                        self._refresh(job)
                        examined += 1
            except Exception:  # noqa: BLE001 - one bad job never stalls the rest.
                _LOGGER.exception("video %s poll failed; will retry", job.video_id)
        return examined

    @contextmanager
    def _job_gate(self, video_id: str, *, blocking: bool = True) -> Iterator[bool]:
        """Retain one job gate through waiters; never hold the registry across I/O."""
        with self._pending_lock:
            gate = self._job_gates.setdefault(video_id, JobGate())
            gate.users += 1
        acquired = False
        try:
            acquired = gate.lock.acquire(blocking=blocking)
            yield acquired
        finally:
            if acquired:
                gate.lock.release()
            with self._pending_lock:
                gate.users -= 1
                if gate.users == 0:
                    del self._job_gates[video_id]

    def refresh(self, job: VideoJob) -> VideoJob:
        """Bring one job up to date with its provider, settling a terminal state.

        Args:
            job: The job as last read.

        Returns:
            The current job (re-read when another writer settled it first).
        """
        with self._job_gate(job.video_id):
            return self._refresh(job)

    def _refresh(self, job: VideoJob) -> VideoJob:
        """Re-read durable state under the job gate before constructing an observation."""
        with self._pending_lock:
            pending = self._pending.get(job.video_id)
        if pending is not None:
            return self._deliver_settlement(pending)
        current = self._jobs.load_any(video_id=job.video_id)
        if current is not None:
            job = current
        if job.settled or self._poll_stop.is_set():
            return job
        now = self._clock()
        if job.provider_job_id is None:
            if now - job.created_at < timedelta(seconds=DISPATCH_GRACE_SECONDS):
                return job
            failure = GatewayFailure(
                failure_class=GatewayFailureClass.INTERNAL,
                safe_message="the video dispatch was interrupted before the provider confirmed it",
            )
            return self._finalize(job, None, failure=failure, observed_zero=False)
        if now >= job.settle_by:
            failure = GatewayFailure(
                failure_class=GatewayFailureClass.TIMEOUT,
                safe_message="the provider did not finish the video in time",
            )
            return self._finalize(
                job, None, failure=failure, observed_zero=False, release_reservation=True
            )
        if job.last_polled_at is not None and now - job.last_polled_at < self._poll_interval:
            return job
        resolved = self._resolve(job)
        if resolved is None:
            return job
        try:
            snapshot = self._poll_snapshot(job, resolved)
        except Exception:  # noqa: BLE001 - transient; the next poll retries.
            _LOGGER.warning("video %s poll unavailable; will retry", job.video_id)
            return self._save(job.model_copy(update={"last_polled_at": now}))
        if snapshot.status in {VideoStatus.COMPLETED, VideoStatus.FAILED}:
            return self._finalize(job, snapshot, failure=None, observed_zero=True)
        return self._save(
            job.model_copy(
                update={
                    "status": snapshot.status,
                    "progress": max(job.progress, snapshot.progress),
                    "last_polled_at": now,
                }
            )
        )

    def _poll_snapshot(
        self, job: VideoJob, resolved: tuple[GatewayWireProfile, VideoProviderClient]
    ) -> VideoProviderSnapshot:
        """Read the same frozen provider job with the client's bounded transport."""
        profile, client = resolved
        assert job.provider_job_id is not None
        snapshot = client.poll(profile, job.provider_job_id, output_stored=job.output_stored)
        if snapshot.status is VideoStatus.COMPLETED and job.output_stored:
            snapshot = self._confirm_stored(job, snapshot)
        return snapshot

    def _confirm_stored(
        self, job: VideoJob, snapshot: VideoProviderSnapshot
    ) -> VideoProviderSnapshot:
        """Bill a stored-output completion only when the upload actually landed.

        A provider that reports ``done`` without the object in host storage
        failed to deliver, so the job fails and bills zero seconds. A store
        error propagates so the next poll retries.
        """
        if self._outputs is None:
            raise RuntimeError("the video output store is not installed")
        landed = self._outputs.download(video_id=job.video_id, organization_id=job.organization_id)
        if landed is not None:
            return snapshot
        return VideoProviderSnapshot(
            status=VideoStatus.FAILED,
            error_code="video_upload_missing",
            error_message="the provider finished but its video upload never arrived",
        )

    def _save(self, job: VideoJob) -> VideoJob:
        """Write one open job, re-reading when another writer settled it first."""
        if self._jobs.save_open(job=job):
            return job
        current = self._jobs.load_any(video_id=job.video_id)
        return job if current is None else current

    def _finalize(
        self,
        job: VideoJob,
        snapshot: VideoProviderSnapshot | None,
        *,
        failure: GatewayFailure | None,
        observed_zero: bool,
        release_reservation: bool = False,
    ) -> VideoJob:
        """Retain one terminal receipt until accounting and job publication both acknowledge it."""
        candidate = terminal_settlement(
            job,
            snapshot,
            failure=failure,
            observed_zero=observed_zero,
            release_reservation=release_reservation,
            now=self._clock(),
        )
        with self._pending_lock:
            pending = self._pending.setdefault(job.video_id, candidate)
        return self._deliver_settlement(pending)

    def _deliver_settlement(self, pending: PendingSettlement) -> VideoJob:
        """Deliver under the job gate; uncertainty never erases the exact terminal meter."""
        try:
            saved = self._settled_job_winner(pending)
            if saved is None:
                if not pending.ledger_written:
                    self._ledger.finish_attempt(
                        attempt_id=pending.job.attempt_id,
                        terminal_event=pending.event,
                        failure=pending.failure,
                        finalize_request=pending.finalize_request,
                    )
                    pending.ledger_written = True
                saved = self._save(pending.closed)
                if not saved.settled:
                    return pending.job
        except Exception as error:  # noqa: BLE001 - retain the exact record for fair retry.
            # Another process may have won after our first read. A settled job
            # is published only after its ledger write; an open or unreadable
            # row cannot establish that the original evidence is recoverable.
            try:
                saved = self._settled_job_winner(
                    pending,
                    recovery_superseded=isinstance(
                        error,
                        (
                            AccountingRecoverySupersededError,
                            AccountingRecoveryUnresolvedTerminalError,
                        ),
                    ),
                )
                if saved is None and isinstance(error, AccountingRecoverySupersededError):
                    return self._recover_superseded(pending)
            except Exception:  # noqa: BLE001 - a failed reread never releases ownership.
                saved = None
            if saved is None:
                _LOGGER.warning("video settlement retained: %s", type(error).__name__)
                return pending.job
        with self._pending_lock:
            if self._pending.get(pending.job.video_id) is pending:
                del self._pending[pending.job.video_id]
        return saved

    def _recover_superseded(self, pending: PendingSettlement) -> VideoJob:
        """Recover lost job publication only from a terminal read and exact ledger replay."""
        now = self._clock()
        if (
            self._poll_stop.is_set()
            or pending.job.provider_job_id is None
            or pending.event is None
            or not pending.event.accounting_release_reservation
            or (
                pending.recovery_polled_at is not None
                and now - pending.recovery_polled_at < self._poll_interval
            )
        ):
            return pending.job
        pending.recovery_polled_at = now
        current = self._jobs.load_any(video_id=pending.job.video_id)
        if current is None:
            return pending.job
        require_same_job(current, pending.job, frozen=True)
        resolved = self._resolve(pending.job)
        if resolved is None:
            return pending.job
        snapshot = self._poll_snapshot(pending.job, resolved)
        if not snapshot.terminal_replayable or snapshot.status not in {
            VideoStatus.COMPLETED,
            VideoStatus.FAILED,
        }:
            return pending.job
        recovered = terminal_settlement(
            pending.job,
            snapshot,
            failure=None,
            observed_zero=True,
            now=now,
        )
        recovered.require_ledger_replay = True
        with self._pending_lock:
            if self._pending.get(pending.job.video_id) is not pending:
                return pending.job
            self._pending[pending.job.video_id] = recovered
        return self._deliver_settlement(recovered)

    def _settled_job_winner(
        self, pending: PendingSettlement, *, recovery_superseded: bool = False
    ) -> VideoJob | None:
        """Require acknowledgment of retained meters and waivers before accepting a winner."""
        current = self._jobs.load_any(video_id=pending.job.video_id)
        if current is None or not current.settled:
            return None
        require_same_job(current, pending.job)
        if not pending.ledger_written and (
            pending.require_ledger_replay
            or (
                current.billed_seconds_milli is None
                and pending.closed.billed_seconds_milli is not None
            )
            or (
                pending.event is not None
                and pending.event.accounting_release_reservation
                and not recovery_superseded
            )
        ):
            # A published job proves neither an exact meter after unknown cleanup
            # nor a separate customer waiver. Accounting must acknowledge each.
            return None
        return current

    def _retry_settlements(self) -> None:
        """Give each organization a bounded turn for delivery and superseded-job recovery."""
        self._request_settlements.retry(_POLLER_BATCH)
        with self._pending_lock:
            selected = self._recovery.select(list(self._pending.values()), _POLLER_BATCH)
        for pending in selected:
            with self._job_gate(pending.job.video_id, blocking=False) as acquired:
                if acquired:
                    with self._pending_lock:
                        current = self._pending.get(pending.job.video_id)
                    if current is pending:
                        self._deliver_settlement(pending)

    def _resolve(self, job: VideoJob) -> tuple[GatewayWireProfile, VideoProviderClient] | None:
        """Re-resolve the job's frozen deployment on a currently loaded catalog.

        The authorizing revision's catalog is preferred; after a reload any
        loaded catalog that still resolves the identical deployment serves.
        A deployment no loaded catalog resolves cannot be polled now, and the
        settlement bound eventually finalizes its job.
        """
        catalogs = self._components.runtime_catalogs
        preferred = (job.alias_revision_id, job.catalog_sha256)
        keys = [preferred, *(key for key in catalogs if key != preferred)]
        client = self._clients.get(job.dialect)
        if client is None:
            return None
        for key in keys:
            catalog = catalogs.get(key)
            if catalog is None:
                continue
            try:
                profile, _wire = _resolve_deployment_profile(catalog, job.deployment)
            except Exception:  # noqa: BLE001 - this catalog cannot serve the deployment.
                continue
            if video_dialect(profile) == job.dialect:
                return profile, client
        return None

    # -- background poller ------------------------------------------------------

    def start_poller(self, *, interval_seconds: float = POLL_INTERVAL_SECONDS) -> threading.Thread:
        """Start the daemon thread that drives every open job to settlement.

        Args:
            interval_seconds: Sleep between passes.

        Returns:
            The started thread.
        """
        if not 0 < interval_seconds <= 300:
            raise ValueError("video poll interval must be positive and at most 300 seconds")

        def loop() -> None:
            """Stop new polls at shutdown but retain ownership of known terminal writes."""
            while True:
                closing = self._poll_stop.wait(interval_seconds)
                try:
                    if closing:
                        self._retry_settlements()
                        if self._recovery_drained():
                            return
                    else:
                        self.poll_open()
                except Exception:  # noqa: BLE001 - the poller must outlive store hiccups.
                    _LOGGER.exception("video poller pass failed")
                if closing:
                    time.sleep(min(interval_seconds, 0.1))

        with self._poller_lock:
            if self._poller is None:
                self._poller = threading.Thread(
                    target=loop, name="gateway-video-poller", daemon=True
                )
                self._poller.start()
            return self._poller

    def close(self, timeout_seconds: float = 10.0) -> bool:
        """Join provider polling and exact-meter recovery before releasing the ledger."""
        if not 0 <= timeout_seconds <= 300:
            raise ValueError("video shutdown timeout must be between zero and 300 seconds")
        with self._pending_lock:
            self._poll_stop.set()
        thread = self.start_poller()
        thread.join(timeout_seconds)
        return not thread.is_alive() and self._recovery_drained()

    def _recovery_drained(self) -> bool:
        """Keep transport ownership for creates, active refreshes and retained terminal writes."""
        with self._pending_lock:
            active = self._active_creates or self._pending or self._job_gates
        return not active and not self._request_settlements.outstanding

    @property
    def idle(self) -> bool:
        """Permit idle retirement only after local owners and durable open jobs finish.

        Explicit process shutdown may stop polling and leave durable jobs for
        restart. Idle host eviction must keep their poller alive instead.
        """
        return self._recovery_drained() and not self._jobs.open_jobs(limit=1)


def _unsupported_failure() -> GatewayFailure:
    """The terminal failure for an alias that does not generate video."""
    return GatewayFailure(
        failure_class=GatewayFailureClass.UNSUPPORTED_CAPABILITY,
        safe_message="the model alias does not generate video",
    )


def _storage_failure() -> GatewayFailure:
    """The failure for a rung whose host upload destination could not be minted."""
    return GatewayFailure(
        failure_class=GatewayFailureClass.INTERNAL,
        safe_message="the video output storage is unavailable",
        failover_eligible=True,
    )


def _unavailable_error() -> OpenAIProtocolError:
    """The 503 for a video whose serving provider or storage is unreachable now."""
    return OpenAIProtocolError(
        status_code=503,
        code="unavailable_route",
        message="The provider serving this video is not available; retry shortly.",
        error_type="api_error",
    )


def _internal_failure() -> GatewayFailure:
    """The terminal failure for a job row that could not be persisted."""
    return GatewayFailure(
        failure_class=GatewayFailureClass.INTERNAL,
        safe_message="the video job could not be recorded",
    )
