"""In-process tests for the video engine over real authority, ledger, and job store.

A seeded root serves a certified two-rung Veo pool (``veo``) and an xAI alias
(``grok-video``); the engine runs in process with an injected clock and
``httpx.MockTransport`` providers, so failover, the settlement bound, a lost
dispatch, and racing settlements are driven deterministically and read back
from the SQLite ledger.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from exp.common.core.artifacts import sha256_json
from exp.common.models import (
    BilledUnitKind,
    ConnectionConfig,
    GatewayDeploymentCapabilities,
    GatewayEquivalenceCertification,
    GatewayTokenPrices,
    GatewayUnitPrices,
    ModelCapabilities,
    NormalizedGatewayCatalog,
)
from exp.runtime.gateway.catalog_authority import upsert_connection, upsert_singleton_deployment
from exp.runtime.gateway.lifecycle import LocalGatewayComponents, load_gateway_components
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.native_accounting_errors import NativeBridgeError
from exp.runtime.gateway.stream_contracts import GatewayEvent, GatewayFailure
from exp.runtime.gateway.tests.certified_pool_fixture_test import upsert_certified_pool
from exp.runtime.gateway.video.contracts import (
    VIDEO_JOB_SETTLEMENT_SECONDS,
    VideoJob,
    VideoRequest,
    VideoStatus,
)
from exp.runtime.gateway.video.engine import DISPATCH_GRACE_SECONDS, VideoEngine
from exp.runtime.gateway.video.output import InMemoryVideoOutputStore, VideoUploadTarget
from exp.runtime.gateway.video.providers import VeoVideoClient, XaiVideoClient

_KEY = "provider-secret-canary"
_VEO_RATE = 100_000_000
_XAI_RATE = 50_000_000
_CERTIFIED = GatewayEquivalenceCertification(
    certification_id="video-fixture",
    provenance="test-only same-model fixture",
    evidence_sha256=sha256_json({"fixture": "veo"}),
    certified_at=datetime(2026, 10, 18, tzinfo=UTC),
)


@dataclass
class _Provider:
    """Scripted provider answers shared by both mock transports.

    Attributes:
        submits: Submit outcomes consumed in order; an int is an HTTP status,
            an exception class is raised, and ``None`` accepts.
        done: Whether operation polls report completion.
        poll_status: HTTP status for polls.
        xai_duration: The rendered duration xAI reports, in seconds.
        posts: Every submit URL received.
        bodies: Every submit body received, decoded.
    """

    submits: list[int | type[Exception] | None] = field(default_factory=list)
    done: bool = True
    poll_status: int = 200
    xai_duration: int = 5
    posts: list[str] = field(default_factory=list)
    bodies: list[dict[str, object]] = field(default_factory=list)

    def handle(self, request: httpx.Request) -> httpx.Response:
        """Answer one provider request from the script."""
        if request.method == "POST":
            self.posts.append(str(request.url))
            self.bodies.append(json.loads(request.content))
            outcome = self.submits.pop(0) if self.submits else None
            if isinstance(outcome, type):
                raise outcome("scripted transport failure")
            if isinstance(outcome, int):
                return httpx.Response(outcome, json={"error": {"message": f"scripted {outcome}"}})
            if request.url.host == "api.x.ai":
                return httpx.Response(200, json={"request_id": f"xai-{len(self.posts)}"})
            model = request.url.path.split("/models/")[1].split(":")[0]
            return httpx.Response(
                200, json={"name": f"models/{model}/operations/op-{len(self.posts)}"}
            )
        if self.poll_status != 200:
            return httpx.Response(self.poll_status, json={"error": {"message": "busy"}})
        if request.url.host == "api.x.ai":
            if not self.done:
                return httpx.Response(200, json={"status": "pending"})
            video = {"url": "https://vidgen.x.ai/v.mp4", "duration": self.xai_duration}
            return httpx.Response(200, json={"status": "done", "video": video})
        if not self.done:
            return httpx.Response(200, json={"done": False})
        samples = [{"video": {"uri": "https://generativelanguage.googleapis.com/v1beta/files/x"}}]
        response = {"generateVideoResponse": {"generatedSamples": samples}}
        return httpx.Response(200, json={"done": True, "response": response})


@dataclass
class _Clock:
    """A settable wall clock."""

    now: datetime = field(default_factory=lambda: datetime(2026, 10, 18, 12, tzinfo=UTC))

    def __call__(self) -> datetime:
        """Return the current fake time."""
        return self.now


@dataclass(frozen=True)
class _Setup:
    """The seeded components, key, engine, and its scripted provider."""

    components: LocalGatewayComponents
    raw_key: str
    engine: VideoEngine
    provider: _Provider
    clock: _Clock
    database: Path


def _seed_deployment(
    root: Path,
    alias: str,
    *,
    connection: str,
    model: str,
    prices: GatewayTokenPrices,
    supports_video: bool = True,
) -> tuple[NormalizedGatewayCatalog, Path]:
    """Register one deployment and return its normalized catalog and snapshot."""
    normalized, snapshot, _changed = upsert_singleton_deployment(
        root,
        deployment_alias=alias,
        connection_name=connection,
        provider_model=model,
        exact_model_id=f"{model}-exact",
        revision=None,
        capabilities=ModelCapabilities(),
        gateway_capabilities=GatewayDeploymentCapabilities(
            supports_video_generation=supports_video
        ),
        prices=prices,
        pricing_source="test-fixture",
        replace=False,
    )
    return normalized, snapshot


def _activate(
    manager: GatewayManagement,
    alias: str,
    pool: str,
    normalized: NormalizedGatewayCatalog,
    snapshot: Path,
) -> None:
    """Activate one alias over a pool and grant it to the default identity."""
    manager.activate_direct_alias(
        alias_id=alias,
        alias_name=alias,
        revision_id=f"revision-{alias}",
        pool_id=pool,
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.add_grant(identity_id="default", alias_id=alias)


@pytest.fixture(name="setup")
def _setup(tmp_path: Path) -> Iterator[_Setup]:
    """Seed a two-rung Veo pool, an xAI alias, and an unpriced video claim."""
    root = tmp_path / "root"
    manager = GatewayManagement(root)
    manager.initialize()
    manager.create_identity(identity_id="default", display_name="Default")
    upsert_connection(
        root,
        name="gemini-main",
        connection=ConnectionConfig(provider="gemini", api_key_env="TEST_PROVIDER_KEY"),
        replace=False,
    )
    upsert_connection(
        root,
        name="xai-main",
        connection=ConnectionConfig(
            provider="openai-compatible",
            base_url="https://api.x.ai/v1",
            api_key_env="TEST_PROVIDER_KEY",
        ),
        replace=False,
    )
    veo_card = GatewayTokenPrices(
        units=GatewayUnitPrices(kind=BilledUnitKind.VIDEO_SECOND, rates={"720p": _VEO_RATE})
    )
    for index in range(2):
        normalized, snapshot = _seed_deployment(
            root,
            f"veo-{index}",
            connection="gemini-main",
            model="veo-3.1-fast-generate-preview",
            prices=veo_card,
        )
    normalized, snapshot, _changed = upsert_certified_pool(
        root,
        pool_id="veo-pool",
        exact_model_id="veo-3.1-fast-generate-preview-exact",
        deployment_aliases=("veo-0", "veo-1"),
        certification=_CERTIFIED,
        expected_catalog_sha256=normalized.identity_sha256(),
        replace=False,
    )
    _activate(manager, "veo", "veo-pool", normalized, snapshot)
    normalized, snapshot = _seed_deployment(
        root,
        "grok-video",
        connection="xai-main",
        model="grok-imagine-video",
        prices=GatewayTokenPrices(
            units=GatewayUnitPrices(kind=BilledUnitKind.VIDEO_SECOND, rates={"": _XAI_RATE})
        ),
    )
    _activate(manager, "grok-video", "grok-video", normalized, snapshot)
    normalized, snapshot = _seed_deployment(
        root,
        "veo-unpriced",
        connection="gemini-main",
        model="veo-3.1-generate-preview",
        prices=GatewayTokenPrices(
            units=GatewayUnitPrices(kind=BilledUnitKind.CHARACTER, rates={"": 1})
        ),
    )
    _activate(manager, "veo-unpriced", "veo-unpriced", normalized, snapshot)
    raw_key = manager.issue_key(identity_id="default", key_id="key-one").raw_key
    components = load_gateway_components(root, environment={"TEST_PROVIDER_KEY": _KEY})
    provider = _Provider()
    clock = _Clock()

    def http() -> httpx.Client:
        """Build one client over the scripted transport."""
        return httpx.Client(transport=httpx.MockTransport(provider.handle))

    assert components.video_jobs is not None
    engine = VideoEngine(
        components=components,
        ledger=components.ledger,
        jobs=components.video_jobs,
        clients={"veo": VeoVideoClient(http=http), "xai_videos": XaiVideoClient(http=http)},
        clock=clock,
        poll_interval_seconds=0,
    )
    try:
        yield _Setup(components, raw_key, engine, provider, clock, manager.database_path)
    finally:
        assert engine.close(5)
        components.write_ledger.close()


def _create(setup: _Setup, alias: str = "veo", seconds: int = 8) -> VideoJob:
    """Create one 720p landscape job on an alias."""
    return setup.engine.create(
        raw_key=setup.raw_key,
        alias=alias,
        request=VideoRequest(prompt="a test clip", seconds=seconds, size="1280x720"),
        reference_image=None,
        client_headers={},
    )


def _attempts(setup: _Setup) -> list[sqlite3.Row]:
    """Read every ledger attempt in dispatch order."""
    with sqlite3.connect(setup.database) as connection:
        connection.row_factory = sqlite3.Row
        return connection.execute(
            """
            SELECT state, budget_reserved_nano_usd, budget_settled_nano_usd,
                   estimated_cost_nano_usd, billed_quantity_milli, usage_source, deployment_id
            FROM gateway_attempts ORDER BY rowid
            """
        ).fetchall()


def _status_of(error: NativeBridgeError) -> int:
    """Read the public status code a bridge error carries."""
    status = json.loads(error.public_error_json)["status_code"]
    assert isinstance(status, int)
    return status


def test_a_throttled_rung_fails_over_before_acceptance(setup: _Setup) -> None:
    """A 429 at submit settles that rung at zero and the next rung accepts the job."""
    setup.provider.submits = [429, None]
    job = _create(setup)
    first, second = _attempts(setup)
    assert first["state"] == "failed"
    assert first["budget_reserved_nano_usd"] == 8 * _VEO_RATE
    assert first["budget_settled_nano_usd"] == 0
    assert first["billed_quantity_milli"] == 0
    assert second["state"] == "dispatched"
    assert second["budget_reserved_nano_usd"] == 8 * _VEO_RATE
    assert job.deployment.deployment_id == second["deployment_id"]
    assert len(setup.provider.posts) == 2
    page, has_more = setup.engine.list_jobs(
        organization_id=job.organization_id,
        identity_id=job.identity_id,
        limit=10,
        after=None,
        order="desc",
    )
    assert [listed.video_id for listed in page] == [job.video_id]
    assert not has_more


def test_a_request_rejection_is_the_callers_and_never_fails_over(setup: _Setup) -> None:
    """A provider 400 is answered to the caller at once; the second rung is not tried."""
    setup.provider.submits = [400, None]
    with pytest.raises(NativeBridgeError) as refused:
        _create(setup)
    assert _status_of(refused.value) == 400
    assert "scripted 400" in refused.value.public_error_json
    (attempt,) = _attempts(setup)
    assert attempt["budget_settled_nano_usd"] == 0
    assert len(setup.provider.posts) == 1


def test_an_ambiguous_submit_settles_conservatively_and_is_never_resent(setup: _Setup) -> None:
    """A lost submit answer may have created a job: no failover, the hold is kept."""
    setup.provider.submits = [httpx.ReadTimeout, None]
    with pytest.raises(NativeBridgeError):
        _create(setup)
    (attempt,) = _attempts(setup)
    assert attempt["state"] == "failed"
    assert attempt["usage_source"] == "unknown"
    assert attempt["budget_settled_nano_usd"] == 8 * _VEO_RATE
    assert len(setup.provider.posts) == 1


def test_a_job_still_running_at_its_bound_finalizes_at_zero(setup: _Setup) -> None:
    """The settlement bound releases a provider job that never finishes."""
    setup.provider.done = False
    job = _create(setup)
    assert setup.engine.refresh(job).status is VideoStatus.IN_PROGRESS
    setup.clock.now += timedelta(seconds=VIDEO_JOB_SETTLEMENT_SECONDS)
    assert setup.engine.poll_open() == 1
    jobs = setup.components.video_jobs
    assert jobs is not None
    current = jobs.load_any(video_id=job.video_id)
    assert current is not None
    assert current.status is VideoStatus.FAILED
    assert current.error_code == "timeout"
    assert current.settled
    (attempt,) = _attempts(setup)
    assert attempt["budget_settled_nano_usd"] == 0
    assert setup.engine.poll_open() == 0


def test_a_lost_dispatch_settles_conservatively_after_the_grace(setup: _Setup) -> None:
    """A job row that never recorded its provider handle closes, keeping the hold."""
    job = _create(setup)
    jobs = setup.components.video_jobs
    assert jobs is not None
    assert jobs.save_open(job=job.model_copy(update={"provider_job_id": None}))
    lost = jobs.load_any(video_id=job.video_id)
    assert lost is not None
    assert setup.engine.refresh(lost).status is VideoStatus.QUEUED
    setup.clock.now += timedelta(seconds=DISPATCH_GRACE_SECONDS)
    closed = setup.engine.refresh(lost)
    assert closed.status is VideoStatus.FAILED
    assert closed.settled
    (attempt,) = _attempts(setup)
    assert attempt["usage_source"] == "unknown"
    assert attempt["budget_settled_nano_usd"] == 8 * _VEO_RATE


def test_a_transient_poll_failure_keeps_the_job_open(setup: _Setup) -> None:
    """A provider 503 while polling changes nothing but the poll time."""
    job = _create(setup)
    setup.provider.poll_status = 503
    polled = setup.engine.refresh(job)
    assert not polled.settled
    assert polled.last_polled_at == setup.clock.now
    assert _attempts(setup)[0]["state"] == "dispatched"


def test_racing_settlements_bill_once(setup: _Setup) -> None:
    """Two observers of the same terminal state settle the attempt exactly once."""
    job = _create(setup)
    first = setup.engine.refresh(job)
    second = setup.engine.refresh(job)  # A stale snapshot from another process.
    assert first.settled and second.settled
    assert first.billed_seconds_milli == second.billed_seconds_milli == 8_000
    (attempt,) = _attempts(setup)
    assert attempt["estimated_cost_nano_usd"] == 8 * _VEO_RATE


def test_failed_terminal_write_keeps_the_original_meter_through_shutdown(setup: _Setup) -> None:
    """A later timeout cannot replace a completed render retained after a ledger failure."""
    job = _create(setup, alias="grok-video", seconds=6)
    finish = setup.components.ledger.finish_attempt
    observed_at = setup.clock.now
    with patch.object(
        setup.components.ledger, "finish_attempt", side_effect=RuntimeError("offline")
    ) as failed:
        assert not setup.engine.refresh(job).settled
    original = failed.call_args.kwargs
    setup.clock.now += timedelta(seconds=VIDEO_JOB_SETTLEMENT_SECONDS + 1)
    setup.provider.poll_status = 503
    with patch.object(setup.components.ledger, "finish_attempt", wraps=finish) as recovered:
        assert setup.engine.close(5)
    assert recovered.call_args.kwargs == original
    jobs = setup.components.video_jobs
    assert jobs is not None
    current = jobs.load_any(video_id=job.video_id)
    assert current is not None and current.settled
    assert current.status is VideoStatus.COMPLETED
    assert current.completed_at == observed_at
    assert current.billed_seconds_milli == 5_000
    assert _attempts(setup)[0]["budget_settled_nano_usd"] == 5 * _XAI_RATE


@pytest.mark.parametrize("outcome", [400, httpx.ReadTimeout])
def test_failed_create_settlement_retains_exact_rejection_or_unknown_usage(
    setup: _Setup, outcome: int | type[Exception]
) -> None:
    """Create failures retain their original known-zero or unknown terminal evidence."""
    setup.provider.submits = [outcome]
    finish = setup.components.ledger.finish_attempt
    with patch.object(
        setup.components.ledger, "finish_attempt", side_effect=RuntimeError("offline")
    ) as failed:
        with pytest.raises(NativeBridgeError):
            _create(setup, alias="grok-video", seconds=6)
        assert not setup.engine.close(0.01)
    original = failed.call_args.kwargs
    with patch.object(setup.components.ledger, "finish_attempt", wraps=finish) as recovered:
        assert setup.engine.close(5)
    assert recovered.call_args.kwargs == original
    assert len(setup.provider.posts) == 1
    (attempt,) = _attempts(setup)
    if isinstance(outcome, int):
        assert attempt["billed_quantity_milli"] == 0
        assert attempt["budget_settled_nano_usd"] == 0
    else:
        assert attempt["usage_source"] == "unknown"
        assert attempt["budget_settled_nano_usd"] == 6 * _XAI_RATE


def test_failed_create_request_finish_is_owned_by_shutdown(setup: _Setup) -> None:
    """An admitted unsupported duration retains its no-effects certificate until committed."""
    finish = setup.components.ledger.finish_request
    with patch.object(
        setup.components.ledger, "finish_request", side_effect=RuntimeError("offline")
    ) as failed:
        with pytest.raises(NativeBridgeError):
            _create(setup, alias="grok-video", seconds=16)
        assert not setup.engine.close(0.01)
    original = failed.call_args.kwargs
    with patch.object(setup.components.ledger, "finish_request", wraps=finish) as recovered:
        assert setup.engine.close(5)
    assert recovered.call_args.kwargs == original
    assert original["certify_no_effects"] is True
    assert not _attempts(setup) and not setup.provider.posts
    with sqlite3.connect(setup.database) as connection:
        assert connection.execute("SELECT state FROM gateway_requests").fetchall() == [("failed",)]


def test_retained_rejection_cannot_finalize_the_next_video_rung(setup: _Setup) -> None:
    """A delayed known-zero first rung keeps its non-final settlement across failover."""
    setup.provider.submits = [503, None]
    finish = setup.components.ledger.finish_attempt
    with patch.object(
        setup.components.ledger, "finish_attempt", side_effect=RuntimeError("offline")
    ):
        job = _create(setup)
    assert len(setup.provider.posts) == 2
    with patch.object(setup.components.ledger, "finish_attempt", wraps=finish) as recovered:
        setup.engine._retry_settlements()
        assert recovered.call_args.kwargs["finalize_request"] is False
    first, second = _attempts(setup)
    assert first["budget_settled_nano_usd"] == 0
    assert second["state"] == "dispatched"
    assert setup.engine.refresh(job).settled
    assert _attempts(setup)[1]["budget_settled_nano_usd"] == 8 * _VEO_RATE


def test_shutdown_retains_a_create_before_its_terminal_write(setup: _Setup) -> None:
    """An active provider submit cannot outlive the recovery owner that receives its failure."""
    entered, release = threading.Event(), threading.Event()
    handle = setup.provider.handle
    setup.provider.submits = [400]
    errors: list[NativeBridgeError] = []

    def blocked_submit(request: httpx.Request) -> httpx.Response:
        """Pause the provider response until close has attempted its bounded join."""
        entered.set()
        assert release.wait(5)
        return handle(request)

    def create() -> None:
        """Finish a rejected create through the real ledger and job store."""
        try:
            _create(setup, alias="grok-video", seconds=6)
        except NativeBridgeError as error:
            errors.append(error)

    with patch.object(setup.provider, "handle", side_effect=blocked_submit):
        worker = threading.Thread(target=create)
        worker.start()
        try:
            assert entered.wait(5)
            assert not setup.engine.close(0.01)
        finally:
            release.set()
            worker.join(5)
        assert not worker.is_alive()
    assert len(errors) == 1
    assert setup.engine.close(5)
    assert _attempts(setup)[0]["budget_settled_nano_usd"] == 0


def test_job_store_retry_does_not_repeat_acknowledged_accounting(setup: _Setup) -> None:
    """Once accounting succeeds, only the exact terminal job write is retried."""
    job = _create(setup)
    jobs = setup.components.video_jobs
    assert jobs is not None
    with patch.object(
        setup.components.ledger, "finish_attempt", wraps=setup.components.ledger.finish_attempt
    ) as finish:
        with patch.object(jobs, "save_open", side_effect=RuntimeError("offline")):
            assert not setup.engine.refresh(job).settled
        assert setup.engine.close(5)
        assert finish.call_count == 1
    assert _attempts(setup)[0]["budget_settled_nano_usd"] == 8 * _VEO_RATE


@pytest.mark.parametrize("delay_winner_job", [False, True])
def test_another_engine_settlement_winner_releases_the_losing_retry_owner(
    setup: _Setup, delay_winner_job: bool
) -> None:
    """A conflicting timeout drains only after the completed winner is durably published."""
    job = _create(setup, alias="grok-video", seconds=6)
    other = _stored_engine(setup, InMemoryVideoOutputStore())
    jobs = setup.components.video_jobs
    assert jobs is not None
    finish = setup.components.ledger.finish_attempt
    observed_at = setup.clock.now
    setup.clock.now += timedelta(seconds=VIDEO_JOB_SETTLEMENT_SECONDS + 1)
    winner_started = False

    def finish_after_winner(
        *,
        attempt_id: str,
        terminal_event: GatewayEvent | None,
        failure: GatewayFailure | None,
        finalize_request: bool,
    ) -> None:
        """Commit another engine's completion after the loser's initial durable read."""
        nonlocal winner_started
        if not winner_started:
            winner_started = True
            setup.clock.now = observed_at
            if delay_winner_job:
                with patch.object(jobs, "save_open", side_effect=RuntimeError("offline")):
                    assert not other.refresh(job).settled
            else:
                assert other.refresh(job).settled
        finish(
            attempt_id=attempt_id,
            terminal_event=terminal_event,
            failure=failure,
            finalize_request=finalize_request,
        )

    with patch.object(setup.components.ledger, "finish_attempt", side_effect=finish_after_winner):
        result = setup.engine.refresh(job)
    if delay_winner_job:
        assert not result.settled
        assert not setup.engine.close(0.01)
    else:
        assert result.settled and result.status is VideoStatus.COMPLETED
        assert result.billed_seconds_milli == 5_000
    assert other.close(5)
    assert setup.engine.close(5)
    current = jobs.load_any(video_id=job.video_id)
    assert current is not None and current.settled
    assert current.status is VideoStatus.COMPLETED
    assert current.billed_seconds_milli == 5_000
    (attempt,) = _attempts(setup)
    assert attempt["state"] == "completed"
    assert attempt["billed_quantity_milli"] == 5_000
    assert attempt["budget_settled_nano_usd"] == 5 * _XAI_RATE


def test_stale_waiter_observes_the_settled_winner_after_deadline(setup: _Setup) -> None:
    """Concurrent stale readers cannot replace completion with a later timeout."""
    job = _create(setup)
    entered = threading.Event()
    release = threading.Event()
    finish = setup.components.ledger.finish_attempt
    results: list[VideoJob] = []

    def blocked_finish(
        *,
        attempt_id: str,
        terminal_event: GatewayEvent | None,
        failure: GatewayFailure | None,
        finalize_request: bool,
    ) -> None:
        """Hold the first completed observation while a stale reader starts waiting."""
        entered.set()
        assert release.wait(5)
        finish(
            attempt_id=attempt_id,
            terminal_event=terminal_event,
            failure=failure,
            finalize_request=finalize_request,
        )

    with patch.object(
        setup.components.ledger, "finish_attempt", side_effect=blocked_finish
    ) as calls:
        first = threading.Thread(target=lambda: results.append(setup.engine.refresh(job)))
        second = threading.Thread(target=lambda: results.append(setup.engine.refresh(job)))
        first.start()
        try:
            assert entered.wait(5)
            second.start()
            setup.clock.now += timedelta(seconds=VIDEO_JOB_SETTLEMENT_SECONDS + 1)
        finally:
            release.set()
            first.join(5)
            if second.ident is not None:
                second.join(5)
        assert calls.call_count == 1
    assert len(results) == 2
    assert all(result.settled and result.status is VideoStatus.COMPLETED for result in results)
    assert setup.engine.close(5)


def test_shutdown_joins_active_poll_and_does_not_start_the_next_job(setup: _Setup) -> None:
    """The owned poller completes its current observation and stops the open-job batch."""
    _create(setup)
    _create(setup)
    entered = threading.Event()
    release = threading.Event()
    handle = setup.provider.handle
    polls: list[str] = []

    def blocked_poll(request: httpx.Request) -> httpx.Response:
        """Retain one in-progress provider response across a bounded close."""
        polls.append(str(request.url))
        entered.set()
        assert release.wait(5)
        return handle(request)

    with patch.object(setup.provider, "handle", side_effect=blocked_poll):
        thread = setup.engine.start_poller(interval_seconds=0.01)
        try:
            assert entered.wait(5)
            assert not setup.engine.close(0.01)
        finally:
            release.set()
        assert setup.engine.close(5)
        assert not thread.is_alive()
    assert len(polls) == 1
    assert sorted(row["state"] for row in _attempts(setup)) == ["completed", "dispatched"]


def test_xai_bills_the_rendered_seconds_never_more_than_requested(setup: _Setup) -> None:
    """xAI reports the rendered length: a shorter render bills only its seconds."""
    job = _create(setup, alias="grok-video", seconds=6)
    assert job.dialect == "xai_videos"
    assert setup.provider.posts == ["https://api.x.ai/v1/videos/generations"]
    assert _attempts(setup)[0]["budget_reserved_nano_usd"] == 6 * _XAI_RATE
    settled = setup.engine.refresh(job)
    assert settled.billed_seconds_milli == 5_000
    assert _attempts(setup)[0]["estimated_cost_nano_usd"] == 5 * _XAI_RATE
    fetch = setup.engine.content(
        organization_id=job.organization_id,
        identity_id=job.identity_id,
        video_id=job.video_id,
        variant="video",
    )
    assert fetch.url == "https://vidgen.x.ai/v.mp4"
    assert fetch.headers == {}


def test_a_video_claim_without_a_video_second_card_is_excluded(setup: _Setup) -> None:
    """Startup excludes an alias whose video claim could never bill."""
    excluded = dict(setup.components.unavailable_aliases)
    assert "veo-unpriced" in excluded
    assert "video_second unit card" in excluded["veo-unpriced"]


@pytest.mark.parametrize(
    ("mutate", "status"),
    [
        (lambda job: job.model_copy(update={"status": VideoStatus.IN_PROGRESS}), 400),
        (lambda job: job, 400),
    ],
)
def test_content_requires_a_completed_job(
    setup: _Setup, mutate: Callable[[VideoJob], VideoJob], status: int
) -> None:
    """A job that is not completed has nothing to download yet."""
    setup.provider.done = False
    job = mutate(_create(setup))
    with pytest.raises(NativeBridgeError) as refused:
        setup.engine.content(
            organization_id=job.organization_id,
            identity_id=job.identity_id,
            video_id=job.video_id,
            variant="video",
        )
    assert _status_of(refused.value) == status


def test_only_the_video_variant_is_served(setup: _Setup) -> None:
    """Thumbnail and spritesheet variants are refused by name."""
    job = _create(setup)
    with pytest.raises(NativeBridgeError) as refused:
        setup.engine.content(
            organization_id=job.organization_id,
            identity_id=job.identity_id,
            video_id=job.video_id,
            variant="thumbnail",
        )
    assert json.loads(refused.value.public_error_json)["param"] == "variant"


class _BrokenOutputs(InMemoryVideoOutputStore):
    """An output store whose upload destinations cannot be minted."""

    def upload_target(self, *, video_id: str, organization_id: str) -> VideoUploadTarget:
        """Fail every mint."""
        raise RuntimeError("bucket unavailable")


def _stored_engine(setup: _Setup, outputs: InMemoryVideoOutputStore) -> VideoEngine:
    """A second engine over the same seams with host output storage installed."""

    def http() -> httpx.Client:
        """Build one client over the scripted transport."""
        return httpx.Client(transport=httpx.MockTransport(setup.provider.handle))

    assert setup.components.video_jobs is not None
    return VideoEngine(
        components=setup.components,
        ledger=setup.components.ledger,
        jobs=setup.components.video_jobs,
        clients={"veo": VeoVideoClient(http=http), "xai_videos": XaiVideoClient(http=http)},
        clock=setup.clock,
        poll_interval_seconds=0,
        outputs=outputs,
    )


def _create_on(engine: VideoEngine, setup: _Setup, alias: str, seconds: int) -> VideoJob:
    """Create one 720p landscape job through ``engine``."""
    return engine.create(
        raw_key=setup.raw_key,
        alias=alias,
        request=VideoRequest(prompt="a test clip", seconds=seconds, size="1280x720"),
        reference_image=None,
        client_headers={},
    )


def test_an_xai_job_uploads_to_host_storage_and_is_served_from_it(setup: _Setup) -> None:
    """With a store installed, xAI gets ``output.upload_url`` and content streams from the store."""
    outputs = InMemoryVideoOutputStore()
    engine = _stored_engine(setup, outputs)
    job = _create_on(engine, setup, "grok-video", 6)
    assert job.output_stored and outputs.minted == [job.video_id]
    upload = f"https://videos.invalid/upload/{job.organization_id}/{job.video_id}.mp4"
    assert setup.provider.bodies[0]["output"] == {"upload_url": upload}
    outputs.put(video_id=job.video_id, organization_id=job.organization_id, data=b"mp4")
    settled = engine.refresh(job)
    assert (settled.status, settled.content_url, settled.billed_seconds_milli) == (
        VideoStatus.COMPLETED,
        None,
        5_000,
    )
    assert _attempts(setup)[0]["estimated_cost_nano_usd"] == 5 * _XAI_RATE
    fetch = engine.content(
        organization_id=job.organization_id,
        identity_id=job.identity_id,
        video_id=job.video_id,
        variant="video",
    )
    assert fetch.url == f"https://videos.invalid/download/{job.organization_id}/{job.video_id}.mp4"
    engine.delete(
        organization_id=job.organization_id, identity_id=job.identity_id, video_id=job.video_id
    )
    assert outputs.stored(video_id=job.video_id, organization_id=job.organization_id) is None


def test_a_done_generation_whose_upload_never_arrived_bills_zero(setup: _Setup) -> None:
    """``done`` without the object in host storage fails the job at $0."""
    engine = _stored_engine(setup, InMemoryVideoOutputStore())
    job = _create_on(engine, setup, "grok-video", 6)
    settled = engine.refresh(job)
    assert (settled.status, settled.error_code, settled.billed_seconds_milli) == (
        VideoStatus.FAILED,
        "video_upload_missing",
        0,
    )
    assert _attempts(setup)[0]["estimated_cost_nano_usd"] == 0
    with pytest.raises(NativeBridgeError):
        engine.content(
            organization_id=job.organization_id,
            identity_id=job.identity_id,
            video_id=job.video_id,
            variant="video",
        )


def test_veo_ignores_host_storage(setup: _Setup) -> None:
    """Veo cannot upload to a caller URL, so its jobs keep the provider download."""
    outputs = InMemoryVideoOutputStore()
    job = _create_on(_stored_engine(setup, outputs), setup, "veo", 8)
    assert not job.output_stored and outputs.minted == []
    assert "output" not in setup.provider.bodies[0]


def test_an_unmintable_upload_destination_settles_unsent_at_zero(setup: _Setup) -> None:
    """A store that cannot mint fails the rung before any provider call, at $0."""
    engine = _stored_engine(setup, _BrokenOutputs())
    with pytest.raises(NativeBridgeError):
        _create_on(engine, setup, "grok-video", 6)
    assert setup.provider.posts == []
    attempts = _attempts(setup)
    assert len(attempts) == 1 and attempts[0]["estimated_cost_nano_usd"] == 0


def test_without_a_store_xai_is_submitted_without_output(setup: _Setup) -> None:
    """No host storage: the plain xAI request, served from the provider URL."""
    job = _create(setup, alias="grok-video", seconds=6)
    assert not job.output_stored
    assert "output" not in setup.provider.bodies[0]
