"""Real local gateway composition releases worker threads and database descriptors."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import cast

import pytest

from exp.common.models import ModelRequest
from exp.common.sqlite.connection import close_idle_connections
from exp.runtime.gateway import lifecycle, routing
from exp.runtime.gateway.guardrails.classifiers import ClassifierRegistry
from exp.runtime.gateway.guardrails.client import DirectClassifierClient
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.store import MappingGuardrailStore
from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.native_bridge import NativeControlPlane
from exp.runtime.gateway.video.engine import VideoEngine
from exp.runtime.models.providers.async_transport import RequestDeadline
from exp.runtime.router.runtime import RouterRuntime


def _descriptors() -> int:
    """Count this Linux process's open descriptors without forcing garbage collection."""
    return len(tuple(Path("/proc/self/fd").iterdir()))


def test_repeated_local_gateway_create_use_close_keeps_resources_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Independent real worlds close pollers, writers, selection lanes and their WAL handles."""
    if not Path("/proc/self/fd").is_dir():
        pytest.skip("descriptor regression requires Linux procfs")
    start = VideoEngine.start_poller
    polled = threading.Event()
    poll = VideoEngine.poll_open

    def quickly(self: VideoEngine, *, interval_seconds: float = 5.0) -> threading.Thread:
        """Use a short real poll interval so every composition opens its worker database."""
        return start(self, interval_seconds=0.01)

    def observed(self: VideoEngine) -> None:
        """Run the real SQLite poll and signal after its connection was used."""
        poll(self)
        polled.set()

    monkeypatch.setattr(VideoEngine, "start_poller", quickly)
    monkeypatch.setattr(VideoEngine, "poll_open", observed)

    def cycle(index: int) -> None:
        """Build, authenticate, let the real poller use SQLite, then explicitly close owners."""
        root = tmp_path / str(index)
        _, key = _configured_gateway(root)
        components = lifecycle.load_gateway_components(
            root, environment={"TEST_PROVIDER_KEY": "local-test-value"}
        )
        control = NativeControlPlane(components)
        try:
            polled.clear()
            control.models(json.dumps({"raw_key": key}))
            assert polled.wait(2), "real video poller never inspected its job store"
            assert components.accounting_healthy
        finally:
            control.close(timeout_seconds=2)
            components.close(timeout_seconds=2)
        control.close(timeout_seconds=2)
        components.close(timeout_seconds=2)
        assert components.write_ledger.stopped

    cycle(0)  # Warm process-wide tokenizer and thread-local library initialization once.
    baseline_fds = _descriptors()
    baseline_threads = threading.active_count()
    for index in range(1, 13):
        cycle(index)
        assert _descriptors() <= baseline_fds + 2
        assert threading.active_count() <= baseline_threads, [
            (thread.name, thread.ident) for thread in threading.enumerate()
        ]


def test_failed_local_gateway_startup_rolls_back_owned_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed authority loader cannot strand the eagerly started ledger writer."""
    _configured_gateway(tmp_path)
    close_idle_connections()
    initial_threads = threading.active_count()
    initial_descriptors = _descriptors()

    def fail(*_args: object, **_kwargs: object) -> None:
        """Fail after the real writer has opened, as an invalid serving catalog would."""
        raise RuntimeError("injected authority startup failure")

    monkeypatch.setattr(lifecycle, "_load_alias_state", fail)
    with pytest.raises(RuntimeError, match="injected authority startup failure"):
        lifecycle.load_gateway_components(
            tmp_path, environment={"TEST_PROVIDER_KEY": "local-test-value"}
        )
    assert threading.active_count() == initial_threads
    assert _descriptors() <= initial_descriptors


def test_control_shutdown_does_not_claim_a_running_video_poller_has_exited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bounded stop keeps a blocked poll visible until the real worker releases its handle."""
    _configured_gateway(tmp_path)
    components = lifecycle.load_gateway_components(
        tmp_path, environment={"TEST_PROVIDER_KEY": "local-test-value"}
    )
    entered, release = threading.Event(), threading.Event()
    poll = VideoEngine.poll_open
    start = VideoEngine.start_poller

    def blocked(self: VideoEngine) -> None:
        """Open the real job-store connection, then hold the worker as a slow provider would."""
        poll(self)
        entered.set()
        assert release.wait(5)

    def quickly(self: VideoEngine, *, interval_seconds: float = 5.0) -> threading.Thread:
        """Start the poll without the normal idle interval."""
        return start(self, interval_seconds=0.01)

    monkeypatch.setattr(VideoEngine, "poll_open", blocked)
    monkeypatch.setattr(VideoEngine, "start_poller", quickly)
    control = NativeControlPlane(components)
    try:
        assert entered.wait(2)
        assert control.close(timeout_seconds=0.02) is False
    finally:
        release.set()
        control.close(timeout_seconds=2)
        components.close(timeout_seconds=2)
    assert components.write_ledger.stopped


def test_closing_one_control_preserves_borrowed_components_and_guardrails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two planes may share host authority and policy; each stops only its own workers."""
    _, key = _configured_gateway(tmp_path)
    components = lifecycle.load_gateway_components(
        tmp_path, environment={"TEST_PROVIDER_KEY": "local-test-value"}
    )
    guardrails = GuardrailEngine(
        store=MappingGuardrailStore(()),
        client=DirectClassifierClient(ClassifierRegistry({})),
        monotonic=time.monotonic,
    )
    first = NativeControlPlane(components, guardrails=guardrails)
    second = NativeControlPlane(components, guardrails=guardrails)
    owner_close = guardrails.close

    def borrowed(*, timeout_seconds: float = 1.0) -> None:
        """Fail if a plane tries to stop the host's shared policy engine."""
        raise AssertionError("a control plane cannot close borrowed guardrails")

    monkeypatch.setattr(guardrails, "close", borrowed)
    try:
        first.close(timeout_seconds=2)
        first.close(timeout_seconds=2)
        assert components.accounting_healthy
        assert first.readiness("{}") == "false"
        assert not first._accounting.running
        assert first._accounting.accounting_healthy
        assert second._accounting.running
        assert second.readiness("{}") == "true"
        assert "coding" in second.models(json.dumps({"raw_key": key}))
        second.close(timeout_seconds=2)
        assert second.readiness("{}") == "false"
        assert not second._accounting.running
        assert second._accounting.accounting_healthy
        assert components.accounting_healthy
    finally:
        monkeypatch.setattr(guardrails, "close", owner_close)
        try:
            first.close(timeout_seconds=2)
        finally:
            try:
                second.close(timeout_seconds=2)
            finally:
                try:
                    components.close(timeout_seconds=2)
                finally:
                    owner_close(timeout_seconds=2)


def test_failed_control_setup_stops_its_already_constructed_accounting_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exception starting video polling cannot leak the earlier settlement sweeper."""
    _configured_gateway(tmp_path)
    components = lifecycle.load_gateway_components(
        tmp_path, environment={"TEST_PROVIDER_KEY": "local-test-value"}
    )
    baseline_threads = threading.active_count()

    def fail(_self: VideoEngine, *, interval_seconds: float = 5.0) -> threading.Thread:
        """Fail after accounting has started but before a complete plane can be returned."""
        raise RuntimeError("injected video poller failure")

    monkeypatch.setattr(VideoEngine, "start_poller", fail)
    try:
        with pytest.raises(RuntimeError, match="injected video poller failure"):
            NativeControlPlane(components)
        assert threading.active_count() == baseline_threads
        assert components.accounting_healthy
    finally:
        components.close(timeout_seconds=2)


def test_component_shutdown_retains_writer_until_running_selection_really_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bounded failed drain remains retryable and never closes dependencies of a live lane."""
    _configured_gateway(tmp_path)
    components = lifecycle.load_gateway_components(
        tmp_path, environment={"TEST_PROVIDER_KEY": "local-test-value"}
    )
    entered, release = threading.Event(), threading.Event()

    def blocked(*_args: object, **_kwargs: object) -> None:
        """Hold a selected worker as a slow synchronous embedding call would."""
        entered.set()
        assert release.wait(5)
        raise RuntimeError("released selection")

    monkeypatch.setattr(routing, "_select_within_deadline", blocked)
    future = components.selection_workers.submit(
        cast("RouterRuntime", None),
        cast("ModelRequest", None),
        episode_id="blocked-shutdown",
        deadline=RequestDeadline.after(5),
    )
    try:
        assert entered.wait(2)
        with pytest.raises(TimeoutError, match="selection workers remain active; ledger retained"):
            components.close(timeout_seconds=0.02)
        assert components.accounting_healthy
        assert not components.write_ledger.stopped
    finally:
        release.set()
        with pytest.raises(RuntimeError, match="released selection"):
            future.result(timeout=2)
        components.close(timeout_seconds=2)
    assert components.write_ledger.stopped
