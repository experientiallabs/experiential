"""Optional release remains bounded when every serving callback worker is occupied."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import exp_gateway_native
import httpx
import pytest

from exp.common.models import BillingSource, GatewayTokenPrices
from exp.runtime.gateway.guardrails.classifiers import ClassifierRegistry
from exp.runtime.gateway.guardrails.client import DirectClassifierClient
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.store import MappingGuardrailStore
from exp.runtime.gateway.guardrails.store_test import _authorization
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.native_bridge import NativeControlPlane
from exp.runtime.gateway.native_capture import (
    CaptureConfiguration,
    CaptureController,
    CaptureDeliveryLimits,
)
from exp.runtime.gateway.native_capture_test import _request_json
from exp.runtime.gateway.native_server import serve_native_gateway
from exp.runtime.gateway.tests.launch_test import _unused_port
from exp.runtime.gateway.tests.native_waterfall_test import _content_chunk, _terminal_frames
from exp.runtime.gateway.tests.parallel_input_guardrails_test import _Classifier, _engine


@contextmanager
def _held_provider(received: threading.Event, release: threading.Event) -> Iterator[str]:
    """Hold provider output until unrelated callbacks occupy the serving pool."""
    frames = _content_chunk("safe-answer") + _terminal_frames(prompt_tokens=80, completion_tokens=5)

    class Provider(BaseHTTPRequestHandler):
        """Return finite metered output after the controlled output boundary."""

        def do_POST(self) -> None:  # noqa: N802 - HTTP protocol method.
            """Acknowledge dispatch, then wait for the output boundary."""
            self.rfile.read(int(self.headers["content-length"]))
            received.set()
            assert release.wait(5)
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(frames)))
            self.end_headers()
            self.wfile.write(frames)

        def log_message(self, format: str, *args: object) -> None:
            """Suppress local fixture access logs."""
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        worker.join(5)


@contextmanager
def _serving_with_capture(
    control: NativeControlPlane, collector: exp_gateway_native.CaptureCollector | None
) -> Iterator[str]:
    """Use four serving workers and the same collector as the admitted capability."""
    port = _unused_port()
    ready = threading.Event()
    shutdown = exp_gateway_native.shutdown_handle()
    worker = threading.Thread(
        target=serve_native_gateway,
        args=(control,),
        kwargs={
            "host": "127.0.0.1",
            "port": port,
            "capture": collector,
            "shutdown": shutdown,
            "on_listening": ready.set,
            "callback_permits": 4,
            "graceful_timeout_seconds": 1.0,
        },
        daemon=True,
    )
    worker.start()
    try:
        assert ready.wait(10)
        yield f"http://127.0.0.1:{port}"
    finally:
        shutdown.request_shutdown()
        worker.join(5)
        assert not worker.is_alive()


@pytest.mark.parametrize(
    ("capture_mode", "distinct_budgets"),
    [(mode, False) for mode in ("off", "on", "full", "hosted", "hosted-checkpoint-discard")]
    + [("off", True)],
)
def test_optional_release_survives_occupied_serving_pool_and_late_denial(
    tmp_path: Path, capture_mode: str, distinct_budgets: bool
) -> None:
    """Expiry releases while all serving workers remain blocked, then settles exactly once."""
    received, provider_release = threading.Event(), threading.Event()
    pool_blocked, callbacks_release = threading.Event(), threading.Event()
    first_response = threading.Event()
    classifier = _Classifier("block")
    template = _engine(classifier)
    policy = template.policies_for(_authorization())[0].model_copy(
        update={
            "fail_open": True,
            "input_max_wait_ms": 250,
            "organization_id": None,
            "identity_id": None,
        }
    )
    template.close()
    engine = GuardrailEngine(
        store=MappingGuardrailStore(
            (policy.model_copy(update={"policy_id": "shorter", "input_max_wait_ms": 50}), policy)
            if distinct_budgets
            else (policy,)
        ),
        client=DirectClassifierClient(ClassifierRegistry({"criminal": classifier})),
        monotonic=time.monotonic,
    )
    records: list[str] = []
    collector = (
        None
        if capture_mode == "off"
        else exp_gateway_native.CaptureCollector(
            CaptureConfiguration(
                settlement_required=capture_mode.startswith("hosted"),
                asynchronous_delivery=capture_mode.startswith("hosted"),
                maximum_pending_records=1,
                checkpoint_failure="discard",
                delivery=(
                    CaptureDeliveryLimits(maximum_record_bytes=1024, maximum_bytes=6400)
                    if capture_mode == "hosted-checkpoint-discard"
                    else CaptureDeliveryLimits()
                ),
            ).model_dump_json(),
            records.append,
        )
    )
    if capture_mode == "full":
        assert collector is not None and collector.begin(_request_json())
    capture = (
        None
        if collector is None
        else CaptureController(collector, application_for=lambda _authorization: "capture-fixture")
    )

    class OccupiedPlane(NativeControlPlane):
        """Block ordinary serving callbacks while settlement capacity stays available."""

        blocked = 0
        block_lock = threading.Lock()

        def settle(self, argument: str) -> str:
            """Grant hosted retention only after the real ledger settles the request."""
            result = super().settle(argument)
            if capture_mode.startswith("hosted"):
                assert collector is not None
                collector.settle(json.loads(argument)["request_id"], True, True)
            return result

        def models(self, argument: str) -> str:
            """Occupy a serving worker until the response escapes its input gate."""
            with self.block_lock:
                self.blocked += 1
                if self.blocked == 4:
                    pool_blocked.set()
            assert callbacks_release.wait(5)
            return super().models(argument)

    with _held_provider(received, provider_release) as base:
        _, key = _configured_gateway(
            tmp_path,
            base_url=base,
            billing_source=BillingSource.HOST_MANAGED,
            prices=GatewayTokenPrices(
                input_nano_usd_per_million_tokens=1_000_000_000,
                output_nano_usd_per_million_tokens=2_000_000_000,
            ),
        )
        components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "fixture"})
        control = OccupiedPlane(components, guardrails=engine, capture=capture)
        with (
            _serving_with_capture(control, collector) as url,
            ThreadPoolExecutor(max_workers=5) as pool,
        ):

            def completion() -> tuple[int, str]:
                """Observe first content independently of mandatory final settlement."""
                lines: list[str] = []
                with (
                    httpx.Client(timeout=5) as client,
                    client.stream(
                        "POST",
                        url + "/v1/chat/completions",
                        headers={"authorization": f"Bearer {key}"},
                        json={
                            "model": "coding",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "synthetic"
                                    * (400 if capture_mode == "hosted-checkpoint-discard" else 1),
                                }
                            ],
                            "stream": True,
                            "max_tokens": 100,
                        },
                    ) as response,
                ):
                    if response.status_code != 200:
                        first_response.set()
                    for line in response.iter_lines():
                        lines.append(line)
                        if "safe-answer" in line:
                            first_response.set()
                    return response.status_code, "\n".join(lines)

            future = pool.submit(completion)
            blockers = []
            try:
                assert received.wait(2) and classifier.started.wait(2)
                for _ in range(4):
                    blockers.append(
                        pool.submit(
                            httpx.get,
                            url + "/v1/models",
                            headers={"authorization": f"Bearer {key}"},
                            timeout=5,
                        )
                    )
                assert pool_blocked.wait(2)
                if collector is not None:
                    assert collector.counts()[0] == 0
                assert records == [] and not future.done()
                started = time.monotonic()
                provider_release.set()
                assert first_response.wait(0.8), "optional delivery queued behind serving callbacks"
                assert time.monotonic() - started < 0.8
                assert not callbacks_release.is_set() and all(not b.done() for b in blockers)
                classifier.release.set()
                callbacks_release.set()
                status, body = future.result(timeout=3)
                assert status == (503 if capture_mode == "full" else 200), body
                assert ("safe-answer" in body) == (capture_mode != "full")
                if capture_mode == "full":
                    assert "capture_unavailable" in body
                for blocker in blockers:
                    assert blocker.result(timeout=2).status_code == 200
            finally:
                provider_release.set()
                callbacks_release.set()
                classifier.release.set()
                if capture_mode == "full":
                    assert collector is not None
                    collector.settle("request", False, False)
            with sqlite3.connect(components.ledger.database_path) as db:
                assert db.execute("select terminal_state from gateway_requests").fetchall() == [
                    ("failed" if capture_mode == "full" else "completed",)
                ]
                assert db.execute(
                    "select budget_settled_nano_usd from gateway_attempts"
                ).fetchall() == [(0 if capture_mode == "full" else 90_000,)]
        engine.close(timeout_seconds=2)
        if collector is not None:
            assert collector.close(2)
            assert len(records) == {"on": 1, "hosted": 2}.get(capture_mode, 0)
            assert collector.checkpoint_discarded() == (
                1 if capture_mode == "hosted-checkpoint-discard" else 0
            )
