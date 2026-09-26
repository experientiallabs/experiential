"""Capture delivery retries safely while keeping cloud failures off inference."""

import hashlib
import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from exp.common.core.artifacts import JsonValue
from exp.runtime.capture.normalization import CapturedExchange, CaptureProtocol, normalize_exchange
from exp.runtime.capture.upload import CaptureUploader

_UPLOAD_ORIGIN = "https://storage.example"
_UPLOAD_PREFIX = "/storage/v1/object/upload/sign/artifacts/orgs/org/telemetry-traces/otlp/"
_UPLOAD_TEMPLATE = f"{_UPLOAD_ORIGIN}{_UPLOAD_PREFIX}{{ingest}}/{{nonce}}?token=signed"


def _signed_url(ingest: str) -> str:
    """Return a synthetic ticket within the run's acknowledged storage scope."""
    return f"{_UPLOAD_ORIGIN}{_UPLOAD_PREFIX}{ingest}/{'a' * 43}?token=signed"


def _exchange() -> CapturedExchange:
    """Return a bounded synthetic request and response for this test."""
    return CapturedExchange(
        protocol="responses",
        host="api.openai.com",
        path="/v1/responses",
        started_ns=1,
        ended_ns=2,
        request=b'{"model":"test","input":"hello","api_key":"secret"}',
        response=b'{"output":[]}',
        status=200,
    )


def _wait(predicate: Callable[[], bool]) -> None:
    """Wait briefly for the bounded background delivery worker to make progress."""
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.01)


def _usage_exchange(input_tokens: int = 3, output_tokens: int = 7) -> CapturedExchange:
    """Build one synthetic response with a provider-reported token pair."""
    return replace(
        _exchange(),
        response=json.dumps(
            {"output": [], "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}}
        ).encode(),
    )


@pytest.mark.parametrize(
    ("protocol", "content_type", "response"),
    [
        (
            "responses",
            "application/json",
            b'{"output":[],"usage":{"input_tokens":3,"output_tokens":7}}',
        ),
        (
            "chat",
            "application/json",
            b'{"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":7}}',
        ),
        (
            "messages",
            "application/json",
            b'{"content":[],"usage":{"input_tokens":3,"output_tokens":7}}',
        ),
        (
            "responses",
            "text/event-stream",
            b'data: {"type":"response.completed","response":{"output":[],'
            b'"usage":{"input_tokens":3,"output_tokens":7}}}\n\n',
        ),
        (
            "chat",
            "text/event-stream",
            b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":7}}\n\n'
            b"data: [DONE]\n\n",
        ),
        (
            "messages",
            "text/event-stream",
            b'data: {"type":"message_start","message":{"usage":{"input_tokens":3}}}\n\n'
            b'data: {"type":"message_delta","usage":{"output_tokens":7}}\n\n',
        ),
    ],
)
def test_provider_usage_reaches_live_and_final_capture_counts(
    tmp_path: Path, protocol: CaptureProtocol, content_type: str, response: bytes
) -> None:
    """Count actual provider usage from each supported JSON and streaming wire protocol."""
    run = str(uuid4())
    uploader = CaptureUploader(
        "https://api.example",
        "org",
        run,
        "KEY",
        tmp_path / run,
        upload_origin=_UPLOAD_ORIGIN,
        upload_path_prefix=_UPLOAD_PREFIX,
        transport=httpx.MockTransport(lambda request: httpx.Response(503)),
    )
    uploader.start()
    try:
        assert uploader.submit(
            replace(
                _exchange(),
                protocol=protocol,
                response_content_type=content_type,
                response=response,
            )
        )
        _wait(lambda: uploader.stats.usage_exchanges == 1)
        assert uploader.stats.input_tokens == 3
        assert uploader.stats.output_tokens == 7
    finally:
        final = uploader.close()
    assert (final.input_tokens, final.output_tokens, final.usage_exchanges) == (3, 7, 1)
    assert uploader.stats == uploader.close() == final


@pytest.mark.parametrize("broken_diagnostic", [False, True])
def test_capture_receipts_correlate_saved_evidence_without_exposing_content(
    tmp_path: Path, broken_diagnostic: bool
) -> None:
    """Receipts expose only bounded identifiers and counts, and a closed terminal loses no data."""
    run = str(uuid4())
    diagnostics: list[str] = []
    response_id = "resp_SYNTHETIC_PRIVATE_ID"
    trace_id = uuid4().hex

    def diagnostic(event: str) -> None:
        """Simulate a terminal that receives one event but may fail to render it."""
        diagnostics.append(event)
        if broken_diagnostic:
            raise OSError("SYNTHETIC_PRIVATE_ERROR")

    uploader = CaptureUploader(
        "https://api.example",
        "org",
        run,
        "KEY",
        tmp_path / run,
        upload_origin=_UPLOAD_ORIGIN,
        upload_path_prefix=_UPLOAD_PREFIX,
        transport=httpx.MockTransport(lambda request: httpx.Response(503)),
        on_diagnostic=diagnostic,
    )
    uploader.start()
    try:
        assert uploader.submit(
            replace(
                _usage_exchange(),
                response=json.dumps(
                    {
                        "id": response_id,
                        "status": "completed",
                        "output": [],
                        "usage": {"input_tokens": 3, "output_tokens": 7},
                    }
                ).encode(),
                failed=True,
                trace_id=trace_id,
            )
        )
        _wait(lambda: any("capture_saved" in event for event in diagnostics))
    finally:
        stats = uploader.close()
    assert (stats.captured_exchanges, stats.pending_batches, stats.dropped_exchanges) == (1, 1, 0)
    assert (stats.input_tokens, stats.output_tokens) == (3, 7)
    saved = next(event for event in diagnostics if "capture_saved" in event)
    assert f"trace {trace_id}" in saved
    assert f"response {hashlib.sha256(response_id.encode()).hexdigest()[:16]}" in saved
    assert "completed=True" in saved and "interrupted=False" in saved
    assert "transport_error=True" in saved and "3 in / 7 out tokens" in saved
    assert not any("SYNTHETIC_PRIVATE" in event or "secret" in event for event in diagnostics)
    payload = json.loads(next((tmp_path / run).glob("*.json")).read_bytes())
    span = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert span["traceId"] == trace_id
    assert span["status"] == {"code": 1}


@pytest.mark.parametrize("fail_after_unblocking", [False, True])
def test_blocked_verbose_reader_cannot_stop_storage_delivery_or_bounded_shutdown(
    tmp_path: Path,
    fail_after_unblocking: bool,
) -> None:
    """Fill the diagnostic queue while both data workers continue with a blocked output sink."""
    blocked = threading.Event()
    release = threading.Event()
    run = str(uuid4())
    diagnostics: list[str] = []
    failures: set[str] = set()

    def diagnostic(event: str) -> None:
        """Block exactly like a full terminal pipe, independently of the uploader workers."""
        blocked.set()
        assert release.wait(10)
        kind = "notice" if event.startswith("diagnostic_events_dropped:") else "receipt"
        if fail_after_unblocking and kind not in failures:
            failures.add(kind)
            raise OSError("synthetic output failure")
        diagnostics.append(event)

    def platform(request: httpx.Request) -> httpx.Response:
        """Accept immutable batches without introducing provider or network dependencies."""
        if request.url.host == "storage.example":
            return httpx.Response(200)
        if request.url.path.endswith("/batches/upload"):
            ingest_id = str(uuid4())
            return httpx.Response(
                200,
                json={
                    "status": "pending",
                    "ingest_id": ingest_id,
                    "signed_url": _signed_url(ingest_id),
                },
            )
        assert request.url.path.endswith("/finalize")
        return httpx.Response(202)

    uploader = CaptureUploader(
        "https://api.example",
        "org",
        run,
        "KEY",
        tmp_path / run,
        upload_origin=_UPLOAD_ORIGIN,
        upload_path_prefix=_UPLOAD_PREFIX,
        transport=httpx.MockTransport(platform),
        on_diagnostic=diagnostic,
    )
    uploader.start()
    try:
        assert uploader.submit(_usage_exchange())
        assert blocked.wait(2)
        _wait(lambda: uploader.stats.uploaded_batches == 1)
        # More saved receipts than the diagnostic queue can hold must still persist.
        for index in range(1, 140):
            assert uploader.submit(_usage_exchange())
            _wait(lambda expected=index + 1: uploader.stats.usage_exchanges == expected)
        assert uploader.stats.uploaded_batches >= 2
        started = time.monotonic()
        stats = uploader.close(timeout=0.25)
        assert time.monotonic() - started < 1
        assert stats.captured_exchanges == 140
        assert stats.dropped_exchanges == 0
        assert stats.pending_batches + stats.uploaded_batches == 140
        assert (stats.input_tokens, stats.output_tokens) == (420, 980)
    finally:
        release.set()
        uploader.close()
        if uploader._diagnostic_thread is not None:
            uploader._diagnostic_thread.join(timeout=1)
            assert not uploader._diagnostic_thread.is_alive()
    lost = sum(
        int(event.split(": ", 1)[1])
        for event in diagnostics
        if event.startswith("diagnostic_events_dropped:")
    )
    receipts = sum(event.startswith(("capture_saved", "upload_accepted")) for event in diagnostics)
    assert lost > 0
    assert receipts + lost == stats.captured_exchanges + stats.uploaded_batches
    assert failures == ({"notice", "receipt"} if fail_after_unblocking else set())


@pytest.mark.parametrize("shutdown", [False, True])
def test_last_failed_receipt_reports_loss_without_another_request(
    tmp_path: Path, shutdown: bool
) -> None:
    """Account for a final output failure during idle time and diagnostic shutdown."""
    run = str(uuid4())
    diagnostics: list[str] = []
    reported = threading.Event()

    def diagnostic(event: str) -> None:
        """Recover immediately after rejecting the last receipt."""
        if event == "capture_saved":
            raise OSError("synthetic output failure")
        diagnostics.append(event)
        reported.set()

    uploader = CaptureUploader(
        "https://api.example",
        "org",
        run,
        "KEY",
        tmp_path / run,
        upload_origin=_UPLOAD_ORIGIN,
        upload_path_prefix=_UPLOAD_PREFIX,
        on_diagnostic=diagnostic,
    )
    uploader._diagnostic("capture_saved")
    if shutdown:
        uploader._diagnostics_done.set()
    reporter = threading.Thread(target=uploader._report_diagnostics, daemon=True)
    reporter.start()
    try:
        assert reported.wait(2)
        assert diagnostics == ["diagnostic_events_dropped: 1"]
    finally:
        uploader._diagnostics_done.set()
        reporter.join(timeout=1)
    assert not reporter.is_alive()


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"input_tokens": 3},
        {"input_tokens": -1, "output_tokens": 7},
        {"input_tokens": 3, "output_tokens": -1},
        {"input_tokens": True, "output_tokens": 7},
        {"input_tokens": 3, "output_tokens": False},
        {"input_tokens": "3", "output_tokens": 7},
        {"input_tokens": 3, "output_tokens": 7.0},
    ],
)
def test_missing_or_invalid_provider_usage_remains_unknown(
    tmp_path: Path, usage: JsonValue
) -> None:
    """Persist supported requests without inventing token counts for missing or invalid usage."""
    run = str(uuid4())
    uploader = CaptureUploader(
        "https://api.example",
        "org",
        run,
        "KEY",
        tmp_path / run,
        upload_origin=_UPLOAD_ORIGIN,
        upload_path_prefix=_UPLOAD_PREFIX,
        transport=httpx.MockTransport(lambda request: httpx.Response(503)),
    )
    uploader.start()
    assert uploader.submit(
        replace(_exchange(), response=json.dumps({"output": [], "usage": usage}).encode())
    )
    final = uploader.close()
    assert final.pending_batches == 1
    assert final.dropped_exchanges == 0
    assert (final.input_tokens, final.output_tokens, final.usage_exchanges) == (0, 0, 0)


def test_usage_totals_accumulate_known_exchanges_including_zero(tmp_path: Path) -> None:
    """Add known counts across this run while distinguishing reported zero from unknown usage."""
    run = str(uuid4())
    uploader = CaptureUploader(
        "https://api.example",
        "org",
        run,
        "KEY",
        tmp_path / run,
        upload_origin=_UPLOAD_ORIGIN,
        upload_path_prefix=_UPLOAD_PREFIX,
        transport=httpx.MockTransport(lambda request: httpx.Response(503)),
    )
    uploader.start()
    for exchange in (_usage_exchange(), _usage_exchange(5, 11), _usage_exchange(0, 0), _exchange()):
        assert uploader.submit(exchange)
    final = uploader.close()
    assert final.captured_exchanges == final.pending_batches == 4
    assert (final.input_tokens, final.output_tokens, final.usage_exchanges) == (8, 18, 3)


def test_current_run_usage_is_counted_once_across_upload_retries(tmp_path: Path) -> None:
    """Retry the same durable batch without adding its provider usage a second time."""
    run, ingest = str(uuid4()), str(uuid4())
    directory = tmp_path / run
    directory.mkdir()
    statuses = iter(("running", "done"))

    def handler(request: httpx.Request) -> httpx.Response:
        """Return two distinct receipts for one synthetic upload batch."""
        return httpx.Response(200, json={"ingest_id": ingest, "status": next(statuses)})

    uploader = CaptureUploader(
        "https://api.example",
        "org",
        run,
        "KEY",
        directory,
        upload_origin=_UPLOAD_ORIGIN,
        upload_path_prefix=_UPLOAD_PREFIX,
    )
    assert uploader.submit(_usage_exchange())
    uploader._stop.set()
    uploader._work()
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        for _ in range(2):
            uploader._retry_at.clear()
            uploader._deliver_one(client)
            assert uploader.stats.input_tokens == 3
            assert uploader.stats.output_tokens == 7
            assert uploader.stats.usage_exchanges == 1
    final = uploader.close()
    assert final.uploaded_batches == 1
    assert final.pending_batches == 0
    assert (final.input_tokens, final.output_tokens, final.usage_exchanges) == (3, 7, 1)


def test_recovery_keeps_original_run_batch_and_never_sends_api_key_to_storage(
    tmp_path: Path,
) -> None:
    """Recover an existing batch with stable identity and separate storage credentials."""
    prior_run, run, batch, ingest = (str(uuid4()) for _ in range(4))
    prior = tmp_path / prior_run
    prior.mkdir()
    (prior / f"{batch}.json").write_bytes(
        normalize_exchange(_usage_exchange(), max_body_bytes=4096)[0]
    )
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Emulate the cloud response needed by this delivery scenario."""
        requests.append(request)
        if request.url.host == "storage.example":
            assert "authorization" not in request.headers
            assert b"secret" not in request.content
            return httpx.Response(409)
        assert request.headers["authorization"] == "Bearer PLATFORM-KEY"
        if request.url.path.endswith(f"/{prior_run}/end"):
            assert json.loads(request.content) == {"pending_batches": 0, "upload_errors": 0}
            return httpx.Response(200)
        if request.url.path.endswith("/batches/upload"):
            assert prior_run in request.url.path
            assert json.loads(request.content)["batch_id"] == batch
            return httpx.Response(
                200,
                json={
                    "status": "pending",
                    "signed_url": _signed_url(ingest),
                    "ingest_id": ingest,
                },
            )
        assert request.url.path.endswith(f"/{ingest}/finalize")
        return httpx.Response(202)

    uploader = CaptureUploader(
        "https://api.example",
        "org",
        run,
        "PLATFORM-KEY",
        tmp_path / run,
        upload_origin=_UPLOAD_ORIGIN,
        upload_path_prefix=_UPLOAD_PREFIX,
        transport=httpx.MockTransport(handler),
    )
    uploader.start()
    try:
        assert uploader.pending_current_run == 0
        _wait(lambda: uploader.stats.uploaded_batches == 1)
    finally:
        uploader.close()
    assert len(requests) == 4
    assert not (prior / f"{batch}.json").exists()
    assert uploader.stats.usage_exchanges == 0
    assert uploader.stats.input_tokens == uploader.stats.output_tokens == 0


def test_slow_cloud_does_not_block_submission_or_local_shutdown_flush(tmp_path: Path) -> None:
    """Keep inference enqueue and local persistence responsive during a cloud stall."""
    entered, release = threading.Event(), threading.Event()
    run = str(uuid4())

    def handler(request: httpx.Request) -> httpx.Response:
        """Emulate the cloud response needed by this delivery scenario."""
        entered.set()
        release.wait(3)
        return httpx.Response(503)

    uploader = CaptureUploader(
        "https://api.example",
        "org",
        run,
        "KEY",
        tmp_path / run,
        upload_origin=_UPLOAD_ORIGIN,
        upload_path_prefix=_UPLOAD_PREFIX,
        transport=httpx.MockTransport(handler),
    )
    uploader.start()
    try:
        assert uploader.submit(_exchange())
        assert entered.wait(2)
        before = time.monotonic()
        assert uploader.submit(_exchange())
        assert time.monotonic() - before < 0.1
        uploader.close(timeout=0.3)
        assert len(list((tmp_path / run).glob("*.json"))) == 2
        assert all(path.stat().st_mode & 0o777 == 0o600 for path in (tmp_path / run).glob("*.json"))
    finally:
        release.set()
        uploader.close()


def test_finite_raw_queue_reports_backpressure_without_starting_worker(tmp_path: Path) -> None:
    """Reject queue overflow immediately and expose a dropped-capture counter."""
    run = str(uuid4())
    uploader = CaptureUploader(
        "https://api.example",
        "org",
        run,
        "KEY",
        tmp_path / run,
        upload_origin=_UPLOAD_ORIGIN,
        upload_path_prefix=_UPLOAD_PREFIX,
        max_queue_bytes=1,
    )
    assert not uploader.submit(_exchange())
    assert uploader.stats.dropped_exchanges == 1


def test_spool_rejects_symbolic_link_ancestors(tmp_path: Path) -> None:
    """Reject spool paths that could follow a symbolic link into another directory."""
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(target)
    run = str(uuid4())
    uploader = CaptureUploader(
        "https://api.example",
        "org",
        run,
        "KEY",
        linked / run,
        upload_origin=_UPLOAD_ORIGIN,
        upload_path_prefix=_UPLOAD_PREFIX,
    )
    with pytest.raises(ValueError, match="symbolic"):
        uploader.start()


@pytest.mark.parametrize(
    "destination_template",
    [
        _UPLOAD_TEMPLATE.replace("storage.example", "unapproved.example"),
        _UPLOAD_TEMPLATE.replace("storage.example", "storage.example.attacker.example"),
        _UPLOAD_TEMPLATE.replace("storage.example", "storage.example:8443"),
        _UPLOAD_TEMPLATE.replace("https://", "http://"),
        _UPLOAD_TEMPLATE.replace("https://", "https://user:password@"),
        _UPLOAD_TEMPLATE.replace("/orgs/org/", "/orgs/other/"),
        _UPLOAD_TEMPLATE.replace("{ingest}", "00000000-0000-0000-0000-000000000000"),
        _UPLOAD_TEMPLATE.replace("/{nonce}", "/unexpected/{nonce}"),
        _UPLOAD_TEMPLATE.replace("/{nonce}", "/%2e%2e/{nonce}"),
        _UPLOAD_TEMPLATE + "#fragment",
        "https://[malformed",
    ],
)
def test_misrouted_signed_ticket_never_sends_capture_bytes(
    tmp_path: Path, destination_template: str
) -> None:
    """Reject an unapproved origin or object path before making any signed PUT."""
    run, ingest = str(uuid4()), str(uuid4())
    directory = tmp_path / run
    directory.mkdir()
    path = directory / f"{uuid4()}.json"
    path.write_bytes(b'{"synthetic":"capture-content-canary"}')
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Return an inconsistent ticket from the otherwise authenticated control API."""
        requests.append(request)
        assert request.method == "POST"
        assert request.url.host == "api.example"
        assert request.headers["authorization"] == "Bearer PLATFORM-KEY"
        assert b"capture-content-canary" not in request.content
        return httpx.Response(
            200,
            json={
                "status": "pending",
                "ingest_id": ingest,
                "signed_url": destination_template.format(ingest=ingest, nonce="a" * 43),
            },
        )

    uploader = CaptureUploader(
        "https://api.example",
        "org",
        run,
        "PLATFORM-KEY",
        directory,
        upload_origin=_UPLOAD_ORIGIN,
        upload_path_prefix=_UPLOAD_PREFIX,
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="capture upload destination"):
            uploader._upload(client, path)
    assert len(requests) == 1
    assert path.exists()


@pytest.mark.parametrize(
    ("api_origin", "upload_origin", "path_prefix"),
    [
        ("https://api.example", "https://storage.example:443", _UPLOAD_PREFIX),
        ("https://preview.example", "https://storage.example:8443", "/proxy" + _UPLOAD_PREFIX),
        (
            "https://preview.example",
            "https://storage.example",
            _UPLOAD_PREFIX.replace("artifacts", "artifact%20bucket"),
        ),
        ("http://127.0.0.1:8000", "http://localhost:55421", _UPLOAD_PREFIX),
        ("http://[::1]:8000", "http://[::1]:55421", _UPLOAD_PREFIX),
    ],
)
def test_approved_storage_scope_supports_preview_and_explicit_local_development(
    tmp_path: Path, api_origin: str, upload_origin: str, path_prefix: str
) -> None:
    """Respect explicit deployment origins without sending Platform credentials to Storage."""
    run, ingest = str(uuid4()), str(uuid4())
    directory = tmp_path / run
    directory.mkdir()
    path = directory / f"{uuid4()}.json"
    content = b'{"synthetic":"capture-content-canary"}'
    path.write_bytes(content)
    requests: list[httpx.Request] = []
    signed_url = f"{upload_origin}{path_prefix}{ingest}/{'a' * 43}?token=signed"

    def handler(request: httpx.Request) -> httpx.Response:
        """Accept the approved signed destination and the separate authenticated finalize."""
        requests.append(request)
        if request.method == "PUT":
            assert request.url == httpx.URL(signed_url)
            assert request.content == content
            assert "authorization" not in request.headers
            return httpx.Response(200)
        assert request.headers["authorization"] == "Bearer PLATFORM-KEY"
        if request.url.path.endswith("/batches/upload"):
            return httpx.Response(
                200, json={"status": "pending", "ingest_id": ingest, "signed_url": signed_url}
            )
        assert request.url.path.endswith(f"/{ingest}/finalize")
        return httpx.Response(202)

    uploader = CaptureUploader(
        api_origin,
        "org",
        run,
        "PLATFORM-KEY",
        directory,
        upload_origin=upload_origin,
        upload_path_prefix=path_prefix,
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        uploader._upload(client, path)
    assert [request.method for request in requests] == ["POST", "PUT", "POST"]


@pytest.mark.parametrize(
    ("api_origin", "upload_origin", "path_prefix"),
    [
        ("https://api.example", "http://localhost:55421", _UPLOAD_PREFIX),
        ("http://127.0.0.1:8000", "http://storage.example", _UPLOAD_PREFIX),
        ("http://api.example", "http://localhost:55421", _UPLOAD_PREFIX),
        ("https://api.example", "https://user:password@storage.example", _UPLOAD_PREFIX),
        ("https://api.example", "https://storage.example?query=true", _UPLOAD_PREFIX),
        ("https://api.example", "https://storage.example/unexpected", _UPLOAD_PREFIX),
        (
            "https://api.example",
            _UPLOAD_ORIGIN,
            _UPLOAD_PREFIX.replace("/orgs/org/", "/orgs/other/"),
        ),
        ("https://api.example", _UPLOAD_ORIGIN, "/../" + _UPLOAD_PREFIX),
        ("https://api.example", _UPLOAD_ORIGIN, _UPLOAD_PREFIX + "?query=true"),
    ],
)
def test_invalid_run_storage_policy_is_rejected_before_uploading(
    tmp_path: Path, api_origin: str, upload_origin: str, path_prefix: str
) -> None:
    """Reject credential-bearing, cross-organization, or unapproved cleartext policies."""
    run = str(uuid4())
    with pytest.raises(ValueError, match="capture run storage"):
        CaptureUploader(
            api_origin,
            "org",
            run,
            "PLATFORM-KEY",
            tmp_path / run,
            upload_origin=upload_origin,
            upload_path_prefix=path_prefix,
        )


@pytest.mark.parametrize("terminal", ["done", "error"])
def test_running_retry_keeps_local_copy_until_verified_completion(
    tmp_path: Path, terminal: str
) -> None:
    """Running without our own Storage receipt is not proof that bytes reached the cloud."""
    run, ingest = str(uuid4()), str(uuid4())
    directory = tmp_path / run
    directory.mkdir()
    path = directory / f"{uuid4()}.json"
    path.write_bytes(normalize_exchange(_exchange(), max_body_bytes=4096)[0])
    statuses = iter(("running", terminal))
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Return processing and terminal receipts without a signed upload destination."""
        requests.append(request)
        assert request.method == "POST"
        assert request.url.path.endswith("/batches/upload")
        return httpx.Response(200, json={"ingest_id": ingest, "status": next(statuses)})

    uploader = CaptureUploader(
        "https://api.example",
        "org",
        run,
        "KEY",
        directory,
        upload_origin=_UPLOAD_ORIGIN,
        upload_path_prefix=_UPLOAD_PREFIX,
    )
    uploader._recover_temporary_files()
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        uploader._deliver_one(client)
        assert path.exists()
        assert uploader.stats.pending_batches == uploader.pending_current_run == 1
        assert uploader.stats.uploaded_batches == 0
        assert uploader.stats.upload_errors == 0
        uploader._retry_at.clear()
        uploader._deliver_one(client)
    assert len(requests) == 2
    assert path.exists() == (terminal == "error")
    assert uploader.stats.uploaded_batches == int(terminal == "done")
    assert uploader.stats.upload_errors == int(terminal == "error")
    assert uploader.stats.pending_batches == int(terminal == "error")


def test_shutdown_counts_unfinished_copies_and_freezes_final_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Discard late normalization work while preserving already durable retry files."""
    run = str(uuid4())
    directory = tmp_path / run
    directory.mkdir()
    existing = directory / f"{uuid4()}.json"
    existing.write_bytes(normalize_exchange(_exchange(), max_body_bytes=4096)[0])
    entered, release = threading.Event(), threading.Event()

    def stalled_normalization(
        exchange: CapturedExchange, *, max_body_bytes: int
    ) -> tuple[bytes, tuple[int, int] | None, str]:
        """Pause an accepted in-memory copy until after the shutdown deadline."""
        entered.set()
        assert release.wait(3)
        return normalize_exchange(exchange, max_body_bytes=max_body_bytes)

    monkeypatch.setattr("exp.runtime.capture.upload.normalize_exchange", stalled_normalization)
    uploader = CaptureUploader(
        "https://api.example",
        "org",
        run,
        "KEY",
        directory,
        upload_origin=_UPLOAD_ORIGIN,
        upload_path_prefix=_UPLOAD_PREFIX,
    )
    uploader.start()
    try:
        assert uploader.submit(_usage_exchange())
        assert entered.wait(1)
        assert uploader.submit(_usage_exchange())
        before = time.monotonic()
        final = uploader.close(timeout=0.02)
        assert time.monotonic() - before < 0.5
        assert final.captured_exchanges == 2
        assert final.dropped_exchanges == 2
        assert final.pending_batches == uploader.pending_current_run == 1
        assert (final.input_tokens, final.output_tokens, final.usage_exchanges) == (0, 0, 0)
    finally:
        release.set()
        assert uploader._thread is not None
        uploader._thread.join(1)
    assert not uploader._thread.is_alive()
    assert uploader.stats == final
    assert uploader.close() == final
    assert list(directory.iterdir()) == [existing]


@pytest.mark.parametrize("stage", ["fsync", "rename"])
def test_shutdown_distinguishes_unfinished_write_from_durable_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    """Keep fsynced copies pending across a late rename without miscounting them as drops."""
    run = str(uuid4())
    directory = tmp_path / run
    entered, release = threading.Event(), threading.Event()
    original_fsync, original_replace = os.fsync, Path.replace

    def stalled_fsync(descriptor: int) -> None:
        """Pause before the file reaches the durable commit point."""
        entered.set()
        assert release.wait(3)
        original_fsync(descriptor)

    def stalled_replace(path: Path, target: str | Path) -> Path:
        """Pause after fsync but before publication of the final filename."""
        entered.set()
        assert release.wait(3)
        return original_replace(path, target)

    if stage == "fsync":
        monkeypatch.setattr("exp.runtime.capture.upload.os.fsync", stalled_fsync)
    else:
        monkeypatch.setattr(Path, "replace", stalled_replace)
    uploader = CaptureUploader(
        "https://api.example",
        "org",
        run,
        "KEY",
        directory,
        upload_origin=_UPLOAD_ORIGIN,
        upload_path_prefix=_UPLOAD_PREFIX,
    )
    uploader.start()
    try:
        assert uploader.submit(_usage_exchange())
        assert entered.wait(1)
        final = uploader.close(timeout=0.02)
        assert final.dropped_exchanges == int(stage == "fsync")
        assert final.pending_batches == uploader.pending_current_run == int(stage == "rename")
        expected_usage = (3, 7, 1) if stage == "rename" else (0, 0, 0)
        assert (final.input_tokens, final.output_tokens, final.usage_exchanges) == expected_usage
    finally:
        release.set()
        assert uploader._thread is not None
        uploader._thread.join(1)
    assert not uploader._thread.is_alive()
    assert len(list(directory.glob("*.json"))) == int(stage == "rename")
    assert not list(directory.glob("*.tmp"))
    assert uploader.stats == final
    assert uploader.pending_current_run == final.pending_batches


def test_start_recovers_complete_temps_and_removes_bounded_partial_files(tmp_path: Path) -> None:
    """Recover complete sanitized crash leftovers while incomplete files cannot accumulate."""
    prior, run = str(uuid4()), str(uuid4())
    previous = tmp_path / prior
    previous.mkdir()
    complete = previous / f"{uuid4()}.tmp"
    complete.write_bytes(normalize_exchange(_usage_exchange(), max_body_bytes=4096)[0])
    for content in (b'{"resourceSpans":', b'{"resourceSpans":[]}', b"x" * 5000):
        (previous / f"{uuid4()}.tmp").write_bytes(content)
    uploader = CaptureUploader(
        "https://api.example",
        "org",
        run,
        "KEY",
        tmp_path / run,
        upload_origin=_UPLOAD_ORIGIN,
        upload_path_prefix=_UPLOAD_PREFIX,
        max_spool_bytes=4096,
    )
    uploader.start()
    final = uploader.close(timeout=0.02)
    assert list(previous.iterdir()) == [complete.with_suffix(".json")]
    assert final.pending_batches == 1
    assert final.dropped_exchanges == 3
    assert uploader.pending_current_run == 0
    assert (final.input_tokens, final.output_tokens, final.usage_exchanges) == (0, 0, 0)


def test_temporary_recovery_respects_combined_batch_count(tmp_path: Path) -> None:
    """Existing final batches and recovered temps share the same finite file budget."""
    run = str(uuid4())
    directory = tmp_path / run
    directory.mkdir()
    content = normalize_exchange(_exchange(), max_body_bytes=4096)[0]
    for _ in range(1024):
        (directory / f"{uuid4()}.json").write_bytes(content)
    temporary = directory / f"{uuid4()}.tmp"
    temporary.write_bytes(content)
    uploader = CaptureUploader(
        "https://api.example",
        "org",
        run,
        "KEY",
        directory,
        upload_origin=_UPLOAD_ORIGIN,
        upload_path_prefix=_UPLOAD_PREFIX,
    )
    uploader._recover_temporary_files()
    assert not temporary.exists()
    assert len(list(directory.iterdir())) == 1024
    assert uploader.stats.pending_batches == 1024
    assert uploader.stats.dropped_exchanges == 1


def test_delivery_of_durable_temp_removes_canonical_copy_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A delivery worker can finish publication without leaving a duplicate retry file."""
    run, ingest = str(uuid4()), str(uuid4())
    directory = tmp_path / run
    entered, release = threading.Event(), threading.Event()
    original_replace = Path.replace
    methods: list[str] = []

    def stalled_persistence_rename(path: Path, target: str | Path) -> Path:
        """Pause only the persistence worker while delivery publishes the same temp."""
        if threading.current_thread().name == "exp-capture-upload":
            entered.set()
            assert release.wait(3)
        return original_replace(path, target)

    def handler(request: httpx.Request) -> httpx.Response:
        """Acknowledge one immutable upload and its normal finalize receipt."""
        methods.append(request.method)
        if request.method == "PUT":
            assert "authorization" not in request.headers
            return httpx.Response(200)
        if request.url.path.endswith("/batches/upload"):
            return httpx.Response(
                200,
                json={"status": "pending", "ingest_id": ingest, "signed_url": _signed_url(ingest)},
            )
        return httpx.Response(202)

    monkeypatch.setattr(Path, "replace", stalled_persistence_rename)
    uploader = CaptureUploader(
        "https://api.example",
        "org",
        run,
        "KEY",
        directory,
        upload_origin=_UPLOAD_ORIGIN,
        upload_path_prefix=_UPLOAD_PREFIX,
        transport=httpx.MockTransport(handler),
    )
    uploader.start()
    try:
        assert uploader.submit(_exchange())
        assert entered.wait(1)
        _wait(lambda: uploader.stats.uploaded_batches == 1)
        assert not list(directory.iterdir())
    finally:
        release.set()
        final = uploader.close()
    assert methods == ["POST", "PUT", "POST"]
    assert final.uploaded_batches == 1
    assert final.pending_batches == final.dropped_exchanges == 0
    assert not list(directory.iterdir())


@pytest.mark.parametrize("stage", ["before_unlink", "after_unlink"])
def test_shutdown_commits_cloud_acceptance_before_cleanup_can_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    """Freeze accurate receipt counters whether local cleanup stalls before or after deletion."""
    run, ingest = str(uuid4()), str(uuid4())
    directory = tmp_path / run
    directory.mkdir()
    path = directory / f"{uuid4()}.json"
    path.write_bytes(normalize_exchange(_exchange(), max_body_bytes=4096)[0])
    entered, release = threading.Event(), threading.Event()
    original_unlink = Path.unlink

    def stalled_unlink(candidate: Path, missing_ok: bool = False) -> None:
        """Pause deletion on either side of the filesystem mutation."""
        if candidate != path:
            original_unlink(candidate, missing_ok=missing_ok)
            return
        if stage == "after_unlink":
            original_unlink(candidate, missing_ok=missing_ok)
        entered.set()
        assert release.wait(3)
        if stage == "before_unlink":
            original_unlink(candidate, missing_ok=missing_ok)

    def handler(request: httpx.Request) -> httpx.Response:
        """Return a verified completed batch without issuing another upload."""
        assert request.url.path.endswith("/batches/upload")
        return httpx.Response(200, json={"ingest_id": ingest, "status": "done"})

    monkeypatch.setattr(Path, "unlink", stalled_unlink)
    uploader = CaptureUploader(
        "https://api.example",
        "org",
        run,
        "KEY",
        directory,
        upload_origin=_UPLOAD_ORIGIN,
        upload_path_prefix=_UPLOAD_PREFIX,
        transport=httpx.MockTransport(handler),
    )
    uploader.start()
    try:
        assert entered.wait(1)
        assert uploader.stats.uploaded_batches == 1
        assert uploader.stats.pending_batches == uploader.pending_current_run == 0
        before = time.monotonic()
        final = uploader.close(timeout=0.02)
        assert time.monotonic() - before < 0.5
        assert final.uploaded_batches == 1
        assert final.pending_batches == uploader.pending_current_run == 0
    finally:
        release.set()
        assert uploader._delivery_thread is not None
        uploader._delivery_thread.join(1)
        uploader.close()
    assert not uploader._delivery_thread.is_alive()
    assert not path.exists()
    assert uploader.stats == final
    assert uploader.pending_current_run == 0


def test_failed_accepted_cleanup_retains_quota_and_retries_without_reupload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep one accepted cleanup slot while disk quota and subsequent uploads remain bounded."""
    run, ingest = str(uuid4()), str(uuid4())
    directory = tmp_path / run
    directory.mkdir()
    path = directory / f"{uuid4()}.json"
    payload = normalize_exchange(_exchange(), max_body_bytes=4096)[0]
    path.write_bytes(payload)
    original_unlink = Path.unlink
    failed = True
    requests: list[httpx.Request] = []

    def failing_unlink(candidate: Path, missing_ok: bool = False) -> None:
        """Simulate a persistent local deletion failure for the accepted batch."""
        if candidate == path and failed:
            raise PermissionError("synthetic cleanup failure")
        original_unlink(candidate, missing_ok=missing_ok)

    def handler(request: httpx.Request) -> httpx.Response:
        """Acknowledge completed cloud validation exactly once."""
        requests.append(request)
        return httpx.Response(200, json={"ingest_id": ingest, "status": "done"})

    monkeypatch.setattr(Path, "unlink", failing_unlink)
    uploader = CaptureUploader(
        "https://api.example",
        "org",
        run,
        "KEY",
        directory,
        upload_origin=_UPLOAD_ORIGIN,
        upload_path_prefix=_UPLOAD_PREFIX,
        max_spool_bytes=len(payload),
    )
    uploader._recover_temporary_files()
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        uploader._deliver_one(client)
        assert path.exists()
        assert uploader.stats.uploaded_batches == 1
        assert uploader.stats.pending_batches == uploader.pending_current_run == 0
        assert uploader.stats.upload_errors == 1
        with pytest.raises(ValueError, match="spool is full"):
            uploader._persist(_exchange())
        uploader._retry_at.clear()
        uploader._deliver_one(client)
        assert len(requests) == uploader.stats.uploaded_batches == 1
        assert uploader.stats.upload_errors == 2
        failed = False
        uploader._retry_at.clear()
        uploader._deliver_one(client)
    assert not path.exists()
    assert uploader._accepted_cleanup is None
    assert len(requests) == uploader.stats.uploaded_batches == 1
    assert uploader.close().pending_batches == 0
