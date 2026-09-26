"""Bounded asynchronous capture delivery with credential-free durable retry files."""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from uuid import UUID, uuid4

import httpx

from exp.common.core.artifacts import JsonValue
from exp.runtime.capture.normalization import CapturedExchange, normalize_exchange

logger = logging.getLogger(__name__)
_MAX_BATCH_BYTES = 50 * 1024 * 1024


@dataclass(frozen=True)
class UploadStats:
    """Content-free capture delivery counters for status and run heartbeats."""

    pending_batches: int
    upload_errors: int
    dropped_exchanges: int
    captured_exchanges: int
    uploaded_batches: int
    input_tokens: int = 0
    output_tokens: int = 0
    usage_exchanges: int = 0


class CaptureUploader:
    """Normalize copies off the inference path and retry signed cloud uploads."""

    def __init__(
        self,
        base_url: str,
        org_id: str,
        run_id: str,
        api_key: str,
        spool_dir: Path,
        *,
        upload_origin: str,
        upload_path_prefix: str,
        max_body_bytes: int = 8 * 1024 * 1024,
        max_queue_bytes: int = 32 * 1024 * 1024,
        max_spool_bytes: int = 64 * 1024 * 1024,
        transport: httpx.BaseTransport | None = None,
        on_diagnostic: Callable[[str], None] | None = None,
    ) -> None:
        """Bind one organization and origin-scoped spool to authenticated delivery.

        Args:
            base_url: Validated Platform API origin.
            org_id: Authenticated organization ID.
            run_id: Client-generated UUID for this local capture run.
            api_key: Normal Platform API key, retained only in process memory.
            spool_dir: Private origin/org/run directory supplied by orchestration.
            upload_origin: Storage origin acknowledged by the trusted Platform at run start.
            upload_path_prefix: Organization-bound signed-upload path pinned at run start.
            max_body_bytes: Maximum decompressed bytes for either model body.
            max_queue_bytes: Maximum raw exchange bytes waiting for the worker.
            max_spool_bytes: Maximum sanitized files across runs in this spool's parent.
            transport: Optional HTTP transport for deterministic integration tests.
            on_diagnostic: Optional content-free receipts emitted by background workers.
        """
        UUID(run_id)
        if min(max_body_bytes, max_queue_bytes, max_spool_bytes) < 1:
            raise ValueError("capture upload limits must be positive")
        if spool_dir.name != run_id:
            raise ValueError("capture spool directory must be named for its run UUID")
        self._upload_origin = _upload_scope(base_url, upload_origin, upload_path_prefix, org_id)
        self._upload_path_prefix = upload_path_prefix
        self._base = f"{base_url.rstrip('/')}/api/orgs/{org_id}"
        self._api_key = api_key
        self._spool_dir = spool_dir
        self._max_body_bytes = max_body_bytes
        self._max_queue_bytes = max_queue_bytes
        self._max_spool_bytes = max_spool_bytes
        self._transport = transport
        self._on_diagnostic = on_diagnostic
        self._diagnostics: queue.Queue[str] = queue.Queue(maxsize=128)
        self._diagnostics_done = threading.Event()
        self._diagnostic_thread: threading.Thread | None = None
        self._diagnostics_lost = 0
        self._queue: queue.Queue[CapturedExchange] = queue.Queue(maxsize=64)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._abandon = threading.Event()
        self._thread: threading.Thread | None = None
        self._delivery_thread: threading.Thread | None = None
        self._queued_bytes = 0
        self._captured = 0
        self._dropped = 0
        self._errors = 0
        self._uploaded = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._usage_exchanges = 0
        self._pending_exchanges = 0
        self._durable_temps: set[Path] = set()
        self._pending_paths: set[Path] = set()
        self._accepted_cleanup: Path | None = None
        self._final_stats: UploadStats | None = None
        self._final_pending_current = 0
        self._retry_at: dict[Path, float] = {}

    def start(self) -> None:
        """Start one background worker without performing cloud requests inline."""
        if self._thread is not None:
            raise RuntimeError("capture uploader already started")
        for directory in (self._spool_dir, *self._spool_dir.parents):
            if directory.is_symlink():
                raise ValueError("capture spool cannot use symbolic links")
        self._spool_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        for directory in (self._spool_dir, self._spool_dir.parent):
            if directory.stat().st_uid != os.getuid():
                raise ValueError("capture spool must belong to the current user")
            directory.chmod(0o700)
        self._recover_temporary_files()
        if self._on_diagnostic is not None:
            self._diagnostic_thread = threading.Thread(
                target=self._report_diagnostics, name="exp-capture-diagnostics", daemon=True
            )
            self._diagnostic_thread.start()
        self._thread = threading.Thread(target=self._work, name="exp-capture-upload", daemon=True)
        self._thread.start()
        self._delivery_thread = threading.Thread(
            target=self._deliver, name="exp-capture-delivery", daemon=True
        )
        self._delivery_thread.start()

    def submit(self, exchange: CapturedExchange) -> bool:
        """Accept a finite raw copy immediately, returning false on queue pressure."""
        with self._lock:
            if (
                self._stop.is_set()
                or self._queued_bytes + exchange.byte_count > self._max_queue_bytes
            ):
                self._dropped += 1
                return False
            try:
                self._queue.put_nowait(exchange)
            except queue.Full:
                self._dropped += 1
                return False
            self._queued_bytes += exchange.byte_count
            self._captured += 1
            self._pending_exchanges += 1
            return True

    @property
    def pending_current_run(self) -> int:
        """Count only this run for its heartbeat, excluding recovered sibling runs."""
        with self._lock:
            if self._final_stats is not None:
                return self._final_pending_current
            files = sum(path.parent == self._spool_dir for path in self._pending_paths)
            return self._pending_exchanges + files

    @property
    def stats(self) -> UploadStats:
        """Read delivery counters without ever exposing model content or credentials."""
        with self._lock:
            if self._final_stats is not None:
                return self._final_stats
            return UploadStats(
                pending_batches=self._pending_exchanges + len(self._pending_paths),
                upload_errors=self._errors,
                dropped_exchanges=self._dropped,
                captured_exchanges=self._captured,
                uploaded_batches=self._uploaded,
                input_tokens=self._input_tokens,
                output_tokens=self._output_tokens,
                usage_exchanges=self._usage_exchanges,
            )

    def close(self, timeout: float = 5.0) -> UploadStats:
        """Stop accepting copies and allow a bounded local drain before returning.

        Sanitized files survive cloud failure and are recovered by the next run.
        A daemon worker never delays application exit beyond the requested timeout.
        Copies unfinished at the deadline are counted as dropped, not durable pending.
        """
        with self._lock:
            if self._final_stats is not None:
                return self._final_stats
        self._stop.set()
        deadline = time.monotonic() + max(timeout, 0.0)
        if self._thread is not None:
            self._thread.join(max(deadline - time.monotonic(), 0.0))
        if self._delivery_thread is not None:
            self._delivery_thread.join(max(deadline - time.monotonic(), 0.0))
        with self._lock:
            self._abandon.set()
            self._dropped += self._pending_exchanges
            self._pending_exchanges = 0
            self._queued_bytes = 0
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
                self._queue.task_done()
            self._final_pending_current = sum(
                path.parent == self._spool_dir for path in self._pending_paths
            )
            self._final_stats = UploadStats(
                pending_batches=len(self._pending_paths),
                upload_errors=self._errors,
                dropped_exchanges=self._dropped,
                captured_exchanges=self._captured,
                uploaded_batches=self._uploaded,
                input_tokens=self._input_tokens,
                output_tokens=self._output_tokens,
                usage_exchanges=self._usage_exchanges,
            )
            final_stats = self._final_stats
        self._diagnostics_done.set()
        if self._diagnostic_thread is not None:
            self._diagnostic_thread.join(max(deadline - time.monotonic(), 0.0))
        return final_stats

    def _work(self) -> None:
        """Drain copied bodies to local storage independently of cloud availability."""
        while not self._stop.is_set() or not self._queue.empty():
            try:
                exchange = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if not self._abandon.is_set():
                    self._persist(exchange)
            except (ValueError, OSError, RecursionError, TypeError):
                with self._lock:
                    if not self._abandon.is_set():
                        self._dropped += 1
                        self._pending_exchanges -= 1
                        self._queued_bytes -= exchange.byte_count
                trace = (
                    exchange.trace_id
                    if re.fullmatch(r"[0-9a-f]{32}", exchange.trace_id)
                    else "unknown"
                )
                self._diagnostic(
                    f"capture_dropped: normalization or local storage failed · trace {trace}"
                )
            finally:
                self._queue.task_done()

    def _deliver(self) -> None:
        """Keep slow cloud requests off both the inference and local persistence paths."""
        with httpx.Client(
            timeout=httpx.Timeout(5.0),
            transport=self._transport,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            while not self._stop.wait(0.1):
                self._deliver_one(client)

    def _persist(self, exchange: CapturedExchange) -> None:
        """Write only sanitized OTLP, with finite per-origin/org spool capacity."""
        payload, usage, receipt = normalize_exchange(exchange, max_body_bytes=self._max_body_bytes)
        if self._abandon.is_set():
            return
        if len(payload) > _MAX_BATCH_BYTES:
            raise ValueError("normalized capture exceeds the cloud batch limit")
        files = self._files()
        occupied = 0
        for path in files:
            try:
                occupied += path.stat().st_size
            except FileNotFoundError:
                # The independent delivery worker can finish a batch during accounting.
                continue
        if occupied + len(payload) > self._max_spool_bytes or len(files) >= 1024:
            raise ValueError("capture spool is full")
        destination = self._spool_dir / f"{uuid4()}.json"
        temporary = destination.with_suffix(".tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            with self._lock:
                if self._abandon.is_set():
                    return
                # The complete sanitized file is durable before publication. A late
                # rename must not turn a deadline-accounted drop into a queued batch.
                self._durable_temps.add(temporary)
                self._pending_paths.add(destination)
                self._pending_exchanges -= 1
                self._queued_bytes -= exchange.byte_count
                if usage is not None:
                    self._input_tokens += usage[0]
                    self._output_tokens += usage[1]
                    self._usage_exchanges += 1
            self._diagnostic(f"capture_saved · batch {destination.stem} · {receipt}")
            try:
                temporary.replace(destination)
            except OSError:
                # A complete temp remains eligible for delivery and crash recovery.
                return
            with self._lock:
                self._durable_temps.discard(temporary)
        finally:
            with self._lock:
                durable = temporary in self._durable_temps
            if not durable:
                temporary.unlink(missing_ok=True)

    def _diagnostic(self, event: str) -> None:
        """Drop excess diagnostic messages rather than block capture storage or delivery."""
        if self._on_diagnostic is not None:
            try:
                self._diagnostics.put_nowait(event)
            except queue.Full:
                with self._lock:
                    self._diagnostics_lost += 1

    def _report_diagnostics(self) -> None:
        """Isolate slow or blocked output in one daemon with a bounded message queue."""
        assert self._on_diagnostic is not None
        while True:
            try:
                event = self._diagnostics.get(timeout=0.1)
            except queue.Empty:
                event = None
            with self._lock:
                lost = self._diagnostics_lost
            if lost:
                try:
                    self._on_diagnostic(f"diagnostic_events_dropped: {lost}")
                except Exception:  # noqa: BLE001 - Preserve the count until output recovers.
                    pass
                else:
                    with self._lock:
                        self._diagnostics_lost -= lost
            if event is None:
                if self._diagnostics_done.is_set():
                    return
                continue
            try:
                self._on_diagnostic(event)
            except Exception:  # noqa: BLE001 - Diagnostics must never drop a captured request.
                with self._lock:
                    self._diagnostics_lost += 1

    def _recover_temporary_files(self) -> None:
        """Adopt bounded complete sanitized temps and remove incomplete crash leftovers."""
        existing = self._files()
        occupied = sum(path.stat().st_size for path in existing)
        file_count = len(existing)
        for path in islice(self._spool_dir.parent.glob("*/*.tmp"), 1024):
            if (
                path.parent.is_symlink()
                or path.is_symlink()
                or not path.is_file()
                or not _uuid(path.parent.name)
                or not _uuid(path.stem)
            ):
                continue
            destination = path.with_suffix(".json")
            if destination.exists():
                path.unlink()
                continue
            try:
                size = path.stat().st_size
                if (
                    file_count >= 1024
                    or size > _MAX_BATCH_BYTES
                    or occupied + size > self._max_spool_bytes
                ):
                    raise ValueError("temporary capture exceeds the spool limit")
                payload = json.loads(path.read_bytes())
                if not _complete_capture_payload(payload):
                    raise ValueError("temporary capture is incomplete")
            except (ValueError, RecursionError):
                path.unlink(missing_ok=True)
                self._dropped += 1
                continue
            path.replace(destination)
            occupied += size
            file_count += 1
        files = self._files()
        with self._lock:
            self._pending_paths = {path.with_suffix(".json") for path in files}

    def _files(self) -> list[Path]:
        """Find bounded retry files only in UUID-named sibling run directories."""
        result: list[Path] = []
        for directory in self._spool_dir.parent.glob("*"):
            if directory.is_symlink() or not directory.is_dir() or not _uuid(directory.name):
                continue
            for path in directory.glob("*.json"):
                if not path.is_symlink() and path.is_file() and _uuid(path.stem):
                    result.append(path)
                    if len(result) >= 1024:
                        break
            if len(result) >= 1024:
                break
        with self._lock:
            temporary_files = tuple(self._durable_temps)
        unique = {(path.parent, path.stem): path for path in result}
        for temporary in temporary_files:
            published = temporary.with_suffix(".json")
            unique[(temporary.parent, temporary.stem)] = (
                published if published.is_file() else temporary
            )
        return sorted(unique.values())

    def _deliver_one(self, client: httpx.Client) -> None:
        """Retry one eligible batch while never retaining an unbounded error history."""
        now = time.monotonic()
        if self._cleanup_accepted(client, now):
            return
        files = self._files()
        self._retry_at = {
            path: deadline for path, deadline in self._retry_at.items() if path in files
        }
        for path in files:
            if self._retry_at.get(path, 0) > now:
                continue
            try:
                if path.suffix == ".tmp":
                    destination = path.with_suffix(".json")
                    try:
                        path.replace(destination)
                    except FileNotFoundError:
                        if not destination.is_file():
                            raise
                    with self._lock:
                        self._durable_temps.discard(path)
                    path = destination
                accepted = self._upload(client, path)
            except (httpx.HTTPError, ValueError, OSError, KeyError, TypeError):
                with self._lock:
                    if not self._abandon.is_set():
                        self._errors += 1
                self._retry_at[path] = now + 10.0
                self._diagnostic(f"upload_deferred · batch {path.stem}")
                logger.warning("Capture upload deferred; sanitized batch remains queued locally")
            else:
                if not accepted:
                    self._retry_at[path] = now + 10.0
                    return
                with self._lock:
                    if self._abandon.is_set():
                        return
                    # Receipt accounting commits before cleanup can block. Shutdown
                    # snapshots this state without racing a filesystem enumeration.
                    self._pending_paths.discard(path)
                    self._accepted_cleanup = path
                    self._uploaded += 1
                self._diagnostic(f"upload_accepted · batch {path.stem}")
                self._cleanup_accepted(client, now)
            return

    def _cleanup_accepted(self, client: httpx.Client, now: float) -> bool:
        """Retry one accepted file's deletion before accepting any more cloud receipts."""
        with self._lock:
            path = self._accepted_cleanup
        if path is None:
            return False
        if self._retry_at.get(path, 0) > now:
            return True
        try:
            path.unlink(missing_ok=True)
        except OSError:
            with self._lock:
                if not self._abandon.is_set():
                    self._errors += 1
            self._retry_at[path] = now + 10.0
            logger.warning("Capture uploaded; local cleanup deferred within the spool quota")
        else:
            self._retry_at.pop(path, None)
            with self._lock:
                self._accepted_cleanup = None
            if path.parent != self._spool_dir and not self._abandon.is_set():
                self._finish_recovered_run(client, path.parent)
        return True

    def _finish_recovered_run(self, client: httpx.Client, directory: Path) -> None:
        """Refresh an old run's pending count without reopening its capture lifetime."""
        with self._lock:
            pending = sum(path.parent == directory for path in self._pending_paths)
        try:
            response = client.post(
                f"{self._base}/capture/runs/{directory.name}/end",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"pending_batches": pending, "upload_errors": 0},
            )
            response.raise_for_status()
        except httpx.HTTPError:
            with self._lock:
                if not self._abandon.is_set():
                    self._errors += 1
            logger.warning("Recovered capture was uploaded; its run status could not be refreshed")

    def _upload(self, client: httpx.Client, path: Path) -> bool:
        """Return true only after our own PUT receipt plus finalize, or cloud completion."""
        if path.stat().st_size > self._max_spool_bytes:
            raise ValueError("capture retry file exceeds the spool limit")
        headers = {"Authorization": f"Bearer {self._api_key}"}
        response = client.post(
            f"{self._base}/capture/runs/{path.parent.name}/batches/upload",
            headers=headers,
            json={"batch_id": path.stem, "source_kind": "otlp"},
        )
        response.raise_for_status()
        ticket = response.json()
        if not isinstance(ticket, dict):
            raise ValueError("capture upload response is not an object")
        ingest_id = ticket.get("ingest_id")
        if not isinstance(ingest_id, str) or not _uuid(ingest_id):
            raise ValueError("capture upload response lacks a canonical ingest ID")
        if ticket.get("status") == "running":
            # Running alone can mean finalize happened before any bytes reached Storage.
            # Keep an uncertain retry until the worker acknowledges verified completion.
            return False
        if ticket.get("status") == "done":
            return True
        if ticket.get("status") == "error":
            raise ValueError("capture batch failed cloud validation; retained locally")
        signed_url = ticket.get("signed_url")
        if not isinstance(signed_url, str):
            raise ValueError("capture upload response lacks a ticket")
        parsed = self._signed_destination(signed_url, ingest_id)
        uploaded = client.put(
            parsed,
            content=path.read_bytes(),
            headers={"Content-Type": "application/octet-stream"},
        )
        if uploaded.status_code != 409:
            uploaded.raise_for_status()
        finalized = client.post(
            f"{self._base}/telemetry/traces/{ingest_id}/finalize",
            headers=headers,
        )
        finalized.raise_for_status()
        return True

    def _signed_destination(self, signed_url: str, ingest_id: str) -> httpx.URL:
        """Reject a later ticket that disagrees with this run's approved storage scope."""
        try:
            parsed = httpx.URL(signed_url)
        except httpx.InvalidURL as error:
            raise ValueError("capture upload destination is malformed") from error
        origin = self._upload_origin
        if (
            parsed.userinfo
            or parsed.fragment
            or (parsed.scheme, parsed.host, parsed.port)
            != (origin.scheme, origin.host, origin.port)
        ):
            raise ValueError("capture upload destination differs from the approved storage origin")
        if str(UUID(ingest_id)) != ingest_id:
            raise ValueError("capture upload response has a noncanonical ingest ID")
        prefix = f"{self._upload_path_prefix}{ingest_id}/"
        raw_path = parsed.raw_path.partition(b"?")[0].decode("ascii")
        if not raw_path.startswith(prefix) or not re.fullmatch(
            r"[A-Za-z0-9_-]{43}", raw_path[len(prefix) :]
        ):
            raise ValueError("capture upload destination differs from the approved ingest path")
        return parsed


def _upload_scope(
    base_url: str, upload_origin: str, upload_path_prefix: str, org_id: str
) -> httpx.URL:
    """Validate the run-start storage policy without claiming independent cloud trust."""
    try:
        api = httpx.URL(base_url)
        origin = httpx.URL(upload_origin)
        destination = httpx.URL(upload_origin.rstrip("/") + upload_path_prefix)
    except httpx.InvalidURL as error:
        raise ValueError("capture run returned a malformed storage destination") from error
    local_http = _loopback_http(api) and _loopback_http(origin)
    if (
        (origin.scheme != "https" and not local_http)
        or not origin.host
        or origin.userinfo
        or origin.query
        or origin.fragment
        or origin.raw_path != b"/"
    ):
        raise ValueError("capture run storage origin must use HTTPS or explicit loopback HTTP")
    if (
        not upload_path_prefix.startswith("/")
        or destination.raw_path.decode("ascii") != upload_path_prefix
        or destination.query
        or destination.fragment
        or not re.search(
            r"/storage/v1/object/upload/sign/[^/]+/orgs/"
            + re.escape(org_id)
            + r"/telemetry-traces/otlp/$",
            upload_path_prefix,
        )
    ):
        raise ValueError("capture run storage prefix must be a canonical organization upload path")
    return origin


def _loopback_http(url: httpx.URL) -> bool:
    """Recognize the explicit local-development HTTP origins accepted by capture login."""
    return url.scheme == "http" and url.host in {"localhost", "127.0.0.1", "::1"}


def _complete_capture_payload(payload: JsonValue) -> bool:
    """Recognize the complete single-span envelope emitted by capture normalization."""
    for key in ("resourceSpans", "scopeSpans", "spans"):
        if not isinstance(payload, dict):
            return False
        children = payload.get(key)
        if not isinstance(children, list) or len(children) != 1:
            return False
        payload = children[0]
    if not isinstance(payload, dict):
        return False
    span = payload
    attributes = span.get("attributes")
    if not isinstance(attributes, list):
        return False
    keys: set[str] = set()
    for attribute in attributes:
        if not isinstance(attribute, dict):
            return False
        key, value = attribute.get("key"), attribute.get("value")
        if not isinstance(key, str) or not isinstance(value, dict) or len(value) != 1:
            return False
        scalar = next(iter(value.values()))
        if not (
            ("stringValue" in value and isinstance(scalar, str))
            or ("boolValue" in value and isinstance(scalar, bool))
            or ("intValue" in value and isinstance(scalar, str) and scalar.lstrip("-").isdigit())
        ):
            return False
        keys.add(key)
    trace_id, span_id = span.get("traceId"), span.get("spanId")
    started, ended = span.get("startTimeUnixNano"), span.get("endTimeUnixNano")
    return (
        isinstance(trace_id, str)
        and re.fullmatch(r"[0-9a-f]{32}", trace_id) is not None
        and isinstance(span_id, str)
        and re.fullmatch(r"[0-9a-f]{16}", span_id) is not None
        and isinstance(started, str)
        and started.isdigit()
        and isinstance(ended, str)
        and ended.isdigit()
        and span.get("name") == "captured model request"
        and span.get("kind") == 3
        and span.get("status") in ({"code": 1}, {"code": 2})
        and {"gen_ai.request.model", "gen_ai.input.messages", "exp.capture.protocol"}.issubset(keys)
    )


def _uuid(value: str) -> bool:
    """Recognize canonical UUID filenames without allowing arbitrary path components."""
    try:
        return str(UUID(value)) == value
    except ValueError:
        return False
