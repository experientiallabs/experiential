"""Disposable subprocess boundary for deadline-aware structured provider calls."""

from __future__ import annotations

import json
import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from enum import StrEnum
from typing import Annotated, Literal, Protocol

from llm_waterfall import ChatRequest, ChatResponse
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from wmh.providers.base import ProviderConfig, ToolCallingProvider
from wmh.providers.failure_attribution import (
    ProviderFailureAttribution,
    ProviderFailureOwner,
    ProviderFailureReason,
    classify_provider_failure,
)
from wmh.providers.registry import get_provider

_FRAME_HEADER = struct.Struct("!I")
_MAX_FRAME_BYTES = 8 * 1024 * 1024
_PROCESS_TERMINATE_TIMEOUT_S = 2.0
_CANCEL_POLL_INTERVAL_S = 0.1


class RequestDeadline(Protocol):
    """Absolute caller-owned deadline used by one provider request."""

    def remaining_s(self) -> float: ...

    @property
    def limiting_source(self) -> RequestDeadlineSource: ...


class RequestDeadlineSource(StrEnum):
    """Which independent budget supplied an effective request deadline."""

    CALLER_BUDGET = "caller_budget"
    OPERATION_LIMIT = "operation_limit"


class ProviderWorkerDeadlineExceeded(TimeoutError):
    """The caller-owned deadline expired and the provider process was reaped."""

    def __init__(
        self,
        message: str,
        *,
        source: RequestDeadlineSource = RequestDeadlineSource.OPERATION_LIMIT,
    ) -> None:
        super().__init__(message)
        self.source = source


class ProviderWorkerUnavailable(RuntimeError):
    """The disposable provider process or its private protocol failed."""


class ProviderWorkerCleanupError(RuntimeError):
    """The disposable provider process could not be proved reaped."""


class ProviderWorkerFailure(RuntimeError):
    """One provider call failed with a sanitized ownership classification."""

    def __init__(self, attribution: ProviderFailureAttribution) -> None:
        super().__init__("provider worker request failed")
        self.attribution = attribution


class _StrictFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _InitializeFrame(_StrictFrame):
    kind: Literal["initialize"] = "initialize"
    provider_config: ProviderConfig


class _ReadyFrame(_StrictFrame):
    kind: Literal["ready"] = "ready"


class _CompletionRequestFrame(_StrictFrame):
    kind: Literal["complete_chat"] = "complete_chat"
    request: ChatRequest


class _CompletionFrame(_StrictFrame):
    kind: Literal["completion"] = "completion"
    response: ChatResponse


class _FailureFrame(_StrictFrame):
    kind: Literal["failure"] = "failure"
    owner: ProviderFailureOwner
    reason: ProviderFailureReason

    @property
    def attribution(self) -> ProviderFailureAttribution:
        return ProviderFailureAttribution(self.owner, self.reason)


_StartupResponse = Annotated[_ReadyFrame | _FailureFrame, Field(discriminator="kind")]
_CompletionResponse = Annotated[_CompletionFrame | _FailureFrame, Field(discriminator="kind")]
_STARTUP_RESPONSE_ADAPTER = TypeAdapter(_StartupResponse)
_COMPLETION_RESPONSE_ADAPTER = TypeAdapter(_CompletionResponse)


class _FrameDeadlineExceeded(TimeoutError):
    """A framed socket operation exhausted its absolute deadline."""


class _FrameProtocolError(RuntimeError):
    """The private worker channel closed or carried an invalid frame."""


class _FrameTooLarge(_FrameProtocolError):
    """A private worker frame exceeded its fixed byte ceiling."""


class ProviderProcessWorker:
    """Serve structured provider calls in one killable process per evaluation trial.

    The worker process inherits provider credentials from the trusted evaluator, but neither the
    candidate runner nor the task environment receives its socket. Killing the process is the hard
    cancellation boundary for synchronous SDK calls and any SDK-owned helper threads.
    """

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config.model_copy(deep=True)
        self._state_lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._cleanup_lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._socket: socket.socket | None = None
        self._starting = False
        self._ready = False
        self._cancelled = False
        self._cancelled_event = threading.Event()
        self._closed = threading.Event()
        self._cleanup_proved = False

    @property
    def is_ready(self) -> bool:
        """Return whether the child acknowledged its provider configuration."""
        with self._state_lock:
            return self._ready and not self._cancelled

    def start(self, deadline: RequestDeadline) -> None:
        """Start the child and prove readiness before the caller-owned deadline."""
        with self._start_lock:
            with self._state_lock:
                if self._ready and not self._cancelled:
                    return
                if self._cancelled:
                    raise ProviderWorkerUnavailable("provider worker is unavailable")
                if os.name != "posix":
                    self._cancelled = True
                    self._cleanup_proved = True
                    self._closed.set()
                    raise ProviderWorkerUnavailable(
                        "provider worker requires inherited POSIX sockets"
                    )
                self._starting = True

            try:
                parent_socket, child_socket = socket.socketpair()
            except OSError:
                with self._state_lock:
                    self._starting = False
                    self._cancelled = True
                    self._cleanup_proved = True
                    self._closed.set()
                raise ProviderWorkerUnavailable("provider worker failed to start") from None
            try:
                process = subprocess.Popen(  # noqa: S603 - fixed interpreter and module
                    _worker_command(child_socket.fileno()),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    close_fds=True,
                    pass_fds=(child_socket.fileno(),),
                    start_new_session=True,
                )
            except (OSError, ValueError, subprocess.SubprocessError):
                parent_socket.close()
                child_socket.close()
                with self._state_lock:
                    self._starting = False
                    self._cancelled = True
                    self._cleanup_proved = True
                    self._closed.set()
                raise ProviderWorkerUnavailable("provider worker failed to start") from None
            finally:
                child_socket.close()

            with self._state_lock:
                self._process = process
                self._socket = parent_socket
                self._starting = False
                cancelled = self._cancelled
            if cancelled:
                self._abort(force=True)
                raise ProviderWorkerUnavailable("provider worker is unavailable")

            try:
                _send_frame(
                    parent_socket,
                    _InitializeFrame(provider_config=self._config),
                    deadline=deadline,
                    cancelled=self._cancelled_event,
                )
                payload = _receive_frame(
                    parent_socket,
                    deadline=deadline,
                    cancelled=self._cancelled_event,
                )
                response = _STARTUP_RESPONSE_ADAPTER.validate_python(payload)
            except _FrameDeadlineExceeded:
                self._abort(force=True)
                raise ProviderWorkerDeadlineExceeded(
                    "provider worker startup deadline exceeded",
                    source=deadline.limiting_source,
                ) from None
            except (OSError, ValidationError, _FrameProtocolError):
                self._abort(force=True)
                raise ProviderWorkerUnavailable("provider worker startup failed") from None

            if isinstance(response, _FailureFrame):
                self._abort(force=True)
                raise ProviderWorkerFailure(response.attribution)
            with self._state_lock:
                if self._cancelled:
                    cancelled = True
                else:
                    self._ready = True
                    cancelled = False
            if cancelled:
                self._abort(force=True)
                raise ProviderWorkerUnavailable("provider worker is unavailable")

    def complete_chat(
        self,
        request: ChatRequest,
        deadline: RequestDeadline,
    ) -> ChatResponse:
        """Complete one structured request or terminate at the absolute deadline."""
        with self._io_lock:
            with self._state_lock:
                connection = self._socket
                process = self._process
                ready = self._ready and not self._cancelled
            if not ready or connection is None or process is None:
                raise ProviderWorkerUnavailable("provider worker is unavailable")
            if process.poll() is not None:
                self._abort(force=True)
                raise ProviderWorkerUnavailable("provider worker exited unexpectedly")
            try:
                _send_frame(
                    connection,
                    _CompletionRequestFrame(request=request),
                    deadline=deadline,
                    cancelled=self._cancelled_event,
                )
            except _FrameTooLarge:
                raise ProviderWorkerFailure(
                    ProviderFailureAttribution(
                        ProviderFailureOwner.CANDIDATE,
                        ProviderFailureReason.INVALID_REQUEST,
                    )
                ) from None
            except _FrameDeadlineExceeded:
                self._abort(force=True)
                raise ProviderWorkerDeadlineExceeded(
                    "provider request deadline exceeded",
                    source=deadline.limiting_source,
                ) from None
            except (OSError, ValidationError, _FrameProtocolError):
                self._abort(force=True)
                raise ProviderWorkerUnavailable("provider worker protocol failed") from None
            try:
                payload = _receive_frame(
                    connection,
                    deadline=deadline,
                    cancelled=self._cancelled_event,
                )
                response = _COMPLETION_RESPONSE_ADAPTER.validate_python(payload)
            except _FrameDeadlineExceeded:
                self._abort(force=True)
                raise ProviderWorkerDeadlineExceeded(
                    "provider request deadline exceeded",
                    source=deadline.limiting_source,
                ) from None
            except (OSError, ValidationError, _FrameProtocolError):
                self._abort(force=True)
                raise ProviderWorkerUnavailable("provider worker protocol failed") from None
            if isinstance(response, _FailureFrame):
                raise ProviderWorkerFailure(response.attribution)
            return response.response

    def cancel(self) -> None:
        """Cancel every in-flight request and reap the disposable child."""
        self._abort(force=True)

    def close(self) -> None:
        """Stop and reap the disposable child. Safe to call more than once."""
        self._abort(force=False)

    def wait_closed(self, timeout_s: float) -> bool:
        """Wait until the worker process has been reaped."""
        if not self._closed.wait(timeout_s):
            return False
        with self._state_lock:
            return self._cleanup_proved

    def _abort(self, *, force: bool) -> None:
        with self._state_lock:
            self._cancelled = True
            self._cancelled_event.set()
            connection = self._socket
            process = self._process
            starting = self._starting
        _close_socket(connection)
        cleanup_failed = False
        try:
            with self._cleanup_lock:
                if process is not None:
                    _stop_and_reap(process, force=force)
        except BaseException:
            cleanup_failed = True
            raise
        finally:
            with self._state_lock:
                if not cleanup_failed and self._process is process:
                    self._process = None
                if self._socket is connection:
                    self._socket = None
                self._ready = False
                if not starting:
                    self._cleanup_proved = not cleanup_failed
                    self._closed.set()


def _worker_command(socket_fd: int) -> list[str]:
    """Return the fixed child command for one inherited private socket."""
    return [sys.executable, "-m", "wmh.providers.process_worker", str(socket_fd)]


def _stop_and_reap(process: subprocess.Popen[bytes], *, force: bool) -> None:
    """Stop the worker process group and prove its direct child has been reaped."""
    process_group = process.pid
    try:
        _signal_process_group(
            process_group,
            signal.SIGKILL if force else signal.SIGTERM,
        )
        if not force:
            try:
                process.wait(timeout=_PROCESS_TERMINATE_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                pass
        if _process_group_exists(process_group):
            _signal_process_group(process_group, signal.SIGKILL)
        try:
            process.wait(timeout=_PROCESS_TERMINATE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            _signal_process_group(process_group, signal.SIGKILL)
            process.wait(timeout=_PROCESS_TERMINATE_TIMEOUT_S)
        _wait_for_process_group_exit(process_group)
    except (OSError, subprocess.TimeoutExpired):
        raise ProviderWorkerCleanupError("provider worker cleanup was not proved") from None
    if process.poll() is None:
        raise ProviderWorkerCleanupError("provider worker cleanup was not proved")


def _signal_process_group(process_group: int, signal_number: signal.Signals) -> None:
    try:
        os.killpg(process_group, signal_number)
    except ProcessLookupError:
        pass


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_for_process_group_exit(process_group: int) -> None:
    expires_at = time.monotonic() + _PROCESS_TERMINATE_TIMEOUT_S
    while _process_group_exists(process_group):
        if time.monotonic() >= expires_at:
            raise ProviderWorkerCleanupError("provider worker cleanup was not proved")
        time.sleep(0.01)


def _close_socket(connection: socket.socket | None) -> None:
    if connection is None:
        return
    try:
        connection.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        connection.close()
    except OSError:
        pass


def _send_frame(
    connection: socket.socket,
    frame: BaseModel,
    *,
    deadline: RequestDeadline | None = None,
    cancelled: threading.Event | None = None,
) -> None:
    body = frame.model_dump_json().encode("utf-8")
    if len(body) > _MAX_FRAME_BYTES:
        raise _FrameTooLarge("provider worker frame exceeded its limit")
    packet = _FRAME_HEADER.pack(len(body)) + body
    sent = 0
    while sent < len(packet):
        _set_socket_deadline(connection, deadline, cancelled=cancelled)
        try:
            written = connection.send(packet[sent:])
        except TimeoutError:
            _check_frame_wait(deadline, cancelled=cancelled)
            continue
        if written == 0:
            raise _FrameProtocolError("provider worker channel closed")
        sent += written


def _receive_frame(
    connection: socket.socket,
    *,
    deadline: RequestDeadline | None = None,
    cancelled: threading.Event | None = None,
) -> dict[str, object]:
    header = _receive_exact(
        connection,
        _FRAME_HEADER.size,
        deadline=deadline,
        cancelled=cancelled,
    )
    size = _FRAME_HEADER.unpack(header)[0]
    if size > _MAX_FRAME_BYTES:
        raise _FrameTooLarge("provider worker frame exceeded its limit")
    body = _receive_exact(connection, size, deadline=deadline, cancelled=cancelled)
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise _FrameProtocolError("provider worker sent invalid JSON") from None
    if not isinstance(payload, dict):
        raise _FrameProtocolError("provider worker sent a non-object frame")
    return payload


def _receive_exact(
    connection: socket.socket,
    size: int,
    *,
    deadline: RequestDeadline | None,
    cancelled: threading.Event | None,
) -> bytes:
    chunks: list[bytes] = []
    received = 0
    while received < size:
        _set_socket_deadline(connection, deadline, cancelled=cancelled)
        try:
            chunk = connection.recv(size - received)
        except TimeoutError:
            _check_frame_wait(deadline, cancelled=cancelled)
            continue
        if not chunk:
            raise _FrameProtocolError("provider worker channel closed")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def _set_socket_deadline(
    connection: socket.socket,
    deadline: RequestDeadline | None,
    *,
    cancelled: threading.Event | None,
) -> None:
    if cancelled is not None and cancelled.is_set():
        raise _FrameProtocolError("provider worker was cancelled")
    if deadline is None:
        timeout_s = None if cancelled is None else _CANCEL_POLL_INTERVAL_S
        connection.settimeout(timeout_s)
        return
    remaining_s = deadline.remaining_s()
    if remaining_s <= 0:
        raise _FrameDeadlineExceeded
    if cancelled is not None:
        remaining_s = min(remaining_s, _CANCEL_POLL_INTERVAL_S)
    connection.settimeout(remaining_s)


def _check_frame_wait(
    deadline: RequestDeadline | None,
    *,
    cancelled: threading.Event | None,
) -> None:
    if cancelled is not None and cancelled.is_set():
        raise _FrameProtocolError("provider worker was cancelled")
    if deadline is not None and deadline.remaining_s() <= 0:
        raise _FrameDeadlineExceeded


def _serve_worker(socket_fd: int) -> int:
    connection = socket.socket(fileno=socket_fd)
    try:
        try:
            initialization = _InitializeFrame.model_validate(_receive_frame(connection))
            provider = get_provider(initialization.provider_config)
            if not isinstance(provider, ToolCallingProvider):
                raise TypeError("configured provider has no structured chat capability")
        except Exception:  # noqa: BLE001 - provider construction errors never cross the channel
            _send_frame(
                connection,
                _FailureFrame(
                    owner=ProviderFailureOwner.INFRASTRUCTURE,
                    reason=ProviderFailureReason.CONFIGURATION,
                ),
            )
            return 1
        _send_frame(connection, _ReadyFrame())
        while True:
            try:
                request = _CompletionRequestFrame.model_validate(_receive_frame(connection))
            except _FrameProtocolError:
                return 0
            except Exception:  # noqa: BLE001 - malformed private requests return one safe frame
                _send_frame(
                    connection,
                    _FailureFrame(
                        owner=ProviderFailureOwner.INFRASTRUCTURE,
                        reason=ProviderFailureReason.UNKNOWN,
                    ),
                )
                return 1
            try:
                response = provider.complete_chat(request.request)
            except Exception as exc:  # noqa: BLE001 - classify, never serialize provider text
                attribution = classify_provider_failure(exc)
                _send_frame(
                    connection,
                    _FailureFrame(
                        owner=attribution.owner,
                        reason=attribution.reason,
                    ),
                )
                continue
            try:
                _send_frame(connection, _CompletionFrame(response=response))
            except _FrameProtocolError:
                return 1
    finally:
        _close_socket(connection)


def _main() -> int:
    if len(sys.argv) != 2:
        return 2
    try:
        socket_fd = int(sys.argv[1])
    except ValueError:
        return 2
    if socket_fd < 0:
        return 2
    return _serve_worker(socket_fd)


if __name__ == "__main__":
    raise SystemExit(_main())
