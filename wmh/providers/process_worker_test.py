"""Offline tests for the disposable provider subprocess boundary."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import cast

import pytest
from llm_waterfall import ChatRequest, ChatResponse

import wmh.providers.process_worker as mod
from wmh.harness.pi_runner import TurnDeadline
from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.providers.failure_attribution import (
    ProviderFailureOwner,
    ProviderFailureReason,
)
from wmh.tracking.budget import (
    BudgetAccount,
    BudgetPolicy,
    BudgetScope,
    ProviderCostMeter,
    ReservationStatus,
    SpendLedger,
    TokenPriceCeiling,
)

pytestmark = pytest.mark.skipif(os.name != "posix", reason="provider worker uses inherited sockets")

_CONFIG = ProviderConfig(
    kind=ProviderKind.BEDROCK,
    model_type="claude-haiku-4-5",
    model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
)
_REQUEST = ChatRequest.model_validate(
    {
        "messages": [{"role": "user", "content": "hello"}],
        "max_completion_tokens": 32,
    }
)
_COMPLETION = {
    "choices": [
        {
            "message": {"role": "assistant", "content": "done"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    "provider_receipt": {
        "provider": "bedrock",
        "provider_request_id": "request-1",
        "response_id": None,
        "requested_model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "response_model": None,
        "system_fingerprint": None,
        "request_digest": "sha256:" + "a" * 64,
        "temperature": 0.7,
        "max_tokens": 32,
        "max_tokens_field": "inferenceConfig.maxTokens",
        "seed_supplied": False,
        "cache_config_supplied": False,
        "started_at_unix_s": 10.0,
        "finished_at_unix_s": 11.0,
    },
}


def _scripted_worker_command(
    monkeypatch: pytest.MonkeyPatch,
    response: dict[str, object],
    *,
    response_delay_s: float = 0.0,
    ready_delay_s: float = 0.0,
) -> None:
    response_json = json.dumps(response, sort_keys=True, separators=(",", ":"))
    script = f"""
import json
import socket
import struct
import sys
import time

connection = socket.socket(fileno=int(sys.argv[1]))

def receive():
    header = b""
    while len(header) < 4:
        chunk = connection.recv(4 - len(header))
        if not chunk:
            raise SystemExit(2)
        header += chunk
    size = struct.unpack("!I", header)[0]
    body = b""
    while len(body) < size:
        chunk = connection.recv(size - len(body))
        if not chunk:
            raise SystemExit(3)
        body += chunk
    return json.loads(body)

def send(payload):
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    connection.sendall(struct.pack("!I", len(body)) + body)

receive()
time.sleep({ready_delay_s!r})
send({{"kind": "ready"}})
receive()
time.sleep({response_delay_s!r})
send(json.loads({response_json!r}))
"""

    def command(fd: int) -> list[str]:
        return [sys.executable, "-c", script, str(fd)]

    monkeypatch.setattr(mod, "_worker_command", command)


def _active_process(worker: mod.ProviderProcessWorker) -> subprocess.Popen[bytes]:
    process = worker._process
    assert process is not None
    return process


def _assert_reaped(process: subprocess.Popen[bytes]) -> None:
    assert process.poll() is not None
    with pytest.raises(ChildProcessError):
        os.waitpid(process.pid, os.WNOHANG)


def test_production_worker_starts_without_making_a_provider_request() -> None:
    worker = mod.ProviderProcessWorker(_CONFIG)

    worker.start(TurnDeadline.after(5))
    assert worker.is_ready
    process = _active_process(worker)

    worker.close()
    assert worker.wait_closed(0)
    _assert_reaped(process)


def test_worker_child_wraps_provider_with_registered_crash_safe_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProvider:
        config = _CONFIG
        paid_request_attempts = 1

        def complete_chat(self, _request: ChatRequest) -> ChatResponse:
            return ChatResponse.model_validate(_COMPLETION)

    policy = BudgetPolicy(
        study_id="worker-budget-test",
        manifest_digest="sha256:" + "b" * 64,
        hard_limit_nano_usd=1_000_000,
        phase_limits_nano_usd={"confirmation": 1_000_000},
        meters={
            "worker": ProviderCostMeter(
                provider_config=_CONFIG,
                price=TokenPriceCeiling(
                    input_nano_usd_per_token=1,
                    output_nano_usd_per_token=5,
                ),
            )
        },
    )
    account = BudgetAccount(
        ledger_path=(tmp_path / "budget.sqlite3").resolve(),
        policy=policy,
        scope=BudgetScope(
            phase="confirmation",
            category="worker",
            run_id="arm-1",
            lane="haiku",
            arm="baseline",
        ),
        meter_id="worker",
    )
    monkeypatch.setattr(mod, "get_provider", lambda _config: FakeProvider())
    server, client = socket.socketpair()
    server_fd = server.detach()
    exit_codes: list[int] = []
    worker_thread = threading.Thread(target=lambda: exit_codes.append(mod._serve_worker(server_fd)))
    worker_thread.start()
    try:
        mod._send_frame(
            client,
            mod._InitializeFrame(provider_config=_CONFIG, budget_account=account),
        )
        assert mod._receive_frame(client) == {"kind": "ready"}
        mod._send_frame(client, mod._CompletionRequestFrame(request=_REQUEST))
        response = mod._CompletionFrame.model_validate(mod._receive_frame(client)).response
        assert response.choices[0].message.content == "done"
    finally:
        client.close()
        worker_thread.join(timeout=5)

    assert worker_thread.is_alive() is False
    assert exit_codes == [0]
    reservations = SpendLedger(account.ledger_path, policy).reservations()
    assert len(reservations) == 1
    assert reservations[0].status is ReservationStatus.SETTLED
    assert reservations[0].input_tokens == 3
    assert reservations[0].output_tokens == 2


def test_socketpair_start_failure_is_sanitized_and_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "socket-secret-sentinel"

    def fail_socketpair() -> tuple[socket.socket, socket.socket]:
        raise OSError(secret)

    monkeypatch.setattr(mod.socket, "socketpair", fail_socketpair)
    worker = mod.ProviderProcessWorker(_CONFIG)

    with pytest.raises(mod.ProviderWorkerUnavailable) as caught:
        worker.start(TurnDeadline.after(2))

    assert secret not in str(caught.value)
    assert worker.wait_closed(0)


def test_cleanup_failure_signals_completion_without_proving_reap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = mod.ProviderProcessWorker(_CONFIG)
    process = cast("subprocess.Popen[bytes]", object())
    worker._process = process

    def fail_cleanup(_process: subprocess.Popen[bytes], *, force: bool) -> None:
        _ = force
        assert _process is process
        raise mod.ProviderWorkerCleanupError("provider worker cleanup was not proved")

    monkeypatch.setattr(mod, "_stop_and_reap", fail_cleanup)

    with pytest.raises(mod.ProviderWorkerCleanupError):
        worker.cancel()

    assert worker._closed.is_set()
    assert worker.wait_closed(0) is False


def test_close_kills_worker_process_group_after_leader_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grandchild_pid_path = tmp_path / "grandchild.pid"
    script = """
import json
import socket
import struct
import subprocess
import sys

connection = socket.socket(fileno=int(sys.argv[1]))
grandchild = subprocess.Popen(
    [
        sys.executable,
        "-c",
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
    ]
)
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    handle.write(str(grandchild.pid))

header = connection.recv(4)
size = struct.unpack("!I", header)[0]
body = b""
while len(body) < size:
    body += connection.recv(size - len(body))
ready = json.dumps({"kind": "ready"}).encode()
connection.sendall(struct.pack("!I", len(ready)) + ready)
while connection.recv(1):
    pass
"""

    def command(fd: int) -> list[str]:
        return [sys.executable, "-c", script, str(fd), str(grandchild_pid_path)]

    monkeypatch.setattr(mod, "_worker_command", command)
    worker = mod.ProviderProcessWorker(_CONFIG)
    worker.start(TurnDeadline.after(2))
    process = _active_process(worker)
    grandchild_pid = int(grandchild_pid_path.read_text(encoding="utf-8"))

    worker.close()

    assert worker.wait_closed(0)
    _assert_reaped(process)
    with pytest.raises(ProcessLookupError):
        os.kill(grandchild_pid, 0)
    with pytest.raises(ProcessLookupError):
        os.killpg(process.pid, 0)


def test_worker_round_trips_a_structured_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _scripted_worker_command(
        monkeypatch,
        {"kind": "completion", "response": _COMPLETION},
    )
    worker = mod.ProviderProcessWorker(_CONFIG)

    worker.start(TurnDeadline.after(2))
    process = _active_process(worker)
    response = worker.complete_chat(_REQUEST, TurnDeadline.after(2))

    assert response.choices[0].message.content == "done"
    assert response.token_usage().input_tokens == 3
    assert response.token_usage().output_tokens == 2
    assert response.provider_receipt is not None
    assert response.provider_receipt.provider_request_id == "request-1"
    assert response.provider_receipt.response_id is None
    worker.close()
    _assert_reaped(process)


def test_worker_preserves_only_typed_candidate_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _scripted_worker_command(
        monkeypatch,
        {
            "kind": "failure",
            "owner": "candidate",
            "reason": "invalid_request",
        },
    )
    worker = mod.ProviderProcessWorker(_CONFIG)
    worker.start(TurnDeadline.after(2))
    process = _active_process(worker)

    with pytest.raises(mod.ProviderWorkerFailure) as caught:
        worker.complete_chat(_REQUEST, TurnDeadline.after(2))

    assert caught.value.attribution.owner is ProviderFailureOwner.CANDIDATE
    assert caught.value.attribution.reason is ProviderFailureReason.INVALID_REQUEST
    worker.close()
    _assert_reaped(process)


def test_oversized_request_is_candidate_owned_but_oversized_response_is_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _scripted_worker_command(
        monkeypatch,
        {"kind": "completion", "response": _COMPLETION},
    )
    request_worker = mod.ProviderProcessWorker(_CONFIG)
    request_worker.start(TurnDeadline.after(2))
    request_process = _active_process(request_worker)
    monkeypatch.setattr(mod, "_MAX_FRAME_BYTES", 64)

    with pytest.raises(mod.ProviderWorkerFailure) as request_failure:
        request_worker.complete_chat(_REQUEST, TurnDeadline.after(2))

    assert request_failure.value.attribution.owner is ProviderFailureOwner.CANDIDATE
    request_worker.close()
    _assert_reaped(request_process)

    monkeypatch.setattr(mod, "_MAX_FRAME_BYTES", 8 * 1024 * 1024)
    _scripted_worker_command(
        monkeypatch,
        {
            "kind": "completion",
            "response": {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "x" * 1024},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
        },
    )
    response_worker = mod.ProviderProcessWorker(_CONFIG)
    response_worker.start(TurnDeadline.after(2))
    response_process = _active_process(response_worker)
    monkeypatch.setattr(mod, "_MAX_FRAME_BYTES", 512)

    with pytest.raises(mod.ProviderWorkerUnavailable):
        response_worker.complete_chat(_REQUEST, TurnDeadline.after(2))

    assert response_worker.wait_closed(0)
    _assert_reaped(response_process)


def test_request_deadline_kills_and_reaps_provider_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _scripted_worker_command(
        monkeypatch,
        {"kind": "completion", "response": _COMPLETION},
        response_delay_s=60,
    )
    worker = mod.ProviderProcessWorker(_CONFIG)
    worker.start(TurnDeadline.after(2))
    process = _active_process(worker)

    with pytest.raises(mod.ProviderWorkerDeadlineExceeded) as caught:
        worker.complete_chat(_REQUEST, TurnDeadline.after(0.05))

    assert caught.value.source is mod.RequestDeadlineSource.CALLER_BUDGET
    assert worker.wait_closed(0)
    _assert_reaped(process)


def test_startup_deadline_kills_and_reaps_provider_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _scripted_worker_command(
        monkeypatch,
        {"kind": "completion", "response": _COMPLETION},
        ready_delay_s=60,
    )
    worker = mod.ProviderProcessWorker(_CONFIG)
    stopped_processes: list[subprocess.Popen[bytes]] = []
    real_stop_and_reap = mod._stop_and_reap

    def record_stop_and_reap(process: subprocess.Popen[bytes], *, force: bool) -> None:
        stopped_processes.append(process)
        real_stop_and_reap(process, force=force)

    monkeypatch.setattr(mod, "_stop_and_reap", record_stop_and_reap)

    with pytest.raises(mod.ProviderWorkerDeadlineExceeded) as caught:
        worker.start(TurnDeadline.after(0.05))

    assert caught.value.source is mod.RequestDeadlineSource.CALLER_BUDGET
    assert worker.wait_closed(0)
    assert len(stopped_processes) == 1
    _assert_reaped(stopped_processes[0])


def test_external_cancel_unblocks_calling_thread_and_reaps_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _scripted_worker_command(
        monkeypatch,
        {"kind": "completion", "response": _COMPLETION},
        response_delay_s=60,
    )
    worker = mod.ProviderProcessWorker(_CONFIG)
    worker.start(TurnDeadline.after(2))
    process = _active_process(worker)
    finished = threading.Event()
    failures: list[Exception] = []

    def call() -> None:
        try:
            worker.complete_chat(_REQUEST, TurnDeadline.after(30))
        except Exception as exc:  # noqa: BLE001 - the stable worker error is asserted below
            failures.append(exc)
        finally:
            finished.set()

    caller = threading.Thread(target=call)
    caller.start()
    time.sleep(0.05)

    worker.cancel()
    caller.join(timeout=2)

    assert finished.is_set()
    assert caller.is_alive() is False
    assert len(failures) == 1
    assert isinstance(failures[0], mod.ProviderWorkerUnavailable)
    assert worker.wait_closed(0)
    _assert_reaped(process)


def test_worker_protocol_rejects_raw_failure_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "provider-secret-sentinel"
    _scripted_worker_command(
        monkeypatch,
        {
            "kind": "failure",
            "owner": "infrastructure",
            "reason": "unknown",
            "detail": secret,
        },
    )
    worker = mod.ProviderProcessWorker(_CONFIG)
    worker.start(TurnDeadline.after(2))
    process = _active_process(worker)

    with pytest.raises(mod.ProviderWorkerUnavailable) as caught:
        worker.complete_chat(_REQUEST, TurnDeadline.after(2))

    assert secret not in str(caught.value)
    assert worker.wait_closed(0)
    _assert_reaped(process)
