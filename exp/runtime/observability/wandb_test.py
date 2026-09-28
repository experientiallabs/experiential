"""Verify explicit egress, history ordering, bounded cleanup, and real offline SDK recording."""

import gzip
import json
import os
import re
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from typing import cast

import pytest
import wandb

import exp.runtime.observability.wandb as adapter
from exp.common.core.artifacts import SecretBoundaryError
from exp.common.observability.metrics import MetricRecord
from exp.runtime.observability.wandb import WandbMetricSink


class FakeRun:
    """Record SDK calls without making provider requests."""

    def __init__(self) -> None:
        """Start with an existing history position and configurable cleanup behavior."""
        self.step = 0
        self.starting_step = 7
        self.disabled = False
        self.offline = False
        self.url = "https://wandb.ai/example/project/runs/fixture"
        self.records: list[tuple[dict[str, float], int, bool]] = []
        self.finish_calls = 0
        self.close_error: Exception | None = None

    def log(self, values: dict[str, float], *, step: int, commit: bool) -> None:
        """Retain exactly the payload, history position, and commit flag submitted."""
        self.records.append((values, step, commit))

    def finish(self, *, exit_code: int = 0) -> None:
        """Count ownership cleanup and optionally simulate upload failure."""
        self.finish_calls += 1
        if self.close_error:
            raise self.close_error


def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeRun, dict[str, object]]:
    """Replace only SDK boundaries and retain explicit initialization arguments."""
    monkeypatch.setenv("WANDB_ERROR_REPORTING", "false")
    monkeypatch.setattr(adapter, "_SDK_SAFE_AT_IMPORT", True)
    run = FakeRun()
    captured: dict[str, object] = {}

    def initialize(**arguments: object) -> wandb.Run:
        """Return the instrumented run and capture its selected destination settings."""
        captured.update(arguments)
        return cast(wandb.Run, run)

    monkeypatch.setattr(wandb, "init", initialize)
    monkeypatch.setattr(wandb, "setup", lambda: SimpleNamespace(settings=wandb.Settings()))
    monkeypatch.setattr(wandb, "login", lambda *, timeout, force: True)
    return run, captured


def test_explicit_payload_history_and_privacy_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify ordered numeric egress, explicit run ownership, and capture suppression."""
    run, arguments = fake_sdk(monkeypatch)
    with WandbMetricSink(
        project="project", run_id="fixture", directory=tmp_path, config={"lora_rank": 16}
    ) as sink:
        assert sink.next_step == 7 and sink.url == run.url
        sink.record(MetricRecord(event_id="gpu-1", step=7, values={"gpu/utilization": 50}))
        sink.record(MetricRecord(event_id="update-1", values={"train/optimizer_step": 1}))
        sink.record(MetricRecord(event_id="eval-1", step=12, values={"eval/successes": 8}))
        assert sink.next_step == 13
        with pytest.raises(ValueError, match="precedes next_step"):
            sink.record(MetricRecord(event_id="old", step=7, values={"loss": 1}))
    sink.close()
    assert run.finish_calls == 1
    assert run.records == [
        ({"gpu/utilization": 50.0}, 7, True),
        ({"train/optimizer_step": 1.0}, 8, True),
        ({"eval/successes": 8.0}, 12, True),
    ]
    assert arguments["config"] == {"lora_rank": 16}
    assert arguments["reinit"] == "create_new" and arguments["sync_tensorboard"] is False
    settings = cast(wandb.Settings, arguments["settings"])
    assert settings.console == "off" and settings.disable_code and settings.disable_git
    assert settings.x_disable_meta and settings.x_disable_stats and settings.x_disable_machine_info
    assert settings.config_paths == [] and settings.capture_loggers == {}
    assert not settings.x_save_requirements and settings.disable_job_creation
    assert settings.finish_timeout == settings.init_timeout == 30 and settings.finish_timeout_raises
    assert settings.stop_fn is not None
    settings.stop_fn()  # The metrics UI must never send SIGINT to the learning process.
    with pytest.raises(ValueError, match="sink is closed"):
        sink.record(MetricRecord(event_id="late", values={"loss": 1}))


def test_bad_config_rejected_before_sdk_or_directory_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject unsafe configuration before allocating reporter state."""
    _, captured = fake_sdk(monkeypatch)
    directory = tmp_path / "not-created"
    with pytest.raises(SecretBoundaryError):
        WandbMetricSink(
            project="p",
            run_id="r",
            directory=directory,
            config={"credentials": {"api_key": "private"}},
        )
    with pytest.raises(ValueError, match="timeouts"):
        WandbMetricSink(project="p", run_id="r", directory=directory, finish_timeout_seconds=0)
    assert captured == {} and not directory.exists()


def test_global_config_is_rejected_before_run_egress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse inherited configuration without mutating the existing SDK session."""
    _, captured = fake_sdk(monkeypatch)
    inherited = wandb.Settings(config_paths=["unrelated-private-config.yaml"])
    monkeypatch.setattr(wandb, "setup", lambda: SimpleNamespace(settings=inherited))
    with pytest.raises(ValueError, match="separate metrics reporter process"):
        WandbMetricSink(project="p", run_id="r", directory=tmp_path / "not-created")
    assert not captured and inherited.config_paths == ["unrelated-private-config.yaml"]


def test_online_auth_failure_does_not_create_a_disabled_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Require actual online admission instead of accepting disabled fallback."""
    run, captured = fake_sdk(monkeypatch)
    monkeypatch.setattr(wandb, "login", lambda *, timeout, force: False)
    with pytest.raises(ValueError, match="authentication failed"):
        WandbMetricSink(project="p", run_id="r", directory=tmp_path)
    assert not captured
    monkeypatch.setattr(wandb, "login", lambda *, timeout, force: True)
    run.disabled = True
    with pytest.raises(ValueError, match="requested reporting mode"):
        WandbMetricSink(project="p", run_id="r", directory=tmp_path)
    assert run.finish_calls == 1


def test_mutation_is_revalidated_and_cleanup_cannot_replace_caller_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Revalidate mutable values and preserve caller failures through SDK cleanup."""
    run, _ = fake_sdk(monkeypatch)
    sink = WandbMetricSink(project="p", run_id="r", directory=tmp_path)
    record = MetricRecord(event_id="r", values={"loss": 1})
    record.values["loss"] = float("nan")
    with pytest.raises(ValueError):
        sink.record(record)
    assert not run.records and sink.next_step == 7
    run.close_error = TimeoutError("upload deadline")
    with pytest.raises(RuntimeError, match="original"):
        with sink:
            raise RuntimeError("original")
    assert "cleanup failed (TimeoutError)" in caplog.text


def test_offline_sdk_writes_real_history_without_provider_access(tmp_path: Path) -> None:
    """Use the real offline SDK to prove separate spools and existing-run isolation."""
    script = """
import pathlib, sys
from exp.common.observability.metrics import MetricRecord
from exp.runtime.observability.wandb import WandbMetricSink
directory=pathlib.Path(sys.argv[1])
with WandbMetricSink(
    project='existing-caller',run_id='existing',directory=directory,mode='offline'
) as existing:
  with WandbMetricSink(
    project='exp-offline-test',run_id='offline-fixture',directory=directory,
    mode='offline',config={'rank':16}
  ) as sink:
    first=sink.next_step
    sink.record(MetricRecord(event_id='eval-1',values={'eval/success':1},step=first))
    sink.record(MetricRecord(event_id='update-1',values={'train/optimizer_step':1,'train/loss':-0.5}))
    assert sink.next_step==first+2
    assert sink.url is None
  existing.record(MetricRecord(event_id='still-open',values={'caller/success':1}))
assert len(list(directory.glob('wandb/offline-run-*/run-*.wandb')))==2
target=next(directory.glob('wandb/offline-run-*-offline-fixture/run-*.wandb')).read_bytes()
existing=next(directory.glob('wandb/offline-run-*-existing/run-*.wandb')).read_bytes()
assert b'eval/success' in target and b'train/loss' in target
assert b'caller/success' not in target
assert b'caller/success' in existing
assert b'train/loss' not in existing

"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "WANDB_ERROR_REPORTING": "false"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    files = list(tmp_path.glob("wandb/offline-run-*/files/*"))
    assert all(path.name not in {"output.log", "wandb-metadata.json"} for path in files)
    assert not [
        file
        for directory in tmp_path.rglob("code")
        for file in directory.rglob("*")
        if file.is_file()
    ]


@pytest.mark.parametrize("unsafe", ["unset", "preloaded", "changed"])
def test_uncertain_sdk_reporting_state_fails_before_run_creation(
    tmp_path: Path, unsafe: str
) -> None:
    """Reject late opt-outs, preloaded SDK state, and changes after safe adapter import."""
    script = """
import os, pathlib, sys
mode = sys.argv[2]
if mode == 'unset':
    os.environ.pop('WANDB_ERROR_REPORTING', None)
if mode == 'preloaded':
    import wandb
from exp.runtime.observability.wandb import WandbMetricSink
os.environ['WANDB_ERROR_REPORTING'] = 'true' if mode == 'changed' else 'false'
directory = pathlib.Path(sys.argv[1]) / 'not-created'
try:
    WandbMetricSink(project='p', run_id='r', directory=directory, mode='offline')
except ValueError as error:
    assert 'fresh metrics reporter' in str(error)
else:
    raise AssertionError('unsafe SDK admission unexpectedly succeeded')
assert not directory.exists()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), unsafe],
        env={**os.environ, "WANDB_ERROR_REPORTING": "false"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


class Backend:
    """Retain acknowledged history and reject unrecognized SDK transport operations."""

    def __init__(self) -> None:
        """Seed six remotely acknowledged history positions before the first resume."""
        self.history = [{"_step": step, "existing": step} for step in range(6)]
        self.operations: list[str] = []
        self.errors: list[str] = []
        self.resumed_steps: list[int] = []
        self.uploaded_steps: list[int] = []
        self.url = ""

    def graphql(self, body: dict) -> dict:
        """Implement only the public SDK operations needed for one scalar-only run."""
        query = body["query"]
        match = re.search(r"(?:query|mutation)\s+(\w+)", query)
        operation = body.get("operationName") or (match.group(1) if match else "anonymous")
        self.operations.append(operation)
        viewer = {
            "id": "viewer-id",
            "entity": "fixture-entity",
            "username": "fixture-user",
            "name": "fixture-user",
            "email": "fixture@example.invalid",
            "admin": False,
            "flags": "{}",
            "deletedAt": None,
            "apiKeys": {"edges": []},
            "teams": {"edges": []},
        }
        if operation in {"Viewer", "GetViewer", "GetCurrentUser", "GetDefaultEntity"}:
            return {"viewer": viewer}
        if operation == "ServerInfo":
            return {"serverInfo": {"cliVersionInfo": {}, "latestLocalVersionInfo": None}}
        if operation == "ServerFeaturesQuery":
            return {"serverInfo": {"features": []}}
        if operation == "RunResumeStatus":
            self.resumed_steps.append(self.history[-1]["_step"] + 1)
            return {
                "model": {
                    "id": "project-id",
                    "name": "fixture-project",
                    "entity": {"id": "entity-id", "name": "fixture-entity"},
                    "bucket": {
                        "id": "run-id",
                        "name": "fixture-run",
                        "displayName": "fixture-run",
                        "historyLineCount": len(self.history),
                        "eventsLineCount": 0,
                        "logLineCount": 0,
                        "historyTail": json.dumps([json.dumps(self.history[-1])]),
                        "eventsTail": "[]",
                        "summaryMetrics": json.dumps(self.history[-1]),
                        "config": "{}",
                        "tags": [],
                        "notes": "",
                        "wandbConfig": '{"t":1}',
                    },
                }
            }
        if operation == "UpsertBucket":
            variables = body["variables"]
            assert variables.get("entity") == "fixture-entity"
            assert variables.get("project") == "fixture-project"
            assert variables.get("name") == "fixture-run"
            selected = json.loads(variables.get("config") or "{}")
            assert set(selected) <= {"lora_rank", "_wandb"}, selected
            if "lora_rank" in selected:
                assert selected["lora_rank"] == {"value": 16}
            return {
                "upsertBucket": {
                    "inserted": False,
                    "bucket": {
                        "id": "run-id",
                        "name": "fixture-run",
                        "displayName": "fixture-run",
                        "config": "{}",
                        "historyLineCount": len(self.history),
                        "project": {
                            "id": "project-id",
                            "name": "fixture-project",
                            "entity": {"id": "entity-id", "name": "fixture-entity"},
                        },
                    },
                }
            }
        if operation == "CreateRunFiles":
            return {
                "createRunFiles": {
                    "runID": "run-id",
                    "uploadHeaders": [],
                    "files": [
                        {"name": name, "uploadUrl": f"{self.url}/upload/{name}"}
                        for name in body["variables"]["files"]
                    ],
                }
            }
        if operation == "RunStopStatus":
            return {"project": {"run": {"shouldStop": False}}}
        raise AssertionError(f"Unexpected GraphQL operation: {operation}: {query}")

    def stream(self, body: dict) -> dict:
        """Acknowledge exact history offsets only after retaining their actual SDK payload."""
        for name, file in body.get("files", {}).items():
            assert name in {"wandb-history.jsonl", "wandb-summary.json"}, name
            if name == "wandb-history.jsonl":
                assert file["offset"] == len(self.history)
                for line in file["content"]:
                    row = json.loads(line)
                    assert set(row) <= {"train/loss", "_step", "_runtime", "_timestamp"}, row
                    assert row["_step"] == len(self.history)
                    self.history.append(row)
                    self.uploaded_steps.append(row["_step"])
        return {"exitcode": 0}


def handler_for(backend: Backend) -> type[BaseHTTPRequestHandler]:
    """Bind a strict loopback HTTP handler without replacing any SDK model or method."""

    class Handler(BaseHTTPRequestHandler):
        """Serve only GraphQL, scalar history, and explicit local file-upload destinations."""

        def log_message(self, format: str, *args: object) -> None:
            """Avoid recording authentication headers or noisy fixture access logs."""

        def do_POST(self) -> None:
            """Decode a bounded SDK request and reject every unrecognized endpoint."""
            try:
                length = int(self.headers["Content-Length"])
                assert length <= 1_000_000
                raw = self.rfile.read(length)
                if self.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                body = json.loads(raw)
                if self.path == "/graphql":
                    result = {"data": backend.graphql(body)}
                elif self.path == "/files/fixture-entity/fixture-project/fixture-run/file_stream":
                    result = backend.stream(body)
                else:
                    raise AssertionError(f"Unexpected POST path: {self.path}")
                encoded = json.dumps(result).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
            except Exception as error:  # noqa: BLE001 - expose protocol failures in the test
                backend.errors.append(
                    f"{self.path} encoding={self.headers.get('Content-Encoding')} "
                    f"type={self.headers.get('Content-Type')}: {error}"
                )
                self.send_error(400)

        def do_PUT(self) -> None:
            """Accept only explicit configuration and summary files on loopback."""
            try:
                assert self.path in {
                    "/upload/config.yaml",
                    "/upload/wandb-summary.json",
                }, self.path
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()
            except Exception as error:  # noqa: BLE001 - retain unexpected upload requests
                backend.errors.append(str(error))
                self.send_error(400)

    return Handler


def run_reporter(
    directory: Path, url: str, expected_step: int, last_step: int
) -> subprocess.CompletedProcess:
    """Replay a durable scalar outbox through a real SDK process with no inherited credentials."""
    directory.mkdir()
    script = """
import json, pathlib, socket, sys
original_connect = socket.socket.connect
def local_connect(self, address):
    '''Reject Python network access outside the fixture before importing the SDK.'''
    if isinstance(address, tuple) and address[0] not in ('127.0.0.1', '::1'):
        raise AssertionError('Non-loopback network destination rejected')
    return original_connect(self, address)
socket.socket.connect = local_connect
from exp.common.observability.metrics import MetricRecord
from exp.runtime.observability.wandb import WandbMetricSink
directory = pathlib.Path(sys.argv[1])
rows = [{'event_id':f'event-{step}', 'step':step, 'values':{'train/loss':float(step)}}
        for step in range(4, int(sys.argv[3]) + 1)]
outbox = directory / 'outbox.jsonl'
outbox.write_text(''.join(json.dumps(row) + '\\n' for row in rows))
with WandbMetricSink(project='fixture-project', entity='fixture-entity',
                     run_id='fixture-run', directory=directory, mode='online',
                     config={'lora_rank':16},
                     initialization_timeout_seconds=10, finish_timeout_seconds=10) as sink:
    assert sink.next_step == int(sys.argv[2]), sink.next_step
    for line in outbox.read_text().splitlines():
        row = MetricRecord.model_validate_json(line)
        if row.step >= sink.next_step:
            sink.record(row)
    assert sink.next_step == int(sys.argv[3]) + 1
"""
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(directory),
        "WANDB_BASE_URL": url,
        "WANDB_API_KEY": "a" * 40,
        "WANDB_CONFIG_DIR": str(directory / "config"),
        "WANDB_CACHE_DIR": str(directory / "cache"),
        "WANDB_DATA_DIR": str(directory / "data"),
        "WANDB_SILENT": "true",
        "WANDB_ERROR_REPORTING": "false",
    }
    return subprocess.run(
        [sys.executable, "-c", script, str(directory), str(expected_step), str(last_step)],
        env=env,
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_real_online_sdk_resumes_remote_history_and_replays_only_pending_rows(
    tmp_path: Path,
) -> None:
    """Two real SDK lifetimes resume server-confirmed cursors without reuploading old rows."""
    backend = Backend()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(backend))
    backend.url = f"http://127.0.0.1:{server.server_port}"
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        first = run_reporter(tmp_path / "first", backend.url, 6, 7)
        assert not backend.errors, (backend.errors, first.stderr, backend.operations)
        assert first.returncode == 0, (first.stderr, backend.operations)
        assert backend.uploaded_steps == [6, 7]
        second = run_reporter(tmp_path / "second", backend.url, 8, 8)
        assert not backend.errors, backend.errors
        assert second.returncode == 0, second.stderr
        assert backend.resumed_steps == [6, 8]
        assert backend.uploaded_steps == [6, 7, 8]
        assert [row["train/loss"] for row in backend.history[6:]] == [6, 7, 8]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()
