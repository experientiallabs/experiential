"""Exercise real online SDK resume and pending-row replay against loopback transport."""

import gzip
import json
import os
import re
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread


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
