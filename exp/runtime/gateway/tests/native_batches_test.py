"""Wire-level batch route certification against the real native data plane.

One real native server binds a loopback socket over real gateway components;
the batch lane runs over in-memory seams with a scripted provider client, so
every assertion crosses actual HTTP bytes through the Rust routes, the
bridge, the control plane, and the engine, with no provider network access.
"""

from __future__ import annotations

import asyncio
import base64
import threading
from pathlib import Path
from typing import TYPE_CHECKING, cast

import httpx
import pytest

from exp.runtime.gateway.batch import BatchControlPlane, BatchEngine, BatchStatus
from exp.runtime.gateway.batch.contracts import BatchLineResult
from exp.runtime.gateway.batch.engine_test import (
    MemoryCatalog,
    MemoryFiles,
    MemoryLedger,
    MemorySecrets,
    MemoryStore,
    ScriptedClient,
    _chat_line,
)
from exp.runtime.gateway.batch.providers import ProviderBatchSnapshot
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.native_bridge import NativeControlPlane
from exp.runtime.gateway.native_server import serve_native_gateway
from exp.runtime.gateway.tests.launch_test import _configure_gateway, _unused_port, _wait_ready

exp_gateway_native = pytest.importorskip("exp_gateway_native")

if TYPE_CHECKING:
    from exp_gateway_native import ShutdownHandle


class _BatchedComponents:
    """Loaded gateway components extended with the optional batch plane."""

    def __init__(self, inner: object, batches: BatchControlPlane) -> None:
        """Wrap the loaded components and attach the batch plane."""
        self._inner = inner
        self.batches = batches

    def __getattr__(self, name: str) -> object:
        """Delegate every other component to the loaded composition."""
        return getattr(self._inner, name)


def test_batch_routes_serve_the_full_job_lifecycle_over_real_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upload, submit, poll, settle, download, refusal: all across the wire."""
    monkeypatch.setenv("LOOPBACK_PROVIDER_KEY", "provider-secret")
    _manager, raw_key = _configure_gateway(tmp_path, base_url="http://127.0.0.1:9/v1")
    provider = ScriptedClient(
        [
            ProviderBatchSnapshot(status=BatchStatus.IN_PROGRESS),
            ProviderBatchSnapshot(status=BatchStatus.COMPLETED, results_ready=True),
        ],
        [
            BatchLineResult(
                custom_id="wire-0",
                status_code=200,
                response={"usage": {"prompt_tokens": 2, "completion_tokens": 3}},
                input_tokens=2,
                output_tokens=3,
            )
        ],
    )
    store = MemoryStore()
    ledger = MemoryLedger()
    engine = BatchEngine(
        store=store,
        files=MemoryFiles(),
        catalog=MemoryCatalog(),
        ledger=ledger,
        secrets_resolver=MemorySecrets(),
        clients={"openrouter": provider, "openai": provider},
    )
    port = _unused_port()
    inner = load_gateway_components(tmp_path)
    batches = BatchControlPlane(engine=engine, control=inner.store)
    thread, shutdown, _plane, _ = _serve_with(tmp_path, port, batches, inner)
    try:
        base = f"http://127.0.0.1:{port}"
        auth = {"Authorization": f"Bearer {raw_key}"}
        with httpx.Client(timeout=10.0) as client:
            content = _chat_line("wire-0").encode("utf-8")
            uploaded = client.post(
                f"{base}/v1/files",
                headers=auth,
                data={"purpose": "batch"},
                files={"file": ("input.jsonl", content, "application/jsonl")},
            )
            assert uploaded.status_code == 200, uploaded.text
            file_id = uploaded.json()["id"]

            created = client.post(
                f"{base}/v1/batches",
                headers=auth,
                json={"input_file_id": file_id, "endpoint": "/v1/chat/completions"},
            )
            assert created.status_code == 200, created.text
            batch = created.json()
            assert batch["object"] == "batch"
            assert batch["status"] == "validating"
            batch_id = batch["id"]

            listing = client.get(f"{base}/v1/batches", headers=auth)
            assert listing.status_code == 200
            assert [item["id"] for item in listing.json()["data"]] == [batch_id]

            asyncio.run(engine.poll_once())
            asyncio.run(engine.poll_once())
            asyncio.run(engine.poll_once())

            finished = client.get(f"{base}/v1/batches/{batch_id}", headers=auth)
            assert finished.status_code == 200
            final = finished.json()
            assert final["status"] == "completed"
            assert final["request_counts"] == {"total": 1, "completed": 1, "failed": 0}
            output_file_id = final["output_file_id"]
            assert output_file_id

            downloaded = client.get(f"{base}/v1/files/{output_file_id}/content", headers=auth)
            assert downloaded.status_code == 200
            assert downloaded.headers["content-type"].startswith("application/jsonl")
            assert b'"custom_id": "wire-0"' in downloaded.content
            assert ledger.settled == [("wire-0", 3)]

            unknown = client.get(f"{base}/v1/batches/batch_missing", headers=auth)
            assert unknown.status_code == 404

            unauthorized = client.get(
                f"{base}/v1/batches", headers={"Authorization": "Bearer xpl_wrong"}
            )
            assert unauthorized.status_code == 404
    finally:
        shutdown.request_shutdown()
        thread.join(timeout=10)
        inner.write_ledger.close()
    assert not thread.is_alive()


def _serve_with(
    root: Path, port: int, batches: BatchControlPlane, inner: object
) -> tuple[threading.Thread, ShutdownHandle, NativeControlPlane, object]:
    """Serve one native gateway over already-loaded components."""
    components = _BatchedComponents(inner, batches)
    plane = NativeControlPlane(
        components,  # type: ignore[arg-type]
        data_plane_metrics=exp_gateway_native.metrics_snapshot_json,
    )
    shutdown = cast("ShutdownHandle", exp_gateway_native.shutdown_handle())
    failures: list[BaseException] = []

    def run() -> None:
        """Run the blocking server, retaining any failure."""
        try:
            serve_native_gateway(plane, host="127.0.0.1", port=port, shutdown=shutdown)
        except BaseException as error:  # noqa: BLE001 - surfaced by the caller.
            failures.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    _wait_ready(port, thread)
    return thread, shutdown, plane, inner


def test_sync_route_refuses_a_batch_model_with_the_batches_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A batch-only model on /v1/chat/completions 404s pointing at /v1/batches."""
    monkeypatch.setenv("LOOPBACK_PROVIDER_KEY", "provider-secret")
    _manager, raw_key = _configure_gateway(tmp_path, base_url="http://127.0.0.1:9/v1")
    engine = BatchEngine(
        store=MemoryStore(),
        files=MemoryFiles(),
        catalog=MemoryCatalog(),
        ledger=MemoryLedger(),
        secrets_resolver=MemorySecrets(),
    )
    port = _unused_port()
    inner = load_gateway_components(tmp_path)
    batches = BatchControlPlane(engine=engine, control=inner.store)
    thread, shutdown, _plane, _ = _serve_with(tmp_path, port, batches, inner)
    try:
        with httpx.Client(timeout=10.0) as client:
            refused = client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                headers={"Authorization": f"Bearer {raw_key}"},
                json={
                    "model": "gpt-oss-120b-batch",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert refused.status_code == 404, refused.text
            message = refused.json()["error"]["message"]
            assert "/v1/batches" in message
            assert "gpt-oss-120b-batch" in message
    finally:
        shutdown.request_shutdown()
        thread.join(timeout=10)
        inner.write_ledger.close()
    assert not thread.is_alive()


def test_batch_routes_answer_not_enabled_without_the_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A gateway composed without a batch plane refuses batch routes uniformly."""
    monkeypatch.setenv("LOOPBACK_PROVIDER_KEY", "provider-secret")
    _manager, raw_key = _configure_gateway(tmp_path, base_url="http://127.0.0.1:9/v1")
    port = _unused_port()
    inner = load_gateway_components(tmp_path)
    plane = NativeControlPlane(inner, data_plane_metrics=exp_gateway_native.metrics_snapshot_json)
    shutdown = cast("ShutdownHandle", exp_gateway_native.shutdown_handle())

    def run() -> None:
        """Run the blocking server for the duration of the test."""
        serve_native_gateway(plane, host="127.0.0.1", port=port, shutdown=shutdown)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    _wait_ready(port, thread)
    try:
        with httpx.Client(timeout=10.0) as client:
            refused = client.get(
                f"http://127.0.0.1:{port}/v1/batches",
                headers={"Authorization": f"Bearer {raw_key}"},
            )
            assert refused.status_code == 404
            assert "not enabled" in refused.json()["error"]["message"]
    finally:
        shutdown.request_shutdown()
        thread.join(timeout=10)
        inner.write_ledger.close()
    assert not thread.is_alive()


def test_file_content_roundtrips_exact_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Uploaded JSONL bytes download identically through the content route."""
    monkeypatch.setenv("LOOPBACK_PROVIDER_KEY", "provider-secret")
    _manager, raw_key = _configure_gateway(tmp_path, base_url="http://127.0.0.1:9/v1")
    engine = BatchEngine(
        store=MemoryStore(),
        files=MemoryFiles(),
        catalog=MemoryCatalog(),
        ledger=MemoryLedger(),
        secrets_resolver=MemorySecrets(),
    )
    port = _unused_port()
    inner = load_gateway_components(tmp_path)
    batches = BatchControlPlane(engine=engine, control=inner.store)
    thread, shutdown, _plane, _ = _serve_with(tmp_path, port, batches, inner)
    try:
        base = f"http://127.0.0.1:{port}"
        auth = {"Authorization": f"Bearer {raw_key}"}
        content = (_chat_line("r-0") + "\n" + _chat_line("r-1")).encode("utf-8")
        with httpx.Client(timeout=10.0) as client:
            uploaded = client.post(
                f"{base}/v1/files",
                headers=auth,
                data={"purpose": "batch"},
                files={"file": ("input.jsonl", content, "application/jsonl")},
            )
            assert uploaded.status_code == 200
            file_id = uploaded.json()["id"]
            fetched = client.get(f"{base}/v1/files/{file_id}", headers=auth)
            assert fetched.status_code == 200
            assert fetched.json()["bytes"] == len(content)
            downloaded = client.get(f"{base}/v1/files/{file_id}/content", headers=auth)
            assert downloaded.content == content
            assert base64.b64encode(downloaded.content) == base64.b64encode(content)
    finally:
        shutdown.request_shutdown()
        thread.join(timeout=10)
        inner.write_ledger.close()
    assert not thread.is_alive()


def test_invalid_keys_never_learn_that_a_model_is_batch_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unauthenticated caller gets the generic 401, not the batch pointer."""
    monkeypatch.setenv("LOOPBACK_PROVIDER_KEY", "provider-secret")
    _configure_gateway(tmp_path, base_url="http://127.0.0.1:9/v1")
    engine = BatchEngine(
        store=MemoryStore(),
        files=MemoryFiles(),
        catalog=MemoryCatalog(),
        ledger=MemoryLedger(),
        secrets_resolver=MemorySecrets(),
    )
    port = _unused_port()
    inner = load_gateway_components(tmp_path)
    batches = BatchControlPlane(engine=engine, control=inner.store)
    thread, shutdown, _plane, _ = _serve_with(tmp_path, port, batches, inner)
    try:
        with httpx.Client(timeout=10.0) as client:
            refused = client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                headers={"Authorization": "Bearer xpl_invalid"},
                json={
                    "model": "gpt-oss-120b-batch",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert refused.status_code == 401, refused.text
            assert "/v1/batches" not in refused.text

            upload = client.post(
                f"http://127.0.0.1:{port}/v1/files",
                headers={"Authorization": "Bearer xpl_invalid"},
                data={"purpose": "batch"},
                files={"file": ("input.jsonl", b"{}", "application/jsonl")},
            )
            assert upload.status_code == 401
    finally:
        shutdown.request_shutdown()
        thread.join(timeout=10)
        inner.write_ledger.close()
    assert not thread.is_alive()
