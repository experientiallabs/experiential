"""Gated live certification: a real Doubleword batch through the native routes.

One real native server binds a loopback socket over real gateway components;
the batch lane runs over in-memory seams but with the real Doubleword provider
client and a real Doubleword API key, so a tiny two-line batch crosses actual
HTTP bytes through the Rust routes, the bridge, the plane, and the engine, then
out to Doubleword's real Batch API and back. Gated the same way as the
per-provider live tests: it spends real money in the smallest amount and runs
only when explicitly selected.

    EXP_LIVE_BATCH=1 DOUBLEWORD_API_KEY=... \
        uv run pytest exp/runtime/gateway/tests/doubleword_batch_live_test.py -q
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import httpx
import pytest

from exp.runtime.gateway.batch import BatchControlPlane, BatchEngine
from exp.runtime.gateway.batch.contracts import BatchDeployment
from exp.runtime.gateway.batch.engine_test import MemoryFiles, MemoryLedger, MemoryStore
from exp.runtime.gateway.batch.providers import DoublewordBatchClient
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.tests.launch_test import _configure_gateway, _unused_port
from exp.runtime.gateway.tests.native_batches_test import _serve_with

_LIVE = os.environ.get("EXP_LIVE_BATCH") == "1"
_KEY = os.environ.get("DOUBLEWORD_API_KEY")
_MODEL_ALIAS = "kimi-k3-batch"
_PROVIDER_MODEL = "moonshotai/kimi-k3"
_DEADLINE_SECONDS = 8 * 60.0


class _DoublewordCatalog:
    """BatchCatalog exposing one Doubleword batch model."""

    def batch_deployment(self, *, model: str) -> BatchDeployment | None:
        """Resolve the single Doubleword batch alias, else None."""
        if model != _MODEL_ALIAS:
            return None
        return BatchDeployment(
            model=_MODEL_ALIAS,
            provider="doubleword",
            provider_model=_PROVIDER_MODEL,
            credential_reference="secret://doubleword",
            surfaces=("/v1/chat/completions",),
            input_micro_usd_per_million_tokens=100_000,
            output_micro_usd_per_million_tokens=200_000,
        )


class _LiveSecrets:
    """BatchSecretResolver returning the real Doubleword key for any reference."""

    def resolve(self, reference: str) -> str:
        """Return the configured live key."""
        return _KEY or ""


def _line(custom_id: str) -> str:
    """Render one real chat batch line addressed to the Doubleword alias."""
    return json.dumps(
        {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": _MODEL_ALIAS,
                "messages": [{"role": "user", "content": "Say ok."}],
                "max_tokens": 8,
            },
        }
    )


@pytest.mark.skipif(
    not (_LIVE and _KEY),
    reason="set EXP_LIVE_BATCH=1 and DOUBLEWORD_API_KEY to run the live Doubleword batch lane",
)
def test_doubleword_batch_lane_over_real_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upload, submit, poll to completion, and download a real Doubleword batch."""
    monkeypatch.setenv("LOOPBACK_PROVIDER_KEY", "provider-secret")
    _manager, raw_key = _configure_gateway(tmp_path, base_url="http://127.0.0.1:9/v1")
    store = MemoryStore()
    ledger = MemoryLedger()
    engine = BatchEngine(
        store=store,
        files=MemoryFiles(),
        catalog=_DoublewordCatalog(),
        ledger=ledger,
        secrets_resolver=_LiveSecrets(),
        clients={"doubleword": DoublewordBatchClient()},
    )
    port = _unused_port()
    inner = load_gateway_components(tmp_path)
    batches = BatchControlPlane(engine=engine, control=inner.store)
    thread, shutdown, _plane, _ = _serve_with(tmp_path, port, batches, inner)
    try:
        base = f"http://127.0.0.1:{port}"
        auth = {"Authorization": f"Bearer {raw_key}"}
        with httpx.Client(timeout=30.0) as client:
            content = "\n".join([_line("live-0"), _line("live-1")]).encode("utf-8")
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
            batch_id = created.json()["id"]

            deadline = time.monotonic() + _DEADLINE_SECONDS
            while True:
                asyncio.run(engine.poll_once())
                job = store.jobs.get(batch_id)
                if job is not None and job.settled:
                    break
                assert time.monotonic() < deadline, "live Doubleword batch did not settle in time"
                time.sleep(3)

            finished = client.get(f"{base}/v1/batches/{batch_id}", headers=auth)
            assert finished.status_code == 200
            final = finished.json()
            assert final["status"] == "completed", final
            assert final["request_counts"] == {"total": 2, "completed": 2, "failed": 0}
            output_file_id = final["output_file_id"]
            assert output_file_id

            downloaded = client.get(f"{base}/v1/files/{output_file_id}/content", headers=auth)
            assert downloaded.status_code == 200
            assert downloaded.headers["content-type"].startswith("application/jsonl")
            assert b'"custom_id": "live-0"' in downloaded.content
            assert b'"custom_id": "live-1"' in downloaded.content
            assert len(ledger.settled) == 2
    finally:
        shutdown.request_shutdown()
        thread.join(timeout=10)
        inner.write_ledger.close()
    assert not thread.is_alive()
