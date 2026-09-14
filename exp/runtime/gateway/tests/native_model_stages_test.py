"""Real Rust loopback execution preserves cross-model identity and request accounting."""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from exp.common.models import load_model_catalog, write_model_catalog
from exp.common.models.gateway_chains import (
    GatewayDeploymentRung,
    GatewayModelChain,
    GatewayModelReferenceRung,
)
from exp.runtime.gateway.catalog_authority import snapshot_current_catalog
from exp.runtime.gateway.native_bridge_test import _configured_pool_gateway
from exp.runtime.gateway.tests.native_waterfall_test import (
    _DRIVER_SOURCE,
    _attempt_rows,
    _PrimaryUpstream,
    _SecondaryUpstream,
    _ServingEngine,
)


@pytest.fixture(name="engine")
def stage_engine(tmp_path: Path) -> Iterator[_ServingEngine]:
    """Serve a genuinely different fallback model, with no false pool equivalence."""
    primary = ThreadingHTTPServer(("127.0.0.1", 0), _PrimaryUpstream)
    secondary = ThreadingHTTPServer(("127.0.0.1", 0), _SecondaryUpstream)
    for server in (primary, secondary):
        threading.Thread(target=server.serve_forever, daemon=True).start()
    manager, raw_key = _configured_pool_gateway(
        tmp_path,
        base_urls=(
            f"http://127.0.0.1:{primary.server_port}/v1",
            f"http://127.0.0.1:{secondary.server_port}/v1",
        ),
    )
    catalog = load_model_catalog(tmp_path / "models.toml")
    models = dict(catalog.models)
    beta = models["beta"]
    assert beta.gateway is not None
    models["beta"] = beta.model_copy(
        update={"gateway": beta.gateway.model_copy(update={"exact_model_id": "secondary-exact"})}
    )
    chain = GatewayModelChain(
        model_id="model-revision-exact",
        pool_id="alpha",
        revision="chain-one",
        rungs=(
            GatewayDeploymentRung(deployment_id="alpha"),
            GatewayModelReferenceRung(model_id="secondary-exact"),
        ),
    )
    authored = catalog.model_copy(
        update={
            "models": models,
            "gateway_pools": {},
            "gateway_model_chains": {"model-revision-exact": chain},
        }
    )
    write_model_catalog(tmp_path / "models.toml", authored)
    _, normalized, snapshot = snapshot_current_catalog(tmp_path)
    manager.activate_direct_alias(
        alias_id="coding",
        alias_name="coding",
        revision_id="revision-stage",
        pool_id="alpha",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    driver = tmp_path / "driver.py"
    driver.write_text(_DRIVER_SOURCE + "\n")
    log_path = tmp_path / "driver.log"
    environment = {**os.environ, "TEST_PROVIDER_KEY": "loopback-secret"}
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                str(driver),
                json.dumps({"root": str(tmp_path), "request_timeout_seconds": 10}),
            ],
            stdout=subprocess.PIPE,
            stderr=log,
            env=environment,
            text=True,
        )  # noqa: S603 - generated test driver.
        ports: list[int] = []

        def collect() -> None:
            """Read port announcements without blocking the readiness deadline."""
            assert process.stdout is not None
            for line in process.stdout:
                ports.append(int(json.loads(line)["port"]))

        threading.Thread(target=collect, daemon=True).start()
        try:
            deadline = time.monotonic() + 30
            while True:
                assert process.poll() is None, log_path.read_text()
                assert time.monotonic() < deadline, log_path.read_text()
                if ports:
                    try:
                        response = httpx.get(
                            f"http://127.0.0.1:{ports[-1]}/health/live", timeout=0.5
                        )
                        if response.status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                time.sleep(0.05)
            yield _ServingEngine(ports[-1], raw_key, manager.database_path)
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
            process.wait(timeout=20)
            for server in (primary, secondary):
                server.shutdown()
                server.server_close()
            assert process.returncode == 0, log_path.read_text()


@pytest.mark.parametrize("surface", ["chat/completions", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_actual_stage_header_alias_and_single_request_ledger(
    engine: _ServingEngine, surface: str, stream: bool
) -> None:
    """One request redials root once then commits different exact model on every surface."""
    payload = {"model": "coding", "stream": stream}
    if surface == "responses":
        payload["input"] = "always-500"
    else:
        payload["messages"] = [{"role": "user", "content": "always-500"}]
        if surface == "messages":
            payload["max_tokens"] = 32
    response = httpx.post(
        f"{engine.base}/v1/{surface}",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json=payload,
        timeout=30,
    )
    assert response.status_code == 200, response.text
    assert response.headers["x-gateway-canonical-model"] == "secondary-exact"
    assert response.headers["x-gateway-alias"] == "coding"
    assert "from-secondary" in response.text
    if not stream:
        assert response.json()["model"] == "coding"
    request_id = response.headers["x-request-id"]
    assert _attempt_rows(engine, request_id) == [
        (0, 0, "failed"),
        (1, 0, "failed"),
        (2, 1, "completed"),
    ]
    with sqlite3.connect(engine.database_path) as db:
        rows = db.execute(
            "SELECT exact_model_id,pool_id FROM gateway_attempts "
            "WHERE request_id=? ORDER BY attempt_ordinal",
            (request_id,),
        ).fetchall()
        assert rows == [
            ("model-revision-exact", "alpha"),
            ("model-revision-exact", "alpha"),
            ("secondary-exact", "beta"),
        ]
        assert db.execute("SELECT count(*) FROM gateway_requests").fetchone() == (1,)
        assert db.execute(
            "SELECT count(*) FROM gateway_attempts WHERE state IN ('dispatched','running')"
        ).fetchone() == (0,)
