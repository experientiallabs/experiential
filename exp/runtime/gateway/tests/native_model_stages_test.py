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

import exp_gateway_native
import httpx
import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import load_model_catalog, write_model_catalog
from exp.common.models.catalog import GatewayDeploymentCapabilities
from exp.common.models.gateway_chains import (
    GatewayDeploymentRung,
    GatewayModelChain,
    GatewayModelReferenceRung,
)
from exp.runtime.gateway.native_bridge_test import _configured_pool_gateway
from exp.runtime.gateway.tests.chain_authority_fixture_test import publish_authored_chain_fixture
from exp.runtime.gateway.tests.native_responses_tool_translation_test import (
    _assert_collision_response,
    _collision_request,
    _ToolUpstream,
)
from exp.runtime.gateway.tests.native_waterfall_test import (
    _DRIVER_SOURCE,
    _attempt_rows,
    _content_chunk,
    _PrimaryUpstream,
    _SecondaryUpstream,
    _ServingEngine,
    _sse_frame,
    _terminal_frames,
)


class _StageSecondaryUpstream(_SecondaryUpstream):
    """Capture the real child wire while retaining the ordinary loopback response."""

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract.
        """Require JSON mode on the child wire and record it in the scripted JSON reply."""
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length))
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        if "provider" in payload:
            text = json.dumps(
                {
                    "from-secondary": True,
                    "provider": payload["provider"],
                    "metadata": self.headers.get("X-OpenRouter-Metadata"),
                }
            )
        elif "response_format" in payload:
            text = json.dumps(
                {
                    "from-secondary": True,
                    "response_format": payload["response_format"],
                    "verbosity_sent": "verbosity" in payload,
                }
            )
        else:
            text = "from-secondary"
        try:
            if payload.get("tools"):
                tools = payload["tools"]
                functions = [tool["function"] for tool in tools]
                assert [tool["name"] for tool in functions] == ["apply_patch", "agents__close"]
                for index, name in enumerate(("apply_patch", "agents__close")):
                    self.wfile.write(
                        _sse_frame(
                            {
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "tool_calls": [
                                                {
                                                    "index": index,
                                                    "id": f"call-{index}",
                                                    "type": "function",
                                                    "function": {
                                                        "name": name,
                                                        "arguments": json.dumps(
                                                            {"input": "portable patch"}
                                                            if index == 0
                                                            else {"id": "agent-one"}
                                                        ),
                                                    },
                                                }
                                            ]
                                        },
                                        "finish_reason": None,
                                    }
                                ]
                            }
                        )
                    )
                self.wfile.write(
                    _sse_frame(
                        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
                    )
                )
                self.wfile.write(_terminal_frames(prompt_tokens=7, completion_tokens=3))
                return
            self.wfile.write(
                _sse_frame(
                    {
                        "provider": "fixture-upstream",
                        "choices": [
                            {"index": 0, "delta": {"content": text}, "finish_reason": None}
                        ],
                    }
                )
                if "provider" in payload
                else _content_chunk(text)
            )
            self.wfile.write(_terminal_frames(prompt_tokens=3, completion_tokens=1))
        except OSError:
            return


def test_installed_native_exports_ordered_stage_contract() -> None:
    """Read the compiled extension's feature marker, independently of package version labels."""
    marker = getattr(exp_gateway_native, "MODEL_STAGE_CONTRACT_VERSION", None)
    assert type(marker) is int
    assert marker == 1


@pytest.fixture(name="engine")
def stage_engine(tmp_path: Path, request: pytest.FixtureRequest) -> Iterator[_ServingEngine]:
    """Serve a genuinely different fallback model, with no false pool equivalence."""
    primary = ThreadingHTTPServer(("127.0.0.1", 0), _PrimaryUpstream)
    child_cancel = getattr(request, "param", None) == "tool-child-cancel"
    child_collision = getattr(request, "param", None) == "tool-child-collision"
    first_token_stall = getattr(request, "param", None) == "first-token-stall"
    unavailable_child = getattr(request, "param", None) == "unavailable-child"
    secondary_handler = _StageSecondaryUpstream
    if child_cancel or child_collision:
        secondary_handler = _ToolUpstream
    secondary = ThreadingHTTPServer(("127.0.0.1", 0), secondary_handler)
    for server in (primary, secondary):
        threading.Thread(target=server.serve_forever, daemon=True).start()
    manager, raw_key = _configured_pool_gateway(
        tmp_path,
        base_urls=(
            f"http://127.0.0.1:{primary.server_port}/v1",
            f"http://127.0.0.1:{secondary.server_port}/v1",
        ),
        gateway_capabilities=(
            GatewayDeploymentCapabilities(
                supports_streaming=True, supports_streaming_tool_arguments=True
            ),
            GatewayDeploymentCapabilities(
                supports_streaming=True, supports_streaming_tool_arguments=True
            ),
        ),
    )
    catalog = load_model_catalog(tmp_path / "models.toml")
    models = dict(catalog.models)
    beta = models["beta"]
    assert beta.gateway is not None
    capabilities = beta.gateway.capabilities
    zdr_child = getattr(request, "param", None) == "zdr-child"
    mapping_isolation = getattr(request, "param", None) == "tool-map-isolation"
    if (
        hasattr(request, "param")
        and not zdr_child
        and not unavailable_child
        and not str(request.param).startswith("capture-")
    ):
        rule = (
            "provider_internal"
            if mapping_isolation or child_cancel or child_collision
            else "timeout"
            if first_token_stall
            else request.param
        )
        capabilities = capabilities.model_copy(update={"failover_only_on": (rule,)})
    models["beta"] = beta.model_copy(
        update={
            "gateway": beta.gateway.model_copy(
                update={"exact_model_id": "secondary-exact", "capabilities": capabilities}
            )
        }
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
    connections = dict(catalog.connections)
    if zdr_child:
        connections[beta.connection] = connections[beta.connection].model_copy(
            update={"provider": "openrouter", "base_url": None}
        )
    authored = catalog.model_copy(
        update={
            "models": models,
            "connections": connections,
            "gateway_pools": {},
            "gateway_model_chains": {"model-revision-exact": chain},
        }
    )
    pool_id = "alpha"
    if unavailable_child:
        parent = chain.model_copy(
            update={
                "pool_id": "coding",
                "rungs": (
                    GatewayDeploymentRung(deployment_id="alpha"),
                    GatewayModelReferenceRung(model_id="retired-child"),
                    GatewayDeploymentRung(deployment_id="beta"),
                ),
            }
        )
        retired = GatewayModelChain(
            model_id="retired-child",
            pool_id="absent-child-pool",
            revision="retired",
            available=False,
        )
        authored = catalog.model_copy(
            update={
                "gateway_model_chains": {parent.model_id: parent, retired.model_id: retired},
            }
        )
        pool_id = "coding"
    write_model_catalog(tmp_path / "models.toml", authored)
    publish_authored_chain_fixture(tmp_path, revision_id="revision-stage", pool_id=pool_id)
    driver = tmp_path / "driver.py"
    source = _DRIVER_SOURCE.replace(
        "from exp.runtime.gateway.lifecycle import load_gateway_components",
        "from exp.runtime.gateway.tests.chain_authority_fixture_test import "
        "chain_components as load_gateway_components",
    )
    if mapping_isolation or child_collision:
        source = source.replace(
            "    control_plane = NativeControlPlane(",
            "    original_admit = NativeControlPlane.admit\n"
            "    def distinct_root_mapping(self, payload):\n"
            '        """Give the failed root a distinct frozen inverse to detect leakage."""\n'
            "        admitted = json.loads(original_admit(self, payload))\n"
            "        for wire in admitted['route']:\n"
            "            if wire['deployment_id'] == 'alpha':\n"
            "                wire['native_tool_translation'] = {\n"
            "                    'apply_patch': ['root-only', None, True],\n"
            "                    'agents__close': ['root-close', 'root-space', False]}\n"
            "        return json.dumps(admitted)\n"
            "    NativeControlPlane.admit = distinct_root_mapping\n"
            "    control_plane = NativeControlPlane(",
        )
    if zdr_child:
        source = source.replace(
            "    components = load_gateway_components(",
            "    from exp.runtime.models import registry\n"
            "    factory, _origin = registry._HTTP_PROVIDERS['openrouter']\n"
            "    registry._HTTP_PROVIDERS['openrouter'] = (factory, config['child_origin'])\n"
            "    components = load_gateway_components(",
        )
        source = source.replace(
            "    control_plane = NativeControlPlane(",
            "    from exp.runtime.gateway.routing import CatalogRouteResolver\n"
            "    original = CatalogRouteResolver.resolve_direct\n"
            "    def constrained(self, authorization):\n"
            '        """Apply only the fixture host\'s frozen child ZDR policy."""\n'
            "        route = original(self, authorization)\n"
            "        return route.model_copy(update={'snapshot': route.snapshot.model_copy(\n"
            "            update={'zdr_constrained_deployment_ids': ('beta',)})})\n"
            "    CatalogRouteResolver.resolve_direct = constrained\n"
            "    control_plane = NativeControlPlane(",
        )
    if getattr(request, "param", None) in {"capture-sync", "capture-async", "capture-references"}:
        source = source.replace(
            "    control_plane = NativeControlPlane(",
            "    from exp.runtime.gateway.native_capture import (\n"
            "        CaptureConfiguration, CaptureController)\n"
            "    records = Path(config['root']) / 'capture.jsonl'\n"
            "    def write_capture(value):\n"
            "        with records.open('a') as sink:\n"
            "            sink.write(value + '\\n')\n"
            "    def write_batch(values):\n"
            "        for value in values: write_capture(value)\n"
            "        return [True] * len(values)\n"
            "    conf = CaptureConfiguration(asynchronous_delivery=config['capture_async'])\n"
            "    collector = (exp_gateway_native.CaptureCollector.batched(\n"
            "        conf.model_dump_json(), write_batch, completion_references=True)\n"
            "        if config['capture_references'] else\n"
            "        exp_gateway_native.CaptureCollector(conf.model_dump_json(), write_capture))\n"
            "    capture = CaptureController(collector, application_for=lambda _auth: 'fixture')\n"
            "    control_plane = NativeControlPlane(",
        ).replace(
            '        request_timeout_seconds=config["request_timeout_seconds"],',
            '        request_timeout_seconds=config["request_timeout_seconds"], capture=capture,',
        )
        source = source.replace(
            "    last_error = None",
            "    original_admit, original_settle = control_plane.admit, control_plane.settle\n"
            "    def funded(argument):\n"
            "        result = json.loads(original_admit(argument))\n"
            "        for wire in result.get('route', []):\n"
            "            wire['billing_customer_managed'] = False\n"
            "        return json.dumps(result)\n"
            "    def settled(argument):\n"
            "        result = original_settle(argument)\n"
            "        payload = json.loads(argument)\n"
            "        if payload.get('finalize', True):\n"
            "            collector.settle(payload['request_id'], True, True)\n"
            "        return result\n"
            "    control_plane.admit, control_plane.settle = funded, settled\n"
            "    last_error = None",
        ).replace(
            "                ),\n            )",
            "                ), capture=collector,\n            )",
        )
    driver.write_text(source + "\n")
    log_path = tmp_path / "driver.log"
    environment = {**os.environ, "TEST_PROVIDER_KEY": "loopback-secret"}
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                str(driver),
                json.dumps(
                    {
                        "root": str(tmp_path),
                        "request_timeout_seconds": 10,
                        "time_to_first_token_seconds": 1.0 if first_token_stall else 120.0,
                        "child_origin": f"http://127.0.0.1:{secondary.server_port}/v1",
                        "capture_async": getattr(request, "param", None) != "capture-sync",
                        "capture_references": getattr(request, "param", None)
                        == "capture-references",
                    }
                ),
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


@pytest.mark.parametrize("engine", ["unavailable-child"], indirect=True)
def test_unavailable_child_has_no_attempt_and_parent_suffix_still_serves(
    engine: _ServingEngine,
) -> None:
    """Actual native dispatch skips the retired model without listing or funding a fake lane."""
    headers = {"authorization": f"Bearer {engine.raw_key}"}
    response = httpx.post(
        f"{engine.base}/v1/chat/completions",
        headers=headers,
        json={"model": "coding", "messages": [{"role": "user", "content": "always-500"}]},
        timeout=30,
    )
    assert response.status_code == 200, response.text
    assert response.headers["x-gateway-canonical-model"] == "model-revision-exact"
    assert response.headers["x-gateway-deployment"] == "beta"
    assert "from-secondary" in response.text
    request_id = response.headers["x-request-id"]
    with sqlite3.connect(engine.database_path) as db:
        rows = db.execute(
            "SELECT exact_model_id,pool_id,deployment_id,state FROM gateway_attempts "
            "WHERE request_id=? ORDER BY attempt_ordinal",
            (request_id,),
        ).fetchall()
    assert rows == [
        ("model-revision-exact", "coding", "alpha", "failed"),
        ("model-revision-exact", "coding", "alpha", "failed"),
        ("model-revision-exact", "coding", "beta", "completed"),
    ]
    listed = httpx.get(f"{engine.base}/v1/models", headers=headers, timeout=10)
    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()["data"]] == ["coding"]


@pytest.mark.parametrize(
    "engine", ["capture-sync", "capture-async", "capture-references"], indirect=True
)
@pytest.mark.parametrize("child", [False, True])
def test_capture_keeps_root_request_model_and_actual_winning_deployment(
    engine: _ServingEngine, child: bool
) -> None:
    """Existing full/reference records retain root identity without disguising the actual child."""
    response = httpx.post(
        f"{engine.base}/v1/chat/completions",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json={
            "model": "coding",
            "messages": [{"role": "user", "content": "always-500" if child else "hello"}],
        },
        timeout=20,
    )
    assert response.status_code == 200, response.text
    expected_model = "secondary-exact" if child else "model-revision-exact"
    expected_deployment = "beta" if child else "alpha"
    assert response.headers["x-gateway-canonical-model"] == expected_model
    capture_path = engine.database_path.parent.parent / "capture.jsonl"
    deadline = time.monotonic() + 3
    records: list[JsonObject] = []
    while time.monotonic() < deadline:
        if capture_path.exists():
            records = [json.loads(line) for line in capture_path.read_text().splitlines()]
            if any(record.get("deployment_id") == expected_deployment for record in records):
                break
        time.sleep(0.01)
    assert len(records) == 2, records
    prompt, terminal = records
    assert prompt["schema_version"] == 1 and prompt["deployment_id"] is None
    assert terminal["deployment_id"] == expected_deployment
    assert terminal["schema_version"] in (1, 2)
    for record in records:
        request = record["request"]
        assert isinstance(request, dict) and request["model_id"] == "model-revision-exact"
        assert "canonical_model_id" not in record
    terminal_request = terminal["request"]
    assert isinstance(terminal_request, dict)
    if terminal["schema_version"] == 2:
        assert "context" not in terminal_request
    else:
        assert "context" in terminal_request
    with sqlite3.connect(engine.database_path) as db:
        row = db.execute(
            "SELECT exact_model_id,deployment_id FROM gateway_attempts WHERE state='completed'"
        ).fetchone()
    assert row == (expected_model, expected_deployment)


@pytest.mark.parametrize("engine", ["first-token-stall"], indirect=True)
def test_semantic_first_token_stall_enters_only_the_timeout_eligible_child(
    engine: _ServingEngine,
) -> None:
    """Root headers and keepalives cannot consume the deadline without reaching the child model."""
    started = time.monotonic()
    response = httpx.post(
        f"{engine.base}/v1/chat/completions",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json={"model": "coding", "messages": [{"role": "user", "content": "stall-after-headers"}]},
        timeout=15,
    )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "from-secondary"
    assert response.headers["x-gateway-canonical-model"] == "secondary-exact"
    assert time.monotonic() - started < 6
    assert _attempt_rows(engine, response.headers["x-request-id"]) == [
        (0, 0, "failed"),
        (1, 1, "completed"),
    ]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "engine,allowed",
    [("provider_internal", True), ("refusal", False), ("tool-map-isolation", True)],
    indirect=["engine"],
)
def test_actual_responses_tool_translation_survives_conditional_child_stage(
    engine: _ServingEngine, allowed: bool, stream: bool
) -> None:
    """Real Rust dispatch inverts selected child tools while preserving rule and ledger identity."""
    response = httpx.post(
        f"{engine.base}/v1/responses",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json={
            "model": "coding",
            "input": "always-500",
            "stream": stream,
            "tools": [
                {"type": "custom", "name": "apply_patch", "description": "Apply a patch."},
                {
                    "type": "namespace",
                    "name": "agents",
                    "tools": [
                        {
                            "type": "function",
                            "name": "close",
                            "parameters": {
                                "type": "object",
                                "properties": {"id": {"type": "string"}},
                                "required": ["id"],
                                "additionalProperties": False,
                            },
                        }
                    ],
                },
            ],
        },
        timeout=30,
    )
    with sqlite3.connect(engine.database_path) as db:
        rows = db.execute(
            "SELECT deployment_id,exact_model_id,state,fallback_reason "
            "FROM gateway_attempts ORDER BY attempt_ordinal"
        ).fetchall()
    assert rows[:2] == [("alpha", "model-revision-exact", "failed", None)] * 2
    if not allowed:
        assert response.status_code == 502
        assert len(rows) == 2
        return
    assert response.status_code == 200, response.text
    assert response.headers["x-gateway-canonical-model"] == "secondary-exact"
    assert rows[2:] == [
        ("beta", "secondary-exact", "completed", "failover_only_on:provider_internal")
    ]
    if stream:
        events = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        completed = next(
            event["response"] for event in events if event["type"] == "response.completed"
        )
    else:
        completed = response.json()
    output = completed["output"]
    custom = next(item for item in output if item["type"] == "custom_tool_call")
    namespaced = next(item for item in output if item["type"] == "function_call")
    assert custom["name"] == "apply_patch" and custom["input"] == "portable patch"
    assert namespaced["name"] == "close" and namespaced["namespace"] == "agents"
    assert json.loads(namespaced["arguments"]) == {"id": "agent-one"}
    assert completed["model"] == "coding"


@pytest.mark.parametrize("engine", ["zdr-child"], indirect=True)
def test_actual_child_zdr_constraint_and_upstream_identity_survive_fallback(
    engine: _ServingEngine,
) -> None:
    """The actual child wire, committed identity and settled upstream retain host ZDR policy."""
    response = httpx.post(
        f"{engine.base}/v1/chat/completions",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json={
            "model": "coding",
            "messages": [{"role": "user", "content": "always-500"}],
            "provider": {"zdr": False, "data_collection": "allow", "order": ["fixture-upstream"]},
        },
        timeout=30,
    )
    assert response.status_code == 200, response.text
    assert response.headers["x-gateway-canonical-model"] == "secondary-exact"
    assert response.headers["x-gateway-zdr-constrained"] == "true"
    assert response.headers["x-gateway-deployment"] == "beta"
    content = json.loads(response.json()["choices"][0]["message"]["content"])
    assert content == {
        "from-secondary": True,
        "metadata": "enabled",
        "provider": {"zdr": True, "data_collection": "deny", "order": ["fixture-upstream"]},
    }
    with sqlite3.connect(engine.database_path) as db:
        assert db.execute(
            "SELECT exact_model_id,upstream_provider,state FROM gateway_attempts "
            "ORDER BY attempt_ordinal"
        ).fetchall() == [
            ("model-revision-exact", None, "failed"),
            ("model-revision-exact", None, "failed"),
            ("secondary-exact", "fixture-upstream", "completed"),
        ]


@pytest.mark.parametrize(
    "engine,allowed", [("provider_internal", True), ("refusal", False)], indirect=["engine"]
)
def test_native_child_stage_honors_conditional_failover(
    engine: _ServingEngine, allowed: bool
) -> None:
    """The actual native child wire is reachable only for its authored upstream failure token."""
    response = httpx.post(
        f"{engine.base}/v1/chat/completions",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json={"model": "coding", "messages": [{"role": "user", "content": "always-500"}]},
        timeout=30,
    )
    with sqlite3.connect(engine.database_path) as db:
        attempts = db.execute(
            "SELECT deployment_id,state,fallback_reason FROM gateway_attempts "
            "ORDER BY attempt_ordinal"
        ).fetchall()
    assert attempts[:2] == [("alpha", "failed", None), ("alpha", "failed", None)]
    assert response.status_code == (200 if allowed else 502)
    if allowed:
        assert response.headers["x-gateway-canonical-model"] == "secondary-exact"
        assert attempts[2:] == [("beta", "completed", "failover_only_on:provider_internal")]
    else:
        assert response.status_code == 502
        assert len(attempts) == 2


@pytest.mark.parametrize("surface", ["chat/completions", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_actual_stage_header_alias_and_single_request_ledger(
    engine: _ServingEngine, surface: str, stream: bool
) -> None:
    """One request redials root once then commits different exact model on every surface."""
    payload: JsonObject = {"model": "coding", "stream": stream}
    if surface == "chat/completions":
        payload["response_format"] = {"type": "json_object"}
        payload["verbosity"] = "low"
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
    if surface == "chat/completions":
        if stream:
            chunks = [
                json.loads(line[6:])
                for line in response.text.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            text = "".join(
                choice.get("delta", {}).get("content", "")
                for chunk in chunks
                for choice in chunk.get("choices", [])
            )
            assert any(
                chunk.get("x-experiential-ignored-parameters") == ["verbosity"] for chunk in chunks
            )
        else:
            message = response.json()["choices"][0]["message"]
            assert "tool_calls" not in message
            text = message["content"]
            assert response.json()["x-experiential-ignored-parameters"] == ["verbosity"]
        assert json.loads(text) == {
            "from-secondary": True,
            "response_format": {"type": "json_object"},
            "verbosity_sent": False,
        }
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


@pytest.mark.parametrize("engine", ["tool-child-cancel"], indirect=True)
def test_actual_child_custom_start_disconnect_settles_once(engine: _ServingEngine) -> None:
    """Only the selected child cancels; the failed parent is never re-entered."""
    with httpx.stream(
        "POST",
        f"{engine.base}/v1/responses",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json={
            "model": "coding",
            "input": "always-500",
            "stream": True,
            "tools": [
                {"type": "custom", "name": "apply_patch"},
                {
                    "type": "namespace",
                    "name": "agents",
                    "tools": [
                        {"type": "function", "name": "close", "parameters": {"type": "object"}},
                    ],
                },
                {"type": "function", "name": "plain", "parameters": {"type": "object"}},
            ],
        },
        timeout=30,
    ) as response:
        assert response.status_code == 200
        assert response.headers["x-gateway-canonical-model"] == "secondary-exact"
        request_id = response.headers["x-request-id"]
        for line in response.iter_lines():
            if line.startswith("data: "):
                event = json.loads(line[6:])
                if event["type"] == "response.output_item.added":
                    assert event["item"]["type"] == "custom_tool_call"
                    break
    assert _attempt_rows(engine, request_id) == [
        (0, 0, "failed"),
        (1, 0, "failed"),
        (2, 1, "cancelled"),
    ]
    with sqlite3.connect(engine.database_path) as db:
        assert db.execute("SELECT count(*) FROM gateway_requests").fetchone() == (1,)
        assert db.execute(
            "SELECT count(*) FROM gateway_attempts WHERE state IN ('dispatched','running')"
        ).fetchone() == (0,)


@pytest.mark.parametrize("engine", ["tool-child-collision"], indirect=True)
@pytest.mark.parametrize("stream", [False, True])
def test_actual_child_collision_uses_its_own_declared_and_inverse_names(
    engine: _ServingEngine, stream: bool
) -> None:
    """Failed-root inverse metadata cannot rename any selected child's colliding tool."""

    response = httpx.post(
        f"{engine.base}/v1/responses",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json=_collision_request(stream),
        timeout=30,
    )
    _assert_collision_response(response, stream)
    assert response.headers["x-gateway-canonical-model"] == "secondary-exact"
    assert _attempt_rows(engine, response.headers["x-request-id"]) == [
        (0, 0, "failed"),
        (1, 0, "failed"),
        (2, 1, "completed"),
    ]
