"""Live end-to-end walk of every locked CLI path against real OpenAI providers.

The ordinary gate excludes this module by name (``-k "not live"``); the nightly integration
workflow runs it with real credentials. It configures an OpenAI catalog around gpt-5.6-luna at
pinned maximum reasoning effort, builds immutable evidence from the pinned public terminal-tasks
export, calibrates the judge with real provider calls, optimizes a router over an attributed copy
of the export, serves the frozen policy over the OpenAI-compatible loopback surface, and proves
``optimize model`` fails closed without a persisted SFT configuration. Every provider budget is
explicit and bounded, and no credential is ever written to disk.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from typer.testing import CliRunner

from wmo.cli.app import app
from wmo.common.models.catalog import load_model_catalog
from wmo.common.project import ProjectStore
from wmo.optimize.router.judging.service import prepare_manual_judge_calibration
from wmo.runtime.models.registry import RuntimeModelCatalog
from wmo.simulation.ingest.environment_capture import canonicalize_environment_capture_payloads

TRACES_URL = (
    "https://huggingface.co/datasets/experiential-labs/wmo-terminal-tasks-traces/resolve/"
    "540883e451dc13d34fb50fdd36b143cb0f1fb0db/traces.otel.jsonl"
)
TRACES_SHA256 = "21c62cba7e3372cbf03df051dc2408699fbf8ea3561ba661b599e4949f0e5d42"
_PROJECT = "terminal-tasks"
_INCUMBENT_ALIAS = "mini"
_CALIBRATION_SAMPLE_SIZE = 10
_CALIBRATION_CEILING_USD = "5"
_ROUTER_CEILING_USD = "40"
_ROUTER_JUDGMENTS = "60"
_SERVER_PORT = 8399
_MODELS: tuple[dict[str, object], ...] = (
    {
        "alias": "world",
        "connection": "openai",
        "model": "gpt-5.6-luna",
        "supports_completions": True,
        "supports_temperature": False,
        "reasoning_effort": "xhigh",
        "supports_tools": True,
        "supports_structured_output": True,
        "maximum_output_tokens": 16000,
        "context_window_tokens": 400000,
        "input_cost_per_million_tokens_usd": 1.25,
        "output_cost_per_million_tokens_usd": 10.0,
        "cached_input_cost_per_million_tokens_usd": 0.125,
        "cache_write_cost_per_million_tokens_usd": 0.0,
    },
    {
        "alias": "judge",
        "connection": "openai",
        "model": "gpt-5.6-luna",
        "supports_completions": True,
        "supports_temperature": False,
        "reasoning_effort": "xhigh",
        "supports_structured_output": True,
        "maximum_output_tokens": 8000,
        "context_window_tokens": 400000,
        "input_cost_per_million_tokens_usd": 1.25,
        "output_cost_per_million_tokens_usd": 10.0,
        "cached_input_cost_per_million_tokens_usd": 0.125,
        "cache_write_cost_per_million_tokens_usd": 0.0,
    },
    {
        "alias": "mini",
        "connection": "openai",
        "model": "gpt-5.4-mini",
        "supports_completions": True,
        "supports_temperature": False,
        "supports_tools": True,
        "supports_structured_output": True,
        "maximum_output_tokens": 64000,
        "context_window_tokens": 400000,
        "input_cost_per_million_tokens_usd": 0.25,
        "output_cost_per_million_tokens_usd": 2.0,
        "cached_input_cost_per_million_tokens_usd": 0.025,
        "cache_write_cost_per_million_tokens_usd": 0.0,
    },
    {
        "alias": "nano",
        "connection": "openai",
        "model": "gpt-5.4-nano",
        "supports_completions": True,
        "supports_temperature": False,
        "supports_tools": True,
        "supports_structured_output": True,
        "maximum_output_tokens": 64000,
        "context_window_tokens": 400000,
        "input_cost_per_million_tokens_usd": 0.05,
        "output_cost_per_million_tokens_usd": 0.4,
        "cached_input_cost_per_million_tokens_usd": 0.005,
        "cache_write_cost_per_million_tokens_usd": 0.0,
    },
    {
        "alias": "embed",
        "connection": "openai",
        "model": "text-embedding-3-small",
        "supports_embeddings": True,
        "input_cost_per_million_tokens_usd": 0.02,
    },
)

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="live OpenAI credentials are required",
)


@pytest.fixture(name="pinned_traces")
def _pinned_traces(request: pytest.FixtureRequest) -> Path:
    """Return the hash-verified pinned public export, downloading it once per cache."""
    cache = Path(str(request.config.cache.mkdir("wmo-public-fixtures")))
    path = cache / "terminal-tasks-traces.otel.jsonl"
    if not path.is_file() or _sha256(path) != TRACES_SHA256:
        headers = {}
        token = os.environ.get("HF_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        fetch = urllib.request.Request(TRACES_URL, headers=headers)
        with urllib.request.urlopen(fetch, timeout=120) as response:
            path.write_bytes(response.read())
    assert _sha256(path) == TRACES_SHA256
    return path


def _sha256(path: Path) -> str:
    """Return the hex digest of one local file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _configure_providers(runner: CliRunner, root: Path) -> None:
    """Write the OpenAI catalog through the locked config providers surface."""
    arguments = [
        "config",
        "providers",
        "--root",
        str(root),
        "--non-interactive",
        "--replace",
        "--connection-json",
        json.dumps({"name": "openai", "provider": "openai", "api_key_env": "OPENAI_API_KEY"}),
    ]
    for model in _MODELS:
        arguments += ["--model-json", json.dumps(model)]
    arguments += ["--world-model", "world", "--judge", "judge", "--embedder", "embed"]
    result = runner.invoke(app, arguments)
    assert result.exit_code == 0, result.output


def _walk_telemetry(runner: CliRunner, root: Path) -> None:
    """Walk the telemetry status, enable, and disable paths."""
    for action, expectation in (
        ("status", "telemetry"),
        ("enable", "enabled"),
        ("disable", "disabled"),
    ):
        result = runner.invoke(app, ["config", "telemetry", action, "--root", str(root)])
        assert result.exit_code == 0, result.output
        assert expectation in result.output.lower()


def _attribute_export(root: Path, export: Path, destination: Path) -> None:
    """Write a copy of the export whose agent spans declare the incumbent identity.

    The public export intentionally carries no model identity, so router fidelity correctly
    fails closed on it. This derives a live-test-only attributed copy by declaring the
    configured incumbent on every canonical ``invoke_agent`` span.
    """
    catalog = RuntimeModelCatalog(load_model_catalog(root / "models.toml"))
    snapshot, _ = catalog.snapshot(_INCUMBENT_ALIAS)
    payloads = [json.loads(raw) for raw in export.read_text().splitlines()]
    canonical = canonicalize_environment_capture_payloads(payloads)
    assert canonical is not None, "pinned export no longer matches the capture profile"
    lines: list[str] = []
    for span in canonical:
        assert isinstance(span, dict)
        attributes = span["attributes"]
        assert isinstance(attributes, list)
        operations: set[str] = set()
        for item in attributes:
            assert isinstance(item, dict)
            if item["key"] != "gen_ai.operation.name":
                continue
            item_value = item["value"]
            assert isinstance(item_value, dict)
            string_value = item_value["stringValue"]
            assert isinstance(string_value, str)
            operations.add(string_value)
        if "invoke_agent" in operations:
            attributes.extend(
                [
                    {"key": "gen_ai.provider.name", "value": {"stringValue": snapshot.provider}},
                    {"key": "gen_ai.request.model", "value": {"stringValue": snapshot.model_id}},
                    {
                        "key": "wmo.model.capabilities_sha256",
                        "value": {"stringValue": snapshot.capabilities_sha256},
                    },
                    {
                        "key": "wmo.model.connection_sha256",
                        "value": {"stringValue": snapshot.connection_sha256},
                    },
                ]
            )
        lines.append(json.dumps(span))
    destination.write_text("\n".join(lines) + "\n")


def _calibrate_judge(runner: CliRunner, root: Path) -> None:
    """Label representative traces programmatically and calibrate with a bounded ceiling."""
    store = ProjectStore(root, _PROJECT)
    plan = prepare_manual_judge_calibration(store, sample_size=_CALIBRATION_SAMPLE_SIZE)
    labels = [
        argument
        for trace in plan.traces
        for argument in ("--label", f"{trace.trace_id}:task-success=4")
    ]
    result = runner.invoke(
        app,
        [
            "config",
            "judge",
            "calibrate",
            _PROJECT,
            "--root",
            str(root),
            "--sample-size",
            str(_CALIBRATION_SAMPLE_SIZE),
            "--input-usd-per-million",
            "1.25",
            "--output-usd-per-million",
            "10",
            "--maximum-cost-usd",
            _CALIBRATION_CEILING_USD,
            "--yes",
            "--approve",
            "--non-interactive",
            *labels,
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Approved judge calibration" in result.output


def _http_json(
    port: int,
    method: str,
    route: str,
    payload: dict[str, object] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, object]]:
    """Send one loopback JSON request and return the status code and parsed body."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    try:
        body = json.dumps(payload) if payload is not None else None
        request_headers = {"Content-Type": "application/json", **(headers or {})}
        connection.request(method, route, body=body, headers=request_headers)
        response = connection.getresponse()
        parsed = json.loads(response.read().decode("utf-8"))
        assert isinstance(parsed, dict)
        return response.status, parsed
    finally:
        connection.close()


def _wait_for_server(port: int, process: subprocess.Popen[bytes]) -> None:
    """Poll the models route until the loopback server answers or the process dies."""
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(f"wmo run exited early with code {process.returncode}")
        try:
            status, _ = _http_json(port, "GET", "/v1/models")
        except (ConnectionError, OSError):
            time.sleep(1)
            continue
        if status == 200:
            return
        time.sleep(1)
    raise AssertionError("wmo run never became ready on loopback")


def _serve_and_exercise(root: Path, *, ghost: bool) -> None:
    """Serve the frozen policy and walk the OpenAI-compatible endpoint surface."""
    command = [
        sys.executable,
        "-m",
        "wmo",
        "run",
        _PROJECT,
        "--root",
        str(root),
        "--port",
        str(_SERVER_PORT),
    ]
    if ghost:
        command.append("--ghost")
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait_for_server(_SERVER_PORT, process)
        status, models = _http_json(_SERVER_PORT, "GET", "/v1/models")
        assert status == 200
        listed = models["data"]
        assert isinstance(listed, list) and listed

        chat_payload: dict[str, object] = {
            "model": _PROJECT,
            "messages": [{"role": "user", "content": "Reply with the single word ready."}],
        }
        key = f"live-{'ghost' if ghost else 'durable'}-{int(time.time())}"
        status, first = _http_json(
            _SERVER_PORT,
            "POST",
            "/v1/chat/completions",
            chat_payload,
            {"Idempotency-Key": key},
        )
        assert status == 200, first
        choices = first["choices"]
        assert isinstance(choices, list) and choices

        if not ghost:
            status, replay = _http_json(
                _SERVER_PORT,
                "POST",
                "/v1/chat/completions",
                chat_payload,
                {"Idempotency-Key": key},
            )
            assert status == 200, replay
            assert replay["id"] == first["id"]

        status, response = _http_json(
            _SERVER_PORT,
            "POST",
            "/v1/responses",
            {"model": _PROJECT, "input": "Reply with the single word ready."},
        )
        assert status == 200, response
        assert response["status"] == "completed"
    finally:
        process.terminate()
        process.wait(timeout=30)


def test_live_openai_pipeline_covers_every_locked_cli_path(
    pinned_traces: Path, tmp_path: Path
) -> None:
    """Walk providers, telemetry, build, judge, router, run, and optimize model live."""
    root = tmp_path / ".wmo"
    root.mkdir()
    runner = CliRunner()

    _configure_providers(runner, root)
    providers = runner.invoke(
        app, ["config", "providers", "--root", str(root), "--non-interactive"]
    )
    assert providers.exit_code == 0, providers.output
    _walk_telemetry(runner, root)

    attributed = tmp_path / "traces.attributed.jsonl"
    _attribute_export(root, pinned_traces, attributed)
    build = runner.invoke(
        app,
        ["build", _PROJECT, str(attributed), "--root", str(root), "--yes"],
    )
    assert build.exit_code == 0, build.output

    setup = runner.invoke(
        app, ["config", "judge", "setup", _PROJECT, "--root", str(root), "--approve"]
    )
    assert setup.exit_code == 0, setup.output
    _calibrate_judge(runner, root)

    optimize = runner.invoke(
        app,
        [
            "optimize",
            "router",
            _PROJECT,
            "--root",
            str(root),
            "--candidate",
            "mini",
            "--candidate",
            "nano",
            "--incumbent",
            _INCUMBENT_ALIAS,
            "--maximum-provider-cost-usd",
            _ROUTER_CEILING_USD,
            "--maximum-judgments",
            _ROUTER_JUDGMENTS,
            "--preferred-fidelity-overlaps",
            "3",
            "--maximum-model-calls",
            "8",
            "--yes",
            "--approve-fidelity",
            "--non-interactive",
        ],
    )
    assert optimize.exit_code == 0, optimize.output

    _serve_and_exercise(root, ghost=False)
    _serve_and_exercise(root, ghost=True)

    sft = runner.invoke(app, ["optimize", "model", _PROJECT, "--root", str(root), "--yes"])
    assert sft.exit_code != 0
