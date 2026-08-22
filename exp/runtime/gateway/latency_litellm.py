"""Pinned LiteLLM proxy process used by the same-run latency comparison.

CI installs PyPI ``litellm[proxy]==1.97.0`` (August 2026 stable) and this module
starts the official config-file proxy without a database. That matches the
LiteLLM "Running without a database" deployment path: one YAML model list,
master-key auth, and ``litellm --config``. The equivalent signed image is
``ghcr.io/berriai/litellm:v1.97.0``. The proxy listens on loopback next to
the Experiential gateway so both share the same mock upstream and sampler.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path

from exp.runtime.gateway.latency_measure import (
    ALIAS_ID,
    MOCK_MODEL,
    unused_loopback_port,
    wait_for_http_ok,
)

LITELLM_PIN = "1.97.0"
LITELLM_IMAGE = "ghcr.io/berriai/litellm:v1.97.0"
LITELLM_MASTER_KEY = "sk-latency-compare"
LITELLM_START_TIMEOUT_S = 45.0
LITELLM_STARTUP = (
    f"litellm[proxy]=={LITELLM_PIN} via `litellm --config <generated.yaml> "
    "--host 127.0.0.1 --port <port>`; official config-file proxy without a "
    f"database (equivalent image {LITELLM_IMAGE}); openai/ prefix model "
    f"{MOCK_MODEL} aliased as {ALIAS_ID}; master_key auth; drop_params true"
)


def write_litellm_config(
    path: Path,
    *,
    api_base: str,
    api_key: str,
    master_key: str = LITELLM_MASTER_KEY,
) -> None:
    """Write the pinned LiteLLM config that forwards one alias to the mock.

    The ``openai/`` model prefix is LiteLLM's documented OpenAI-compatible
    custom endpoint form. ``api_base`` is the mock's ``/v1`` URL.

    Args:
        path: Destination YAML path.
        api_base: Mock OpenAI-compatible base URL, including ``/v1``.
        api_key: Credential the mock accepts.
        master_key: Proxy bearer token required on client requests.
    """
    path.write_text(
        "\n".join(
            (
                "model_list:",
                f"  - model_name: {ALIAS_ID}",
                "    litellm_params:",
                f"      model: openai/{MOCK_MODEL}",
                f"      api_base: {api_base}",
                f"      api_key: {api_key}",
                "litellm_settings:",
                "  drop_params: true",
                "  set_verbose: false",
                "general_settings:",
                f"  master_key: {master_key}",
                "",
            )
        ),
        encoding="utf-8",
    )


def resolve_litellm_executable() -> Path:
    """Return the ``litellm`` CLI next to this interpreter, or on PATH.

    Returns:
        Absolute path to the LiteLLM proxy CLI.

    Raises:
        RuntimeError: LiteLLM is not installed in this environment.
    """
    sibling = Path(sys.executable).with_name("litellm")
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return sibling
    found = shutil.which("litellm")
    if found:
        return Path(found)
    raise RuntimeError(
        "LiteLLM is not installed. CI installs "
        f"litellm[proxy]=={LITELLM_PIN} before --compare-litellm "
        f"(`uv run --with 'litellm[proxy]=={LITELLM_PIN}'`)."
    )


def installed_litellm_version() -> str:
    """Return the installed LiteLLM version string, or the pin if unknown.

    Returns:
        Version from ``importlib.metadata``, otherwise ``LITELLM_PIN``.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.metadata as metadata; print(metadata.version('litellm'))",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    version = result.stdout.strip()
    return version or LITELLM_PIN


def start_litellm_process(
    *,
    config_path: Path,
    port: int | None = None,
    executable: Path | None = None,
) -> tuple[subprocess.Popen[str], int]:
    """Launch the pinned LiteLLM proxy on loopback and wait until it is live.

    Args:
        config_path: YAML written by :func:`write_litellm_config`.
        port: Loopback TCP port, or ``None`` to choose one.
        executable: Optional CLI path. Defaults to the resolved install.

    Returns:
        Live process and the bound loopback port.

    Raises:
        RuntimeError: The process exits or does not become ready in time.
    """
    bound_port = unused_loopback_port() if port is None else port
    command = executable or resolve_litellm_executable()
    env = os.environ.copy()
    env.update(
        {
            "DO_NOT_TRACK": "1",
            "LITELLM_LOG": "ERROR",
            "LITELLM_MASTER_KEY": LITELLM_MASTER_KEY,
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            "PYTHONUNBUFFERED": "1",
        }
    )
    process = subprocess.Popen(
        [
            str(command),
            "--config",
            str(config_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(bound_port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    lines: list[str] = []
    pump = threading.Thread(
        target=_pump_process_output,
        args=(process, lines),
        daemon=True,
    )
    pump.start()
    try:
        wait_for_http_ok(
            f"http://127.0.0.1:{bound_port}/health/liveliness",
            process=process,
            lines=lines,
            timeout_s=LITELLM_START_TIMEOUT_S,
            label="litellm",
        )
    except Exception:
        stop_litellm_process(process)
        raise
    return process, bound_port


def stop_litellm_process(process: subprocess.Popen[str]) -> None:
    """Signal the LiteLLM proxy and wait for a bounded exit.

    Args:
        process: Live or already terminated proxy process.
    """
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
    try:
        process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=5)


def _pump_process_output(process: subprocess.Popen[str], lines: list[str]) -> None:
    """Drain proxy stdout so the pipe cannot fill.

    Args:
        process: LiteLLM subprocess with a captured stdout pipe.
        lines: Destination list for complete output lines.
    """
    if process.stdout is None:
        return
    for line in process.stdout:
        lines.append(line)
