"""Release archive and metadata checks for the current single-package distribution."""

from __future__ import annotations

import errno
import os
import pty
import re
import select
import shutil
import signal
import subprocess
import sys
import tarfile
import termios
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import cast

if os.environ.get("WMO_INSTALLED_RELEASE_EVIDENCE") != "1":
    import pytest

BUILT_DIST_ENV = "WMO_BUILT_DIST_DIR"
FORBIDDEN_REQUIREMENT = re.compile(
    r"(?mi)^Requires-Dist:\s*(?:anthropic|boto3|environment-capture|gepa|mlx-lm|"
    r"opentelemetry-proto|scikit-learn|transformers)(?:\s|[<>=;~!])"
)
REQUIRED_CORE_REQUIREMENTS = frozenset(
    {
        "click",
        "fastapi",
        "filelock",
        "httpx",
        "numpy",
        "openai",
        "posthog",
        "pydantic",
        "rich",
        "tomli-w",
        "typer",
        "uvicorn",
    }
)
REQUIRED_WHEEL_MODULES = frozenset(
    {
        "wmo/common/judging/calibration.py",
        "wmo/common/judging/labels.py",
        "wmo/common/judging/review.py",
        "wmo/common/models/model.py",
        "wmo/runtime/environments/local.py",
        "wmo/runtime/models/registry.py",
        "wmo/runtime/router/application.py",
        "wmo/simulation/comparison.py",
        "wmo/simulation/engines/sandbox.py",
        "wmo/optimize/router/automatic/service.py",
        "wmo/optimize/router/composition.py",
        "wmo/optimize/router/evaluation/setup.py",
        "wmo/optimize/router/fit/workflow.py",
        "wmo/optimize/router/judging/service.py",
        "wmo/optimize/router/judgment_budget.py",
    }
)
REQUIRED_SDIST_MEMBERS = frozenset(
    {
        "README.md",
        "assets/wmo-workflow.png",
        "pyproject.toml",
        "wmo/optimize/router/automatic/service.py",
        "wmo/optimize/router/composition.py",
        "wmo/optimize/router/evaluation/setup.py",
        "wmo/optimize/router/fit/workflow.py",
        "wmo/optimize/router/judging/service.py",
        "wmo/optimize/router/judgment_budget.py",
    }
)
FORBIDDEN_FLAT_ROUTER_MODULES = frozenset(
    {
        "wmo/optimize/router/automatic_router.py",
        "wmo/optimize/router/automatic_router_artifacts.py",
        "wmo/optimize/router/automatic_router_judge.py",
        "wmo/optimize/router/automatic_router_preflight.py",
        "wmo/optimize/router/automatic_router_replay.py",
        "wmo/optimize/router/automatic_router_reservations.py",
        "wmo/optimize/router/completed_build.py",
        "wmo/optimize/router/manual_judge.py",
        "wmo/optimize/router/manual_judge_artifacts.py",
        "wmo/optimize/router/manual_judge_contracts.py",
        "wmo/optimize/router/manual_judge_protocol.py",
        "wmo/optimize/router/manual_judge_selection.py",
        "wmo/optimize/router/optimizer.py",
        "wmo/optimize/router/persistence.py",
        "wmo/optimize/router/report.py",
        "wmo/optimize/router/router_attribution.py",
        "wmo/optimize/router/router_execution_contract.py",
        "wmo/optimize/router/router_setup.py",
        "wmo/optimize/router/router_simulation_spec.py",
        "wmo/optimize/router/simulation_spend.py",
        "wmo/optimize/router/spec.py",
        "wmo/optimize/router/workflow.py",
    }
)
_RELEASE_REVISION = "1" * 40


def _normalized_path(member_name: str) -> str:
    """Remove the versioned sdist root from one archive path."""
    parts = PurePosixPath(member_name).parts
    if parts and parts[0].startswith("world_model_optimizer-"):
        parts = parts[1:]
    return PurePosixPath(*parts).as_posix()


def _wheel_metadata(archive: zipfile.ZipFile) -> str:
    """Return the wheel's unique core metadata document."""
    paths = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
    assert len(paths) == 1, f"wheel has unexpected METADATA paths: {paths}"
    return archive.read(paths[0]).decode("utf-8")


def _sdist_metadata(archive: tarfile.TarFile) -> str:
    """Return the sdist's unique core metadata document."""
    members = [
        member
        for member in archive.getmembers()
        if member.isfile() and member.name.endswith("/PKG-INFO")
    ]
    assert len(members) == 1, f"sdist has unexpected PKG-INFO paths: {members}"
    extracted = archive.extractfile(members[0])
    assert extracted is not None
    return extracted.read().decode("utf-8")


def _core_requirement_names(metadata: str) -> frozenset[str]:
    """Return normalized non-extra dependency names from package metadata."""
    names: set[str] = set()
    for line in metadata.splitlines():
        if not line.startswith("Requires-Dist:") or "; extra ==" in line:
            continue
        requirement = line.removeprefix("Requires-Dist:").strip()
        name = re.split(r"[<>=;~!\s]", requirement, maxsplit=1)[0].casefold()
        names.add(name)
    return frozenset(names)


def _assert_current_archive_members(
    names: tuple[str, ...],
    *,
    required: frozenset[str],
    allow_tests: bool,
) -> None:
    """Validate archive membership against the current release contract.

    Args:
        names: Normalized paths contained in the built archive.
        required: Current paths that the archive must contain.
        allow_tests: Whether test modules are valid archive members.
    """
    file_names = frozenset(name for name in names if name and not name.endswith("/"))
    removed_package = PurePosixPath("wmo", "workflow").as_posix() + "/"
    removed_members = sorted(name for name in file_names if name.startswith(removed_package))
    assert not removed_members, f"archive contains removed package: {removed_members}"
    flat_router_members = sorted(file_names & FORBIDDEN_FLAT_ROUTER_MODULES)
    assert not flat_router_members, (
        f"archive contains flat router implementation modules: {flat_router_members}"
    )
    assert required.issubset(file_names), (
        f"archive is missing current members: {required - file_names}"
    )
    if allow_tests:
        return
    tests = sorted(
        name for name in file_names if name.endswith("_test.py") or name == "wmo/conftest.py"
    )
    assert not tests, f"wheel contains test members: {tests}"


def _tracked_sdist_members() -> frozenset[str]:
    """Return current tracked members admitted by the explicit sdist include contract."""
    result = subprocess.run(
        [
            "git",
            "ls-files",
            ".gitignore",
            "README.md",
            "assets",
            "pyproject.toml",
            "conftest.py",
            "wmo",
        ],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        check=True,
    )
    return frozenset(result.stdout.splitlines())


def _tracked_wheel_members() -> frozenset[str]:
    """Return the exact current source members admitted by the wheel contract."""
    return frozenset(
        name
        for name in _tracked_sdist_members()
        if name.startswith("wmo/") and not name.endswith("_test.py") and name != "wmo/conftest.py"
    )


def _run_checked(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str] | None = None,
    timeout: float = 300,
) -> subprocess.CompletedProcess[str]:
    """Run one release-evidence subprocess and retain readable failure output.

    Args:
        command: Exact executable and argument vector.
        cwd: Isolated working directory for the subprocess.
        environment: Optional complete child environment.
        timeout: Maximum subprocess duration in seconds.

    Returns:
        Completed successful subprocess result.

    Raises:
        AssertionError: The subprocess exits unsuccessfully.
    """
    result = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    assert result.returncode == 0, (
        f"command failed ({result.returncode}): {' '.join(command)}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


def _run_tty_child(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    answers: list[tuple[str, str]],
    completion_marker: str | None = None,
    timeout: float = 30,
) -> str:
    """Drive one interactive child and require a clean bounded exit.

    Args:
        command: Exact executable and argument vector.
        cwd: Isolated working directory for the child.
        environment: Complete child environment.
        answers: Ordered prompt substring and terminal answer pairs.
        completion_marker: Optional output required before terminal input closes.
        timeout: Maximum total child lifetime in seconds.

    Returns:
        Combined terminal transcript.

    Raises:
        AssertionError: The child times out, exits unsuccessfully, omits a prompt, or omits the
            required completion marker.
        OSError: The pseudo-terminal fails for a reason other than normal slave closure.
    """
    master, slave = pty.openpty()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdin=slave,
        stdout=slave,
        stderr=slave,
        close_fds=True,
    )
    os.close(slave)
    transcript = ""
    search_from = 0
    pending = list(answers)
    input_closed = False
    completion_seen = completion_marker is None
    terminal_closed = False
    deadline = time.monotonic() + timeout
    try:
        while process.poll() is None and time.monotonic() < deadline:
            readable, _, _ = select.select([master], [], [], 0.1)
            if readable:
                try:
                    chunk = os.read(master, 65_536)
                except OSError as error:
                    if error.errno != errno.EIO:
                        raise
                    terminal_closed = True
                    break
                if not chunk:
                    terminal_closed = True
                    break
                transcript += chunk.decode(errors="replace")
            if pending and (prompt_position := transcript.find(pending[0][0], search_from)) >= 0:
                prompt, answer = pending[0]
                search_from = prompt_position + len(prompt)
                try:
                    os.write(master, (answer + "\n").encode())
                except OSError as error:
                    if error.errno != errno.EIO:
                        raise
                    terminal_closed = True
                    break
                pending.pop(0)
            if completion_marker is not None and completion_marker in transcript:
                completion_seen = True
            if not pending and completion_seen and not input_closed:
                try:
                    canonical_input = bool(termios.tcgetattr(master)[3] & termios.ICANON)
                except termios.error as error:
                    if error.args and error.args[0] == errno.EIO:
                        terminal_closed = True
                        break
                    if process.poll() is None and not terminal_closed:
                        raise
                    continue
                if canonical_input:
                    try:
                        os.write(master, b"\x04")
                    except OSError as error:
                        if error.errno != errno.EIO:
                            raise
                        terminal_closed = True
                        break
                    input_closed = True
        if terminal_closed and process.poll() is None:
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=5)
            raise AssertionError(f"interactive CLI timed out:\n{transcript}")
        if not terminal_closed:
            while True:
                readable, _, _ = select.select([master], [], [], 0)
                if not readable:
                    break
                try:
                    chunk = os.read(master, 65_536)
                except OSError as error:
                    if error.errno != errno.EIO:
                        raise
                    break
                if not chunk:
                    break
                transcript += chunk.decode(errors="replace")
    finally:
        os.close(master)
    assert process.returncode == 0, transcript
    assert not pending, f"unanswered prompts {pending}:\n{transcript}"
    assert completion_seen, f"missing completion marker {completion_marker!r}:\n{transcript}"
    return transcript


def _installed_release_driver() -> None:
    """Execute the isolated installed-wheel no-spend release evidence flow.

    Raises:
        AssertionError: Any package, CLI, API, artifact, replay, or no-spend invariant fails.
    """
    import hashlib
    import json
    import math
    import socket
    import threading
    from collections import Counter
    from datetime import UTC, datetime
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from openai import OpenAI
    from openai.types.responses import FunctionToolParam

    import wmo
    from wmo.common.models import EmbeddingCostReservation, load_model_catalog
    from wmo.common.project import ProjectStore, artifact_input
    from wmo.optimize.model.sft import (
        RuntimeInteractionExampleSource,
        load_sft_model_optimization_config,
        load_verified_sft_dataset,
        prepare_runtime_sft_model_optimization,
    )
    from wmo.optimize.model.sft.selection import load_latest_sft_model_optimization
    from wmo.optimize.router.judging.service import prepare_manual_judge_calibration
    from wmo.runtime.models import CapabilityRequirement, RuntimeModelCatalog
    from wmo.runtime.router import (
        RuntimeAcceptedEvent,
        RuntimeCompletedEvent,
        RuntimeInteractionJournal,
    )
    from wmo.simulation.retrieval import (
        RAGEmbedderBinding,
        load_completed_build_rag_lineage_bindings,
        refresh_runtime_trace_rag,
    )

    execution_root = Path.cwd().resolve()
    source_checkout = Path(os.environ["WMO_SOURCE_CHECKOUT"]).resolve()
    assert execution_root != source_checkout
    assert not execution_root.is_relative_to(source_checkout)
    assert source_checkout not in tuple(Path(item).resolve() for item in sys.path if item)
    package_path = Path(wmo.__file__).resolve()
    assert "site-packages" in package_path.parts, package_path
    assert not any(part == "world-model-optimizer" for part in package_path.parts)

    class ProviderState:
        """Thread-safe deterministic request log for the loopback fake provider."""

        def __init__(self) -> None:
            """Initialize an empty provider request log."""
            self._lock = threading.Lock()
            self.requests: list[dict[str, object]] = []

        def append(self, path: str, payload: dict[str, object]) -> int:
            """Record one provider request and return its stable ordinal.

            Args:
                path: Loopback HTTP request path.
                payload: Decoded JSON request body.

            Returns:
                One-based request ordinal.
            """
            with self._lock:
                self.requests.append({"path": path, "payload": payload})
                return len(self.requests)

        def counts(self) -> Counter[str]:
            """Return request counts grouped by HTTP path."""
            with self._lock:
                return Counter(str(item["path"]) for item in self.requests)

        def snapshot(self) -> tuple[dict[str, object], ...]:
            """Return a detached ordered copy of recorded provider requests."""
            with self._lock:
                return tuple(dict(item) for item in self.requests)

    state = ProviderState()

    class ProviderHandler(BaseHTTPRequestHandler):
        """Minimal OpenAI-compatible handler for deterministic zero-price evidence."""

        protocol_version = "HTTP/1.0"

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            """Suppress nondeterministic HTTP server logs."""
            del format, args

        def do_POST(self) -> None:
            """Serve deterministic embeddings and chat completions on loopback only.

            Raises:
                AssertionError: A structured judge request contains no real evidence span.
            """
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                self.send_error(400)
                return
            ordinal = state.append(self.path, payload)
            if self.path == "/v1/embeddings":
                values = payload.get("input", [])
                texts = [values] if isinstance(values, str) else list(values)
                data = []
                for index, text in enumerate(texts):
                    digest = hashlib.sha256(str(text).encode()).digest()
                    raw = [float(value + 1) for value in digest[:8]]
                    norm = math.sqrt(sum(value * value for value in raw))
                    data.append(
                        {
                            "object": "embedding",
                            "index": index,
                            "embedding": [value / norm for value in raw],
                        }
                    )
                response = {
                    "object": "list",
                    "data": data,
                    "model": payload.get("model"),
                    "usage": {
                        "prompt_tokens": max(len(texts), 1),
                        "total_tokens": max(len(texts), 1),
                    },
                }
            else:
                prompt = json.dumps(payload, sort_keys=True)
                model = str(payload.get("model"))
                if model == "core-model" and "evidence_span_ids" in prompt:
                    messages = payload.get("messages")
                    assert isinstance(messages, list) and messages, prompt
                    user_content = messages[-1].get("content")
                    assert isinstance(user_content, str), prompt
                    rollout_text = user_content.partition("ROLLOUT:\n")[2].partition(
                        "\n\nRUBRIC:\n"
                    )[0]
                    rollout = json.loads(rollout_text)
                    spans = rollout.get("spans")
                    assert isinstance(spans, list) and spans, prompt
                    span_id = spans[0].get("span_id")
                    assert isinstance(span_id, str) and span_id, prompt
                    content = json.dumps(
                        {
                            "dimensions": [
                                {
                                    "dimension_id": "task-success",
                                    "raw_score": 5,
                                    "evidence_span_ids": [span_id],
                                    "feedback": "Deterministic loopback evidence.",
                                }
                            ]
                        }
                    )
                    message: dict[str, object] = {"role": "assistant", "content": content}
                    finish_reason = "stop"
                elif model == "core-model" and "terminal" in prompt:
                    message = {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "message": "P17 generated world observation",
                                "terminal": True,
                            }
                        ),
                    }
                    finish_reason = "stop"
                elif payload.get("tools"):
                    tool = payload["tools"][0]
                    function = tool["function"]
                    message = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call-{ordinal}",
                                "type": "function",
                                "function": {
                                    "name": function["name"],
                                    "arguments": json.dumps({"ticket_id": "42"}),
                                },
                            }
                        ],
                    }
                    finish_reason = "tool_calls"
                else:
                    content = (
                        "Duplicate routed target"
                        if "Duplicate routed example" in prompt
                        else f"Deterministic loopback response {ordinal}"
                    )
                    message = {
                        "role": "assistant",
                        "content": content,
                    }
                    finish_reason = "stop"
                response = {
                    "id": f"chatcmpl-{ordinal}",
                    "object": "chat.completion",
                    "created": 1_760_000_000,
                    "model": model,
                    "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                    "usage": {
                        "prompt_tokens": 8,
                        "completion_tokens": 4,
                        "total_tokens": 12,
                    },
                }
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), ProviderHandler)
    server.daemon_threads = True
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    provider_url = f"http://127.0.0.1:{server.server_port}/v1"

    root = execution_root / ".wmo"
    traces = execution_root / "support.otel.jsonl"
    one_trace = execution_root / "one.otel.jsonl"
    executable = Path(sys.executable).with_name("wmo")
    child_environment = os.environ.copy()
    child_environment.update(
        {
            "WMO_RELEASE_REVISION": _RELEASE_REVISION,
            "WMO_INSTALLED_RELEASE_EVIDENCE": "0",
            "P17_PROVIDER_KEY": "deterministic-loopback-placeholder",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            "HTTP_PROXY": "http://127.0.0.1:9",
            "HTTPS_PROXY": "http://127.0.0.1:9",
            "ALL_PROXY": "http://127.0.0.1:9",
            "http_proxy": "http://127.0.0.1:9",
            "https_proxy": "http://127.0.0.1:9",
            "all_proxy": "http://127.0.0.1:9",
        }
    )

    def attribute(key: str, value: str) -> dict[str, object]:
        """Encode one text OTLP attribute.

        Args:
            key: Semantic-convention attribute name.
            value: Text attribute value.

        Returns:
            OTLP JSON attribute envelope.
        """
        return {"key": key, "value": {"stringValue": value}}

    def write_traces(
        path: Path,
        count: int,
        *,
        terminal_last_trace: bool = False,
    ) -> tuple[str, ...]:
        """Write observed traces attributed across router candidates.

        Args:
            path: Destination OTLP JSONL file.
            count: Number of distinct source lineages.
            terminal_last_trace: Whether the final trace lacks a later real observation.

        Returns:
            Ordered trace identities written to the file.
        """
        records: list[dict[str, object]] = []
        trace_ids = []
        for index in range(count):
            trace_id = f"{index + 1:032x}"
            trace_ids.append(trace_id)
            model = "core-model" if index % 2 == 0 else "candidate-b-model"
            base = 1_760_000_000_000_000_000 + index * 10_000_000_000
            common = [
                attribute("gen_ai.operation.name", "chat"),
                attribute("gen_ai.provider.name", "openai-compatible"),
                attribute("gen_ai.request.model", model),
                attribute("wmo.customer.id", f"customer-{index}"),
                attribute("wmo.conversation.id", f"conversation-{index}"),
            ]
            records.append(
                {
                    "traceId": trace_id,
                    "spanId": f"{index * 2 + 1:016x}",
                    "name": "agent.model_call",
                    "startTimeUnixNano": str(base),
                    "endTimeUnixNano": str(base + 1_000_000_000),
                    "attributes": common
                    + [
                        attribute(
                            "gen_ai.input.messages",
                            json.dumps([{"role": "user", "content": f"Support request {index}"}]),
                        ),
                        attribute(
                            "gen_ai.output.messages",
                            json.dumps(
                                [
                                    {
                                        "role": "assistant",
                                        "content": "What account email?",
                                    }
                                ]
                            ),
                        ),
                    ],
                }
            )
            if terminal_last_trace and index == count - 1:
                continue
            records.append(
                {
                    "traceId": trace_id,
                    "spanId": f"{index * 2 + 2:016x}",
                    "name": "agent.model_call",
                    "startTimeUnixNano": str(base + 2_000_000_000),
                    "endTimeUnixNano": str(base + 3_000_000_000),
                    "attributes": common
                    + [
                        attribute(
                            "gen_ai.input.messages",
                            json.dumps(
                                [
                                    {
                                        "role": "assistant",
                                        "content": "What account email?",
                                    },
                                    {
                                        "role": "user",
                                        "content": f"customer-{index}@example.test",
                                    },
                                ]
                            ),
                        ),
                        attribute(
                            "gen_ai.output.messages",
                            json.dumps(
                                [
                                    {
                                        "role": "assistant",
                                        "content": "Reset instructions sent.",
                                    }
                                ]
                            ),
                        ),
                    ],
                }
            )
        path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )
        return tuple(trace_ids)

    support_trace_ids = write_traces(traces, 10, terminal_last_trace=True)
    write_traces(one_trace, 1)

    def run_cli(*arguments: str, expected: int = 0) -> subprocess.CompletedProcess[str]:
        """Run the installed CLI without a terminal and validate its exit status.

        Args:
            arguments: CLI arguments after the installed executable.
            expected: Required process exit code.

        Returns:
            Completed CLI subprocess.
        """
        result = subprocess.run(
            [str(executable), *arguments],
            cwd=execution_root,
            env=child_environment,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert result.returncode == expected, (
            f"CLI exit {result.returncode}, expected {expected}: {arguments}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        return result

    def directory_digest(path: Path) -> str:
        """Hash every regular file below a directory in relative-path order.

        Args:
            path: Directory containing immutable evidence.

        Returns:
            SHA-256 digest of relative paths and file bytes.
        """
        digest = hashlib.sha256()
        for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
            digest.update(item.relative_to(path).as_posix().encode())
            digest.update(b"\0")
            digest.update(item.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()

    def assert_embedded_revision(value: object) -> None:
        """Require every recursively present code revision to match the installed release.

        Args:
            value: Decoded artifact JSON value to inspect recursively.
        """
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "code_revision":
                    assert item == _RELEASE_REVISION
                assert_embedded_revision(item)
        elif isinstance(value, list):
            for item in value:
                assert_embedded_revision(item)

    def unused_loopback_port() -> int:
        """Reserve and release one currently unused loopback TCP port."""
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            return int(probe.getsockname()[1])

    def wait_for_loopback(port: int, process: subprocess.Popen[str]) -> None:
        """Wait for one local server without allowing an unbounded startup hang.

        Args:
            port: Expected loopback TCP port.
            process: Server process that must remain live.

        Raises:
            AssertionError: The process exits or does not listen before the deadline.
        """
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout is not None else ""
                raise AssertionError(f"wmo run exited before startup:\n{output}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    return
            except OSError:
                time.sleep(0.05)
        raise AssertionError(f"wmo run did not listen on port {port}")

    def run_tty(
        arguments: list[str],
        answers: list[tuple[str, str]],
        *,
        completion_marker: str | None = None,
    ) -> str:
        """Drive one installed interactive CLI process through ordered prompt matches.

        Args:
            arguments: CLI arguments after the installed executable.
            answers: Ordered prompt substring and terminal answer pairs.
            completion_marker: Optional output that must appear before terminal input closes.

        Returns:
            Combined terminal transcript.
        """
        return _run_tty_child(
            [str(executable), *arguments],
            cwd=execution_root,
            environment=child_environment,
            answers=answers,
            completion_marker=completion_marker,
        )

    def model_answers(
        alias: str,
        model: str,
        *,
        embeddings: bool,
        add_another: bool,
    ) -> list[tuple[str, str]]:
        """Return ordered interactive answers for one explicit model declaration.

        Args:
            alias: Stable local model alias.
            model: Provider model identifier.
            embeddings: Whether the alias supports embeddings.
            add_another: Whether setup should collect another model afterward.

        Returns:
            Ordered prompt and answer pairs.
        """
        answers = [
            ("Connection", "loopback"),
            ("Model alias", alias),
            ("Provider model ID", model),
            ("Supports tools?", "y"),
            ("Supports embeddings?", "y" if embeddings else "n"),
            ("Supports structured output?", "y"),
            ("Supports chat completions?", "y"),
            ("Record context window tokens?", "y"),
            ("Context window tokens", "128000"),
            ("Record maximum output tokens?", "y"),
            ("Maximum output tokens", "16000"),
            ("Input cost per million tokens in USD", "0"),
            ("Output cost per million tokens in USD", "0"),
            ("Cached input cost per million tokens in USD", "0"),
            ("Cache write cost per million tokens in USD", "0"),
            ("Add another available model?", "y" if add_another else "n"),
        ]
        return answers

    setup_answers = [
        ("Add a OpenAI connection?", "n"),
        ("Add a OpenRouter connection?", "n"),
        ("Add a Anthropic connection?", "n"),
        ("Add a Gemini connection?", "n"),
        ("Add a OpenAI-compatible connection?", "y"),
        ("Connection name", "loopback"),
        ("API key environment variable", "P17_PROVIDER_KEY"),
        ("Base URL", provider_url),
        ("Add another OpenAI-compatible connection?", "n"),
        *model_answers("core", "core-model", embeddings=True, add_another=False),
        ("World model alias", "core"),
        ("Use 'core' as the judge?", "y"),
        ("Embedder alias", "core"),
        ("Save this configuration?", "y"),
        ("Proceed with at most", ""),
    ]
    try:
        assert not root.exists()
        build_output = run_tty(
            ["build", "support-agent", str(traces), "--root", str(root)],
            setup_answers,
        )
        assert "Model setup is required" in build_output
        assert "Candidate aliases" not in build_output
        support_store = ProjectStore(root, "support-agent")
        support_project = support_store.load_project()
        assert support_project.build is not None
        assert support_project.models is not None
        assert support_project.models.candidates == ()
        catalog_text = (root / "models.toml").read_text(encoding="utf-8")
        assert "deterministic-loopback-placeholder" not in catalog_text
        assert "P17_PROVIDER_KEY" in catalog_text
        build_counts = state.counts()
        assert build_counts["/v1/embeddings"] > 0
        assert build_counts["/v1/chat/completions"] == 0
        assert "Proceed with at most $" in build_output
        assert "embedding spend?" in build_output
        assert "World model:" in build_output
        assert "core-model" in build_output
        assert "Conservative maximum embedding cost:" in build_output
        assert "Configured build-cost ceiling:" in build_output
        dry_run = run_cli(
            "build",
            "support-agent",
            str(traces),
            "--root",
            str(root),
            "--dry-run",
            "--no-interactive",
        )
        assert "Dry run complete" in dry_run.stdout
        assert "Proceed?" not in dry_run.stdout
        assert "Conservative maximum embedding cost:" in dry_run.stdout
        assert state.counts()["/v1/embeddings"] == build_counts["/v1/embeddings"]
        assert support_store.load_project().build is not None
        replay_build = run_cli(
            "build",
            "support-agent",
            str(traces),
            "--root",
            str(root),
            "--no-interactive",
        )
        assert "Reusing completed grounded artifacts." in replay_build.stdout
        assert "Proceed?" not in replay_build.stdout
        assert state.counts()["/v1/embeddings"] == build_counts["/v1/embeddings"]
        world_model = wmo.load_world_model(
            "support-agent",
            root=root,
            environment={"P17_PROVIDER_KEY": "deterministic-loopback-placeholder"},
        )
        session = world_model.new_session(task="Help a customer reset their password")
        observation = world_model.step(
            session.id,
            {"role": "assistant", "content": "What account email is associated?"},
        )
        assert observation.message == {
            "role": "user",
            "content": "P17 generated world observation",
        }
        assert observation.terminal is True
        world_requests = state.snapshot()[sum(build_counts.values()) :]
        assert [item["path"] for item in world_requests] == [
            "/v1/embeddings",
            "/v1/chat/completions",
        ]
        world_prompt = json.dumps(world_requests[-1]["payload"], sort_keys=True)
        assert "What account email?" in world_prompt
        assert "customer-" in world_prompt
        setup_call_count = sum(state.counts().values())
        setup_result = run_cli(
            "config",
            "judge",
            "setup",
            "support-agent",
            "--root",
            str(root),
            "--preview-count",
            "10",
            "--approve",
            "--non-interactive",
        )
        assert "Saved judge setup" in setup_result.stdout
        assert sum(state.counts().values()) == setup_call_count
        calibration_plan = prepare_manual_judge_calibration(support_store, sample_size=10)
        assert len(calibration_plan.previews) >= 2
        calibration_arguments = [
            "config",
            "judge",
            "calibrate",
            "support-agent",
            "--root",
            str(root),
            "--sample-size",
            "10",
        ]
        for preview in calibration_plan.previews:
            calibration_arguments.extend(["--label", f"{preview.trace_id}:task-success=5"])
        calibration_arguments.extend(
            [
                "--input-usd-per-million",
                "0",
                "--output-usd-per-million",
                "0",
                "--maximum-cost-usd",
                "0.000001",
                "--yes",
                "--approve",
                "--accept-insufficient-labels",
                "--non-interactive",
            ]
        )
        calibration_result = run_cli(*calibration_arguments)
        assert "Approved judge calibration" in calibration_result.stdout
        assert sum(state.counts().values()) - setup_call_count == len(calibration_plan.previews)
        review = support_store.read_review()
        assert isinstance(review, dict)
        manual_judge = review.get("manual_judge")
        assert isinstance(manual_judge, dict)
        assert manual_judge.get("approved_calibration") is not None
        optimize_arguments = [
            "optimize",
            "router",
            "support-agent",
            "--root",
            str(root),
            "--maximum-provider-cost-usd",
            "0.000001",
            "--maximum-judgments",
            "100",
            "--preferred-fidelity-overlaps",
            "2",
            "--maximum-model-calls",
            "1",
            "--simulation-maximum-output-tokens",
            "8000",
            "--maximum-concurrency",
            "1",
            "--yes",
            "--approve-fidelity",
        ]
        optimization_output = run_tty(
            optimize_arguments,
            [
                ("Candidate alias", "candidate-b"),
                ("Provider connection", "loopback"),
                ("Provider model ID", "candidate-b-model"),
                ("Supports tools?", "y"),
                ("Context window tokens", "128000"),
                ("Maximum output tokens", "16000"),
                ("Input USD per million tokens", "0"),
                ("Output USD per million tokens", "0"),
                ("Cached input USD per million tokens", "0"),
                ("Cache write USD per million tokens", "0"),
                ("Candidate aliases (comma separated)", "core,candidate-b"),
                ("Incumbent alias", "core"),
                ("Save these router candidates?", "y"),
            ],
        )
        assert optimization_output.count("Candidate aliases (comma separated)") == 1
        assert "policy:" in optimization_output
        assert "report:" in optimization_output
        optimized_artifacts = directory_digest(support_store.paths.artifacts_directory)
        optimized_catalog = (root / "models.toml").read_bytes()
        optimized_project = support_store.paths.project_toml.read_bytes()
        optimized_review = support_store.paths.review_json.read_bytes()
        optimized_provider_requests = state.snapshot()
        replay_result = run_cli(
            *optimize_arguments[:-2],
            "--non-interactive",
        )
        assert "replay: verified completed optimization" in replay_result.stdout
        assert state.snapshot() == optimized_provider_requests
        assert (root / "models.toml").read_bytes() == optimized_catalog
        assert support_store.paths.project_toml.read_bytes() == optimized_project
        assert support_store.paths.review_json.read_bytes() == optimized_review
        assert directory_digest(support_store.paths.artifacts_directory) == optimized_artifacts

        router_port = unused_loopback_port()
        provider_calls_before_run = state.snapshot()
        router_process = subprocess.Popen(
            [
                str(executable),
                "run",
                "support-agent",
                "--root",
                str(root),
                "--port",
                str(router_port),
            ],
            cwd=execution_root,
            env=child_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            wait_for_loopback(router_port, router_process)
            assert state.snapshot() == provider_calls_before_run
            client = OpenAI(
                base_url=f"http://127.0.0.1:{router_port}/v1",
                api_key="unused-loopback-client-key",
            )
            chat = client.chat.completions.create(
                model="support-agent",
                messages=[{"role": "user", "content": "Look up ticket 42"}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup_ticket",
                            "description": "Look up one support ticket.",
                            "parameters": {
                                "type": "object",
                                "properties": {"ticket_id": {"type": "string"}},
                                "required": ["ticket_id"],
                                "additionalProperties": False,
                            },
                        },
                    }
                ],
            )
            assert chat.choices[0].message.tool_calls
            response_calls_before = state.counts()
            response_tool = FunctionToolParam(
                type="function",
                name="lookup_ticket",
                description="Look up one support ticket.",
                parameters={
                    "type": "object",
                    "properties": {"ticket_id": {"type": "string"}},
                    "required": ["ticket_id"],
                    "additionalProperties": False,
                },
                strict=None,
            )
            first_response = client.responses.create(
                model="support-agent",
                input="Look up ticket 42 through the Responses API",
                tools=[response_tool],
            )
            function_call = next(
                item for item in first_response.output if item.type == "function_call"
            )
            first_response_counts = state.counts()
            second_response = client.responses.create(
                model="support-agent",
                previous_response_id=first_response.id,
                input=[
                    {
                        "type": "function_call_output",
                        "call_id": function_call.call_id,
                        "output": "Ticket 42 is ready for reset.",
                    }
                ],
                tools=[response_tool],
            )
            assert second_response.previous_response_id == first_response.id
            second_response_counts = state.counts()
            assert first_response_counts["/v1/embeddings"] == (
                response_calls_before["/v1/embeddings"] + 1
            )
            assert (
                second_response_counts["/v1/embeddings"] == first_response_counts["/v1/embeddings"]
            )
            keyed_chat = client.chat.completions.create(
                model="support-agent",
                messages=[{"role": "user", "content": "Durable keyed request"}],
                extra_headers={"Idempotency-Key": "p17-durable-request"},
            )
            client.close()
        finally:
            router_process.terminate()
            try:
                router_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                router_process.kill()
                router_process.wait(timeout=5)
        journal = RuntimeInteractionJournal(support_store.paths)
        events_before_public_replay = journal.read_events()
        provider_before_public_replay = state.snapshot()
        with wmo.load_router(
            "support-agent",
            root=root,
            environment={"P17_PROVIDER_KEY": "deterministic-loopback-placeholder"},
        ) as loaded_router:
            replayed_chat = loaded_router.chat.completions.create(
                model="support-agent",
                messages=[{"role": "user", "content": "Durable keyed request"}],
                extra_headers={"Idempotency-Key": "p17-durable-request"},
            )
            assert replayed_chat.id == keyed_chat.id
            public_response = loaded_router.responses.create(
                model="support-agent",
                input="Programmatic router response",
            )
            assert public_response.output
            duplicate_first = loaded_router.chat.completions.create(
                model="support-agent",
                messages=[{"role": "user", "content": "Duplicate routed example"}],
            )
            duplicate_second = loaded_router.chat.completions.create(
                model="support-agent",
                messages=[{"role": "user", "content": "Duplicate routed example"}],
            )
            assert duplicate_first.choices[0].message.content == "Duplicate routed target"
            assert duplicate_second.choices[0].message.content == "Duplicate routed target"
        events = journal.read_events()
        assert state.snapshot() != provider_before_public_replay
        accepted = tuple(event for event in events if isinstance(event, RuntimeAcceptedEvent))
        completed = tuple(event for event in events if isinstance(event, RuntimeCompletedEvent))
        prior_completed = tuple(
            event
            for event in events_before_public_replay
            if isinstance(event, RuntimeCompletedEvent)
        )
        assert len(completed) == len(prior_completed) + 3
        assert len({event.interaction_id for event in completed}) == len(completed)
        assert accepted[1].lineage_id == accepted[2].lineage_id
        assert accepted[1].selected_alias == accepted[2].selected_alias
        current_project = support_store.load_project()
        assert current_project.build is not None
        assert current_project.models is not None
        completed_build = current_project.build
        completed_build_digests = {
            pointer.artifact_id: directory_digest(
                support_store.paths.artifacts_directory / pointer.artifact_id
            )
            for pointer in (
                completed_build.trace_dataset,
                completed_build.task_set,
                completed_build.serving_rag,
                completed_build.fit_rag,
                completed_build.world_model,
            )
        }
        imported_bindings = load_completed_build_rag_lineage_bindings(
            support_store.artifacts,
            completed_build,
        )
        assert {binding.trace_id for binding in imported_bindings} == set(support_trace_ids)
        catalog = load_model_catalog(root / "models.toml")
        resolved_embedder = RuntimeModelCatalog(
            catalog,
            environment={"P17_PROVIDER_KEY": "deterministic-loopback-placeholder"},
        ).preflight(
            current_project.models.embedder,
            CapabilityRequirement(requires_embeddings=True),
        )
        assert resolved_embedder.embedding_client is not None
        embedding_price = resolved_embedder.capabilities.input_cost_per_million_tokens_usd
        assert embedding_price == 0
        embedding_binding = RAGEmbedderBinding(
            client=resolved_embedder.embedding_client,
            snapshot=resolved_embedder.snapshot,
            maximum_attempts=3,
            input_usd_per_million_tokens=embedding_price,
        )
        embedding_reservation = EmbeddingCostReservation(
            model=resolved_embedder.snapshot,
            input_usd_per_million_tokens=embedding_price,
            maximum_attempts=embedding_binding.maximum_attempts,
            maximum_input_tokens=1_000_000,
        )
        refresh_time = datetime.now(UTC)
        provider_before_refresh = state.snapshot()
        refresh = refresh_runtime_trace_rag(
            journal,
            support_store.artifacts,
            (completed_build.trace_dataset,),
            imported_bindings,
            embedder=embedding_binding,
            embedding_reservation=embedding_reservation,
            maximum_embedding_cost_usd=0,
            created_at=refresh_time,
            code_revision=_RELEASE_REVISION,
        )
        provider_after_refresh = state.snapshot()
        assert len(provider_after_refresh) > len(provider_before_refresh)
        assert all(
            request["path"] == "/v1/embeddings"
            for request in provider_after_refresh[len(provider_before_refresh) :]
        )
        assert refresh.snapshot_export.snapshot.completed_target_count == len(completed)
        assert refresh.retrieval.index.rag_id not in {
            completed_build.serving_rag.artifact_id,
            completed_build.fit_rag.artifact_id,
        }
        assert refresh.dataset.dataset.dataset_id != completed_build.trace_dataset.artifact_id
        response_acceptances = tuple(
            event for event in accepted if event.lineage_id == accepted[1].lineage_id
        )
        assert len(response_acceptances) == 2
        observed_response = response_acceptances[0]
        terminal_response = response_acceptances[1]
        tool_transitions = tuple(
            transition
            for transition in refresh.retrieval.transitions
            if transition.trace_id == observed_response.interaction_id
            and transition.action.kind == "tool_call"
            and transition.observation.kind == "tool_result"
        )
        assert len(tool_transitions) == 1
        assert tool_transitions[0].action.tool_name == "lookup_ticket"
        assert tool_transitions[0].observation.content == "Ticket 42 is ready for reset."
        assert terminal_response.interaction_id not in {
            transition.trace_id for transition in refresh.retrieval.transitions
        }
        refresh_payload = json.dumps(
            [trace.model_dump(mode="json") for trace in refresh.dataset.traces],
            sort_keys=True,
        )
        assert "P17 generated world observation" not in refresh_payload
        assert support_store.load_project().build == completed_build
        assert {
            artifact_id: directory_digest(support_store.paths.artifacts_directory / artifact_id)
            for artifact_id in completed_build_digests
        } == completed_build_digests
        replayed_refresh = refresh_runtime_trace_rag(
            journal,
            support_store.artifacts,
            (completed_build.trace_dataset,),
            imported_bindings,
            embedder=embedding_binding,
            embedding_reservation=embedding_reservation,
            maximum_embedding_cost_usd=0,
            created_at=refresh_time,
            code_revision=_RELEASE_REVISION,
        )
        assert replayed_refresh.refresh.refresh_id == refresh.refresh.refresh_id
        assert replayed_refresh.retrieval.index.rag_id == refresh.retrieval.index.rag_id
        assert state.snapshot() == provider_after_refresh
        provider_before_model_optimization = state.snapshot()
        model_optimization_output = run_tty(
            [
                "optimize",
                "model",
                "support-agent",
                "--root",
                str(root),
                "--tinker-connection",
                "tinker-local",
                "--tinker-api-key-env",
                "P17_MISSING_TINKER_KEY",
                "--base-model-alias",
                "tinker-base",
                "--base-model",
                "fake-base-model",
                "--maximum-cost-usd",
                "1",
                "--training-usd-per-million-tokens",
                "0",
            ],
            [
                ("Use Tinker connection 'tinker-local'", "y"),
                ("Proceed?", "n"),
            ],
            completion_marker="Managed Tinker SFT was not started.",
        )
        assert "Managed Tinker SFT was not started." in model_optimization_output
        assert state.snapshot() == provider_before_model_optimization
        assert "P17_MISSING_TINKER_KEY" not in child_environment
        latest = load_latest_sft_model_optimization(support_store)
        assert latest is not None
        config = load_sft_model_optimization_config(
            support_store,
            latest.config.artifact_id,
        )
        dataset = load_verified_sft_dataset(support_store, latest.dataset.artifact_id)
        assert config.dataset == latest.dataset
        assert dataset.build_spec is not None
        assert dataset.build_spec.held_out_fraction == 0
        assert dataset.rows
        assert all(row.partition == "train" for row in dataset.rows)
        runtime_rows = tuple(
            row
            for row in dataset.rows
            if isinstance(row.example.source, RuntimeInteractionExampleSource)
        )
        assert len(runtime_rows) == len(completed)
        assert {
            cast(RuntimeInteractionExampleSource, row.example.source).interaction_id
            for row in runtime_rows
        } == {event.interaction_id for event in completed}
        duplicate_rows = tuple(
            row for row in runtime_rows if row.example.target.content == "Duplicate routed target"
        )
        assert len(duplicate_rows) == 2
        assert len({row.example.example_id for row in duplicate_rows}) == 2
        assert len({row.fingerprint for row in duplicate_rows}) == 1
        duplicate_groups = {row.example.leakage_group_id for row in duplicate_rows}
        assert len(duplicate_groups) == 2
        assert any(
            duplicate_groups.issubset(set(partition.leakage_group_ids))
            for partition in dataset.partitions
        )
        assert any(row.example.target.tool_calls for row in runtime_rows)
        assert "P17 generated world observation" not in json.dumps(
            [row.model_dump(mode="json") for row in dataset.rows],
            sort_keys=True,
        )
        replayed_preparation = prepare_runtime_sft_model_optimization(
            support_store,
            created_at=dataset.dataset.created_at,
            code_revision=_RELEASE_REVISION,
        )
        assert replayed_preparation.created is False
        assert replayed_preparation.dataset.dataset.dataset_id == dataset.dataset.dataset_id
        one_result = run_cli(
            "build",
            "one-trace",
            str(one_trace),
            "--root",
            str(root),
            "--no-interactive",
        )
        assert "100 to 1,000 traces is the usual starting range" in one_result.stdout
        one_store = ProjectStore(root, "one-trace")
        assert one_store.load_project().build is not None
        assert support_trace_ids
        for project_store in (support_store, one_store):
            for artifact_id in project_store.artifacts.list_ids():
                manifest = project_store.artifacts.read(artifact_id).manifest
                assert manifest.code_revision == _RELEASE_REVISION
                for file in manifest.files:
                    if not (file.path.endswith(".json") or file.path.endswith(".jsonl")):
                        continue
                    payload = project_store.artifacts.read_bytes(artifact_id, file.path)
                    for line in payload.decode("utf-8").splitlines():
                        if line.strip():
                            assert_embedded_revision(json.loads(line))
                for input_item in manifest.inputs:
                    input_manifest = project_store.artifacts.read(input_item.artifact_id).manifest
                    assert artifact_input(input_manifest) == input_item
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
        assert not server_thread.is_alive()


def test_tty_child_exit_survives_terminal_close_races(tmp_path: Path) -> None:
    """Treat terminal closure before child reaping as a clean bounded exit.

    Args:
        tmp_path: Pytest-owned working directory shared by isolated child processes.
    """
    from concurrent.futures import ThreadPoolExecutor

    child = (
        "import os, sys, time; "
        "answer = input('Prompt: '); "
        "assert answer == 'yes'; "
        "print('COMPLETE', flush=True); "
        "os.close(0); os.close(1); os.close(2); "
        "time.sleep(0.02)"
    )

    def invoke(_: int) -> str:
        """Run one child that closes its terminal before its process exits.

        Args:
            _: Unused invocation index supplied by the concurrent map.

        Returns:
            Complete prompt and completion-marker transcript.
        """
        return _run_tty_child(
            [sys.executable, "-c", child],
            cwd=tmp_path,
            environment=os.environ.copy(),
            answers=[("Prompt:", "yes")],
            completion_marker="COMPLETE",
            timeout=2,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        transcripts = tuple(executor.map(invoke, range(32)))
    assert all("Prompt:" in transcript for transcript in transcripts)
    assert all("COMPLETE" in transcript for transcript in transcripts)


def test_installed_wheel_no_spend_release_evidence(tmp_path: Path) -> None:
    """Prove the installed release happy path with deterministic loopback providers.

    Args:
        tmp_path: Pytest-owned build, virtual environment, and execution root.
    """
    repository = Path(__file__).resolve().parent.parent
    distribution = tmp_path / "dist"
    virtual_environment = tmp_path / "venv"
    execution = tmp_path / "execution"
    distribution.mkdir()
    execution.mkdir()
    uv = shutil.which("uv")
    assert uv is not None, "release evidence requires the repository's uv toolchain"
    environment = os.environ.copy()
    environment["UV_CACHE_DIR"] = str(tmp_path / "uv-cache")
    _run_checked(
        [uv, "build", "--wheel", "--out-dir", str(distribution)],
        cwd=repository,
        environment=environment,
    )
    wheel = tuple(distribution.glob("*.whl"))
    assert len(wheel) == 1
    _run_checked([uv, "venv", str(virtual_environment)], cwd=execution, environment=environment)
    installed_python = virtual_environment / "bin" / "python"
    _run_checked(
        [uv, "pip", "install", "--python", str(installed_python), str(wheel[0])],
        cwd=execution,
        environment=environment,
    )
    driver = execution / "installed_release_evidence.py"
    driver.write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    driver_environment = environment.copy()
    driver_environment.update(
        {
            "WMO_INSTALLED_RELEASE_EVIDENCE": "1",
            "WMO_RELEASE_REVISION": _RELEASE_REVISION,
            "WMO_TELEMETRY": "0",
            "PYTHONNOUSERSITE": "1",
            "WMO_SOURCE_CHECKOUT": str(repository),
        }
    )
    _run_checked(
        [str(installed_python), str(driver)],
        cwd=execution,
        environment=driver_environment,
        timeout=600,
    )


def test_built_archives_match_current_package_contract() -> None:
    """Prove fresh wheel and sdist match the current package contract.

    The test compares archive membership to tracked release sources, rejects package leakage and
    forbidden requirements, and validates the exact minimal core dependency set.
    """
    configured_dir = os.environ.get(BUILT_DIST_ENV)
    if configured_dir is None:
        pytest.skip(f"set {BUILT_DIST_ENV} to scan freshly built release archives")
    dist_dir = Path(configured_dir)
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    assert len(wheels) == 1, f"expected one wheel in {dist_dir}, found {wheels}"
    assert len(sdists) == 1, f"expected one sdist in {dist_dir}, found {sdists}"

    with zipfile.ZipFile(wheels[0]) as wheel:
        names = tuple(_normalized_path(name) for name in wheel.namelist())
        metadata = _wheel_metadata(wheel)
        _assert_current_archive_members(
            names,
            required=REQUIRED_WHEEL_MODULES,
            allow_tests=False,
        )
        assert frozenset(name for name in names if name.startswith("wmo/")) == (
            _tracked_wheel_members()
        )
        outside_package = sorted(
            name
            for name in wheel.namelist()
            if not name.startswith("wmo/") and ".dist-info/" not in name
        )
        assert not outside_package, f"wheel carries members outside the package: {outside_package}"
        assert FORBIDDEN_REQUIREMENT.search(metadata) is None
        assert _core_requirement_names(metadata) == REQUIRED_CORE_REQUIREMENTS

    with tarfile.open(sdists[0], mode="r:gz") as sdist:
        names = tuple(
            _normalized_path(member.name) for member in sdist.getmembers() if member.isfile()
        )
        metadata = _sdist_metadata(sdist)
        _assert_current_archive_members(names, required=REQUIRED_SDIST_MEMBERS, allow_tests=True)
        assert frozenset(name for name in names if name and not name.endswith("/")) == (
            _tracked_sdist_members() | {"PKG-INFO"}
        )
        assert FORBIDDEN_REQUIREMENT.search(metadata) is None
        assert _core_requirement_names(metadata) == REQUIRED_CORE_REQUIREMENTS


def test_w16_public_evidence_apis_resolve_from_release_owners() -> None:
    """W16 customer and comparison workflows resolve without test-only API owners."""
    import wmo
    from wmo.common.judging import HumanScoreReview, JudgeCalibrationService, RubricReview
    from wmo.runtime.environments import LocalProcessEnvironmentRuntime
    from wmo.simulation import compare_text_and_sandbox
    from wmo.simulation.engines import SandboxSimulator

    assert callable(wmo.compose_router)
    assert callable(wmo.load_project_router)
    assert callable(wmo.load_router)
    assert callable(wmo.create_project_router_app)
    assert callable(HumanScoreReview.open)
    assert callable(JudgeCalibrationService)
    assert callable(RubricReview.open)
    assert callable(compare_text_and_sandbox)
    assert SandboxSimulator.__module__ == "wmo.simulation.engines.sandbox"
    assert LocalProcessEnvironmentRuntime.__module__ == "wmo.runtime.environments.local"


def test_documentation_index_commands_and_release_scope_are_current() -> None:
    """Every indexed doc exists and release docs name current commands and explicit exclusions."""
    repository = Path(__file__).resolve().parent.parent
    docs = repository / "docs"
    index = (docs / "README.md").read_text(encoding="utf-8")
    indexed_paths = re.findall(r"\| `([^`]+\.md)` \|", index)
    assert indexed_paths
    assert not [path for path in indexed_paths if not (docs / path).is_file()]

    usage = (docs / "usage.md").read_text(encoding="utf-8")
    assert "wmo optimize router" in usage
    assert "wmo optimize model" in usage
    assert "wmo optimize route" not in usage.replace("wmo optimize router", "")

    scope = (docs / "release-scope.md").read_text(encoding="utf-8")
    for exclusion in (
        "No paid E2B or Harbor cloud smoke ran",
        "No real Tinker training ran",
        "No trained-versus-base behavioral comparison ran",
        "exactly $0.00 observed service spend",
    ):
        assert exclusion in scope

    ingest = (docs / "reference" / "ingest.md").read_text(encoding="utf-8")
    assert "PostHogPullRequest" in ingest
    assert "pull_posthog_traces" in ingest


if __name__ == "__main__" and os.environ.get("WMO_INSTALLED_RELEASE_EVIDENCE") == "1":
    _installed_release_driver()
