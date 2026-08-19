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
    r"(?mi)^Requires-Dist:\s*(?:anthropic|environment-capture|gepa|mlx-lm|"
    r"opentelemetry-proto|scikit-learn|transformers)(?:\s|[<>=;~!])"
)
REQUIRED_CORE_REQUIREMENTS = frozenset(
    {
        "boto3",
        "botocore",
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
        "wmo/cli/gateway/app.py",
        "wmo/common/judging/calibration.py",
        "wmo/common/judging/labels.py",
        "wmo/common/judging/review.py",
        "wmo/common/models/model.py",
        "wmo/runtime/environments/local.py",
        "wmo/runtime/gateway/lifecycle.py",
        "wmo/runtime/gateway/ledger.py",
        "wmo/runtime/gateway/openai/requests.py",
        "wmo/runtime/gateway/provider_certification.py",
        "wmo/runtime/gateway/service.py",
        "wmo/runtime/gateway/usage.py",
        "wmo/runtime/models/registry.py",
        "wmo/runtime/router/application.py",
        "wmo/simulation/comparison.py",
        "wmo/simulation/engines/sandbox.py",
        "wmo/optimize/router/automatic/service.py",
        "wmo/optimize/router/composition.py",
        "wmo/optimize/router/evaluation/setup.py",
        "wmo/optimize/router/fit/workflow.py",
        "wmo/optimize/router/judging/service.py",
        "wmo/optimize/router/judging/review.py",
        "wmo/cli/judge_review.py",
        "wmo/optimize/router/judgment_budget.py",
    }
)
REQUIRED_SDIST_MEMBERS = frozenset(
    {
        "README.md",
        "assets/wmo-workflow.png",
        "docs/reference/gateway-architecture.md",
        "docs/release-scope.md",
        "docs/usage.md",
        "pyproject.toml",
        "wmo/optimize/router/automatic/service.py",
        "wmo/optimize/router/composition.py",
        "wmo/optimize/router/evaluation/setup.py",
        "wmo/optimize/router/fit/workflow.py",
        "wmo/optimize/router/judging/service.py",
        "wmo/optimize/router/judging/review.py",
        "wmo/cli/judge_review.py",
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
_TEST_RELEASE_REVISION = "1" * 40


def _assert_gateway_canaries_absent(
    root: Path,
    *,
    channels: dict[str, bytes | str],
    canaries: tuple[str, ...],
) -> None:
    """Reject raw content or secrets across durable and observable gateway channels.

    Args:
        root: Gateway root containing SQLite, WAL, backups, and catalog snapshots.
        channels: Named stdout, stderr, log, or HTTP response bodies.
        canaries: Raw content and secret values that must never appear.

    Raises:
        AssertionError: A canary appears in any scanned file or named channel.
    """
    encoded = tuple(canary.encode() for canary in canaries)
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        payload = path.read_bytes()
        for index, canary in enumerate(encoded):
            assert canary not in payload, (
                f"forbidden gateway canary {index} persisted in {path.relative_to(root)}"
            )
    for name, value in sorted(channels.items()):
        payload = value.encode() if isinstance(value, str) else value
        for index, canary in enumerate(encoded):
            assert canary not in payload, f"forbidden gateway canary {index} exposed in {name}"


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
            "docs/reference/gateway-architecture.md",
            "docs/release-scope.md",
            "docs/usage.md",
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


def _tty_answer_bytes(answer: str) -> bytes:
    """Encode one scripted terminal answer.

    A keyboard sequence, recognized by any escape or carriage return it contains, is written
    exactly so a raw-mode picker reads only the intended keys. Line-oriented prompts still
    receive a trailing newline.

    Args:
        answer: Scripted key sequence or line-oriented prompt answer.

    Returns:
        Bytes written to the child pseudo-terminal.
    """
    if "\x1b" in answer or "\r" in answer:
        return answer.encode()
    return (answer + "\n").encode()


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
                    os.write(master, _tty_answer_bytes(answer))
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
    import asyncio
    import hashlib
    import json
    import math
    import socket
    import sqlite3
    import stat
    import threading
    from collections import Counter
    from collections.abc import Sequence
    from datetime import UTC, datetime
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    import httpx
    import numpy as np
    import openai
    import uvicorn
    from openai import AsyncOpenAI, OpenAI
    from openai.types.responses import FunctionToolParam

    import wmo
    from wmo.cli.gateway.key_output import key_output_marker_path
    from wmo.common.models import (
        BillingSource,
        ConnectionConfig,
        Embedding,
        GatewayDeploymentCapabilities,
        GatewayEquivalenceCertification,
        GatewayTokenPrices,
        ModelCapabilities,
        ModelRequest,
        ModelResponse,
        ModelSnapshot,
        RoutedCandidateSnapshot,
    )
    from wmo.common.project import ProjectStore, artifact_input
    from wmo.common.routing import KnnGuard, KnnRouterPolicy
    from wmo.common.routing.bank import (
        CandidateEvidenceCount,
        KnnBankManifest,
        KnnEvidenceBank,
        bank_bytes,
    )
    from wmo.common.routing.features import ROUTER_FEATURE_SCHEMA_SHA256
    from wmo.optimize.router.judging.contracts import ManualJudgeTraceReviewArtifact
    from wmo.optimize.router.judging.service import prepare_manual_judge_calibration
    from wmo.runtime.gateway.catalog_authority import (
        upsert_certified_pool,
        upsert_connection,
        upsert_singleton_deployment,
    )
    from wmo.runtime.gateway.lifecycle import load_local_gateway
    from wmo.runtime.gateway.management import GatewayManagement
    from wmo.runtime.models import (
        CatalogRoleName,
        ResolvedModel,
        RuntimeModelCatalog,
    )
    from wmo.runtime.router import RouterRuntime, RuntimeInteractionJournal

    release_revision = os.environ["WMO_RELEASE_REVISION"]
    assert re.fullmatch(r"[0-9a-f]{40}", release_revision), release_revision
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

        def count_containing(self, value: str) -> int:
            """Count requests whose canonical record contains one marker.

            Args:
                value: Marker expected in a request path or payload.

            Returns:
                Number of matching physical provider requests.
            """
            with self._lock:
                return sum(value in json.dumps(item, sort_keys=True) for item in self.requests)

        def snapshot(self) -> tuple[dict[str, object], ...]:
            """Return a detached ordered copy of recorded provider requests."""
            with self._lock:
                return tuple(dict(item) for item in self.requests)

    class InstalledProjectClient:
        """Deterministic selection client that forbids router-owned completion."""

        def __init__(self) -> None:
            """Initialize selection and completion counters."""
            self.embed_calls = 0
            self.complete_calls = 0

        def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
            """Return one stable feature embedding for learned selection."""
            assert len(texts) == 1
            self.embed_calls += 1
            return (Embedding(values=(1.0, 0.0)),)

        def complete(self, request: ModelRequest) -> ModelResponse:
            """Reject completion because the gateway owns physical provider work."""
            del request
            self.complete_calls += 1
            raise AssertionError("project router completion must remain unused")

    class InstalledProjectCatalog:
        """Resolve frozen selection-only models without provider network work."""

        def __init__(
            self,
            snapshots: dict[str, ModelSnapshot],
            client: InstalledProjectClient,
        ) -> None:
            """Bind exact snapshots and the deterministic selection client."""
            self._snapshots = snapshots
            self._client = client

        def resolve(
            self,
            alias: str,
            *,
            role: CatalogRoleName | None = None,
        ) -> ResolvedModel:
            """Return one candidate or embedder binding by frozen alias."""
            del role
            snapshot = self._snapshots[alias]
            capabilities = ModelCapabilities(
                supports_embeddings=alias == "embedder",
            )
            return ResolvedModel(
                alias=alias,
                snapshot=snapshot,
                capabilities=capabilities,
                client=self._client,
                embedding_client=(self._client if capabilities.supports_embeddings else None),
            )

    def installed_project_runtime() -> tuple[RouterRuntime, InstalledProjectClient]:
        """Build one real packaged RouterRuntime for installed selection-only evidence."""
        digest = "a" * 64
        created_at = datetime(2026, 8, 19, tzinfo=UTC)
        snapshots: dict[str, ModelSnapshot] = {}
        for alias in ("baseline", "cheap", "embedder"):
            capabilities = ModelCapabilities(supports_embeddings=alias == "embedder")
            snapshots[alias] = ModelSnapshot(
                provider="installed-fixture",
                model_id=alias,
                revision="fixture",
                billing_source=BillingSource.CUSTOMER_MANAGED,
                capabilities_sha256=capabilities.identity_sha256(),
                connection_sha256="b" * 64,
            )
        bank = KnnEvidenceBank(
            task_ids=tuple(f"task-{index}" for index in range(8)),
            candidate_aliases=("baseline", "cheap"),
            embeddings=np.asarray(((1.0, 0.0),) * 8, dtype=np.float32),
            scores=np.asarray(((0.4, 1.0),) * 8, dtype=np.float32),
            candidate_costs=np.asarray(((0.5, 0.1),) * 8, dtype=np.float64),
            score_counts=np.ones((8, 2), dtype=np.int32),
            cost_counts=np.ones((8, 2), dtype=np.int32),
            workload_weights=np.ones(8, dtype=np.float64),
            novelty_floor=0.5,
        )
        bank_sha256 = hashlib.sha256(bank_bytes(bank)).hexdigest()
        manifest = KnnBankManifest(
            schema_version=1,
            created_at=created_at,
            code_revision=release_revision,
            bank_artifact_id="installed-project-bank",
            fit_evaluation_id="installed-project-fit",
            evaluation_plan_id="installed-project-plan",
            evaluation_plan_sha256=digest,
            task_set_id="installed-project-tasks",
            task_set_sha256=digest,
            task_ids=bank.task_ids,
            candidate_aliases=bank.candidate_aliases,
            evaluation_protocols_sha256=digest,
            embedder_alias="embedder",
            embedder=snapshots["embedder"],
            feature_extractor_id="request-visible-v2",
            feature_schema_sha256=ROUTER_FEATURE_SCHEMA_SHA256,
            pricing_snapshot_id="installed-project-pricing",
            pricing_snapshot_sha256=digest,
            bank_sha256=bank_sha256,
            embedding_dimension=2,
            novelty_floor=0.5,
            evidence_counts=tuple(
                CandidateEvidenceCount(
                    candidate_alias=alias,
                    scored_task_count=8,
                    costed_task_count=8,
                )
                for alias in bank.candidate_aliases
            ),
        )
        policy = KnnRouterPolicy(
            schema_version=1,
            created_at=created_at,
            code_revision=release_revision,
            policy_id="installed-project-policy",
            baseline_alias="baseline",
            candidates=tuple(
                RoutedCandidateSnapshot(alias=alias, model=snapshots[alias])
                for alias in bank.candidate_aliases
            ),
            embedder_alias="embedder",
            embedder=snapshots["embedder"],
            feature_extractor_id=manifest.feature_extractor_id,
            feature_schema_sha256=manifest.feature_schema_sha256,
            pricing_snapshot_id=manifest.pricing_snapshot_id,
            pricing_snapshot_sha256=manifest.pricing_snapshot_sha256,
            bank_artifact_id=manifest.bank_artifact_id,
            bank_sha256=bank_sha256,
            guard=KnnGuard(
                maximum_neighbors=8,
                minimum_paired_observations=8,
                relative_similarity_threshold=0.95,
                uncertainty_multiplier=0.5,
                quality_tolerance=0,
            ),
            fit_evaluation_id=manifest.fit_evaluation_id,
            evaluation_plan_id=manifest.evaluation_plan_id,
            evaluation_plan_sha256=manifest.evaluation_plan_sha256,
            task_set_id=manifest.task_set_id,
            task_set_sha256=manifest.task_set_sha256,
            evaluation_protocols_sha256=manifest.evaluation_protocols_sha256,
            judgment_status="provisional",
        )
        client = InstalledProjectClient()
        runtime = RouterRuntime(
            policy,
            manifest,
            bank,
            cast(RuntimeModelCatalog, InstalledProjectCatalog(snapshots, client)),
            pricing_snapshot_id=manifest.pricing_snapshot_id,
            pricing_snapshot_sha256=manifest.pricing_snapshot_sha256,
            pricing_candidate_aliases=manifest.candidate_aliases,
        )
        return runtime, client

    state = ProviderState()

    class ProviderHandler(BaseHTTPRequestHandler):
        """Minimal multi-protocol handler for deterministic gateway evidence."""

        protocol_version = "HTTP/1.0"

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            """Suppress nondeterministic HTTP server logs."""
            del format, args

        def do_GET(self) -> None:
            """Publish the model IDs this loopback account may call."""
            state.append(self.path, {})
            body = json.dumps(
                {
                    "object": "list",
                    "data": [{"id": "core-model"}, {"id": "candidate-b-model"}],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            """Serve deterministic embeddings and chat completions on loopback only.

            Raises:
                AssertionError: A structured judge request is missing required score fields.
            """
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                self.send_error(400)
                return
            ordinal = state.append(self.path, payload)
            if payload.get("model") in {"gateway-primary-model", "project-primary-model"}:
                self._send_primary_gateway_stream(payload, ordinal=ordinal)
                return
            if payload.get("model") in {"gateway-secondary-model", "project-secondary-model"}:
                self._send_gateway_stream(payload, ordinal=ordinal)
                return
            if payload.get("stream") is True and payload.get("model") in {
                "core-model",
                "candidate-b-model",
            }:
                self._send_gateway_stream(payload, ordinal=ordinal)
                return
            if self.path.endswith("/embeddings"):
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
                if model == "core-model" and (
                    "optional nullable rationale" in prompt or "Rationale is optional" in prompt
                ):
                    content = json.dumps(
                        {
                            "dimensions": [
                                {
                                    "dimension_id": "task-success",
                                    "raw_score": 1,
                                    "rationale": "Deterministic loopback evidence.",
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

        def _send_gateway_stream(self, payload: dict[str, object], *, ordinal: int) -> None:
            """Return one deterministic Chat SSE stream for gateway certification.

            Args:
                payload: Gateway-normalized upstream Chat request.
                ordinal: Stable provider request ordinal.
            """
            assert payload.get("stream") is True
            assert payload.get("stream_options") == {"include_usage": True}
            prompt = json.dumps(payload, sort_keys=True)
            if "sdk-retry-input-canary-P9" in prompt:
                self._send_retryable_failure()
                return
            if payload.get("tools"):
                choices = (
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": f"call-gateway-{ordinal}",
                                            "type": "function",
                                            "function": {
                                                "name": "lookup_ticket",
                                                "arguments": json.dumps(
                                                    {"ticket": tool_argument_canary},
                                                    separators=(",", ":"),
                                                ),
                                            },
                                        }
                                    ],
                                },
                                "finish_reason": "tool_calls",
                            }
                        ]
                    },
                )
            else:
                serialized = json.dumps(payload, sort_keys=True)
                if project_prompt_canary in serialized:
                    content = project_response_canary
                elif "Duplicate routed example" in serialized:
                    content = "Duplicate routed target"
                else:
                    content = response_canary
                choices = (
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "content": content,
                                },
                                "finish_reason": "stop",
                            }
                        ]
                    },
                )
            frames = [
                f"data: {json.dumps(frame, separators=(',', ':'))}\n\n".encode()
                for frame in choices
            ]
            frames.append(
                b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n'
            )
            frames.append(b"data: [DONE]\n\n")
            body = b"".join(frames)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_primary_gateway_stream(
            self,
            payload: dict[str, object],
            *,
            ordinal: int,
        ) -> None:
            """Return deterministic compatible SSE failures and committed output.

            Args:
                payload: Gateway-normalized compatible request.
                ordinal: Stable provider request ordinal.
            """
            prompt = json.dumps(payload, sort_keys=True)
            if (
                "prompt-content-canary-P9" in prompt
                or "tool-argument-canary" in prompt
                or "sdk-retry-input-canary-P9" in prompt
            ):
                self._send_retryable_failure()
                return
            if "project-prompt-canary-P9" in prompt:
                self._send_retryable_failure()
                return
            if "provider-auth-input-canary-P9" in prompt:
                body = b'{"error":{"message":"fixture credential rejected"}}'
                self.send_response(401)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if "refusal-input-canary-P9" in prompt:
                frames: tuple[dict[str, object], ...] = (
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"refusal": "policy refusal"},
                                "finish_reason": "content_filter",
                            }
                        ],
                    },
                )
                self._send_sse_frames(frames, done=True)
                return
            if "postcommit-input-canary-P9" in prompt:
                frames = cast(
                    tuple[dict[str, object], ...],
                    (
                        {
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": postcommit_response_canary},
                                    "finish_reason": None,
                                }
                            ]
                        },
                    ),
                )
                self._send_sse_frames(frames)
                return
            if "cancellation-input-canary-P9" in prompt:
                first = cast(
                    tuple[dict[str, object], ...],
                    (
                        {
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": cancellation_response_canary},
                                    "finish_reason": None,
                                }
                            ]
                        },
                    ),
                )
                terminal = cast(
                    tuple[dict[str, object], ...],
                    (
                        {
                            "choices": [],
                            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                        },
                    ),
                )
                self._send_sse_frames(
                    first,
                    trailing=terminal,
                    pause_seconds=1.0,
                    done=True,
                )
                return
            raise AssertionError(f"unexpected primary gateway request {ordinal}")

        def _send_sse_frames(
            self,
            frames: tuple[dict[str, object], ...],
            *,
            trailing: tuple[dict[str, object], ...] = (),
            pause_seconds: float = 0,
            done: bool = False,
        ) -> None:
            """Write deterministic SSE frames with an optional cancellation window.

            Args:
                frames: Frames written and flushed immediately.
                trailing: Frames written after the optional pause.
                pause_seconds: Delay before trailing frames.
                done: Whether to append the compatible terminal sentinel.
            """
            first = b"".join(
                f"data: {json.dumps(frame, separators=(',', ':'))}\n\n".encode() for frame in frames
            )
            remainder = b"".join(
                f"data: {json.dumps(frame, separators=(',', ':'))}\n\n".encode()
                for frame in trailing
            )
            if done:
                remainder += b"data: [DONE]\n\n"
            body = first + remainder
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(first)
            self.wfile.flush()
            if pause_seconds:
                time.sleep(pause_seconds)
            if remainder:
                try:
                    self.wfile.write(remainder)
                except BrokenPipeError:
                    pass

        def _send_retryable_failure(self) -> None:
            """Return one content-free retryable provider failure."""
            body = b'{"error":{"message":"temporary fixture failure"}}'
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), ProviderHandler)
    server.daemon_threads = True
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    provider_url = f"http://127.0.0.1:{server.server_port}/v1"
    azure_endpoint = provider_url.removesuffix("/v1")
    provider_embeddings_path = "/openai/v1/embeddings"
    provider_chat_path = "/openai/v1/chat/completions"
    azure_connection_sha256 = ConnectionConfig(
        provider="azure",
        base_url=azure_endpoint,
        api_key_env="AZURE_OPENAI_API_KEY",
        api_version="v1",
    ).identity_sha256()
    core_capabilities_sha256 = ModelCapabilities(
        supports_completions=True,
        supports_embeddings=True,
        supports_tools=True,
        supports_structured_output=True,
        context_window_tokens=128000,
        maximum_output_tokens=32000,
        input_cost_per_million_tokens_usd=0,
        output_cost_per_million_tokens_usd=0,
        cached_input_cost_per_million_tokens_usd=0,
        cache_write_cost_per_million_tokens_usd=0,
    ).identity_sha256()
    candidate_capabilities_sha256 = ModelCapabilities(
        supports_completions=True,
        supports_tools=True,
        context_window_tokens=128000,
        maximum_output_tokens=32000,
        input_cost_per_million_tokens_usd=0,
        output_cost_per_million_tokens_usd=0,
        cached_input_cost_per_million_tokens_usd=0,
        cache_write_cost_per_million_tokens_usd=0,
    ).identity_sha256()

    root = execution_root / ".wmo"
    traces = execution_root / "support.otel.jsonl"
    one_trace = execution_root / "one.otel.jsonl"
    executable = Path(sys.executable).with_name("wmo")
    child_environment = os.environ.copy()
    child_environment.update(
        {
            "WMO_RELEASE_REVISION": release_revision,
            "WMO_INSTALLED_RELEASE_EVIDENCE": "0",
            "AZURE_OPENAI_API_KEY": "deterministic-loopback-placeholder",
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
            capabilities_sha256 = (
                core_capabilities_sha256 if model == "core-model" else candidate_capabilities_sha256
            )
            base = 1_760_000_000_000_000_000 + index * 10_000_000_000
            common = [
                attribute("gen_ai.operation.name", "chat"),
                attribute("gen_ai.provider.name", "azure"),
                attribute("gen_ai.request.model", model),
                attribute("wmo.model.capabilities_sha256", capabilities_sha256),
                attribute("wmo.model.connection_sha256", azure_connection_sha256),
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
    cli_transcripts: list[str] = []

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
        cli_transcripts.extend((result.stdout, result.stderr))
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

    def gateway_attempt_rows(database: Path) -> tuple[dict[str, object], ...]:
        """Read stable content-free attempt evidence from the installed database.

        Args:
            database: Gateway SQLite path.

        Returns:
            Attempt rows ordered by durable creation identity.
        """
        with sqlite3.connect(database) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT attempt_id, request_id, attempt_ordinal, route_depth,
                       deployment_id, billing_source, state, failure_class,
                       input_tokens, cached_input_tokens, output_tokens,
                       reasoning_tokens, estimated_cost_micro_usd
                FROM gateway_attempts
                ORDER BY started_at, request_id, attempt_ordinal, attempt_id
                """
            ).fetchall()
        return tuple(dict(row) for row in rows)

    def frozen_billing_evidence(
        rows: tuple[dict[str, object], ...],
    ) -> dict[str, tuple[object, object]]:
        """Index immutable billing source and cost by physical attempt.

        Args:
            rows: Content-free attempt rows.

        Returns:
            Billing source and attributed cost keyed by attempt ID.
        """
        return {
            str(row["attempt_id"]): (row["billing_source"], row["estimated_cost_micro_usd"])
            for row in rows
        }

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

    def start_gateway(root: Path) -> tuple[subprocess.Popen[str], int, str]:
        """Start one installed gateway subprocess on an unused loopback port.

        Args:
            root: Configured gateway root.

        Returns:
            Process, selected port, and OpenAI-compatible base URL.
        """
        port = unused_loopback_port()
        process = subprocess.Popen(
            [
                str(executable),
                "run",
                "--root",
                str(root),
                "--port",
                str(port),
                "--non-interactive",
                "--graceful-timeout",
                "2",
            ],
            cwd=execution_root,
            env=child_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        wait_for_loopback(port, process)
        return process, port, f"http://127.0.0.1:{port}/v1"

    def stop_gateway(process: subprocess.Popen[str]) -> tuple[str, str]:
        """Stop one gateway process and return its complete observable output.

        Args:
            process: Live or already terminated gateway process.

        Returns:
            Complete stdout and stderr text.
        """
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 0, stdout + stderr
        return stdout, stderr

    gateway_root = execution_root / "gateway-empty"
    gateway_database = gateway_root / "gateway" / "gateway.db"
    empty_gateway = run_cli(
        "run",
        "--root",
        str(gateway_root),
        "--non-interactive",
        "--json",
        expected=2,
    )
    empty_payload = json.loads(empty_gateway.stdout)
    assert empty_payload["error"]["code"] == "gateway_not_initialized"
    assert not gateway_root.exists()
    initialized_gateway = run_cli(
        "config",
        "gateway",
        "init",
        "--root",
        str(gateway_root),
        "--non-interactive",
        "--json",
    )
    assert json.loads(initialized_gateway.stdout)["schema_version"] == 1

    canary_suffix = hashlib.sha256(str(execution_root).encode()).hexdigest()[:16]
    provider_secret = f"provider-secret-canary-P9-{canary_suffix}"
    prompt_canary = f"prompt-content-canary-P9-{canary_suffix}"
    response_canary = f"gateway-response-canary-{canary_suffix}"
    tool_argument_canary = f"tool-argument-canary-{canary_suffix}"
    invalid_key_canary = f"invalid-virtual-key-canary-P9-{canary_suffix}"
    refusal_input_canary = f"refusal-input-canary-P9-{canary_suffix}"
    auth_input_canary = f"provider-auth-input-canary-P9-{canary_suffix}"
    sdk_retry_input_canary = f"sdk-retry-input-canary-P9-{canary_suffix}"
    postcommit_input_canary = f"postcommit-input-canary-P9-{canary_suffix}"
    postcommit_response_canary = f"postcommit-response-canary-P9-{canary_suffix}"
    cancellation_input_canary = f"cancellation-input-canary-P9-{canary_suffix}"
    cancellation_response_canary = f"cancellation-response-canary-P9-{canary_suffix}"
    project_prompt_canary = f"project-prompt-canary-P9-{canary_suffix}"
    project_response_canary = f"project-response-canary-P9-{canary_suffix}"
    child_environment["P9_LOOPBACK_PROVIDER_KEY"] = provider_secret
    provider_commands = (
        (
            "config",
            "gateway",
            "provider",
            "add",
            "gateway-primary",
            "--provider",
            "openai-compatible",
            "--credential-env",
            "P9_LOOPBACK_PROVIDER_KEY",
            "--base-url",
            provider_url,
            "--root",
            str(gateway_root),
            "--non-interactive",
            "--json",
        ),
        (
            "config",
            "gateway",
            "provider",
            "add",
            "gateway-secondary",
            "--provider",
            "openai-compatible",
            "--credential-env",
            "P9_LOOPBACK_PROVIDER_KEY",
            "--base-url",
            provider_url,
            "--root",
            str(gateway_root),
            "--non-interactive",
            "--json",
        ),
    )
    for command in provider_commands:
        receipt = run_cli(*command)
        assert json.loads(receipt.stdout)["schema_version"] == 1

    catalog_sha256 = ""
    for alias, deployment, billing_source in (
        ("primary", "gateway-primary:gateway-primary-model", "host_managed"),
        ("secondary", "gateway-secondary:gateway-secondary-model", "customer_managed"),
    ):
        receipt = run_cli(
            "config",
            "gateway",
            "alias",
            "create",
            alias,
            "--deployment",
            deployment,
            "--exact-model",
            "gateway-model-revision",
            "--supports-tools",
            "--supports-structured-output",
            "--supports-developer-messages",
            "--supports-strict-tools",
            "--supports-parallel-tool-calls",
            "--maximum-output-tokens",
            "4096",
            "--input-price",
            "1000000",
            "--cached-input-price",
            "500000",
            "--output-price",
            "2000000",
            "--reasoning-price",
            "3000000",
            "--pricing-source",
            "deterministic loopback fixture",
            "--billing-source",
            billing_source,
            "--root",
            str(gateway_root),
            "--non-interactive",
            "--json",
        )
        catalog_sha256 = json.loads(receipt.stdout)["data"]["catalog_sha256"]

    for alias, revision, refusal_failover in (("coding", "installed-pool-revision", False),):
        command = [
            "config",
            "gateway",
            "pool",
            "certify",
            alias,
            "--deployment-alias",
            "primary",
            "--deployment-alias",
            "secondary",
            "--exact-model",
            "gateway-model-revision",
            "--certification-id",
            f"{alias}-certification",
            "--provenance",
            "installed-wheel deterministic loopback",
            "--evidence-sha256",
            "a" * 64,
            "--certified-at",
            "2026-08-19T00:00:00Z",
            "--expected-catalog-sha256",
            catalog_sha256,
            "--revision",
            revision,
            "--root",
            str(gateway_root),
            "--non-interactive",
            "--json",
        ]
        if refusal_failover:
            command.append("--refusal-failover")
        receipt = run_cli(*command)
        catalog_sha256 = json.loads(receipt.stdout)["data"]["catalog_sha256"]

    authority_commands = (
        (
            "config",
            "gateway",
            "identity",
            "create",
            "default",
            "--root",
            str(gateway_root),
            "--non-interactive",
            "--json",
        ),
        (
            "config",
            "gateway",
            "grant",
            "add",
            "default",
            "coding",
            "--root",
            str(gateway_root),
            "--non-interactive",
            "--json",
        ),
    )
    for command in authority_commands:
        receipt = run_cli(*command)
        assert json.loads(receipt.stdout)["schema_version"] == 1

    key_output_directory = execution_root / "gateway-key-output"
    key_output_directory.mkdir(mode=0o700)
    assert stat.S_IMODE(key_output_directory.stat().st_mode) == 0o700
    key_output = key_output_directory / "key.txt"
    issue_receipt = run_cli(
        "config",
        "gateway",
        "key",
        "issue",
        "default",
        "--key-id",
        "installed-key",
        "--output",
        str(key_output),
        "--root",
        str(gateway_root),
        "--non-interactive",
        "--json",
    )
    raw_key = key_output.read_text(encoding="utf-8").strip()
    assert raw_key.startswith("wmo_vk_")
    assert stat.S_IMODE(key_output.stat().st_mode) == 0o600
    assert raw_key not in issue_receipt.stdout

    def assert_key_and_canaries_confined() -> None:
        """Require one authorized key output and no duplicate secret or content artifact."""
        assert not key_output_marker_path(key_output).exists()
        assert tuple(key_output.parent.glob(f".{key_output.name}.*.reserve")) == ()
        forbidden = raw_key.encode()
        entries = tuple(key_output_directory.iterdir())
        assert entries == (key_output,)
        assert all(
            forbidden not in path.read_bytes()
            for path in entries
            if path != key_output and path.is_file() and not path.is_symlink()
        )

    assert_key_and_canaries_confined()

    gateway_process, gateway_port, base_url = start_gateway(gateway_root)
    gateway_stdout_parts: list[str] = []
    gateway_stderr_parts: list[str] = []
    usage_json_body = ""
    usage_html_body = ""
    error_bodies: list[str] = []
    try:
        assert openai.__version__ == "3.0.0"
        default_refusal = httpx.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {raw_key}"},
            json={
                "model": "coding",
                "messages": [{"role": "user", "content": refusal_input_canary}],
            },
            timeout=10,
        )
        default_refusal.raise_for_status()
        assert default_refusal.json()["choices"][0]["message"]["refusal"] == "policy refusal"
        assert state.count_containing("gateway-secondary-model") == 0

        auth_attempt_ids = {
            str(row["attempt_id"]) for row in gateway_attempt_rows(gateway_database)
        }
        auth_secondary = state.count_containing("gateway-secondary-model")
        auth_fallback = httpx.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {raw_key}"},
            json={
                "model": "coding",
                "messages": [{"role": "user", "content": auth_input_canary}],
            },
            timeout=10,
        )
        auth_fallback.raise_for_status()
        assert auth_fallback.json()["choices"][0]["message"]["content"] == response_canary
        assert state.count_containing("gateway-secondary-model") == auth_secondary + 1
        auth_attempts = tuple(
            row
            for row in gateway_attempt_rows(gateway_database)
            if str(row["attempt_id"]) not in auth_attempt_ids
        )
        assert len(auth_attempts) == 2
        assert len({row["request_id"] for row in auth_attempts}) == 1
        assert [row["attempt_ordinal"] for row in auth_attempts] == [0, 1]
        assert [row["route_depth"] for row in auth_attempts] == [0, 1]
        assert [row["billing_source"] for row in auth_attempts] == [
            "host_managed",
            "customer_managed",
        ]
        assert [row["state"] for row in auth_attempts] == ["failed", "completed"]
        assert auth_attempts[0]["failure_class"] == "provider_authentication"
        assert auth_attempts[0]["estimated_cost_micro_usd"] is None
        assert auth_attempts[1]["input_tokens"] == 3
        assert auth_attempts[1]["output_tokens"] == 2
        assert auth_attempts[1]["estimated_cost_micro_usd"] == 7

        with OpenAI(api_key=raw_key, base_url=base_url, timeout=10) as client:
            assert [model.id for model in client.models.list().data] == ["coding"]
            chat = client.chat.completions.create(
                model="coding",
                messages=[{"role": "user", "content": prompt_canary}],
            )
            assert chat.choices[0].message.content == response_canary
            chat_chunks = list(
                client.chat.completions.create(
                    model="coding",
                    messages=[{"role": "user", "content": prompt_canary}],
                    stream=True,
                )
            )
            assert (
                "".join(
                    chunk.choices[0].delta.content or "" for chunk in chat_chunks if chunk.choices
                )
                == response_canary
            )
            response = client.responses.create(model="coding", input=prompt_canary)
            assert response.output_text == response_canary
            response_events = list(
                client.responses.create(model="coding", input=prompt_canary, stream=True)
            )
            assert response_events[-1].type == "response.completed"
            tool_chat = client.chat.completions.create(
                model="coding",
                messages=[{"role": "user", "content": prompt_canary}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup_ticket",
                            "description": "Look up one ticket.",
                            "parameters": {
                                "type": "object",
                                "properties": {"ticket": {"type": "string"}},
                                "required": ["ticket"],
                                "additionalProperties": False,
                            },
                        },
                    }
                ],
            )
            assert tool_chat.choices[0].message.tool_calls
            tool_call = tool_chat.choices[0].message.tool_calls[0]
            assert tool_call.type == "function"
            assert tool_call.function.arguments == json.dumps(
                {"ticket": tool_argument_canary}, separators=(",", ":")
            )

        async def exercise_async_gateway() -> None:
            """Run all async SDK stream and non-stream gateway quadrants."""
            async with AsyncOpenAI(api_key=raw_key, base_url=base_url, timeout=10) as client:
                chat = await client.chat.completions.create(
                    model="coding",
                    messages=[{"role": "user", "content": prompt_canary}],
                )
                assert chat.choices[0].message.content == response_canary
                chat_stream = await client.chat.completions.create(
                    model="coding",
                    messages=[{"role": "user", "content": prompt_canary}],
                    stream=True,
                )
                chat_chunks = [chunk async for chunk in chat_stream]
                assert (
                    "".join(
                        chunk.choices[0].delta.content or ""
                        for chunk in chat_chunks
                        if chunk.choices
                    )
                    == response_canary
                )
                response = await client.responses.create(model="coding", input=prompt_canary)
                assert response.output_text == response_canary
                response_stream = await client.responses.create(
                    model="coding",
                    input=prompt_canary,
                    stream=True,
                )
                response_events = [event async for event in response_stream]
                assert response_events[-1].type == "response.completed"

        asyncio.run(exercise_async_gateway())

        primary_requests = state.count_containing("gateway-primary-model")
        secondary_requests = state.count_containing("gateway-secondary-model")
        assert primary_requests >= 2
        assert secondary_requests == 10

        matrix_stdout, matrix_stderr = stop_gateway(gateway_process)
        gateway_stdout_parts.append(matrix_stdout)
        gateway_stderr_parts.append(matrix_stderr)
        frozen_before_restart = frozen_billing_evidence(gateway_attempt_rows(gateway_database))
        gateway_process, gateway_port, base_url = start_gateway(gateway_root)
        assert frozen_billing_evidence(gateway_attempt_rows(gateway_database)) == (
            frozen_before_restart
        )

        usage_before_retry = httpx.get(
            f"http://127.0.0.1:{gateway_port}/usage.json",
            timeout=5,
        ).json()["totals"]
        provider_before_retry = state.count_containing(sdk_retry_input_canary)
        try:
            with OpenAI(api_key=raw_key, base_url=base_url, timeout=10) as client:
                client.chat.completions.create(
                    model="coding",
                    messages=[{"role": "user", "content": sdk_retry_input_canary}],
                )
        except openai.APIStatusError as exc:
            assert exc.status_code in {502, 503}
            error_bodies.append(exc.response.text)
        else:
            raise AssertionError("default SDK retry scenario unexpectedly succeeded")
        usage_after_retry = httpx.get(
            f"http://127.0.0.1:{gateway_port}/usage.json",
            timeout=5,
        ).json()["totals"]
        provider_retry_calls = (
            state.count_containing(sdk_retry_input_canary) - provider_before_retry
        )
        assert usage_after_retry["requests"] - usage_before_retry["requests"] == 3
        assert usage_after_retry["attempts"] - usage_before_retry["attempts"] == 4
        assert provider_retry_calls == 4

        retry_stdout, retry_stderr = stop_gateway(gateway_process)
        gateway_stdout_parts.append(retry_stdout)
        gateway_stderr_parts.append(retry_stderr)
        gateway_process, gateway_port, base_url = start_gateway(gateway_root)

        postcommit_secondary = state.count_containing("gateway-secondary-model")
        with httpx.stream(
            "POST",
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {raw_key}"},
            json={
                "model": "coding",
                "messages": [{"role": "user", "content": postcommit_input_canary}],
                "stream": True,
            },
            timeout=10,
        ) as postcommit:
            assert postcommit.status_code == 200
            postcommit_body = "".join(postcommit.iter_text())
        assert postcommit_response_canary in postcommit_body
        assert state.count_containing("gateway-secondary-model") == postcommit_secondary

        postcommit_stdout, postcommit_stderr = stop_gateway(gateway_process)
        gateway_stdout_parts.append(postcommit_stdout)
        gateway_stderr_parts.append(postcommit_stderr)
        gateway_process, gateway_port, base_url = start_gateway(gateway_root)

        keyed_headers = {
            "Authorization": f"Bearer {raw_key}",
            "Idempotency-Key": "installed-restart-operation",
        }
        keyed_request = {
            "model": "coding",
            "messages": [{"role": "user", "content": prompt_canary}],
        }
        keyed_first = httpx.post(
            f"{base_url}/chat/completions",
            headers=keyed_headers,
            json=keyed_request,
            timeout=10,
        )
        keyed_first.raise_for_status()
        physical_before_replay = len(state.snapshot())
        keyed_replay = httpx.post(
            f"{base_url}/chat/completions",
            headers=keyed_headers,
            json=keyed_request,
            timeout=10,
        )
        keyed_replay.raise_for_status()
        assert keyed_replay.content == keyed_first.content
        assert len(state.snapshot()) == physical_before_replay

        with sqlite3.connect(gateway_database) as connection:
            assert connection.execute("PRAGMA journal_mode").fetchone() == ("wal",)
        assert gateway_database.with_name("gateway.db-wal").exists()

        invalid_request = httpx.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {raw_key}"},
            json={
                "model": "coding",
                "messages": [{"role": "user", "content": prompt_canary}],
                "n": 2,
            },
            timeout=5,
        )
        assert invalid_request.status_code == 400
        error_bodies.append(invalid_request.text)

        first_stdout, first_stderr = stop_gateway(gateway_process)
        gateway_stdout_parts.append(first_stdout)
        gateway_stderr_parts.append(first_stderr)
        gateway_process, gateway_port, base_url = start_gateway(gateway_root)
        provider_before_restart_replay = len(state.snapshot())
        attempts_before_restart_replay = gateway_attempt_rows(gateway_database)
        restart_replay = httpx.post(
            f"{base_url}/chat/completions",
            headers=keyed_headers,
            json=keyed_request,
            timeout=10,
        )
        assert restart_replay.status_code == 409
        assert restart_replay.json()["error"]["code"] == "idempotency_replay_unavailable"
        assert len(state.snapshot()) == provider_before_restart_replay
        assert gateway_attempt_rows(gateway_database) == attempts_before_restart_replay
        error_bodies.append(restart_replay.text)

        cancellation_secondary = state.count_containing("gateway-secondary-model")
        with httpx.stream(
            "POST",
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {raw_key}"},
            json={
                "model": "coding",
                "messages": [{"role": "user", "content": cancellation_input_canary}],
                "stream": True,
            },
            timeout=10,
        ) as cancellation:
            assert cancellation.status_code == 200
            for line in cancellation.iter_lines():
                if cancellation_response_canary in line:
                    break
            else:
                raise AssertionError("gateway cancellation stream produced no semantic output")
        assert state.count_containing("gateway-secondary-model") == cancellation_secondary

        second_stdout, second_stderr = stop_gateway(gateway_process)
        gateway_stdout_parts.append(second_stdout)
        gateway_stderr_parts.append(second_stderr)
        opted_policy = run_cli(
            "config",
            "gateway",
            "pool",
            "certify",
            "coding",
            "--deployment-alias",
            "primary",
            "--deployment-alias",
            "secondary",
            "--exact-model",
            "gateway-model-revision",
            "--certification-id",
            "coding-refusal-certification",
            "--provenance",
            "installed-wheel deterministic loopback refusal policy",
            "--evidence-sha256",
            "b" * 64,
            "--certified-at",
            "2026-08-19T00:00:01Z",
            "--expected-catalog-sha256",
            catalog_sha256,
            "--revision",
            "installed-refusal-revision",
            "--replace",
            "--refusal-failover",
            "--root",
            str(gateway_root),
            "--non-interactive",
            "--json",
        )
        opted_catalog_sha256 = json.loads(opted_policy.stdout)["data"]["catalog_sha256"]
        assert opted_catalog_sha256 != catalog_sha256
        billing_after_pool_replace = frozen_billing_evidence(gateway_attempt_rows(gateway_database))
        assert {
            attempt_id: billing_after_pool_replace[attempt_id]
            for attempt_id in frozen_before_restart
        } == frozen_before_restart
        gateway_process, gateway_port, base_url = start_gateway(gateway_root)
        opted_secondary = state.count_containing("gateway-secondary-model")
        opted_refusal = httpx.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {raw_key}"},
            json={
                "model": "coding",
                "messages": [{"role": "user", "content": refusal_input_canary}],
            },
            timeout=10,
        )
        opted_refusal.raise_for_status()
        assert opted_refusal.json()["choices"][0]["message"]["content"] == response_canary
        assert state.count_containing("gateway-secondary-model") == opted_secondary + 1

        revoked = run_cli(
            "config",
            "gateway",
            "key",
            "revoke",
            "installed-key",
            "--root",
            str(gateway_root),
            "--non-interactive",
            "--json",
        )
        assert json.loads(revoked.stdout)["changed"] is True
        provider_before_revoke = len(state.snapshot())
        revoked_auth = httpx.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {raw_key}"},
            timeout=5,
        )
        assert revoked_auth.status_code == 401
        assert len(state.snapshot()) == provider_before_revoke
        error_bodies.append(revoked_auth.text)

        usage_json_response = httpx.get(f"http://127.0.0.1:{gateway_port}/usage.json", timeout=5)
        usage_html_response = httpx.get(f"http://127.0.0.1:{gateway_port}/usage", timeout=5)
        usage_json_response.raise_for_status()
        usage_html_response.raise_for_status()
        usage_json_body = usage_json_response.text
        usage_html_body = usage_html_response.text
        usage_payload = usage_json_response.json()
        assert usage_payload["schema_version"] == 2
        identity_usage = usage_payload["identities"][0]
        assert usage_payload["totals"]["requests"] >= 14
        assert usage_payload["totals"]["attempts"] > usage_payload["totals"]["requests"]
        assert usage_payload["totals"]["known_estimated_cost_micro_usd"] > 0
        terminal_counts = {
            item["state"]: item["attempts"] for item in usage_payload["totals"]["terminal_counts"]
        }
        assert terminal_counts["completed"] >= 10
        assert terminal_counts["failed"] >= 2
        assert terminal_counts["cancelled"] >= 1
        attempt_rows = gateway_attempt_rows(gateway_database)
        billing_sources = {str(row["billing_source"]) for row in attempt_rows}
        assert billing_sources == {"customer_managed", "host_managed"}
        assert (
            sum(cast(int | None, row["input_tokens"]) or 0 for row in attempt_rows)
            == (usage_payload["totals"]["input_tokens"])
        )
        assert (
            sum(cast(int | None, row["cached_input_tokens"]) or 0 for row in attempt_rows)
            == (usage_payload["totals"]["cached_input_tokens"])
        )
        assert (
            sum(cast(int | None, row["output_tokens"]) or 0 for row in attempt_rows)
            == (usage_payload["totals"]["output_tokens"])
        )
        assert (
            sum(cast(int | None, row["reasoning_tokens"]) or 0 for row in attempt_rows)
            == (usage_payload["totals"]["reasoning_tokens"])
        )
        assert (
            sum(cast(int | None, row["estimated_cost_micro_usd"]) or 0 for row in attempt_rows)
            == (usage_payload["totals"]["known_estimated_cost_micro_usd"])
        )
        assert (
            sum(row["estimated_cost_micro_usd"] is None for row in attempt_rows)
            == (usage_payload["totals"]["unknown_cost_attempts"])
        )
        source_buckets = usage_payload["by_billing_source"]
        assert [bucket["billing_source"] for bucket in source_buckets] == [
            "customer_managed",
            "host_managed",
        ]
        for bucket in source_buckets:
            billing_source = bucket["billing_source"]
            source_rows = tuple(
                row for row in attempt_rows if row["billing_source"] == billing_source
            )
            assert source_rows
            assert all(row["state"] != "dispatched" for row in source_rows)
            assert bucket["attempts"] == len(source_rows)
            assert bucket["input_tokens"] == sum(
                cast(int | None, row["input_tokens"]) or 0 for row in source_rows
            )
            assert bucket["cached_input_tokens"] == sum(
                cast(int | None, row["cached_input_tokens"]) or 0 for row in source_rows
            )
            assert bucket["output_tokens"] == sum(
                cast(int | None, row["output_tokens"]) or 0 for row in source_rows
            )
            assert bucket["reasoning_tokens"] == sum(
                cast(int | None, row["reasoning_tokens"]) or 0 for row in source_rows
            )
            assert bucket["known_estimated_cost_micro_usd"] == sum(
                cast(int | None, row["estimated_cost_micro_usd"]) or 0 for row in source_rows
            )
            assert bucket["unknown_cost_attempts"] == sum(
                row["estimated_cost_micro_usd"] is None for row in source_rows
            )
            expected_source_terminals = {
                state: sum(row["state"] == state for row in source_rows)
                for state in sorted({str(row["state"]) for row in source_rows})
            }
            assert {
                item["state"]: item["attempts"] for item in bucket["terminal_counts"]
            } == expected_source_terminals
        expected_cells = (
            identity_usage["identity_id"],
            identity_usage["requests"],
            identity_usage["attempts"],
            identity_usage["input_tokens"],
            identity_usage["cached_input_tokens"],
            identity_usage["output_tokens"],
            identity_usage["reasoning_tokens"],
            identity_usage["known_estimated_cost_micro_usd"],
            identity_usage["unknown_cost_attempts"],
            identity_usage["total_latency_ms"],
            ", ".join(
                f"{item['state']}: {item['attempts']}" for item in identity_usage["terminal_counts"]
            ),
        )
        expected_source_cells = tuple(
            value
            for bucket in source_buckets
            for value in (
                bucket["billing_source"],
                bucket["attempts"],
                bucket["input_tokens"],
                bucket["cached_input_tokens"],
                bucket["output_tokens"],
                bucket["reasoning_tokens"],
                bucket["known_estimated_cost_micro_usd"],
                bucket["unknown_cost_attempts"],
                ", ".join(
                    f"{item['state']}: {item['attempts']}" for item in bucket["terminal_counts"]
                ),
            )
        )
        assert tuple(re.findall(r"<td>(.*?)</td>", usage_html_body)) == tuple(
            str(value) for value in (*expected_cells, *expected_source_cells)
        )

        invalid_auth = httpx.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {invalid_key_canary}"},
            timeout=5,
        )
        assert invalid_auth.status_code == 401
        error_bodies.append(invalid_auth.text)
        _assert_gateway_canaries_absent(
            gateway_root,
            channels={
                "cli": "".join(cli_transcripts),
                "http_errors": "".join(error_bodies),
                "usage_html": usage_html_body,
                "usage_json": usage_json_body,
            },
            canaries=(
                raw_key,
                provider_secret,
                prompt_canary,
                response_canary,
                tool_argument_canary,
                invalid_key_canary,
                refusal_input_canary,
                auth_input_canary,
                sdk_retry_input_canary,
                postcommit_input_canary,
                postcommit_response_canary,
                cancellation_input_canary,
                cancellation_response_canary,
            ),
        )
    finally:
        gateway_stdout, gateway_stderr = stop_gateway(gateway_process)
        gateway_stdout_parts.append(gateway_stdout)
        gateway_stderr_parts.append(gateway_stderr)

    project_root = execution_root / "gateway-project"
    project_manager = GatewayManagement(project_root)
    project_manager.initialize()
    for alias in ("cheap", "baseline"):
        upsert_connection(
            project_root,
            name=f"project-{alias}",
            connection=ConnectionConfig(
                provider="openai-compatible",
                base_url=provider_url,
                api_key_env="P9_LOOPBACK_PROVIDER_KEY",
            ),
            replace=False,
        )
    project_catalog = None
    for alias, provider_model, billing_source in (
        ("cheap", "project-primary-model", BillingSource.HOST_MANAGED),
        ("baseline", "project-secondary-model", BillingSource.CUSTOMER_MANAGED),
    ):
        project_catalog, _snapshot, _changed = upsert_singleton_deployment(
            project_root,
            deployment_alias=alias,
            connection_name=f"project-{alias}",
            provider_model=provider_model,
            exact_model_id="project-exact-model",
            revision=None,
            capabilities=ModelCapabilities(),
            gateway_capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
            prices=GatewayTokenPrices(
                input_micro_usd_per_million_tokens=1_000_000,
                output_micro_usd_per_million_tokens=2_000_000,
            ),
            pricing_source="installed project fixture",
            billing_source=billing_source,
            replace=False,
        )
    assert project_catalog is not None
    project_catalog, project_snapshot, _changed = upsert_certified_pool(
        project_root,
        pool_id="project-pool",
        exact_model_id="project-exact-model",
        deployment_aliases=("cheap", "baseline"),
        certification=GatewayEquivalenceCertification(
            certification_id="installed-project-certification",
            provenance="installed deterministic exact-model fixture",
            evidence_sha256="c" * 64,
            certified_at=datetime(2026, 8, 19, tzinfo=UTC),
        ),
        expected_catalog_sha256=project_catalog.identity_sha256(),
        replace=False,
    )
    project_manager.activate_project_alias(
        alias_id="project-coding",
        alias_name="project-coding",
        revision_id="installed-project-revision",
        project_ref="installed-project",
        activation_ref="installed-project-policy",
        snapshot_ref=f"catalog-snapshots/{project_snapshot.name}",
        catalog_sha256=project_catalog.identity_sha256(),
    )
    project_manager.create_identity(identity_id="project-identity", display_name="Project")
    project_manager.add_grant(identity_id="project-identity", alias_id="project-coding")
    project_key = project_manager.issue_key(
        identity_id="project-identity",
        key_id="project-key",
    ).raw_key
    learned_runtime, learned_client = installed_project_runtime()

    def load_installed_project(
        project: str,
        root: Path,
        *,
        policy_id: str,
        runtime_catalog: RuntimeModelCatalog,
    ) -> RouterRuntime:
        """Inject one real installed selection runtime without project completion."""
        del root, runtime_catalog
        assert project == "installed-project"
        assert policy_id == "installed-project-policy"
        return learned_runtime

    project_provider_before_load = state.snapshot()
    project_primary_before = state.count_containing("project-primary-model")
    project_secondary_before = state.count_containing("project-secondary-model")
    installed_project_gateway = load_local_gateway(
        project_root,
        graceful_timeout_seconds=2,
        environment={"P9_LOOPBACK_PROVIDER_KEY": provider_secret},
        project_loader=load_installed_project,
    )
    assert state.snapshot() == project_provider_before_load
    project_port = unused_loopback_port()
    project_server = uvicorn.Server(
        uvicorn.Config(
            installed_project_gateway.app,
            host="127.0.0.1",
            port=project_port,
            log_level="critical",
            access_log=False,
        )
    )
    project_thread = threading.Thread(target=project_server.run, daemon=True)
    project_thread.start()
    project_deadline = time.monotonic() + 20
    while not project_server.started and time.monotonic() < project_deadline:
        time.sleep(0.01)
    assert project_server.started
    assert state.snapshot() == project_provider_before_load
    try:
        with OpenAI(
            api_key=project_key,
            base_url=f"http://127.0.0.1:{project_port}/v1",
            timeout=10,
        ) as client:
            project_response = client.chat.completions.create(
                model="project-coding",
                messages=[{"role": "user", "content": project_prompt_canary}],
            )
        assert project_response.choices[0].message.content == project_response_canary
    finally:
        project_server.should_exit = True
        project_thread.join(timeout=10)
    assert not project_thread.is_alive()
    assert state.count_containing("project-primary-model") == project_primary_before + 2
    assert state.count_containing("project-secondary-model") == project_secondary_before + 1
    assert learned_client.embed_calls == 1
    assert learned_client.complete_calls == 0
    assert learned_runtime.records_decisions is False
    assert not (project_root / "projects").exists()
    with sqlite3.connect(project_manager.database_path) as connection:
        project_attempts = connection.execute(
            """
            SELECT attempt_ordinal, route_depth, exact_model_id, billing_source, state
            FROM gateway_attempts ORDER BY attempt_ordinal
            """
        ).fetchall()
    assert project_attempts == [
        (0, 0, "project-exact-model", "host_managed", "failed"),
        (1, 0, "project-exact-model", "host_managed", "failed"),
        (2, 1, "project-exact-model", "customer_managed", "completed"),
    ]
    _assert_gateway_canaries_absent(
        project_root,
        channels={},
        canaries=(
            project_key,
            provider_secret,
            project_prompt_canary,
            project_response_canary,
        ),
    )
    gateway_stdout = "".join(gateway_stdout_parts)
    gateway_stderr = "".join(gateway_stderr_parts)
    assert (gateway_root / "gateway" / "gateway.db").is_file()
    assert tuple((gateway_root / "gateway" / "catalog-snapshots").glob("*.json"))
    _assert_gateway_canaries_absent(
        gateway_root,
        channels={
            "cli": "".join(cli_transcripts),
            "http_errors": "".join(error_bodies),
            "stderr": gateway_stderr,
            "stdout": gateway_stdout,
            "usage_html": usage_html_body,
            "usage_json": usage_json_body,
        },
        canaries=(
            raw_key,
            provider_secret,
            prompt_canary,
            response_canary,
            tool_argument_canary,
            invalid_key_canary,
            refusal_input_canary,
            auth_input_canary,
            sdk_retry_input_canary,
            postcommit_input_canary,
            postcommit_response_canary,
            cancellation_input_canary,
            cancellation_response_canary,
        ),
    )

    def assert_embedded_revision(value: object) -> None:
        """Require every recursively present code revision to match the installed release.

        Args:
            value: Decoded artifact JSON value to inspect recursively.
        """
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "code_revision":
                    assert item == release_revision
                assert_embedded_revision(item)
        elif isinstance(value, list):
            for item in value:
                assert_embedded_revision(item)

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

    down = "\x1b[B"
    enter = "\r"
    space = " "
    setup_answers = [
        (
            "Select the providers you want to use",
            (down * 5) + enter + (down * 2) + enter,
        ),
        ("Azure OpenAI base URL", azure_endpoint),
        ("Azure OpenAI API version", ""),
        ("Select the models to configure", space + down + enter),
        ("Connection for the declared model", enter),
        ("Provider model ID", "core-model"),
        ("Supports chat completions?", "y"),
        ("Supports embeddings?", "y"),
        ("Supports tools?", "y"),
        ("Supports structured output?", "y"),
        ("Record context window tokens?", "y"),
        ("Context window tokens", "128000"),
        ("Record maximum output tokens?", "y"),
        ("Maximum output tokens", "32000"),
        ("Input cost per million tokens in USD", "0"),
        ("Output cost per million tokens in USD", "0"),
        ("Cached input cost per million tokens in USD", "0"),
        ("Cache write cost per million tokens in USD", "0"),
        ("Reasoning effort", enter),
        ("/core-model)", (down * 2) + enter),
        ("World model", enter),
        ("Judge model", enter),
        ("Embedder model", enter),
        ("Save this configuration?", "y"),
    ]
    try:
        assert not root.exists()
        build_output = run_tty(
            ["build", "support-agent", str(traces), "--root", str(root)],
            setup_answers,
        )
        assert "Model setup is required" in build_output
        assert "Candidate aliases" not in build_output
        assert "Complete" in build_output
        support_store = ProjectStore(root, "support-agent")
        support_project = support_store.load_project()
        assert support_project.build is not None
        assert support_project.models is not None
        assert support_project.models.candidates == ()
        catalog_text = (root / "models.toml").read_text(encoding="utf-8")
        assert "deterministic-loopback-placeholder" not in catalog_text
        assert "AZURE_OPENAI_API_KEY" in catalog_text
        build_counts = state.counts()
        assert build_counts[provider_embeddings_path] > 0
        assert build_counts[provider_chat_path] == 0
        world_model = wmo.load_world_model(
            "support-agent",
            root=root,
            environment={"AZURE_OPENAI_API_KEY": "deterministic-loopback-placeholder"},
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
            provider_embeddings_path,
            provider_chat_path,
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
        calibration_plan = prepare_manual_judge_calibration(support_store)
        assert len(calibration_plan.previews) == 5
        assert len(calibration_plan.previews) >= 2
        calibration_arguments = [
            "config",
            "judge",
            "calibrate",
            "support-agent",
            "--root",
            str(root),
        ]
        for preview in calibration_plan.previews:
            calibration_arguments.extend(["--label", f"{preview.trace_id}:task-success=1"])
        calibration_arguments.extend(
            [
                "--maximum-cost-usd",
                "0.000001",
                "--yes",
                "--approve",
                "--non-interactive",
            ]
        )
        calibration_result = run_cli(*calibration_arguments)
        assert "Approved judge calibration" in calibration_result.stdout
        assert "Trace 1 of 5" in calibration_result.stdout
        assert "Original user request:" in calibration_result.stdout
        assert "Configured judge proposals" in calibration_result.stdout
        assert "Description: The agent successfully completed" in calibration_result.stdout
        assert "Numeric range: 0 to 1" in calibration_result.stdout
        assert "Proposed score: 1" in calibration_result.stdout
        assert "Proposed judgment:" in calibration_result.stdout
        assert sum(state.counts().values()) - setup_call_count == len(calibration_plan.previews)
        review = support_store.read_review()
        assert isinstance(review, dict)
        manual_judge = review.get("manual_judge")
        assert isinstance(manual_judge, dict)
        assert manual_judge.get("approved_calibration") is not None
        trace_review_pointers = manual_judge.get("trace_reviews")
        assert isinstance(trace_review_pointers, list)
        assert len(trace_review_pointers) == 5
        for pointer in trace_review_pointers:
            assert isinstance(pointer, dict)
            review_id = pointer.get("artifact_id")
            assert isinstance(review_id, str)
            trace_review = ManualJudgeTraceReviewArtifact.model_validate_json(
                support_store.artifacts.read_bytes(review_id, "review.json")
            )
            assert trace_review.provenance.proposal_author == "configured_judge"
            assert trace_review.provenance.decision_author == "human"
            assert trace_review.axes[0].human_correction is None
            assert trace_review.axes[0].final_accepted_label.score_source == "configured_judge"
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
            "--maximum-model-calls",
            "1",
            "--simulation-maximum-output-tokens",
            "8000",
            "--maximum-concurrency",
            "1",
            "--candidate",
            "candidate-b",
            "--candidate",
            "core-model",
            "--incumbent",
            "core-model",
            "--candidate-model",
            json.dumps(
                {
                    "alias": "candidate-b",
                    "connection": "azure",
                    "model": "candidate-b-model",
                    "capabilities": {
                        "supports_completions": True,
                        "supports_tools": True,
                        "context_window_tokens": 128_000,
                        "maximum_output_tokens": 32_000,
                        "input_cost_per_million_tokens_usd": 0,
                        "output_cost_per_million_tokens_usd": 0,
                        "cached_input_cost_per_million_tokens_usd": 0,
                        "cache_write_cost_per_million_tokens_usd": 0,
                    },
                }
            ),
            "--yes",
        ]
        optimization_output = run_tty(
            optimize_arguments,
            [("Save these router candidates?", "y")],
        )
        assert "candidates: candidate-b, core-model" in optimization_output
        assert "incumbent: core-model" in optimization_output
        assert "policy:" in optimization_output
        assert "report:" in optimization_output
        optimized_artifacts = directory_digest(support_store.paths.artifacts_directory)
        optimized_catalog = (root / "models.toml").read_bytes()
        optimized_project = support_store.paths.project_toml.read_bytes()
        optimized_review = support_store.paths.review_json.read_bytes()
        optimized_provider_requests = state.snapshot()
        replay_result = run_cli(
            *optimize_arguments[:-1],
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
            compatibility_key_directory = root / "gateway" / "compatibility-keys"
            compatibility_key_files = tuple(compatibility_key_directory.glob("*.txt"))
            assert len(compatibility_key_files) == 1
            compatibility_key_file = compatibility_key_files[0]
            compatibility_key = compatibility_key_file.read_text(encoding="utf-8").strip()
            assert compatibility_key.startswith("wmo_vk_")
            assert stat.S_IMODE(compatibility_key_file.stat().st_mode) == 0o600
            assert not key_output_marker_path(compatibility_key_file).exists()
            assert (
                tuple(compatibility_key_directory.glob(f".{compatibility_key_file.name}.*.reserve"))
                == ()
            )
            client = OpenAI(
                base_url=f"http://127.0.0.1:{router_port}/v1",
                api_key=compatibility_key,
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
            assert first_response_counts[provider_embeddings_path] == (
                response_calls_before[provider_embeddings_path] + 1
            )
            assert (
                second_response_counts[provider_embeddings_path]
                == first_response_counts[provider_embeddings_path]
            )
            keyed_chat = client.chat.completions.create(
                model="support-agent",
                messages=[{"role": "user", "content": "Durable keyed request"}],
                extra_headers={"Idempotency-Key": "p17-durable-request"},
            )
            assert keyed_chat.id
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
        gateway_database = root / "gateway" / "gateway.db"
        artifacts_before_public_replay = directory_digest(support_store.paths.artifacts_directory)
        project_before_public_replay = support_store.paths.project_toml.read_bytes()
        with sqlite3.connect(gateway_database) as connection:
            gateway_requests_before_public_replay = connection.execute(
                "SELECT COUNT(*) FROM gateway_requests"
            ).fetchone()[0]
            gateway_attempts_before_public_replay = connection.execute(
                "SELECT COUNT(*) FROM gateway_attempts"
            ).fetchone()[0]
        with wmo.load_router(
            "support-agent",
            root=root,
            environment={"AZURE_OPENAI_API_KEY": "deterministic-loopback-placeholder"},
        ) as loaded_router:
            try:
                loaded_router.chat.completions.create(
                    model="support-agent",
                    messages=[{"role": "user", "content": "Durable keyed request"}],
                    extra_headers={"Idempotency-Key": "p17-durable-request"},
                )
            except openai.ConflictError as error:
                assert error.status_code == 409
                assert "idempotency_replay_unavailable" in str(error)
            else:
                raise AssertionError("restarted project gateway replay must fail closed")
            assert state.snapshot() == provider_before_public_replay
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
        assert journal.read_events() == events_before_public_replay
        assert state.snapshot() != provider_before_public_replay
        assert support_store.paths.project_toml.read_bytes() == project_before_public_replay
        assert directory_digest(support_store.paths.artifacts_directory) == (
            artifacts_before_public_replay
        )
        with sqlite3.connect(gateway_database) as connection:
            gateway_requests_after_public_replay = connection.execute(
                "SELECT COUNT(*) FROM gateway_requests"
            ).fetchone()[0]
            gateway_attempts_after_public_replay = connection.execute(
                "SELECT COUNT(*) FROM gateway_attempts"
            ).fetchone()[0]
        assert gateway_requests_after_public_replay == gateway_requests_before_public_replay + 3
        assert gateway_attempts_after_public_replay == gateway_attempts_before_public_replay + 3
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
                assert manifest.code_revision == release_revision
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
        assert_key_and_canaries_confined()
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


def test_gateway_canary_scanner_covers_every_persistent_and_observable_channel(
    tmp_path: Path,
) -> None:
    """Every security-sensitive gateway artifact class reaches one scanner.

    Args:
        tmp_path: Pytest-owned gateway certification root.
    """
    cases = (
        ("gateway.db", None),
        ("gateway.db-wal", None),
        ("gateway.db.backup-v3-test", None),
        ("catalog-snapshots/snapshot.json", None),
        (None, "stdout"),
        (None, "stderr"),
        (None, "log"),
        (None, "http_error"),
        (None, "usage_html"),
        (None, "usage_json"),
    )
    canary = "release-scanner-secret-content-canary"
    for index, (relative_path, channel) in enumerate(cases):
        case_root = tmp_path / f"case-{index}"
        channels: dict[str, bytes | str] = {}
        if relative_path is not None:
            path = case_root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(canary, encoding="utf-8")
        if channel is not None:
            channels[channel] = canary

        with pytest.raises(AssertionError, match="forbidden gateway canary 0"):
            _assert_gateway_canaries_absent(
                case_root,
                channels=channels,
                canaries=(canary,),
            )


def test_package_workflow_installs_the_exact_certified_openai_sdk() -> None:
    """The archive smoke lane constrains the SDK version claimed by certification."""
    repository = Path(__file__).resolve().parent.parent
    workflow = (repository / ".github" / "workflows" / "python-package.yml").read_text(
        encoding="utf-8"
    )

    assert 'dist/*.whl "openai==3.0.0"' in workflow


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
    _run_checked(
        [uv, "venv", "--python", sys.executable, str(virtual_environment)],
        cwd=execution,
        environment=environment,
    )
    installed_python = virtual_environment / "bin" / "python"
    _run_checked(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(installed_python),
            str(wheel[0]),
            "openai==3.0.0",
        ],
        cwd=execution,
        environment=environment,
    )
    driver = execution / "installed_release_evidence.py"
    driver.write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    driver_environment = environment.copy()
    driver_environment.update(
        {
            "WMO_INSTALLED_RELEASE_EVIDENCE": "1",
            "WMO_RELEASE_REVISION": _TEST_RELEASE_REVISION,
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
    from wmo.common.judging import HumanScoreReview, JudgeCalibrationService, RubricReview
    from wmo.runtime.environments import LocalProcessEnvironmentRuntime
    from wmo.simulation import compare_text_and_sandbox
    from wmo.simulation.engines import SandboxSimulator

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
    assert "wmo config gateway" in usage
    assert "wmo config gateway pool certify" in usage
    assert "wmo run --root ROOT" in usage
    assert "OpenAI `3.0.0`" in usage
    assert "schema-v2" in usage
    assert "by_billing_source" in usage
    assert "wmo optimize route" not in usage.replace("wmo optimize router", "")

    scope = (docs / "release-scope.md").read_text(encoding="utf-8")
    assert "local gateway" in scope
    assert "Gateway provider evidence matrix" in scope
    assert "not_run_requires_credentials" in scope
    for exclusion in (
        "No paid E2B or Harbor cloud smoke ran",
        "No real Tinker training ran",
        "No trained-versus-base behavioral comparison ran",
        "exactly $0.00 observed service spend",
    ):
        assert exclusion in scope

    architecture = (docs / "reference" / "gateway-architecture.md").read_text(encoding="utf-8")
    assert "GET /v1/models" in architecture
    assert "POST /v1/chat/completions" in architecture
    assert "POST /v1/responses" in architecture
    assert "provider_certification.py" in architecture
    assert "schema-v2" in architecture
    assert "by_billing_source" in architecture
    assert "inert contracts" not in architecture
    assert "does not claim that a gateway server" not in architecture

    ingest = (docs / "reference" / "ingest.md").read_text(encoding="utf-8")
    assert "PostHogPullRequest" in ingest
    assert "pull_posthog_traces" in ingest


if __name__ == "__main__" and os.environ.get("WMO_INSTALLED_RELEASE_EVIDENCE") == "1":
    _installed_release_driver()
