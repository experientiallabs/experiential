"""End-to-end regression over the pinned public terminal-tasks trace export.

The published quickstart hands `wmo build` one unmodified Hugging Face OTLP export whose
environment-capture spans carry no provider or model identity, then sets up a judge, reviews real
traces, calibrates, and approves. This module walks that whole path with deterministic local
model clients and proves the source stays provider free, that completed trace reviews survive a
provider failure, and that replay and approval make no provider calls or review prompts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path

import pytest
from click import unstyle
from typer.testing import CliRunner

from wmo.cli.app import app
from wmo.common.config.settings import set_maximum_command_cost_usd
from wmo.common.models import (
    AssistantAction,
    ConnectionConfig,
    Embedding,
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
    ModelRequest,
    ModelResponse,
    ModelRoles,
    ModelSnapshot,
    OperationEconomics,
    write_model_catalog,
)
from wmo.common.project import ProjectStore
from wmo.common.rollouts import RolloutArtifact
from wmo.common.traces import load_trace_dataset
from wmo.optimize.router.judging.contracts import ManualJudgeTraceReviewArtifact
from wmo.optimize.router.judging.service import prepare_manual_judge_calibration
from wmo.runtime.models import ResolvedModel
from wmo.runtime.models.registry import RuntimeModelCatalog

TRACES_URL = (
    "https://huggingface.co/datasets/experiential-labs/wmo-terminal-tasks-traces/resolve/"
    "540883e451dc13d34fb50fdd36b143cb0f1fb0db/traces.otel.jsonl"
)
TRACES_SHA256 = "21c62cba7e3372cbf03df051dc2408699fbf8ea3561ba661b599e4949f0e5d42"
_JUDGE_MODEL_ID = "judge-id"
_PROJECT = "terminal-tasks"
_SAMPLE_SIZE = 5


class _EmbeddingClient:
    """Return deterministic unit vectors so build needs no embedding provider."""

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Return one deterministic normalized vector per text."""
        embeddings: list[Embedding] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            raw = tuple(float(value + 1) for value in digest[:8])
            norm = math.sqrt(sum(value * value for value in raw))
            embeddings.append(Embedding(values=tuple(value / norm for value in raw)))
        return tuple(embeddings)


class _JudgeClient:
    """Return one cited scalar score per call, or fail after a chosen number of calls."""

    def __init__(self, model: ModelSnapshot, *, fail_after: int | None = None) -> None:
        """Bind the configured judge identity and an optional interruption point."""
        self.model = model
        self.fail_after = fail_after
        self.calls = 0

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Return a schema-valid score after confirming the request includes a rollout span."""
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("simulated judge provider interruption")
        self.calls += 1
        content = request.messages[1].content or ""
        match = re.search(r'"span_id":\s*"([^"]+)"', content)
        assert match is not None
        return ModelResponse(
            output=AssistantAction(
                content=json.dumps(
                    {
                        "dimensions": [
                            {
                                "dimension_id": "task-success",
                                "raw_score": 1,
                                "rationale": "The trace shows the requested task was handled.",
                            }
                        ]
                    }
                )
            ),
            model=self.model,
            economics=OperationEconomics(),
        )


class _RuntimeCatalog:
    """Resolve configured aliases to deterministic local clients over real static identity."""

    judge_clients: list[_JudgeClient] = []
    judge_fail_after: int | None = None

    def __init__(self, catalog: ModelCatalog) -> None:
        """Wrap the real catalog so every snapshot stays exactly as configured."""
        self._real = RuntimeModelCatalog(catalog)
        self._embedding = _EmbeddingClient()

    def snapshot(self, alias: str) -> tuple[ModelSnapshot, ModelCapabilities]:
        """Return one credential-free static model identity.

        Args:
            alias: Configured local model alias.

        Returns:
            Exact model snapshot and capability declaration.
        """
        return self._real.snapshot(alias)

    def preflight(self, alias: str, requirement: object | None = None) -> ResolvedModel:
        """Return a deterministic resolved model for one configured alias."""
        del requirement
        snapshot, capabilities = self._real.snapshot(alias)
        client = _JudgeClient(snapshot, fail_after=type(self).judge_fail_after)
        type(self).judge_clients.append(client)
        embedding = self._embedding if alias == "embed" else None
        return ResolvedModel(alias, snapshot, capabilities, client, embedding)

    def resolve(self, alias: str) -> ResolvedModel:
        """Resolve one alias with the same deterministic clients as preflight."""
        return self.preflight(alias)


@pytest.fixture(name="pinned_traces")
def _pinned_traces(request: pytest.FixtureRequest) -> Path:
    """Return the cached pinned public export, skipping when it cannot be fetched."""
    cache = Path(str(request.config.cache.mkdir("wmo-public-fixtures")))
    path = cache / "terminal-tasks-traces.otel.jsonl"
    if not path.is_file() or _sha256(path) != TRACES_SHA256:
        headers = {}
        token = os.environ.get("HF_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            fetch = urllib.request.Request(TRACES_URL, headers=headers)
            with urllib.request.urlopen(fetch, timeout=120) as response:
                payload = response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            pytest.skip(f"pinned public terminal-tasks export is unavailable: {exc}")
        path.write_bytes(payload)
    assert _sha256(path) == TRACES_SHA256
    return path


def _sha256(path: Path) -> str:
    """Return the hex digest of one local file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_catalog(root: Path) -> None:
    """Write a secret-free catalog covering every build and judge role."""
    write_model_catalog(
        root / "models.toml",
        ModelCatalog(
            connections={
                "fixture": ConnectionConfig(provider="openai", api_key_env="FIXTURE_API_KEY")
            },
            models={
                "world": ModelRecord(
                    connection="fixture",
                    model="world-id",
                    capabilities=ModelCapabilities(maximum_output_tokens=16_000),
                ),
                "judge": ModelRecord(
                    connection="fixture",
                    model=_JUDGE_MODEL_ID,
                    capabilities=ModelCapabilities(
                        input_cost_per_million_tokens_usd=0,
                        output_cost_per_million_tokens_usd=0,
                    ),
                ),
                "embed": ModelRecord(
                    connection="fixture",
                    model="embed-id",
                    capabilities=ModelCapabilities(
                        supports_embeddings=True,
                        input_cost_per_million_tokens_usd=0,
                    ),
                ),
            },
            roles=ModelRoles(world_model="world", judge="judge", embedder="embed"),
        ),
    )


def _calibrate_arguments(
    root: Path,
    labels: Sequence[str],
    sample_size: int | None = None,
) -> list[str]:
    """Return one noninteractive calibrate invocation using catalog prices.

    Args:
        root: Local test project root.
        labels: Repeatable explicit label arguments.
        sample_size: Optional override; ``None`` exercises the installed default of five.

    Returns:
        Complete nested CLI argument list.
    """
    arguments = [
        "config",
        "judge",
        "calibrate",
        _PROJECT,
        "--root",
        str(root),
        "--yes",
        "--approve",
        "--non-interactive",
        *labels,
    ]
    if sample_size is not None:
        arguments[6:6] = ["--sample-size", str(sample_size)]
    return arguments


def _rollout_payloads(store: ProjectStore) -> tuple[tuple[RolloutArtifact, str], ...]:
    """Return every persisted rollout of the project with its exact stored text.

    Args:
        store: Project whose immutable artifacts are read.

    Returns:
        Parsed rollout and its raw stored JSON text, in artifact order.
    """
    payloads: list[tuple[RolloutArtifact, str]] = []
    for artifact_id in store.artifacts.list_ids():
        record = store.artifacts.read(artifact_id)
        if record.manifest.artifact_type != "rollout":
            continue
        text = (record.directory / "rollout.json").read_text(encoding="utf-8")
        payloads.append((RolloutArtifact.model_validate_json(text), text))
    return tuple(payloads)


def test_public_terminal_tasks_path_stays_provider_free_and_keeps_labels(
    pinned_traces: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Walk build, setup, labeling, calibration, and approval on the pinned public export."""
    monkeypatch.setattr("wmo.cli.build_cmd.RuntimeModelCatalog", _RuntimeCatalog)
    monkeypatch.setattr("wmo.cli.build_cmd.capture_build_completed", lambda **_kwargs: None)
    monkeypatch.setattr("wmo.cli.judge_config.RuntimeModelCatalog", _RuntimeCatalog)
    _RuntimeCatalog.judge_clients = []
    _RuntimeCatalog.judge_fail_after = None
    root = tmp_path / ".wmo"
    root.mkdir()
    _write_catalog(root)
    runner = CliRunner()

    build = runner.invoke(
        app, ["build", _PROJECT, "--traces", str(pinned_traces), "--root", str(root)]
    )
    assert build.exit_code == 0, build.output
    setup = runner.invoke(
        app, ["config", "judge", "setup", _PROJECT, "--root", str(root), "--approve"]
    )
    assert setup.exit_code == 0, setup.output

    store = ProjectStore(root, _PROJECT)
    plan = prepare_manual_judge_calibration(store)
    assert len(plan.traces) == _SAMPLE_SIZE
    assert all(span.model is None for trace in plan.traces for span in trace.spans)
    labels = [
        argument
        for trace in plan.traces
        for argument in ("--label", f"{trace.trace_id}:task-success=1")
    ]

    before_preflight = store.read_review()
    set_maximum_command_cost_usd(0.5, root)
    over_budget = [
        *_calibrate_arguments(root, ()),
        "--input-usd-per-million",
        "1",
        "--output-usd-per-million",
        "1",
    ]
    _RuntimeCatalog.judge_clients = []
    refused = runner.invoke(app, over_budget)
    assert refused.exit_code == 2
    refused_text = " ".join(unstyle(refused.output).replace("│", " ").split())
    assert "Cost preflight wmo config judge calibrate terminal-tasks" in refused_text
    assert "estimated cost $0.55296" in refused_text
    assert "of the $0.50 per-command budget" in refused_text
    assert "judge judge: openai/judge-id" in refused_text
    assert "5 remaining judge calls with up to 3 attempts each" in refused_text
    assert "--yes cannot override" in refused_text
    assert "missing labels" not in refused_text
    assert _RuntimeCatalog.judge_clients == []
    assert store.read_review() == before_preflight
    set_maximum_command_cost_usd(25.0, root)

    _RuntimeCatalog.judge_fail_after = 1
    interrupted = runner.invoke(app, _calibrate_arguments(root, labels))
    assert interrupted.exit_code != 0
    assert "Trace 1 of 5" in interrupted.output
    assert "Original user request:" in interrupted.output
    assert "Tool call:" in interrupted.output
    assert "Tool arguments:" in interrupted.output
    assert "Tool result:" in interrupted.output
    assert "Final outcome:" in interrupted.output
    assert "Configured judge proposals" in interrupted.output
    assert "Proposed score: 1" in interrupted.output
    assert "Proposed judgment:" in interrupted.output
    assert "0: The agent did not complete the requested task." in interrupted.output
    assert "1: The agent successfully completed the requested task." in interrupted.output
    assert len(_trace_reviews(store)) == 1

    _RuntimeCatalog.judge_fail_after = None
    _RuntimeCatalog.judge_clients = []
    resumed = runner.invoke(
        app,
        [argument for argument in _calibrate_arguments(root, labels) if argument != "--approve"],
    )
    assert resumed.exit_code == 2
    resumed_text = " ".join(unstyle(resumed.output).replace("│", " ").split())
    assert "review progress: 1/5 distinct trace lineages complete" in resumed_text
    assert "Trace 1 of 5" not in resumed.output
    assert "Trace 2 of 5" in resumed.output
    assert "--approve" in resumed_text
    assert sum(client.calls for client in _RuntimeCatalog.judge_clients) == _SAMPLE_SIZE - 1
    drafted = _drafted_labels(store)
    assert len(drafted) == _SAMPLE_SIZE
    assert {item["score"] for item in drafted} == {1}
    assert len(_trace_reviews(store)) == _SAMPLE_SIZE
    assert _review_completion(store) == (True, False)

    _RuntimeCatalog.judge_fail_after = 0
    _RuntimeCatalog.judge_clients = []
    unapproved_replay = runner.invoke(
        app,
        [
            argument
            for argument in _calibrate_arguments(root, ())
            if argument not in {"--yes", "--approve"}
        ],
    )
    assert unapproved_replay.exit_code == 2
    assert "already complete" in unapproved_replay.output
    replay_text = unstyle(unapproved_replay.output)
    assert "--approve" in replay_text
    assert "Cost preflight" not in unapproved_replay.output
    assert "Resuming" not in unapproved_replay.output
    assert _RuntimeCatalog.judge_clients == []
    assert _review_completion(store) == (True, False)

    approved = runner.invoke(
        app,
        [argument for argument in _calibrate_arguments(root, ()) if argument != "--yes"],
    )
    assert approved.exit_code == 0, approved.output
    assert "Approved judge calibration" in approved.output
    assert "Cost preflight" not in approved.output
    assert _RuntimeCatalog.judge_clients == []
    assert _review_completion(store) == (True, True)

    replay = runner.invoke(
        app,
        [
            *[
                argument
                for argument in _calibrate_arguments(root, ())
                if argument not in {"--yes", "--approve"}
            ],
            "--page",
        ],
    )
    assert replay.exit_code == 0, replay.output
    assert "Cost preflight" not in replay.output
    assert "authorization:" not in replay.output
    assert "Proceed?" not in replay.output
    assert "Approved judge calibration" in replay.output
    assert _RuntimeCatalog.judge_clients == []

    resampled = runner.invoke(
        app,
        [
            argument
            for argument in _calibrate_arguments(root, (), sample_size=_SAMPLE_SIZE - 2)
            if argument not in {"--yes", "--approve"}
        ],
    )
    assert resampled.exit_code == 0, resampled.output
    assert "already complete" in resampled.output
    assert "Approved judge calibration" in resampled.output
    assert len(_drafted_labels(store)) == _SAMPLE_SIZE

    rollouts = _rollout_payloads(store)
    assert len(rollouts) >= _SAMPLE_SIZE
    for rollout, text in rollouts:
        assert rollout.evidence_source == "production"
        assert rollout.candidate is None
        provenance = rollout.provider_free_source
        assert provenance is not None
        assert provenance.reason == "source_trace_records_no_model_identity"
        assert provenance.checked_span_count == len(rollout.spans)
        assert all(span.model is None for span in rollout.spans)
        assert _JUDGE_MODEL_ID not in text

    reviews = _trace_reviews(store)
    assert len(reviews) == _SAMPLE_SIZE
    for review in reviews:
        assert review.provenance.historical_source == "provider_free_production_trace"
        assert review.provenance.proposal_author == "configured_judge"
        assert review.provenance.decision_author == "human"
        assert review.original_judge_response
        assert review.judge_model.model_id == _JUDGE_MODEL_ID
        assert review.rubric_revision == plan.setup.rubric
        assert review.axes[0].human_correction is None
        assert review.axes[0].final_accepted_label.score_source == "configured_judge"
        assert review.axes[0].final_accepted_label.judgment_source == "configured_judge"

    dataset = load_trace_dataset(store.artifacts, plan.setup.trace_dataset.artifact_id)
    assert len(dataset.traces) >= 100
    assert all(span.model is None for trace in dataset.traces for span in trace.spans)
    assert _JUDGE_MODEL_ID not in json.dumps(
        [trace.model_dump(mode="json") for trace in dataset.traces]
    )


def _drafted_labels(store: ProjectStore) -> tuple[dict[str, object], ...]:
    """Return the persisted resumable human labels of the project review state."""
    review = store.read_review()
    assert isinstance(review, dict)
    manual_judge = review["manual_judge"]
    assert isinstance(manual_judge, dict)
    drafts = manual_judge["label_drafts"]
    assert isinstance(drafts, list)
    assert len(drafts) == 1
    draft = drafts[0]
    assert isinstance(draft, dict)
    entries = draft["labels"]
    assert isinstance(entries, list)
    labels: list[dict[str, object]] = []
    for entry in entries:
        assert isinstance(entry, dict)
        labels.append(entry)
    return tuple(labels)


def _review_completion(store: ProjectStore) -> tuple[bool, bool]:
    """Return whether judge audit and explicit approval pointers are present.

    Args:
        store: Project whose mutable review pointers are inspected.

    Returns:
        Audit-present and approved-calibration-present flags.
    """
    review = store.read_review()
    assert isinstance(review, dict)
    manual_judge = review["manual_judge"]
    assert isinstance(manual_judge, dict)
    return manual_judge.get("audit") is not None, manual_judge.get(
        "approved_calibration"
    ) is not None


def _trace_reviews(store: ProjectStore) -> tuple[ManualJudgeTraceReviewArtifact, ...]:
    """Return completed immutable trace review artifacts from mutable pointer state.

    Args:
        store: Project whose completed trace review pointers are inspected.

    Returns:
        Parsed trace review artifacts in incremental completion order.
    """
    review = store.read_review()
    assert isinstance(review, dict)
    manual_judge = review["manual_judge"]
    assert isinstance(manual_judge, dict)
    pointers = manual_judge["trace_reviews"]
    assert isinstance(pointers, list)
    artifacts: list[ManualJudgeTraceReviewArtifact] = []
    for pointer in pointers:
        assert isinstance(pointer, dict)
        artifact_id = pointer["artifact_id"]
        assert isinstance(artifact_id, str)
        artifacts.append(
            ManualJudgeTraceReviewArtifact.model_validate_json(
                store.artifacts.read_bytes(artifact_id, "review.json")
            )
        )
    return tuple(artifacts)
