"""End-to-end local tests for the canonical trace-to-task-set build command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from wmo.cli.app import app
from wmo.common.observability import RunRecord
from wmo.common.observability.telemetry import BuildTelemetryStats
from wmo.common.project import ProjectStore, artifact_input
from wmo.common.tasks import TaskSet
from wmo.common.traces import TraceDataset

_RUNNER = CliRunner()


def _attribute(key: str, value: str) -> dict[str, object]:
    """Encode one textual OpenTelemetry attribute for a local canonical fixture."""
    return {"key": key, "value": {"stringValue": value}}


def _otlp_export(tmp_path: Path, count: int = 100) -> Path:
    """Write distinct valid OTLP JSONL traces that exercise the approved 50/20 default split."""
    records = []
    for index in range(count):
        trace_id = f"{index + 1:032x}"
        span_id = f"{index + 1:016x}"
        records.append(
            {
                "traceId": trace_id,
                "spanId": span_id,
                "name": "agent.model_call",
                "startTimeUnixNano": str(1_760_000_000_000_000_000 + index * 1_000_000_000),
                "endTimeUnixNano": str(1_760_000_001_000_000_000 + index * 1_000_000_000),
                "attributes": [
                    _attribute("gen_ai.operation.name", "chat"),
                    _attribute("gen_ai.provider.name", "openai"),
                    _attribute("gen_ai.request.model", "gpt-test"),
                    _attribute(
                        "gen_ai.input.messages",
                        json.dumps(
                            [
                                {
                                    "role": "user",
                                    "content": f"Resolve support request {index}",
                                }
                            ]
                        ),
                    ),
                    _attribute("wmo.customer.id", f"customer-{index}"),
                    _attribute("wmo.conversation.id", f"conversation-{index}"),
                ],
            }
        )
    path = tmp_path / "traces.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    return path


def _posthog_export(tmp_path: Path, count: int = 100) -> Path:
    """Write distinct local PostHog generation traces without an HTTP call."""
    path = tmp_path / "posthog.json"
    path.write_text(
        json.dumps(
            [
                {
                    "event": "$ai_generation",
                    "timestamp": f"2026-08-11T00:{index // 60:02d}:{index % 60:02d}Z",
                    "properties": {
                        "$ai_trace_id": f"{index + 1:032x}",
                        "$ai_span_id": f"generation-{index}",
                        "$ai_provider": "openai",
                        "$ai_model": "gpt-test",
                        "$ai_input": [
                            {
                                "role": "user",
                                "content": f"Resolve distinct support request {index}",
                            }
                        ],
                        "$ai_output_choices": [
                            {"role": "assistant", "content": "I sent reset instructions."}
                        ],
                        "wmo.customer.id": f"customer-{index}",
                        "wmo.conversation.id": f"conversation-{index}",
                        "wmo.outcome.status": "success",
                    },
                }
                for index in range(count)
            ]
        ),
        encoding="utf-8",
    )
    return path


def _task_set_for(root: Path, project_id: str) -> tuple[TraceDataset, TaskSet]:
    """Read the two canonical artifacts written by a successful local build."""
    artifacts = ProjectStore(root, project_id).artifacts
    manifests = tuple(artifacts.read(artifact_id).manifest for artifact_id in artifacts.list_ids())
    trace_manifest = next(
        manifest for manifest in manifests if manifest.artifact_type == "trace-dataset"
    )
    task_manifest = next(manifest for manifest in manifests if manifest.artifact_type == "task-set")
    trace_dataset = TraceDataset.model_validate_json(
        artifacts.read_bytes(trace_manifest.artifact_id, "trace-dataset.json")
    )
    task_set = TaskSet.model_validate_json(
        artifacts.read_bytes(task_manifest.artifact_id, "task-set.json")
    )
    return trace_dataset, task_set


def test_build_reads_the_raw_otlp_file_once_and_persists_the_immutable_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The CLI performs exactly one raw read, then derives TaskSet only from TraceDataset input."""
    source = _otlp_export(tmp_path)
    original_read_bytes = Path.read_bytes
    reads = 0
    captured: list[BuildTelemetryStats] = []
    telemetry_calls: list[RunRecord] = []

    def count_source_reads(path: Path) -> bytes:
        nonlocal reads
        if path == source:
            reads += 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", count_source_reads)

    def capture(*, stats: BuildTelemetryStats, record: RunRecord, **_kwargs: object) -> None:
        captured.append(stats)
        telemetry_calls.append(record)

    monkeypatch.setattr("wmo.cli.build_cmd.capture_build_completed", capture)
    root = tmp_path / ".wmo"
    result = _RUNNER.invoke(
        app,
        [
            "build",
            str(source),
            "--source",
            "otlp",
            "--project",
            "support",
            "--root",
            str(root),
        ],
    )

    assert result.exit_code == 0, result.output
    assert reads == 1
    trace_dataset, task_set = _task_set_for(root, "support")
    manifest = ProjectStore(root, "support").artifacts.read(trace_dataset.dataset_id).manifest
    assert task_set.inputs == (artifact_input(manifest),)
    assert len(trace_dataset.trace_ids) == 100
    assert sum(task_id.startswith("task-") for task_id in task_set.task_ids) == 70
    tasks = ProjectStore(root, "support").artifacts.read_bytes(task_set.task_set_id, "tasks.jsonl")
    task_records = tuple(line for line in tasks.decode("utf-8").splitlines() if line)
    assert sum('"partition":"fit"' in line for line in task_records) == 50
    assert sum('"partition":"held_out"' in line for line in task_records) == 20
    assert len(captured) == 1
    stats = captured[0]
    assert stats.input_trace_count == 100
    assert stats.train_trace_count == 50
    assert stats.heldout_trace_count == 20
    assert telemetry_calls[0].total.calls == 0


def test_build_accepts_a_local_posthog_export_without_using_the_hogql_transport(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """PostHog local exports use the focused canonical converter and persist normal evidence."""

    def fail_pull(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("local PostHog build must not invoke the HogQL pull transport")

    def fail_http_client(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("local PostHog build must not construct an HTTP client")

    monkeypatch.setattr("wmo.simulation.ingest.posthog.pull_posthog_traces", fail_pull)
    monkeypatch.setattr("wmo.simulation.ingest.posthog_pull.httpx.Client", fail_http_client)
    monkeypatch.setattr("wmo.cli.build_cmd.capture_build_completed", lambda **_kwargs: None)
    source = _posthog_export(tmp_path)
    root = tmp_path / ".wmo"
    original_read_bytes = Path.read_bytes
    reads = 0

    def count_source_reads(path: Path) -> bytes:
        nonlocal reads
        if path == source:
            reads += 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", count_source_reads)

    result = _RUNNER.invoke(
        app,
        [
            "build",
            str(source),
            "--source",
            "posthog",
            "--project",
            "support",
            "--root",
            str(root),
        ],
    )

    assert result.exit_code == 0, result.output
    trace_dataset, task_set = _task_set_for(root, "support")
    assert trace_dataset.source is not None
    assert trace_dataset.source.kind == "file"
    assert trace_dataset.invalid_trace_count == 0
    assert len(trace_dataset.trace_ids) == 100
    assert task_set.task_ids
    assert reads == 1


@pytest.mark.parametrize("source_kind", ["otlp", "posthog"])
@pytest.mark.parametrize(
    ("trace_count", "accepted"),
    [(99, False), (100, True), (1_000, True), (1_001, False)],
)
def test_build_enforces_normalized_trace_operating_range_for_each_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source_kind: str,
    trace_count: int,
    accepted: bool,
) -> None:
    """Both canonical loaders enforce the exact 100 through 1000 trace build range."""
    monkeypatch.setattr("wmo.cli.build_cmd.capture_build_completed", lambda **_kwargs: None)
    source = (
        _otlp_export(tmp_path, trace_count)
        if source_kind == "otlp"
        else _posthog_export(tmp_path, trace_count)
    )
    root = tmp_path / ".wmo"

    result = _RUNNER.invoke(
        app,
        [
            "build",
            str(source),
            "--source",
            source_kind,
            "--project",
            "boundary",
            "--root",
            str(root),
        ],
    )

    if accepted:
        assert result.exit_code == 0, result.output
        trace_dataset, _task_set = _task_set_for(root, "boundary")
        assert len(trace_dataset.trace_ids) == trace_count
    else:
        assert result.exit_code == 2
        assert "requires 100 to 1000 valid normalized traces" in result.output
        assert ProjectStore(root, "boundary").artifacts.list_ids() == ()


def test_build_rejects_unknown_source_and_missing_local_evidence(tmp_path: Path) -> None:
    """The direct build surface names only its two canonical input formats and local file need."""
    source = _otlp_export(tmp_path, count=1)

    unknown = _RUNNER.invoke(
        app,
        ["build", str(source), "--source", "langsmith", "--project", "support"],
    )
    missing = _RUNNER.invoke(
        app,
        ["build", str(tmp_path / "missing.json"), "--project", "support"],
    )

    assert unknown.exit_code == 2
    assert "choose one of: otlp, posthog" in " ".join(unknown.output.replace("│", " ").split())
    assert missing.exit_code == 2
    assert "trace file not found" in missing.output


def test_build_rejects_the_removed_name_compatibility_alias(tmp_path: Path) -> None:
    """The direct task-set build accepts only --project for its local destination."""
    source = _otlp_export(tmp_path, count=1)

    result = _RUNNER.invoke(app, ["build", str(source), "--name", "support"])

    assert result.exit_code == 2
    assert "No such option: --name" in result.output


def test_build_rejects_removed_file_alias_and_requires_project(tmp_path: Path) -> None:
    """The locked build surface has one positional trace and one explicit project."""
    source = _otlp_export(tmp_path, count=1)

    alias = _RUNNER.invoke(app, ["build", "--file", str(source), "--project", "support"])
    missing_project = _RUNNER.invoke(app, ["build", str(source)])

    assert alias.exit_code == 2
    assert "No such option: --file" in alias.output
    assert missing_project.exit_code == 2
    assert "--project" in missing_project.output
