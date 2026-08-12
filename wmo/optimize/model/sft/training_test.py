"""Deterministic offline SFT-run tests with an injected no-network trainer."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest

from wmo.common.core.artifacts import ArtifactInput
from wmo.common.models import AssistantAction, NumericMeasurement
from wmo.optimize.model.sft.contracts import (
    AssistantActionEvent,
    PartitionedSFTExample,
    SFTDataset,
    SFTDatasetArtifact,
    SFTExample,
    SFTInspectionReport,
    SFTMessage,
    TraceExampleSource,
)
from wmo.optimize.model.sft.rendering import partitioned_rows_sha256
from wmo.optimize.model.sft.training import (
    TinkerSFTBudgetExceeded,
    TinkerSFTOptimizer,
    TinkerSFTResult,
    TinkerSFTResumeError,
    TinkerSFTSpec,
    TrainerBatchResult,
    TrainerDatum,
    train_tinker_sft,
)

_TIME = datetime(2026, 8, 12, tzinfo=UTC)
_DIGEST = "d" * 64


@dataclass(frozen=True)
class _FakeDatum:
    """One rendered datum returned by the fake trainer without an SDK dependency."""

    example_id: str
    supervised_token_count: int


class _FakeSession:
    """Record deterministic training calls without creating a Tinker client."""

    def __init__(self, backend: _FakeBackend) -> None:
        self._backend = backend

    def render_examples(self, examples: Sequence[SFTExample]) -> tuple[_FakeDatum, ...]:
        self._backend.rendered_example_ids.extend(example.example_id for example in examples)
        return tuple(
            _FakeDatum(example_id=example.example_id, supervised_token_count=index + 3)
            for index, example in enumerate(examples)
        )

    def train_batch(
        self, datums: Sequence[TrainerDatum], *, learning_rate: float
    ) -> TrainerBatchResult:
        self._backend.train_calls += 1
        if self._backend.fail_on_train_call == self._backend.train_calls:
            raise RuntimeError("injected training failure")
        self._backend.trained_example_ids.extend(datum.example_id for datum in datums)
        return TrainerBatchResult(
            loss=1.0 / self._backend.train_calls,
            gradient_norm=0.5,
            cost_usd=NumericMeasurement(
                value=self._backend.cost_per_batch,
                provenance="observed",
            ),
        )

    def save_state(self, checkpoint_name: str) -> str:
        self._backend.saved_state_names.append(checkpoint_name)
        return f"fake://state/{checkpoint_name}"

    def save_sampling_handle(self, model_name: str) -> str:
        self._backend.saved_model_names.append(model_name)
        return f"fake://model/{model_name}"


class _FakeBackend:
    """A complete backend double proving the runner needs neither credentials nor a network."""

    def __init__(
        self,
        *,
        cost_per_batch: float = 0.10,
        fail_on_train_call: int | None = None,
    ) -> None:
        self.cost_per_batch = cost_per_batch
        self.fail_on_train_call = fail_on_train_call
        self.open_resume_paths: list[str | None] = []
        self.rendered_example_ids: list[str] = []
        self.trained_example_ids: list[str] = []
        self.saved_state_names: list[str] = []
        self.saved_model_names: list[str] = []
        self.train_calls = 0

    def open(self, spec: TinkerSFTSpec, resume_state_path: str | None) -> _FakeSession:
        self.open_resume_paths.append(resume_state_path)
        return _FakeSession(self)


def _example(name: str, partition: Literal["train", "held_out"]) -> PartitionedSFTExample:
    """Build one immutable W12-shaped SFT row for a deterministic training test."""
    return PartitionedSFTExample(
        partition=partition,
        fingerprint=hashlib.sha256(name.encode("utf-8")).hexdigest(),
        example=SFTExample(
            example_id=f"example-{name}",
            leakage_group_id=f"lineage-{name}",
            task=f"Resolve request {name}.",
            history=(
                SFTMessage(role="system", content="Follow the support policy."),
                SFTMessage(role="user", content=f"Customer request {name}."),
                AssistantActionEvent(action=AssistantAction(content="I will investigate.")),
            ),
            target=AssistantAction(content=f"Resolved {name}."),
            source=TraceExampleSource(
                trace_id=f"trace-{name}",
                acceptance_evidence=ArtifactInput(
                    artifact_id="acceptance-evidence", sha256=_DIGEST
                ),
            ),
            source_step_index=1,
        ),
    )


def _dataset() -> SFTDatasetArtifact:
    """Build one accepted frozen dataset with two train rows and one held-out row."""
    rows = (
        _example("train-a", "train"),
        _example("train-b", "train"),
        _example("held-out", "held_out"),
    )
    train_rows = tuple(row for row in rows if row.partition == "train")
    held_out_rows = tuple(row for row in rows if row.partition == "held_out")
    dataset = SFTDataset(
        schema_version=1,
        created_at=_TIME,
        inputs=(ArtifactInput(artifact_id="acceptance-evidence", sha256=_DIGEST),),
        code_revision="w12-fixture",
        dataset_id="sft-dataset-fixture",
        build_sha256="b" * 64,
        status="accepted",
        acceptance_rule_ids=("acceptance-rule",),
        acceptance_evidence_ids=("acceptance-evidence",),
        train_leakage_group_ids=tuple(sorted(row.example.leakage_group_id for row in train_rows)),
        held_out_leakage_group_ids=tuple(
            sorted(row.example.leakage_group_id for row in held_out_rows)
        ),
        train_example_ids=tuple(sorted(row.example.example_id for row in train_rows)),
        held_out_example_ids=tuple(sorted(row.example.example_id for row in held_out_rows)),
        examples_path="examples.jsonl",
        examples_sha256=partitioned_rows_sha256(rows),
    )
    inspection = SFTInspectionReport(
        report_id="sft-inspection-fixture",
        dataset_id=dataset.dataset_id,
        build_sha256=dataset.build_sha256,
        source_count=1,
        accepted_source_count=1,
        eligible_action_count=len(rows),
        fingerprint_count=len(rows),
        connected_component_count=len(rows),
        train_example_count=len(train_rows),
        held_out_example_count=len(held_out_rows),
        exclusions=(),
        representative_train_example_ids=tuple(row.example.example_id for row in train_rows),
        representative_held_out_example_ids=tuple(row.example.example_id for row in held_out_rows),
    )
    return SFTDatasetArtifact(
        dataset=dataset,
        sources=(),
        partitions=(),
        inspection=inspection,
        representative_samples=rows,
        rows=rows,
    )


def _spec(**overrides: int | float | str | None) -> TinkerSFTSpec:
    """Return one small deterministic training specification."""
    fields: dict[str, int | float | str | None] = {
        "base_model": "test-base-model",
        "lora_rank": 8,
        "learning_rate": 0.01,
        "batch_size": 1,
        "epochs": 1,
        "checkpoint_every_steps": 1,
        "maximum_steps": None,
        "maximum_datum_tokens": 128,
        "maximum_cost_usd": None,
    }
    return TinkerSFTSpec.model_validate(fields | overrides)


def _run(
    output_dir: Path,
    backend: _FakeBackend,
    *,
    spec: TinkerSFTSpec | None = None,
) -> TinkerSFTResult:
    """Run the public operation with immutable test provenance."""
    return train_tinker_sft(
        _dataset(),
        spec or _spec(),
        output_dir,
        backend=backend,
        created_at=_TIME,
        code_revision="w13-test",
    )


def test_fake_backend_consumes_only_train_rows_and_writes_terminal_provenance(
    tmp_path: Path,
) -> None:
    """The injected fake completes without a service client, credentials, or network call."""
    backend = _FakeBackend()

    result = _run(tmp_path / "run", backend)

    assert backend.open_resume_paths == [None]
    assert backend.rendered_example_ids == ["example-train-a", "example-train-b"]
    assert backend.trained_example_ids == ["example-train-a", "example-train-b"]
    assert result.dataset_id == "sft-dataset-fixture"
    assert result.training_step_count == 2
    assert result.total_cost_usd == NumericMeasurement(value=0.20, provenance="observed")
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    model = json.loads((tmp_path / "run" / "model.json").read_text(encoding="utf-8"))
    terminal = json.loads((tmp_path / "run" / "result.json").read_text(encoding="utf-8"))
    metrics = (tmp_path / "run" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert manifest["dataset_build_sha256"] == "b" * 64
    assert manifest["dataset_examples_sha256"] == _dataset().dataset.examples_sha256
    assert model["sampling_handle"].startswith("fake://model/")
    assert terminal["model_id"] == model["model_id"]
    assert [json.loads(line)["record_type"] for line in metrics] == [
        "metric",
        "checkpoint",
        "metric",
        "checkpoint",
    ]
    assert "quality" not in json.dumps(terminal)


def test_resume_uses_last_durable_checkpoint_without_replaying_completed_batch(
    tmp_path: Path,
) -> None:
    """A retry restores the recorded state before training only the remaining batch."""
    backend = _FakeBackend(fail_on_train_call=2)
    output_dir = tmp_path / "run"

    with pytest.raises(RuntimeError, match="injected training failure"):
        _run(output_dir, backend)

    saved_state = f"fake://state/{backend.saved_state_names[-1]}"
    backend.fail_on_train_call = None
    result = _run(output_dir, backend)

    assert backend.open_resume_paths == [None, saved_state]
    assert backend.trained_example_ids == ["example-train-a", "example-train-b"]
    assert result.training_step_count == 2
    records = [
        json.loads(line)
        for line in (output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["record_type"] for record in records] == [
        "metric",
        "checkpoint",
        "resume",
        "metric",
        "checkpoint",
    ]


def test_retry_after_an_uncheckpointed_metric_starts_a_new_base_attempt(tmp_path: Path) -> None:
    """Abandoned pre-checkpoint metrics never become a later checkpoint's durable lineage."""
    backend = _FakeBackend(fail_on_train_call=2)
    output_dir = tmp_path / "run"
    spec = _spec(checkpoint_every_steps=2)

    with pytest.raises(RuntimeError, match="injected training failure"):
        _run(output_dir, backend, spec=spec)

    assert backend.saved_state_names == []
    backend.fail_on_train_call = None
    result = _run(output_dir, backend, spec=spec)

    assert backend.open_resume_paths == [None, None]
    assert backend.trained_example_ids == [
        "example-train-a",
        "example-train-a",
        "example-train-b",
    ]
    records = [
        json.loads(line)
        for line in (output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [(record["record_type"], record["attempt_id"]) for record in records] == [
        ("metric", 1),
        ("metric", 2),
        ("metric", 2),
        ("checkpoint", 2),
    ]
    assert result.training_metric_count == 2
    assert result.total_cost_usd == NumericMeasurement(value=0.20, provenance="observed")


def test_completed_run_is_idempotent_and_does_not_open_another_trainer(tmp_path: Path) -> None:
    """A completed terminal result is returned from local provenance before backend composition."""
    backend = _FakeBackend()
    output_dir = tmp_path / "run"

    first = _run(output_dir, backend)
    second = _run(output_dir, backend)

    assert second == first
    assert backend.open_resume_paths == [None]


def test_budget_exhaustion_persists_a_checkpoint_and_refuses_further_spend(
    tmp_path: Path,
) -> None:
    """Observed spend over the cap leaves a resumable checkpoint but no terminal model."""
    backend = _FakeBackend(cost_per_batch=0.20)
    output_dir = tmp_path / "run"

    with pytest.raises(TinkerSFTBudgetExceeded, match="maximum_cost_usd"):
        _run(output_dir, backend, spec=_spec(maximum_cost_usd=0.30, checkpoint_every_steps=10))

    records = [
        json.loads(line)
        for line in (output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["record_type"] for record in records] == ["metric", "metric", "checkpoint"]
    assert not (output_dir / "model.json").exists()
    opens_before_retry = list(backend.open_resume_paths)
    with pytest.raises(TinkerSFTBudgetExceeded, match="already reached"):
        _run(output_dir, backend, spec=_spec(maximum_cost_usd=0.30, checkpoint_every_steps=10))
    assert backend.open_resume_paths == opens_before_retry


def test_changed_spec_cannot_reuse_an_append_only_run_directory(tmp_path: Path) -> None:
    """A run directory remains tied to its original frozen dataset and optimization settings."""
    backend = _FakeBackend()
    output_dir = tmp_path / "run"
    _run(output_dir, backend)

    with pytest.raises(TinkerSFTResumeError, match="training spec"):
        _run(output_dir, backend, spec=_spec(learning_rate=0.02))


def test_optimizer_delegates_to_the_same_injected_backend(tmp_path: Path) -> None:
    """The narrow optimizer seam adds no registry or alternate training implementation."""
    backend = _FakeBackend()
    optimizer = TinkerSFTOptimizer(backend)

    result = optimizer.optimize(
        dataset=_dataset(),
        spec=_spec(),
        output_dir=tmp_path / "run",
        created_at=_TIME,
        code_revision="w13-test",
    )

    assert result.training_step_count == 2
    assert backend.open_resume_paths == [None]
