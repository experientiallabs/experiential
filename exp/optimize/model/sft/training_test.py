"""Deterministic offline SFT-run tests with an injected no-network trainer."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import canonical_json_bytes, sha256_json, stable_id
from exp.common.models import NumericMeasurement
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.model.sft.builder import write_sft_dataset
from exp.optimize.model.sft.builder_test import (
    _build as _build_fixture_dataset,
)
from exp.optimize.model.sft.builder_test import (
    _store as _fixture_store,
)
from exp.optimize.model.sft.builder_test import (
    _write_production_source,
)
from exp.optimize.model.sft.contracts import SFTDatasetArtifact, SFTExample
from exp.optimize.model.sft.rendering import (
    canonical_partitioned_rows_jsonl,
    partitioned_rows_sha256,
)
from exp.optimize.model.sft.training import (
    TinkerSFTAmbiguousStepError,
    TinkerSFTBudgetExceeded,
    TinkerSFTError,
    TinkerSFTOptimizer,
    TinkerSFTResult,
    TinkerSFTResumeError,
    TinkerSFTSpec,
    train_tinker_sft,
)
from exp.optimize.model.sft.training_contracts import TrainerBatchResult, TrainerDatum

_TIME = datetime(2026, 8, 12, tzinfo=UTC)


@dataclass(frozen=True)
class _FakeDatum:
    """One rendered datum returned by the fake trainer without an SDK dependency."""

    example_id: str
    supervised_token_count: int


class _FakeSession:
    """Record deterministic training calls without creating a Tinker client."""

    def __init__(self, backend: _FakeBackend) -> None:
        """Bind the shared fake backend call journal."""
        self._backend = backend

    def render_examples(self, examples: Sequence[SFTExample]) -> tuple[_FakeDatum, ...]:
        """Render deterministic datum identities and supervised-token counts.

        Args:
            examples: Frozen W12 examples supplied to the trainer.

        Returns:
            One fake datum for each example in the original order.
        """
        self._backend.rendered_example_ids.extend(example.example_id for example in examples)
        return tuple(
            _FakeDatum(example_id=example.example_id, supervised_token_count=index + 3)
            for index, example in enumerate(examples)
        )

    def train_batch(
        self, datums: Sequence[TrainerDatum], *, learning_rate: float
    ) -> TrainerBatchResult:
        """Record one batch and return deterministic metrics or an injected failure.

        Args:
            datums: Rendered examples scheduled for this optimizer step.
            learning_rate: Frozen learning rate accepted by the backend seam.

        Returns:
            Deterministic loss, gradient, and observed cost facts.

        Raises:
            RuntimeError: The configured failure boundary is reached.
        """
        del learning_rate
        self._backend.train_calls += 1
        if self._backend.fail_on_train_call == self._backend.train_calls:
            raise RuntimeError("injected training failure")
        self._backend.trained_example_ids.extend(datum.example_id for datum in datums)
        result = TrainerBatchResult(
            loss=1.0 / self._backend.train_calls,
            gradient_norm=0.5,
            cost_usd=NumericMeasurement(
                value=self._backend.cost_per_batch,
                provenance="observed",
            ),
        )
        if self._backend.fail_after_train_call == self._backend.train_calls:
            raise RuntimeError("injected failure after optimizer dispatch")
        return result

    def save_state(self, checkpoint_name: str) -> str:
        """Record a checkpoint request and return its deterministic provider handle.

        Args:
            checkpoint_name: Stable checkpoint name selected by the trainer.

        Returns:
            Configured or derived fake provider state handle.
        """
        self._backend.saved_state_names.append(checkpoint_name)
        return self._backend.state_resource or f"fake://state/{checkpoint_name}"

    def save_sampling_handle(self, model_name: str) -> str:
        """Record terminal model persistence and return a fake sampling handle.

        Args:
            model_name: Stable terminal model name selected by the trainer.

        Returns:
            Deterministic fake sampling handle.
        """
        self._backend.saved_model_names.append(model_name)
        return f"fake://model/{model_name}"


class _FakeBackend:
    """A complete backend double proving the runner needs neither credentials nor a network."""

    def __init__(
        self,
        *,
        cost_per_batch: float = 0.10,
        conservative_cost_per_batch: float | None = 0.10,
        fail_on_train_call: int | None = None,
        fail_after_train_call: int | None = None,
        fail_on_cost_call: int | None = None,
        state_resource: str | None = None,
    ) -> None:
        """Configure deterministic costs, failures, state, and empty call journals.

        Args:
            cost_per_batch: Observed cost returned after each completed batch.
            conservative_cost_per_batch: Pre-dispatch upper bound, or ``None`` when unknown.
            fail_on_train_call: Batch number that fails before a result is returned.
            fail_after_train_call: Batch number that fails after simulated provider work.
            fail_on_cost_call: Estimate call number that raises a planning failure.
            state_resource: Optional fixed provider checkpoint handle.
        """
        self.cost_per_batch = cost_per_batch
        self.conservative_cost_per_batch = conservative_cost_per_batch
        self.fail_on_train_call = fail_on_train_call
        self.fail_after_train_call = fail_after_train_call
        self.fail_on_cost_call = fail_on_cost_call
        self.state_resource = state_resource
        self.open_resume_paths: list[str | None] = []
        self.rendered_example_ids: list[str] = []
        self.trained_example_ids: list[str] = []
        self.saved_state_names: list[str] = []
        self.saved_model_names: list[str] = []
        self.train_calls = 0
        self.cost_calls = 0

    def conservative_step_cost(
        self, spec: TinkerSFTSpec, *, batch_example_count: int
    ) -> NumericMeasurement | None:
        """Return the configured batch bound or an injected planning failure.

        Args:
            spec: Frozen training settings accepted by the backend seam.
            batch_example_count: Exact examples in the planned batch.

        Returns:
            Configured conservative measurement, or ``None`` when unknown.

        Raises:
            RuntimeError: The configured estimate failure boundary is reached.
        """
        del spec, batch_example_count
        self.cost_calls += 1
        if self.fail_on_cost_call == self.cost_calls:
            raise RuntimeError("injected pre-dispatch planning failure")
        if self.conservative_cost_per_batch is None:
            return None
        return NumericMeasurement(
            value=self.conservative_cost_per_batch,
            provenance="estimated",
        )

    def open(self, spec: TinkerSFTSpec, resume_state_path: str | None) -> _FakeSession:
        """Record the resume handle and return a session sharing this call journal.

        Args:
            spec: Frozen training settings accepted without provider access.
            resume_state_path: Optional durable checkpoint handle.

        Returns:
            A deterministic fake training session.
        """
        del spec
        self.open_resume_paths.append(resume_state_path)
        return _FakeSession(self)


@dataclass(frozen=True)
class _PersistedDataset:
    """Canonical persisted W12 fixture used through the public W13 entry boundary."""

    store: ProjectStore
    artifact: SFTDatasetArtifact


def _persisted_dataset(tmp_path: Path) -> _PersistedDataset:
    """Build and persist two verified production lineages through W12 services."""
    store = _fixture_store(tmp_path / "project")
    sources = (
        _write_production_source(store, "train-source-a"),
        _write_production_source(store, "train-source-b"),
    )
    artifact = _build_fixture_dataset(store, production=sources)
    write_sft_dataset(store, artifact)
    return _PersistedDataset(store=store, artifact=artifact)


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
        "training_usd_per_million_tokens": None,
    }
    return TinkerSFTSpec.model_validate(fields | overrides)


def _run(
    fixture: _PersistedDataset,
    output_dir: Path,
    backend: _FakeBackend,
    *,
    spec: TinkerSFTSpec | None = None,
) -> TinkerSFTResult:
    """Run the public operation with immutable test provenance."""
    return train_tinker_sft(
        fixture.store,
        fixture.artifact.dataset.dataset_id,
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
    fixture = _persisted_dataset(tmp_path)
    backend = _FakeBackend()
    train_ids = tuple(
        row.example.example_id for row in fixture.artifact.rows if row.partition == "train"
    )

    result = _run(fixture, tmp_path / "run", backend)

    assert backend.open_resume_paths == [None]
    assert backend.rendered_example_ids == list(train_ids)
    assert sorted(backend.trained_example_ids) == sorted(train_ids)
    assert result.dataset_id == fixture.artifact.dataset.dataset_id
    assert result.training_step_count == 2
    assert result.total_cost_usd == NumericMeasurement(value=0.20, provenance="observed")
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    model = json.loads((tmp_path / "run" / "model.json").read_text(encoding="utf-8"))
    terminal = json.loads((tmp_path / "run" / "result.json").read_text(encoding="utf-8"))
    metrics = (tmp_path / "run" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert manifest["dataset_build_sha256"] == fixture.artifact.dataset.build_sha256
    expected_dataset_input = artifact_input(
        fixture.store.artifacts.read(fixture.artifact.dataset.dataset_id).manifest
    )
    assert manifest["dataset_manifest_sha256"] == expected_dataset_input.sha256
    assert manifest["inputs"] == [expected_dataset_input.model_dump(mode="json")]
    assert model["inputs"] == [expected_dataset_input.model_dump(mode="json")]
    assert expected_dataset_input.model_dump(mode="json") in terminal["inputs"]
    assert manifest["dataset_examples_sha256"] == fixture.artifact.dataset.examples_sha256
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
    fixture = _persisted_dataset(tmp_path)
    backend = _FakeBackend(fail_on_cost_call=3)
    output_dir = tmp_path / "run"

    with pytest.raises(RuntimeError, match="planning failure"):
        _run(fixture, output_dir, backend)

    saved_state = f"fake://state/{backend.saved_state_names[-1]}"
    backend.fail_on_cost_call = None
    result = _run(fixture, output_dir, backend)

    assert backend.open_resume_paths == [None, saved_state]
    assert len(backend.trained_example_ids) == 2
    assert len(set(backend.trained_example_ids)) == 2
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


def test_crash_after_optimizer_dispatch_never_replays_and_reports_unknown_spend(
    tmp_path: Path,
) -> None:
    """An unmatched pre-dispatch intent is an honest terminal ambiguity, never a retry cue."""
    fixture = _persisted_dataset(tmp_path)
    backend = _FakeBackend(fail_after_train_call=1)
    output_dir = tmp_path / "run"

    with pytest.raises(RuntimeError, match="after optimizer dispatch"):
        _run(fixture, output_dir, backend)

    assert backend.train_calls == 1
    assert backend.saved_state_names == []
    backend.fail_after_train_call = None
    with pytest.raises(TinkerSFTAmbiguousStepError, match="spend are unknown"):
        _run(fixture, output_dir, backend)
    assert backend.train_calls == 1
    assert backend.open_resume_paths == [None]


def test_completed_run_is_idempotent_and_does_not_open_another_trainer(tmp_path: Path) -> None:
    """A completed terminal result is returned from local provenance before backend composition."""
    fixture = _persisted_dataset(tmp_path)
    backend = _FakeBackend()
    output_dir = tmp_path / "run"

    first = _run(fixture, output_dir, backend)
    second = _run(fixture, output_dir, backend)

    assert second == first
    assert backend.open_resume_paths == [None]


def test_budget_refuses_a_step_whose_upper_bound_exceeds_remaining_spend(
    tmp_path: Path,
) -> None:
    """A $0.10 cap against a $0.20 bound dispatches no trainer or optimizer call."""
    fixture = _persisted_dataset(tmp_path)
    backend = _FakeBackend(cost_per_batch=0.20, conservative_cost_per_batch=0.20)
    output_dir = tmp_path / "run"

    with pytest.raises(TinkerSFTBudgetExceeded, match="remaining budget"):
        _run(fixture, output_dir, backend, spec=_spec(maximum_cost_usd=0.10))

    assert backend.open_resume_paths == []
    assert backend.train_calls == 0


def test_unknown_cost_refuses_before_open_or_optimizer_dispatch(tmp_path: Path) -> None:
    """A budgeted concrete-style backend with no supported bound performs zero remote work."""
    fixture = _persisted_dataset(tmp_path)
    backend = _FakeBackend(conservative_cost_per_batch=None)

    with pytest.raises(TinkerSFTBudgetExceeded, match="backend-proven"):
        _run(fixture, tmp_path / "run", backend, spec=_spec(maximum_cost_usd=1.0))

    assert backend.open_resume_paths == []
    assert backend.train_calls == 0


def test_changed_spec_cannot_reuse_an_append_only_run_directory(tmp_path: Path) -> None:
    """A run directory remains tied to its original frozen dataset and optimization settings."""
    fixture = _persisted_dataset(tmp_path)
    backend = _FakeBackend()
    output_dir = tmp_path / "run"
    _run(fixture, output_dir, backend)

    with pytest.raises(TinkerSFTResumeError, match="training spec"):
        _run(fixture, output_dir, backend, spec=_spec(learning_rate=0.02))


def test_optimizer_delegates_to_the_same_injected_backend(tmp_path: Path) -> None:
    """The narrow optimizer seam adds no registry or alternate training implementation."""
    fixture = _persisted_dataset(tmp_path)
    backend = _FakeBackend()
    optimizer = TinkerSFTOptimizer(backend)

    result = optimizer.optimize(
        store=fixture.store,
        dataset_id=fixture.artifact.dataset.dataset_id,
        spec=_spec(),
        output_dir=tmp_path / "run",
        created_at=_TIME,
        code_revision="w13-test",
    )

    assert result.training_step_count == 2
    assert backend.open_resume_paths == [None]


def test_seed_deterministically_controls_schedule_order(tmp_path: Path) -> None:
    """The same seed repeats exact batch order while a selected alternate seed changes it."""
    fixture = _persisted_dataset(tmp_path)
    first = _FakeBackend()
    repeat = _FakeBackend()
    alternate = _FakeBackend()

    _run(fixture, tmp_path / "first", first, spec=_spec(seed=7))
    _run(fixture, tmp_path / "repeat", repeat, spec=_spec(seed=7))
    _run(fixture, tmp_path / "alternate", alternate, spec=_spec(seed=9))

    assert first.trained_example_ids == repeat.trained_example_ids
    assert first.trained_example_ids != alternate.trained_example_ids


def test_forged_cross_split_persisted_input_never_opens_training(tmp_path: Path) -> None:
    """Even manifest-digested forged rows fail W12 invariants before backend composition."""
    fixture = _persisted_dataset(tmp_path)
    original = fixture.artifact
    held_out = next(row for row in original.rows if row.partition == "held_out")
    forged_row = held_out.model_copy(
        update={
            "partition": "train",
            "example": held_out.example.model_copy(update={"example_id": "forged-example"}),
        }
    )
    rows = (*original.rows, forged_row)
    dataset = original.dataset.model_copy(
        update={
            "dataset_id": "forged-cross-split",
            "build_sha256": "f" * 64,
            "train_example_ids": tuple(
                sorted((*original.dataset.train_example_ids, forged_row.example.example_id))
            ),
            "examples_sha256": partitioned_rows_sha256(rows),
        }
    )
    inspection = original.inspection.model_copy(
        update={
            "dataset_id": dataset.dataset_id,
            "build_sha256": dataset.build_sha256,
            "train_example_count": original.inspection.train_example_count + 1,
        }
    )
    forged = original.model_copy(
        update={"dataset": dataset, "inspection": inspection, "rows": rows}
    )
    fixture.store.artifacts.write(
        artifact_id=dataset.dataset_id,
        artifact_type="sft-dataset",
        envelope=dataset,
        files={
            "dataset.json": canonical_json_bytes(forged.metadata()),
            "examples.jsonl": canonical_partitioned_rows_jsonl(rows),
        },
    )
    backend = _FakeBackend()

    with pytest.raises(TinkerSFTError, match="not safe for training"):
        train_tinker_sft(
            fixture.store,
            dataset.dataset_id,
            _spec(),
            tmp_path / "forged-run",
            backend=backend,
            created_at=_TIME,
            code_revision="w13-test",
        )

    assert backend.open_resume_paths == []
    assert backend.train_calls == 0


def test_completed_reuse_refuses_when_event_log_is_deleted(tmp_path: Path) -> None:
    """A terminal result is not trusted without its hash-bound checkpoint event lineage."""
    fixture = _persisted_dataset(tmp_path)
    backend = _FakeBackend()
    output_dir = tmp_path / "run"
    _run(fixture, output_dir, backend)
    (output_dir / "events.jsonl").unlink()

    with pytest.raises(TinkerSFTResumeError):
        _run(fixture, output_dir, backend)


def test_completed_reuse_rejects_coherent_result_dataset_build_input_edit(
    tmp_path: Path,
) -> None:
    """A self-consistent mutable result cannot replace the canonical W12 manifest binding."""
    fixture = _persisted_dataset(tmp_path)
    backend = _FakeBackend()
    output_dir = tmp_path / "run"
    _run(fixture, output_dir, backend)
    result_path = output_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    forged_build = "0" * 64
    result["dataset_build_sha256"] = forged_build
    for input_item in result["inputs"]:
        if input_item["artifact_id"] == fixture.artifact.dataset.dataset_id:
            input_item["sha256"] = forged_build
    result_path.write_bytes(canonical_json_bytes(result) + b"\n")

    with pytest.raises(TinkerSFTResumeError):
        _run(fixture, output_dir, backend)


def test_completed_reuse_rejects_coherently_truncated_schedule(tmp_path: Path) -> None:
    """A hash-consistent one-step terminal history cannot complete a frozen two-step run."""
    fixture = _persisted_dataset(tmp_path)
    backend = _FakeBackend()
    output_dir = tmp_path / "run"
    _run(fixture, output_dir, backend)

    event_path = output_dir / "events.jsonl"
    events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    first_events = [event for event in events if event.get("step") == 1]
    first_checkpoint = next(event for event in first_events if event["record_type"] == "checkpoint")
    first_metric = next(event for event in first_events if event["record_type"] == "metric")
    event_payload = b"".join(canonical_json_bytes(event) + b"\n" for event in first_events)
    event_path.write_bytes(event_payload)
    events_sha256 = hashlib.sha256(event_payload).hexdigest()

    for path in output_dir.glob("*step-2.step-intent.json"):
        path.unlink()
    second_checkpoint = next(
        event for event in events if event["record_type"] == "checkpoint" and event["step"] == 2
    )
    (output_dir / f"{second_checkpoint['checkpoint_id']}.checkpoint-intent.json").unlink()
    (output_dir / "model-intent.json").unlink()

    model_path = output_dir / "model.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    model.update(
        {
            "events_sha256": events_sha256,
            "final_checkpoint_id": first_checkpoint["checkpoint_id"],
            "final_checkpoint_state_path": first_checkpoint["state_path"],
            "training_step_count": 1,
            "training_metric_count": 1,
            "total_cost_usd": first_metric["cumulative_cost_usd"],
        }
    )
    model["model_id"] = stable_id(
        "tinker-sft-model",
        {
            "run_id": model["run_id"],
            "checkpoint_id": model["final_checkpoint_id"],
            "sampling_handle": model["sampling_handle"],
        },
    )
    model_path.write_bytes(canonical_json_bytes(model) + b"\n")
    model_sha256 = sha256_json(model)

    result_path = output_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update(
        {
            "events_sha256": events_sha256,
            "final_checkpoint_id": first_checkpoint["checkpoint_id"],
            "model_id": model["model_id"],
            "model_sha256": model_sha256,
            "training_step_count": 1,
            "training_metric_count": 1,
            "checkpoint_count": 1,
            "total_cost_usd": first_metric["cumulative_cost_usd"],
        }
    )
    for input_item in result["inputs"]:
        if input_item["artifact_id"] != fixture.artifact.dataset.dataset_id:
            input_item.update({"artifact_id": model["model_id"], "sha256": model_sha256})
    result["inputs"] = sorted(result["inputs"], key=lambda item: item["artifact_id"])
    result["result_id"] = stable_id(
        "tinker-sft-result",
        {"run_id": result["run_id"], "model_sha256": model_sha256},
    )
    result_path.write_bytes(canonical_json_bytes(result) + b"\n")

    with pytest.raises(TinkerSFTResumeError, match="frozen schedule"):
        _run(fixture, output_dir, backend)
    assert backend.open_resume_paths == [None]


def test_provider_token_url_is_rejected_before_checkpoint_persistence(tmp_path: Path) -> None:
    """Provider resource strings with token query material never enter local artifacts."""
    fixture = _persisted_dataset(tmp_path)
    backend = _FakeBackend(state_resource="https://provider.test/state?token=secret-value")

    with pytest.raises(TinkerSFTError, match="opaque provider resource ID"):
        _run(fixture, tmp_path / "run", backend)

    assert not (tmp_path / "run" / "result.json").exists()
