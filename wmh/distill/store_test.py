"""Tests for the distillation run store, adapter store, and handoff snippet."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
from pydantic import BaseModel

from wmh.distill.config import (
    DistillConfig,
    HarborConfig,
    StudentConfig,
    TeacherConfig,
)
from wmh.distill.gate import DistillGateRecord
from wmh.distill.store import (
    CHAMPION_ALIAS,
    DEFAULT_TINKER_OPENAI_ENDPOINT,
    AdapterStore,
    DistillModelCard,
    DistillRunStore,
    build_handoff_toml,
)


class _MetricsRow(BaseModel):
    solve_rate: float
    reverse_kl_per_token: float
    usd: float


class _EvalReport(BaseModel):
    split: str
    solve_rate: float


def _config() -> DistillConfig:
    return DistillConfig(
        student=StudentConfig(base_model="Qwen/Qwen3-8B"),
        teacher=TeacherConfig(model="Qwen/Qwen3-235B-A22B-Instruct-2507"),
        harbor=HarborConfig(job_template="jobs/tb2.yaml"),
    )


def _card(sampler: str = "tinker://fake/sampler/final/0") -> DistillModelCard:
    return DistillModelCard(
        base_model="Qwen/Qwen3-8B",
        lora_rank=32,
        teacher_model="Qwen/Qwen3-235B-A22B-Instruct-2507",
        sampler_path=sampler,
        state_path="tinker://fake/state/5",
        steps_completed=40,
    )


def _gate_record() -> DistillGateRecord:
    return DistillGateRecord(
        accepted=True,
        reason="accepted: after 0.500 >= 0.70 x teacher 0.600 = 0.420 (k=3 attempts)",
        teacher_solve_rate=0.6,
        student_before_solve_rate=0.3,
        student_after_solve_rate=0.5,
        min_teacher_fraction=0.7,
    )


# -- DistillRunStore ---------------------------------------------------------------------------


def test_snapshot_config_round_trips(tmp_path: Path) -> None:
    store = DistillRunStore(tmp_path / "run")
    path = store.snapshot_config(_config())
    assert path == store.config_path
    assert store.load_config() == _config()


def test_append_metrics_appends_rows_with_step(tmp_path: Path) -> None:
    store = DistillRunStore(tmp_path / "run")
    store.append_metrics(0, _MetricsRow(solve_rate=0.1, reverse_kl_per_token=2.0, usd=3.5))
    store.append_metrics(1, _MetricsRow(solve_rate=0.2, reverse_kl_per_token=1.5, usd=4.0))
    rows = store.read_metrics()
    assert [row["step"] for row in rows] == [0, 1]
    assert rows[0]["solve_rate"] == 0.1
    assert store.last_step() == 1
    assert store.budget_spent() == pytest.approx(7.5)


def test_metrics_helpers_on_fresh_run(tmp_path: Path) -> None:
    store = DistillRunStore(tmp_path / "run")
    assert store.read_metrics() == []
    assert store.last_step() is None
    assert store.budget_spent() == 0.0


def test_append_metrics_rejects_conflicting_step(tmp_path: Path) -> None:
    class _RowWithStep(BaseModel):
        step: int
        usd: float

    store = DistillRunStore(tmp_path / "run")
    store.append_metrics(3, _RowWithStep(step=3, usd=1.0))  # matching step is fine
    with pytest.raises(ValueError, match="carries step 4"):
        store.append_metrics(5, _RowWithStep(step=4, usd=1.0))


def test_corrupt_metrics_line_names_the_line(tmp_path: Path) -> None:
    store = DistillRunStore(tmp_path / "run")
    store.append_metrics(0, _MetricsRow(solve_rate=0.1, reverse_kl_per_token=2.0, usd=1.0))
    with store.metrics_path.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    with pytest.raises(ValueError, match="line 2"):
        store.read_metrics()


def test_spend_ledger_round_trips_and_updates(tmp_path: Path) -> None:
    store = DistillRunStore(tmp_path / "run")
    assert store.read_spend() is None  # fresh dir: no ledger yet
    store.write_spend(1.25)
    assert store.read_spend() == pytest.approx(1.25)
    store.write_spend(3.5)  # every charge replaces the cumulative total
    assert store.read_spend() == pytest.approx(3.5)


def test_spend_ledger_rejects_negative_totals(tmp_path: Path) -> None:
    store = DistillRunStore(tmp_path / "run")
    with pytest.raises(ValueError, match=">= 0"):
        store.write_spend(-0.01)


def test_corrupt_spend_ledger_is_actionable(tmp_path: Path) -> None:
    store = DistillRunStore(tmp_path / "run")
    store.write_spend(2.0)
    store.spend_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt spend ledger"):
        store.read_spend()


def test_write_eval_places_payload_under_evals(tmp_path: Path) -> None:
    store = DistillRunStore(tmp_path / "run")
    path = store.write_eval("baseline-teacher", _EvalReport(split="holdout", solve_rate=0.6))
    assert path == store.evals_dir / "baseline-teacher.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {"split": "holdout", "solve_rate": 0.6}


def test_write_eval_rejects_path_traversal_names(tmp_path: Path) -> None:
    store = DistillRunStore(tmp_path / "run")
    with pytest.raises(ValueError, match="invalid"):
        store.write_eval("../escape", _EvalReport(split="holdout", solve_rate=0.6))


def test_checkpoint_manifest_round_trips(tmp_path: Path) -> None:
    store = DistillRunStore(tmp_path / "run")
    assert store.latest_checkpoint() is None
    store.record_checkpoint(8, "tinker://fake/state/0", "tinker://fake/sampler/s/0")
    store.record_checkpoint(16, "tinker://fake/state/1", "tinker://fake/sampler/s/1")
    # A fresh store over the same dir reads the same manifest (the resume path).
    reopened = DistillRunStore(tmp_path / "run")
    latest = reopened.latest_checkpoint()
    assert latest is not None
    assert (latest.step, latest.state_path) == (16, "tinker://fake/state/1")
    assert [record.step for record in reopened.checkpoints()] == [8, 16]


def test_checkpoint_same_step_replaces_earlier_record(tmp_path: Path) -> None:
    store = DistillRunStore(tmp_path / "run")
    store.record_checkpoint(8, "tinker://fake/state/0", "tinker://fake/sampler/s/0")
    store.record_checkpoint(8, "tinker://fake/state/1", "tinker://fake/sampler/s/1")
    records = store.checkpoints()
    assert len(records) == 1
    assert records[0].state_path == "tinker://fake/state/1"


def test_gate_model_card_and_handoff_writes(tmp_path: Path) -> None:
    store = DistillRunStore(tmp_path / "run")
    store.write_gate(_gate_record())
    store.write_model_card(_card())
    store.write_handoff(build_handoff_toml("tinker://fake/sampler/final/0"))
    gate = DistillGateRecord.model_validate_json(store.gate_path.read_text(encoding="utf-8"))
    assert gate == _gate_record()
    card = DistillModelCard.model_validate_json(store.model_card_path.read_text(encoding="utf-8"))
    assert card.sampler_path == "tinker://fake/sampler/final/0"
    parsed = tomllib.loads(store.handoff_path.read_text(encoding="utf-8"))
    assert parsed["models"]["agent"]["provider"] == "openai"


def test_write_handoff_rejects_invalid_toml(tmp_path: Path) -> None:
    store = DistillRunStore(tmp_path / "run")
    with pytest.raises(ValueError, match="not valid TOML"):
        store.write_handoff("[models.agent\nbroken")


# -- AdapterStore ------------------------------------------------------------------------------


def test_save_version_assigns_incrementing_versions_and_champion(tmp_path: Path) -> None:
    store = AdapterStore(tmp_path)
    assert store.save_version("nano-distill", _card()) == 1
    assert store.save_version("nano-distill", _card("tinker://fake/sampler/final/1")) == 2
    assert store.versions("nano-distill") == [1, 2]
    assert store.aliases("nano-distill") == {CHAMPION_ALIAS: 2}
    loaded = store.resolve("nano-distill")
    assert (loaded.name, loaded.version) == ("nano-distill", 2)
    assert loaded.sampler_path == "tinker://fake/sampler/final/1"


def test_second_save_does_not_mutate_v1(tmp_path: Path) -> None:
    store = AdapterStore(tmp_path)
    store.save_version("a", _card())
    v1_card = store.dir_for("a") / "v1" / "model_card.json"
    before = v1_card.read_text(encoding="utf-8")
    store.save_version("a", _card("tinker://fake/sampler/final/9"))
    assert v1_card.read_text(encoding="utf-8") == before  # v1 untouched
    assert store.resolve("a", "v1").sampler_path == "tinker://fake/sampler/final/0"


def test_save_version_without_alias_keeps_champion(tmp_path: Path) -> None:
    store = AdapterStore(tmp_path)
    store.save_version("a", _card())
    store.save_version("a", _card("tinker://fake/sampler/final/1"), alias=None)
    assert store.aliases("a") == {CHAMPION_ALIAS: 1}
    assert store.resolve("a").version == 1  # champion wins over latest
    assert store.resolve("a", "2").version == 2


def test_rollback_is_repointing_the_alias(tmp_path: Path) -> None:
    store = AdapterStore(tmp_path)
    store.save_version("a", _card())
    store.save_version("a", _card("tinker://fake/sampler/final/1"))
    store.set_alias("a", CHAMPION_ALIAS, 1)
    assert store.resolve("a").version == 1


def test_unknown_adapter_ref_and_alias_are_friendly(tmp_path: Path) -> None:
    store = AdapterStore(tmp_path)
    with pytest.raises(FileNotFoundError, match="no adapter named"):
        store.resolve("ghost")
    store.save_version("a", _card())
    with pytest.raises(ValueError, match="no version v9"):
        store.resolve("a", "v9")
    with pytest.raises(ValueError, match="no version or alias"):
        store.resolve("a", "prod")
    with pytest.raises(ValueError, match="no version v9"):
        store.set_alias("a", CHAMPION_ALIAS, 9)


def test_list_names_ignores_dirs_without_versions(tmp_path: Path) -> None:
    store = AdapterStore(tmp_path)
    store.save_version("real", _card())
    (store.adapters_dir / "empty").mkdir(parents=True)
    assert store.list_names() == ["real"]
    assert store.exists("real")
    assert not store.exists("empty")


# -- handoff snippet ---------------------------------------------------------------------------


def test_handoff_snippet_content_and_default_endpoint() -> None:
    text = build_handoff_toml("tinker://runs/abc/sampler/final")
    parsed = tomllib.loads(text)
    assert parsed == {
        "models": {
            "agent": {
                "provider": "openai",
                "model": "tinker://runs/abc/sampler/final",
                "endpoint": DEFAULT_TINKER_OPENAI_ENDPOINT,
            }
        }
    }
    # The auth note rides along as TOML comments, naming the env keys.
    assert "WMH_ENDPOINT_API_KEY" in text
    assert "TINKER_API_KEY" in text


def test_handoff_snippet_honors_custom_endpoint() -> None:
    text = build_handoff_toml(
        "tinker://runs/abc/sampler/final", endpoint="http://localhost:8000/v1"
    )
    parsed = tomllib.loads(text)
    assert parsed["models"]["agent"]["endpoint"] == "http://localhost:8000/v1"


def test_handoff_rejects_non_tinker_sampler_path() -> None:
    with pytest.raises(ValueError, match="not a tinker://"):
        build_handoff_toml("s3://bucket/weights")


def test_handoff_rejects_unembeddable_values() -> None:
    with pytest.raises(ValueError, match="cannot be embedded"):
        build_handoff_toml('tinker://bad"path')
