"""Tests for the distillation run store, adapter store, and handoff snippet."""

from __future__ import annotations

import contextlib
import json
import threading
import tomllib
from pathlib import Path

import pytest
from pydantic import BaseModel

from wmo.distill.config import (
    DistillConfig,
    HarborConfig,
    StudentConfig,
    TeacherConfig,
)
from wmo.distill.gate import DistillGateRecord
from wmo.distill.loop import StepMetrics
from wmo.distill.store import (
    CHAMPION_ALIAS,
    DEFAULT_TINKER_OPENAI_ENDPOINT,
    AdapterStore,
    DistillModelCard,
    DistillRunStore,
    WarmupRecord,
    WarmupTrialsManifest,
    build_handoff_toml,
    is_tinker_endpoint,
    student_pool_entry,
)
from wmo.distill.tokens import TrialRecord
from wmo.distill.tripwire import TripwireBaseline
from wmo.providers.base import ProviderKind
from wmo.providers.pool import load_pool, upsert_pool_entry
from wmo.providers.tinker import TokenSpan


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


def test_step_metrics_row_round_trips_the_rl_and_cost_fields(tmp_path: Path) -> None:
    """The real StepMetrics shape (cumulative_usd, RL metrics included) persists."""
    metrics = StepMetrics(
        tasks=2,
        trials=4,
        solve_rate=0.5,
        loss="ppo",
        empty_span_trials=0,
        datums=4,
        fragments=0,
        fragmentation_rate=0.0,
        overflow_drops=0,
        overlong_drops=0,
        mismatch_drops=0,
        clipped_tokens=3,
        loss_tokens=40,
        context_tokens=200,
        reverse_kl_per_token=-0.25,
        entropy_per_token=0.181,
        mean_generation_tokens=7577.0,
        entropy_baseline=0.2,
        entropy_ratio=0.905,
        generation_tokens_baseline=8000.0,
        generation_tokens_ratio=0.947,
        reward_mean=0.5,
        advantage_mean=0.05,
        advantage_std=1.2,
        clip_fraction=0.075,
        pg_loss=1.5,
        grad_norm=None,
        sampler_path="tinker://fake/sampler/0001",
        student_prefill_tokens=120,
        student_cached_prefill_tokens=30,
        student_sample_tokens=40,
        student_train_tokens=200,
        teacher_prefill_tokens=160,
        teacher_cached_prefill_tokens=0,
        teacher_sample_tokens=0,
        usd=0.75,
        cumulative_usd=3.25,
    )
    store = DistillRunStore(tmp_path / "run")
    store.append_metrics(4, metrics)
    (row,) = store.read_metrics()
    assert row == {"step": 4, **metrics.model_dump(mode="json")}
    assert row["cumulative_usd"] == 3.25
    assert row["reward_mean"] == 0.5
    # The degeneration pair persists with its baseline and ratio, so a row read
    # back months later still says what the collapse was measured against.
    assert row["entropy_per_token"] == 0.181
    assert row["mean_generation_tokens"] == 7577.0
    assert row["entropy_baseline"] == 0.2
    assert row["entropy_ratio"] == 0.905
    assert row["generation_tokens_baseline"] == 8000.0
    assert row["generation_tokens_ratio"] == 0.947
    # The objective persists in the row: advantage_mean and clip_fraction are
    # only interpretable next to the loss that produced them.
    assert row["loss"] == "ppo"
    assert row["clip_fraction"] == 0.075
    assert row["pg_loss"] == 1.5
    assert row["grad_norm"] is None  # unreported backend metrics persist as null


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


def test_a_tolerant_read_drops_only_a_half_written_last_line(tmp_path: Path) -> None:
    """What a run killed mid-append leaves: a torn tail a read-only reader must survive."""
    store = DistillRunStore(tmp_path / "run")
    store.append_metrics(0, _MetricsRow(solve_rate=0.1, reverse_kl_per_token=2.0, usd=1.0))
    with store.metrics_path.open("a", encoding="utf-8") as handle:
        handle.write('{"step": 1, "solve_')

    rows = store.read_metrics(tolerate_partial_tail=True)

    assert [row["step"] for row in rows] == [0]
    with pytest.raises(ValueError, match="line 2"):
        store.read_metrics()  # strict is still the default, for everything that resumes a run


def test_a_tolerant_read_still_refuses_damage_above_the_last_line(tmp_path: Path) -> None:
    """A broken row with complete rows after it means content was lost, not a torn write."""
    store = DistillRunStore(tmp_path / "run")
    store.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    store.metrics_path.write_text('{"step": 0}\n[1, 2, 3]\n{"step": 2}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        store.read_metrics(tolerate_partial_tail=True)


@pytest.mark.parametrize("tail", ["[]", "null", '"step 1"', "[1, 2, 3]"])
def test_a_tolerant_read_still_refuses_a_last_line_that_parses(tmp_path: Path, tail: str) -> None:
    """A whole non-object value is not a torn append: every prefix of `{...}` fails to PARSE.

    So a final line json.loads accepts was written whole, and dropping it would hide real
    corruption while reporting an older step as the run's latest state.
    """
    store = DistillRunStore(tmp_path / "run")
    store.append_metrics(0, _MetricsRow(solve_rate=0.1, reverse_kl_per_token=2.0, usd=1.0))
    with store.metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{tail}\n")

    with pytest.raises(ValueError, match="line 2.*expected a JSON object"):
        store.read_metrics(tolerate_partial_tail=True)


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


def test_warmup_record_round_trips(tmp_path: Path) -> None:
    store = DistillRunStore(tmp_path / "run")
    assert store.read_warmup() is None  # fresh dir: warmup never finished
    record = WarmupRecord(
        steps=2,
        trials=16,
        kept_trials=4,
        datums=4,
        state_path="tinker://fake/state/1",
        sampler_path="tinker://fake/sampler/warmup/0",
    )
    path = store.write_warmup(record)
    assert path == store.warmup_path
    assert store.read_warmup() == record


def test_warmup_record_skip_shape(tmp_path: Path) -> None:
    store = DistillRunStore(tmp_path / "run")
    store.write_warmup(
        WarmupRecord(steps=0, trials=8, kept_trials=0, datums=0, skipped_reason="0 passing")
    )
    read = store.read_warmup()
    assert read is not None
    assert read.skipped_reason == "0 passing"
    assert read.state_path is None and read.sampler_path is None


def test_corrupt_warmup_record_is_actionable(tmp_path: Path) -> None:
    store = DistillRunStore(tmp_path / "run")
    store.write_warmup(WarmupRecord(steps=1, trials=1, kept_trials=1, datums=1))
    store.warmup_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt warmup record"):
        store.read_warmup()


def _trial_record(task_id: str, *, passed: bool) -> TrialRecord:
    return TrialRecord(
        task_id=task_id,
        attempt=1,
        trial_name=f"{task_id}__s1",
        reward=1.0 if passed else 0.0,
        passed=passed,
        spans=[
            TokenSpan(
                call_index=0,
                prompt_token_ids=[65, 66],
                sampled_token_ids=[67, 68],
                sampled_logprobs=[-0.5, -0.25],
            )
        ],
        stop_reason="submitted",
        artifact_dir=f"/trials/{task_id}",
    )


def test_warmup_trials_manifest_round_trips(tmp_path: Path) -> None:
    store = DistillRunStore(tmp_path / "run")
    assert store.read_warmup_trials() is None  # fresh dir: nothing collected
    manifest = WarmupTrialsManifest(
        teacher_model="Qwen/Qwen3-235B-A22B-Instruct-2507",
        records=[_trial_record("task-a", passed=True), _trial_record("task-b", passed=False)],
    )
    path = store.write_warmup_trials(manifest)
    assert path == store.warmup_trials_path
    assert store.read_warmup_trials() == manifest


def test_corrupt_warmup_trials_manifest_is_actionable(tmp_path: Path) -> None:
    store = DistillRunStore(tmp_path / "run")
    store.write_warmup_trials(
        WarmupTrialsManifest(teacher_model="teacher", records=[_trial_record("t", passed=True)])
    )
    store.warmup_trials_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt warmup trial manifest"):
        store.read_warmup_trials()


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


def test_write_samples_places_markdown_under_samples(tmp_path: Path) -> None:
    store = DistillRunStore(tmp_path / "run")
    path = store.write_samples("step-0000", "### trial t\nepisode text\n")
    assert path == store.samples_dir / "step-0000.md"
    assert path.read_text(encoding="utf-8") == "### trial t\nepisode text\n"
    # A rewrite replaces the file whole (tmp plus atomic replace, no append).
    store.write_samples("step-0000", "replaced\n")
    assert path.read_text(encoding="utf-8") == "replaced\n"
    assert sorted(p.name for p in store.samples_dir.iterdir()) == ["step-0000.md"]


def test_write_samples_rejects_path_traversal_names(tmp_path: Path) -> None:
    store = DistillRunStore(tmp_path / "run")
    with pytest.raises(ValueError, match="invalid"):
        store.write_samples("../escape", "boom")


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


def test_tripwire_baseline_survives_in_the_manifest_beside_checkpoints(tmp_path: Path) -> None:
    """The baseline must survive `--resume`, so it rides the run manifest, and
    the checkpoint cadence must not clobber it (nor it the checkpoints)."""
    store = DistillRunStore(tmp_path / "run")
    assert store.read_tripwire_baseline() is None
    baseline = TripwireBaseline(
        step=0,
        entropy_per_token=0.181,
        mean_generation_tokens=7577.0,
        episodes=47,
        sampled_tokens=356122,
    )
    store.write_tripwire_baseline(baseline)
    store.record_checkpoint(4, "tinker://fake/state/0", "tinker://fake/sampler/s/0")

    reopened = DistillRunStore(tmp_path / "run")
    assert reopened.read_tripwire_baseline() == baseline
    assert [record.step for record in reopened.checkpoints()] == [4]


def test_a_manifest_without_a_baseline_still_loads(tmp_path: Path) -> None:
    """Run dirs written before the tripwires existed resume unchanged."""
    store = DistillRunStore(tmp_path / "run")
    store.run_dir.mkdir(parents=True)
    store.checkpoints_path.write_text(
        json.dumps(
            {
                "checkpoints": [
                    {
                        "step": 2,
                        "state_path": "tinker://fake/state/0",
                        "sampler_path": "tinker://fake/sampler/s/0",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert store.read_tripwire_baseline() is None
    latest = store.latest_checkpoint()
    assert latest is not None
    assert latest.step == 2


def test_gate_model_card_and_handoff_writes(tmp_path: Path) -> None:
    store = DistillRunStore(tmp_path / "run")
    store.write_gate(_gate_record())
    store.write_model_card(_card())
    store.write_handoff(
        build_handoff_toml("tinker://fake/sampler/final/0", base_model="Qwen/Qwen3-4B")
    )
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


def test_concurrent_adapter_alias_moves_both_land(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two promotions of DIFFERENT aliases must not erase each other.

    The write here was already atomic, which is the wrong half for this failure: neither write is
    torn, the second simply overwrites a table the first had already added a key to, and both
    calls return normally. The race is forced with a barrier on every read of `aliases.toml`, so
    both writers provably hold the same snapshot; with the lock held the second cannot reach its
    read until the first is done, the barrier times out, and the reads happen in sequence.
    """
    store = AdapterStore(tmp_path)
    store.save_version("a", _card())
    store.save_version("a", _card("tinker://fake/sampler/final/1"))
    read_together = threading.Barrier(2, timeout=1.0)
    real_read = Path.read_text

    def _park_on_every_alias_read(self: Path, **kwargs: object) -> str:
        text = real_read(self, **kwargs)  # ty: ignore[invalid-argument-type]
        if self.name == "aliases.toml":
            with contextlib.suppress(threading.BrokenBarrierError):
                read_together.wait()
        return text

    monkeypatch.setattr(Path, "read_text", _park_on_every_alias_read)
    failures: list[BaseException] = []

    def _promote(alias: str, version: int) -> None:
        try:
            store.set_alias("a", alias, version)
        except BaseException as exc:  # noqa: BLE001 - reported by the assertions below
            failures.append(exc)

    writers = [
        threading.Thread(target=_promote, args=args)
        for args in ((CHAMPION_ALIAS, 2), ("staging", 1))
    ]
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join(timeout=30)
        assert not writer.is_alive(), "the alias lock wait is not bounded"
    monkeypatch.undo()  # the assertion below reads the table without parking on the barrier

    assert failures == []
    assert store.aliases("a") == {CHAMPION_ALIAS: 2, "staging": 1}


def test_a_corrupt_adapter_alias_file_names_itself(tmp_path: Path) -> None:
    """The user never edited this file, so a bare `tomllib` message gives them nowhere to start."""
    store = AdapterStore(tmp_path)
    store.save_version("a", _card())
    path = store.dir_for("a") / "aliases.toml"
    path.write_text("[aliases]\nchampion = \n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"is not valid TOML") as caught:
        store.aliases("a")

    assert str(path) in str(caught.value)


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
    text = build_handoff_toml("tinker://runs/abc/sampler/final", base_model="Qwen/Qwen3-4B")
    parsed = tomllib.loads(text)
    assert parsed == {
        "models": {
            "agent": {
                "provider": "openai",
                "model": "tinker://runs/abc/sampler/final",
                "model_type": "Qwen/Qwen3-4B",
                "endpoint": DEFAULT_TINKER_OPENAI_ENDPOINT,
                # Added deliberately, not loosened: a tinker:// path is outside the built-in
                # catalog, so capability resolution falls back to `max_completion_tokens`, which
                # Tinker's endpoint answers with a 400 on every call. The snippet has to pin the
                # name that endpoint takes or the promoted student is unusable.
                "chat_max_tokens_field": "max_tokens",
            }
        }
    }
    # The auth note rides along as TOML comments, naming the env keys.
    assert "WMO_ENDPOINT_API_KEY" in text
    assert "TINKER_API_KEY" in text


def test_handoff_snippet_honors_custom_endpoint() -> None:
    text = build_handoff_toml(
        "tinker://runs/abc/sampler/final",
        base_model="Qwen/Qwen3-4B",
        endpoint="http://localhost:8000/v1",
    )
    parsed = tomllib.loads(text)
    assert parsed["models"]["agent"]["endpoint"] == "http://localhost:8000/v1"


def test_handoff_rejects_non_tinker_sampler_path() -> None:
    with pytest.raises(ValueError, match="not a tinker://"):
        build_handoff_toml("s3://bucket/weights", base_model="Qwen/Qwen3-4B")


def test_handoff_rejects_unembeddable_values() -> None:
    with pytest.raises(ValueError, match="cannot be embedded"):
        build_handoff_toml('tinker://bad"path', base_model="Qwen/Qwen3-4B")


def test_student_pool_entry_addresses_the_adapter_as_an_openai_endpoint() -> None:
    entry = student_pool_entry(_card(), name="student", input_per_mtok=0.1, output_per_mtok=0.4)

    assert entry.name == "student"
    assert entry.kind is ProviderKind.OPENAI
    assert entry.model == "tinker://fake/sampler/final/0"  # the weights, not a family name
    assert entry.model_type == "Qwen/Qwen3-8B"  # capability resolution keys on the base model
    assert entry.endpoint == DEFAULT_TINKER_OPENAI_ENDPOINT
    assert entry.api_key_env == "TINKER_API_KEY"
    assert entry.tier == "open"


def test_student_pool_entry_asks_for_the_classic_output_budget_field() -> None:
    """Tinker's OpenAI-compatible endpoint 400s on `max_completion_tokens`.

    A `tinker://` path is not in the built-in catalog, so nothing else can supply this: without
    the explicit field every routed call to the student fails.
    """
    entry = student_pool_entry(_card(), name="s", input_per_mtok=0.1, output_per_mtok=0.4)

    assert entry.chat_max_tokens_field == "max_tokens"
    assert entry.provider_config().resolved_chat_max_tokens_field() == "max_tokens"


def test_student_pool_entry_prices_are_required_and_carried() -> None:
    """An unpriced candidate reports $0 and a cost-aware policy would route everything to it."""
    entry = student_pool_entry(_card(), name="s", input_per_mtok=0.1, output_per_mtok=0.4)

    assert entry.price().input_per_mtok == 0.1
    assert entry.price().output_per_mtok == 0.4
    with pytest.raises(TypeError):  # no default: the caller must state a price
        student_pool_entry(_card(), name="s")  # ty: ignore[missing-argument]


def test_student_pool_entry_honors_an_endpoint_override() -> None:
    entry = student_pool_entry(
        _card(),
        name="s",
        input_per_mtok=0.1,
        output_per_mtok=0.4,
        endpoint="https://self-hosted.example/v1",
    )

    assert entry.endpoint == "https://self-hosted.example/v1"


def test_student_pool_entry_rejects_a_non_tinker_sampler_path() -> None:
    with pytest.raises(ValueError, match="tinker://"):
        student_pool_entry(
            _card(sampler="Qwen/Qwen3-8B"), name="s", input_per_mtok=0.1, output_per_mtok=0.4
        )


def test_student_pool_entry_round_trips_through_the_pool_file(tmp_path: Path) -> None:
    """The end of the keystone path: card -> entry -> pool.toml -> loadable candidate."""
    path = tmp_path / "pool.toml"
    entry = student_pool_entry(_card(), name="student", input_per_mtok=0.1, output_per_mtok=0.4)

    upsert_pool_entry(entry, path)

    assert load_pool(path).entry("student") == entry


def test_student_pool_entry_never_sends_the_tinker_key_to_another_host() -> None:
    """A custom --endpoint must not inherit TINKER_API_KEY.

    `pool_provider` resolves `api_key_env` and hands the value straight to the client as an
    explicit credential, so inheriting the default here would send a Tinker bearer token to
    whatever host the caller named.
    """
    entry = student_pool_entry(
        _card(),
        name="s",
        input_per_mtok=0.1,
        output_per_mtok=0.4,
        endpoint="https://someone-elses-host.example/v1",
    )

    assert entry.api_key_env is None  # falls back to WMO_ENDPOINT_API_KEY in the provider
    assert entry.chat_max_tokens_field == "max_completion_tokens"  # not Tinker's convention


def test_student_pool_entry_takes_an_explicit_credential_for_a_custom_host() -> None:
    entry = student_pool_entry(
        _card(),
        name="s",
        input_per_mtok=0.1,
        output_per_mtok=0.4,
        endpoint="https://my-vllm.example/v1",
        api_key_env="MY_VLLM_KEY",
        chat_max_tokens_field="max_tokens",
    )

    assert entry.api_key_env == "MY_VLLM_KEY"
    assert entry.chat_max_tokens_field == "max_tokens"


def test_student_pool_entry_keeps_the_tinker_defaults_on_an_explicit_tinker_endpoint() -> None:
    """Naming Tinker's endpoint by hand is still Tinker, so the defaults must still apply."""
    entry = student_pool_entry(
        _card(),
        name="s",
        input_per_mtok=0.1,
        output_per_mtok=0.4,
        endpoint=DEFAULT_TINKER_OPENAI_ENDPOINT,
    )

    assert entry.api_key_env == "TINKER_API_KEY"
    assert entry.chat_max_tokens_field == "max_tokens"


@pytest.mark.parametrize(
    "endpoint",
    [
        DEFAULT_TINKER_OPENAI_ENDPOINT,
        DEFAULT_TINKER_OPENAI_ENDPOINT + "/",
        DEFAULT_TINKER_OPENAI_ENDPOINT + "//",
        f"  {DEFAULT_TINKER_OPENAI_ENDPOINT}  ",
    ],
)
def test_equivalent_spellings_of_the_tinker_endpoint_keep_its_defaults(endpoint: str) -> None:
    """A pasted trailing slash must not reclassify Tinker's own endpoint as somebody else's host.

    Getting this wrong is silent: the entry would swap TINKER_API_KEY for the
    WMO_ENDPOINT_API_KEY fallback and `max_tokens` for `max_completion_tokens`, so the student
    400s on a URL that is Tinker's.
    """
    entry = student_pool_entry(
        _card(), name="s", input_per_mtok=0.1, output_per_mtok=0.4, endpoint=endpoint
    )

    assert entry.api_key_env == "TINKER_API_KEY"
    assert entry.chat_max_tokens_field == "max_tokens"
    parsed = tomllib.loads(
        build_handoff_toml("tinker://w/1", base_model="Qwen/Qwen3-8B", endpoint=endpoint)
    )
    assert parsed["models"]["agent"]["chat_max_tokens_field"] == "max_tokens"


def test_a_genuinely_different_host_is_not_read_as_tinker() -> None:
    """The normalization must not become a prefix match: a lookalike host is still custom."""
    entry = student_pool_entry(
        _card(),
        name="s",
        input_per_mtok=0.1,
        output_per_mtok=0.4,
        endpoint=DEFAULT_TINKER_OPENAI_ENDPOINT + "/proxy",
    )

    assert entry.api_key_env is None
    assert entry.chat_max_tokens_field == "max_completion_tokens"


@pytest.mark.parametrize(
    "endpoint",
    [
        "HTTPS://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1",
        "https://Tinker.ThinkingMachines.Dev/services/tinker-prod/oai/api/v1",
        "https://TINKER.THINKINGMACHINES.DEV/services/tinker-prod/oai/api/v1/",
    ],
)
def test_case_variant_tinker_urls_keep_its_defaults(endpoint: str) -> None:
    """Scheme and host are case-insensitive per RFC 3986, so these are all Tinker's endpoint."""
    entry = student_pool_entry(
        _card(), name="s", input_per_mtok=0.1, output_per_mtok=0.4, endpoint=endpoint
    )

    assert entry.api_key_env == "TINKER_API_KEY"
    assert entry.chat_max_tokens_field == "max_tokens"


def test_a_case_variant_PATH_is_not_read_as_tinker() -> None:
    """The path is case-SENSITIVE, so a different route is a different resource, not Tinker's.

    Treating it as Tinker's would hand `TINKER_API_KEY` to a route the operator never named.
    """
    entry = student_pool_entry(
        _card(),
        name="s",
        input_per_mtok=0.1,
        output_per_mtok=0.4,
        endpoint="https://tinker.thinkingmachines.dev/services/tinker-prod/OAI/API/V1",
    )

    assert entry.api_key_env is None
    assert entry.chat_max_tokens_field == "max_completion_tokens"


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://tinker.thinkingmachines.dev:443/services/tinker-prod/oai/api/v1",
        "https://tinker.thinkingmachines.dev:443/services/tinker-prod/oai/api/v1/",
        "  HTTPS://TINKER.THINKINGMACHINES.DEV:443/services/tinker-prod/oai/api/v1  ",
    ],
)
def test_the_spelled_out_default_https_port_keeps_the_tinker_defaults(endpoint: str) -> None:
    """`:443` on an https URL is redundant (RFC 3986 section 3.2.3), so this IS Tinker's endpoint.

    A URL copied out of a proxy config or a log line often carries the port. Reading it as a
    custom host is silent: the entry swaps TINKER_API_KEY for the WMO_ENDPOINT_API_KEY fallback
    and `max_tokens` for `max_completion_tokens`, so every routed call to the student fails auth
    or 400s on a URL that is Tinker's own.
    """
    assert is_tinker_endpoint(endpoint)
    entry = student_pool_entry(
        _card(), name="s", input_per_mtok=0.1, output_per_mtok=0.4, endpoint=endpoint
    )

    assert entry.api_key_env == "TINKER_API_KEY"
    assert entry.chat_max_tokens_field == "max_tokens"
    parsed = tomllib.loads(
        build_handoff_toml("tinker://w/1", base_model="Qwen/Qwen3-8B", endpoint=endpoint)
    )
    assert parsed["models"]["agent"]["chat_max_tokens_field"] == "max_tokens"


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://tinker.thinkingmachines.dev:8443/services/tinker-prod/oai/api/v1",
        # Not parseable as the default port, so it is compared as written: guessing what an
        # operator meant is the wrong side to err on when a credential rides on the answer.
        "https://tinker.thinkingmachines.dev:0443/services/tinker-prod/oai/api/v1",
        "https://tinker.thinkingmachines.dev:notaport/services/tinker-prod/oai/api/v1",
    ],
)
def test_a_port_that_is_not_the_scheme_default_is_a_different_service(endpoint: str) -> None:
    """Only the redundant port may be dropped: `:8443` on that host is a proxy, not Tinker."""
    assert not is_tinker_endpoint(endpoint)  # must also not raise on an unparseable port
    entry = student_pool_entry(
        _card(), name="s", input_per_mtok=0.1, output_per_mtok=0.4, endpoint=endpoint
    )

    assert entry.api_key_env is None
    assert entry.chat_max_tokens_field == "max_completion_tokens"


def test_only_the_schemes_own_default_port_is_redundant() -> None:
    """http implies 80 and https implies 443; each other combination is a distinct endpoint."""
    from wmo.distill.store import _normalize_endpoint

    assert _normalize_endpoint("http://h.example/v1") == _normalize_endpoint(
        "http://h.example:80/v1"
    )
    assert _normalize_endpoint("https://h.example/v1") == _normalize_endpoint(
        "https://h.example:443/v1"
    )
    # 443 is not http's default port, nor 80 https's: those name a real port on that host.
    assert _normalize_endpoint("http://h.example/v1") != _normalize_endpoint(
        "http://h.example:443/v1"
    )
    assert _normalize_endpoint("https://h.example/v1") != _normalize_endpoint(
        "https://h.example:80/v1"
    )


@pytest.mark.parametrize("padded", ["  {}  ", "\t{}\n", "{} "])
def test_a_padded_endpoint_is_stripped_before_it_is_stored(padded: str) -> None:
    """Classification tolerates whitespace, so storage must remove it, not preserve it.

    The entry's endpoint becomes the OpenAI client's `base_url` verbatim. Persisting the padded
    string would pick Tinker's defaults correctly and then fail every routed call on a URL the
    client cannot use.
    """
    entry = student_pool_entry(
        _card(),
        name="s",
        input_per_mtok=0.1,
        output_per_mtok=0.4,
        endpoint=padded.format(DEFAULT_TINKER_OPENAI_ENDPOINT),
    )

    assert entry.endpoint == DEFAULT_TINKER_OPENAI_ENDPOINT
    assert entry.api_key_env == "TINKER_API_KEY"  # still classified as Tinker


def test_a_padded_custom_endpoint_is_also_stripped() -> None:
    entry = student_pool_entry(
        _card(),
        name="s",
        input_per_mtok=0.1,
        output_per_mtok=0.4,
        endpoint="  https://my-vllm.example/v1  ",
    )

    assert entry.endpoint == "https://my-vllm.example/v1"
    assert entry.api_key_env is None  # and still a custom host


def test_a_padded_endpoint_is_stripped_in_the_handoff_snippet() -> None:
    text = build_handoff_toml(
        "tinker://w/1", base_model="Qwen/Qwen3-8B", endpoint=f"  {DEFAULT_TINKER_OPENAI_ENDPOINT} "
    )
    parsed = tomllib.loads(text)

    assert parsed["models"]["agent"]["endpoint"] == DEFAULT_TINKER_OPENAI_ENDPOINT
    assert parsed["models"]["agent"]["chat_max_tokens_field"] == "max_tokens"
