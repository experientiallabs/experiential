"""Tests for the distillation run config: loading, validators, snapshotting."""

import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from wmo.distill.config import (
    PROBE_BASELINE_ENTROPY_NATS,
    PROBE_BASELINE_EPISODE_TOKENS,
    DistillConfig,
    EvalConfig,
    HarborConfig,
    PricingConfig,
    StudentConfig,
    TeacherConfig,
    TrainConfig,
    TripwireConfig,
    WandbConfig,
    WarmupConfig,
    load_distill_config,
    snapshot_toml,
)

MINIMAL_TOML = """
[student]
base_model = "Qwen/Qwen3-8B"

[teacher]
model = "Qwen/Qwen3-235B-A22B-Instruct-2507"

[harbor]
job_template = "jobs/tb2.yaml"
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "distill.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _minimal_config() -> DistillConfig:
    return DistillConfig(
        student=StudentConfig(base_model="Qwen/Qwen3-8B"),
        teacher=TeacherConfig(model="Qwen/Qwen3-235B-A22B-Instruct-2507"),
        harbor=HarborConfig(job_template="jobs/tb2.yaml"),
    )


def test_load_minimal_applies_defaults(tmp_path: Path) -> None:
    cfg = load_distill_config(_write(tmp_path, MINIMAL_TOML))
    assert cfg.student.base_model == "Qwen/Qwen3-8B"
    assert cfg.student.lora_rank == 32
    assert cfg.teacher.backend == "tinker"
    assert cfg.teacher.checkpoint is None
    assert cfg.rollout_source == "harbor"
    assert cfg.harbor is not None
    assert cfg.harbor.backend == "local"
    assert cfg.harbor.reward_key == "reward"
    # 100, not 20: at 20 the cap fired mid-tool-call on 45% of Ultra TerminalBench-2 trials and
    # scored every one of them 0, and the reference terminus-2 agent is effectively unbounded.
    assert cfg.rollout.max_turns == 100
    # 1800s, not the 300s evaluation default every rollout used to inherit silently (which ended
    # 31% of Super trials on the wall clock).
    assert cfg.rollout.episode_timeout_s == pytest.approx(1800.0)
    assert cfg.rollout.context_budget_tokens == 65536
    assert cfg.rollout.compaction is False
    assert cfg.train.steps == 40
    assert cfg.train.tasks_per_batch == 8
    assert cfg.train.group_size == 4
    assert cfg.train.learning_rate == pytest.approx(1e-4)
    # The default objective is the OpenClaw-RL / Slime form: the RAW per-token
    # teacher-minus-student gap, unclipped and uncentered, regularized by the
    # loss's own ratio clipping rather than by reshaping the advantage.
    assert cfg.train.advantage_clip is None
    assert cfg.train.center_advantages is False
    assert cfg.train.max_datum_tokens == 65536
    assert cfg.train.sampler_refresh_every == 1
    assert cfg.train.save_state_every == 8
    assert cfg.train.trial_concurrency == 8
    # 1.0 keeps issued sampler logprobs comparable to untempered teacher logprobs.
    assert cfg.sampling.temperature == pytest.approx(1.0)
    assert cfg.sampling.max_tokens == 8192
    assert cfg.warmup.steps == 0
    assert cfg.warmup.rollouts_per_task == 1
    assert cfg.warmup.keep == "passed"
    assert cfg.warmup.learning_rate is None
    # Interim evals default OFF: the holdout gate is the run's measurement.
    assert cfg.eval.every == 0
    assert cfg.eval.tasks == 12
    assert cfg.eval.k == 1
    assert cfg.eval.teacher_baseline_from is None
    assert cfg.eval.student_baseline_from is None
    assert cfg.gate.k == 3
    assert cfg.gate.min_teacher_fraction == pytest.approx(0.7)
    assert cfg.gate.require_no_regression is True
    assert cfg.pricing.student_prefill is None
    assert cfg.budget.max_usd is None
    # Degeneration tripwires are ON by default and every bound is a FRACTION of
    # the baseline the run measures itself. Absolutes are the trap: our healthy
    # untrained entropy is 0.181 nats/token, under the sibling lane's absolute
    # 0.2 "collapse" floor, so an absolute rule would fire at step 0.
    assert cfg.tripwire.enabled is True
    assert cfg.tripwire.entropy_warn_frac == pytest.approx(0.5)
    assert cfg.tripwire.entropy_kill_frac == pytest.approx(0.3)
    assert cfg.tripwire.length_warn_frac == pytest.approx(0.5)
    assert cfg.tripwire.length_kill_frac == pytest.approx(0.25)
    assert cfg.tripwire.kill_consecutive_steps == 2
    assert cfg.wandb.enabled is False
    assert cfg.wandb.project == "wmo-distill"
    assert cfg.wandb.entity is None
    assert cfg.wandb.run_name is None
    assert cfg.wandb.tags == []


def test_load_full_overrides(tmp_path: Path) -> None:
    text = """
[student]
base_model = "Qwen/Qwen3-4B"
lora_rank = 8

[teacher]
backend = "tinker"
model = "big-teacher"
checkpoint = "tinker://run/weights/ck-7"

[harbor]
job_template = "jobs/tb2.json"
backend = "e2b"
reward_key = "score"

[rollout]
max_turns = 5
context_budget_tokens = 2048

[train]
steps = 2
tasks_per_batch = 1
group_size = 2
learning_rate = 5e-5
advantage_clip = 2.0
center_advantages = false
max_datum_tokens = 4096
sampler_refresh_every = 2
save_state_every = 1
trial_concurrency = 4

[sampling]
temperature = 0.0
max_tokens = 128

[warmup]
steps = 2
rollouts_per_task = 3
keep = "all"
learning_rate = 2e-5

[eval]
every = 5
tasks = 2
k = 2
teacher_baseline_from = "runs/prior/evals/baseline-teacher.json"
student_baseline_from = "runs/prior/evals/baseline-student-before.json"

[gate]
k = 1
min_teacher_fraction = 1.0
require_no_regression = false

[pricing]
student_prefill = 0.1
student_cached_prefill = 0.03
student_sample = 0.4
student_train = 0.6
teacher_prefill = 1.2
teacher_cached_prefill = 0.3
teacher_sample = 3.0

[budget]
max_usd = 50.0

[wandb]
enabled = true
project = "my-distill"
entity = "my-team"
run_name = "smoke-01"
tags = ["smoke", "tb2"]
"""
    cfg = load_distill_config(_write(tmp_path, text))
    assert cfg.student.lora_rank == 8
    assert cfg.teacher.checkpoint == "tinker://run/weights/ck-7"
    assert cfg.harbor is not None
    assert cfg.harbor.backend == "e2b"
    assert cfg.harbor.reward_key == "score"
    assert cfg.rollout.max_turns == 5
    assert cfg.train.advantage_clip == pytest.approx(2.0)
    assert cfg.train.center_advantages is False
    assert cfg.sampling.temperature == 0.0
    assert cfg.warmup.steps == 2
    assert cfg.warmup.rollouts_per_task == 3
    assert cfg.warmup.keep == "all"
    assert cfg.warmup.learning_rate == pytest.approx(2e-5)
    assert cfg.eval.every == 5
    assert cfg.eval.teacher_baseline_from == "runs/prior/evals/baseline-teacher.json"
    assert cfg.eval.student_baseline_from == "runs/prior/evals/baseline-student-before.json"
    assert cfg.gate.min_teacher_fraction == 1.0
    assert cfg.pricing.student_prefill == pytest.approx(0.1)
    assert cfg.pricing.student_sample == pytest.approx(0.4)
    assert cfg.pricing.student_train == pytest.approx(0.6)
    assert cfg.pricing.teacher_prefill == pytest.approx(1.2)
    # Explicit cached rates win over the 20%-of-prefill derivation.
    assert cfg.pricing.effective_student_cached_prefill == pytest.approx(0.03)
    assert cfg.pricing.effective_teacher_cached_prefill == pytest.approx(0.3)
    assert cfg.pricing.teacher_sample == pytest.approx(3.0)
    assert cfg.budget.max_usd == pytest.approx(50.0)
    assert cfg.wandb.enabled is True
    assert cfg.wandb.project == "my-distill"
    assert cfg.wandb.entity == "my-team"
    assert cfg.wandb.run_name == "smoke-01"
    assert cfg.wandb.tags == ["smoke", "tb2"]


def test_compaction_rejected_with_a_verbatim_renderer(tmp_path: Path) -> None:
    """The one combination that silently corrupts: merge an episode whose prefix moved."""
    text = MINIMAL_TOML + (
        "\n[rollout]\ncompaction = true\n\n"
        '[rollout.renderers]\n"Qwen/Qwen3-8B" = "wmo/qwen3_verbatim"\n'
    )
    with pytest.raises(ValueError, match="prefix property"):
        load_distill_config(_write(tmp_path, text))


def test_compaction_allowed_with_a_strip_history_renderer(tmp_path: Path) -> None:
    """Compaction costs nothing once the episode is already one datum per turn.

    Leaving it off is what let a context overflow FAIL the trial instead of
    ending the episode, which took 24% of teacher trials out of the denominator
    rather than scoring them zero.
    """
    text = MINIMAL_TOML + (
        "\n[rollout]\ncompaction = true\n\n"
        '[rollout.renderers]\n"Qwen/Qwen3-8B" = "wmo/qwen3_5_strip_history"\n'
    )
    assert load_distill_config(_write(tmp_path, text)).rollout.compaction is True


def test_defer_baselines_requires_an_imported_student_before(tmp_path: Path) -> None:
    """Deferred, student-before runs after the LoRA moved -- it MUST be imported, not measured."""
    text = MINIMAL_TOML + "\n[eval]\ndefer_baselines = true\n"
    with pytest.raises(ValueError, match="student_baseline_from"):
        load_distill_config(_write(tmp_path, text))


def test_defer_baselines_accepted_with_a_student_baseline(tmp_path: Path) -> None:
    """The teacher may be measured late (fixed base model); the student may not."""
    text = MINIMAL_TOML + (
        "\n[eval]\ndefer_baselines = true\n"
        'student_baseline_from = "runs/base/evals/baseline-student-before.json"\n'
    )
    cfg = load_distill_config(_write(tmp_path, text))
    assert cfg.eval.defer_baselines is True


def test_compaction_defaults_off(tmp_path: Path) -> None:
    """Off unless asked for: it is a real change to what the model sees."""
    assert load_distill_config(_write(tmp_path, MINIMAL_TOML)).rollout.compaction is False


@pytest.mark.parametrize("steps", [0, 1, 5])
def test_warmup_steps_accepted(tmp_path: Path, steps: int) -> None:
    # Warmup was config-reserved in v1 (steps > 0 rejected); it is implemented now.
    text = MINIMAL_TOML + f"\n[warmup]\nsteps = {steps}\n"
    cfg = load_distill_config(_write(tmp_path, text))
    assert cfg.warmup.steps == steps


def test_warmup_keep_rejects_unknown_value(tmp_path: Path) -> None:
    text = MINIMAL_TOML + "\n[warmup]\nkeep = 'best'\n"
    with pytest.raises(ValueError, match="warmup.keep"):
        load_distill_config(_write(tmp_path, text))


def test_log_sample_rollouts_defaults_to_two_and_zero_disables(tmp_path: Path) -> None:
    cfg = load_distill_config(_write(tmp_path, MINIMAL_TOML))
    assert cfg.train.log_sample_rollouts == 2
    disabled = load_distill_config(
        _write(tmp_path, MINIMAL_TOML + "\n[train]\nlog_sample_rollouts = 0\n")
    )
    assert disabled.train.log_sample_rollouts == 0


def test_train_loss_defaults_to_importance_sampling(tmp_path: Path) -> None:
    cfg = load_distill_config(_write(tmp_path, MINIMAL_TOML))
    assert cfg.train.loss == "importance_sampling"
    assert cfg.train.topk == 8


@pytest.mark.parametrize("loss", ["importance_sampling", "ppo", "topk_ce"])
def test_train_loss_accepts_every_mode(tmp_path: Path, loss: str) -> None:
    text = MINIMAL_TOML + f"\n[train]\nloss = '{loss}'\n"
    cfg = load_distill_config(_write(tmp_path, text))
    assert cfg.train.loss == loss


def test_train_loss_rejects_unknown_value(tmp_path: Path) -> None:
    text = MINIMAL_TOML + "\n[train]\nloss = 'forward_kl'\n"
    with pytest.raises(ValueError, match="train.loss"):
        load_distill_config(_write(tmp_path, text))


def test_ppo_is_a_loss_the_installed_tinker_sdk_accepts() -> None:
    """The `ppo` mode is only legal because the pinned SDK lists it.

    `train.loss = "ppo"` becomes the wire `loss_fn`, which the service
    validates against `tinker.types.LossFnType`. Pinning the membership here
    means an SDK bump that drops or renames the value fails offline instead of
    mid-run, the same discipline the loss_fn_inputs keysets get in
    `data_test.py`.
    """
    pytest.importorskip("tinker")
    from typing import get_args

    from tinker.types import LossFnType

    assert "ppo" in get_args(LossFnType)
    assert "importance_sampling" in get_args(LossFnType)
    assert "cross_entropy" in get_args(LossFnType)


def test_advantage_clip_can_be_switched_off_by_omission(tmp_path: Path) -> None:
    """No clipping is the default, and a positive bound is still accepted.

    TOML has no null, so "no clipping" is expressed by leaving the key out;
    that is why the field defaults to None rather than to a number.
    """
    off = load_distill_config(_write(tmp_path, MINIMAL_TOML + "\n[train]\nloss = 'ppo'\n"))
    assert off.train.advantage_clip is None
    assert off.train.center_advantages is False
    on = load_distill_config(_write(tmp_path, MINIMAL_TOML + "\n[train]\nadvantage_clip = 2.5\n"))
    assert on.train.advantage_clip == pytest.approx(2.5)


def test_centering_can_be_switched_back_on(tmp_path: Path) -> None:
    cfg = load_distill_config(
        _write(tmp_path, MINIMAL_TOML + "\n[train]\ncenter_advantages = true")
    )
    assert cfg.train.center_advantages is True


@pytest.mark.parametrize("clip", ["0.0", "-1.0"])
def test_advantage_clip_still_rejects_non_positive_bounds(tmp_path: Path, clip: str) -> None:
    """Optional does not mean lax: a set bound must be > 0 (0 is not "off")."""
    with pytest.raises(ValueError, match="advantage_clip"):
        load_distill_config(
            _write(tmp_path, MINIMAL_TOML + f"\n[train]\nadvantage_clip = {clip}\n")
        )


def test_snapshot_round_trips_the_ppo_raw_gap_objective(tmp_path: Path) -> None:
    """The snapshot a run dir keeps must reproduce the objective exactly.

    `snapshot_toml` drops None fields, so an unset `advantage_clip` survives
    the round trip only because "unset" and "no clipping" are the same state.
    """
    text = MINIMAL_TOML + "\n[train]\nloss = 'ppo'\ncenter_advantages = false\n"
    cfg = load_distill_config(_write(tmp_path, text))
    snapshot = snapshot_toml(cfg)
    assert "advantage_clip" not in snapshot
    restored = load_distill_config(_write(tmp_path, snapshot))
    assert restored == cfg
    assert restored.train.loss == "ppo"
    assert restored.train.advantage_clip is None
    assert restored.train.center_advantages is False


def test_snapshot_round_trips_an_explicit_clip_and_centering(tmp_path: Path) -> None:
    text = MINIMAL_TOML + "\n[train]\nadvantage_clip = 4.0\ncenter_advantages = true\n"
    cfg = load_distill_config(_write(tmp_path, text))
    restored = load_distill_config(_write(tmp_path, snapshot_toml(cfg)))
    assert restored == cfg
    assert restored.train.advantage_clip == pytest.approx(4.0)
    assert restored.train.center_advantages is True


@pytest.mark.parametrize("topk", [1, 8, 64])
def test_train_topk_bounds_accepted(tmp_path: Path, topk: int) -> None:
    text = MINIMAL_TOML + f"\n[train]\nloss = 'topk_ce'\ntopk = {topk}\n"
    cfg = load_distill_config(_write(tmp_path, text))
    assert cfg.train.topk == topk


def test_snapshot_round_trips_the_topk_ce_loss(tmp_path: Path) -> None:
    text = MINIMAL_TOML + "\n[train]\nloss = 'topk_ce'\ntopk = 16\n"
    cfg = load_distill_config(_write(tmp_path, text))
    restored = load_distill_config(_write(tmp_path, snapshot_toml(cfg)))
    assert restored == cfg
    assert restored.train.loss == "topk_ce"
    assert restored.train.topk == 16


@pytest.mark.parametrize(
    "snippet",
    [
        "[rollout]\nmax_turns = 0",
        "[rollout]\ncontext_budget_tokens = 512",
        "[train]\nsteps = 0",
        "[train]\nlearning_rate = 0.0",
        "[train]\nadvantage_clip = 0.0",
        "[train]\ngroup_size = 0",
        "[train]\ntopk = 0",
        "[train]\ntopk = 65",
        "[train]\nlog_sample_rollouts = -1",
        "[sampling]\ntemperature = -0.1",
        "[sampling]\ntemperature = 2.5",
        "[sampling]\nmax_tokens = 0",
        "[warmup]\nsteps = -1",
        "[warmup]\nrollouts_per_task = 0",
        "[warmup]\nlearning_rate = 0.0",
        "[warmup]\nlearning_rate = -1e-5",
        "[eval]\nevery = -1",
        "[eval]\ntasks = 0",
        "[gate]\nk = 0",
        "[gate]\nmin_teacher_fraction = 0.0",
        "[gate]\nmin_teacher_fraction = 1.5",
        "[pricing]\nstudent_prefill = -0.1",
        "[pricing]\nstudent_cached_prefill = -0.1",
        "[pricing]\nteacher_cached_prefill = -0.1",
        "[pricing]\nteacher_sample = -0.1",
        "[budget]\nmax_usd = 0.0",
    ],
)
def test_bad_ranges_rejected(tmp_path: Path, snippet: str) -> None:
    base = MINIMAL_TOML if "[harbor]" not in snippet else MINIMAL_TOML.split("[harbor]")[0]
    with pytest.raises(ValueError, match="invalid distill config"):
        load_distill_config(_write(tmp_path, base + "\n" + snippet + "\n"))


def test_unknown_key_rejected(tmp_path: Path) -> None:
    text = MINIMAL_TOML + "\n[train]\nnot_a_field = 1\n"
    with pytest.raises(ValueError, match="train.not_a_field"):
        load_distill_config(_write(tmp_path, text))


def test_unknown_section_rejected(tmp_path: Path) -> None:
    text = MINIMAL_TOML + "\n[mystery]\nx = 1\n"
    with pytest.raises(ValueError, match="mystery"):
        load_distill_config(_write(tmp_path, text))


def test_load_error_names_file_and_field(tmp_path: Path) -> None:
    path = _write(tmp_path, MINIMAL_TOML + "\n[train]\nsteps = 0\n")
    with pytest.raises(ValueError) as excinfo:
        load_distill_config(path)
    message = str(excinfo.value)
    assert str(path) in message
    assert "train.steps" in message


def test_missing_required_field_named(tmp_path: Path) -> None:
    text = "[student]\nbase_model = 'm'\n[teacher]\nmodel = 't'\n"
    path = _write(tmp_path, text)
    with pytest.raises(ValueError) as excinfo:
        load_distill_config(path)
    assert "harbor" in str(excinfo.value)
    assert str(path) in str(excinfo.value)


def test_missing_file_error_names_path(tmp_path: Path) -> None:
    path = tmp_path / "nope.toml"
    with pytest.raises(FileNotFoundError, match="nope.toml"):
        load_distill_config(path)


def test_invalid_toml_error_names_file(tmp_path: Path) -> None:
    path = _write(tmp_path, "[student\n")
    with pytest.raises(ValueError, match="not valid TOML"):
        load_distill_config(path)


def test_snapshot_round_trip_minimal(tmp_path: Path) -> None:
    cfg = _minimal_config()
    rendered = snapshot_toml(cfg)
    parsed = DistillConfig.model_validate(tomllib.loads(rendered))
    assert parsed == cfg


def test_snapshot_round_trip_via_file(tmp_path: Path) -> None:
    cfg = load_distill_config(_write(tmp_path, MINIMAL_TOML))
    cfg = cfg.model_copy(deep=True)
    snap_path = tmp_path / "snapshot.toml"
    snap_path.write_text(snapshot_toml(cfg), encoding="utf-8")
    assert load_distill_config(snap_path) == cfg


def test_snapshot_preserves_overrides(tmp_path: Path) -> None:
    cfg = _minimal_config().model_copy(deep=True)
    cfg.train.learning_rate = 5e-5
    cfg.pricing.student_prefill = 0.25
    cfg.budget.max_usd = 10.0
    parsed = DistillConfig.model_validate(tomllib.loads(snapshot_toml(cfg)))
    assert parsed.train.learning_rate == pytest.approx(5e-5)
    assert parsed.pricing.student_prefill == pytest.approx(0.25)
    assert parsed.budget.max_usd == pytest.approx(10.0)
    assert parsed == cfg


def test_snapshot_round_trips_the_wandb_section(tmp_path: Path) -> None:
    # Both shapes must survive snapshot -> parse: the all-defaults section
    # (entity and run_name are None, so exclude_none omits them) and a fully
    # populated one.
    for wandb in (
        WandbConfig(),
        WandbConfig(
            enabled=True,
            project="my-distill",
            entity="my-team",
            run_name="smoke-01",
            tags=["smoke", "tb2"],
        ),
    ):
        cfg = _minimal_config().model_copy(update={"wandb": wandb}, deep=True)
        snap_path = tmp_path / "snapshot.toml"
        snap_path.write_text(snapshot_toml(cfg), encoding="utf-8")
        assert load_distill_config(snap_path) == cfg


def test_snapshot_round_trips_the_warmup_section(tmp_path: Path) -> None:
    # Both shapes must survive snapshot -> parse: the all-defaults section
    # (learning_rate is None, so exclude_none omits it) and a fully populated
    # one.
    for warmup in (
        WarmupConfig(),
        WarmupConfig(
            steps=2,
            rollouts_per_task=2,
            keep="all",
            learning_rate=5e-5,
            trajectories_from=".wmo/distill/runs/source-run",
        ),
    ):
        cfg = _minimal_config().model_copy(update={"warmup": warmup}, deep=True)
        snap_path = tmp_path / "snapshot.toml"
        snap_path.write_text(snapshot_toml(cfg), encoding="utf-8")
        assert load_distill_config(snap_path) == cfg


def test_snapshot_round_trips_the_eval_section(tmp_path: Path) -> None:
    # Both shapes must survive snapshot -> parse: the all-defaults section
    # (both baseline reuse paths are None, so exclude_none omits them) and a
    # fully populated one.
    for eval_cfg in (
        EvalConfig(),
        EvalConfig(
            every=5,
            tasks=4,
            k=2,
            teacher_baseline_from="runs/prior/evals/baseline-teacher.json",
            student_baseline_from="runs/prior/evals/baseline-student-before.json",
        ),
    ):
        cfg = _minimal_config().model_copy(update={"eval": eval_cfg}, deep=True)
        snap_path = tmp_path / "snapshot.toml"
        snap_path.write_text(snapshot_toml(cfg), encoding="utf-8")
        assert load_distill_config(snap_path) == cfg


def test_snapshot_round_trips_the_tripwire_section(tmp_path: Path) -> None:
    # Both shapes must survive snapshot -> parse: the all-defaults section and a
    # deliberately loosened one (a run may widen the bounds; it may not turn them
    # into absolute nats or token counts, which the schema makes impossible).
    for tripwire in (
        TripwireConfig(),
        TripwireConfig(
            enabled=False,
            entropy_warn_frac=0.4,
            entropy_kill_frac=0.2,
            length_warn_frac=0.35,
            length_kill_frac=0.1,
            kill_consecutive_steps=3,
        ),
    ):
        cfg = _minimal_config().model_copy(update={"tripwire": tripwire}, deep=True)
        snap_path = tmp_path / "snapshot.toml"
        snap_path.write_text(snapshot_toml(cfg), encoding="utf-8")
        assert load_distill_config(snap_path) == cfg


def test_tripwire_section_loads_from_toml(tmp_path: Path) -> None:
    text = MINIMAL_TOML + "\n[tripwire]\nentropy_warn_frac = 0.6\nkill_consecutive_steps = 3\n"
    cfg = load_distill_config(_write(tmp_path, text))
    assert cfg.tripwire.entropy_warn_frac == pytest.approx(0.6)
    assert cfg.tripwire.kill_consecutive_steps == 3
    assert cfg.tripwire.entropy_kill_frac == pytest.approx(0.3)  # untouched default


@pytest.mark.parametrize(
    "snippet",
    [
        "[tripwire]\nentropy_warn_frac = 0.0\n",
        "[tripwire]\nentropy_warn_frac = 1.5\n",
        "[tripwire]\nlength_kill_frac = -0.1\n",
        "[tripwire]\nkill_consecutive_steps = 0\n",
    ],
)
def test_tripwire_bad_ranges_rejected(tmp_path: Path, snippet: str) -> None:
    """A fraction must stay a fraction of the measured baseline: 0 would disarm
    the bound silently and anything above 1.0 would fire on a healthy step."""
    with pytest.raises(ValueError, match="tripwire"):
        load_distill_config(_write(tmp_path, MINIMAL_TOML + "\n" + snippet))


def test_a_kill_fraction_above_its_warn_fraction_is_rejected(tmp_path: Path) -> None:
    """An abort that was never warned about first is a configuration mistake."""
    text = MINIMAL_TOML + "\n[tripwire]\nentropy_warn_frac = 0.3\nentropy_kill_frac = 0.5\n"
    with pytest.raises(ValueError, match="entropy_kill_frac"):
        load_distill_config(_write(tmp_path, text))


def test_tripwire_unknown_key_rejected(tmp_path: Path) -> None:
    """No absolute threshold can be smuggled in under a new key."""
    text = MINIMAL_TOML + "\n[tripwire]\nentropy_min_nats = 0.2\n"
    with pytest.raises(ValueError, match="tripwire.entropy_min_nats"):
        load_distill_config(_write(tmp_path, text))


def test_the_recorded_probe_baseline_is_under_the_sibling_absolute_floor() -> None:
    """The documented reason the thresholds are relative, pinned as a test: a
    healthy, untrained Super-120B on TB2 measures 0.181 nats/token, so the
    sibling lane's absolute "entropy < 0.2 means collapse" rule would have fired
    on this project's step 0."""
    assert PROBE_BASELINE_ENTROPY_NATS == pytest.approx(0.181)
    assert PROBE_BASELINE_ENTROPY_NATS < 0.2
    assert PROBE_BASELINE_EPISODE_TOKENS == 7577


def test_wandb_unknown_key_rejected(tmp_path: Path) -> None:
    text = MINIMAL_TOML + "\n[wandb]\nteam = 'nope'\n"
    with pytest.raises(ValueError, match="wandb.team"):
        load_distill_config(_write(tmp_path, text))


def test_pricing_cached_defaults_derive_20_percent_of_prefill() -> None:
    derived = PricingConfig(student_prefill=1.0, teacher_prefill=10.0)
    assert derived.effective_student_cached_prefill == pytest.approx(0.2)
    assert derived.effective_teacher_cached_prefill == pytest.approx(2.0)
    # No full prefill price means no derivation: the cached meter is unpriced.
    assert PricingConfig().effective_student_cached_prefill is None
    assert PricingConfig().effective_teacher_cached_prefill is None
    # An explicit cached price stands alone, even without the full price.
    explicit = PricingConfig(student_cached_prefill=0.07, teacher_cached_prefill=0.9)
    assert explicit.effective_student_cached_prefill == pytest.approx(0.07)
    assert explicit.effective_teacher_cached_prefill == pytest.approx(0.9)


def test_snapshot_round_trips_the_pricing_section(tmp_path: Path) -> None:
    cfg = _minimal_config().model_copy(
        update={
            "pricing": PricingConfig(
                student_prefill=0.3,
                student_sample=0.7,
                student_train=0.6,
                teacher_prefill=2.49,
                teacher_cached_prefill=0.498,
                teacher_sample=6.225,
            )
        },
        deep=True,
    )
    snap_path = tmp_path / "snapshot.toml"
    snap_path.write_text(snapshot_toml(cfg), encoding="utf-8")
    parsed = load_distill_config(snap_path)
    assert parsed == cfg
    # The unset student cached rate stays unset (derived at charge time).
    assert parsed.pricing.student_cached_prefill is None
    assert parsed.pricing.effective_student_cached_prefill == pytest.approx(0.06)


def test_direct_model_validation_rejects_extra() -> None:
    with pytest.raises(ValidationError):
        StudentConfig.model_validate({"base_model": "m", "surprise": 1})


@pytest.mark.parametrize("value", [0, -65536])
def test_train_max_datum_tokens_must_be_positive(value: int) -> None:
    # A zero or negative cap (a dropped minus sign survives TOML) would silently
    # reject every episode from training after the rollout budget was already spent.
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({"max_datum_tokens": value})


XTOKEN_TOML = """
[student]
base_model = "Qwen/Qwen3-8B"

[teacher]
backend = "openai_compat"
model = "nvidia/GLM-5.2-NVFP4"
tokenizer = "zai-org/GLM-5.2"
alignment = "chunk"
endpoint = "http://127.0.0.1:8000/v1"

[harbor]
job_template = "jobs/tb2.yaml"
"""


def test_teacher_defaults_to_tinker_with_same_tokenizer() -> None:
    teacher = TeacherConfig(model="Qwen/Qwen3-235B-A22B-Instruct-2507")
    assert teacher.backend == "tinker"
    assert teacher.alignment == "same_tokenizer"
    assert teacher.endpoint is None
    assert teacher.tokenizer is None


def test_load_cross_tokenizer_teacher(tmp_path: Path) -> None:
    cfg = load_distill_config(_write(tmp_path, XTOKEN_TOML))
    assert cfg.teacher.backend == "openai_compat"
    assert cfg.teacher.model == "nvidia/GLM-5.2-NVFP4"
    assert cfg.teacher.tokenizer == "zai-org/GLM-5.2"
    assert cfg.teacher.alignment == "chunk"
    assert cfg.teacher.endpoint == "http://127.0.0.1:8000/v1"
    # importance_sampling (the default) is the only loss the chunk path supports.
    assert cfg.train.loss == "importance_sampling"


def test_openai_compat_teacher_requires_endpoint() -> None:
    # Without a URL there is nothing to score against, and the failure has to name
    # the missing key rather than surface as a connection error mid-run.
    with pytest.raises(ValidationError, match="teacher.endpoint"):
        TeacherConfig.model_validate(
            {
                "backend": "openai_compat",
                "model": "nvidia/GLM-5.2-NVFP4",
                "tokenizer": "zai-org/GLM-5.2",
                "alignment": "chunk",
            }
        )


def test_openai_compat_teacher_requires_tokenizer() -> None:
    with pytest.raises(ValidationError, match="teacher.tokenizer"):
        TeacherConfig.model_validate(
            {
                "backend": "openai_compat",
                "model": "nvidia/GLM-5.2-NVFP4",
                "alignment": "chunk",
                "endpoint": "http://127.0.0.1:8000/v1",
            }
        )


def test_openai_compat_teacher_requires_chunk_alignment() -> None:
    # A self-hosted teacher never shares the student's vocabulary, so leaving the
    # default same_tokenizer alignment in place would score the wrong token ids.
    with pytest.raises(ValidationError, match='alignment = "chunk"'):
        TeacherConfig.model_validate(
            {
                "backend": "openai_compat",
                "model": "nvidia/GLM-5.2-NVFP4",
                "tokenizer": "zai-org/GLM-5.2",
                "endpoint": "http://127.0.0.1:8000/v1",
            }
        )


def test_openai_compat_teacher_rejects_tinker_checkpoint() -> None:
    with pytest.raises(ValidationError, match="teacher.checkpoint"):
        TeacherConfig.model_validate(
            {
                "backend": "openai_compat",
                "model": "nvidia/GLM-5.2-NVFP4",
                "tokenizer": "zai-org/GLM-5.2",
                "alignment": "chunk",
                "endpoint": "http://127.0.0.1:8000/v1",
                "checkpoint": "tinker://run/weights/7",
            }
        )


def test_tinker_teacher_rejects_endpoint() -> None:
    with pytest.raises(ValidationError, match="teacher.endpoint is only for"):
        TeacherConfig.model_validate(
            {"model": "big-teacher", "endpoint": "http://127.0.0.1:8000/v1"}
        )


def test_chunk_alignment_requires_openai_compat_backend() -> None:
    # Chunk alignment is implemented only against the self-hosted backend; asking a
    # Tinker teacher for it would silently score same-vocabulary ids twice over.
    with pytest.raises(ValidationError, match='alignment = "same_tokenizer"'):
        TeacherConfig.model_validate({"model": "big-teacher", "alignment": "chunk"})


def test_chunk_alignment_rejects_topk_ce_loss(tmp_path: Path) -> None:
    # topk_ce would feed teacher-vocabulary candidate ids to the student as targets.
    text = XTOKEN_TOML + '\n[train]\nloss = "topk_ce"\n'
    with pytest.raises(ValueError, match="topk_ce"):
        load_distill_config(_write(tmp_path, text))


def test_same_tokenizer_alignment_still_accepts_topk_ce_loss(tmp_path: Path) -> None:
    text = MINIMAL_TOML + '\n[train]\nloss = "topk_ce"\n'
    cfg = load_distill_config(_write(tmp_path, text))
    assert cfg.train.loss == "topk_ce"
    assert cfg.teacher.alignment == "same_tokenizer"


def test_snapshot_round_trips_the_cross_tokenizer_teacher(tmp_path: Path) -> None:
    cfg = load_distill_config(_write(tmp_path, XTOKEN_TOML))
    snap_path = tmp_path / "snapshot.toml"
    snap_path.write_text(snapshot_toml(cfg), encoding="utf-8")
    assert load_distill_config(snap_path) == cfg


def test_checked_in_run_configs_resolve_cookbook_renderers() -> None:
    # The configs pin real Tinker lineup names; the cookbook must know a renderer
    # for each or the very first rollout completion of that run fails in build_renderer
    # (regression: the Nemotron smoke config once carried lineup names absent from the
    # cookbook 0.4.x catalog). The student always trains on Tinker, so it always needs
    # a renderer. An openai_compat teacher is a self-hosted HF repo id the cookbook has
    # never heard of (zai-org/GLM-* is absent from the 0.4.3 catalog); what it must
    # carry instead is the endpoint serving it and its own tokenizer.
    pytest.importorskip("tinker_cookbook")
    from tinker_cookbook.model_info import get_recommended_renderer_name

    paths = sorted((Path(__file__).parent / "configs").glob("*.toml"))
    assert paths, "the reference configs that ship with the package are missing"
    for path in paths:
        cfg = load_distill_config(path)
        models = [cfg.student.base_model]
        if cfg.teacher.backend == "tinker":
            models.append(cfg.teacher.model)
        else:
            assert cfg.teacher.endpoint, f"{path.name}: openai_compat teacher needs endpoint"
            assert cfg.teacher.tokenizer, f"{path.name}: openai_compat teacher needs tokenizer"
        for model in models:
            assert get_recommended_renderer_name(model), f"{path.name}: {model}"


def test_checked_in_run_configs_name_a_verbatim_renderer_for_every_tinker_model() -> None:
    """Every model a run SAMPLES needs an explicit renderer, and it must be a wmo one.

    The auto-discovered reasoning renderer of every model in this lineup kills the
    trial before it grades anything (harbor's terminus-2 parsers are handed a list),
    so a config that leaves `[rollout.renderers]` unset for a sampled model does not
    run at all. The value check lives in the validator; this pins that the checked-in
    configs actually carry the entries.
    """
    pytest.importorskip("tinker_cookbook")
    from wmo.distill.renderers import WMO_RENDERERS

    paths = sorted((Path(__file__).parent / "configs").glob("*.toml"))
    assert paths, "the reference configs that ship with the package are missing"
    for path in paths:
        cfg = load_distill_config(path)
        # An openai_compat teacher is served outside Tinker and renders with its own
        # template, so it never reaches terminus-2's renderer.
        sampled = [cfg.student.base_model]
        if cfg.teacher.backend == "tinker":
            sampled.append(cfg.teacher.model)
        for model in sampled:
            renderer = cfg.rollout.renderers.get(model)
            assert renderer is not None, f"{path.name}: no rollout.renderers entry for {model}"
            assert renderer in WMO_RENDERERS, f"{path.name}: {model} = {renderer}"


def test_the_config_module_never_imports_the_distill_extra_at_module_scope() -> None:
    """`wmo --help` must work on a base install: the CLI imports this module eagerly.

    Renderer-name validation needs tinker-cookbook, so it lives behind a lazy import
    inside the validator and only runs when a config actually names renderers.
    """
    import ast

    import wmo.distill.config as config_module

    assert config_module.__file__ is not None
    tree = ast.parse(Path(config_module.__file__).read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            roots.add(node.module.split(".")[0])
    assert not roots & {"tinker", "tinker_cookbook", "harbor"}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            assert node.module != "wmo.distill.renderers"


def test_a_renderer_name_the_cookbook_cannot_build_is_rejected_at_load(tmp_path: Path) -> None:
    """A typo must fail here, not on the first (paid) rollout of the run."""
    pytest.importorskip("tinker_cookbook")
    text = MINIMAL_TOML + (
        '\n[rollout.renderers]\n"Qwen/Qwen3-8B" = "wmo/qwen3_5_verbatm"\n'  # codespell:ignore
    )
    with pytest.raises(ValueError, match="tinker-cookbook cannot build"):
        load_distill_config(_write(tmp_path, text))


def test_the_wmo_verbatim_and_builtin_renderer_names_load(tmp_path: Path) -> None:
    pytest.importorskip("tinker_cookbook")
    text = MINIMAL_TOML + (
        "\n[rollout.renderers]\n"
        '"Qwen/Qwen3-8B" = "wmo/qwen3_verbatim"\n'
        '"Qwen/Qwen3-235B-A22B-Instruct-2507" = "qwen3_disable_thinking"\n'
    )
    cfg = load_distill_config(_write(tmp_path, text))
    assert cfg.rollout.renderers["Qwen/Qwen3-8B"] == "wmo/qwen3_verbatim"


def test_a_renderer_key_naming_a_model_the_run_never_samples_is_rejected(tmp_path: Path) -> None:
    """The more dangerous typo: an unmatched key is silently ignored everywhere else."""
    pytest.importorskip("tinker_cookbook")
    text = MINIMAL_TOML + '\n[rollout.renderers]\n"Qwen/Qwen3-9B" = "wmo/qwen3_verbatim"\n'
    with pytest.raises(ValueError, match="never samples"):
        load_distill_config(_write(tmp_path, text))


def _checked_in_config(name: str) -> DistillConfig:
    """Load one of the reference configs that ship beside this module.

    They live in the package rather than in `.agents/` so they reach users of the
    wheel, which is the point of a reference config. No skip guard: these are
    shipped files, so an absent one is a packaging regression and should fail.
    """
    return load_distill_config(Path(__file__).parent / "configs" / name)


def test_training_turn_cap_defaults_to_the_rollout_cap(tmp_path: Path) -> None:
    """Unset means one cap everywhere, which is the pre-existing behaviour."""
    cfg = load_distill_config(_write(tmp_path, MINIMAL_TOML))
    assert cfg.train.rollout_max_turns is None
    assert cfg.rollout.max_turns == 100


def test_training_turn_cap_is_separate_from_the_eval_cap(tmp_path: Path) -> None:
    """Evals must stay at rollout.max_turns or the paired before/after delta is void."""
    text = MINIMAL_TOML + "\n[rollout]\nmax_turns = 100\n\n[train]\nrollout_max_turns = 20\n"
    cfg = load_distill_config(_write(tmp_path, text))
    assert cfg.train.rollout_max_turns == 20
    assert cfg.rollout.max_turns == 100, "the eval-side cap must not move"
