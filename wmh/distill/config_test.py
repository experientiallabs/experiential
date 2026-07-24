"""Tests for the distillation run config: loading, validators, snapshotting."""

import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from wmh.distill.config import (
    DistillConfig,
    HarborConfig,
    PricingConfig,
    StudentConfig,
    TeacherConfig,
    TrainConfig,
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
    assert cfg.harbor.backend == "local"
    assert cfg.harbor.reward_key == "reward"
    assert cfg.rollout.max_turns == 20
    assert cfg.rollout.context_budget_tokens == 65536
    assert cfg.rollout.compaction is False
    assert cfg.train.steps == 40
    assert cfg.train.tasks_per_batch == 8
    assert cfg.train.group_size == 4
    assert cfg.train.learning_rate == pytest.approx(1e-4)
    assert cfg.train.advantage_clip == pytest.approx(4.0)
    assert cfg.train.center_advantages is True
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
    assert cfg.eval.every == 10
    assert cfg.eval.tasks == 12
    assert cfg.eval.k == 1
    assert cfg.gate.k == 3
    assert cfg.gate.min_teacher_fraction == pytest.approx(0.7)
    assert cfg.gate.require_no_regression is True
    assert cfg.pricing.student_prefill is None
    assert cfg.budget.max_usd is None
    assert cfg.wandb.enabled is False
    assert cfg.wandb.project == "wmh-distill"
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
every = 0
tasks = 2
k = 2

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
    assert cfg.harbor.backend == "e2b"
    assert cfg.harbor.reward_key == "score"
    assert cfg.rollout.max_turns == 5
    assert cfg.train.center_advantages is False
    assert cfg.sampling.temperature == 0.0
    assert cfg.warmup.steps == 2
    assert cfg.warmup.rollouts_per_task == 3
    assert cfg.warmup.keep == "all"
    assert cfg.warmup.learning_rate == pytest.approx(2e-5)
    assert cfg.eval.every == 0
    assert cfg.gate.min_teacher_fraction == 1.0
    assert cfg.pricing.is_complete()
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


def test_compaction_true_rejected(tmp_path: Path) -> None:
    text = MINIMAL_TOML + "\n[rollout]\ncompaction = true\n"
    with pytest.raises(ValueError, match="prefix property"):
        load_distill_config(_write(tmp_path, text))


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


@pytest.mark.parametrize(
    "snippet",
    [
        "[rollout]\nmax_turns = 0",
        "[rollout]\ncontext_budget_tokens = 512",
        "[train]\nsteps = 0",
        "[train]\nlearning_rate = 0.0",
        "[train]\nadvantage_clip = 0.0",
        "[train]\ngroup_size = 0",
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
        WarmupConfig(steps=2, rollouts_per_task=2, keep="all", learning_rate=5e-5),
    ):
        cfg = _minimal_config().model_copy(update={"warmup": warmup}, deep=True)
        snap_path = tmp_path / "snapshot.toml"
        snap_path.write_text(snapshot_toml(cfg), encoding="utf-8")
        assert load_distill_config(snap_path) == cfg


def test_wandb_unknown_key_rejected(tmp_path: Path) -> None:
    text = MINIMAL_TOML + "\n[wandb]\nteam = 'nope'\n"
    with pytest.raises(ValueError, match="wandb.team"):
        load_distill_config(_write(tmp_path, text))


def test_pricing_is_complete() -> None:
    assert not PricingConfig().is_complete()
    partial = PricingConfig(student_prefill=0.1, student_sample=0.2)
    assert not partial.is_complete()
    # The four full prices alone are not complete: teacher-in-harness episodes
    # bill teacher_sample, so it must be priced too.
    no_teacher_sample = PricingConfig(
        student_prefill=0.1, student_sample=0.2, student_train=0.3, teacher_prefill=0.4
    )
    assert not no_teacher_sample.is_complete()
    # The cached rates never block completeness: their 20% defaults derive
    # from the full prefill prices.
    full = PricingConfig(
        student_prefill=0.1,
        student_sample=0.2,
        student_train=0.3,
        teacher_prefill=0.4,
        teacher_sample=1.0,
    )
    assert full.is_complete()


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


def test_checked_in_run_configs_resolve_cookbook_renderers() -> None:
    # The smoke configs pin real Tinker lineup names; the cookbook must know a renderer
    # for each or the very first rollout completion of that run fails in build_renderer
    # (regression: the Nemotron smoke config once carried lineup names absent from the
    # cookbook 0.4.x catalog).
    pytest.importorskip("tinker_cookbook")
    from tinker_cookbook.model_info import get_recommended_renderer_name

    config_dir = Path(__file__).resolve().parents[2] / ".agents" / "distill"
    paths = sorted(config_dir.glob("*.toml"))
    if not paths:
        pytest.skip("no checked-in distill run configs in this tree")
    for path in paths:
        cfg = load_distill_config(path)
        for model in (cfg.student.base_model, cfg.teacher.model):
            assert get_recommended_renderer_name(model), f"{path.name}: {model}"
