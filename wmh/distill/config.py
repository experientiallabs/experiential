"""Per-run TOML configuration for an on-policy distillation run.

A distillation run is described by one TOML file with sections mirroring the
sub-models below (student, teacher, harbor, rollout, train, sampling, warmup,
eval, gate, pricing, budget, wandb). `load_distill_config` reads and validates
the file; `snapshot_toml` renders a validated config back to TOML so a run dir
can carry an exact snapshot of the configuration it ran with.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated, Literal

import tomli_w
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class StudentConfig(BaseModel):
    """The Tinker LoRA student under training."""

    model_config = ConfigDict(extra="forbid")

    base_model: str
    lora_rank: int = 32


class TeacherConfig(BaseModel):
    """The teacher that scores student tokens via compute_logprobs."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["tinker"] = "tinker"
    model: str
    checkpoint: str | None = None
    """Optional tinker:// checkpoint path to serve the teacher from."""


class HarborConfig(BaseModel):
    """How rollouts are produced: the pi agent on harbor tasks.

    Attempts per task are NOT configured here: training rollouts use
    `train.group_size` (the on-policy group) and evals use `eval.k` / `gate.k`.
    """

    model_config = ConfigDict(extra="forbid")

    job_template: str
    """Path to a Harbor JobConfig YAML/JSON used as the task template."""

    backend: Literal["local", "e2b"] = "local"
    reward_key: str = "reward"

    retries: int = Field(default=1, ge=0)
    """Harbor-level retries per failed trial. Distill batches see transient
    sandbox/runner deaths (e.g. an E2B transport drop killing the pi runner
    mid-episode); one retry absorbs them, and any trial that still ends
    without a verifier reward scores 0.0 instead of aborting the run."""


class RolloutConfig(BaseModel):
    """Per-episode rollout limits."""

    model_config = ConfigDict(extra="forbid")

    max_turns: int = Field(default=20, ge=1)
    """Episode turn cap; pinned into the harness doc's `param:max-turns`."""

    context_budget_tokens: int = Field(default=65536, ge=1024)
    """Context cap: episodes where any call's prompt plus sampled tokens exceed
    it are dropped whole from training (`build_datums`), and the cost estimate
    caps per-episode tokens here."""

    compaction: bool = False

    @field_validator("compaction")
    @classmethod
    def _reject_compaction(cls, value: bool) -> bool:
        """Reject compaction: it breaks the prefix property the cost model relies on.

        Every turn's prompt must extend the previous turn's tokens verbatim so
        prefill work amortizes across turns and sampled spans stay aligned for
        teacher scoring. Compacting mid-rollout rewrites the prefix, so each
        later turn would be a full re-prefill and issued spans would no longer
        appear verbatim in the episode tokens.
        """
        if value:
            raise ValueError(
                "rollout.compaction = true is not supported: compaction rewrites the "
                "token prefix mid-episode, breaking the prefix property that keeps "
                "prefill costs amortized across turns and sampled spans verbatim in "
                "the episode; set compaction = false (episodes that outgrow "
                "context_budget_tokens are dropped whole from training instead)"
            )
        return value


class TrainConfig(BaseModel):
    """Optimizer-loop schedule and batch shape."""

    model_config = ConfigDict(extra="forbid")

    steps: int = Field(default=40, ge=1)
    tasks_per_batch: int = Field(default=8, ge=1)
    group_size: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    advantage_clip: float = Field(default=4.0, gt=0)
    center_advantages: bool = True
    max_datum_tokens: int = 65536
    sampler_refresh_every: int = Field(default=1, ge=1)
    save_state_every: int = Field(default=8, ge=1)
    trial_concurrency: int = Field(default=8, ge=1)


class SamplingConfig(BaseModel):
    """Student sampling parameters used during rollouts.

    Both values are pinned into the harness document's param surfaces
    (`param:temperature`, `param:max-output-tokens`), which is how the pi
    runtimes stamp them onto every worker request.
    """

    model_config = ConfigDict(extra="forbid")

    temperature: float = Field(default=1.0, ge=0, le=2)
    """Rollout sampling temperature. 1.0 (the default) keeps the sampler's
    issued logprobs directly comparable to the teacher's untempered
    compute_logprobs values; any other value biases the reverse-KL advantages
    and is warned about at run start."""

    max_tokens: int = Field(default=8192, ge=1)
    """Per-completion output cap for every rollout request."""


class WarmupConfig(BaseModel):
    """Supervised warmup on the teacher's own pi trajectories before OPD steps.

    The remedy for a student that samples only failing trajectories (on-policy
    distillation then matches the teacher on failures): before the OPD step
    loop, the teacher runs the pi harness on the TRAIN tasks, its kept trials
    become cross_entropy SFT datums via the same prefix merge, and the student
    trains `steps` full-batch passes over them. 0 steps (the default) disables
    the phase entirely.
    """

    model_config = ConfigDict(extra="forbid")

    steps: int = Field(default=0, ge=0)
    """Full-batch SFT passes over the kept teacher trajectories; 0 disables."""

    rollouts_per_task: int = Field(default=1, ge=1)
    """Teacher attempts per train task when collecting warmup trajectories."""

    keep: Literal["passed", "all"] = "passed"
    """Which teacher trials feed the SFT set: reward-passing only, or all."""

    learning_rate: Annotated[float, Field(gt=0)] | None = None
    """Warmup optimizer LR; None uses `train.learning_rate`."""


class EvalConfig(BaseModel):
    """Periodic held-out evaluation schedule."""

    model_config = ConfigDict(extra="forbid")

    every: int = Field(default=10, ge=0)
    """Evaluate every N train steps; 0 means final eval only."""

    tasks: int = Field(default=12, ge=1)
    k: int = Field(default=1, ge=1)


class GateConfig(BaseModel):
    """Promotion gate for the distilled student."""

    model_config = ConfigDict(extra="forbid")

    k: int = Field(default=3, ge=1)
    min_teacher_fraction: float = Field(default=0.7, gt=0, le=1)
    require_no_regression: bool = True


CACHED_PREFILL_FRACTION = 0.2
"""Default cached-prefill price as a fraction of the full prefill price.

Tinker bills a request's verbatim repeated prompt prefix at 20% of the full
prefill price (the ratio its console lists, e.g. Ultra 0.498 vs 2.49
USD/Mtok). An explicit `*_cached_prefill` field overrides this derivation.
"""


class PricingConfig(BaseModel):
    """Per-model per-meter prices, USD per million tokens (all optional).

    Tinker bills prefill PER REQUEST over each call's full prompt: every
    agent turn re-bills its whole context, with the verbatim repeated prefix
    billed at a discounted cached rate. The `*_cached_prefill` fields carry
    that cached rate; when left unset they default to
    `CACHED_PREFILL_FRACTION` (20%) of the corresponding full prefill price
    whenever that price is set, and stay unpriced otherwise. `teacher_sample`
    prices the tokens teacher-in-harness episodes (warmup collection and the
    gate's teacher baseline) SAMPLE, which bill at the sampling rate
    (~2.5x prefill on the live price list), not the prefill rate.
    """

    model_config = ConfigDict(extra="forbid")

    student_prefill: Annotated[float, Field(ge=0)] | None = None
    student_cached_prefill: Annotated[float, Field(ge=0)] | None = None
    """Student cached-prefill rate; None means 20% of student_prefill when set."""

    student_sample: Annotated[float, Field(ge=0)] | None = None
    student_train: Annotated[float, Field(ge=0)] | None = None
    teacher_prefill: Annotated[float, Field(ge=0)] | None = None
    teacher_cached_prefill: Annotated[float, Field(ge=0)] | None = None
    """Teacher cached-prefill rate; None means 20% of teacher_prefill when set."""

    teacher_sample: Annotated[float, Field(ge=0)] | None = None
    """Sampling rate for teacher-in-harness episodes (warmup, gate baseline)."""

    @property
    def effective_student_cached_prefill(self) -> float | None:
        """The student cached-prefill price actually charged (see class docstring)."""
        if self.student_cached_prefill is not None:
            return self.student_cached_prefill
        if self.student_prefill is not None:
            return self.student_prefill * CACHED_PREFILL_FRACTION
        return None

    @property
    def effective_teacher_cached_prefill(self) -> float | None:
        """The teacher cached-prefill price actually charged (see class docstring)."""
        if self.teacher_cached_prefill is not None:
            return self.teacher_cached_prefill
        if self.teacher_prefill is not None:
            return self.teacher_prefill * CACHED_PREFILL_FRACTION
        return None

    def is_complete(self) -> bool:
        """Whether every meter has a price, so run cost can be fully accounted.

        The cached-prefill meters count as priced through their derived 20%
        defaults, so completeness needs exactly the four full prices plus
        `teacher_sample`.
        """
        return (
            self.student_prefill is not None
            and self.effective_student_cached_prefill is not None
            and self.student_sample is not None
            and self.student_train is not None
            and self.teacher_prefill is not None
            and self.effective_teacher_cached_prefill is not None
            and self.teacher_sample is not None
        )


class BudgetConfig(BaseModel):
    """Optional hard USD budget for the whole run."""

    model_config = ConfigDict(extra="forbid")

    max_usd: Annotated[float, Field(gt=0)] | None = None


class WandbConfig(BaseModel):
    """Optional Weights & Biases run tracking (off by default).

    Enabling it requires the wandb SDK (`uv sync --extra distill`) and
    credentials (WANDB_API_KEY or a prior `wandb login`); both are checked
    before the run spends anything (see `wmh.distill.tracking`).
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    project: str = "wmh-distill"
    entity: str | None = None
    run_name: str | None = None
    """The wandb run name; None derives one from the agent name and run dir."""

    tags: list[str] = Field(default_factory=list)


class DistillConfig(BaseModel):
    """Top-level configuration for one distillation run.

    The student, teacher, and harbor sections are required (each carries a
    required field); every other section has complete defaults and may be
    omitted from the TOML file.
    """

    model_config = ConfigDict(extra="forbid")

    student: StudentConfig
    teacher: TeacherConfig
    harbor: HarborConfig
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    warmup: WarmupConfig = Field(default_factory=WarmupConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    gate: GateConfig = Field(default_factory=GateConfig)
    pricing: PricingConfig = Field(default_factory=PricingConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)


def load_distill_config(path: Path) -> DistillConfig:
    """Load and validate a distillation run config from a TOML file.

    Args:
        path: Path to the per-run TOML file.

    Returns:
        The validated DistillConfig.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid TOML or fails validation; the
            message names the file and each failing field.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"distill config not found: {path}") from exc
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"invalid distill config {path}: not valid TOML ({exc})") from exc
    try:
        return DistillConfig.model_validate(data)
    except ValidationError as exc:
        details = "; ".join(
            "{field}: {msg}".format(
                field=".".join(str(part) for part in err["loc"]) or "(top level)",
                msg=err["msg"],
            )
            for err in exc.errors()
        )
        raise ValueError(f"invalid distill config {path}: {details}") from exc


def snapshot_toml(cfg: DistillConfig) -> str:
    """Render a validated config back to TOML for run-dir snapshotting.

    Unset optional fields (None) are omitted; parsing the result back yields
    an identical DistillConfig.

    Args:
        cfg: The config to snapshot.

    Returns:
        A valid TOML document string.
    """
    data = cfg.model_dump(mode="json", exclude_none=True)
    return tomli_w.dumps(data)
