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
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


class StudentConfig(BaseModel):
    """The Tinker LoRA student under training."""

    model_config = ConfigDict(extra="forbid")

    base_model: str
    lora_rank: int = 32


class TeacherConfig(BaseModel):
    """The teacher that scores the student's sampled tokens.

    Two backends, distinguished by whether the teacher shares the student's
    vocabulary. A `tinker` teacher scores the student's exact token ids through
    `compute_logprobs`, which is only meaningful when both tokenizers agree. An
    `openai_compat` teacher is a self-hosted vLLM server that cannot read the
    student's ids at all, so it scores its OWN tokenization of the same
    conversation and the two are compared span by span
    (`alignment = "chunk"`, see `wmh.distill.xtoken`).
    """

    model_config = ConfigDict(extra="forbid")

    backend: Literal["tinker", "openai_compat"] = "tinker"
    model: str
    checkpoint: str | None = None
    """Optional tinker:// checkpoint path to serve the teacher from."""

    endpoint: str | None = None
    """Base URL of the OpenAI-compatible server, e.g. `http://host:8000/v1`.
    Required by (and exclusive to) the `openai_compat` backend."""

    tokenizer: str | None = None
    """HuggingFace repo id of the teacher's tokenizer, e.g. `zai-org/GLM-5.2`.
    The cross-tokenizer path needs it to render and tokenize the conversation
    on the teacher's side; the model name is a served id the cookbook does not
    know."""

    alignment: Literal["same_tokenizer", "chunk"] = "same_tokenizer"
    """How teacher logprobs line up with student tokens. `same_tokenizer`
    scores the student's ids position for position; `chunk` aligns byte-
    identical message content between the two tokenizations."""

    @model_validator(mode="after")
    def _check_backend_coherence(self) -> TeacherConfig:
        """Require each backend's fields and reject the other backend's.

        A half-configured cross-tokenizer teacher is the dangerous case: an
        endpoint with `alignment = "same_tokenizer"` would send the student's
        token ids to a foreign vocabulary and score noise without erroring.
        """
        if self.backend == "openai_compat":
            missing = [
                name
                for name, value in (("endpoint", self.endpoint), ("tokenizer", self.tokenizer))
                if not value
            ]
            if missing:
                raise ValueError(
                    f"teacher.backend = 'openai_compat' requires {' and '.join(missing)}; "
                    "set endpoint to the vLLM server base URL (e.g. "
                    "'http://host:8000/v1') and tokenizer to the teacher's HuggingFace "
                    "repo id (e.g. 'zai-org/GLM-5.2')"
                )
            if self.alignment != "chunk":
                raise ValueError(
                    "teacher.backend = 'openai_compat' requires teacher.alignment = "
                    "'chunk': a served teacher cannot score the student's token ids, so "
                    "its logprobs must be chunk-aligned against its own tokenization"
                )
            if self.checkpoint is not None:
                raise ValueError(
                    "teacher.checkpoint is a tinker:// weights path and has no meaning "
                    "for the 'openai_compat' backend; drop it and point teacher.endpoint "
                    "at the server that already serves those weights"
                )
            return self
        if self.endpoint is not None:
            raise ValueError(
                "teacher.endpoint is only for teacher.backend = 'openai_compat'; the "
                "tinker backend reaches its teacher through the Tinker service, so "
                "either drop endpoint or set backend = 'openai_compat'"
            )
        if self.alignment != "same_tokenizer":
            raise ValueError(
                "teacher.alignment = 'chunk' requires teacher.backend = "
                "'openai_compat'; a tinker teacher shares the student's vocabulary and "
                "is scored position for position"
            )
        return self


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

    max_turns: int = Field(default=100, ge=1)
    """Episode turn cap; pinned into the harness doc's `param:max-turns`.

    Measured on real TerminalBench-2 episodes: p50 21 calls, p95 57, max 72. A
    cap of 20 truncated the majority of episodes mid-task, so it is not a
    safety limit but a silent rollout filter."""

    episode_timeout_s: float = Field(default=1800.0, gt=0)
    """Per-episode wall clock, passed through to the harbor scorer.

    Without this the scorer's own default of 300 s applies, against a measured
    per-trial p50 of 597 s and p95 of 1,185 s on TerminalBench-2 -- so most
    trials would die on the clock and score 0, which reads as a capability
    result rather than a timeout. Only honoured for the e2b backend; the local
    runner rejects a non-default value because it shares one runner dir."""

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
    loss: Literal["importance_sampling", "topk_ce"] = "importance_sampling"
    """The distillation loss. `importance_sampling` (the default) trains on
    per-token reverse-KL advantages over the student's realized tokens;
    `topk_ce` trains a weighted cross-entropy over the teacher's top-k
    candidate tokens at every loss position (renormalized teacher probs as
    weights), which carries dense supervision from tokens the student did
    NOT sample at roughly k times the training-token volume."""

    topk: int = Field(default=8, ge=1, le=64)
    """How many teacher candidates per position under `loss = "topk_ce"`
    (ignored by `importance_sampling`). Training volume scales linearly
    with it (k replicated cross_entropy datums per source datum)."""

    advantage_clip: float = Field(default=4.0, gt=0)
    center_advantages: bool = True
    max_datum_tokens: int = Field(default=65536, ge=1)
    sampler_refresh_every: int = Field(default=1, ge=1)
    save_state_every: int = Field(default=8, ge=1)
    trial_concurrency: int = Field(default=8, ge=1)

    log_sample_rollouts: int = Field(default=2, ge=0)
    """How many sample episodes each batch renders to human-readable text:
    after every training step, the warmup collection, and each eval batch,
    the first N span-bearing trials are decoded WITH the chat template's
    special tokens and written to the run dir's `samples/` files plus the
    tracker's samples table (see `wmh.distill.samples`). 0 disables."""


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

    trajectories_from: str | None = None
    """Path to another run dir whose warmup COLLECTION completed: the warmup
    phase loads that run's `warmup-trials.json` manifest instead of collecting
    teacher rollouts here. The manifest's teacher must match this run's
    teacher, the `keep` filter applies at load time (so it may differ from the
    source run's), and loading charges no meter (the source run paid for the
    collection); the CE training passes still run per this run's config."""


class EvalConfig(BaseModel):
    """Held-out evaluation schedule plus cross-run baseline reuse."""

    model_config = ConfigDict(extra="forbid")

    every: int = Field(default=0, ge=0)
    """Evaluate every N train steps; 0 (the default) means final eval only."""

    tasks: int = Field(default=12, ge=1)
    k: int = Field(default=1, ge=1)

    teacher_baseline_from: str | None = None
    """Path to a prior run's `evals/baseline-teacher.json` to reuse instead of
    re-running the teacher-in-harness holdout baseline (the teacher's solve
    rate is a property of the teacher, not of the training run). The report
    must cover exactly this run's holdout task ids, carry at least `gate.k`
    attempts, and name the same teacher model; it is copied into this run's
    `evals/` with a provenance `source` note."""

    student_baseline_from: str | None = None
    """Path to a prior run's `evals/baseline-student-before.json` to reuse
    instead of re-running the pre-training student baseline (parallel runs
    from the same base model share it). Validated like
    `teacher_baseline_from`, except the model check is on the report's
    recorded `base_model` field: the student's provider model is a per-run
    sampler path and never matches across runs."""


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


def _effective_cached(explicit: float | None, full_prefill: float | None) -> float | None:
    """The cached-prefill rate charged: the explicit override, else 20% of the full rate."""
    if explicit is not None:
        return explicit
    if full_prefill is not None:
        return full_prefill * CACHED_PREFILL_FRACTION
    return None


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
        return _effective_cached(self.student_cached_prefill, self.student_prefill)

    @property
    def effective_teacher_cached_prefill(self) -> float | None:
        """The teacher cached-prefill price actually charged (see class docstring)."""
        return _effective_cached(self.teacher_cached_prefill, self.teacher_prefill)

    def is_complete(self) -> bool:
        """Whether every meter has a price, so run cost can be fully accounted.

        The cached-prefill meters count as priced through their derived 20%
        defaults whenever the corresponding full prefill price is set (which
        this requires), so completeness needs exactly the four full prices
        plus `teacher_sample`.
        """
        return (
            self.student_prefill is not None
            and self.student_sample is not None
            and self.student_train is not None
            and self.teacher_prefill is not None
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

    @model_validator(mode="after")
    def _check_loss_supports_alignment(self) -> DistillConfig:
        """Reject the loss/alignment pairs that cannot be expressed on the wire.

        `topk_ce` trains a weighted cross-entropy whose targets are the
        teacher's candidate TOKEN IDS. Under a foreign vocabulary those ids
        index a different embedding table, so they would be trained as if they
        were student tokens: silently meaningless targets, not an error.
        """
        if self.teacher.alignment == "chunk" and self.train.loss == "topk_ce":
            raise ValueError(
                "train.loss = 'topk_ce' cannot be used with teacher.alignment = "
                "'chunk': topk_ce trains on the teacher's candidate token ids as "
                "student targets, and those ids mean something different in the "
                "student's vocabulary. Use train.loss = 'importance_sampling', which "
                "needs only scalar logprob sums per aligned chunk"
            )
        return self


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
