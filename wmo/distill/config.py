"""Per-run TOML configuration for an on-policy distillation run.

A distillation run is described by one TOML file with sections mirroring the
sub-models below (student, teacher, harbor, rollout, train, sampling, warmup,
eval, gate, pricing, budget, tripwire, wandb). `load_distill_config` reads and
validates the file; `snapshot_toml` renders a validated config back to TOML so a
run dir can carry an exact snapshot of the configuration it ran with.
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
    JsonValue,
    ValidationError,
    field_validator,
    model_validator,
)

from wmo.distill.rendering import MISSING_DISTILL_EXTRA


class StudentConfig(BaseModel):
    """The Tinker LoRA student under training."""

    model_config = ConfigDict(extra="forbid")

    base_model: str
    lora_rank: int = 32


class TeacherConfig(BaseModel):
    """The teacher that scores student tokens, and how its vocabulary lines up.

    Two backends, and the backend choice fixes the token alignment:

    - `tinker` (the default) serves the teacher from the Tinker lineup and
      requires the teacher to share the student's tokenizer, so student token
      ids are scored verbatim (`alignment = "same_tokenizer"`).
    - `openai_compat` scores against a self-hosted vLLM OpenAI-compatible
      server, which is how a teacher outside the Tinker lineup (e.g. a
      quantized GLM checkpoint) is reached. Its vocabulary differs from the
      student's, so scoring goes through byte-aligned chunks
      (`alignment = "chunk"`) and the teacher's own `tokenizer` must be named.
    """

    model_config = ConfigDict(extra="forbid")

    backend: Literal["tinker", "openai_compat"] = "tinker"
    model: str
    checkpoint: str | None = None
    """Optional tinker:// checkpoint path to serve the teacher from
    (tinker backend only)."""

    endpoint: str | None = None
    """Base URL of the self-hosted vLLM OpenAI-compatible server serving the
    teacher (openai_compat backend only, where it is required)."""

    tokenizer: str | None = None
    """HF repo id of the teacher's tokenizer, e.g. `zai-org/GLM-5.2`. Required
    by the openai_compat backend: chunk alignment has to tokenize teacher-side
    text itself to map student token spans onto teacher token spans."""

    alignment: Literal["same_tokenizer", "chunk"] = "same_tokenizer"
    """How student token positions map onto teacher token positions.
    `same_tokenizer` scores the student's own ids directly; `chunk` splits each
    sampled span into byte-aligned chunks scored in the teacher's vocabulary."""

    @model_validator(mode="after")
    def _check_backend_axis(self) -> TeacherConfig:
        """Reject backend/alignment/field combinations that cannot be served.

        Returns:
            This config, unchanged, when the combination is coherent.

        Raises:
            ValueError: If the backend does not match the fields it needs (or
                forbids); every message names the key to add or drop.
        """
        if self.backend == "openai_compat":
            if self.checkpoint is not None:
                raise ValueError(
                    "teacher.checkpoint is a tinker:// path and only applies to "
                    'teacher.backend = "tinker"; drop teacher.checkpoint and point '
                    "teacher.endpoint at a server already serving the weights you want"
                )
            if self.endpoint is None:
                raise ValueError(
                    'teacher.backend = "openai_compat" needs teacher.endpoint: set it to '
                    "the base URL of the vLLM OpenAI-compatible server serving "
                    f'{self.model!r} (e.g. endpoint = "http://127.0.0.1:8000/v1"), or '
                    'switch to backend = "tinker" for a Tinker-lineup teacher'
                )
            if self.tokenizer is None:
                raise ValueError(
                    'teacher.backend = "openai_compat" needs teacher.tokenizer: set it to '
                    "the HF repo id of the teacher's tokenizer (e.g. tokenizer = "
                    '"zai-org/GLM-5.2") so chunk alignment can tokenize teacher-side text'
                )
            if self.alignment != "chunk":
                raise ValueError(
                    'teacher.backend = "openai_compat" requires teacher.alignment = '
                    f'"chunk", got {self.alignment!r}: a self-hosted teacher does not '
                    "share the student's vocabulary, so its logprobs cannot be read off "
                    "the student's token ids"
                )
            return self
        if self.endpoint is not None:
            raise ValueError(
                'teacher.endpoint is only for teacher.backend = "openai_compat"; the '
                '"tinker" backend reaches the teacher through the Tinker service, so '
                'drop teacher.endpoint or set backend = "openai_compat"'
            )
        if self.alignment != "same_tokenizer":
            raise ValueError(
                'teacher.backend = "tinker" requires teacher.alignment = '
                f'"same_tokenizer", got {self.alignment!r}: chunk alignment is only '
                'implemented for the "openai_compat" backend, so either drop '
                'teacher.alignment or set backend = "openai_compat" with an endpoint '
                "and tokenizer"
            )
        return self


class HarborConfig(BaseModel):
    """How rollouts are produced: harbor's terminus-2 agent on harbor tasks.

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


class Tau2Config(BaseModel):
    """How rollouts are produced: tau2-bench's own harness on real tau2 tasks.

    The alternative to `[harbor]` (exactly one of the two selects the run's rollout
    source). Every episode runs Sierra's real benchmark unmodified - tau2's own
    `llm_agent`, LLM user simulator, orchestrator, and deterministic evaluator - via
    one `tau2 run` subprocess per (task x attempt). The agent's LLM calls come back
    into this process through a local OpenAI-compatible proxy backed by a per-episode
    `TinkerChatProvider`, which is what records the student's exact sampled token
    spans (`wmo.distill.tau2`).

    tau2 needs Python 3.12+ and a heavy dependency tree, so it lives in its own venv
    (see `packages/environment-capture/tau-bench/README.md`) and wmo never imports
    it; `tau2_bin` points into that venv.

    Attempts per task are NOT configured here: training rollouts use
    `train.group_size` and evals use `eval.k` / `gate.k`, exactly as with harbor.
    """

    model_config = ConfigDict(extra="forbid")

    tau2_bin: str
    """Path to the `tau2` CLI executable inside its own venv."""

    data_dir: str
    """The tau2 data directory, exported as TAU2_DATA_DIR to every runner."""

    user_llm: str = "azure/gpt-5.4-mini"
    """The user simulator's litellm model spec, pinned across every arm of a study.

    The user simulator is part of the environment: two runs facing different
    simulated customers are runs on different benchmarks. The default is the pin
    the 720-episode sim-to-real study ran against. The azure/ route needs
    AZURE_API_KEY, AZURE_API_BASE, and AZURE_API_VERSION in the environment; the
    preflight checks for them by name."""

    user_llm_args: dict[str, JsonValue] = Field(default_factory=dict)
    """Extra litellm kwargs for the user simulator, forwarded verbatim as
    `--user-llm-args`. Empty (the pinned default) sends `{}`, which is what the
    prior tau2 captures ran with."""

    backend: Literal["local", "e2b"] = "local"
    """Where the tau2 runner subprocesses execute.

    `local` runs them in the tau2 venv on this machine (the environment is an
    in-memory JSON DB; the heavy lifting is remote LLM calls either way). `e2b`
    isolates each runner in an E2B sandbox per the harbor pattern."""

    max_errors: int = Field(default=10, ge=1)
    """tau2's per-episode tool/format error budget (`--max-errors`); its default."""

    episode_retries: int = Field(default=1, ge=0)
    """Fresh-episode retries after an infrastructure failure (no verifier evidence).

    The retry lives HERE, not in tau2's runner (`--max-retries` is pinned to 0):
    tau2's own retry re-runs the simulation into the same span sink, so training
    datums would carry an abandoned attempt's tokens under another attempt's
    reward. Each wmo-level retry starts a fresh sink and a fresh recorder. One
    retry absorbs the transient user-sim API blips observed in practice; an
    episode that still ends without evidence stays an `infra_failed` record."""


class RolloutConfig(BaseModel):
    """Per-episode rollout limits."""

    model_config = ConfigDict(extra="forbid")

    max_turns: int = Field(default=100, ge=1)
    """Episode turn cap; passed to terminus-2 as `max_turns` (f.k.a. `max_episodes`).

    100, not the harness-wide 20 that world-model closed-loop eval uses. Terminus-2 is unbounded by
    default (`_max_episodes = 1_000_000`) and TerminalBench-2 tasks routinely need 50 to 200 turns;
    at 20 the cap fired mid-tool-call on 45% of Ultra trials and 12% of Super trials, and every one
    of those scored reward 0. 100 is the point where a cap still bounds a looping agent (a real
    100-turn episode also runs into `episode_timeout_s` first) without being the thing that decides
    the score. Also pinned into the harness doc's `param:max-turns`, which no longer steers the
    rollout agent but still keys the per-candidate harbor job dir."""

    episode_timeout_s: float = Field(default=1800.0, gt=0)
    """Per-episode wall budget.

    Terminus-2 has no internal wall clock, so this is applied as harbor's own agent-phase timeout
    (`AgentConfig.override_timeout_sec`), which raises `AgentTimeoutError`; harbor swallows that
    and still verifies the work, so an episode cut off by the clock stays a graded trial. Without
    it every rollout and eval wave inherited the task's declared agent timeout. 1800s covers the
    real work in this suite (a single `apt-get install build-essential` observation was 45,131 chars
    and tens of seconds; the suite also compiles CompCert and boots QEMU) while staying inside the
    E2B sandbox lease ceiling (`MAX_EVAL_EPISODE_LIFETIME_S`, 3600s, minus one lease of cleanup
    headroom)."""

    context_budget_tokens: int = Field(default=65536, ge=1024)
    """Context cap: episodes where any call's prompt plus sampled tokens exceed
    it are dropped whole from training (`build_datums`), and the cost estimate
    caps per-episode tokens here.

    Also the context window the rollout agent's LLM is built with (terminus-2's
    `TinkerLLM(context_limit=...)`), so the agent measures its own prompts against the real
    serving limit instead of harbor's 32,000 default. Keep it at least `sampling.max_tokens`
    below the served window, or a full-budget prompt plus its output cap exceeds the window and
    the sampling call 400s."""

    renderers: dict[str, str] = Field(default_factory=dict)
    """Per-base-model override of the chat renderer terminus-2's `TinkerLLM` builds prompts with,
    keyed by the base model name (`student.base_model`, `teacher.model`). A model with no entry
    auto-discovers its renderer through
    `tinker_cookbook.model_info.get_recommended_renderer_name`.

    Keyed per model, not one value, because a run samples MORE than one base model: the student's
    rollouts, plus the teacher's own rollouts for the warmup collection and the teacher baseline
    eval. A Nemotron Nano student and a Nemotron Ultra teacher need different renderer names.

    Exists because the auto-discovered renderer of every REASONING model in this lineup is
    unusable under terminus-2 as the cookbook ships it, measured offline against the real
    tokenizers. Terminus-2 keeps only `parse_response(...)["content"]` in its chat history, and
    `nemotron3`, `nemotron3_ultra` and `qwen3_5` (the auto renderer for both Qwen3.5 and Qwen3.6)
    return that content as a LIST of thinking/text parts, which harbor's
    `TerminusJSONPlainParser` raises `TypeError` on, killing the trial before it grades anything.
    Those same renderers also strip the thinking block when they re-render an assistant turn that
    is no longer last, so turn N+1's prompt diverges from turn N's and EVERY turn becomes its own
    datum fragment, at a cost quadratic in turn count (2.7x the tokens at 6 turns, 7.8x at 20,
    15.1x at 40).

    What to name here: the wmo VERBATIM renderer for every Nemotron-3 or Qwen3.5/3.6 model in the
    run. `wmo.distill.renderers` wraps each reasoning renderer so the parser sees a plain `str`
    of the action text while the model's own reasoning is replayed into history token for token,
    which fixes both failures at once and keeps the episode a single datum:

        [rollout.renderers]
        "Qwen/Qwen3.5-9B" = "wmo/qwen3_5_verbatim"
        "Qwen/Qwen3.6-27B" = "wmo/qwen3_5_verbatim"
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16" = "wmo/nemotron3_verbatim"
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16" = "wmo/nemotron3_ultra_verbatim"

    The disable-thinking renderers (`nemotron3_disable_thinking`,
    `nemotron3_ultra_disable_thinking`) also hold the prefix property and parse, but they buy that
    by throwing the reasoning away, which is exactly the behavior distillation is meant to teach.
    Name one only when a run deliberately trains a non-reasoning policy."""

    @field_validator("renderers")
    @classmethod
    def _reject_blank_renderers(cls, value: dict[str, str]) -> dict[str, str]:
        """Reject blank or unknown renderer entries at config load, not at trial 1.

        Args:
            value: The raw `model -> renderer name` mapping.

        Returns:
            The mapping unchanged when every entry names a renderer the cookbook can build.

        Raises:
            ValueError: If either side of an entry is blank, or the renderer name is one
                `tinker_cookbook.renderers.get_renderer` does not know.
            ImportError: If the distill extra is missing, so no name can be checked.
        """
        for model, renderer in value.items():
            if not model.strip() or not renderer.strip():
                raise ValueError(
                    "rollout.renderers maps a base model name to a tinker-cookbook renderer "
                    f"name; got the entry {model!r} = {renderer!r}, which has an empty side"
                )
        if not value:
            return value
        # Deferred: wmo.distill.renderers subclasses cookbook classes at module scope, and
        # wmo.distill.config is imported by the CLI, which must work without the distill extra.
        try:
            from wmo.distill.renderers import WMO_RENDERERS, is_known_renderer
        except ImportError as exc:
            raise ImportError(MISSING_DISTILL_EXTRA) from exc
        for model, renderer in value.items():
            if is_known_renderer(renderer):
                continue
            raise ValueError(
                f"rollout.renderers names the renderer {renderer!r} for {model!r}, which "
                "tinker-cookbook cannot build; a name that does not resolve would only fail on "
                "the run's first rollout. Use one of the wmo verbatim renderers "
                f"({', '.join(sorted(WMO_RENDERERS))}) or a built-in cookbook name "
                "(qwen3, qwen3_5, nemotron3, nemotron3_ultra, their *_disable_thinking "
                "variants, ...)"
            )
        return value

    compaction: bool = False
    """Whether terminus-2 may summarize its own history to stay inside the window.

    Maps to terminus-2's `enable_summarize`, whose own default is True; this
    defaults False because it is only SAFE under a history-editing renderer.

    Compaction rewrites the prompt prefix mid-episode, so it is incompatible
    with the prefix property and therefore with one-datum-per-episode: under a
    verbatim renderer it would turn amortized prefill into a full re-prefill
    per turn and stop sampled spans appearing verbatim in the episode tokens.
    That is why it used to be rejected outright.

    Under a strip-history renderer the prefix property is ALREADY gone by
    design (the episode is one datum per turn either way), so compaction costs
    nothing structurally and buys back the episodes that would otherwise die.
    Two things make it safe there, both verified in harbor's source rather than
    assumed:

    - `Chat.reset_response_chain()` clears only `_last_response_id`; nothing
      clears `_prompt_token_ids_list`/`_completion_token_ids_list`, so every
      real turn is still recorded even though `chat._messages` is truncated to
      three. Harbor's "rollout details will be incomplete" warning is about the
      single-linear-trajectory assumption, which strip-history already broke.
    - the three summarization subagents call `self._llm.call(...)` directly,
      bypassing the `Chat`, so their turns never enter the training data.

    Leaving it off costs whole trials: on the 17-task holdout with reasoning
    dropped from history, terminal output alone still pushed 24% of teacher
    episodes past the window, and an overflow FAILS the trial rather than
    ending the episode.
    """

    @model_validator(mode="after")
    def _compaction_needs_a_history_editing_renderer(self) -> RolloutConfig:
        """Reject compaction when a renderer is relying on the prefix property.

        Combining the two silently produces the worst case: the datum builder
        would still try to merge an episode whose prefix compaction has
        rewritten, so spans would stop matching the episode tokens.
        """
        if not self.compaction:
            return self
        from wmo.distill.renderers import VERBATIM_RENDERERS

        verbatim = sorted(
            f"{model} = {name!r}"
            for model, name in self.renderers.items()
            if name in VERBATIM_RENDERERS
        )
        if verbatim:
            raise ValueError(
                "rollout.compaction = true cannot be combined with a verbatim renderer "
                f"({'; '.join(verbatim)}): compaction rewrites the token prefix mid-episode, "
                "which breaks the prefix property those renderers exist to preserve. Either "
                "set compaction = false, or point [rollout.renderers] at a strip-history "
                "renderer, whose episodes are already one datum per turn"
            )
        return self


class TrainConfig(BaseModel):
    """Optimizer-loop schedule and batch shape."""

    model_config = ConfigDict(extra="forbid")

    steps: int = Field(default=40, ge=0)
    """On-policy optimizer steps. 0 makes the run WARMUP-ONLY: the supervised
    phase is the whole training run (teacher rollouts -> keep-filter -> CE),
    then the student-after eval and the gate run as usual. Rejected unless
    `warmup.steps > 0`, since a run with neither phase trains nothing and
    would put a no-op behind the gate."""

    tasks_per_batch: int = Field(default=8, ge=1)
    group_size: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    loss: Literal["importance_sampling", "ppo", "topk_ce"] = "importance_sampling"
    """The distillation loss.

    `importance_sampling` (the default) trains per-token reverse-KL
    advantages over the student's realized tokens.

    `ppo` trains those same advantages (identical wire datums) under the
    service's clipped-ratio surrogate: the update is bounded by clipping the
    policy ratio rather than by bounding the advantage, which is the
    OpenClaw-RL / Slime formulation, so pair it with `advantage_clip` unset
    and `center_advantages = false`. The clip epsilon is the service default
    (reported symmetric 0.2); no `loss_fn_config` is sent. Note what the clip
    can and cannot do here: this loop takes ONE forward/backward plus one
    optimizer step per batch and refreshes the sampler every
    `sampler_refresh_every` steps, so at `sampler_refresh_every = 1` the
    ratio is ~1 for every token (only sampler-vs-trainer numerical drift,
    ~0.08 nat) and the clip almost never binds. It starts protecting the run
    when the policy drifts from the sampler that produced the batch (a larger
    `sampler_refresh_every`, or reused batches).

    `topk_ce` trains a weighted cross-entropy over the teacher's top-k
    candidate tokens at every loss position (renormalized teacher probs as
    weights), which carries dense supervision from tokens the student did
    NOT sample at roughly k times the training-token volume."""

    topk: int = Field(default=8, ge=1, le=64)
    """How many teacher candidates per position under `loss = "topk_ce"`
    (ignored by `importance_sampling` and `ppo`). Training volume scales
    linearly with it (k replicated cross_entropy datums per source datum)."""

    advantage_clip: Annotated[float, Field(gt=0)] | None = None
    """Symmetric bound on each per-token advantage,
    `clip(teacher_lp - student_lp, +-advantage_clip)`, applied before any
    centering.

    None (the default) trains the RAW gap, which is the OpenClaw-RL / Slime
    form and what `loss = "ppo"` expects: nothing bounds one token's
    magnitude, and the PPO ratio clip is the regularizer instead. A positive
    value caps outliers (a token the teacher is far more confident about
    than the student can otherwise dominate its batch) at the cost of
    biasing the reverse-KL estimate. `clip_fraction` in the metrics row
    reports how often the bound bit, and is 0.0 whenever clipping is off.
    Ignored by `topk_ce`, which builds no advantages."""

    center_advantages: bool = False
    """Whether to subtract the batch mean over all loss tokens from every
    loss token (after clipping), forcing the batch-mean advantage to zero.

    False (the default) trains the raw uncentered gap: a token's sign says
    whether the teacher liked it more than the student did, not whether it
    beat the batch average, and `advantage_mean` in the metrics row reads
    the objective itself (the mean teacher-minus-student gap) rather than a
    trivial 0. True restores the variance-reduced baseline form, which
    removes the shared offset but pushes DOWN every below-average token even
    when the teacher scored it above the student. Ignored by `topk_ce`.

    Uncentered is the unbiased reverse-KL estimator, and on-policy its mean is
    `-KL(student||teacher) <= 0`, so the average sampled token is pushed down
    and only tokens the teacher likes better than the student are pushed up.
    That is the correct gradient, and also why the mean is worth watching: a
    mean that stops moving toward 0 means the student stopped closing."""

    max_datum_tokens: int = Field(default=65536, ge=1)
    sampler_refresh_every: int = Field(default=1, ge=1)
    save_state_every: int = Field(default=8, ge=1)
    trial_concurrency: int = Field(default=8, ge=1)

    num_substeps: int = Field(default=1, ge=1)
    """Optimizer updates to take per collected rollout batch.

    1 (the default, and what every run before 2026-07-26 did) means ONE
    `forward_backward` over every datum followed by ONE `optim_step`: a whole
    batch of agent rollouts buys a single gradient step. Measured, that was
    **$66 per update**, and a 15-step run would be fifteen updates total --
    for a rank-32 LoRA that is a nudge, not training.

    Splitting the same datums into N shuffled minibatches and taking N updates
    costs NOTHING extra in rollouts, which are 94% of the bill and already
    paid for by the time we get here. At 1,095 datums a 64-datum minibatch is
    17 updates per batch instead of 1.

    The tradeoff is that updates 2..N are OFF-POLICY with respect to the
    weights that sampled the batch -- which is the standard PPO setting and
    the reason `train.loss = "ppo"` has a ratio clip at all. With
    `num_substeps = 1` that clip is inert by construction, because the
    sampling policy and the training policy are the same weights and the ratio
    is identically 1. Raising this is what makes it do its job. Shuffle order
    is seeded per step, so a rerun reproduces the same minibatches.

    Keep it modest relative to the batch: too many updates on one batch drifts
    far from the sampling policy, and our advantages are unclipped
    (`advantage_clip` unset), so the ratio clip is the only bound in the
    system.
    """

    rollout_max_turns: int | None = Field(default=None, ge=1)
    """Turn cap for TRAINING rollouts only; `None` uses `rollout.max_turns`.

    Evals deliberately keep `rollout.max_turns`, because student-before and
    student-after must run under identical settings or the paired delta
    measures the config change instead of the training. Capping only the
    training side is safe in a way it would NOT be for RL: on-policy
    distillation's signal is the per-token teacher-minus-student gap on
    whatever the student did, so a truncated episode is still valid
    supervision for the turns it did take. It does not need the episode to
    succeed.

    This is the cost lever that matters, because both cost and latency grow
    superlinearly in turns: every turn re-prefills the whole conversation, and
    under a history-editing renderer every turn is also its own teacher-scored
    datum. Measured over 193 real TB2 episodes (Qwen3.6-27B and Qwen3.5-9B):

        cap   solved episodes kept   input tokens kept   est USD/step
         10            34%                   6%                14
         20            63%                  18%                44
         30            81%                  33%                79
        100           100%                 100%               242

    tinker-cookbook's own harbor multi-turn distillation recipe pins
    `max_turns=10` with `max_trajectory_tokens=24576` for terminal-bench-2.0,
    which is the same reconciliation reached independently.
    """

    log_sample_rollouts: int = Field(default=2, ge=0)
    """How many sample episodes each batch renders to human-readable text:
    after every training step, the warmup collection, and each eval batch,
    the first N span-bearing trials are decoded WITH the chat template's
    special tokens and written to the run dir's `samples/` files plus the
    tracker's samples table (see `wmo.distill.samples`). 0 disables."""


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
    loop, the teacher runs the same terminus-2 rollouts on the TRAIN tasks, its kept trials
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

    defer_baselines: bool = False
    """Start training immediately and resolve the gate baselines at finalize.

    The two baselines feed only `gate_distillation`; nothing in the training
    loop reads them. Measured up front they are pure latency, so this moves
    them off the critical path and lets a SEPARATE process measure them while
    training runs.

    That process needs its own `HOME`. Harbor's task cache is a hardcoded
    `~/.cache/harbor` with no env override, and `_copy_task_source_to_target`
    does an unconditional `rmtree` + `copytree` of the task directory even when
    the cache is already warm -- so two concurrent harbor jobs under one HOME
    delete each other's tasks out from under a running trial.

    Requires `student_baseline_from`. The student-before baseline is the one
    measurement whose meaning depends on WHEN it runs: at finalize the LoRA has
    trained, so re-measuring it there would quietly report the trained student
    as the "before" number and collapse the very delta the run exists to show.
    Deferred, it must be imported. The teacher is a fixed base model, so it may
    be measured late without harm.
    """

    @model_validator(mode="after")
    def _deferred_baselines_need_an_imported_student(self) -> EvalConfig:
        """Refuse to defer without a student-before to import."""
        if self.defer_baselines and not self.student_baseline_from:
            raise ValueError(
                "eval.defer_baselines = true requires eval.student_baseline_from: deferring "
                "runs the student-before baseline AFTER training, when the LoRA has already "
                "moved, so it would measure the TRAINED student and report it as the "
                "pre-training number. Point it at the concurrent baseline run's "
                "evals/baseline-student-before.json"
            )
        return self


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


class BudgetConfig(BaseModel):
    """Optional hard USD budget for the whole run."""

    model_config = ConfigDict(extra="forbid")

    max_usd: Annotated[float, Field(gt=0)] | None = None


PROBE_BASELINE_ENTROPY_NATS = 0.181
"""Batch-pooled entropy proxy measured on healthy, UNTRAINED Super-120B weights.

Pooled over the 356,122 sampled tokens of a live 48-episode TerminalBench-2
probe at `sampling.temperature = 0.7`
(`.wmo/distill-runs/probe-scaffold/tokens/step-0000/*.jsonl`, 47 episodes that
recorded spans). Per-episode: mean 0.200, p50 0.184, p10 0.135, min 0.082.
Recorded here as documentation for the `[tripwire]` defaults; nothing reads it
as a threshold, because each run measures its own baseline (see
`TripwireConfig`).
"""

PROBE_BASELINE_EPISODE_TOKENS = 7577
"""Batch-pooled sampled tokens per episode on the same probe.

Per-episode: p50 5,543, p10 2,242, p90 16,932, min 349, max 30,869. The 88x
spread between the shortest and longest healthy episode is why the tripwires
pool over the batch and why the length fractions are as loose as they are.
"""


class TripwireConfig(BaseModel):
    """Degeneration tripwires on the student's own sampled tokens.

    Every threshold is a FRACTION of the baseline this run measures at its own
    first training step, never an absolute number. Two sibling cross-tokenizer
    lanes died of a degeneration no KL curve shows (KL can fall while the
    policy collapses): one had generation length collapse 50x (2,866 to about
    50 tokens) while task accuracy fell 0.813 to 0.596, the other reports the
    mirror pathology, "pure KL gives EOS no gradient, so the student never
    learns to stop". `entropy_per_token` and `mean_generation_tokens` on every
    metrics row are what make both visible here.

    Why fractions and not the sibling lane's absolute "entropy below 0.2 nats
    means collapse": our own healthy, untrained baseline is
    `PROBE_BASELINE_ENTROPY_NATS` = 0.181 nats/token, so that threshold would
    fire at step 0, before a single gradient step. Terminal-command tokens are
    far more predictable than the math reasoning the sibling measured, and
    sampling at temperature 0.7 biases a sampled-token entropy estimate
    downward on its own (see `wmo.distill.tripwire.policy_health`). A tripwire
    that always fires gets muted, which is strictly worse than no tripwire, so
    do not replace any fraction below with an absolute nats or token count.

    Bounds are TWO-SIDED, as of 2026-07-26. They used to be downside-only, on
    the reasoning that "an episode cap bounds the runaway direction already".
    A measured run refuted that: with `train.rollout_max_turns = 20` in force
    the whole time, `mean_generation_tokens` still reached **5.33x** baseline
    over four steps, because the cap bounds TURNS while each turn grew — 172
    turns pinned the 12,288-token output cap in a single step. The real damage
    was invisible to both bounds: episodes overflowing the context budget are
    dropped whole, so trainable datums fell 1,129 -> 689 and 22 of 64 episodes
    were discarded, while every downside tripwire stayed silent and a human had
    to stop the run by eye. Entropy climbed 1.68x over the same steps, which is
    the signature a sibling lane saw before its model scored 0/30 on AIME with
    78k-character repetition loops.

    So each metric now answers to a floor (`*_frac`, a fraction) AND a ceiling
    (`*_mult`, a multiple). The asymmetry in the defaults is deliberate: a
    policy can legitimately grow somewhat (this student was distilling toward a
    teacher that genuinely takes longer turns, 1,068 median context tokens per
    turn against the student's 770), so the ceilings sit further from 1.0 than
    the floors do.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    """Whether a breach may warn or abort the run.

    False still computes both metrics, still captures and persists the
    baseline, and still records the baseline and the ratio on every metrics row
    (all of that is free, from data the step already holds); it only silences
    the warning and the abort."""

    entropy_warn_frac: float = Field(default=0.5, gt=0.0, le=1.0)
    """Warn when the batch-pooled entropy falls to this fraction of baseline.

    Half the measured 0.181 nats/token is about 0.091, which is below the
    LOWEST single healthy episode on the probe (0.082 is the per-episode min,
    and the pooled batch statistic is far tighter than any single episode)."""

    entropy_kill_frac: float = Field(default=0.3, gt=0.0, le=1.0)
    """Abort (after `kill_consecutive_steps`) at this fraction of baseline.

    About 0.054 nats/token against the measured baseline. The sibling lane's
    collapsing steps read 0.06, 0.05 and 0.13 nats against a pre-registered
    floor of 0.2, i.e. ratios of roughly 0.25 to 0.65 of their own threshold,
    so a collapse of that severity trips this while ordinary drift does not."""

    length_warn_frac: float = Field(default=0.5, gt=0.0, le=1.0)
    """Warn when the batch's mean sampled tokens per episode falls to this
    fraction of baseline (about 3,789 tokens against the measured 7,577)."""

    length_kill_frac: float = Field(default=0.25, gt=0.0, le=1.0)
    """Abort (after `kill_consecutive_steps`) at this fraction of baseline.

    A quarter of the measured 7,577 is about 1,894 tokens, still under the
    probe's per-episode p10 of 2,242: the whole batch has to average shorter
    than nine out of ten healthy episodes. The sibling's 50x collapse (2,866 to
    about 50 tokens) is a ratio of 0.017, two orders of magnitude past this."""

    entropy_warn_mult: float = Field(default=1.5, gt=1.0)
    """Warn when the batch-pooled entropy RISES to this multiple of baseline.

    The measured refutation run climbed 0.513 -> 0.862 nats/token, a ratio of
    1.68, and nothing fired. A sibling lane's degenerate model (AIME 0/30, 78k
    repetition loops) climbed 0.26 -> 0.82, a ratio of 3.15, and rising entropy
    was its ONLY warning. 1.5 sits below the run we know went wrong."""

    entropy_kill_mult: float = Field(default=2.5, gt=1.0)
    """Abort (after `kill_consecutive_steps`) at this multiple of baseline.

    Above the 1.68 that a run reached while still producing a measurable (if
    insignificant) gain, and below the 3.15 of the run that ended degenerate,
    so this kills the established pathology without killing a run that is still
    arguably learning."""

    length_warn_mult: float = Field(default=2.0, gt=1.0)
    """Warn when mean sampled tokens per episode RISES to this multiple.

    Reached at step 1 of the refutation run (2.31x), which is when a human
    should have been asked to look, and comfortably above the upside of ordinary
    batch-composition noise: the probe resampling that set the floors put the
    downside p0.1 at 0.62 for this batch shape, whose reciprocal is 1.61."""

    length_kill_mult: float = Field(default=3.0, gt=1.0)
    """Abort (after `kill_consecutive_steps`) at this multiple of baseline.

    Chosen against measured damage rather than taste: the refutation run passed
    3.0x at step 2, which is the step where context-overflow drops reached 14 of
    64 episodes and the batch began collapsing. Two consecutive steps past this
    is the point where further spend buys less supervision each step."""

    kill_consecutive_steps: int = Field(default=2, ge=1)
    """Consecutive kill-level steps tolerated before the run aborts.

    Same reasoning as `MAX_CONSECUTIVE_EMPTY_STEPS`: one batch is a small task
    draw and can be transiently short or flat, two in a row is a trend. A step
    that is not at kill level resets the streak."""

    @model_validator(mode="after")
    def _check_kill_below_warn(self) -> TripwireConfig:
        """Keep every kill threshold on the far side of its warn threshold.

        Both directions, and the comparison inverts between them: a kill FLOOR
        must sit at or below its warn floor, while a kill CEILING must sit at or
        above its warn ceiling. Getting that inversion wrong is the whole reason
        this is checked rather than left to the reader.

        Returns:
            This config, unchanged, when each kill threshold is reachable only
            through its warn threshold.

        Raises:
            ValueError: If a kill threshold sits on the near side of its warn
                threshold, which would abort the run without ever having warned
                about it.
        """
        floors = (
            ("entropy", self.entropy_kill_frac, self.entropy_warn_frac),
            ("length", self.length_kill_frac, self.length_warn_frac),
        )
        for metric, kill, warn in floors:
            if kill > warn:
                raise ValueError(
                    f"tripwire.{metric}_kill_frac ({kill}) must be <= "
                    f"tripwire.{metric}_warn_frac ({warn}), or the run would abort at a "
                    "ratio it never warned about first"
                )
        ceilings = (
            ("entropy", self.entropy_kill_mult, self.entropy_warn_mult),
            ("length", self.length_kill_mult, self.length_warn_mult),
        )
        for metric, kill, warn in ceilings:
            if kill < warn:
                raise ValueError(
                    f"tripwire.{metric}_kill_mult ({kill}) must be >= "
                    f"tripwire.{metric}_warn_mult ({warn}), or the run would abort at a "
                    "ratio it never warned about first"
                )
        return self


class WandbConfig(BaseModel):
    """Optional Weights & Biases run tracking (off by default).

    Enabling it requires the wandb SDK (`uv sync --extra distill`) and
    credentials (WANDB_API_KEY or a prior `wandb login`); both are checked
    before the run spends anything (see `wmo.distill.tracking`).
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    project: str = "wmo-distill"
    entity: str | None = None
    run_name: str | None = None
    """The wandb run name; None derives one from the agent name and run dir."""

    tags: list[str] = Field(default_factory=list)


class DistillConfig(BaseModel):
    """Top-level configuration for one distillation run.

    The student and teacher sections are required, plus EXACTLY ONE rollout
    source section: `[harbor]` (terminus-2 on harbor benchmark tasks) or
    `[tau2]` (tau2-bench's own harness on real tau2 tasks). Every other
    section has complete defaults and may be omitted from the TOML file.
    """

    model_config = ConfigDict(extra="forbid")

    student: StudentConfig
    teacher: TeacherConfig
    harbor: HarborConfig | None = None
    tau2: Tau2Config | None = None
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    warmup: WarmupConfig = Field(default_factory=WarmupConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    gate: GateConfig = Field(default_factory=GateConfig)
    pricing: PricingConfig = Field(default_factory=PricingConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    tripwire: TripwireConfig = Field(default_factory=TripwireConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    @model_validator(mode="after")
    def _check_some_training_phase(self) -> DistillConfig:
        """Reject a run that would train nothing (no OPD steps, no warmup).

        Returns:
            This config, unchanged, when at least one phase trains.

        Raises:
            ValueError: If `train.steps` and `warmup.steps` are both 0.
        """
        if self.train.steps == 0 and self.warmup.steps == 0:
            raise ValueError(
                "train.steps = 0 makes the run warmup-only, but warmup.steps is also 0, "
                "so nothing would train and the gate would measure a no-op; set "
                "warmup.steps > 0 for a warmup-only run or train.steps > 0 for OPD"
            )
        return self

    @model_validator(mode="after")
    def _check_rollout_source(self) -> DistillConfig:
        """Require exactly one rollout source section.

        Returns:
            This config, unchanged, when exactly one of `[harbor]` / `[tau2]` is set.

        Raises:
            ValueError: If both or neither source section is present.
        """
        if (self.harbor is None) == (self.tau2 is None):
            raise ValueError(
                "a distill run needs exactly one rollout source: set [harbor] "
                "(terminus-2 on harbor benchmark tasks) or [tau2] (tau2-bench's own "
                "harness on real tau2 tasks), not "
                + ("both" if self.harbor is not None else "neither")
            )
        if self.tau2 is not None:
            # Both knobs steer harbor's terminus-2 agent and would be silently
            # ignored here, which is exactly the trap this config forbids. The
            # tau2 proxy path renders through each model's auto-discovered
            # cookbook renderer and keeps history verbatim by splicing exact
            # sampled ids, so neither knob has a tau2 meaning.
            if self.rollout.renderers:
                raise ValueError(
                    "[rollout.renderers] is a terminus-2 (harbor) knob and has no effect "
                    "under the [tau2] source, whose proxy renders through each model's "
                    "auto-discovered cookbook renderer; remove the table"
                )
            if self.rollout.compaction:
                raise ValueError(
                    "rollout.compaction is terminus-2's summarizer and has no effect under "
                    "the [tau2] source (tau2's own agent keeps its full history); set it "
                    "false or leave it unset"
                )
        return self

    @property
    def rollout_source(self) -> Literal["harbor", "tau2"]:
        """Which rollout source this run selected (validated to be exactly one)."""
        return "harbor" if self.harbor is not None else "tau2"

    @model_validator(mode="after")
    def _check_cross_tokenizer_loss(self) -> DistillConfig:
        """Reject the losses the cross-tokenizer (chunk-aligned) path cannot express.

        Returns:
            This config, unchanged, when the loss suits the alignment.

        Raises:
            ValueError: If chunk alignment is paired with `topk_ce`.
        """
        if self.teacher.alignment == "chunk" and self.train.loss == "topk_ce":
            raise ValueError(
                'train.loss = "topk_ce" is not supported with teacher.alignment = '
                '"chunk": top-k CE trains the student on the teacher\'s candidate '
                "token ids as targets, and those ids index the TEACHER's vocabulary, "
                "which is a different vocabulary from the student's under chunk "
                "alignment (they name different text); use loss = "
                '"importance_sampling" or "ppo", which only need the teacher\'s '
                "total logprob over each chunk of the student's own tokens"
            )
        return self

    @model_validator(mode="after")
    def _check_renderer_models(self) -> DistillConfig:
        """Reject a `rollout.renderers` key naming a model this run never samples.

        The value side is checked where it is declared (`RolloutConfig`); the KEY
        is the more dangerous typo, because a key that matches nothing is not an
        error anywhere downstream. The lookup simply misses, the model falls back
        to its auto-discovered renderer, and the run dies on trial 1 with the
        failure this setting existed to prevent.

        Returns:
            This config, unchanged, when every key names a sampled base model.

        Raises:
            ValueError: If a key is neither `student.base_model` nor `teacher.model`.
        """
        sampled = {self.student.base_model, self.teacher.model}
        unknown = sorted(set(self.rollout.renderers) - sampled)
        if unknown:
            raise ValueError(
                f"rollout.renderers names the model(s) {unknown} that this run never samples; "
                "the keys are base model names and must be student.base_model "
                f"({self.student.base_model!r}) or teacher.model ({self.teacher.model!r}). An "
                "unmatched key is silently ignored, and the model it meant to fix would fall "
                "back to its auto-discovered renderer"
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
