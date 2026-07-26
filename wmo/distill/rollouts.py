"""Collect one distillation training step's rollouts as real harbor trials.

`collect_rollouts` turns one train batch (task ids x group size) into scored
harbor trials of **harbor's own terminus-2 agent** sampling the Tinker
student, and joins each trial's exact token spans back into `TrialRecord`s.

Terminus-2, not the WMO pi bridge, because the scaffold is not what this run
is measuring: on the same TerminalBench-2 tasks pi took 2-3x the turns
terminus-2's published p50 of 10 (p50 21-30 across four models) with 0%
wasted turns, so the gap is scaffold efficiency, not a bug, and it drove
39-59% of the harness loss. Terminus-2 supports this natively:
`llm_backend="tinker"` builds `harbor.llms.tinker.TinkerLLM`, and
`collect_rollout_details=True` records per-turn `prompt_token_ids`,
`completion_token_ids` and `logprobs` that harbor persists into each trial's
`result.json` (`agent_result.rollout_details`). Those ids ARE the training
targets; nothing is re-encoded from text.

Three consequences of that agent swap are load-bearing here:

- **No token sink directory.** Spans come from harbor's own per-trial
  `result.json`, which the scorer's entry prune deletes together with the
  trial it belongs to, so there is nothing left to keep outside the trial dir.
- **Terminus-2 runs IN-PROCESS**, driving the task environment over tmux, so
  a trial holds one sandbox (the task environment), not two, and the Tinker
  API key must be present in THIS process rather than forwarded into a
  harness sandbox.
- **Terminus-2 has no wall clock of its own**, so `rollout.episode_timeout_s`
  is applied as harbor's agent-phase timeout
  (`AgentConfig.override_timeout_sec`) instead of an agent kwarg.

Two directory choices are unchanged and still load-bearing:

- The harbor jobs dir is FRESH per training step (`run_dir/harbor/step-NNNN`):
  harbor job dirs are keyed by the candidate doc hash only, so reusing one
  jobs dir across steps would resume step N's completed trials as step N+1's
  "results", carrying tokens sampled from the previous weights.
- A job dir left by a PREVIOUS SESSION under different sampler weights is
  wiped before scoring (`_wipe_stale_policy_dir`): sampler paths carry a
  per-session nonce, so such a dir can never satisfy the scorer's strict
  job-config resume check, and its trials sampled a policy this session did
  not restore. Wiping makes a crash-resumed step re-run whole from the
  current weights. A dir whose recorded provider matches (the teacher's
  stable identity) is kept for harbor's native trial-level resume.

The harbor SDK is an optional extra imported lazily here, the same contract
as the CLI's harbor commands; `import wmo.distill.rollouts` succeeds without
it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from wmo.core.types import JsonObject
from wmo.distill.config import DistillConfig
from wmo.distill.tokens import TrialRecord, assemble_harbor_trial_records
from wmo.harness.doc import HarnessDoc
from wmo.harness.runtime import StopReason
from wmo.providers.base import ProviderConfig, ProviderKind

if TYPE_CHECKING:
    from harbor.models.job.config import JobConfig

logger = logging.getLogger(__name__)

MISSING_HARBOR_EXTRA = (
    "the harbor SDK is not installed; distillation rollouts run as real harbor trials. "
    "Run `uv sync --extra harbor` (or `pip install 'world-model-optimizer[harbor]'`) and retry"
)

HARBOR_TERMINUS_2_AGENT_IMPORT_PATH = "harbor.agents.terminus_2.terminus_2:Terminus2"
"""Harbor's own terminus-2 agent, the rollout agent for distillation.

Imported by string through harbor's agent factory (never at module scope here), so
`import wmo.distill.rollouts` keeps working without the harbor extra."""

E2B_SANDBOXES_PER_TRIAL = 1
"""Concurrent E2B sandboxes one `harbor.backend = "e2b"` trial holds at once.

One, not two: terminus-2 runs inside the harbor process and drives the task environment over
tmux, so the only sandbox a trial holds is harbor's task environment. (The pi bridge needed a
second, pooled sandbox to host the harness process; that path is no longer used for
distillation rollouts.) Capacity planning (`wmo.cli.model_app`) multiplies by this."""


def _recorded_provider_config(config_path: Path) -> JsonObject | None:
    """The provider config a persisted harbor job config ran with, or None.

    None means the file is unreadable or not shaped like a scorer-produced
    JobConfig dump; callers leave those dirs alone so the scorer can raise its
    own actionable error instead of evidence being destroyed silently.
    """
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    agents = payload.get("agents")
    if not isinstance(agents, list) or not agents or not isinstance(agents[0], dict):
        return None
    kwargs = agents[0].get("kwargs")
    if not isinstance(kwargs, dict):
        return None
    recorded = kwargs.get("provider_config")
    return recorded if isinstance(recorded, dict) else None


def _wipe_stale_policy_dir(candidate_dir: Path, provider_config: ProviderConfig) -> bool:
    """Wipe a candidate job dir recorded under different provider weights.

    Sampler paths carry a per-session nonce, so a job dir left by a previous
    session (a crash mid-batch, then `--resume`) can never satisfy the
    scorer's strict job-config resume check, and its completed trials sampled
    a policy this session did not restore, so resuming them would mix
    policies. Deleting the dir makes the batch re-run whole from the current
    weights: correct, at the price of re-running that batch's completed
    trials. The trials' recorded token spans go with it, which is right -
    they were sampled from the policy being discarded.

    A recorded provider that MATCHES the current one (the teacher's stable
    identity, whose baseline eval legitimately resumes across sessions) is
    left for harbor's native trial-level resume, as is an unreadable config
    (the scorer raises its own actionable error for those).

    Args:
        candidate_dir: The scorer's deterministic per-candidate job dir.
        provider_config: The provider the batch is about to sample.

    Returns:
        True when the directory was wiped.
    """
    recorded = _recorded_provider_config(candidate_dir / "config.json")
    if recorded is None or recorded == provider_config.model_dump(mode="json"):
        return False
    logger.warning(
        "harbor job dir %s was produced under provider %r, not the current %r; "
        "wiping it so the batch re-runs from the current weights instead of "
        "resuming another policy's trials",
        candidate_dir,
        recorded.get("model"),
        provider_config.model,
    )
    shutil.rmtree(candidate_dir)
    return True


def terminus_2_agent_kwargs(cfg: DistillConfig, provider_config: ProviderConfig) -> JsonObject:
    """The harbor `AgentConfig.kwargs` that make terminus-2 sample the Tinker student.

    Every value here is either a WMO config knob or an invariant the
    tokens-in-tokens-out contract depends on:

    - `llm_backend="tinker"` selects `harbor.llms.tinker.TinkerLLM`, and
      `llm_kwargs.model_path` points it at the EXACT weights being sampled
      (`provider_config.model`, a `tinker://.../sampler_weights/...` path).
      The base model that names the renderer and tokenizer travels separately
      as harbor's `AgentConfig.model_name` (`provider_config.model_type`),
      because `TinkerLLM(model_name=...)` means the base model, not the
      checkpoint. A provider whose `model` IS its `model_type` (the teacher
      sampling a base model directly) sends no `model_path` at all.
    - `collect_rollout_details=True` is what records `prompt_token_ids`,
      `completion_token_ids` and `logprobs` per turn. Without it a trial
      produces rewards and no training data at all.
    - `enable_summarize` follows `rollout.compaction`, which defaults False and
      which `RolloutConfig` only allows to be true under a history-editing
      renderer. Off, a context overflow raises and the trial FAILS on a
      recorded `ContextLengthExceededError` -- it does not score zero, it
      leaves the denominator. On, terminus-2 compacts and the episode
      continues; under a strip-history renderer that costs nothing structural,
      because every real turn is still recorded (the summarization subagents
      bypass the `Chat` entirely) and the episode was already one datum per
      turn. See `RolloutConfig.compaction` for why this is not the blanket
      rejection it used to be.
    - `max_turns` is terminus-2's `max_episodes`, and
      `llm_kwargs.context_limit` / `max_tokens` are the context and output
      budgets the agent measures itself against.
    - `llm_kwargs.renderer_name` is sent only when `rollout.renderers` names
      one for THIS provider's base model, so a run whose teacher rollouts use
      a different base model than the student still gets each model's own
      renderer. The wmo verbatim renderers are registered with the cookbook
      here, because this is the one call site every rollout path shares, and
      terminus-2 resolves the name from that registry in this process.

    Args:
        cfg: The validated run config.
        provider_config: The provider the batch samples; must be the tinker
            kind, since only Tinker sampling records token ids and logprobs.

    Returns:
        The kwargs dict, safe to pass as the scorer's `extra_agent_kwargs`
        (it collides with none of the scorer-owned keys).

    Raises:
        ValueError: If the provider is not the tinker kind.
    """
    if provider_config.kind is not ProviderKind.TINKER:
        raise ValueError(
            "distillation rollouts must sample through Tinker so the student's exact token "
            f"spans are recorded, got provider kind {provider_config.kind.value!r}; configure "
            "the worker provider with kind 'tinker'"
        )
    llm_kwargs: JsonObject = {
        "max_tokens": cfg.sampling.max_tokens,
        "context_limit": cfg.rollout.context_budget_tokens,
        "output_limit": cfg.sampling.max_tokens,
    }
    if provider_config.model != provider_config.model_type:
        llm_kwargs["model_path"] = provider_config.model
    renderer_name = cfg.rollout.renderers.get(provider_config.model_type)
    if renderer_name is not None:
        llm_kwargs["renderer_name"] = renderer_name
    # Terminus-2 resolves `renderer_name` through the cookbook's GLOBAL registry, inside the
    # TinkerLLM it builds in this process. This is the single place every rollout path (training,
    # warmup collection, eval waves) passes through, so registering the wmo verbatim renderers
    # here is what makes a `[rollout.renderers]` entry naming one resolvable. The import is
    # deferred because wmo.distill.renderers subclasses cookbook classes at module scope, and
    # `import wmo.distill.rollouts` must keep working without the distill extra (the CLI imports
    # it eagerly). Registration is idempotent.
    from wmo.distill.renderers import register_wmo_renderers

    register_wmo_renderers()
    _share_harbor_tinker_service_client()
    return {
        "llm_backend": "tinker",
        "llm_kwargs": llm_kwargs,
        "collect_rollout_details": True,
        "temperature": cfg.sampling.temperature,
        "max_turns": cfg.rollout.max_turns,
        "enable_summarize": cfg.rollout.compaction,
        # The cap is deliberate here, so terminus-2's "consider removing this limit" warning is
        # noise on every trial.
        "suppress_max_turns_warning": True,
    }


_HARBOR_TINKER_CLIENT_SHARED = False


def _share_harbor_tinker_service_client() -> None:
    """Make harbor's `TinkerLLM` reuse wmo's process-wide `ServiceClient`.

    Harbor's `TinkerLLM._ensure_client` does `tinker.ServiceClient()` per
    instance and never closes it, and terminus-2 builds one `TinkerLLM` per
    TRIAL. The SDK's service client starts a heartbeat task that strongly
    references its holder, so each one pins a live server-side session for the
    rest of the process -- the exact leak `wmo.providers.tinker.
    shared_service_client` exists to prevent for wmo's own call paths. Harbor's
    class was the one path that still bypassed it.

    Measured cost of not doing this, on a 64-episode-per-step run: step 0 clean,
    step 1 three failures, step 2 THIRTY-ONE trials rejected with
    `400 Too many active sessions. Please ensure you are not creating extra
    ServiceClient objects`, halving that step's datums. The cap is ~240
    sessions and each step burns 64, so the failure arrives on schedule and
    worsens every step.

    Only the client construction changes; the sampling-client selection below
    it is harbor's own logic, reproduced exactly. Idempotent, and a no-op if
    harbor's internals move -- a rollout that samples successfully matters more
    than this optimization, so a shape mismatch leaves the original in place.
    """
    global _HARBOR_TINKER_CLIENT_SHARED
    if _HARBOR_TINKER_CLIENT_SHARED:
        return
    from harbor.llms.tinker import TinkerLLM

    from wmo.providers import tinker as tinker_provider

    if not hasattr(TinkerLLM, "_ensure_client"):  # pragma: no cover - upstream moved
        logger.warning(
            "harbor's TinkerLLM has no _ensure_client; leaving its service-client "
            "construction alone. Watch for '400 Too many active sessions' errors"
        )
        _HARBOR_TINKER_CLIENT_SHARED = True
        return

    async def _ensure_client_shared(self: object) -> object:
        """Harbor's `_ensure_client`, but on the shared service client."""
        existing = getattr(self, "_sampling_client", None)
        if existing is not None:
            return existing
        # Resolved through the MODULE, not captured at patch time, so
        # `rebuild_shared_service_client()` (the wedge-recovery path) is honoured.
        service = tinker_provider.shared_service_client()
        self._service_client = service  # ty: ignore[unresolved-attribute]
        model_path = getattr(self, "_model_path", None)
        if model_path:
            client = await service.create_sampling_client_async(model_path=model_path)
        else:
            client = await service.create_sampling_client_async(
                base_model=self._model_name  # ty: ignore[unresolved-attribute]
            )
        self._sampling_client = client  # ty: ignore[unresolved-attribute]
        return client

    TinkerLLM._ensure_client = _ensure_client_shared  # ty: ignore[invalid-assignment]
    _HARBOR_TINKER_CLIENT_SHARED = True
    logger.info("harbor TinkerLLM patched to share wmo's process-wide tinker ServiceClient")


class RolloutStats(BaseModel):
    """Aggregate health metrics for one collected rollout batch."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trials: int = Field(ge=0)
    trials_with_spans: int = Field(ge=0)
    solve_rate: float = Field(ge=0.0, le=1.0)
    """Fraction of EXECUTED trials whose verifier reward passed (metrics/gating signal).

    Infrastructure failures are excluded from the denominator: a trial with no verifier evidence
    behind its stand-in 0.0 is an UNKNOWN outcome, and counting it as a task failure biases this
    rate in whichever direction the failures fell. Both directions have been measured: three Super
    `student-before` baselines were reported as 0.0% from 51/51 rate-limited trials, and 2 of 48
    TerminalBench-2 probe trials whose verifier timed out on submitted work held a probe at 20.8%
    when its gradeable denominator was 46. 0.0 when nothing executed (`executed_trials == 0`),
    which callers must treat as a null measurement rather than a score."""

    graded_solve_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    """Mean GRADED test-pass score over the trials that have one: the power metric beside
    `solve_rate`.

    Same trials, read at test resolution instead of the benchmark's one bit (see
    `wmo.harness.scoring.GradedTests`). `solve_rate` stays the headline because binary IS the
    benchmark's definition of success; this exists because a small binary holdout cannot resolve a
    real effect. On the 48-episode TerminalBench-2 probe it reads 0.319 against a 0.217 binary solve
    rate, and it moves 2 of 12 tasks off a flat 0.00 where no improvement could ever have shown up.

    Denominator: trials that are NOT `infra_failed` AND carry a readable test report
    (`graded_trials`). A verifier that timed out wrote no report either, so such a trial is excluded
    here exactly as it is from `solve_rate`, never averaged in as 0.0. 0.0 when there are none,
    which callers must read as a null measurement, not a score.

    Coarse, not continuous: the probe's tasks carried 1 to 6 tests, so most trial scores are 0, 1/2,
    or 1, and a single-test task is exactly as binary as its reward."""

    graded_trials: int = Field(default=0, ge=0)
    """Gradeable trials that carried a readable test report: the `graded_solve_rate` denominator.

    Below `executed_trials` means some graded trials produced a reward but no parseable test report,
    so the two rates ran over different trial sets; the gap is what says so instead of a fabricated
    0.0 hiding it."""

    raw_solve_rate: float = Field(ge=0.0, le=1.0)
    """Passing trials over ALL trials, infra failures included.

    What the training path actually optimizes against (`missing_reward="zero"` makes a dead trial a
    failed trial for advantage estimation), kept alongside `solve_rate` so the two can be compared
    instead of one silently standing in for the other."""

    executed_trials: int = Field(ge=0)
    """Trials that produced verifier evidence (`trials` minus `infra_failed_trials`)."""

    infra_failed_trials: int = Field(ge=0)
    """Trials with no verifier evidence, so no measured outcome.

    One count over two causes, which share this denominator treatment but not their diagnosis:
    the agent never ran (sandbox creation, rate limit, transport death) or the agent ran and was
    never graded (verifier timeout, unwritten/unparseable reward file). The per-cell note the
    scorer writes (`infra-failure: <exception type>; ...`) is what tells them apart."""

    empty_span_trials: int = Field(ge=0)
    """Trials that recorded no token span (died before the first completion)."""

    truncated_spans: int = Field(default=0, ge=0)
    """Turns that sampled the full output cap, so the model was cut off mid-answer.

    Nothing upstream reports this. Harbor's `TinkerLLM` means to
    (`harbor/llms/tinker.py:243` raises `OutputLengthExceededError` when a
    response hit `max_tokens`), but it gates on `not parse_success`, and
    `parse_success` is a `ParseTermination` StrEnum whose members are ALL
    non-empty strings and therefore ALL truthy. The guard can never fire, so a
    turn cut off at the cap flows on as an ordinary turn: the agent reads a
    half-written action, the parser reports an error, and the episode continues
    as if the model had simply answered badly.

    A truncated turn also has no stop token, so it cannot be replayed verbatim
    (`wmo.distill.renderers`) and shows up downstream as a prefix break, which
    means a fragmented episode and multiplied teacher-scoring cost. Counted per
    SPAN, since one bad turn does not spoil the episode's other turns; a rising
    count means `sampling.max_tokens` is too low for the reasoning these tasks
    provoke, not that the model got worse.

    0 on records assembled before this field existed."""

    truncated_span_trials: int = Field(default=0, ge=0)
    """Trials carrying at least one `truncated_spans` turn: the episode-level view."""

    stop_reason_counts: dict[str, int] = Field(default_factory=dict)
    """How many trials ended on each recorded stop reason (`"unknown"` when no trace was readable).

    The per-reason breakdown behind `scaffold_loss_rate`: `max_turns` (turn cap), `budget` (wall
    clock), `no_tool_call`, `output_truncated`, `unparsed_tool_call`, `provider_error`, and
    `submitted`."""

    scaffold_loss_rate: float = Field(ge=0.0, le=1.0)
    """Share of trials WITH A READABLE STOP REASON that never reached an explicit `submit`.

    The headline number of the pi/Nemotron-3 scaffold audit: it was 88.8% for Super and 92.2% for
    Ultra, every one of those trials scored reward 0, and nothing in the metrics surfaced it. An
    episode the harness cut off measures where the guillotine fell, not what the model can do, so
    this belongs beside every solve rate. 0.0 when no trial reported a stop reason.

    Deliberately NOT the `executed` denominator that `solve_rate` uses, because the two answer
    different questions. "Did the harness cut this episode off?" is answered by the stop reason
    alone and needs no verifier; "did the model solve it?" needs a grade. An episode that reached
    `submit` and then had its VERIFIER time out is a scaffold SUCCESS whose task outcome is
    unknown, so it belongs in this denominator (as a non-loss) while being excluded from
    `solve_rate`. Sharing one denominator conflated them and inflated this rate: on the 48-episode
    probe it read 15.22% over 46 gradeable trials when the true scaffold loss was 14.58% (7 of 48
    episodes reported a non-submit stop reason). A trial with no readable trace at all has no
    stop reason and is excluded from both."""

    # TODO: surface TokenRecorder.fallback_count (incremental-prompt re-renders) here;
    # it needs the per-trial span sink format to carry the counter across the
    # agent-process boundary, so it is not trivially wireable today.


UNKNOWN_STOP_REASON = "unknown"
"""Stop-reason bucket for a trial whose run trace was missing or unreadable."""


def rollout_stats(records: Sequence[TrialRecord], *, max_tokens: int) -> RolloutStats:
    """The batch health stats over a set of assembled trial records.

    A pure function of the records and the output cap they sampled under, so a
    batch loaded back from persisted trial records (the warmup-trials manifest)
    reports the same stats its original collection did.

    Args:
        records: The batch's trial records.
        max_tokens: The per-turn output cap the batch sampled under
            (`sampling.max_tokens`); a span that reached it was cut off, which
            nothing upstream reports (see `RolloutStats.truncated_spans`). Pass
            THIS run's cap when re-reading another run's records: the count
            then answers "would these turns be truncated here", which is what
            the reader of this run's metrics needs.

    Returns:
        The aggregate stats. An empty batch, and a batch where every trial was
        an infrastructure failure, both report a 0.0 solve rate over zero
        executed trials: those are null measurements, and the counts are what
        distinguish them. `scaffold_loss_rate` runs over its own denominator
        (trials that reported a stop reason), so a batch whose agents all ran
        but whose verifiers all failed still reports a real scaffold rate.
        `graded_solve_rate` runs over `graded_trials` (gradeable trials with a
        readable test report) and is 0.0 over zero of them, likewise a null
        measurement rather than a score.
    """
    with_spans = sum(1 for record in records if record.spans)
    truncated_per_record = [
        sum(1 for span in record.spans if len(span.sampled_token_ids) >= max_tokens)
        for record in records
    ]
    executed = [record for record in records if not record.infra_failed]
    # The graded rate's own denominator: gradeable trials whose verifier also left a readable test
    # report. A missing report is an absent measurement, so it is excluded rather than scored 0.0,
    # the same rule that keeps an ungradeable trial out of `solve_rate`.
    graded = [record.graded_score for record in executed if record.graded_score is not None]
    counts: dict[str, int] = {}
    for record in records:
        key = record.stop_reason or UNKNOWN_STOP_REASON
        counts[key] = counts.get(key, 0) + 1
    # The scaffold question ("did the harness cut this off?") is answered by the stop reason and
    # needs no verifier, so it gets its own denominator: every trial that reported one. Sharing
    # `executed` with solve_rate dropped submit-then-ungradeable episodes out of a rate they
    # belong in as successes, which inflated it.
    with_stop_reason = [record for record in records if record.stop_reason]
    submitted = sum(
        1 for record in with_stop_reason if record.stop_reason == StopReason.SUBMITTED.value
    )
    return RolloutStats(
        trials=len(records),
        trials_with_spans=with_spans,
        solve_rate=(
            sum(1 for record in executed if record.passed) / len(executed) if executed else 0.0
        ),
        graded_solve_rate=(sum(graded) / len(graded) if graded else 0.0),
        graded_trials=len(graded),
        raw_solve_rate=(
            sum(1 for record in records if record.passed) / len(records) if records else 0.0
        ),
        executed_trials=len(executed),
        infra_failed_trials=len(records) - len(executed),
        empty_span_trials=len(records) - with_spans,
        truncated_spans=sum(truncated_per_record),
        truncated_span_trials=sum(1 for count in truncated_per_record if count),
        stop_reason_counts=dict(sorted(counts.items())),
        scaffold_loss_rate=(
            (len(with_stop_reason) - submitted) / len(with_stop_reason) if with_stop_reason else 0.0
        ),
    )


def _with_agent_wall_budget(template: JobConfig, episode_timeout_s: float) -> JobConfig:
    """Return the job template with the episode wall budget as harbor's agent timeout.

    Terminus-2 has no wall clock of its own (the WMO pi bridge enforced
    `episode_timeout_sec` itself), so the budget has to be harbor's: harbor
    runs `agent.run()` under `asyncio.wait_for(..., agent_timeout_sec)`,
    derived from `AgentConfig.override_timeout_sec` when set and the task's
    own declared agent timeout otherwise. Overriding it here makes every
    rollout and eval wave honor `rollout.episode_timeout_s` instead of
    whatever each task happens to declare.

    `max_timeout_sec` is cleared alongside it, because harbor resolves the
    effective budget as `min(base, max) * multiplier` and a template-provided
    ceiling would silently shorten a deliberately raised budget.

    The scorer's template validation permits both fields (it owns agent
    identity, model, skills, env and kwargs, not the timeouts), so this is a
    supported use of the seam rather than a bypass.

    Args:
        template: The validated harbor `JobConfig`.
        episode_timeout_s: The per-episode wall budget in seconds.

    Returns:
        A revalidated `JobConfig` whose template agents carry the budget.
    """
    from harbor.models.job.config import JobConfig
    from harbor.models.trial.config import AgentConfig

    agents = []
    for agent in template.agents:
        agent_fields = {name: getattr(agent, name) for name in AgentConfig.model_fields}
        agent_fields["override_timeout_sec"] = episode_timeout_s
        agent_fields["max_timeout_sec"] = None
        agents.append(AgentConfig(**agent_fields))
    job_fields = {name: getattr(template, name) for name in JobConfig.model_fields}
    job_fields["agents"] = agents
    return JobConfig(**job_fields)


def collect_rollouts(
    step_index: int,
    task_ids: Sequence[str],
    cfg: DistillConfig,
    harness: HarnessDoc,
    provider_config: ProviderConfig,
    run_dir: Path,
    *,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[list[TrialRecord], RolloutStats]:
    """Run one train batch of harbor trials and join rewards with token spans.

    Each task in `task_ids` runs `cfg.train.group_size` attempts (the
    on-policy group), every trial running harbor's terminus-2 agent against
    the Tinker student, with the per-turn token ids harbor records in the
    trial's `result.json` joined back as the trial's spans.

    This is a synchronous entry point, mirroring how the CLI drives the
    scorer: the async `HarborScorer.create` runs on its own loop and the
    (task x attempts) batch runs inside the blocking `score` call.

    Args:
        step_index: 0-based training step; keys the fresh jobs dir.
        task_ids: Exact task ids for this batch (a subset of the train split).
        cfg: The validated run config (harbor template/backend/reward key,
            group size, trial concurrency).
        harness: The document whose hash keys this candidate's harbor job dir.
            Terminus-2 runs its own scaffold, so the document no longer
            steers the agent; the rollout knobs pinned into it
            (`pin_rollout_params`) still change the job identity, which is
            what keeps a config change from resuming another config's trials.
        provider_config: The student provider config; `model` must point at
            the CURRENT sampler weights (`tinker://` path) and `model_type`
            at the base model naming the renderer/tokenizer.
        run_dir: The distillation run directory.
        should_cancel: Optional cooperative cancellation poll, forwarded to
            the harbor runner.

    Returns:
        The `TrialRecord`s (one per task x attempt, in report order) and the
        batch stats. A trial whose `result.json` records no rollout details
        is kept with empty spans and counted in `empty_span_trials`, never
        dropped: its reward is real batch signal and dropping it would
        silently bias the solve rate.

    Raises:
        ValueError: If `step_index` is negative, the harbor job template
            cannot be loaded/validated, or the provider is not the tinker kind.
        ImportError: If the harbor extra is not installed.
    """
    if step_index < 0:
        raise ValueError(f"step_index must be >= 0, got {step_index}")
    try:
        import yaml
        from harbor.models.job.config import JobConfig

        from wmo.evals.harbor.scorer import HarborScorer
    except ImportError as error:
        raise ImportError(MISSING_HARBOR_EXTRA) from error

    step_name = f"step-{step_index:04d}"
    jobs_dir = run_dir / "harbor" / step_name

    template_path = Path(cfg.harbor.job_template)
    try:
        raw = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(
            f"cannot load the harbor job template from {template_path}: {error}; point "
            "harbor.job_template at a harbor JobConfig YAML/JSON file"
        ) from error
    if not isinstance(raw, dict):
        raise ValueError(f"the harbor job template in {template_path} must be a mapping")
    try:
        template = JobConfig.model_validate({**raw, "jobs_dir": str(jobs_dir)})
    except ValueError as error:
        raise ValueError(
            f"invalid harbor job template {template_path}: {error}; fix the template "
            "so it validates as a harbor JobConfig"
        ) from error
    template = _with_agent_wall_budget(template, cfg.rollout.episode_timeout_s)

    backend = cfg.harbor.backend
    scorer = asyncio.run(
        HarborScorer.create(
            template,
            task_ids,
            provider_config=provider_config,
            reward_key=cfg.harbor.reward_key,
            attempts=cfg.train.group_size,
            task_environment="e2b" if backend == "e2b" else "docker",
            harness_backend=backend,
            # Terminus-2 runs in this process, but the scorer's local backend still pins
            # concurrency to 1 (that guard belongs to the pi runner's shared runner dir);
            # e2b parallelizes up to the configured trial concurrency.
            agent_concurrency=1 if backend == "local" else cfg.train.trial_concurrency,
            harbor_retries=cfg.harbor.retries,
            agent_import_path=HARBOR_TERMINUS_2_AGENT_IMPORT_PATH,
            extra_agent_kwargs=terminus_2_agent_kwargs(cfg, provider_config),
            # TinkerLLM reads the base model, not the checkpoint, to pick its renderer and
            # tokenizer; the checkpoint travels as llm_kwargs.model_path.
            agent_model_name=provider_config.model_type,
            # Kept for the scorer's own bookkeeping and the pi bridge; terminus-2 takes its
            # wall budget from harbor's agent-phase timeout (set on the template above) and
            # its context budget from llm_kwargs.context_limit.
            episode_timeout_s=cfg.rollout.episode_timeout_s,
            context_window=cfg.rollout.context_budget_tokens,
            # A trial with no verifier evidence (the agent never ran, or the verifier never
            # graded it) keeps a stand-in 0.0 so advantage estimation stays defined, and is
            # flagged `infra_failed` so it never enters a reported solve rate. Never a reason
            # to abort the run.
            missing_reward="zero",
        )
    )
    _wipe_stale_policy_dir(scorer.candidate_job_dir(harness), provider_config)
    logger.info(
        "collecting rollouts for %s: %d task(s) x %d attempt(s), backend %s -> %s",
        step_name,
        len(task_ids),
        cfg.train.group_size,
        backend,
        jobs_dir,
    )
    report = scorer.score(harness, should_cancel=should_cancel)
    records = assemble_harbor_trial_records(report.cells, max_turns=cfg.rollout.max_turns)
    stats = rollout_stats(records, max_tokens=cfg.sampling.max_tokens)
    if stats.truncated_spans:
        logger.warning(
            "%d turn(s) across %d/%d trial(s) in %s sampled the full sampling.max_tokens = %d "
            "and were cut off mid-answer; harbor cannot report this (its OutputLengthExceededError "
            "guard is unreachable), so the agent read a half-written action and the episode "
            "continued as if the model had answered badly. A truncated turn also cannot be "
            "replayed verbatim, so it fragments the episode; raise sampling.max_tokens",
            stats.truncated_spans,
            stats.truncated_span_trials,
            stats.trials,
            step_name,
            cfg.sampling.max_tokens,
        )
    if stats.empty_span_trials:
        logger.warning(
            "%d/%d trial(s) in %s recorded no token spans in their result.json (the agent "
            "died before its first student completion, or harbor logged no rollout details); "
            "they carry reward signal but no training data",
            stats.empty_span_trials,
            stats.trials,
            step_name,
        )
    if stats.infra_failed_trials:
        logger.warning(
            "%d/%d trial(s) in %s never produced verifier evidence (no sandbox, a rate limit, a "
            "dead transport, or a VERIFIER that never graded the work); they are EXCLUDED from "
            "the reported solve rate (%d executed) and carry a stand-in 0.0 reward for advantage "
            "estimation only; grep the cells' `infra-failure:` notes for the exact causes",
            stats.infra_failed_trials,
            stats.trials,
            step_name,
            stats.executed_trials,
        )
    if stats.graded_trials < stats.executed_trials:
        logger.warning(
            "%d/%d gradeable trial(s) in %s carry a verifier reward but no readable CTRF test "
            "report, so they are EXCLUDED from the graded solve rate (%.3f over %d trial(s)) "
            "instead of counted as 0.0; the binary solve rate still covers all %d",
            stats.executed_trials - stats.graded_trials,
            stats.executed_trials,
            step_name,
            stats.graded_solve_rate,
            stats.graded_trials,
            stats.executed_trials,
        )
    if stats.scaffold_loss_rate > 0:
        logger.warning(
            "scaffold loss rate %.1f%% in %s: %d/%d executed trial(s) never reached an "
            "explicit submit (stop reasons %s); those rewards measure where the harness cut "
            "the episode off, not model capability",
            100.0 * stats.scaffold_loss_rate,
            step_name,
            round(stats.scaffold_loss_rate * stats.executed_trials),
            stats.executed_trials,
            stats.stop_reason_counts,
        )
    return records, stats
