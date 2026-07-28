"""Collect one distillation training step's rollouts as real tau2-bench episodes.

`collect_tau2_rollouts` is the tau2 counterpart of the harbor collector
(`wmo.distill.rollouts.collect_rollouts`): one train batch (task ids x group
size) becomes real tau2 episodes, each an unmodified run of Sierra's own
benchmark harness - tau2's `llm_agent`, its LLM user simulator, its
orchestrator, and its deterministic evaluator - joined back with the exact
token spans the sampled model produced.

Token-exactness works differently from harbor's in-process terminus-2. tau2
lives in its own Python 3.13 venv (wmo never imports it) and its agent calls
an OpenAI-compatible endpoint through litellm, which is a TEXT boundary that
would lose token identity. So every episode runs as its own `tau2 run`
subprocess whose agent LLM points at a loopback proxy
(`wmo.distill.tau2_proxy.EpisodeProxy`), where a PER-EPISODE
`TinkerChatProvider` samples the current weights and its `TokenRecorder`
writes the episode's spans. The provider's prompt state splices the next
prompt as `prompt(N) + sampled(N) + suffix` and matches replayed assistant
turns by role, so the ids are never re-encoded from the text tau2 echoes
back. One episode per subprocess is what makes span attribution exact: the
model alias in the request names the episode, and no transcript matching is
ever needed.

Episode identity and resume mirror the harbor collector's rules:

- The step's episodes dir is keyed by step (`run_dir/tau2/step-NNNN`), fresh
  weights per step, so a completed episode dir is only ever reused within the
  same step.
- Each episode dir records the provider config it sampled
  (`provider.json`). A dir recorded under DIFFERENT weights is wiped and the
  episode re-runs from the current weights (sampler paths carry a per-session
  nonce, so a crash-resumed step cannot silently mix policies); a dir whose
  recorded provider matches and whose evidence is complete is reused without
  re-running (the teacher's stable identity is what makes warmup and the
  teacher baseline resumable).

The verifier is tau2's own: `results.json` `reward_info.reward`, in [0, 1]
with 1.0 = the benchmark's definition of success. The user simulator is part
of the environment and stays pinned (`Tau2Config.user_llm`). Episodes that
died without verifier evidence (`termination_reason = infrastructure_error`
or no recorded reward) are `infra_failed`: excluded from reported solve
rates, kept as stand-in 0.0 for advantage estimation, exactly as harbor
trials are.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from collections.abc import Callable, Sequence
from hashlib import blake2s
from pathlib import Path

from wmo.core.types import JsonObject
from wmo.distill.config import DistillConfig
from wmo.distill.data import CONTEXT_OVERFLOW_STOP_REASON
from wmo.distill.rollouts import RolloutStats, rollout_stats
from wmo.distill.tau2_proxy import EpisodeProxy
from wmo.distill.tokens import TrialRecord, load_trial_spans
from wmo.harness.doc import HarnessDoc
from wmo.harness.runtime import StopReason
from wmo.providers.base import ProviderConfig, ProviderKind
from wmo.providers.retry import RetryingToolCallingProvider
from wmo.providers.tinker import TinkerChatProvider, TokenRecorder

logger = logging.getLogger(__name__)

TAU2_DOMAINS = ("airline", "retail", "telecom")
"""The tau2 domains the pinned corpus covers; a task id must name one."""

TASK_SPLIT_OVERRIDES = {"telecom": "full"}
"""Per-domain `--task-split-name` overrides.

A property of the corpus, not a knob: the pinned telecom scenarios were
captured from telecom's 2285-task "full" split, and tau2's default "base"
split raises on the missing ids (see
the tau-bench data bundle's README)."""

RESULTS_FILENAME = "results.json"
SPANS_SINK_DIR = "spans"
PROVIDER_SNAPSHOT_FILENAME = "provider.json"
RUNNER_LOG_FILENAME = "tau2.log"

_SUBPROCESS_KILL_MARGIN_S = 120.0
"""Wall clock granted past tau2's own `--timeout` before the subprocess is killed.

tau2's per-simulation timeout ends the episode gracefully (a recorded
`timeout` termination with the evaluator still run); the hard kill exists
only for a wedged runner and forfeits the episode's evidence."""

_TERMINATION_STOP_REASONS: dict[str, str] = {
    # The conversation ended by its own protocol: the user or agent declared
    # the interaction over. tau2's evaluator grades the completed episode, so
    # this is the tau2 analog of an explicit submit.
    "user_stop": StopReason.SUBMITTED.value,
    "agent_stop": StopReason.SUBMITTED.value,
    "max_steps": StopReason.MAX_TURNS.value,
    "timeout": StopReason.BUDGET.value,
    # The agent burned tau2's per-episode error budget on malformed or invalid
    # tool calls; the closest scaffold-loss bucket names the cause.
    "too_many_errors": StopReason.UNPARSED_TOOL_CALL.value,
    # The datum builder's explicit drop keys on this exact string
    # (`wmo.distill.data.CONTEXT_OVERFLOW_STOP_REASON`): an episode that
    # outgrew the window must leave training whole, tokens aligned.
    "context_window_exceeded": CONTEXT_OVERFLOW_STOP_REASON,
    "agent_error": StopReason.ERROR.value,
    "user_error": StopReason.ERROR.value,
    "unexpected_error": StopReason.ERROR.value,
    # infrastructure_error is deliberately ABSENT: it carries no verifier
    # evidence and maps to `infra_failed`, not to a stop reason.
}


def parse_tau2_task_id(task_id: str) -> tuple[str, str]:
    """Split a composite `domain/benchmark_task_id` into its parts.

    Args:
        task_id: The split-file task id, e.g. `"airline/12"`.

    Returns:
        The `(domain, tau2_task_id)` pair.

    Raises:
        ValueError: If the id has no domain prefix or names an unknown domain.
    """
    domain, separator, rest = task_id.partition("/")
    if not separator or not rest or domain not in TAU2_DOMAINS:
        raise ValueError(
            f"invalid tau2 task id {task_id!r}: expected 'domain/task_id' with the "
            f"domain one of {', '.join(TAU2_DOMAINS)} (e.g. 'airline/12'); fix the "
            "task-id split files so every entry carries its domain prefix"
        )
    return domain, rest


def _episode_name(task_id: str, attempt: int) -> str:
    """The filesystem- and alias-safe episode name for one (task, attempt)."""
    return f"{task_id.replace('/', '-')}-a{attempt:02d}"


class _EpisodeSpec:
    """One (task, attempt) episode's identities, resolved once up front."""

    def __init__(self, task_id: str, attempt: int, step_dir: Path) -> None:
        self.task_id = task_id
        self.attempt = attempt
        self.domain, self.tau2_task_id = parse_tau2_task_id(task_id)
        self.name = _episode_name(task_id, attempt)
        self.episode_dir = step_dir / self.name
        self.sink_dir = step_dir / SPANS_SINK_DIR
        # Unique per (rollout root, step, episode), because tau2 keys its
        # simulations dir by this name and --auto-resume RESUMES a matching
        # sim: a bare episode name would silently replay step N's episode as
        # step N+1's "fresh" rollout under new weights, and two runs sharing
        # an eval key would steal each other's sims. Hashing the absolute
        # step dir scopes the name to exactly one step of one rollout root.
        step_scope = blake2s(str(step_dir.resolve()).encode("utf-8"), digest_size=6).hexdigest()
        self.save_name = f"wmo-{step_scope}-{self.name}"


def _provider_snapshot(provider_config: ProviderConfig) -> JsonObject:
    """The provider identity an episode dir records (and is compared against)."""
    return provider_config.model_dump(mode="json")


def _has_verifier_evidence(spec: _EpisodeSpec) -> bool:
    """Whether the episode dir holds a GRADED simulation.

    The shared predicate behind both the retry decider and resume reuse: the
    results must parse to a simulation whose termination is not an
    infrastructure error and whose reward is a number. Anything less is an
    absent measurement, never evidence.
    """
    simulation = _read_simulation(spec)
    if simulation is None:
        return False
    if simulation.get("termination_reason") == "infrastructure_error":
        return False
    reward_info = simulation.get("reward_info")
    return isinstance(reward_info, dict) and isinstance(reward_info.get("reward"), int | float)


def _episode_complete(spec: _EpisodeSpec, provider_config: ProviderConfig) -> bool:
    """Whether an episode dir holds complete evidence reusable under THIS provider.

    Reuse demands more than file existence, or a crash-resumed step inherits
    junk permanently: the episode must carry VERIFIER evidence (the same bar
    the retry loop uses, so a recorded infrastructure failure is retried fresh
    on resume instead of being reused as a permanent zero), and the recorded
    provider snapshot must parse AND equal the current provider (an unreadable
    snapshot cannot prove which weights sampled the episode, so it re-runs).
    A missing span sink is deliberately NOT a reuse blocker: the sink is
    written before the results copy, so a valid results file with no sink
    means the episode genuinely made no successful completion, which is the
    recorded `empty_span_trials` shape.
    """
    if not _has_verifier_evidence(spec):
        return False
    try:
        recorded = json.loads(
            (spec.episode_dir / PROVIDER_SNAPSHOT_FILENAME).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False
    return recorded == _provider_snapshot(provider_config)


def _wipe_stale_episode_dir(spec: _EpisodeSpec, provider_config: ProviderConfig) -> bool:
    """Wipe an episode dir recorded under different provider weights.

    Mirrors the harbor collector's `_wipe_stale_policy_dir`: a dir left by a
    previous session under other weights would resume another policy's
    episode. A MATCHING snapshot leaves the dir for `_episode_complete` to
    reuse. An unreadable snapshot also leaves the dir in place, but such a dir
    never passes `_episode_complete`, so the episode re-runs and its fresh
    attempt overwrites the evidence rather than destroying it here.

    Args:
        spec: The episode's resolved identities.
        provider_config: The provider this batch is about to sample.

    Returns:
        True when the directory was wiped.
    """
    snapshot_path = spec.episode_dir / PROVIDER_SNAPSHOT_FILENAME
    try:
        recorded = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if recorded == _provider_snapshot(provider_config):
        return False
    logger.warning(
        "tau2 episode dir %s was produced under provider %r, not the current %r; "
        "wiping it so the episode re-runs from the current weights",
        spec.episode_dir,
        recorded.get("model") if isinstance(recorded, dict) else recorded,
        provider_config.model,
    )
    shutil.rmtree(spec.episode_dir)
    sink = spec.sink_dir / f"{spec.name}.jsonl"
    sink.unlink(missing_ok=True)
    return True


def _agent_llm_args(cfg: DistillConfig, proxy_base_url: str) -> JsonObject:
    """The litellm kwargs tau2 passes on every agent completion call."""
    return {
        "api_base": proxy_base_url,
        # litellm requires a key for the openai/ route; the proxy ignores it.
        "api_key": "wmo-local-proxy",
        "temperature": cfg.sampling.temperature,
        "max_tokens": cfg.sampling.max_tokens,
        # ZERO litellm-level retries, for the same span-sink integrity reason as
        # the runner-level --max-retries 0: tau2's generate() defaults
        # num_retries to 3, and a retried HTTP request does not stop the
        # abandoned one, so two threads would call complete_chat on the SAME
        # per-episode provider and race its prompt state and recorder (both
        # single-thread by contract). Retry on this side of the proxy belongs
        # to wmo's RetryingToolCallingProvider. The explicit timeout keeps
        # litellm's client clock from firing before a slow long-prompt Tinker
        # sample returns; the episode budget is the real wall clock.
        "num_retries": 0,
        "timeout": cfg.rollout.episode_timeout_s,
    }


def _tau2_command(spec: _EpisodeSpec, cfg: DistillConfig, proxy_base_url: str) -> list[str]:
    """The `tau2 run` argv for one episode."""
    assert cfg.tau2 is not None  # validated by the caller
    command = [
        cfg.tau2.tau2_bin,
        "run",
        "--domain",
        spec.domain,
    ]
    split_override = TASK_SPLIT_OVERRIDES.get(spec.domain)
    if split_override is not None:
        command += ["--task-split-name", split_override]
    command += [
        "--task-ids",
        spec.tau2_task_id,
        "--num-trials",
        "1",
        "--max-steps",
        str(cfg.rollout.max_turns),
        "--max-errors",
        str(cfg.tau2.max_errors),
        "--timeout",
        str(cfg.rollout.episode_timeout_s),
        "--agent-llm",
        f"openai/{spec.name}",
        "--agent-llm-args",
        json.dumps(_agent_llm_args(cfg, proxy_base_url)),
        "--user-llm",
        cfg.tau2.user_llm,
        "--user-llm-args",
        json.dumps(cfg.tau2.user_llm_args),
        "--save-to",
        spec.save_name,
        # ZERO tau2-internal retries, deliberately. tau2's runner-level retry
        # re-runs the whole simulation, which (a) multiplies the episode wall
        # clock past the collector's hard deadline, and (b) would append the
        # abandoned attempt's turns into the SAME span sink, so training datums
        # would carry tokens from an attempt whose recorded reward is another
        # attempt's. Transient failures retry at the episode level instead
        # (`Tau2Config.episode_retries`), each attempt on a fresh sink.
        "--max-retries",
        "0",
        # Headless reruns must never block on tau2's interactive resume prompt.
        "--auto-resume",
    ]
    return command


async def _run_episode_subprocess(
    spec: _EpisodeSpec, cfg: DistillConfig, proxy_base_url: str
) -> None:
    """Run one tau2 episode subprocess and copy its results into the episode dir.

    The subprocess gets tau2's own graceful `--timeout` plus a hard-kill
    margin; stdout/stderr land in the episode dir for postmortems. The
    authoritative artifact (`results.json`) is copied out of tau2's
    simulations dir so the episode dir stays self-contained even if the tau2
    data dir is cleaned.

    Args:
        spec: The episode's resolved identities.
        cfg: The validated run config.
        proxy_base_url: The loopback proxy the agent LLM calls.

    Raises:
        RuntimeError: If the runner was killed on the hard deadline or exited
            without writing a results file.
    """
    assert cfg.tau2 is not None
    spec.episode_dir.mkdir(parents=True, exist_ok=True)
    command = _tau2_command(spec, cfg, proxy_base_url)
    log_path = spec.episode_dir / RUNNER_LOG_FILENAME
    # A fresh launch means fresh evidence: a leftover sim under this save name
    # (a crashed earlier session, or an episode dir wiped for a policy change)
    # would be silently RESUMED by --auto-resume instead of re-run.
    sim_dir = Path(cfg.tau2.data_dir) / "simulations" / spec.save_name
    shutil.rmtree(sim_dir, ignore_errors=True)
    env = os.environ | {"TAU2_DATA_DIR": cfg.tau2.data_dir}
    with log_path.open("wb") as log_file:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=log_file,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
        try:
            await asyncio.wait_for(
                process.wait(),
                timeout=cfg.rollout.episode_timeout_s + _SUBPROCESS_KILL_MARGIN_S,
            )
        except asyncio.CancelledError:
            # A cancelled await does NOT signal the child: without this, a
            # batch aborted mid-flight leaves tau2 runners burning user-sim
            # tokens against a proxy that is about to stop.
            process.kill()
            await process.wait()
            raise
        except TimeoutError:
            process.kill()
            await process.wait()
            raise RuntimeError(
                f"tau2 runner for {spec.name} exceeded its hard deadline "
                f"({cfg.rollout.episode_timeout_s:.0f}s episode budget "
                f"+ {_SUBPROCESS_KILL_MARGIN_S:.0f}s margin) and was killed; "
                f"see {log_path}"
            ) from None
    results_path = sim_dir / RESULTS_FILENAME
    if not results_path.exists():
        raise RuntimeError(
            f"tau2 runner for {spec.name} exited with code {process.returncode} without "
            f"writing {results_path}; see {log_path}"
        )
    shutil.copyfile(results_path, spec.episode_dir / RESULTS_FILENAME)


def _read_simulation(spec: _EpisodeSpec) -> JsonObject | None:
    """The episode's single simulation object from its copied results file."""
    try:
        payload = json.loads((spec.episode_dir / RESULTS_FILENAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    simulations = payload.get("simulations") if isinstance(payload, dict) else None
    if not isinstance(simulations, list) or not simulations:
        return None
    first = simulations[0]
    return first if isinstance(first, dict) else None


def _assemble_record(spec: _EpisodeSpec) -> TrialRecord:
    """Join one episode's tau2 verdict with its recorded token spans."""
    spans = load_trial_spans(spec.sink_dir, spec.name)
    simulation = _read_simulation(spec)
    termination = simulation.get("termination_reason") if simulation else None
    reward_info = simulation.get("reward_info") if simulation else None
    reward = reward_info.get("reward") if isinstance(reward_info, dict) else None
    infra_failed = (
        simulation is None
        or termination == "infrastructure_error"
        or not isinstance(reward, (int, float))
    )
    reward_value = float(reward) if isinstance(reward, (int, float)) else 0.0
    stop_reason = (
        _TERMINATION_STOP_REASONS.get(termination) if isinstance(termination, str) else None
    )
    return TrialRecord(
        task_id=spec.task_id,
        attempt=spec.attempt,
        trial_name=spec.name,
        reward=max(0.0, min(1.0, reward_value)),
        passed=not infra_failed and reward_value >= 1.0 - 1e-9,
        spans=spans,
        stop_reason=stop_reason,
        infra_failed=infra_failed,
        tests=None,
        artifact_dir=str(spec.episode_dir),
    )


def collect_tau2_rollouts(
    step_index: int,
    task_ids: Sequence[str],
    cfg: DistillConfig,
    harness: HarnessDoc,
    provider_config: ProviderConfig,
    run_dir: Path,
    *,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[list[TrialRecord], RolloutStats]:
    """Run one train batch of real tau2 episodes and join rewards with token spans.

    Each task in `task_ids` runs `cfg.train.group_size` attempts, every
    attempt a fresh `tau2 run` subprocess sampling the given provider through
    the loopback proxy. Signature and return contract are identical to the
    harbor collector's, so the training loop stays source-agnostic.

    Args:
        step_index: 0-based training step; keys the fresh episodes dir.
        task_ids: Composite `domain/task_id` ids for this batch.
        cfg: The validated run config (`cfg.tau2` selected this collector).
        harness: Accepted for signature parity with the harbor collector; the
            rollout knobs pinned into it already live in `cfg`, and tau2's own
            agent is not steered by a harness document.
        provider_config: The provider to sample; `model` must point at the
            CURRENT sampler weights and `model_type` at the base model.
        run_dir: The distillation run directory.
        should_cancel: Optional cooperative cancellation poll; once it returns
            True no further episode starts, and unstarted episodes are
            recorded as infra failures.

    Returns:
        The `TrialRecord`s (one per task x attempt, in report order) and the
        batch stats, matching the harbor collector's semantics: an episode
        with no verifier evidence is kept as an `infra_failed` record, never
        dropped.

    Raises:
        ValueError: If `step_index` is negative, `cfg.tau2` is unset, a task
            id is malformed, or the provider is not the tinker kind.
    """
    del harness  # signature parity; see the docstring
    if step_index < 0:
        raise ValueError(f"step_index must be >= 0, got {step_index}")
    if cfg.tau2 is None:
        raise ValueError("collect_tau2_rollouts needs a config whose [tau2] section is set")
    if provider_config.kind is not ProviderKind.TINKER:
        raise ValueError(
            "distillation rollouts must sample through Tinker so the sampled model's exact "
            f"token spans are recorded, got provider kind {provider_config.kind.value!r}; "
            "configure the worker provider with kind 'tinker'"
        )
    if cfg.tau2.backend == "e2b":
        raise NotImplementedError(
            "the tau2 e2b backend is not wired yet; run with tau2.backend = 'local' "
            "(the runner is an API-only subprocess; the heavy work is remote either way)"
        )

    step_name = f"step-{step_index:04d}"
    step_dir = run_dir / "tau2" / step_name
    specs = [
        _EpisodeSpec(task_id, attempt, step_dir)
        for task_id in task_ids
        for attempt in range(1, cfg.train.group_size + 1)
    ]
    logger.info(
        "collecting tau2 rollouts for %s: %d task(s) x %d attempt(s) -> %s",
        step_name,
        len(task_ids),
        cfg.train.group_size,
        step_dir,
    )

    proxy = EpisodeProxy()
    proxy.start()
    try:
        asyncio.run(_run_batch(specs, cfg, provider_config, proxy, should_cancel))
    finally:
        proxy.stop()

    records = [_assemble_record(spec) for spec in specs]
    stats = rollout_stats(records, max_tokens=cfg.sampling.max_tokens)
    _log_batch_health(step_name, stats)
    return records, stats


async def _run_batch(
    specs: Sequence[_EpisodeSpec],
    cfg: DistillConfig,
    provider_config: ProviderConfig,
    proxy: EpisodeProxy,
    should_cancel: Callable[[], bool] | None,
) -> None:
    """Run the batch's episodes under the configured concurrency."""
    semaphore = asyncio.Semaphore(cfg.train.trial_concurrency)

    assert cfg.tau2 is not None  # validated by the caller

    async def _one_attempt(spec: _EpisodeSpec) -> None:
        """One fresh simulation attempt: fresh sink, fresh recorder, fresh provider."""
        spec.episode_dir.mkdir(parents=True, exist_ok=True)
        spec.sink_dir.mkdir(parents=True, exist_ok=True)
        sink_path = spec.sink_dir / f"{spec.name}.jsonl"
        # A leftover sink from a wiped, crashed, or retried earlier attempt would
        # break load_trial_spans' contiguous call_index contract, and worse,
        # would splice an abandoned attempt's turns into this attempt's datums.
        sink_path.unlink(missing_ok=True)
        (spec.episode_dir / RESULTS_FILENAME).unlink(missing_ok=True)
        # The explicit tool-calling wrapper (not `wrap_provider_with_retries`)
        # because the proxy's registry is typed to the structured seam; same
        # retry contract as the harbor agent bridge.
        provider = RetryingToolCallingProvider(
            TinkerChatProvider(provider_config, recorder=TokenRecorder(jsonl_path=sink_path))
        )
        (spec.episode_dir / PROVIDER_SNAPSHOT_FILENAME).write_text(
            json.dumps(_provider_snapshot(provider_config), sort_keys=True),
            encoding="utf-8",
        )
        proxy.register(spec.name, provider)
        try:
            await _run_episode_subprocess(spec, cfg, proxy.base_url)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - one episode must not kill the batch
            # RuntimeError is the expected shape (hard deadline, missing
            # results), but ANY per-episode failure is still just an episode
            # with no verifier evidence: letting it escape would cancel every
            # sibling through asyncio.gather and orphan their tau2 runners.
            # The retry loop (or, on the last attempt, the assembled
            # infra_failed record) owns what happens next.
            logger.warning("episode %s attempt failed: %s", spec.name, error)
        finally:
            proxy.release(spec.name)

    episode_retries = cfg.tau2.episode_retries

    async def _one(spec: _EpisodeSpec) -> None:
        async with semaphore:
            if should_cancel is not None and should_cancel():
                logger.warning("cancellation requested; skipping episode %s", spec.name)
                return
            _wipe_stale_episode_dir(spec, provider_config)
            if _episode_complete(spec, provider_config):
                logger.info("episode %s already complete; reusing its evidence", spec.name)
                return
            for attempt in range(1 + episode_retries):
                await _one_attempt(spec)
                if _has_verifier_evidence(spec):
                    return
                if attempt < episode_retries:
                    logger.warning(
                        "episode %s produced no verifier evidence; retrying with a fresh "
                        "simulation and a fresh span sink (%d retr%s left)",
                        spec.name,
                        episode_retries - attempt,
                        "y" if episode_retries - attempt == 1 else "ies",
                    )

    await asyncio.gather(*(_one(spec) for spec in specs))


def _log_batch_health(step_name: str, stats: RolloutStats) -> None:
    """Mirror the harbor collector's batch health warnings."""
    if stats.truncated_spans:
        logger.warning(
            "%d turn(s) across %d/%d episode(s) in %s sampled the full sampling.max_tokens "
            "and were cut off mid-answer; a truncated turn cannot be replayed verbatim and "
            "fragments the episode; raise sampling.max_tokens",
            stats.truncated_spans,
            stats.truncated_span_trials,
            stats.trials,
            step_name,
        )
    if stats.empty_span_trials:
        logger.warning(
            "%d/%d episode(s) in %s recorded no token spans (the runner died before its "
            "first completion); they carry reward signal but no training data",
            stats.empty_span_trials,
            stats.trials,
            step_name,
        )
    if stats.infra_failed_trials:
        logger.warning(
            "%d/%d episode(s) in %s never produced verifier evidence; they are EXCLUDED "
            "from the reported solve rate (%d executed) and carry a stand-in 0.0 reward "
            "for advantage estimation only; read each episode dir's %s",
            stats.infra_failed_trials,
            stats.trials,
            step_name,
            stats.executed_trials,
            RUNNER_LOG_FILENAME,
        )
    if stats.scaffold_loss_rate > 0:
        logger.warning(
            "scaffold loss rate %.1f%% in %s: episodes that never reached a protocol stop "
            "(stop reasons %s); those rewards measure where the harness cut the episode "
            "off, not model capability",
            100.0 * stats.scaffold_loss_rate,
            step_name,
            stats.stop_reason_counts,
        )
