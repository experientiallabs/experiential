"""Standalone scaffold-loss probe: one wave of real pi rollouts, no teacher, no training.

Purpose: cheap live evidence that the pi/Nemotron-3 scaffold defects are fixed,
BEFORE paying for a distillation sweep. The forensic audit
(`.agents/distill/pi-nemotron-scaffold-audit.md`) found that only 7.8% of Ultra
episodes and 11.2% of Super episodes ever reached an explicit `submit`: every
other trial's reward measured where the harness cut the episode off, not what
the model can do. The audit's Tier 1 experiment is exactly this probe, and its
primary metric is the SCAFFOLD LOSS RATE (share of executed episodes that never
reached `submit`), not the pass rate, which is far too noisy at n=12.

What it does and does not do:

- Loads a real `DistillConfig` from a TOML (the same `load_distill_config` the
  CLI uses), pins the harness with `pin_rollout_params`, and calls
  `wmh.distill.rollouts.collect_rollouts` ONCE. That is the same rollout path a
  training step drives, so what it measures is what a real run would see.
- Samples the BASE student directly: `tinker_provider_config(base, base)` and a
  sampling client for the base model name. No training client is created, so
  no `forward_backward`, `optim_step`, or `save_state` can happen, and there is
  no adapter, no teacher scoring, and no eval report.
- `wmh optimize harness --mode distill` cannot do this cheaply: `TrainConfig.steps` has
  `ge=1`, so the smallest run still pays for teacher scoring plus the teacher
  baseline, student-before, and student-after evals.

Spend: student sampling (plus one 1-token preflight ping) and the E2B sandboxes
the trials hold. Nothing else is metered.

Backend: the checked-in configs say `harbor.backend = "local"` while every real
run recorded `e2b` (the CLI's own `--backend` override), so `--backend` is
offered here too and defaults to whatever the config says.

Run from the repo root (TINKER_API_KEY plus, for `--backend e2b`, E2B_API_KEY):

    uv run python .agents/distill/probe_scaffold.py \
        --config .agents/distill/distill-super-anchor.toml \
        --task-ids .agents/distill/tb2-probe-task-ids.json \
        --backend e2b --run-dir .wmh/distill-runs/probe-scaffold

Progress goes to stderr through the module logger (the rollout collector's own
warnings included); the plain-language report goes to stdout, so
`> report.txt` keeps the two apart. Raw per-episode evidence lands in
`<run-dir>/probe-scaffold.json`.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from wmh.agents.default import default_agent
from wmh.distill.config import DistillConfig, load_distill_config
from wmh.distill.cost import BudgetMeter, batch_billing
from wmh.distill.loop import (
    SdkServiceClient,
    TokenizerSource,
    pin_rollout_params,
    tinker_provider_config,
)
from wmh.distill.rollouts import (
    E2B_SANDBOXES_PER_TRIAL,
    UNKNOWN_STOP_REASON,
    RolloutStats,
    collect_rollouts,
)
from wmh.distill.tokens import TrialRecord
from wmh.harness.doc import HarnessDoc
from wmh.harness.e2b_reap import check_capacity, is_credential_error
from wmh.harness.population import write_json_atomic
from wmh.harness.runtime import StopReason
from wmh.harness.scoring import ScoreRequest
from wmh.providers.base import ProviderConfig
from wmh.providers.tinker import (
    TINKER_API_KEY_ENV,
    served_context_window,
    shared_service_client,
)

logger = logging.getLogger("probe_scaffold")

PROBE_STEP_INDEX = 0
"""The step index the single rollout wave is filed under (`harbor/step-0000`)."""

REPORT_FILENAME = "probe-scaffold.json"
"""Per-episode evidence plus the stats, written into the run dir."""

DEFAULT_MAX_SCAFFOLD_LOSS = 0.05
"""Pass threshold: at most this share of executed episodes may miss `submit`."""

SEED_AGENT = "pi"
"""The built-in pi harness the distill CLI seeds every rollout from."""

PI_RUNTIME_KIND = "pi-node"

HARBOR_TIMEOUT_STOP_REASON = "cancelled-by-harbor-timeout"
"""Harbor cancelled the trial on ITS own timeout (`wmh.evals.harbor.agent` writes this marker)."""

AGENT_EXCEPTION_PREFIX = "agent-exception:"
"""Prefix of the partial-trace marker for a trial whose agent process raised."""

STOP_REASON_MEANINGS: dict[str, str] = {
    StopReason.SUBMITTED.value: "the model called submit itself (a real completion)",
    StopReason.MAX_TURNS.value: "cut off at the turn cap",
    StopReason.NO_TOOL_CALL.value: "prose-only turns exhausted the nudge budget",
    StopReason.OUTPUT_TRUNCATED.value: "last turn was cut at the output-token cap",
    StopReason.UNPARSED_TOOL_CALL.value: "the renderer could not parse the tool call it emitted",
    StopReason.PROVIDER_ERROR.value: "the worker LLM calls kept failing (context overflow, outage)",
    StopReason.NO_ACTION.value: "no parseable tool call (non-pi runtimes)",
    StopReason.ERROR.value: "harness code raised",
    StopReason.BUDGET.value: "ran past the episode wall budget (rollout.episode_timeout_s)",
    HARBOR_TIMEOUT_STOP_REASON: "harbor cancelled the trial on its own timeout",
    UNKNOWN_STOP_REASON: "no readable run trace (usually an infra-failed trial)",
}
"""Plain-language gloss per recorded stop reason, for the report's breakdown.

Covers the `StopReason` vocabulary plus the two markers harbor's agent bridge
writes into a PARTIAL trace, which are stop reasons a real wave will show and
which `rollout_stats` counts (correctly) as scaffold losses."""


def stop_reason_meaning(reason: str) -> str:
    """The plain-language gloss for one recorded stop reason."""
    if reason.startswith(AGENT_EXCEPTION_PREFIX):
        return f"the agent process raised {reason[len(AGENT_EXCEPTION_PREFIX) :]}"
    return STOP_REASON_MEANINGS.get(reason, "unrecognized stop reason (a new StopReason member?)")


@dataclass(frozen=True)
class EpisodeSummary:
    """One episode's shape: what it did and how it ended."""

    task_id: str
    attempt: int
    trial_name: str
    stop_reason: str
    passed: bool
    reward: float
    infra_failed: bool
    turns: int
    """Recorded student completions (one span per successful sampling call)."""

    sampled_tokens: int
    final_prompt_tokens: int
    """Prompt length of the last recorded call: the episode's context high-water mark."""

    artifact_dir: str
    token_sink: str


def load_task_ids(path: Path) -> tuple[str, ...]:
    """Load the probe's task ids from a JSON array, validated canonically.

    Same validation the CLI applies to `--task-ids` (the `ScoreRequest` rules:
    non-empty, unique, no blanks), so a list that probes cleanly here is a list
    a real run accepts.

    Args:
        path: JSON file holding one array of task-id strings.

    Returns:
        The ordered task ids.

    Raises:
        ValueError: If the file cannot be read or is not a valid id list.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load task ids from {path}: {error}") from error
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError(f"{path} must contain one JSON array of task-id strings")
    try:
        return ScoreRequest(task_ids=tuple(raw), attempts=1).task_ids
    except ValidationError as error:
        raise ValueError(f"invalid task ids in {path}: {error}") from error


def probe_harness() -> HarnessDoc:
    """The seed pi harness every distillation rollout runs.

    Raises:
        RuntimeError: If the built-in seed stops being a pi-node harness (the
            distill rollout path drives pi through harbor and nothing else).
    """
    doc = default_agent(SEED_AGENT)
    kind = doc.runtime_kind()
    if kind != PI_RUNTIME_KIND:
        raise RuntimeError(
            f"the built-in {SEED_AGENT!r} harness has runtime kind {kind!r}, not "
            f"{PI_RUNTIME_KIND!r}; this probe measures the pi scaffold, so it cannot "
            "report anything meaningful about another runtime"
        )
    return doc


def student_provider(cfg: DistillConfig) -> ProviderConfig:
    """The rollout provider for the BASE student: no adapter, no sampler weights.

    The same helper both loop sites use (`wmh.distill.loop`), with `model` and
    `model_type` both the base model name: Tinker serves a base model directly,
    so measuring the scaffold needs no training client and therefore cannot
    train by accident.
    """
    return tinker_provider_config(cfg.student.base_model, cfg.student.base_model)


def preflight_student_sampler(cfg: DistillConfig) -> None:
    """Prove the student serves before 48 sandboxes spin up, for one token.

    Mirrors the loop's own student preflight ping minus everything training
    needs: a sampling client for the base model (`SdkServiceClient`, the same
    shared-cache adapter the rollout providers use), one token sampled from a
    tokenized "ping". No training client, so no weights call is possible.

    Raises:
        RuntimeError: If the client cannot be built, exposes no tokenizer, or
            samples nothing; the message names what to check.
    """
    service = SdkServiceClient(shared_service_client())
    client = service.create_sampling_client(cfg.student.base_model)
    if not isinstance(client, TokenizerSource):
        raise RuntimeError(
            f"the sampling client for {cfg.student.base_model!r} exposes no tokenizer, so "
            "the preflight ping cannot be built; check the pinned tinker SDK"
        )
    prompt = client.get_tokenizer().encode("ping")
    try:
        sequence = client.sample(prompt, max_tokens=1, temperature=0.0)
    except Exception as exc:  # noqa: BLE001 - re-raised with actionable context
        raise RuntimeError(
            f"student preflight ping failed for {cfg.student.base_model!r}: {exc}; check "
            f"that the base model is still in Tinker's lineup and that {TINKER_API_KEY_ENV} "
            "is valid"
        ) from exc
    if not sequence.tokens:
        raise RuntimeError(
            f"student preflight ping for {cfg.student.base_model!r} sampled no tokens; "
            "check the model name and the served context tier"
        )
    logger.info("student preflight ping ok: %s serves", cfg.student.base_model)


def preflight_context_window(cfg: DistillConfig) -> None:
    """Warn when the configured context budget cannot fit the served window.

    The audit's defect 4: a 128,000-token assumption against a 65,536-token
    deployment produced 118 context-overflow 400s in one run, every one of
    which ended its episode. The served window is read from the service's own
    capabilities, never from a table, and the check is
    `context_budget_tokens + sampling.max_tokens <= served window` because that
    sum is what one request sends.
    """
    served = served_context_window(cfg.student.base_model)
    needed = cfg.rollout.context_budget_tokens + cfg.sampling.max_tokens
    if served is None:
        logger.warning(
            "Tinker lists no context window for %s, so the %d-token budget + %d-token "
            "output cap cannot be checked against the deployment; a prompt over the real "
            "window ends its episode with a 400",
            cfg.student.base_model,
            cfg.rollout.context_budget_tokens,
            cfg.sampling.max_tokens,
        )
        return
    if needed > served:
        logger.warning(
            "rollout.context_budget_tokens %d + sampling.max_tokens %d = %d exceeds the "
            "%d-token window Tinker serves %s with; episodes will hit context-overflow "
            "400s and end early, which this probe will count as scaffold loss",
            cfg.rollout.context_budget_tokens,
            cfg.sampling.max_tokens,
            needed,
            served,
            cfg.student.base_model,
        )
        return
    logger.info(
        "context budget ok: %d + %d = %d tokens per request against the %d served for %s",
        cfg.rollout.context_budget_tokens,
        cfg.sampling.max_tokens,
        needed,
        served,
        cfg.student.base_model,
    )


def preflight_e2b_capacity(*, concurrency: int) -> None:
    """Refuse to start when the account cannot hold the wave's sandboxes.

    A distill trial holds `E2B_SANDBOXES_PER_TRIAL` sandboxes at once (harbor's
    task environment plus the pooled pi worker), and the audit's defect 5 was
    24.4% of Super trials dying at sandbox creation on the account cap, every
    one recorded as a measured zero. `check_capacity` counts what is running
    and reclaims only this machine's provable orphans.

    Args:
        concurrency: Trials this wave can have in flight at once.

    Raises:
        RuntimeError: If capacity cannot be measured or too few slots are free.
    """
    required = concurrency * E2B_SANDBOXES_PER_TRIAL
    try:
        check = check_capacity(required=required)
    except Exception as error:  # noqa: BLE001 - a monitoring call must not block a probe
        if isinstance(error, ImportError | ValueError) or is_credential_error(error):
            raise
        logger.warning(
            "could not check E2B sandbox capacity (%s: %s); starting anyway",
            type(error).__name__,
            error,
        )
        return
    if check.reaped:
        logger.info(
            "reaped %d orphaned E2B sandbox(es) from dead local runs (%d -> %d of %d in use)",
            check.reaped,
            check.alive_before,
            check.alive,
            check.cap,
        )
    if not check.ok:
        raise RuntimeError(
            f"not enough free E2B sandbox slots: {check.alive} of {check.cap} concurrent "
            f"sandboxes are in use, leaving {check.free} free, but this wave needs "
            f"{required} ({E2B_SANDBOXES_PER_TRIAL} per trial x {concurrency} concurrent "
            "trials). Run `wmh e2b reap --stale-minutes 60 --yes` (account-wide: it can "
            "kill another machine's run), lower train.trial_concurrency in the config, or "
            "wait for the other runs to finish"
        )
    logger.info(
        "e2b capacity ok: %d/%d sandbox(es) in use, %d free, %d needed (%d per trial x %d "
        "concurrent trials)",
        check.alive,
        check.cap,
        check.free,
        required,
        E2B_SANDBOXES_PER_TRIAL,
        concurrency,
    )


def summarize_episodes(
    records: Sequence[TrialRecord], token_sink_dir: Path
) -> list[EpisodeSummary]:
    """One summary row per trial: turns, sampled tokens, and how it ended."""
    summaries: list[EpisodeSummary] = []
    for record in records:
        summaries.append(
            EpisodeSummary(
                task_id=record.task_id,
                attempt=record.attempt,
                trial_name=record.trial_name,
                stop_reason=record.stop_reason or UNKNOWN_STOP_REASON,
                passed=record.passed,
                reward=record.reward,
                infra_failed=record.infra_failed,
                turns=len(record.spans),
                sampled_tokens=sum(len(span.sampled_token_ids) for span in record.spans),
                final_prompt_tokens=(len(record.spans[-1].prompt_token_ids) if record.spans else 0),
                artifact_dir=record.artifact_dir,
                token_sink=str(token_sink_dir / f"{record.trial_name}.jsonl"),
            )
        )
    return summaries


def percentile(values: Sequence[int], quantile: float) -> int:
    """Nearest-rank percentile over integers (no interpolation, so p99 is a real episode).

    Args:
        values: The sample; must be non-empty.
        quantile: Fraction in (0, 1].

    Returns:
        The value at the nearest rank at or above `quantile`.
    """
    ordered = sorted(values)
    rank = math.ceil(quantile * len(ordered))
    return ordered[min(max(rank, 1), len(ordered)) - 1]


def distribution_line(label: str, values: Sequence[int]) -> str:
    """One `mean / p50 / p99` line, or a plain note when nothing was recorded."""
    if not values:
        return f"  {label:<32} not recorded (no episode captured a completion)"
    return (
        f"  {label:<32} mean {statistics.fmean(values):,.0f}  "
        f"p50 {percentile(values, 0.5):,}  p99 {percentile(values, 0.99):,}"
        f"  (n={len(values)})"
    )


def suspicious_submits(summaries: Sequence[EpisodeSummary]) -> list[EpisodeSummary]:
    """Episodes reported as `submitted` that look like the pre-fix misread.

    A `done` frame with no `reason` still degrades to `submitted`
    (`wmh.harness.runner_link.stop_reason_for_done` logs a warning and assumes
    the old contract), so a runner that predates termination reporting would
    show 0% scaffold loss. The signature the audit documented for that failure
    is a zero-reward "submission" that barely ran: no recorded completion at
    all, or at most two turns. Not proof, a prompt to read those trial dirs.
    """
    return [
        summary
        for summary in summaries
        if summary.stop_reason == StopReason.SUBMITTED.value
        and not summary.passed
        and not summary.infra_failed
        and summary.turns <= 2
    ]


def student_spend_usd(cfg: DistillConfig, records: Sequence[TrialRecord]) -> float | None:
    """Measured student spend for the wave, or None when `[pricing]` prices nothing.

    Charged exactly as the loop charges a student batch: unique episode tokens
    at the prefill rate, the repeated per-request volume at the cached rate,
    and sampled tokens at the sampling rate.
    """
    billing = batch_billing(records)
    rates = (
        cfg.pricing.student_prefill,
        cfg.pricing.student_cached_prefill,
        cfg.pricing.student_sample,
    )
    if all(rate is None for rate in rates):
        return None
    meter = BudgetMeter(cfg.pricing)
    meter.charge("student_prefill", billing.unique_tokens)
    meter.charge("student_cached_prefill", billing.cached_tokens)
    meter.charge("student_sample", billing.sampled_tokens)
    return meter.spent_usd


def report_lines(
    *,
    cfg: DistillConfig,
    config_path: Path,
    backend: str,
    task_ids: Sequence[str],
    stats: RolloutStats,
    summaries: Sequence[EpisodeSummary],
    records: Sequence[TrialRecord],
    max_scaffold_loss: float,
    report_path: Path,
) -> list[str]:
    """The plain-language report, one string per line."""
    executed = [summary for summary in summaries if not summary.infra_failed]
    rule = "=" * 88
    lines = [
        rule,
        f"pi scaffold probe: {len(task_ids)} task(s) x {cfg.train.group_size} attempt(s) "
        f"= {stats.trials} episode(s)",
        f"config {config_path} | student {cfg.student.base_model} (base weights, no adapter)",
        f"backend {backend} | max_turns {cfg.rollout.max_turns} | episode wall "
        f"{cfg.rollout.episode_timeout_s:.0f}s | context budget "
        f"{cfg.rollout.context_budget_tokens:,} + {cfg.sampling.max_tokens:,} out",
        rule,
        f"  {'episodes run':<32} {stats.trials}",
        f"  {'executed (verifier graded it)':<32} {stats.executed_trials}",
        f"  {'infra-failed (no verdict)':<32} {stats.infra_failed_trials}"
        "   nothing ran (sandbox / rate limit / transport) or nothing graded it (verifier)",
        f"  {'episodes with no completion':<32} {stats.empty_span_trials}"
        "   (died before the first student sample)",
        "",
        f"how the {stats.trials} episode(s) ended:",
    ]
    for reason, count in stats.stop_reason_counts.items():
        share = 100.0 * count / stats.trials if stats.trials else 0.0
        meaning = stop_reason_meaning(reason)
        lines.append(f"  {reason:<32} {count:>4}  {share:5.1f}%   {meaning}")
    # Counted over EXECUTED episodes only, exactly as `rollout_stats` computes the rate.
    scaffold_lost = sum(
        1 for summary in executed if summary.stop_reason != StopReason.SUBMITTED.value
    )
    lines.extend(
        [
            "",
            f"  {'SCAFFOLD LOSS RATE':<32} {100.0 * stats.scaffold_loss_rate:5.1f}%"
            f"   ({scaffold_lost} of {stats.executed_trials} executed episode(s) never "
            "reached submit)",
            f"  {'threshold (--max-scaffold-loss)':<32} {100.0 * max_scaffold_loss:5.1f}%",
            f"  {'solve rate (executed only)':<32} {100.0 * stats.solve_rate:5.1f}%"
            f"   ({sum(1 for s in executed if s.passed)} of {stats.executed_trials})",
            f"  {'solve rate (raw, all episodes)':<32} {100.0 * stats.raw_solve_rate:5.1f}%"
            "   what advantage estimation sees",
            "",
        ]
    )
    turns = [summary.turns for summary in summaries if summary.turns]
    sampled = [summary.sampled_tokens for summary in summaries if summary.turns]
    context = [summary.final_prompt_tokens for summary in summaries if summary.turns]
    lines.append(distribution_line("turns per episode", turns))
    lines.append(distribution_line("sampled tokens per episode", sampled))
    lines.append(distribution_line("final prompt tokens", context))
    spend = student_spend_usd(cfg, records)
    lines.append(
        f"  {'student spend (measured)':<32} "
        + ("unpriced ([pricing] has no student rates)" if spend is None else f"${spend:,.2f}")
    )
    suspicious = suspicious_submits(summaries)
    if suspicious:
        lines.extend(
            [
                "",
                f"  WARNING {len(suspicious)} episode(s) reported `submitted` with reward 0 and "
                "<= 2 turns.",
                "  That is the signature of a runner whose `done` frame carried no reason (it "
                "degrades",
                "  to `submitted`), i.e. a possible FALSE PASS. Read these trial dirs:",
            ]
        )
        lines.extend(f"    {summary.artifact_dir}" for summary in suspicious[:5])
    lines.append("")
    if stats.trials and not stats.executed_trials:
        lines.append(
            "NULL MEASUREMENT: no episode carries a verifier reward, so there is no "
            "scaffold-loss rate to report. Either nothing ran (almost always the E2B "
            "concurrent-sandbox cap) or nothing graded it (verifier timeouts)."
        )
    elif not stats.trials:
        lines.append("NULL MEASUREMENT: harbor scored no trials at all; check the job template.")
    elif stats.scaffold_loss_rate <= max_scaffold_loss:
        lines.append(
            f"PASS: scaffold loss {100.0 * stats.scaffold_loss_rate:.1f}% is at or under the "
            f"{100.0 * max_scaffold_loss:.1f}% threshold. The scaffold is no longer the "
            "dominant term; solve rates from this configuration are worth measuring."
        )
    else:
        lines.append(
            f"FAIL: scaffold loss {100.0 * stats.scaffold_loss_rate:.1f}% exceeds the "
            f"{100.0 * max_scaffold_loss:.1f}% threshold. Read the breakdown above: every "
            "non-`submitted` bucket is a harness defect, not a task failure, and a "
            "distillation sweep run now would train on truncated episodes."
        )
    lines.append(f"per-episode evidence: {report_path}")
    lines.append(rule)
    return lines


def write_report(
    path: Path,
    *,
    cfg: DistillConfig,
    config_path: Path,
    backend: str,
    provider: ProviderConfig,
    harness: HarnessDoc,
    task_ids: Sequence[str],
    stats: RolloutStats,
    summaries: Sequence[EpisodeSummary],
    records: Sequence[TrialRecord],
    max_scaffold_loss: float,
    passed: bool,
) -> None:
    """Persist the stats plus every episode's record for later auditing.

    The `TrialRecord`s are written verbatim EXCEPT for the token spans, which
    are replaced by per-call lengths: the verbatim ids already live in the
    per-trial sink files (named in each episode's `token_sink`), and copying
    them here would duplicate gigabytes for a 100-turn TB2 wave.
    """
    payload = {
        "probe": "scaffold",
        "config_path": str(config_path),
        "config": cfg.model_dump(mode="json"),
        "backend": backend,
        "provider_config": provider.model_dump(mode="json"),
        "harness_doc_hash": harness.doc_hash,
        "task_ids": list(task_ids),
        "group_size": cfg.train.group_size,
        "trial_concurrency": cfg.train.trial_concurrency,
        "max_scaffold_loss": max_scaffold_loss,
        "passed": passed,
        "stats": stats.model_dump(mode="json"),
        "student_spend_usd": student_spend_usd(cfg, records),
        "episodes": [
            {
                **record.model_dump(mode="json", exclude={"spans"}),
                "turns": summary.turns,
                "sampled_tokens": summary.sampled_tokens,
                "final_prompt_tokens": summary.final_prompt_tokens,
                "token_sink": summary.token_sink,
                "spans": [
                    {
                        "call_index": span.call_index,
                        "prompt_tokens": len(span.prompt_token_ids),
                        "sampled_tokens": len(span.sampled_token_ids),
                    }
                    for span in record.spans
                ],
            }
            for record, summary in zip(records, summaries, strict=True)
        ],
    }
    write_json_atomic(path, payload)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """The probe's CLI surface."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", required=True, help="distill run TOML (loaded exactly as the CLI loads it)"
    )
    parser.add_argument("--task-ids", required=True, help="JSON array of harbor task ids to probe")
    parser.add_argument(
        "--run-dir", required=True, help="directory for the harbor jobs, token sinks, and report"
    )
    parser.add_argument(
        "--max-scaffold-loss",
        type=float,
        default=DEFAULT_MAX_SCAFFOLD_LOSS,
        help=f"pass threshold on the scaffold loss rate (default {DEFAULT_MAX_SCAFFOLD_LOSS})",
    )
    parser.add_argument(
        "--backend",
        choices=("local", "e2b"),
        default=None,
        help="override harbor.backend for this wave (default: whatever the config says)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one rollout wave and report the scaffold loss rate.

    Returns:
        0 when the scaffold loss rate is at or under the threshold, 1 when it
        is over it or the wave was a null measurement, 2 on a setup error.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not 0.0 <= args.max_scaffold_loss <= 1.0:
        logger.error(
            "--max-scaffold-loss must be a fraction in [0, 1], got %s", args.max_scaffold_loss
        )
        return 2
    if not os.environ.get(TINKER_API_KEY_ENV):
        # The literal is deliberate: interpolating `TINKER_API_KEY_ENV` here logs the env var's
        # NAME, not its value, but CodeQL flags any credential-shaped identifier reaching a log
        # sink (py/clear-text-logging-sensitive-data) and it is right to. The name is a constant,
        # so spelling it out loses nothing and removes the ambiguity for both the analyzer and a
        # reader who would otherwise have to check which of the two this is.
        logger.error("TINKER_API_KEY is not set; the probe samples the student live")
        return 2

    config_path = Path(args.config)
    run_dir = Path(args.run_dir)
    report_path = run_dir / REPORT_FILENAME
    try:
        cfg = load_distill_config(config_path)
        task_ids = load_task_ids(Path(args.task_ids))
        harness = probe_harness()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 2
    if report_path.exists():
        logger.error(
            "%s already holds a probe report; move it aside or choose a fresh --run-dir so "
            "the two waves' harbor dirs and token sinks cannot mix",
            report_path,
        )
        return 2
    if args.backend is not None and args.backend != cfg.harbor.backend:
        cfg = cfg.model_copy(
            update={"harbor": cfg.harbor.model_copy(update={"backend": args.backend})}
        )
    trials = len(task_ids) * cfg.train.group_size
    concurrency = min(cfg.train.trial_concurrency, trials)

    provider = student_provider(cfg)
    pinned = pin_rollout_params(harness, cfg)
    logger.info(
        "probing the pi scaffold: %d task(s) x %d attempt(s) = %d episode(s) of %s, backend "
        "%s, up to %d trial(s) in flight -> %s",
        len(task_ids),
        cfg.train.group_size,
        trials,
        cfg.student.base_model,
        cfg.harbor.backend,
        concurrency,
        run_dir,
    )
    try:
        preflight_context_window(cfg)
        preflight_student_sampler(cfg)
        if cfg.harbor.backend == "e2b":
            preflight_e2b_capacity(concurrency=concurrency)
    except (ImportError, RuntimeError, ValueError) as exc:
        logger.error("preflight failed: %s", exc)
        return 2

    try:
        records, stats = collect_rollouts(
            PROBE_STEP_INDEX, task_ids, cfg, pinned, provider, run_dir
        )
    except (ImportError, ValueError) as exc:
        logger.error(
            "the rollout wave could not start: %s. The harbor job template's "
            "environment.type and the effective backend (%s) must agree, so the "
            "checked-in TB2 template (environment.type: e2b) needs --backend e2b",
            exc,
            cfg.harbor.backend,
        )
        return 2
    token_sink_dir = run_dir / "tokens" / f"step-{PROBE_STEP_INDEX:04d}"
    summaries = summarize_episodes(records, token_sink_dir)
    passed = bool(stats.executed_trials) and stats.scaffold_loss_rate <= args.max_scaffold_loss
    write_report(
        report_path,
        cfg=cfg,
        config_path=config_path,
        backend=cfg.harbor.backend,
        provider=provider,
        harness=pinned,
        task_ids=task_ids,
        stats=stats,
        summaries=summaries,
        records=records,
        max_scaffold_loss=args.max_scaffold_loss,
        passed=passed,
    )
    for line in report_lines(
        cfg=cfg,
        config_path=config_path,
        backend=cfg.harbor.backend,
        task_ids=task_ids,
        stats=stats,
        summaries=summaries,
        records=records,
        max_scaffold_loss=args.max_scaffold_loss,
        report_path=report_path,
    ):
        print(line)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
