"""Bench-VERIFY corpus loader: bird-sql trajectories with deterministic recorded outcomes.

Ground truth is the bird-sql execution-match grade recorded in each trace's metadata
(`reward` in {0.0, 1.0}); the gold SQL lives in the corpus `gold/<base_task_id>.json` sidecar.
No LLM ever assigns a label here, so the VERIFY benchmark measures the outcome judge against a
truly independent signal.

We reuse the production OTel adapter (`wmh.ingest.otel_genai.OtelGenAIAdapter`) to turn the span
corpus into `Trace` objects, and reproduce `RunResult.transcript()` verbatim so the judge sees the
exact transcript format it receives in closed-loop eval (`wmh.evals.closed_loop`).

The bird-sql corpus is not materialized inside this worktree; by default we read it from the main
checkout (read-only). Override with --corpus-root / --gold-dir.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from wmh.core.types import ActionKind, Trace
from wmh.harness.runtime import TRANSCRIPT_OBS_CHARS
from wmh.ingest.otel_genai import OtelGenAIAdapter

# The committed corpus lives in the main checkout (the worktree carries only model artifacts).
DEFAULT_CORPUS_ROOT = Path(
    "/Users/silen/Desktop/Projects/world-model-harness/packages/environment-capture/bird-sql"
)


@dataclass(frozen=True)
class VerifyCase:
    """One labeled VERIFY trajectory: judge inputs plus the deterministic recorded outcome."""

    trace_id: str
    base_task_id: str
    task_id: str
    model: str
    instruction: str  # the question + evidence hint (what the judge grades against)
    answer: str  # the agent's submitted SQL (final_answer)
    transcript: str  # production-format judge transcript
    gold: list[str]  # gold assertion(s) derived from the reference SQL
    gold_sql: str
    recorded_pass: bool  # ground truth: reward >= 1.0 (execution match)
    n_steps: int


def _transcript(trace: Trace) -> str:
    """Reproduce `RunResult.transcript()` so the judge sees the production transcript format."""
    lines: list[str] = []
    for i, step in enumerate(trace.steps, 1):
        act = step.action
        desc = act.name or (act.content or "")
        if act.kind == ActionKind.TOOL_CALL and act.arguments:
            desc = f"{act.name} {act.arguments}"
        lines.append(f"[{i}] {act.kind.value}: {desc}")
        lines.append(f"    -> {step.observation.content[:TRANSCRIPT_OBS_CHARS]}")
    return "\n".join(lines)


def _gold_assertion(gold_sql: str) -> list[str]:
    """One semantic post-condition mirroring the deterministic execution-match grade.

    bird-sql scores a run by executing the agent's submitted SQL and the reference SQL against the
    same database and comparing result sets. We hand the outcome judge that same success condition
    as a gold assertion: the transcript shows the queries the agent ran and the rows they returned,
    so the judge has the evidence to decide whether the submitted query is equivalent.
    """
    return [
        "The agent's final submitted SQL query answers the question and is semantically "
        "equivalent to (returns the same result set as) this reference query: " + gold_sql.strip()
    ]


def load_cases(
    corpus_root: Path = DEFAULT_CORPUS_ROOT,
    gold_dir: Path | None = None,
) -> list[VerifyCase]:
    """Load every bird-sql trace that has a recorded reward and a gold-SQL sidecar."""
    traces_path = corpus_root / "traces.otel.jsonl"
    gold_dir = gold_dir if gold_dir is not None else corpus_root / "gold"
    traces = OtelGenAIAdapter().from_file(str(traces_path))

    cases: list[VerifyCase] = []
    for trace in traces:
        meta = trace.metadata
        reward = meta.get("reward")
        base_task_id = meta.get("base_task_id")
        final_answer = meta.get("final_answer") or ""
        if not isinstance(reward, (int, float)) or not isinstance(base_task_id, str):
            continue
        gold_path = gold_dir / f"{base_task_id}.json"
        if not gold_path.exists():
            continue
        gold_sql = json.loads(gold_path.read_text(encoding="utf-8")).get("gold_sql", "")
        if not gold_sql or not trace.steps:
            continue
        instruction = trace.steps[0].task or ""
        cases.append(
            VerifyCase(
                trace_id=trace.trace_id,
                base_task_id=base_task_id,
                task_id=str(meta.get("task_id", trace.trace_id)),
                model=str(meta.get("model", "")),
                instruction=instruction,
                answer=str(final_answer),
                transcript=_transcript(trace),
                gold=_gold_assertion(gold_sql),
                gold_sql=gold_sql,
                recorded_pass=float(reward) >= 1.0,
                n_steps=len(trace.steps),
            )
        )
    return cases


def select_balanced(
    cases: list[VerifyCase],
    n_per_class: int = 20,
    seed: int = 7,
    max_transcript_chars: int = 12_000,
) -> list[VerifyCase]:
    """Deterministic balanced sample: n_per_class pass + n_per_class fail, distinct base tasks.

    Distinct `base_task_id` keeps the sample from stacking many reruns of one question (task
    diversity), and the transcript cap keeps judge cost/latency bounded. Selection is seeded so the
    exact sample is reproducible.
    """
    rng = random.Random(seed)
    eligible = [c for c in cases if len(c.transcript) <= max_transcript_chars]
    passes = [c for c in eligible if c.recorded_pass]
    fails = [c for c in eligible if not c.recorded_pass]
    rng.shuffle(passes)
    rng.shuffle(fails)

    def take_distinct(pool: list[VerifyCase], k: int) -> list[VerifyCase]:
        seen: set[str] = set()
        picked: list[VerifyCase] = []
        for c in pool:
            if c.base_task_id in seen:
                continue
            seen.add(c.base_task_id)
            picked.append(c)
            if len(picked) == k:
                break
        return picked

    picked_pass = take_distinct(passes, n_per_class)
    # Keep the fail base tasks disjoint from the pass base tasks so no question appears twice.
    pass_bases = {c.base_task_id for c in picked_pass}
    fails_disjoint = [c for c in fails if c.base_task_id not in pass_bases]
    picked_fail = take_distinct(fails_disjoint, n_per_class)

    sample = picked_pass + picked_fail
    rng.shuffle(sample)
    return sample
