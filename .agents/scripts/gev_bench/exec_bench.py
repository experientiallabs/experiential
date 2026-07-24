"""Bench-EXEC: closed-loop rollouts inside the world model vs the real bird-sql environment.

The EXECUTE leg of the GEV triple. It runs the SAME candidate policy (the fixed wmh
``AgentRuntime``) and the SAME verifiers on each scenario against (a) the REAL sqlite environment
(the bird-sql ``LocalBashEnv`` staged by ``BirdSqlAdapter``) and (b) the SIMULATED environment (the
committed bird-sql world model), then reports per-scenario outcome agreement, the sim-optimism gap,
and policy rank agreement between two candidate models.

Two verifiers score every episode, kept as separate columns and never blended:

- deterministic: the agent's final submitted SQL is executed against a pristine read-only copy of
  the REAL database and its rows compared to the gold query's rows (``BirdSqlAdapter.grade``). This
  is environment-agnostic: it grades the agent's OUTPUT, so the same grader scores both a real-env
  episode and a sim-env episode, which is exactly what "does sim behavior succeed in reality" asks.
- judge: an LLM ``GoldJudge`` (Opus 4.8) scores each transcript+answer against a gold assertion
  derived from the gold result set. One common verifier across both environments; its agreement
  with the deterministic grader on real episodes is reported as a bonus verify datapoint.

Usage (from the worktree root):
    uv run python .agents/scripts/gev_bench/exec_bench.py --smoke      # 1 scenario x 1 candidate
    uv run python .agents/scripts/gev_bench/exec_bench.py --scenarios 8 --k 2   # full grid
"""

from __future__ import annotations

import argparse
import json
import shlex
import sqlite3
import time
from dataclasses import replace
from pathlib import Path
from statistics import fmean

from environment_capture.benchmarks.bird_sql import (
    BirdSqlAdapter,
    extract_sql,
    question_implies_order,
    rows_match,
)
from environment_capture.trajectory import Task as BirdTask

from wmh.core.types import Action, Observation
from wmh.engine import load_world_model
from wmh.evals.closed_loop import RolloutEvidence, evaluate_with_env
from wmh.evals.gold import GoldJudge
from wmh.evals.tasks import TaskSpec
from wmh.harness.runtime import AgentRuntime
from wmh.providers import get_provider
from wmh.providers.base import ProviderConfig, ProviderKind

_REPO = Path(__file__).resolve().parents[3]
_BIRD_ROOT = _REPO / "packages" / "environment-capture" / "bird-sql"
_MODEL_DIR = _BIRD_ROOT / "models" / "bird-sql"
_OUT_DIR = _REPO / ".agents" / "docs" / "research" / "gev_bench_results" / "exec"

# The how-to-submit-SQL framing the generic agent prompt does not carry (mirrors capture.py). The
# SAME instruction is handed to both environments so any outcome difference is the environment's.
_SQL_INSTRUCTIONS = (
    "\n\nThe SQLite database is ./database.db and its DDL schema is ./schema.sql. Read the schema, "
    'then explore the data with the sqlite3 CLI (e.g. `sqlite3 database.db "SELECT ..."`). When '
    "confident, call submit with your final answer set to a single SQLite SELECT query (no prose) "
    "that answers the question."
)

_JUDGE_REGION = "us-east-1"
_AGENT_MAX_TURNS = 12  # matches the capture agent's max_steps; bounds cost per episode
_GOLD_ROWS_CAP = 40  # rows of the gold result set shown to the judge
_GOLD_REPR_CHARS = 1500

# Candidate policy models (both Bedrock). model_type -> runtime model id.
_CANDIDATES: dict[str, str] = {
    "haiku-4.5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "opus-4.8": "us.anthropic.claude-opus-4-8",
}
_CANDIDATE_TYPE = {
    "haiku-4.5": "claude-haiku-4-5",
    "opus-4.8": "claude-opus-4-8",
}


class RealBirdEnvironment:
    """AgentEnvironment backed by the real bird-sql sqlite workspace (a LocalBashEnv).

    The wmh runtime emits ``bash``/``read_file``/``write_file`` actions; the sqlite workspace is a
    plain shell, so file tools are expressed as shell commands. Every observation is the REAL
    command output, the counterpart to the world model's generated observation.
    """

    def __init__(self, adapter: BirdSqlAdapter, task: BirdTask) -> None:
        self._env = adapter.open_env(task)

    def execute(self, action: Action) -> Observation:
        name = action.name
        args = action.arguments
        if name == "bash":
            command = args.get("command", "")
        elif name == "read_file":
            command = f"cat {shlex.quote(str(args.get('path', '')))}"
        elif name == "write_file":
            path = shlex.quote(str(args.get("path", "")))
            content = str(args.get("content", ""))
            command = f"cat > {path} <<'WMH_EOF'\n{content}\nWMH_EOF"
        else:
            return Observation(content=f"tool {name!r} not available", is_error=True)
        result = self._env.execute(command if isinstance(command, str) else str(command))
        return Observation(content=result.output, is_error=result.returncode != 0)

    def close(self) -> None:
        self._env.close()


def _gold_rows(bird_task: BirdTask, adapter: BirdSqlAdapter) -> tuple[list[tuple], str]:
    """Execute the task's gold SQL against the real db; return (rows, compact repr for the judge)."""
    gold = json.loads((_BIRD_ROOT / "gold" / f"{bird_task.task_id}.json").read_text())
    db = _BIRD_ROOT / "databases" / f"{bird_task.data['db_name']}.sqlite"
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(gold["gold_sql"]).fetchall()
    finally:
        con.close()
    shown = rows[:_GOLD_ROWS_CAP]
    repr_str = "\n".join(str(r) for r in shown)
    if len(rows) > _GOLD_ROWS_CAP:
        repr_str += f"\n... ({len(rows) - _GOLD_ROWS_CAP} more rows)"
    return rows, repr_str[:_GOLD_REPR_CHARS]


def _gold_assertion(bird_task: BirdTask, adapter: BirdSqlAdapter) -> str:
    """A single observable post-condition for the LLM judge, phrased from the gold result set."""
    _rows, repr_str = _gold_rows(bird_task, adapter)
    return (
        "The agent's final submitted answer is a single SQLite SELECT/WITH query that correctly "
        "answers the question, and executing it would return exactly this result set "
        "(row order matters only if the question implies an ordering):\n"
        f"{repr_str}"
    )


def _load_scenarios(n: int, adapter: BirdSqlAdapter) -> tuple[list[TaskSpec], dict[str, BirdTask]]:
    """Pick `n` test-split tasks spread across databases; build TaskSpecs and the bird-task map.

    Test-split tasks are held-out questions on the SAME databases the world model trained on, so the
    world model must simulate new questions rather than replay a memorized task.
    """
    tasks = adapter.tasks("test")
    by_db: dict[str, list[BirdTask]] = {}
    for t in tasks:
        by_db.setdefault(t.data["db_name"], []).append(t)
    # Round-robin across databases for schema variety, deterministic order.
    ordered: list[BirdTask] = []
    dbs = sorted(by_db)
    i = 0
    while len(ordered) < n and any(by_db.values()):
        db = dbs[i % len(dbs)]
        if by_db[db]:
            ordered.append(by_db[db].pop(0))
        i += 1
        if i > n * len(dbs):
            break
    ordered = ordered[:n]
    specs: list[TaskSpec] = []
    bird_map: dict[str, BirdTask] = {}
    for t in ordered:
        specs.append(
            TaskSpec(
                task_id=t.task_id,
                instruction=t.prompt + _SQL_INSTRUCTIONS,
                gold=[_gold_assertion(t, adapter)],
            )
        )
        bird_map[t.task_id] = t
    return specs, bird_map


def _grade_detail(bird_task: BirdTask, evidence: RolloutEvidence) -> tuple[float, str, str]:
    """Grade the episode's final SQL against the REAL db (execution match), with an error class.

    Same logic as ``BirdSqlAdapter.grade`` (extract_sql -> execute against a pristine read-only
    copy -> row multiset compare, order-sensitive only when the question implies order), plus an
    ``error_kind`` so the report can count execute-step defects. ``no_such_table``/
    ``no_such_column`` mean the agent's final query references schema the REAL database does not
    have: in a SIM episode that is the world model having let the agent operate on a HALLUCINATED
    schema. Returns (score, extracted_sql, error_kind).
    """
    sql = extract_sql(evidence.answer)
    if not sql:
        return 0.0, "", "no_sql"
    gold_sql = json.loads(
        (_BIRD_ROOT / "gold" / f"{bird_task.task_id}.json").read_text()
    )["gold_sql"]
    db = _BIRD_ROOT / "databases" / f"{bird_task.data['db_name']}.sqlite"
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        gold_rows = con.execute(gold_sql).fetchall()
        try:
            pred_rows = con.execute(sql).fetchall()
        except sqlite3.Error as exc:
            msg = str(exc).lower()
            if "no such table" in msg:
                kind = "no_such_table"
            elif "no such column" in msg:
                kind = "no_such_column"
            else:
                kind = "sql_error"
            return 0.0, sql, kind
    finally:
        con.close()
    ok = rows_match(pred_rows, gold_rows, order_sensitive=question_implies_order(bird_task.prompt))
    return (1.0 if ok else 0.0), sql, ("none" if ok else "wrong_result")


# Final-SQL error classes that mean the agent referenced schema absent from the real database.
_SCHEMA_HALLUCINATION_KINDS = frozenset({"no_such_table", "no_such_column"})


def _run_condition(
    label: str,
    candidate: str,
    env_kind: str,
    specs: list[TaskSpec],
    make_env,
    agent_provider,
    judge: GoldJudge,
    bird_map: dict[str, BirdTask],
    adapter: BirdSqlAdapter,
    k: int,
    concurrency: int,
) -> list[dict]:
    """Run one (candidate, environment) condition; return per-episode rows with both scores."""
    runtime = AgentRuntime(agent_provider, max_turns=_AGENT_MAX_TURNS)
    report = evaluate_with_env(
        specs, make_env, runtime, judge, label=label, k=k, concurrency=concurrency
    )
    rows: list[dict] = []
    for task_id, outcome in report.per_task.items():
        bird_task = bird_map[task_id]
        for attempt_idx, (verdict, attempt) in enumerate(
            zip(outcome.verdicts, outcome.attempts)
        ):
            det_score, sql, error_kind = _grade_detail(bird_task, attempt)
            rows.append(
                {
                    "candidate": candidate,
                    "env": env_kind,
                    "task_id": task_id,
                    "db": bird_task.data["db_name"],
                    "attempt": attempt_idx,  # 0-based pass index, NOT a pass/fail flag
                    "det_score": det_score,  # 1.0 = final SQL execution-matches gold on the real db
                    "judge_passed": verdict.passed,
                    "judge_fraction": verdict.fraction,
                    "error_kind": error_kind,  # none|wrong_result|no_such_table|no_such_column|sql_error|no_sql
                    "schema_hallucination": error_kind in _SCHEMA_HALLUCINATION_KINDS,
                    "pred_sql": sql,
                    "stop_reason": attempt.stop_reason.value,
                    "turns": attempt.turns,
                }
            )
    return rows


def _cell_rate(rows: list[dict], candidate: str, env: str, task_id: str, key: str) -> float:
    """Mean of `key` over the k passes of one (candidate, env, task) cell."""
    vals = [
        r[key]
        for r in rows
        if r["candidate"] == candidate and r["env"] == env and r["task_id"] == task_id
    ]
    return fmean(vals) if vals else 0.0


def _confusion(
    rows: list[dict], candidate: str, task_ids: list[str], key: str, threshold: float = 0.5
) -> dict:
    """Sim-vs-real confusion for one candidate over its scenarios, binarized at `threshold`."""
    c = {"sim_pass_real_pass": 0, "sim_pass_real_fail": 0, "sim_fail_real_pass": 0, "sim_fail_real_fail": 0}
    for task_id in task_ids:
        sim = _cell_rate(rows, candidate, "sim", task_id, key) >= threshold
        real = _cell_rate(rows, candidate, "real", task_id, key) >= threshold
        if sim and real:
            c["sim_pass_real_pass"] += 1
        elif sim and not real:
            c["sim_pass_real_fail"] += 1
        elif not sim and real:
            c["sim_fail_real_pass"] += 1
        else:
            c["sim_fail_real_fail"] += 1
    return c


def _agreement(confusion: dict) -> float | None:
    total = sum(confusion.values())
    if total == 0:
        return None
    return (confusion["sim_pass_real_pass"] + confusion["sim_fail_real_fail"]) / total


def _mean_rate(rows: list[dict], candidate: str, env: str, task_ids: list[str], key: str) -> float:
    return fmean(_cell_rate(rows, candidate, env, t, key) for t in task_ids) if task_ids else 0.0


def _metrics(rows: list[dict], candidates: list[str], task_ids: list[str]) -> dict:
    """Compute outcome agreement, sim-optimism gap, rank agreement, judge-vs-deterministic."""
    out: dict = {
        "per_candidate": {},
        "rank_agreement": {},
        "judge_vs_deterministic": {},
        "schema_hallucination": {},
    }
    for cand in candidates:
        det_conf = _confusion(rows, cand, task_ids, "det_score")
        judge_conf = _confusion(rows, cand, task_ids, "judge_passed")
        cand_sim = [r for r in rows if r["candidate"] == cand and r["env"] == "sim"]
        cand_real = [r for r in rows if r["candidate"] == cand and r["env"] == "real"]
        out["schema_hallucination"][cand] = {
            "sim_rate": _rate(cand_sim, "schema_hallucination"),
            "real_rate": _rate(cand_real, "schema_hallucination"),
            "sim_error_kinds": _error_kind_counts(cand_sim),
            "real_error_kinds": _error_kind_counts(cand_real),
        }
        out["per_candidate"][cand] = {
            "deterministic": {
                "sim_mean_pass": _mean_rate(rows, cand, "sim", task_ids, "det_score"),
                "real_mean_pass": _mean_rate(rows, cand, "real", task_ids, "det_score"),
                "sim_optimism_gap": _mean_rate(rows, cand, "sim", task_ids, "det_score")
                - _mean_rate(rows, cand, "real", task_ids, "det_score"),
                "outcome_agreement": _agreement(det_conf),
                "confusion": det_conf,
            },
            "judge": {
                "sim_mean_pass": _mean_rate(rows, cand, "sim", task_ids, "judge_passed"),
                "real_mean_pass": _mean_rate(rows, cand, "real", task_ids, "judge_passed"),
                "sim_optimism_gap": _mean_rate(rows, cand, "sim", task_ids, "judge_passed")
                - _mean_rate(rows, cand, "real", task_ids, "judge_passed"),
                "outcome_agreement": _agreement(judge_conf),
                "confusion": judge_conf,
            },
        }

    # Rank agreement (deterministic): does the sim pick the same winning candidate as reality?
    if len(candidates) == 2:
        a, b = candidates
        for verifier in ("deterministic", "judge"):
            key = "det_score" if verifier == "deterministic" else "judge_passed"
            real_a = _mean_rate(rows, a, "real", task_ids, key)
            real_b = _mean_rate(rows, b, "real", task_ids, key)
            sim_a = _mean_rate(rows, a, "sim", task_ids, key)
            sim_b = _mean_rate(rows, b, "sim", task_ids, key)
            real_winner = a if real_a > real_b else (b if real_b > real_a else "tie")
            sim_winner = a if sim_a > sim_b else (b if sim_b > sim_a else "tie")
            # Per-scenario rank agreement: sign of (a - b) matches between sim and real.
            per_scenario = []
            for t in task_ids:
                rd = _cell_rate(rows, a, "real", t, key) - _cell_rate(rows, b, "real", t, key)
                sd = _cell_rate(rows, a, "sim", t, key) - _cell_rate(rows, b, "sim", t, key)
                per_scenario.append(_sign(rd) == _sign(sd))
            out["rank_agreement"][verifier] = {
                "real_scores": {a: real_a, b: real_b},
                "sim_scores": {a: sim_a, b: sim_b},
                "real_winner": real_winner,
                "sim_winner": sim_winner,
                "overall_agree": real_winner == sim_winner,
                "per_scenario_agreement": fmean(per_scenario) if per_scenario else None,
            }

    # Judge-vs-deterministic agreement on REAL episodes (bonus verify datapoint).
    real_rows = [r for r in rows if r["env"] == "real"]
    if real_rows:
        agree = sum(1 for r in real_rows if (r["judge_passed"]) == (r["det_score"] >= 0.5))
        false_pass = sum(1 for r in real_rows if r["judge_passed"] and r["det_score"] < 0.5)
        false_fail = sum(1 for r in real_rows if not r["judge_passed"] and r["det_score"] >= 0.5)
        out["judge_vs_deterministic"] = {
            "n_real_episodes": len(real_rows),
            "accuracy": agree / len(real_rows),
            "false_pass": false_pass,  # judge says pass, deterministic says fail
            "false_fail": false_fail,  # judge says fail, deterministic says pass
        }
    return out


def _sign(x: float) -> int:
    return (x > 1e-9) - (x < -1e-9)


def _rate(rows: list[dict], key: str) -> float | None:
    """Fraction of `rows` where boolean `key` is true (None over an empty set)."""
    return (sum(1 for r in rows if r[key]) / len(rows)) if rows else None


def _error_kind_counts(rows: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["error_kind"]] = counts.get(r["error_kind"], 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=int, default=8)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--smoke", action="store_true", help="1 scenario x 1 candidate x k")
    parser.add_argument("--candidates", default="haiku-4.5,opus-4.8")
    parser.add_argument("--real-concurrency", type=int, default=4)
    parser.add_argument("--sim-concurrency", type=int, default=3)
    parser.add_argument("--out-prefix", default="")
    args = parser.parse_args()

    candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]
    n_scen = args.scenarios
    k = args.k
    if args.smoke:
        candidates = candidates[:1]
        n_scen = 1
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    adapter = BirdSqlAdapter(data_root=_BIRD_ROOT)
    specs, bird_map = _load_scenarios(n_scen, adapter)
    task_ids = [s.task_id for s in specs]
    print(f"scenarios ({len(specs)}): {task_ids}")

    judge_provider = get_provider(
        ProviderConfig(
            kind=ProviderKind.BEDROCK,
            model_type="claude-opus-4-8",
            model="us.anthropic.claude-opus-4-8",
            region=_JUDGE_REGION,
        )
    )
    judge = GoldJudge(judge_provider)

    world_model, _serve_provider = load_world_model(_MODEL_DIR)

    candidate_providers = {
        cand: get_provider(
            ProviderConfig(
                kind=ProviderKind.BEDROCK,
                model_type=_CANDIDATE_TYPE[cand],
                model=_CANDIDATES[cand],
                region=_JUDGE_REGION,
            )
        )
        for cand in candidates
    }

    started = time.time()
    all_rows: list[dict] = []
    for cand in candidates:
        agent_provider = candidate_providers[cand]
        # Real environment.
        print(f"[{cand}] real env, k={k} ...")
        real_rows = _run_condition(
            f"real@{cand}",
            cand,
            "real",
            specs,
            lambda spec: RealBirdEnvironment(adapter, bird_map[spec.task_id]),
            agent_provider,
            judge,
            bird_map,
            adapter,
            k,
            args.real_concurrency,
        )
        all_rows.extend(real_rows)
        # Simulated environment (world model). Freeze the index so concurrent stepping cannot
        # mutate the shared retrieval buffer mid-eval; sessions are already enrich=False.
        print(f"[{cand}] sim env (world model), k={k} ...")
        from wmh.evals.closed_loop import WorldModelEnvironment

        with world_model.frozen() as wm:
            sim_rows = _run_condition(
                f"sim@{cand}",
                cand,
                "sim",
                specs,
                lambda spec: WorldModelEnvironment(wm, task=spec.instruction),
                agent_provider,
                judge,
                bird_map,
                adapter,
                k,
                args.sim_concurrency,
            )
        all_rows.extend(sim_rows)

    elapsed = time.time() - started
    metrics = _metrics(all_rows, candidates, task_ids)
    metrics["meta"] = {
        "scenarios": task_ids,
        "candidates": candidates,
        "k": k,
        "n_episodes": len(all_rows),
        "elapsed_s": round(elapsed, 1),
        "world_model": str(_MODEL_DIR.relative_to(_REPO)),
        "smoke": args.smoke,
    }

    prefix = args.out_prefix or ("smoke_" if args.smoke else "")
    raw_path = _OUT_DIR / f"{prefix}episodes.jsonl"
    metrics_path = _OUT_DIR / f"{prefix}metrics.json"
    with raw_path.open("w", encoding="utf-8") as fh:
        for r in all_rows:
            fh.write(json.dumps(r) + "\n")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nwrote {len(all_rows)} episodes -> {raw_path}")
    print(f"wrote metrics -> {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
