"""Reward-agreement demo: replace the real financebench environment with a world model.

The SAME agent runs the held-out test tasks twice — once against the real workspace environment
and once against a world model of it — and the SAME deterministic grader scores both. The bridge
is one small class: `WorldModelCommandEnv` exposes a `WorldModel` session through the
`CommandEnv.execute` seam, so the identical agent loop drives either backend. This is the
mechanical meaning of "the world model replaces the benchmark".

Known seam limitation: `Observation.is_error` is a boolean, so the bridge coarsens bash exit
codes to {0, 1}. Agents that branch on specific non-zero codes (e.g. grep's 1-vs-2) see the
same behavior on both backends only up to that coarsening.

Usage (after `wmh build --name financebench --file .../traces.otel.jsonl`):
    uv run python packages/environment-capture/financebench/wm_replace_demo.py \
        --model-dir .wmh/models/financebench --limit 5
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from environment_capture import ExecResult
from environment_capture.agent import BedrockBashAgent
from environment_capture.benchmarks.financebench import FinanceBenchAdapter
from environment_capture.trajectory import JsonValue, Task

from wmh.core.types import Action, ActionKind
from wmh.engine.loader import load_world_model
from wmh.engine.world_model import WorldModel
from wmh.env import Env, WorldModelEnv

_HERE = Path(__file__).parent


class WorldModelCommandEnv:
    """CommandEnv backed by a live WorldModelEnv episode (bash commands become tool_call steps)."""

    def __init__(self, env: Env, *, task: str) -> None:
        self._env = env
        self._env.reset(task=task)

    def execute(self, command: str) -> ExecResult:
        observation = self._env.step(
            Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": command})
        )
        return ExecResult(output=observation.content, returncode=1 if observation.is_error else 0)

    def close(self) -> None:
        self._env.close()


def _run_task(
    adapter: FinanceBenchAdapter,
    agent: BedrockBashAgent,
    task: Task,
    world_model: WorldModel,
) -> dict[str, JsonValue]:
    """Run one task against both backends; each env is closed even when the agent run raises."""
    real_env = adapter.open_env(task)
    try:
        real_run = agent.run(task, real_env)
    finally:
        real_env.close()
    real_reward = adapter.grade(task, real_run.final_answer)

    # The episode reset (inside WorldModelCommandEnv.__init__) can itself raise on a provider
    # error, so the underlying env's close is guarded from construction — symmetric with the
    # real-env branch above.
    wm_inner = WorldModelEnv(world_model)
    try:
        wm_env = WorldModelCommandEnv(wm_inner, task=task.prompt)
        wm_run = agent.run(task, wm_env)
    finally:
        wm_inner.close()
    wm_reward = adapter.grade(task, wm_run.final_answer)

    return {
        "task_id": task.task_id,
        "real_reward": real_reward,
        "real_steps": len(real_run.steps),
        "wm_reward": wm_reward,
        "wm_steps": len(wm_run.steps),
        "agree": real_reward == wm_reward,
        "real_answer": real_run.final_answer,
        "wm_answer": wm_run.final_answer,
        # Full transcripts so WM behavior can be audited against the real env's.
        "real_transcript": [
            {"command": s.action.arguments.get("command"), "output": s.output}
            for s in real_run.steps
        ],
        "wm_transcript": [
            {"command": s.action.arguments.get("command"), "output": s.output}
            for s in wm_run.steps
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=".wmh/models/financebench")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--agent-model", default="us.anthropic.claude-opus-4-8")
    parser.add_argument("--max-steps", type=int, default=10)
    args = parser.parse_args()

    adapter = FinanceBenchAdapter(data_root=_HERE)
    agent = BedrockBashAgent(args.agent_model, max_steps=args.max_steps)
    world_model, _provider = load_world_model(args.model_dir)

    rows: list[dict[str, JsonValue]] = []
    failures: list[dict[str, str]] = []
    for task in adapter.tasks(args.split)[: args.limit]:
        # Per-task isolation, mirroring run_capture: one throttled or crashed task must not
        # take down the rest of the comparison run.
        try:
            row = _run_task(adapter, agent, task, world_model)
        except Exception:  # noqa: BLE001 - isolation is the contract; error recorded
            failures.append({"task_id": task.task_id, "error": traceback.format_exc()})
            print(f"{task.task_id}: FAILED (see failures in the output JSON)")
            continue
        rows.append(row)
        print(
            f"{task.task_id}: real={row['real_reward']:.1f} ({row['real_steps']} steps)  "
            f"wm={row['wm_reward']:.1f} ({row['wm_steps']} steps)  "
            f"{'AGREE' if row['agree'] else 'DISAGREE'}"
        )

    n = len(rows)
    agreement = sum(1 for row in rows if row["agree"]) / n if n else 0.0
    real_mean = sum(float(row["real_reward"]) for row in rows) / n if n else 0.0
    wm_mean = sum(float(row["wm_reward"]) for row in rows) / n if n else 0.0
    summary = {
        "n_tasks": n,
        "reward_agreement": agreement,
        "real_mean_reward": real_mean,
        "wm_mean_reward": wm_mean,
        "agent_model": args.agent_model,
        "model_dir": args.model_dir,
        "tasks": rows,
        "failures": failures,
    }
    print(
        f"\nagreement {agreement:.2f} over {n} tasks | mean reward real {real_mean:.2f} "
        f"vs wm {wm_mean:.2f}"
        + (f" | {len(failures)} task(s) failed" if failures else "")
    )
    runs_dir = _HERE / "runs"
    runs_dir.mkdir(exist_ok=True)
    out = runs_dir / f"wm-replace-{int(time.time())}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
