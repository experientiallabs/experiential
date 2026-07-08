"""Filtered behavior cloning collection: the BASE student generates its own training data.

No teacher, no hint. The base Qwen student rolls each scenario in the frozen world model
(DeepSeek v3.2 backend); a Kimi K2.5 judge grades each episode against the scenario checklist;
only passing episodes are kept (capped per scenario). The student's own reasoning (vLLM's
`reasoning` field) is preserved as <think> blocks in the exported SFT targets.

Usage (from the repo root, with the student tunnel up on :18002):
    uv run python .agents/scripts/collect_bc.py --pool bc_pool_mined.json --out bc_mined.jsonl
    uv run python .agents/scripts/collect_bc.py --pool bc_pool_random.json --out bc_random.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / ".agents" / "scripts"))

from collect_teacher import domain_tool_hint  # noqa: E402
from eval_student import student_system  # noqa: E402
from collect_teacher import foundry, opus_judge  # noqa: E402
from run_scenario_e2e import TRACES, WM_DIR  # noqa: E402

from wmh.core.types import Action, ActionKind, Observation, Step  # noqa: E402
from wmh.engine.world_model import WorldModel  # noqa: E402
from wmh.env.base import WorldModelEnv  # noqa: E402
from wmh.ingest import get_adapter  # noqa: E402
from wmh.scenarios import ChecklistJudge, ScenarioSet  # noqa: E402

DISTILL = REPO / ".agents" / "docs" / "research" / "distill"
WM_MODEL = "gpt-5.4"  # Azure Foundry
# judge: Opus 4.8 (AWS claas-bedrock us-east-2) via opus_judge()
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
MAX_STEPS = 10
KEEP_PER_SCENARIO = 2
WORKERS = 5


def run_self_episode(client, model, world_model, scenario, tool_hint, temperature):  # noqa: ANN001, ANN201
    """One base-student rollout; returns (steps_for_judge, sft_turns, final_text)."""
    env = WorldModelEnv(world_model)
    env.reset(task=scenario.task, seed_state=scenario.seed_state)
    system = student_system(tool_hint)
    user = scenario.task
    if scenario.seed_state.scratchpad:
        user += f"\n\nEnvironment notes: {scenario.seed_state.scratchpad}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    steps: list[Step] = []
    sft_turns: list[dict] = []  # {think, raw_visible, observation}
    final_text = ""
    try:
        for _ in range(MAX_STEPS):
            response = client.chat.completions.create(
                model=model, messages=messages, max_tokens=10240, temperature=temperature
            )
            message = response.choices[0].message
            visible = (message.content or "").strip()
            think = (getattr(message, "reasoning", None) or "").strip()
            stripped = THINK_RE.sub("", visible)
            match = TOOL_CALL_RE.search(stripped)
            if match is None:
                final_text = stripped.strip()
                sft_turns.append({"think": think, "visible": final_text, "observation": None})
                break
            try:
                data = json.loads(match.group(1).strip())
                arguments = data.get("arguments") or {}
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                action = Action(
                    kind=ActionKind.TOOL_CALL, name=str(data.get("name", "")), arguments=arguments
                )
            except (json.JSONDecodeError, TypeError):
                final_text = stripped.strip()[:400]
                break
            observation = env.step(action)
            steps.append(Step(action=action, observation=observation))
            call = json.dumps({"name": action.name, "arguments": action.arguments}, default=str)
            sft_turns.append(
                {
                    "think": think,
                    "visible": f"<tool_call>{call}</tool_call>",
                    "observation": observation.content,
                }
            )
            messages.append({"role": "assistant", "content": f"<tool_call>{call}</tool_call>"})
            messages.append(
                {"role": "user", "content": f"<tool_response>{observation.content}</tool_response>"}
            )
    finally:
        env.close()
    return steps, sft_turns, final_text, system, user


def to_messages(system, user, sft_turns):  # noqa: ANN001, ANN201
    """Render kept episode as chat messages; each assistant turn = <think> + visible text."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for turn in sft_turns:
        content = turn["visible"]
        if turn["think"]:
            content = f"<think>{turn['think']}</think>\n{content}"
        messages.append({"role": "assistant", "content": content})
        if turn["observation"] is not None:
            messages.append(
                {"role": "user", "content": f"<tool_response>{turn['observation']}</tool_response>"}
            )
    return messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--endpoint", default="http://localhost:18002/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--samples", type=int, default=6)
    args = parser.parse_args()

    from openai import OpenAI

    client = OpenAI(base_url=args.endpoint, api_key="not-needed")
    judge = ChecklistJudge(opus_judge())
    world_model = WorldModel.load(str(WM_DIR), foundry(WM_MODEL), telemetry_root=str(REPO / ".wmh"))
    pool = ScenarioSet.load(DISTILL / args.pool)
    traces = get_adapter("otel-genai").from_file(str(TRACES))
    t0 = time.time()

    def collect_for(scenario):  # noqa: ANN001, ANN202
        tool_hint = domain_tool_hint(traces, scenario.provenance[0])
        kept = []
        for _ in range(args.samples):
            try:
                steps, sft_turns, final_text, system, user = run_self_episode(
                    client, args.model, world_model, scenario, tool_hint, temperature=1.0
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  episode error {scenario.scenario_id}: {str(exc)[:100]}", flush=True)
                continue
            judged = list(steps)
            if final_text:
                judged.append(
                    Step(
                        action=Action(kind=ActionKind.MESSAGE, content=final_text),
                        observation=Observation(content=""),
                    )
                )
            verdict = judge.score(scenario.task, scenario.checklist, judged)
            if verdict.success and steps:
                kept.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "source_trace": scenario.provenance[0],
                        "pass_rate": verdict.pass_rate,
                        "n_turns": len(sft_turns),
                        "messages": to_messages(system, user, sft_turns),
                    }
                )
        kept.sort(key=lambda r: (-r["pass_rate"], r["n_turns"]))
        kept = kept[:KEEP_PER_SCENARIO]
        print(
            f"  {scenario.scenario_id}: kept {len(kept)}/{args.samples} ({time.time() - t0:.0f}s)",
            flush=True,
        )
        return kept

    records = []
    with world_model.frozen(), ThreadPoolExecutor(max_workers=WORKERS) as pool_executor:
        for kept in pool_executor.map(collect_for, pool.scenarios):
            records.extend(kept)

    out = DISTILL / args.out
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"kept {len(records)} episodes -> {out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
