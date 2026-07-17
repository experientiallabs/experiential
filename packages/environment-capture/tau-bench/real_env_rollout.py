#!/usr/bin/env python3
"""Roll a student endpoint against the REAL tau2 environment on mined eval scenarios.

Sim2real leg of the distillation verification: the same scenarios the world-model eval uses, but
tool calls execute against Sierra's real domain environments (real JSON DB). Transcripts are
dumped as JSONL; judging happens wmh-side (score the transcripts with the pinned RubricJudge
from a wmh checkout) so this venv stays
tau2+openai only — no wmh import (see README: wmh never imports tau2 and vice versa).

Usage (from examples/tau-bench, in the tau2 venv):
    TAU2_DATA_DIR="$PWD/tau2-bench/data" .venv/bin/python real_env_rollout.py \
        --scenarios ../../.agents/docs/research/distill/eval_pool.json \
        --endpoint http://localhost:18001/v1 --model Qwen/Qwen3.5-9B \
        --label student-before --out ../../.agents/docs/research/distill/real_rollouts_before.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from openai import OpenAI
from tau2.registry import registry

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
MAX_STEPS = 10
_CORPUS = Path(__file__).resolve().parent / "traces.otel.jsonl"


def trace_domains(corpus: Path) -> dict[str, str]:
    domains: dict[str, str] = {}
    for line in corpus.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        span = json.loads(line)
        attrs = {a["key"]: a.get("value", {}).get("stringValue", "") for a in span.get("attributes", [])}
        meta = attrs.get("wmh.trace.metadata")
        if meta:
            domains[span["traceId"]] = json.loads(meta).get("domain", "")
    return domains


def system_prompt(tool_names: list[str]) -> str:
    return (
        "You are a customer-service agent operating in a tool environment.\n"
        f"Available tools (from prior episodes in this domain): {', '.join(tool_names)}\n"
        "Think before every action inside <think></think>, then either emit exactly one\n"
        '<tool_call>{"name": ..., "arguments": {...}}</tool_call>\n'
        "or reply to the user in plain text when the task is complete. Always check policy "
        "constraints before irreversible actions."
    )


def run_episode(client: OpenAI, model: str, env, tool_names: list[str], scenario: dict, temperature: float) -> dict:
    messages = [{"role": "system", "content": system_prompt(tool_names)}]
    user = scenario["task"]
    seed = (scenario.get("seed_state") or {}).get("scratchpad", "")
    if seed:
        user += f"\n\nEnvironment notes: {seed}"
    messages.append({"role": "user", "content": user})
    steps = []
    final_text = ""
    for _ in range(MAX_STEPS):
        response = client.chat.completions.create(
            model=model, messages=messages, max_tokens=10240, temperature=temperature
        )
        text = response.choices[0].message.content or ""
        stripped = THINK_RE.sub("", text)
        match = TOOL_CALL_RE.search(stripped)
        if match is None:
            final_text = stripped.strip()
            break
        try:
            data = json.loads(match.group(1).strip())
            name = str(data.get("name", ""))
            arguments = data.get("arguments") or {}
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            final_text = stripped.strip()[:400]
            break
        try:
            result = env.use_tool(name, **arguments)
            observation = str(result)
            is_error = False
        except Exception as exc:  # noqa: BLE001 - real env raises on bad calls; that IS the signal
            observation = f"Error: {type(exc).__name__}: {exc}"
            is_error = True
        steps.append({"tool": name, "arguments": arguments, "observation": observation[:2000], "is_error": is_error})
        messages.append({"role": "assistant", "content": f"<tool_call>{match.group(1).strip()}</tool_call>"})
        messages.append({"role": "user", "content": f"<tool_response>{observation[:2000]}</tool_response>"})
    return {"steps": steps, "final_text": final_text}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    pool = json.loads(Path(args.scenarios).read_text(encoding="utf-8"))
    scenarios = pool["scenarios"][: args.limit] if args.limit else pool["scenarios"]
    domains = trace_domains(_CORPUS)
    client = OpenAI(base_url=args.endpoint, api_key="not-needed")

    envs: dict[str, object] = {}
    tools: dict[str, list[str]] = {}
    records = []
    for scenario in scenarios:
        domain = domains.get(scenario["provenance"][0], "")
        if domain not in ("airline", "retail", "telecom"):
            continue
        for pass_index in range(args.passes):
            # Fresh environment per episode: real tool calls mutate the domain DB.
            env = registry.get_env_constructor(domain)()
            if domain not in tools:
                tools[domain] = [t.name for t in env.get_tools()]
            episode = run_episode(client, args.model, env, tools[domain], scenario, args.temperature)
            records.append(
                {
                    "label": args.label,
                    "scenario_id": scenario["scenario_id"],
                    "task": scenario["task"],
                    "checklist": scenario["checklist"],
                    "domain": domain,
                    "pass_index": pass_index,
                    **episode,
                }
            )
            print(f"{scenario['scenario_id']} pass {pass_index}: {len(episode['steps'])} steps", flush=True)

    Path(args.out).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(records)} real-env episodes -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
