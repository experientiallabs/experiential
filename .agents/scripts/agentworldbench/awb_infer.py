"""AgentWorldBench `infer` stage backed by a wmh world model.

Replaces `eval.py infer` from github.com/QwenLM/Qwen-AgentWorld. Reads the benchmark's
per-domain `*_test.jsonl` rows, feeds each row's FULL interaction history to a wmh world
model, and writes `predictions.jsonl` with the `gen` field their `judge`/`score` stages
consume unchanged.

Protocol note (verified against their repo @354f733): their shipped `infer` sends only
`system_str` + `current_prompt` — NO prior turns — while `build_judge_messages` scores the
prediction against the full history context. We follow the paper protocol (the world model
receives the interaction history): turns `1..turn_idx-1` are seeded as teacher-forced
history, exactly what the judge sees.

Two modes:
- `wm`: a built wmh model (optimized prompt + RAG index), via the serving session path.
- `base`: BASE_ENV_PROMPT + no retrieval, any provider — for domains without a wmh corpus
  and for the RAG-vs-base ablation.

Usage (from the repo root):
    uv run python .agents/scripts/agentworldbench/awb_infer.py \
        --data .wmh/agentworldbench/data/terminal_test.jsonl --limit 3 \
        --mode wm --model-dir packages/environment-capture/terminal-tasks/models/terminal-tasks \
        --output .wmh/agentworldbench/results/terminal_wm/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from wmh.core.types import Action, ActionKind, EnvState, JsonObject, Observation, Step
from wmh.engine.loader import load_world_model
from wmh.engine.prompts import BASE_ENV_PROMPT
from wmh.optimize.gepa import predict_observation
from wmh.providers import get_provider
from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.tracking.metered import MeteredProvider
from wmh.tracking.tracker import Phase, RunTracker


def history_steps(row: JsonObject) -> list[Step]:
    """Turns 1..turn_idx-1 as teacher-forced Steps (prompt/response are parallel lists)."""
    prompts, responses = row["prompt"], row["response"]
    n_history = min(int(row["turn_idx"]) - 1, len(prompts), len(responses))
    return [
        Step(
            action=Action(kind=ActionKind.MESSAGE, content=prompts[i]),
            observation=Observation(content=responses[i]),
        )
        for i in range(n_history)
    ]


def current_action(row: JsonObject) -> Action:
    content = row.get("current_prompt") or row["prompt"][int(row["turn_idx"]) - 1]
    return Action(kind=ActionKind.MESSAGE, content=content)


def wrap_gen(observation: Observation) -> str:
    """Their output_parser extracts the LAST <predicted_observation> block (tag optional but safest)."""
    return f"<predicted_observation>\n{observation.content}\n</predicted_observation>"


def infer_wm(rows: list[JsonObject], model_dir: str) -> list[JsonObject]:
    """Predict via a built model's serving path: seeded session -> one step (retrieval included)."""
    world_model, provider = load_world_model(model_dir)
    for i, row in enumerate(rows):
        row["gen"] = ""
        session = None
        try:
            session = world_model.new_session(task=None)
            world_model.seed_session(session.id, history_steps(row))
            observation = world_model.step(session.id, current_action(row))
            row["gen"] = wrap_gen(observation)
        except Exception:  # per-row isolation; their judge marks gen == "" as failed
            print(f"[{i + 1}/{len(rows)}] infer failed for id={row.get('id')}", file=sys.stderr)
            traceback.print_exc()
        finally:
            usage = world_model.end_session(session.id) if session is not None else None
        row["wmh_infer"] = {
            "mode": "wm",
            "model_dir": model_dir,
            "serve_model": provider.config.model,
            "input_tokens": usage.total.input_tokens if usage else 0,
            "output_tokens": usage.total.output_tokens if usage else 0,
            "cost_usd": usage.total.cost_usd if usage else 0.0,
        }
        print(f"[{i + 1}/{len(rows)}] id={row.get('id')} turn={row.get('turn_idx')} ok")
    return rows


def infer_base(rows: list[JsonObject], provider_model: str, region: str) -> list[JsonObject]:
    """Predict with BASE_ENV_PROMPT and no retrieval (base-prompt-only rows / ablation arm)."""
    tracker = RunTracker(run_id="awb-infer-base", kind="eval")
    tracker.start()
    provider = MeteredProvider(
        get_provider(
            ProviderConfig(kind=ProviderKind.BEDROCK, model=provider_model, region=region)
        ),
        tracker,
        base_phase=Phase.SERVE,
    )
    for i, row in enumerate(rows):
        before = tracker.record_summary().total
        try:
            observation = predict_observation(
                provider,
                BASE_ENV_PROMPT,
                None,
                EnvState(),
                current_action(row),
                demos=[],
                history=history_steps(row),
            )
            row["gen"] = wrap_gen(observation)
        except Exception:
            row["gen"] = ""
            print(f"[{i + 1}/{len(rows)}] infer failed for id={row.get('id')}", file=sys.stderr)
            traceback.print_exc()
        after = tracker.record_summary().total
        row["wmh_infer"] = {
            "mode": "base",
            "serve_model": provider_model,
            "input_tokens": after.input_tokens - before.input_tokens,
            "output_tokens": after.output_tokens - before.output_tokens,
            "cost_usd": (after.cost_usd or 0.0) - (before.cost_usd or 0.0),
        }
        print(f"[{i + 1}/{len(rows)}] id={row.get('id')} turn={row.get('turn_idx')} ok")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="one AgentWorldBench {domain}_test.jsonl")
    parser.add_argument("--mode", choices=["wm", "base"], required=True)
    parser.add_argument("--model-dir", help="built wmh model dir (required for --mode wm)")
    parser.add_argument("--provider-model", default="us.anthropic.claude-opus-4-8")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--limit", type=int, default=None, help="first N rows only")
    parser.add_argument("--output", required=True, help="predictions.jsonl path")
    args = parser.parse_args()
    if args.mode == "wm" and not args.model_dir:
        parser.error("--mode wm requires --model-dir")

    rows: list[JsonObject] = [
        json.loads(line)
        for line in Path(args.data).read_text(encoding="utf-8").splitlines()
        if line
    ]
    if args.limit is not None:
        rows = rows[: args.limit]
    print(f"{len(rows)} rows from {args.data} (mode={args.mode})")

    if args.mode == "wm":
        rows = infer_wm(rows, args.model_dir)
    else:
        rows = infer_base(rows, args.provider_model, args.region)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    total_cost = sum(r["wmh_infer"].get("cost_usd") or 0.0 for r in rows)
    n_ok = sum(1 for r in rows if r["gen"])
    print(f"wrote {out} — {n_ok}/{len(rows)} predictions, infer cost ${total_cost:.4f}")


if __name__ == "__main__":
    main()
