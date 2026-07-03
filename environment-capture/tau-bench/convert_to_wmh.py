#!/usr/bin/env python3
"""Convert a real tau2-bench `results.json` into the wmh OTel-GenAI trace corpus.

This runs in the ISOLATED tau2 capture environment (Python 3.13, `tau2` installed). It does NOT
import `wmh` — it only emits the OTel-GenAI span JSONL shape that `wmh.ingest.otel_genai` reads, so
the world-model-harness package stays free of any tau2 dependency. Only the produced
`traces.otel.jsonl` is carried back into this example folder.

What it produces, per tau2 simulation (one solved task), faithful to the contract open-loop replay
needs:
  - one Step per AGENT TOOL CALL: action = the real tool call, observation = the REAL recorded tool
    result the agent saw (verbatim, with its error flag).
  - `Trace.metadata` carries the benchmark name, domain, task id, the task's GOLD evaluation
    criteria (expected actions + assertions), and the achieved reward. Gold rides along for the
    deferred closed-loop eval; open-loop replay ignores it.

`state_before` is intentionally left EMPTY for tau2. The environment's full DB (flight catalog, all
reservations, all users) is megabytes per step AND would leak the answer: handing the model the DB
that already contains `reservation NM1VX1` makes predicting `get_reservation_details(NM1VX1)` a
lookup, not a reconstruction. Open-loop replay reconstructs the env from the action + retrieved
similar past steps + the teacher-forced session history, which is the point. (The wmh adapter still
*reads* `wmh.state.*` when present, for future benchmarks whose state is small and non-leaky.)

Pure-conversational turns (no tool call) are not Steps — open-loop replay scores predicted
observations for `(state, action)`, and a chat turn has no environment observation to score.

Usage:
    TAU2_DATA_DIR=.../data python convert_to_wmh.py <results.json> --out traces.otel.jsonl --benchmark tau2-bench
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _tool_observation_by_id(messages: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map tool_call id -> the recorded ToolMessage (the real observation the agent saw)."""
    out: dict[str, dict[str, Any]] = {}
    for m in messages:
        if m.get("role") == "tool" and m.get("id"):
            out[str(m["id"])] = m
    return out


def _gold(task: dict[str, Any]) -> dict[str, Any]:
    """The task's evaluation criteria (expected actions + assertions) as plain JSON."""
    crit = task.get("evaluation_criteria")
    return crit if isinstance(crit, dict) else {}


def _as_text(value: Any) -> str:  # noqa: ANN401 - tau2 fields are loosely typed JSON
    """Render a loosely-typed value as a JSON-clean string: strings pass through, else JSON-encode.

    Used for both the task field and tool observations. tau2's `user_scenario.instructions` may be a
    plain string OR a structured dict (domain/reason_for_call/known_info/...), and a tool result's
    content is usually a string but is not guaranteed to be. Encoding non-strings with json.dumps
    keeps the trace JSON-clean end to end, so downstream can json.loads it (vs. a Python `repr()` with
    single quotes, which needs ast.literal_eval).
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _task_text(task: dict[str, Any]) -> str:
    """The agent-visible request: the user-scenario instructions, else the ticket."""
    scenario = task.get("user_scenario")
    if isinstance(scenario, dict) and scenario.get("instructions"):
        return _as_text(scenario["instructions"])
    if task.get("ticket"):
        return _as_text(task["ticket"])
    return ""


def _attr(key: str, value: str) -> dict[str, Any]:
    return {"key": key, "value": {"stringValue": value}}


def _spans_for_simulation(
    sim: dict[str, Any],
    task: dict[str, Any],
    *,
    benchmark: str,
    domain: str,
    trace_id: str,
) -> list[dict[str, Any]]:
    """Emit ordered action/observation span pairs for one simulation's agent tool calls."""
    messages = sim.get("messages", []) or []
    tool_obs = _tool_observation_by_id(messages)

    metadata = {
        "benchmark": benchmark,
        "domain": domain,
        "task_id": str(task.get("id", sim.get("task_id", ""))),
        "gold": _gold(task),
        "reward": (sim.get("reward_info") or {}).get("reward"),
    }
    task_text = _task_text(task)

    spans: list[dict[str, Any]] = []
    ordinal = 0
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            name = tc.get("name", "")
            args = tc.get("arguments") or {}

            # The authoritative observation is what the agent actually saw (recorded ToolMessage).
            recorded = tool_obs.get(str(tc.get("id")))
            obs_content = "" if recorded is None else _as_text(recorded.get("content", ""))
            obs_error = bool(recorded.get("error", False)) if recorded is not None else False

            action_attrs = [
                _attr("gen_ai.operation.name", "chat"),
                _attr("gen_ai.request.model", "tau2-agent"),
                _attr("gen_ai.tool.name", str(name)),
                _attr("gen_ai.tool.call.arguments", json.dumps(args)),
            ]
            if ordinal == 0 and task_text:
                action_attrs.append(_attr("gen_ai.prompt", task_text))
            if ordinal == 0:
                action_attrs.append(_attr("wmh.trace.metadata", json.dumps(metadata)))

            spans.append({
                "traceId": trace_id,
                "spanId": f"{trace_id[:12]}{ordinal:04x}a",
                "parentSpanId": "",
                "name": "chat tau2",
                "startTimeUnixNano": ordinal * 10,
                "endTimeUnixNano": ordinal * 10 + 1,
                "status": {"code": "STATUS_CODE_OK"},
                "attributes": action_attrs,
            })
            spans.append({
                "traceId": trace_id,
                "spanId": f"{trace_id[:12]}{ordinal:04x}b",
                "parentSpanId": "",
                "name": "execute_tool tau2",
                "startTimeUnixNano": ordinal * 10 + 2,
                "endTimeUnixNano": ordinal * 10 + 3,
                "status": {"code": "STATUS_CODE_ERROR" if obs_error else "STATUS_CODE_OK"},
                "attributes": [
                    _attr("gen_ai.operation.name", "execute_tool"),
                    _attr("gen_ai.tool.name", str(name)),
                    _attr("gen_ai.tool.message", obs_content),
                ],
            })
            ordinal += 1
    return spans


def _trace_id(benchmark: str, domain: str, sim_id: str) -> str:
    return hashlib.sha256(f"{benchmark}|{domain}|{sim_id}".encode()).hexdigest()[:32]


def _infer_domain(data: dict[str, Any]) -> str:
    """Pull the domain from the run info/config."""
    info = data.get("info", {})
    env_info = info.get("environment_info", {}) if isinstance(info, dict) else {}
    if isinstance(env_info, dict) and isinstance(env_info.get("domain_name"), str):
        return env_info["domain_name"]
    for key in ("domain", "domain_name"):
        if isinstance(info.get(key), str):
            return info[key]
    cfg = info.get("config", {}) if isinstance(info.get("config"), dict) else {}
    dom = cfg.get("domain")
    return dom if isinstance(dom, str) else "airline"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", help="Path to a tau2 results.json")
    parser.add_argument("--out", required=True, help="Output OTel JSONL path")
    parser.add_argument("--benchmark", default="tau2-bench", help="Benchmark name for metadata")
    parser.add_argument(
        "--only-rewarded",
        action="store_true",
        help="Skip simulations with reward < 1.0 (keep only fully-correct runs).",
    )
    args = parser.parse_args()

    data = json.loads(Path(args.results).read_text())
    sims = data.get("simulations", []) or []
    tasks_by_id = {str(t["id"]): t for t in data.get("tasks", []) if isinstance(t, dict)}
    domain = _infer_domain(data)

    n_spans = n_traces = 0
    with Path(args.out).open("w", encoding="utf-8") as f:
        for sim in sims:
            if sim.get("termination_reason") == "infrastructure_error":
                continue
            reward = (sim.get("reward_info") or {}).get("reward")
            if args.only_rewarded and reward != 1.0:
                continue
            task = tasks_by_id.get(str(sim.get("task_id", "")))
            if task is None:
                continue
            sim_id = str(sim.get("id", sim.get("task_id", "")))
            trace_id = _trace_id(args.benchmark, domain, sim_id)
            spans = _spans_for_simulation(
                sim, task, benchmark=args.benchmark, domain=domain, trace_id=trace_id
            )
            if not spans:
                continue
            for span in spans:
                f.write(json.dumps(span) + "\n")
                n_spans += 1
            n_traces += 1
    print(f"wrote {n_traces} traces, {n_spans} spans -> {args.out}")


if __name__ == "__main__":
    main()
