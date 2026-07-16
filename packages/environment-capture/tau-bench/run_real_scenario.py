#!/usr/bin/env python3
"""Run one real tau2-bench scenario against the real tau2 environment.

This constructs Sierra's real tau2 domain environment (the airline/retail Python tools over the
real JSON database) and calls the recorded tool calls in order, printing the real tool results and
wall-clock time.

Unlike swe-bench/terminal-tasks, tau2 actions are TOOL CALLS (e.g. `get_user_details(user_id=...)`),
not shell commands, so this imports the real tau2 package and uses `Environment.use_tool`. It must
therefore run in the tau2 `.venv` set up by this directory's README — NOT under `wmh` (which never
imports tau2). There is no container/process to boot; the "startup cost" the world model saves is
loading the real domain DB into the environment.

Stdlib + tau2 only (no `wmh` import). It reads this example's `traces.otel.jsonl`, picks a held-out
trace using the harness's deterministic blake2b split, reads the `domain` from trace metadata, and
replays the trace's recorded `(tool_name, arguments)` calls.

Usage (from this directory, in the tau2 venv):
    TAU2_DATA_DIR="$PWD/tau2-bench/data" .venv/bin/python run_real_scenario.py --trace 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

_DEFAULT_CORPUS = Path(__file__).resolve().parent / "traces.otel.jsonl"


def _attr_map(span: dict[str, Any]) -> dict[str, str]:
    return {a["key"]: a.get("value", {}).get("stringValue", "") for a in span.get("attributes", [])}


def _load_traces(corpus: Path) -> "list[dict[str, Any]]":
    """Group the OTel spans into ordered traces: [{trace_id, domain, calls:[(name, args)]}]."""
    spans = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines() if line]
    order: list[str] = []
    by_trace: dict[str, list[dict[str, Any]]] = {}
    for span in spans:
        tid = span["traceId"]
        if tid not in by_trace:
            by_trace[tid] = []
            order.append(tid)
        by_trace[tid].append(span)

    traces: list[dict[str, Any]] = []
    for tid in order:
        domain = ""
        calls: list[tuple[str, dict[str, Any]]] = []
        for span in by_trace[tid]:
            attrs = _attr_map(span)
            if "wmh.trace.metadata" in attrs:
                domain = json.loads(attrs["wmh.trace.metadata"]).get("domain", "")
            args = attrs.get("gen_ai.tool.call.arguments")
            name = attrs.get("gen_ai.tool.name")
            if args and name:  # an action span (the observation span has no arguments)
                calls.append((name, json.loads(args)))
        traces.append({"trace_id": tid, "domain": domain, "calls": calls})
    return traces


def _holdout(traces: list[dict[str, Any]], train_split: float) -> list[dict[str, Any]]:
    """The held-out traces, by the SAME deterministic blake2b split the wmh harness uses."""
    held: list[dict[str, Any]] = []
    for trace in traces:
        digest = hashlib.blake2b(trace["trace_id"].encode("utf-8"), digest_size=8).digest()
        fraction = int.from_bytes(digest, "big") / 2**64
        if fraction >= train_split:
            held.append(trace)
    return held or traces  # tiny corpora: no held-out -> fall back to all (matches the wmh side)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=str(_DEFAULT_CORPUS), help="tau2-bench OTel corpus.")
    parser.add_argument(
        "--trace", type=int, default=None,
        help="Held-out trace to replay by index (default: the simplest = fewest tool calls).",
    )
    parser.add_argument(
        "--trace-id", default=None,
        help="Replay the trace with this exact trace_id (overrides --trace). Use this to pin the "
        "SAME scenario the world-model side ran, since index order differs between the two sides.",
    )
    parser.add_argument("--train-split", type=float, default=0.7, help="Train/holdout ratio.")
    args = parser.parse_args()

    traces = _load_traces(Path(args.corpus))
    pool = _holdout(traces, args.train_split)
    if not pool:
        raise SystemExit(f"no traces in {args.corpus}; nothing to run")
    if args.trace_id is not None:
        # Pin by stable trace_id so this side replays EXACTLY the world side's scenario, regardless
        # of how the two loaders order the corpus.
        by_id = {t["trace_id"]: t for t in traces}
        if args.trace_id not in by_id:
            raise SystemExit(f"--trace-id {args.trace_id} not found in {args.corpus}")
        trace = by_id[args.trace_id]
    elif args.trace is None:
        # Default: the simplest scenario — fewest recorded tool calls (matches the wmh side).
        trace = min(pool, key=lambda t: len(t["calls"]))
    elif 0 <= args.trace < len(pool):
        trace = pool[args.trace]
    else:
        raise SystemExit(f"--trace {args.trace} out of range; {len(pool)} held-out trace(s)")
    domain, calls = trace["domain"] or "airline", trace["calls"]

    print(
        f"REAL tau2 env: domain={domain}, trace {trace['trace_id'][:8]} "
        f"({len(calls)} tool calls) — standing up the real environment "
        "(import tau2 -> registry -> load domain DB), then replaying the recorded tool calls\n"
    )

    # Standup = the real cost of bringing Sierra's environment up in-process: importing the heavy
    # tau2 package + registering its components + loading the domain DB. (The one-time
    # `pip install tau2-bench` is the venv Setup in the README; this is the per-run standup.)
    start = time.monotonic()
    try:
        from tau2.registry import registry
    except ImportError as exc:  # pragma: no cover - depends on the isolated venv
        raise SystemExit(
            "tau2 is not importable; run this from packages/environment-capture/tau-bench/ in the tau2 .venv "
            "(see this directory's README), with TAU2_DATA_DIR set."
        ) from exc
    env = registry.get_env_constructor(domain)()  # loads the real domain DB
    env_ready = time.monotonic()
    print(f"[environment stood up (import + registry + DB) in {env_ready - start:.2f}s]\n")

    for i, (name, kwargs) in enumerate(calls):
        print(f"--- step {i} ---\n> {name}({json.dumps(kwargs)})")
        try:
            result = env.use_tool(name, **kwargs)
            print(result)
        except Exception as exc:  # noqa: BLE001 - surface the REAL tool error, like the agent saw
            print(f"[tool error] {type(exc).__name__}: {exc}")
        print()

    total = time.monotonic() - start
    print(
        f"done (REAL tau2 env): standup {env_ready - start:.2f}s + "
        f"{len(calls)} tool calls, {total:.2f}s total"
    )


if __name__ == "__main__":
    main()
