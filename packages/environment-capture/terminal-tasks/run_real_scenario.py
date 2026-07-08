#!/usr/bin/env python3
"""Run one real terminal-tasks scenario in a freshly-built container.

This builds a fresh container from scratch first (a base image + the real `apt-get install` of the
tools these tasks need: `curl`, `python3`, `jq`, `ca-certificates`), streams the build, counts it in
the total time, and then `docker exec`s the recorded `bash` commands.

Because terminal-tasks commands hit live public APIs, a real re-run reflects *current* data, so the
output may differ from the recorded observation (rates change, releases bump) — that is the honest
real environment.

Stdlib-only and self-contained (no `wmh` import). It reads this example's `traces.otel.jsonl`, picks
a held-out trace using the harness's deterministic blake2b split, and runs that trace's recorded
commands inside the built container.

Usage:
    python run_real_scenario.py --trace 1            # cold --no-cache build (default)
    python run_real_scenario.py --trace 1 --cache    # reuse the cached tools image
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

_DEFAULT_CORPUS = Path(__file__).resolve().parent / "traces.otel.jsonl"

# The recorded terminal-tasks commands use curl (-> public APIs), python3 (JSON parsing), and jq.
# A fresh container installs exactly those — the real tool standup the world model never pays.
_IMAGE_TAG = "wmh-terminal-tasks:latest"
_DOCKERFILE = """\
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \\
        curl ca-certificates python3 jq \\
    && rm -rf /var/lib/apt/lists/*
WORKDIR /work
"""


def _attr_map(span: dict[str, Any]) -> dict[str, str]:
    return {a["key"]: a.get("value", {}).get("stringValue", "") for a in span.get("attributes", [])}


def _load_traces(corpus: Path) -> "list[dict[str, Any]]":
    """Group the OTel spans into ordered traces: [{trace_id, task, commands:[...]}]."""
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
        task = ""
        commands: list[str] = []
        for span in by_trace[tid]:
            attrs = _attr_map(span)
            task = task or attrs.get("gen_ai.prompt", "")
            args = attrs.get("gen_ai.tool.call.arguments")
            if args:  # an action span (the observation span has no arguments)
                command = json.loads(args).get("command")
                if isinstance(command, str) and command.strip():
                    commands.append(command)
        traces.append({"trace_id": tid, "task": task, "commands": commands})
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


def _exists(image: str) -> bool:
    return subprocess.run(
        ["docker", "image", "inspect", image], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0


def _build_tools_image(*, no_cache: bool) -> None:
    """`docker build` the fresh tools image (curl/python3/jq), streaming the apt install live."""
    with tempfile.TemporaryDirectory(prefix="wmh-tt-build-") as ctx:
        (Path(ctx) / "Dockerfile").write_text(_DOCKERFILE, encoding="utf-8")
        cmd = ["docker", "build", "-t", _IMAGE_TAG]
        if no_cache:
            cmd.append("--no-cache")
        cmd.append(ctx)
        print(f"$ {' '.join(cmd)}")
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            raise SystemExit(f"docker build failed (exit {rc})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=str(_DEFAULT_CORPUS), help="terminal-tasks OTel corpus.")
    parser.add_argument(
        "--trace", type=int, default=None,
        help="Held-out trace to replay (default: the simplest = fewest commands).",
    )
    parser.add_argument("--train-split", type=float, default=0.7, help="Train/holdout ratio.")
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Reuse the cached tools image if present. Default: cold --no-cache build.",
    )
    parser.add_argument("--exec-timeout", type=int, default=120, help="Per-command timeout (s).")
    args = parser.parse_args()

    traces = _load_traces(Path(args.corpus))
    pool = _holdout(traces, args.train_split)
    if not pool:
        raise SystemExit(f"no traces in {args.corpus}; nothing to run")
    if args.trace is None:
        # Default: the simplest scenario — fewest recorded commands (matches the wmh side's default).
        trace = min(pool, key=lambda t: len(t["commands"]))
    elif 0 <= args.trace < len(pool):
        trace = pool[args.trace]
    else:
        raise SystemExit(f"--trace {args.trace} out of range; {len(pool)} held-out trace(s)")
    commands = trace["commands"]
    task = (trace["task"] or "").strip().splitlines()[0] if trace["task"] else ""
    no_cache = not args.cache
    print(
        f"REAL sandbox: trace {trace['trace_id'][:8]} ({len(commands)} commands)"
        + (f" — {task[:70]}" if task else "")
        + " — building a fresh container (apt install curl/python3/jq), then running the commands"
        + (" [--no-cache]\n" if no_cache else " [cached]\n")
    )

    start = time.monotonic()
    if args.cache and _exists(_IMAGE_TAG):
        print(f"--- tools image {_IMAGE_TAG} already built (cached) ---\n")
    else:
        print("--- building the tools image ---")
        _build_tools_image(no_cache=no_cache)
        print()
    build_done = time.monotonic()
    print(f"[container built from scratch in {build_done - start:.1f}s]\n")

    container = f"wmh-real-{uuid.uuid4().hex[:8]}"
    rc = subprocess.run(
        ["docker", "run", "-d", "--name", container, "-w", "/work", "--rm", _IMAGE_TAG,
         "sleep", "2h"],
        stdout=subprocess.DEVNULL,
    ).returncode
    if rc != 0:
        raise SystemExit(f"failed to start container (docker run exit {rc})")
    try:
        for i, command in enumerate(commands):
            print(f"--- step {i} ---\n$ {command}")
            try:
                subprocess.run(
                    ["docker", "exec", "-w", "/work", container, "bash", "-lc", command],
                    timeout=args.exec_timeout,
                )
            except subprocess.TimeoutExpired:
                print(f"[timed out after {args.exec_timeout}s]")
            print()
    finally:
        subprocess.run(["docker", "rm", "-f", container], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)

    total = time.monotonic() - start
    print(
        f"done (REAL sandbox): build {build_done - start:.1f}s + "
        f"{len(commands)} commands, {total:.1f}s total"
    )


if __name__ == "__main__":
    main()
