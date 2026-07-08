#!/usr/bin/env python3
"""Capture many terminal-tasks trajectories: an LLM agent runs bash in a container, live.

The environment being reconstructed is a Unix shell. This harness stands up ONE fresh
`debian:bookworm-slim` container (with `curl python3 jq git ca-certificates` installed), gives an
Opus agent a task, and loops: the model emits exactly one ```bash``` command, we `docker exec` it in
the container, and feed the REAL stdout/stderr + exit code back. The recorded (command -> real
output) pairs are the trace. This mirrors self-improvement-bench's `run_container_harness` (agents
issue bash inside a container, real output is recorded), reduced to a single-file, wmh-free tool.

Isolated exactly like the other capture tools: stdlib + boto3 only, never imports `wmh`. It writes a
trajectory JSONL in the shape `convert_to_wmh.py` already reads
(`{task, task_category, tool_calls:[{name, arguments:{command}, output, isError}]}`), so:

    AWS_REGION=us-east-1 packages/environment-capture/terminal-tasks/.venv/bin/python capture_terminal.py \
        --per-category 60 --out trajectories.jsonl
    # then, with the committed converter:
    python convert_to_wmh.py trajectories.jsonl --out traces.otel.jsonl --benchmark terminal-tasks

To beat per-model Bedrock throttling on a long run, tasks are sharded across several Opus models
(4.6/4.7/4.8) run concurrently — the same trick tau-bench's telecom capture used.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any

import boto3

# The five task families the committed terminal-tasks corpus covers. Each is a seed the task-writer
# model expands into concrete, self-contained shell tasks solvable with the installed tools.
CATEGORIES = [
    "Public API queries via curl (no auth)",
    "Package registry / version lookups (curl + jq)",
    "Filesystem + text processing",
    "GitHub via curl (public)",
    "Misc dev/research tasks",
]

# Opus inference-profile ids that invoke on this account; sharding across them dodges the per-model
# Bedrock quota (litellm/converse ServiceUnavailableError) on a sustained run.
DEFAULT_MODELS = [
    "us.anthropic.claude-opus-4-6-v1",
    "us.anthropic.claude-opus-4-7",
    "us.anthropic.claude-opus-4-8",
]

IMAGE = "debian:bookworm-slim"
SETUP = "apt-get update -qq && apt-get install -y -qq curl python3 jq git ca-certificates >/dev/null 2>&1"

_AGENT_SYSTEM = """You are a terminal agent solving a task in a Debian shell. Available: curl, \
python3, jq, git, standard coreutils. Network is available.

Each turn, respond with EXACTLY ONE bash command in a fenced block:
```bash
<your command>
```
You will see its real stdout/stderr and exit code. Work step by step. When the task is done, run a \
final command that prints the answer, then on the NEXT turn reply with exactly `DONE` (no fence)."""

_TASKGEN_SYSTEM = """You write concrete, self-contained terminal tasks for an agent with curl, \
python3, jq, git and coreutils in a throwaway Debian container with network. Each task must be \
solvable in 3-8 shell commands and have a checkable result. Reply with ONLY a JSON array of \
strings, each a task instruction. No prose."""

_FENCE_RE = re.compile(r"```(?:bash|sh)?\s*\n(.*?)```", re.DOTALL)


# Capacity/transient error codes worth retrying — same set as environment_capture.agent. A bad
# request or auth error must RAISE immediately: retrying it 6x with backoff only hides the bug.
_THROTTLE_CODES = {
    "ThrottlingException",
    "TooManyRequestsException",
    "ServiceUnavailableException",
    "ModelTimeoutException",
    "InternalServerException",
}


def _retriable(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    code = response.get("Error", {}).get("Code") if isinstance(response, dict) else None
    if code is not None:
        return code in _THROTTLE_CODES
    transient = {"ReadTimeoutError", "ConnectTimeoutError", "EndpointConnectionError"}
    return type(exc).__name__ in transient


def _converse(client: Any, model: str, system: str, messages: list[dict[str, str]], max_tokens: int) -> str:  # noqa: ANN401, E501
    """One Bedrock converse call -> assistant text (retries throttling/transient errors ONLY)."""
    conv = [{"role": m["role"], "content": [{"text": m["content"]}]} for m in messages]
    last_err: Exception | None = None
    for attempt in range(6):
        try:
            r = client.converse(
                modelId=model,
                system=[{"text": system}],
                messages=conv,
                inferenceConfig={"maxTokens": max_tokens},
            )
            return r["output"]["message"]["content"][0]["text"]
        except Exception as e:  # noqa: BLE001 - classified just below
            if not _retriable(e):
                raise
            last_err = e
            _sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"converse failed after retries: {last_err}")


def _sleep(seconds: float) -> None:
    import time

    time.sleep(seconds)


def _gen_tasks(client: Any, model: str, category: str, n: int) -> list[str]:  # noqa: ANN401
    """Ask the model for `n` concrete tasks in `category`; parse the JSON array, best-effort."""
    text = _converse(
        client,
        model,
        _TASKGEN_SYSTEM,
        [{"role": "user", "content": f"Write {n} tasks in the category: {category}"}],
        max_tokens=4096,
    )
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        tasks = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [t for t in tasks if isinstance(t, str)][:n]


def _docker(args: list[str], *, timeout: int = 120) -> tuple[str, int]:
    """Run a docker CLI command, return (combined output, returncode)."""
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, errors="replace"
        )
        return (proc.stdout + proc.stderr), proc.returncode
    except subprocess.TimeoutExpired:
        return "(timed out)", 124


def _start_container() -> str:
    """Start a fresh tools container; return its id. Caller must `docker rm -f` it."""
    name = f"wmh-term-{uuid.uuid4().hex[:12]}"
    out, rc = _docker(
        ["docker", "run", "-d", "--name", name, IMAGE, "sleep", "3600"], timeout=120
    )
    if rc != 0:
        raise RuntimeError(f"docker run failed: {out}")
    cid = out.strip()
    _docker(["docker", "exec", cid, "sh", "-c", SETUP], timeout=300)  # best-effort tool install
    return cid


def _exec(cid: str, command: str, *, workdir: str = "/root", timeout: int = 60) -> tuple[str, int]:
    return _docker(
        ["docker", "exec", "-w", workdir, cid, "sh", "-c", command], timeout=timeout
    )


def _run_task(client: Any, model: str, task: str, category: str, *, step_limit: int) -> dict[str, Any]:  # noqa: ANN401, E501
    """Run one agent-in-container episode; return a trajectory dict in the converter's schema."""
    cid = _start_container()
    tool_calls: list[dict[str, Any]] = []
    last_rc = 0
    try:
        messages = [{"role": "user", "content": task}]
        for _ in range(step_limit):
            reply = _converse(client, model, _AGENT_SYSTEM, messages, max_tokens=2048)
            messages.append({"role": "assistant", "content": reply})
            if reply.strip() == "DONE":
                break
            m = _FENCE_RE.search(reply)
            if not m:
                messages.append({"role": "user", "content": "Reply with ONE ```bash``` block or DONE."})
                continue
            command = m.group(1).strip()
            output, rc = _exec(cid, command)
            last_rc = rc
            tool_calls.append({
                "name": "bash",
                "arguments": {"command": command},
                "output": output,
                "isError": rc != 0,
            })
            messages.append({
                "role": "user",
                "content": f"<returncode>{rc}</returncode>\n<output>\n{output}\n</output>",
            })
    finally:
        _docker(["docker", "rm", "-f", cid], timeout=60)

    return {
        "task": task,
        "task_category": category,
        "returncode": last_rc,
        "tool_calls": tool_calls,
    }


def _shard(client: Any, model: str, tasks: list[tuple[str, str]], step_limit: int, out: list, lock) -> None:  # noqa: ANN401, E501
    """Run a shard of (task, category) pairs on one model; append trajectories under `lock`."""
    for task, category in tasks:
        try:
            traj = _run_task(client, model, task, category, step_limit=step_limit)
        except Exception as e:  # noqa: BLE001 - one bad episode shouldn't kill the shard
            print(f"[{model}] task failed: {e}", flush=True)
            continue
        if traj["tool_calls"]:  # a trajectory with no commands has nothing to score
            with lock:
                out.append(traj)
                print(f"[{model}] {len(out)} trajectories ({len(traj['tool_calls'])} steps)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-category", type=int, default=60, help="Tasks per category.")
    parser.add_argument("--step-limit", type=int, default=10, help="Max agent steps per task.")
    parser.add_argument("--region", default="us-east-1", help="AWS region (Bedrock).")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="Comma-separated model ids.")
    parser.add_argument("--out", default="trajectories.jsonl", help="Output trajectory JSONL.")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    client = boto3.client("bedrock-runtime", region_name=args.region)

    # Generate the task list up front (cheap), then shard the (task, category) pairs across models.
    all_tasks: list[tuple[str, str]] = []
    for category in CATEGORIES:
        tasks = _gen_tasks(client, models[0], category, args.per_category)
        all_tasks.extend((t, category) for t in tasks)
        print(f"generated {len(tasks)} tasks for {category!r}", flush=True)
    print(f"total tasks: {len(all_tasks)}, sharding across {len(models)} models\n", flush=True)

    shards: list[list[tuple[str, str]]] = [[] for _ in models]
    for i, item in enumerate(all_tasks):
        shards[i % len(models)].append(item)

    out: list[dict[str, Any]] = []
    lock = threading.Lock()
    threads = [
        threading.Thread(
            target=_shard,
            args=(boto3.client("bedrock-runtime", region_name=args.region), model, shard,
                  args.step_limit, out, lock),
        )
        for model, shard in zip(models, shards, strict=True)
        if shard
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    with Path(args.out).open("w", encoding="utf-8") as f:
        for traj in out:
            f.write(json.dumps(traj) + "\n")
    print(f"\nwrote {len(out)} trajectories -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
