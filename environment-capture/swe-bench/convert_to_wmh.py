#!/usr/bin/env python3
"""Convert real SWE-bench (mini-swe-agent) trajectories into the wmh OTel-GenAI trace corpus.

The source is real SWE-bench Verified runs driven by the standard
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) harness: per instance, the agent runs
shell commands inside the instance's Docker container and the REAL command output is recorded
(stdout/stderr + a `<returncode>`), including tracebacks, build errors, and pytest logs. That maps
directly to the harness contract: one Step per shell command, with the real `(action) -> observation`
the agent actually saw. The environment being reconstructed is a Unix shell inside a buggy repo:
predict the command's real output given the command.

This is a stdlib-only converter (no `wmh` import, no third-party deps) so it stays a self-contained
capture tool. It reads the SOURCE trajectories in place and never copies them into the repo — only
the produced OTel JSONL is written to ``--out``.

Source schema — mini-swe-agent `<instance_id>.traj.json` (trajectory_format "mini-swe-agent-1.x"):
either a bare list of messages, or a dict ``{"messages": [...], "info": {...}, "instance_id": ...}``.
Each message is ``{"role": "system"|"user"|"assistant", "content": str, "extra"?: {...}}``. An
assistant turn carries one bash command in a fenced ```` ```...bash...``` ```` block (the agent's
action); the following user turn is the environment's reply, wrapped as
``<returncode>N</returncode>\n<output>...</output>``.

Per trajectory, per assistant command + its following observation:
  - action      = bash {"command": "<the fenced command>"}.
  - observation = the recorded ``<output>`` text, ``is_error`` from a non-zero ``<returncode>``.
  - task        = the instance problem statement (first user turn / info), on the first step.
  - Trace.metadata = benchmark, instance_id, repo, gold model_patch + exit_status when present.

``state_before`` is left empty: the env state is a whole repo working tree (huge, and would leak the
answer); open-loop replay reconstructs from the action + retrieved steps + teacher-forced history.

Usage:
    python convert_to_wmh.py <run_dir_or_traj.json> --out traces.otel.jsonl --benchmark swe-bench
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

# A fenced command block: ```<lang>\n<command>\n```  (mini-swe-agent uses langs like
# `mswea_bash_command` / `bash`). We take the first fenced block in the assistant message.
_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
# The environment reply wraps real stdout/stderr; capture the returncode and the output body.
_RETURNCODE_RE = re.compile(r"<returncode>\s*(-?\d+)\s*</returncode>", re.DOTALL)
_OUTPUT_RE = re.compile(r"<output>\n?(.*?)\n?</output>", re.DOTALL)


def _attr(key: str, value: str) -> dict[str, Any]:
    return {"key": key, "value": {"stringValue": value}}


def _messages_of(traj: Any) -> list[dict[str, Any]]:  # noqa: ANN401 - loosely-typed source JSON
    """A trajectory file is either a bare message list or a dict carrying ``messages``."""
    if isinstance(traj, list):
        return [m for m in traj if isinstance(m, dict)]
    if isinstance(traj, dict):
        msgs = traj.get("messages")
        if isinstance(msgs, list):
            return [m for m in msgs if isinstance(m, dict)]
    return []


def _info_of(traj: Any) -> dict[str, Any]:  # noqa: ANN401 - loosely-typed source JSON
    return traj.get("info", {}) if isinstance(traj, dict) else {}


def _command_of(message: dict[str, Any]) -> str | None:
    """The shell command an assistant turn ran, or None for a reasoning-only turn.

    Handles both mini-swe-agent model formats:
    - tool-call models (what Bedrock Opus 4.8 produces): the command is in
      ``extra.actions[0].command``.
    - text-based models: the command is the first fenced ```` ``` ```` block in ``content``.
    """
    extra = message.get("extra")
    if isinstance(extra, dict):
        actions = extra.get("actions")
        if isinstance(actions, list) and actions:
            first = actions[0]
            if isinstance(first, dict):
                command = first.get("command")
                if isinstance(command, str) and command.strip():
                    return command.strip()
    content = message.get("content")
    if isinstance(content, str):
        match = _FENCE_RE.search(content)
        if match is not None and match.group(1).strip():
            return match.group(1).strip()
    return None


def _observation_of(message: dict[str, Any]) -> tuple[str, bool]:
    """The real output + error flag from an environment reply (a ``tool`` or ``user`` message).

    Prefers the structured ``extra.returncode`` + the ``<output>`` body; falls back to the raw
    content when the reply isn't wrapped (older/text formats), so no real observation is dropped.
    """
    content = message.get("content")
    text = content if isinstance(content, str) else ""
    out_match = _OUTPUT_RE.search(text)
    body = out_match.group(1) if out_match is not None else text

    extra = message.get("extra")
    if isinstance(extra, dict) and extra.get("returncode") is not None:
        # Structured returncode from the tool-call format is authoritative. Coerce via int() so a
        # JSON returncode that arrives as a string ("0") still compares correctly (str "0" != int 0).
        return body, _nonzero(extra["returncode"])
    rc_match = _RETURNCODE_RE.search(text)
    is_error = bool(rc_match) and rc_match.group(1) != "0"
    return body, is_error


def _nonzero(returncode: Any) -> bool:  # noqa: ANN401 - loosely-typed source JSON
    """True if `returncode` is a non-zero exit code, tolerant of int or numeric-string encodings."""
    try:
        return int(returncode) != 0
    except (TypeError, ValueError):
        # Unparseable returncode: treat a truthy non-empty value as an error, else not.
        return bool(returncode)


def _task_of(messages: list[dict[str, Any]], info: dict[str, Any]) -> str:
    """The instance problem statement: explicit info field, else the first user turn."""
    for key in ("problem_statement", "task", "prompt"):
        value = info.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for m in messages:
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            return m["content"]
    return ""


def _metadata(info: dict[str, Any], instance_id: str, benchmark: str) -> dict[str, Any]:
    meta: dict[str, Any] = {"benchmark": benchmark, "instance_id": instance_id}
    for key in ("repo", "submission", "model_patch", "exit_status"):
        if key in info:
            meta[key] = info[key]
    return meta


def _spans_for_trajectory(
    traj: Any,  # noqa: ANN401 - loosely-typed source JSON
    *,
    benchmark: str,
    instance_id: str,
) -> list[dict[str, Any]]:
    """Emit ordered action/observation span pairs for one instance's command turns.

    Walk the messages; for each assistant turn that ran a command, pair it with the NEXT message
    (the environment reply) as the observation. The trace_id is the instance id, so a benchmark run
    maps one trace per instance.
    """
    messages = _messages_of(traj)
    info = _info_of(traj)
    task_text = _task_of(messages, info)
    metadata = _metadata(info, instance_id, benchmark)
    trace_id = _stable_trace_id(instance_id)

    spans: list[dict[str, Any]] = []
    ordinal = 0
    for i, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        command = _command_of(message)
        if command is None:
            continue  # reasoning-only turn: no environment observation to score
        nxt = messages[i + 1] if i + 1 < len(messages) else None
        if nxt is None or nxt.get("role") not in ("tool", "user"):
            continue  # no recorded env reply followed this command; skip (never invent one)
        obs_content, obs_error = _observation_of(nxt)

        action_attrs = [
            _attr("gen_ai.operation.name", "chat"),
            _attr("gen_ai.request.model", "swe-bench-agent"),
            _attr("gen_ai.tool.name", "bash"),
            _attr("gen_ai.tool.call.arguments", json.dumps({"command": command})),
        ]
        if ordinal == 0 and task_text:
            action_attrs.append(_attr("gen_ai.prompt", task_text))
        if ordinal == 0:
            action_attrs.append(_attr("wmh.trace.metadata", json.dumps(metadata)))

        spans.append({
            "traceId": trace_id,
            "spanId": f"{trace_id[:12]}{ordinal:04x}a",
            "parentSpanId": "",
            "name": "chat swe-bench",
            "startTimeUnixNano": ordinal * 10,
            "endTimeUnixNano": ordinal * 10 + 1,
            "status": {"code": "STATUS_CODE_OK"},
            "attributes": action_attrs,
        })
        spans.append({
            "traceId": trace_id,
            "spanId": f"{trace_id[:12]}{ordinal:04x}b",
            "parentSpanId": "",
            "name": "execute_tool swe-bench",
            "startTimeUnixNano": ordinal * 10 + 2,
            "endTimeUnixNano": ordinal * 10 + 3,
            "status": {"code": "STATUS_CODE_ERROR" if obs_error else "STATUS_CODE_OK"},
            "attributes": [
                _attr("gen_ai.operation.name", "execute_tool"),
                _attr("gen_ai.tool.name", "bash"),
                _attr("gen_ai.tool.message", obs_content),
            ],
        })
        ordinal += 1
    return spans


def _stable_trace_id(instance_id: str) -> str:
    """A 32-hex-char trace id derived from the instance id (stable across reruns)."""
    import hashlib

    return hashlib.sha256(instance_id.encode()).hexdigest()[:32]


def _iter_traj_files(source: Path) -> list[Path]:
    """Every `*.traj.json` under `source` (or `source` itself if it is one)."""
    if source.is_file():
        return [source]
    return sorted(source.rglob("*.traj.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="A run dir (searched for *.traj.json) or a single traj.json")
    parser.add_argument("--out", required=True, help="Output OTel JSONL path")
    parser.add_argument("--benchmark", default="swe-bench", help="Benchmark name")
    parser.add_argument(
        "--min-steps",
        type=int,
        default=1,
        help="Skip trajectories with fewer than this many command steps (default 1).",
    )
    args = parser.parse_args()

    source = Path(args.source)
    traj_files = _iter_traj_files(source)
    n_traces = n_spans = n_skipped = 0
    with Path(args.out).open("w", encoding="utf-8") as out:
        for traj_file in traj_files:
            try:
                traj = json.loads(traj_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                # One corrupt/partial trajectory must not abort the whole batch (and leave a
                # truncated --out): skip it, count it, and report at the end.
                print(f"skipping unreadable trajectory {traj_file}: {exc}")
                n_skipped += 1
                continue
            # instance id: the dict field, else the filename stem (strip the .traj suffix).
            instance_id = (
                traj.get("instance_id")
                if isinstance(traj, dict) and traj.get("instance_id")
                else traj_file.name.removesuffix(".traj.json")
            )
            spans = _spans_for_trajectory(traj, benchmark=args.benchmark, instance_id=instance_id)
            if len(spans) < args.min_steps * 2:  # two spans (action + observation) per step
                n_skipped += 1
                continue
            for span in spans:
                out.write(json.dumps(span) + "\n")
                n_spans += 1
            n_traces += 1
    print(
        f"wrote {n_traces} traces, {n_spans} spans -> {args.out} "
        f"(from {len(traj_files)} traj files, skipped {n_skipped})"
    )


if __name__ == "__main__":
    main()
