#!/usr/bin/env python3
"""Replay admitted teacher bash actions and run the real task verifier."""

# ruff: noqa: ANN401

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BashAction:
    """One exact bash action extracted from a source trajectory."""

    assistant_message_index: int
    tool_call_index: int
    tool_call_id: str
    command: str
    recorded_output: str


@dataclass(frozen=True)
class AdmittedTrace:
    """One admitted source trace with its complete audit record."""

    source_row_index: int
    source_row_sha256: str
    task_id: str
    rollout_id: str
    first_user_content: str
    actions: tuple[BashAction, ...]


def sha256_text(value: str) -> str:
    """Return the SHA-256 digest of UTF-8 text."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON without exposing a partially written result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_arguments(arguments: Any, *, context: str) -> dict[str, Any]:
    """Normalize OpenAI tool arguments while failing closed on malformed JSON."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{context}: malformed tool argument JSON") from exc
    if not isinstance(arguments, dict):
        raise ValueError(f"{context}: tool arguments are not an object")
    return arguments


def extract_bash_actions(messages: list[dict[str, Any]]) -> tuple[BashAction, ...]:
    """Extract exact bash calls and pair each with its recorded tool output."""
    tool_outputs: dict[str, str] = {}
    for message_index, message in enumerate(messages):
        if message.get("role") != "tool":
            continue
        tool_call_id = message.get("tool_call_id")
        content = message.get("content")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise ValueError(f"message {message_index}: missing tool_call_id")
        if not isinstance(content, str):
            raise ValueError(f"message {message_index}: tool output is not text")
        if tool_call_id in tool_outputs:
            raise ValueError(f"message {message_index}: duplicate tool output ID")
        tool_outputs[tool_call_id] = content

    actions: list[BashAction] = []
    consumed_outputs: set[str] = set()
    for message_index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            raise ValueError(f"message {message_index}: tool_calls is not a list")
        for tool_call_index, tool_call in enumerate(tool_calls):
            context = f"message {message_index} tool call {tool_call_index}"
            if not isinstance(tool_call, dict):
                raise ValueError(f"{context}: tool call is not an object")
            tool_call_id = tool_call.get("id")
            function = tool_call.get("function")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise ValueError(f"{context}: missing tool call ID")
            if not isinstance(function, dict) or function.get("name") != "bash":
                raise ValueError(f"{context}: only the bash tool can be replayed")
            arguments = parse_arguments(function.get("arguments"), context=context)
            if not {"command"} <= set(arguments) <= {"command", "description"}:
                raise ValueError(
                    f"{context}: expected command and optional description only"
                )
            if not isinstance(arguments["command"], str):
                raise ValueError(f"{context}: command is not a string")
            if "description" in arguments and not isinstance(
                arguments["description"], str
            ):
                raise ValueError(f"{context}: description is not a string")
            if tool_call_id not in tool_outputs:
                raise ValueError(f"{context}: recorded tool output is missing")
            actions.append(
                BashAction(
                    assistant_message_index=message_index,
                    tool_call_index=tool_call_index,
                    tool_call_id=tool_call_id,
                    command=arguments["command"],
                    recorded_output=tool_outputs[tool_call_id],
                )
            )
            consumed_outputs.add(tool_call_id)

    unused_outputs = sorted(set(tool_outputs) - consumed_outputs)
    if unused_outputs:
        raise ValueError(f"unpaired recorded tool outputs: {unused_outputs}")
    if not actions:
        raise ValueError("trajectory has no replayable bash actions")
    return tuple(actions)


def load_admitted_traces(path: Path) -> list[AdmittedTrace]:
    """Load a fail-closed set of replay-selected or legacy SFT-admitted traces."""
    traces: list[AdmittedTrace] = []
    task_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            admission = record.get("admission")
            source = record.get("source")
            selected_for_replay = isinstance(admission, dict) and bool(
                admission.get("selected_for_replay")
                or admission.get("selected_for_sft")
            )
            if not selected_for_replay:
                raise ValueError(f"line {line_number}: trace is not selected for replay")
            if not isinstance(source, dict):
                raise ValueError(f"line {line_number}: source record is missing")
            task_id = source.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"line {line_number}: task_id is missing")
            if task_id in task_ids:
                raise ValueError(f"line {line_number}: duplicate task_id {task_id}")
            messages = json.loads(source["message_log_json"])
            if not isinstance(messages, list):
                raise ValueError(f"line {line_number}: message log is not a list")
            user_messages = [
                message.get("content") for message in messages if message.get("role") == "user"
            ]
            if not user_messages or not isinstance(user_messages[0], str):
                raise ValueError(f"line {line_number}: first user message is missing")
            traces.append(
                AdmittedTrace(
                    source_row_index=int(record["source_row_index"]),
                    source_row_sha256=str(record["source_row_sha256"]),
                    task_id=task_id,
                    rollout_id=str(source["rollout_id"]),
                    first_user_content=user_messages[0],
                    actions=extract_bash_actions(messages),
                )
            )
            task_ids.add(task_id)
    if not traces:
        raise ValueError("audit dataset is empty")
    return traces


def load_completed_results(
    out_root: Path, traces: list[AdmittedTrace]
) -> dict[str, dict[str, Any]]:
    """Load terminal episode results that can be skipped during an exact resume."""
    completed: dict[str, dict[str, Any]] = {}
    for trace in traces:
        result_path = out_root / "episodes" / trace.task_id / "replay_result.json"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("task_id") != trace.task_id:
            raise ValueError(f"resume task mismatch in {result_path}")
        if result.get("source_row_sha256") != trace.source_row_sha256:
            raise ValueError(f"resume source hash mismatch in {result_path}")
        if result.get("finished_at") is None:
            continue
        completed[trace.task_id] = result
    return completed


async def replay_trace(
    *,
    trace: AdmittedTrace,
    task: Any,
    pool: Any,
    out_root: Path,
    bash_timeout_s: int,
) -> dict[str, Any]:
    """Replay a source trace in a fresh task sandbox and invoke its verifier."""
    result_path = out_root / "episodes" / trace.task_id / "replay_result.json"
    started_at = time.time()
    result: dict[str, Any] = {
        "schema": "xtoken-glm52-teacher-real-verifier-replay-v1",
        "model_judgment_is_official_task_verification": False,
        "source_row_index": trace.source_row_index,
        "source_row_sha256": trace.source_row_sha256,
        "task_id": trace.task_id,
        "rollout_id": trace.rollout_id,
        "source_action_count": len(trace.actions),
        "started_at": started_at,
        "finished_at": None,
        "status": "starting",
        "actions": [],
    }
    write_json_atomic(result_path, result)
    try:
        if task.instruction not in trace.first_user_content:
            result["status"] = "source_task_instruction_mismatch"
            return result
        result["task_instruction_sha256"] = sha256_text(task.instruction)
        result["source_user_message_sha256"] = sha256_text(trace.first_user_content)
        result["task_instruction_embedded_in_source_prompt"] = True
        async with pool.session(task) as sandbox:
            result["setup"] = asdict(sandbox.report)
            if not sandbox.report.ok:
                result["status"] = "infrastructure_setup_failed"
                return result
            initial = await sandbox.check_initial_state()
            result["initial_state"] = asdict(initial)
            if not initial.passed:
                result["status"] = "infrastructure_initial_state_failed"
                return result

            for action_index, action in enumerate(trace.actions):
                action_started = time.time()
                shell_result = await sandbox.shell(
                    action.command,
                    timeout_s=bash_timeout_s,
                )
                replay_output = shell_result.output
                action_record = {
                    "action_index": action_index,
                    "assistant_message_index": action.assistant_message_index,
                    "tool_call_index": action.tool_call_index,
                    "tool_call_id": action.tool_call_id,
                    "command_sha256": sha256_text(action.command),
                    "command_characters": len(action.command),
                    "recorded_output_sha256": sha256_text(action.recorded_output),
                    "recorded_output_characters": len(action.recorded_output),
                    "replay_output_sha256": sha256_text(replay_output),
                    "replay_output_characters": len(replay_output),
                    "recorded_output_exact_match": replay_output == action.recorded_output,
                    "exit_code": shell_result.exit_code,
                    "timed_out": shell_result.timed_out,
                    "wall_s": round(time.time() - action_started, 3),
                }
                result["actions"].append(action_record)
                write_json_atomic(result_path, result)
                if shell_result.timed_out:
                    result["status"] = "replay_action_timeout"
                    return result

            verifier = await sandbox.verify()
            result["final_verifier"] = asdict(verifier)
            result["reward"] = verifier.reward
            result["status"] = "scored"
            result["shell_restarts"] = sandbox.report.shell_restarts
    except Exception as exc:  # noqa: BLE001
        result["status"] = "episode_error"
        result["error_class"] = type(exc).__name__
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["finished_at"] = time.time()
        result["wall_s"] = round(result["finished_at"] - started_at, 3)
        write_json_atomic(result_path, result)
    return result


async def async_main(args: argparse.Namespace) -> int:
    """Replay every admitted trace with bounded E2B concurrency."""
    if not os.environ.get("E2B_API_KEY"):
        raise RuntimeError("E2B_API_KEY is required")
    from nemo_rl.environments.tmax import SandboxConfig, TMaxSandboxPool, load_task

    traces = load_admitted_traces(args.audit_dataset)
    tasks = {}
    for trace in traces:
        task_dir = args.tasks / trace.task_id
        if not task_dir.is_dir():
            raise FileNotFoundError(f"missing task directory: {task_dir}")
        tasks[trace.task_id] = load_task(task_dir)

    audit_sha256 = hashlib.sha256(args.audit_dataset.read_bytes()).hexdigest()
    run_spec = {
        "schema": "xtoken-glm52-teacher-real-verifier-replay-run-v1",
        "audit_dataset": str(args.audit_dataset),
        "audit_dataset_sha256": audit_sha256,
        "task_ids": [trace.task_id for trace in traces],
        "task_count": len(traces),
        "template_alias": args.template_alias,
        "run_id": args.run_id,
        "concurrency": args.concurrency,
        "sandbox_timeout_s": args.sandbox_timeout_s,
        "bash_timeout_s": args.bash_timeout_s,
        "exact_teacher_bash_actions_replayed": True,
        "final_environment_verifier_used": True,
        "model_judgment_is_official_task_verification": False,
    }
    run_spec_path = args.out / "run_spec.json"
    if args.out.exists():
        if not args.resume:
            raise FileExistsError(f"output exists; pass --resume: {args.out}")
        if not run_spec_path.exists():
            raise FileNotFoundError(f"resume run_spec is missing: {run_spec_path}")
        existing_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
        if existing_spec != run_spec:
            raise ValueError("resume run_spec differs from requested protocol")
    else:
        args.out.mkdir(parents=True)
        write_json_atomic(run_spec_path, run_spec)
    completed_results = load_completed_results(args.out, traces)
    pending_traces = [trace for trace in traces if trace.task_id not in completed_results]
    LOGGER.info(
        "resume gate completed=%d pending=%d total=%d",
        len(completed_results),
        len(pending_traces),
        len(traces),
    )
    pool = TMaxSandboxPool(
        SandboxConfig(
            template_alias=args.template_alias,
            run_id=args.run_id,
            lane="glm52-teacher-replay",
            sandbox_timeout_s=args.sandbox_timeout_s,
            bash_timeout_s=args.bash_timeout_s,
        ),
        max_concurrent=args.concurrency,
    )
    completed = len(completed_results)
    lock = asyncio.Lock()

    async def one(trace: AdmittedTrace) -> dict[str, Any]:
        nonlocal completed
        result = await replay_trace(
            trace=trace,
            task=tasks[trace.task_id],
            pool=pool,
            out_root=args.out,
            bash_timeout_s=args.bash_timeout_s,
        )
        async with lock:
            completed += 1
            LOGGER.info(
                "replay progress %d/%d task=%s status=%s reward=%s",
                completed,
                len(traces),
                trace.task_id,
                result["status"],
                result.get("reward"),
            )
        return result

    new_results = await asyncio.gather(*(one(trace) for trace in pending_traces))
    result_by_task = {
        **completed_results,
        **{str(result["task_id"]): result for result in new_results},
    }
    results = [result_by_task[trace.task_id] for trace in traces]
    scored = [result for result in results if result.get("reward") is not None]
    positive = [result for result in scored if float(result["reward"]) > 0]
    perfect = [result for result in scored if float(result["reward"]) == 1]
    summary = {
        "schema": "xtoken-glm52-teacher-real-verifier-replay-summary-v1",
        "audit_dataset_sha256": audit_sha256,
        "attempted": len(results),
        "scored": len(scored),
        "positive": len(positive),
        "perfect": len(perfect),
        "reward_sum": sum(float(result["reward"]) for result in scored),
        "mean_reward": (
            sum(float(result["reward"]) for result in scored) / len(scored) if scored else None
        ),
        "status_counts": {
            status: sum(result["status"] == status for result in results)
            for status in sorted({result["status"] for result in results})
        },
        "pool": asdict(pool.stats),
        "model_judgment_is_official_task_verification": False,
        "final_environment_verifier_used": True,
    }
    write_json_atomic(args.out / "summary.json", summary)
    LOGGER.info("replay summary %s", json.dumps(summary, sort_keys=True))
    return 0 if len(scored) == len(results) else 1


def parse_args() -> argparse.Namespace:
    """Parse the durable replay command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dataset", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--template-alias", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--sandbox-timeout-s", type=int, default=3600)
    parser.add_argument("--bash-timeout-s", type=int, default=120)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run the replay and return a fail-closed process status."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.concurrency <= 0:
        raise ValueError("concurrency must be positive")
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
