"""Run the frozen SWE-rebench development matrix on resumable E2B workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from e2b import CommandResult, Sandbox, Template

logger = logging.getLogger("coding-router-swerebench-execute")

PROTOCOL = "coding-router-swerebench-development-execution-v1"
TEMPLATE_NAME = "deepswe-router-responses-v2"
TEMPLATE_ID = "j1a2bxbpllu3rp84b4qj"
TEMPLATE_BUILD_ID = "e971c040-95bd-45c1-89ee-fb597bf75671"
MODEL = "gpt-5.6-luna"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
CORPUS_SHA256 = "7d846b5576d15e68fd18ac21bfe0610cc1614b3b35ec0ae0cb8cfae0b82962c1"
SMOKE_REPORT_SHA256 = "ee76a57040cbe7aaef692d2fc3f3df66d7a556cbf6dda74119e0802cb4230e13"
SMOKE_ARCHIVE_SHA256 = {
    "xhigh": "bf1d576d25f1b56ae3a9484db5d5599576519a218aec3073db29272345f4015b",
    "max": "c449dc999a4d604546c358affcf5e1cba1865aba8ca312789b92b5eb27bb4e6a",
}
REUSED_TASKS = {
    "0xs34n__starknet.js-538",
    "acloudguru__serverless-plugin-aws-alerts-13",
}
DOCKER_ADAPTER_REPORT_SHA256 = (
    "08499c87fa93b9ec58c76fabbf16c388df169506bd5160e2ca604f3a7b62938a"
)
RESPONSES_ADAPTER_REPORT_SHA256 = (
    "476f4a5e0a67fc4880fc80ed27e52d333620461da63775689e8f5be38e66179c"
)
E2B_ACCOUNT_CAP = 1_000
TASK_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+$")
IMAGE_PATTERN = re.compile(r"^docker\.io/swerebenchv2/[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+$")
STATE_LOCK = threading.Lock()

REMOTE_VALIDATOR = r'''"""Audit new matrix traces and write a compact report."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--traces", type=Path, required=True)
parser.add_argument("--task", required=True)
parser.add_argument("--effort", required=True)
parser.add_argument("--expected", type=int, required=True)
parser.add_argument("--attempt-offset", type=int, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

records = [
    json.loads(line)
    for line in args.traces.read_bytes().split(b"\n")
    if line.strip()
]
if len(records) != args.expected:
    raise SystemExit(f"expected {args.expected} outer rows, found {len(records)}")
cells = []
totals = {field: 0 for field in (
    "prompt_tokens", "cached_input_tokens", "completion_tokens", "reasoning_tokens"
)}
for index, outer in enumerate(records):
    traces = outer.get("traces")
    if not isinstance(traces, list) or len(traces) != 1:
        raise SystemExit(f"row {index} lacks exactly one official trace")
    trace = traces[0]
    if trace.get("task", {}).get("data", {}).get("name") != args.task:
        raise SystemExit(f"row {index} has a different task")
    if trace.get("verifiers", {}).get("commit") != "f6e420b9908ae14d625f079881f13c15011ee1c9":
        raise SystemExit(f"row {index} has a different verifier commit")
    calls = trace.get("calls")
    if not isinstance(calls, list) or not calls or len(calls) > 40:
        raise SystemExit(f"row {index} has invalid provider calls")
    reward = trace.get("rewards", {}).get("solved", {}).get("score")
    trace_errors = trace.get("errors", [])
    timeout_error = isinstance(trace_errors, list) and any(
        isinstance(error, dict)
        and error.get("type") == "HarnessError"
        and isinstance(error.get("message"), str)
        and error["message"].startswith("agent timeout:")
        for error in trace_errors
    )
    mini_swe_agent_exit_137 = isinstance(trace_errors, list) and any(
        isinstance(error, dict)
        and error.get("type") == "HarnessError"
        and isinstance(error.get("message"), str)
        and error["message"].startswith("harness 'mini_swe_agent' exited 137:")
        for error in trace_errors
    )
    post_execution_agent_failure = (
        reward is None
        and trace.get("ok") is False
        and outer.get("ok") is False
        and trace.get("stop_condition") in {"error", "max_turns"}
        and trace.get("info", {}).get("patch") is None
        and isinstance(trace_errors, list)
        and (timeout_error or mini_swe_agent_exit_137)
    )
    if post_execution_agent_failure:
        reward = 0.0
        reward_provenance = (
            "gradeable post-execution agent timeout"
            if timeout_error
            else "gradeable post-execution mini-swe-agent exit 137"
        )
    elif (
        isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or float(reward) not in {0.0, 1.0}
    ):
        raise SystemExit(f"row {index} lacks an official binary reward")
    else:
        reward_provenance = "official verifier"
    scoring = trace.get("timing", {}).get("scoring", {})
    start, end = scoring.get("start"), scoring.get("end")
    if post_execution_agent_failure:
        scoring_seconds = None
    elif not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or end <= start:
        raise SystemExit(f"row {index} lacks official scoring timing")
    else:
        scoring_seconds = end - start
    usage = {field: 0 for field in totals}
    provider_errors = []
    for call_index, call in enumerate(calls):
        if call.get("model") != "gpt-5.6-luna":
            raise SystemExit(f"row {index} call {call_index} has a different model")
        if call.get("endpoint") != "/responses":
            raise SystemExit(f"row {index} call {call_index} has a different endpoint")
        sampling = call.get("sampling", {})
        if sampling.get("reasoning_effort") != args.effort or sampling.get("max_tokens") != 32768:
            raise SystemExit(f"row {index} call {call_index} has different sampling")
        call_usage = call.get("usage")
        if call_usage is None:
            error = call.get("error")
            if not isinstance(error, dict):
                raise SystemExit(f"row {index} call {call_index} lacks usage and error")
            status = error.get("status_code")
            if not isinstance(status, int) or not 429 <= status <= 599:
                raise SystemExit(f"row {index} call {call_index} has ungradeable missing usage")
            provider_errors.append({
                "call_index": call_index,
                "type": error.get("type"),
                "status_code": status,
                "usage_charge": "zero; provider returned no inference usage",
            })
            continue
        if not isinstance(call_usage, dict):
            raise SystemExit(f"row {index} call {call_index} has invalid usage")
        for field in totals:
            value = call_usage.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SystemExit(f"row {index} call {call_index} has invalid {field}")
            usage[field] += value
            totals[field] += value
    if usage["reasoning_tokens"] > usage["completion_tokens"]:
        raise SystemExit(f"row {index} reasoning exceeds output tokens")
    inference_calls = len(calls) - len(provider_errors)
    if not 1 <= inference_calls <= 20:
        raise SystemExit(f"row {index} has invalid provider inference calls")
    patch = trace.get("info", {}).get("patch")
    if post_execution_agent_failure:
        patch_bytes = 0
        patch_sha256 = None
    elif not isinstance(patch, str):
        raise SystemExit(f"row {index} lacks a patch string")
    else:
        patch_bytes = len(patch.encode())
        patch_sha256 = hashlib.sha256(patch.encode()).hexdigest()
    cells.append({
        "attempt_number": args.attempt_offset + index,
        "reward": float(reward),
        "reward_provenance": reward_provenance,
        "official_verifier_reached": not post_execution_agent_failure,
        "provider_calls": len(calls),
        "provider_inference_calls": inference_calls,
        "provider_errors": provider_errors,
        "stop_condition": trace.get("stop_condition"),
        "trace_ok": trace.get("ok"),
        "outer_ok": outer.get("ok"),
        "trace_errors": trace_errors,
        "outer_errors": outer.get("errors", []),
        "patch_bytes": patch_bytes,
        "patch_sha256": patch_sha256,
        "scoring_seconds": scoring_seconds,
        "usage": usage,
    })
report = {
    "protocol": "coding-router-swerebench-effort-artifact-v1",
    "valid": True,
    "task_id": args.task,
    "model": "gpt-5.6-luna",
    "effort": args.effort,
    "new_cells": args.expected,
    "attempt_offset": args.attempt_offset,
    "cells": cells,
    "usage": totals,
    "usage_provenance": "exact token counts from pinned verifier Responses traces",
}
args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
'''


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _capacity() -> int:
    paginator = Sandbox.list(limit=100)
    count = 0
    while True:
        count += len(paginator.next_items())
        if not paginator.has_next:
            return count


def _docker_image(alias: str) -> str:
    prefix = "prime/primeintellect/"
    if not alias.startswith(prefix):
        raise ValueError(f"unexpected frozen image alias: {alias!r}")
    image = "docker.io/swerebenchv2/" + alias.removeprefix(prefix)
    if not IMAGE_PATTERN.fullmatch(image):
        raise ValueError(f"unsafe frozen image: {image!r}")
    return image


def _effort_order(task_index: int) -> tuple[str, ...]:
    offset = task_index % len(EFFORTS)
    return EFFORTS[offset:] + EFFORTS[:offset]


def _new_rollouts(task_id: str, effort: str) -> tuple[int, int]:
    if task_id in REUSED_TASKS and effort in SMOKE_ARCHIVE_SHA256:
        return 1, 1
    return 2, 0


def _config(task_id: str, effort: str, rollouts: int, output_dir: str) -> str:
    if not TASK_PATTERN.fullmatch(task_id):
        raise ValueError(f"unsafe frozen task id: {task_id!r}")
    return f'''model = "{MODEL}"
num_tasks = 1
num_rollouts = {rollouts}
shuffle = false
max_concurrent = 2
verbose = false
rich = false
server = false
push = false
output_dir = "{output_dir}"

[client]
type = "eval"
base_url = "https://api.openai.com/v1"
api_key_var = "OPENAI_API_KEY"

[sampling]
temperature = 1.0
reasoning_effort = "{effort}"
max_tokens = 32768

[env]
max_concurrent_agents = 1

[env.taskset]
id = "swerebench-v2-v1"
dataset_name = "PrimeIntellect/SWE-rebench-V2-Filtered-Verified"
split = "train"
filter_fn = "lambda row: row['instance_id'] == '{task_id}'"

[env.agent]
max_turns = 20
max_output_tokens = 131072

[env.agent.harness]
id = "mini_swe_agent"
version = "2.4.5"

[env.agent.runtime]
type = "docker"

[env.agent.timeout]
setup = 900
rollout = 900
finalize = 300
scoring = 900

[env.retries]
max_retries = 0
'''


def _run(
    sandbox: Sandbox,
    command: str,
    *,
    timeout: float,
    check: bool = True,
) -> CommandResult:
    result = sandbox.commands.run(command, timeout=timeout)
    if check and result.exit_code:
        raise RuntimeError(
            f"remote command failed exit={result.exit_code} stderr={result.stderr[-1000:]!r}"
        )
    return result


def _run_durable_eval(
    sandbox: Sandbox,
    command: str,
    *,
    effort: str,
    exit_status_path: str,
    state: dict[str, Any],
    state_path: Path,
    attempt: dict[str, Any],
    timeout: float,
    poll_interval: float = 10.0,
) -> tuple[CommandResult, Sandbox]:
    """Run one scientific command once and poll its persisted exit status.

    Long-lived E2B output streams have intermittently failed at the HTTP/2
    control plane while the remote command kept running. A short launcher starts
    a detached wrapper exactly once, while remote PID and atomic exit markers
    make completion recoverable without ever issuing the scientific command a
    second time.
    """
    temporary_status_path = f"{exit_status_path}.tmp"
    wrapper_path = f"{exit_status_path}.wrapper.sh"
    pid_path = f"{exit_status_path}.pid"
    temporary_pid_path = f"{pid_path}.tmp"
    lock_path = f"{exit_status_path}.launch-lock"
    log_path = f"{exit_status_path}.log"
    wrapped = (
        "set +e\n"
        f"{command}\n"
        "router_eval_status=$?\n"
        f"printf '%s\\n' \"$router_eval_status\" > {temporary_status_path}\n"
        f"mv {temporary_status_path} {exit_status_path}\n"
        'exit "$router_eval_status"'
    )
    sandbox.files.write(wrapper_path, wrapped)
    launcher = (
        "set -eu\n"
        f"if mkdir {lock_path} 2>/dev/null; then\n"
        f"  nohup bash {wrapper_path} > {log_path} 2>&1 </dev/null &\n"
        "  router_eval_pid=$!\n"
        f"  printf '%s\\n' \"$router_eval_pid\" > {temporary_pid_path}\n"
        f"  mv {temporary_pid_path} {pid_path}\n"
        "else\n"
        f"  test -s {pid_path}\n"
        "fi\n"
        f"cat {pid_path}"
    )
    launch_result = sandbox.commands.run(launcher, timeout=120)
    pid = int(launch_result.stdout.strip())
    if pid <= 1:
        raise ValueError(f"invalid durable eval pid: {pid}")
    process = {
        "pid": pid,
        "exit_status_path": exit_status_path,
        "pid_path": pid_path,
        "wrapper_path": wrapper_path,
        "scientific_command_starts": 1,
    }
    processes = attempt.setdefault("effort_processes", {})
    if not isinstance(processes, dict):
        raise ValueError("invalid durable effort process state")
    processes[effort] = process
    _write_json(state_path, state)

    deadline = time.monotonic() + timeout
    active = sandbox
    missing_pid_polls = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"durable eval timed out effort={effort} pid={pid}; "
                "the scientific command was not rerun"
            )
        try:
            if active.files.exists(exit_status_path, request_timeout=60):
                raw_status = active.files.read(exit_status_path).strip()
                exit_code = int(raw_status)
                if exit_code < 0 or exit_code > 255:
                    raise ValueError(f"invalid durable eval exit status: {raw_status!r}")
                process["exit_code"] = exit_code
                process["completed"] = True
                _write_json(state_path, state)
                return (
                    CommandResult(
                        stdout="",
                        stderr="",
                        exit_code=exit_code,
                        error=None,
                    ),
                    active,
                )
            process_result = active.commands.run(
                (
                    f"if kill -0 {pid} 2>/dev/null; then "
                    "printf running; else printf stopped; fi"
                ),
                timeout=60,
            )
            running = process_result.stdout.strip() == "running"
        except Exception as error:  # noqa: BLE001 - reconnect exact remote PID state
            logger.warning(
                "durable eval poll reconnect effort=%s pid=%d error=%r",
                effort,
                pid,
                error,
            )
            poll_errors = process.setdefault("poll_errors", [])
            if isinstance(poll_errors, list):
                poll_errors.append(repr(error))
            _write_json(state_path, state)
            active = Sandbox.connect(sandbox.sandbox_id, request_timeout=60)
        else:
            if not running:
                if active.files.exists(exit_status_path, request_timeout=60):
                    continue
                missing_pid_polls += 1
                process["missing_pid_polls"] = missing_pid_polls
                _write_json(state_path, state)
                if missing_pid_polls <= 5:
                    time.sleep(min(1.0, remaining))
                    continue
                raise RuntimeError(
                    f"durable eval pid={pid} ended without exit marker; "
                    "the scientific command was not rerun"
                )
            missing_pid_polls = 0
        time.sleep(min(poll_interval, remaining))


def _sync(sandbox: Sandbox, remote: str, local: Path) -> None:
    stream = sandbox.files.read(
        remote,
        format="stream",
        request_timeout=180,
        stream_idle_timeout=180,
        gzip=True,
    )
    temporary = local.with_suffix(local.suffix + ".tmp")
    with stream, temporary.open("wb") as handle:
        for chunk in stream:
            handle.write(chunk)
    temporary.replace(local)


def _verify_completed_effort(task_dir: Path, effort: str, payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    archive = task_dir / f"{effort}.tar.gz"
    report = task_dir / f"{effort}.report.json"
    if not archive.is_file() or not report.is_file():
        return False
    if _sha256(archive) != payload.get("archive_sha256"):
        return False
    if _sha256(report) != payload.get("report_sha256"):
        return False
    report_data = _read_object(report)
    return bool(report_data.get("valid")) and report_data.get("effort") == effort


def _task_complete(task_dir: Path, state: dict[str, Any]) -> bool:
    efforts = state.get("efforts", {})
    return isinstance(efforts, dict) and all(
        _verify_completed_effort(task_dir, effort, efforts.get(effort))
        for effort in EFFORTS
    )


def _task_excluded(state: dict[str, Any]) -> bool:
    """Return whether a task has an audited infrastructure-cell exclusion."""
    exclusion = state.get("exclusion")
    return (
        state.get("stage") == "excluded-infrastructure"
        and isinstance(exclusion, dict)
        and exclusion.get("scope") == "whole-task"
        and isinstance(exclusion.get("effort"), str)
        and isinstance(exclusion.get("reason"), str)
        and isinstance(exclusion.get("evidence_sha256"), str)
        and isinstance(exclusion.get("usage"), dict)
        and isinstance(exclusion.get("provider_calls"), int)
        and isinstance(exclusion.get("observed_scientific_cells"), int)
        and exclusion.get("scientific_cells_rerun") == 0
    )


def _update_summary(root: Path, total_tasks: int) -> None:
    with STATE_LOCK:
        states = list((root / "tasks").glob("*/state.json"))
        completed_efforts = 0
        completed_new_cells = 0
        reused_cells = 0
        complete_tasks = 0
        excluded_tasks = 0
        failed_tasks = 0
        provider_calls = 0
        usage = {
            "prompt_tokens": 0,
            "cached_input_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
        }
        for state_path in states:
            state = _read_object(state_path)
            efforts = state.get("efforts", {})
            if isinstance(efforts, dict):
                for payload in efforts.values():
                    if not isinstance(payload, dict):
                        continue
                    completed_efforts += 1
                    completed_new_cells += int(payload.get("new_cells", 0))
                    reused_cells += int(payload.get("reused_cells", 0))
                    provider_calls += int(payload.get("provider_calls", 0))
                    payload_usage = payload.get("usage", {})
                    if isinstance(payload_usage, dict):
                        for field in usage:
                            usage[field] += int(payload_usage.get(field, 0))
            if _task_excluded(state):
                exclusion = state["exclusion"]
                excluded_usage = exclusion["usage"]
                completed_new_cells += int(exclusion["observed_scientific_cells"])
                provider_calls += int(exclusion["provider_calls"])
                for field in usage:
                    usage[field] += int(excluded_usage.get(field, 0))
            if state.get("stage") == "complete":
                complete_tasks += 1
            if _task_excluded(state):
                excluded_tasks += 1
            if state.get("stage") == "failed":
                failed_tasks += 1
        cost = (
            usage["prompt_tokens"] / 1_000_000
            + usage["cached_input_tokens"] * 0.1 / 1_000_000
            + usage["completion_tokens"] * 6.0 / 1_000_000
        )
        _write_json(
            root / "progress.json",
            {
                "protocol": PROTOCOL,
                "total_tasks": total_tasks,
                "expected_cells": total_tasks * 10,
                "complete_tasks": complete_tasks,
                "excluded_tasks": excluded_tasks,
                "retained_task_coverage": (total_tasks - excluded_tasks) / total_tasks,
                "failed_tasks": failed_tasks,
                "completed_efforts": completed_efforts,
                "completed_new_cells": completed_new_cells,
                "reused_smoke_cells": reused_cells,
                "completed_scientific_cells": completed_new_cells + reused_cells,
                "provider_calls": provider_calls,
                "usage": usage,
                "matrix_cost_usd": cost,
                "rough_cumulative_experiment_spend_usd": 405.7678502 + cost,
            },
        )


def _run_task(
    root: Path,
    task_index: int,
    row: dict[str, Any],
    api_key: str,
    total_tasks: int,
) -> None:
    task_id = str(row["task_id"])
    image = _docker_image(str(row["image_name"]))
    task_dir = root / "tasks" / f"{task_index:04d}"
    task_dir.mkdir(parents=True, exist_ok=True)
    state_path = task_dir / "state.json"
    if state_path.is_file():
        state = _read_object(state_path)
        if state.get("task_id") != task_id or state.get("image") != image:
            raise ValueError(f"task state identity drift at {task_dir}")
        if _task_excluded(state):
            return
        if _task_complete(task_dir, state):
            return
    else:
        state = {
            "protocol": PROTOCOL,
            "task_index": task_index,
            "task_id": task_id,
            "image": image,
            "effort_order": list(_effort_order(task_index)),
            "efforts": {},
            "sandbox_attempts": [],
            "stage": "pending",
        }
        _write_json(state_path, state)

    effort_state = state["efforts"]
    if not isinstance(effort_state, dict):
        raise ValueError(f"invalid effort state for {task_id}")
    missing = [
        effort
        for effort in _effort_order(task_index)
        if not _verify_completed_effort(task_dir, effort, effort_state.get(effort))
    ]
    if not missing:
        state["stage"] = "complete"
        _write_json(state_path, state)
        return

    sandbox = Sandbox.create(
        TEMPLATE_NAME,
        timeout=6 * 3_600,
        secure=True,
        allow_internet_access=True,
        envs={"OPENAI_API_KEY": api_key},
        metadata={
            "owner": "coding-router-v40",
            "phase": "swerebench-development-matrix",
            "task_index": str(task_index),
            "task_id": task_id,
        },
    )
    attempts = state["sandbox_attempts"]
    if not isinstance(attempts, list):
        raise ValueError(f"invalid sandbox attempt state for {task_id}")
    attempt: dict[str, Any] = {
        "sandbox_id": sandbox.sandbox_id,
        "missing_efforts": missing,
        "terminated": False,
    }
    attempts.append(attempt)
    state["stage"] = "running"
    _write_json(state_path, state)
    verified = False
    remote_root = f"/home/user/router-v40-development/{task_index:04d}"
    try:
        _run(sandbox, f"mkdir -p {remote_root}/runtime", timeout=120)
        _run(
            sandbox,
            (
                "test \"$(sha256sum /opt/coding-router/"
                "swerebench-docker-adapter-report.json | cut -d' ' -f1)\" = "
                f"{DOCKER_ADAPTER_REPORT_SHA256}"
            ),
            timeout=120,
        )
        _run(
            sandbox,
            (
                "test \"$(sha256sum /opt/coding-router/"
                "verifiers-responses-adapter-report.json | cut -d' ' -f1)\" = "
                f"{RESPONSES_ADAPTER_REPORT_SHA256}"
            ),
            timeout=120,
        )
        _run(
            sandbox,
            (
                f"cp /opt/coding-router/*-adapter-report.json {remote_root}/runtime/ "
                f"&& sha256sum {remote_root}/runtime/* > {remote_root}/runtime/sha256sums"
            ),
            timeout=120,
        )
        _run(sandbox, f"sudo docker pull {image}", timeout=1_800)
        image_id = _run(
            sandbox,
            f"sudo docker image inspect {image} --format '{{{{.Id}}}}'",
            timeout=120,
        ).stdout.strip()
        state["docker_image_id"] = image_id
        sandbox.files.write(f"{remote_root}/validate.py", REMOTE_VALIDATOR)

        for effort in missing:
            rollouts, attempt_offset = _new_rollouts(task_id, effort)
            output_dir = f"{remote_root}/{effort}"
            config_path = f"{remote_root}/{effort}.toml"
            report_path = f"{remote_root}/{effort}.report.json"
            archive_path = f"{remote_root}/{effort}.tar.gz"
            sandbox.files.write(
                config_path,
                _config(task_id, effort, rollouts, output_dir),
            )
            state["stage"] = f"running-{effort}"
            _write_json(state_path, state)
            eval_result, sandbox = _run_durable_eval(
                sandbox,
                f"cd /opt/verifiers && sudo -E .venv/bin/eval @ {config_path}",
                effort=effort,
                exit_status_path=f"{remote_root}/{effort}.eval-exit-status",
                state=state,
                state_path=state_path,
                attempt=attempt,
                timeout=3 * 3_600,
            )
            _run(
                sandbox,
                (
                    f"sudo /opt/verifiers/.venv/bin/python {remote_root}/validate.py "
                    f"--traces {output_dir}/traces.jsonl --task {task_id} "
                    f"--effort {effort} --expected {rollouts} "
                    f"--attempt-offset {attempt_offset} --output {report_path}"
                ),
                timeout=120,
            )
            _run(
                sandbox,
                (
                    f"sudo tar -C {remote_root} -czf {archive_path} "
                    f"runtime {effort}.toml {effort}.report.json {effort}"
                ),
                timeout=300,
            )
            local_archive = task_dir / f"{effort}.tar.gz"
            local_report = task_dir / f"{effort}.report.json"
            _sync(sandbox, archive_path, local_archive)
            local_report.write_text(
                sandbox.files.read(report_path), encoding="utf-8"
            )
            report = _read_object(local_report)
            if report.get("valid") is not True or report.get("task_id") != task_id:
                raise ValueError(f"downloaded report failed validation for {task_id}/{effort}")
            cells = report.get("cells", [])
            provider_calls = sum(
                int(cell.get("provider_calls", 0))
                for cell in cells
                if isinstance(cell, dict)
            )
            reused = 1 if attempt_offset else 0
            effort_state[effort] = {
                "archive_sha256": _sha256(local_archive),
                "report_sha256": _sha256(local_report),
                "new_cells": rollouts,
                "reused_cells": reused,
                "reused_smoke_archive_sha256": SMOKE_ARCHIVE_SHA256.get(effort)
                if reused
                else None,
                "provider_calls": provider_calls,
                "usage": report.get("usage"),
                "eval_exit_code": eval_result.exit_code,
                "sandbox_id": sandbox.sandbox_id,
            }
            state["stage"] = f"completed-{effort}"
            _write_json(state_path, state)
            _update_summary(root, total_tasks)
            logger.info(
                "task=%d/%d id=%s effort=%s new_cells=%d reused=%d",
                task_index + 1,
                total_tasks,
                task_id,
                effort,
                rollouts,
                reused,
            )
        state["stage"] = "complete"
        verified = True
    except Exception as error:
        state["stage"] = "failed"
        state["error"] = repr(error)
        attempt["error"] = repr(error)
        logger.exception("task failed index=%d id=%s", task_index, task_id)
        raise
    finally:
        if verified:
            sandbox.kill()
            attempt["terminated"] = True
            state["sandbox_terminated"] = True
        _write_json(state_path, state)
        _update_summary(root, total_tasks)


def execute(
    root: Path,
    corpus_path: Path,
    *,
    concurrency: int,
    limit_tasks: int | None,
) -> None:
    """Validate the frozen launch and execute or resume every missing task."""
    if _sha256(corpus_path) != CORPUS_SHA256:
        raise ValueError("development corpus hash mismatch")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is unavailable")
    if concurrency < 1 or concurrency > 100:
        raise ValueError("concurrency must be between 1 and 100")
    if not Template.exists(TEMPLATE_NAME):
        raise RuntimeError(f"required E2B template is absent: {TEMPLATE_NAME}")
    active = _capacity()
    if active + concurrency > E2B_ACCOUNT_CAP:
        raise RuntimeError(
            f"E2B capacity is insufficient: active={active} launch={concurrency} "
            f"cap={E2B_ACCOUNT_CAP}"
        )
    corpus = _read_object(corpus_path)
    rows = corpus.get("tasks")
    if not isinstance(rows, list) or len(rows) != 200:
        raise ValueError("development corpus does not contain exactly 200 tasks")
    selected = rows[:limit_tasks] if limit_tasks is not None else rows
    root.mkdir(parents=True, exist_ok=True)
    (root / "tasks").mkdir(exist_ok=True)
    launch_path = root / "launch.json"
    launch = {
        "protocol": PROTOCOL,
        "corpus_path": str(corpus_path.resolve()),
        "corpus_sha256": CORPUS_SHA256,
        "template": TEMPLATE_NAME,
        "template_id": TEMPLATE_ID,
        "template_build_id": TEMPLATE_BUILD_ID,
        "model": MODEL,
        "efforts": list(EFFORTS),
        "attempts_per_effort": 2,
        "tasks": len(selected),
        "expected_cells": len(selected) * 10,
        "reused_smoke_cells": 4 if len(selected) >= 2 else 2,
        "smoke_report_sha256": SMOKE_REPORT_SHA256,
        "smoke_archive_sha256": SMOKE_ARCHIVE_SHA256,
        "concurrency": concurrency,
        "active_e2b_before": active,
        "e2b_account_cap": E2B_ACCOUNT_CAP,
        "cost_ceiling_usd": 20_000.0,
        "prior_spend_usd": 405.7678502,
        "deep_swe_outcomes_accessed": False,
        "model_persisted": False,
    }
    if launch_path.is_file():
        prior_launch = _read_object(launch_path)
        operational = {"active_e2b_before", "concurrency"}
        frozen_prior = {
            key: value for key, value in prior_launch.items() if key not in operational
        }
        frozen_resume = {
            key: value for key, value in launch.items() if key not in operational
        }
        if frozen_prior != frozen_resume:
            raise ValueError("resume launch manifest differs from the frozen experiment")
    else:
        _write_json(launch_path, launch)
    _update_summary(root, len(selected))
    errors: list[Exception] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                _run_task,
                root,
                index,
                row,
                api_key,
                len(selected),
            ): index
            for index, raw_row in enumerate(selected)
            for row in [_read_row(raw_row, index)]
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:  # noqa: BLE001 - isolate task workers
                errors.append(error)
    _update_summary(root, len(selected))
    if errors:
        raise RuntimeError(f"{len(errors)} task workers failed; inspect task states")


def _read_row(value: object, index: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"corpus task {index} is not an object")
    return value


def main() -> None:
    """Parse command line arguments and run the external development matrix."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=25)
    parser.add_argument("--limit-tasks", type=int)
    args = parser.parse_args()
    execute(
        args.root,
        args.corpus,
        concurrency=args.concurrency,
        limit_tasks=args.limit_tasks,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
