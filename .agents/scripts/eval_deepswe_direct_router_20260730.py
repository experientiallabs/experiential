"""Controlled fresh DeepSWE quality comparison using an Azure Docker daemon.

The local process only orchestrates provider calls. Repository execution and grading happen on
the Docker daemon selected by ``DOCKER_HOST=ssh://...``. The driver has a hard spend ceiling and
defaults to two tasks; broader runs require an explicit task limit.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
import tomllib
import uuid
from pathlib import Path
from typing import Any, Literal

from wmo.config.dotenv import load_env_file

MAX_TURNS = 60
# Match the existing DeepSWE WMO evaluation harness. The reasoning effort remains arm-specific.
MAX_OUTPUT_TOKENS = 16_384
TOOL_TIMEOUT_S = 120.0
VERIFY_TIMEOUT_S = 1_800.0
TOOL_OUT_CAP = 30_000
PATCH_OUT_CAP = 5_000_000
CHARS_PER_TOKEN = 4.0
PREFILL_SWITCH_RATIO = 4.0
DEFAULT_MAX_TASKS = 2
HARD_CAP_USD = 900.0
BOILERPLATE = (
    "\nIMPORTANT: Please work on this in a new branch from main and commit everything when you "
    "are done.\n"
)
SYSTEM = """You are a software engineer working in a checked-out git repository at /app.

You have one tool: bash. Use it to explore the repository, edit files, and run tests.
The repository is at a base commit with no future history; there is no network access.

Work until the task is fully implemented. Prefer reading the existing code and tests before
editing. When you are done, make sure every change is saved to disk.

Be efficient: batch independent shell commands into one call when possible."""
BASH_SCHEMA = {
    "type": "object",
    "properties": {"command": {"type": "string", "description": "bash command in /app"}},
    "required": ["command"],
}
BASH_DESC = "Run a bash command in the repository at /app and return its combined output."
REWARD_RE = re.compile(r"\{.*?\"f2p_total\".*?\}", re.S)


@dataclasses.dataclass(frozen=True)
class Arm:
    name: str
    provider: Literal["openai", "anthropic"]
    model: str
    effort: str
    input_per_mtok: float
    cached_input_per_mtok: float
    output_per_mtok: float
    cache_write_per_mtok: float


@dataclasses.dataclass
class Usage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0

    def add(self, *, input_tokens: int, cached_input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.cached_input_tokens += cached_input_tokens
        self.output_tokens += output_tokens
        self.requests += 1

    def cost(self, arm: Arm) -> float:
        fresh = max(0, self.input_tokens - self.cached_input_tokens)
        fresh_rate = arm.cache_write_per_mtok if arm.provider == "openai" else arm.input_per_mtok
        return (
            fresh * fresh_rate
            + self.cached_input_tokens * arm.cached_input_per_mtok
            + self.output_tokens * arm.output_per_mtok
        ) / 1_000_000


@dataclasses.dataclass
class Step:
    turn: int
    text: str
    calls: list[dict[str, Any]]
    results: list[str]


@dataclasses.dataclass
class TurnResult:
    step: Step | None
    native: list[Any]
    usage: Usage
    stop: str
    error: str | None


class SpendGuard:
    """Hard stop for new model turns; completed work is never silently discarded."""

    def __init__(self, cap_usd: float):
        if cap_usd <= 0 or cap_usd > HARD_CAP_USD:
            raise ValueError(f"--cap-usd must be in (0, {HARD_CAP_USD:g}]")
        self.cap_usd = cap_usd
        self.spent_usd = 0.0

    def may_start_turn(self) -> bool:
        return self.spent_usd < self.cap_usd

    def add(self, amount: float) -> None:
        self.spent_usd += max(0.0, amount)


class RemoteContainer:
    """Container lifecycle through Docker CLI and the caller-selected remote daemon."""

    def __init__(self, image: str, tag: str):
        self.image = image
        self.tag = tag
        self.cid: str | None = None

    def __enter__(self) -> "RemoteContainer":
        result = _docker(
            "run", "-d", "--rm=false", "--network", "none", "--cpus", "2", "--memory", "8g",
            "--label", "coding-router=deepswe-quality-20260730", "--label",
            f"coding-router-job={self.tag}", "-w", "/app", self.image, "sleep", "infinity",
        )
        self.cid = result.stdout.strip()
        return self

    def __exit__(self, *_: object) -> None:
        if self.cid:
            _docker("rm", "-f", self.cid, check=False, timeout=120.0)

    def bash(
        self, command: str, *, timeout: float = TOOL_TIMEOUT_S, output_cap: int = TOOL_OUT_CAP
    ) -> str:
        if not self.cid:
            raise RuntimeError("container is not running")
        try:
            result = _docker(
                "exec", "-w", "/app", self.cid, "bash", "-lc", command,
                check=False, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return f"[command timed out after {timeout:.0f}s and was killed]"
        output = (result.stdout or "") + (result.stderr or "")
        if result.returncode != 0:
            output += f"\n[exit code {result.returncode}]"
        return output[:output_cap] if output else "[no output]"

    def patch(self, base_commit: str) -> str:
        return self.bash(
            "git add -A >/dev/null 2>&1; "
            f"git diff --binary --cached {shlex.quote(base_commit)}",
            timeout=180.0,
            output_cap=PATCH_OUT_CAP,
        )


def _docker(*args: str, timeout: float = 300.0, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"docker {args[0]} failed: {(result.stderr or '')[:800]}")
    return result


def _load_credentials(path: Path | None) -> None:
    candidates = [path] if path is not None else []
    candidates.extend([
        Path.cwd() / ".env.local",
        Path("/Users/admin/Documents/experientiallabs/coding-router/.env.local"),
    ])
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            load_env_file(candidate)
            break
    missing = [key for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY") if not os.environ.get(key)]
    if missing:
        raise RuntimeError(f"credentials unavailable by environment name: {missing}")


def _arms(policy_path: Path) -> tuple[dict[str, Arm], dict[str, Any]]:
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    arms: dict[str, Arm] = {}
    for row in policy["pool"]:
        name = str(row["name"])
        model, effort = name.rsplit("__", 1)
        provider: Literal["openai", "anthropic"] = "anthropic" if model.startswith("claude-") else "openai"
        arms[name] = Arm(
            name=name,
            provider=provider,
            model=model,
            effort=effort,
            input_per_mtok=float(row["input_per_mtok"]),
            cached_input_per_mtok=float(row.get("cached_input_per_mtok") or row["input_per_mtok"]),
            output_per_mtok=float(row["output_per_mtok"]),
            cache_write_per_mtok=float(row.get("cache_write_per_mtok") or row["input_per_mtok"]),
        )
    return arms, policy


def _profile_model(policy: dict[str, Any], text_chars: int) -> str:
    bucket = sum(text_chars > float(bound) for bound in policy["profile_bins"])
    return str(policy["profile_models"][bucket])


def _route(
    policy: dict[str, Any], arms: dict[str, Arm], *, mode: Literal["task", "transcript"],
    route_chars: int, incumbent: str | None,
) -> tuple[str, str, str]:
    desired = _profile_model(policy, route_chars)
    if incumbent is None:
        return desired, "cold-start", "not-applicable"
    if mode == "task":
        return incumbent, "sticky-task-route", "not-applicable"
    if desired == incumbent:
        return desired, "profile-kept-incumbent", "not-needed"
    current, candidate = arms[incumbent], arms[desired]
    warm = (route_chars / CHARS_PER_TOKEN) * current.cached_input_per_mtok / 1_000_000
    cold = (route_chars / CHARS_PER_TOKEN) * candidate.input_per_mtok / 1_000_000
    if warm <= 0.0 or cold <= PREFILL_SWITCH_RATIO * warm:
        return desired, f"cache-aware-switch ${cold:.6f} <= {PREFILL_SWITCH_RATIO:g}x ${warm:.6f}", "passed"
    return incumbent, f"cache-aware-keep ${cold:.6f} > {PREFILL_SWITCH_RATIO:g}x ${warm:.6f}", "reverted"


def _transcript_text(prompt: str, steps: list[Step]) -> str:
    parts = [prompt]
    for step in steps:
        parts.extend([step.text])
        for call, result in zip(step.calls, step.results, strict=True):
            parts.extend([str(call.get("name", "")), json.dumps(call.get("input", {}), sort_keys=True), result])
    return "\n".join(parts)


def _tasks(task_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for directory in sorted(task_root.iterdir()):
        if not directory.is_dir() or not (directory / "task.toml").is_file():
            continue
        config = tomllib.loads((directory / "task.toml").read_text(encoding="utf-8"))
        tests = directory / "tests"
        environment, metadata = config.get("environment") or {}, config.get("metadata") or {}
        if not environment.get("docker_image") or not (tests / "config.json").is_file():
            continue
        if not (json.loads((tests / "config.json").read_text(encoding="utf-8")).get("f2p_node_ids")):
            continue
        prompt = (directory / "instruction.md").read_text(encoding="utf-8")
        if not prompt.endswith(BOILERPLATE):
            prompt += BOILERPLATE
        rows.append({
            "task_id": directory.name,
            "repo": str(metadata.get("repository_url") or directory.name),
            "image": str(environment["docker_image"]),
            "base_commit": str(metadata["base_commit_hash"]),
            "prompt": prompt,
            "tests_dir": tests,
        })
    return rows


def _select_tasks(all_tasks: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    by_id = {task["task_id"]: task for task in all_tasks}
    if args.task_id:
        missing = sorted(set(args.task_id) - set(by_id))
        if missing:
            raise ValueError(f"task ids missing from the full DeepSWE tree: {missing}")
        selected = [by_id[task_id] for task_id in args.task_id]
    elif args.holdout:
        policy = json.loads(args.policy.read_text(encoding="utf-8"))
        fit_ids = {value.split(":", 1)[1] for value in policy["fit_scenario_ids"]}
        selected = [task for task in all_tasks if task["task_id"] not in fit_ids]
        if args.n_tasks is not None:
            selected = sorted(
                selected,
                key=lambda task: hashlib.sha256(
                    f"{args.sample_seed}|deepswe-holdout|{task['task_id']}".encode()
                ).hexdigest(),
            )[:args.n_tasks]
    else:
        ordered = sorted(all_tasks, key=lambda task: hashlib.sha256(
            f"{args.sample_seed}|deepswe|{task['task_id']}".encode()).hexdigest())
        selected = ordered if args.n_tasks is None else ordered[:args.n_tasks]
    if len(selected) > args.max_tasks:
        raise ValueError(f"selected {len(selected)} tasks exceeds explicit --max-tasks={args.max_tasks}")
    return selected


class TurnClient:
    """One structured provider request, preserving native history when the arm is unchanged."""

    def __init__(self, arm: Arm, max_output_tokens: int):
        self.arm = arm
        self.max_output_tokens = max_output_tokens
        if arm.provider == "anthropic":
            import anthropic

            self.client = anthropic.Anthropic(max_retries=5, timeout=900.0)
        else:
            import openai

            self.client = openai.OpenAI(max_retries=5, timeout=900.0)

    def call(self, prompt: str, steps: list[Step], native: list[Any], same_model: bool) -> TurnResult:
        messages = self._seed(prompt, steps, native, same_model)
        usage = Usage()
        try:
            if self.arm.provider == "anthropic":
                return self._anthropic(messages, steps, usage)
            return self._openai(messages, steps, usage)
        except Exception as exc:  # noqa: BLE001
            return TurnResult(None, native, usage, "error", f"{type(exc).__name__}: {exc}"[:600])

    def _seed(self, prompt: str, steps: list[Step], native: list[Any], same_model: bool) -> list[Any]:
        if not steps:
            return [{"role": "user", "content": prompt}]
        if same_model and native:
            return list(native)
        if self.arm.provider == "anthropic":
            messages: list[Any] = [{"role": "user", "content": prompt}]
            for step in steps:
                blocks: list[dict[str, Any]] = []
                if step.text:
                    blocks.append({"type": "text", "text": step.text})
                blocks.extend({
                    "type": "tool_use", "id": call["id"], "name": call["name"], "input": call["input"],
                } for call in step.calls)
                messages.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "(continuing)"}]})
                if step.calls:
                    messages.append({"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": call["id"], "content": result[:TOOL_OUT_CAP]}
                        for call, result in zip(step.calls, step.results, strict=True)
                    ]})
            return messages
        history: list[Any] = [{"role": "user", "content": prompt}]
        for step in steps:
            if step.text:
                history.append({"role": "assistant", "content": step.text})
            history.extend({
                "type": "function_call", "call_id": call["id"], "name": call["name"],
                "arguments": json.dumps(call["input"]),
            } for call in step.calls)
            history.extend({
                "type": "function_call_output", "call_id": call["id"], "output": result[:TOOL_OUT_CAP],
            } for call, result in zip(step.calls, step.results, strict=True))
        return history

    def _anthropic(self, messages: list[Any], steps: list[Step], usage: Usage) -> TurnResult:
        response = self.client.messages.create(
            model=self.arm.model,
            max_tokens=self.max_output_tokens,
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
            tools=[{"name": "bash", "description": BASH_DESC, "input_schema": BASH_SCHEMA}],
            thinking={"type": "adaptive"},
            output_config={"effort": self.arm.effort},
        )
        ru = response.usage
        usage.add(input_tokens=int(ru.input_tokens or 0), cached_input_tokens=int(getattr(ru, "cache_read_input_tokens", 0) or 0), output_tokens=int(ru.output_tokens or 0))
        native = messages + [{"role": "assistant", "content": response.content}]
        calls = [{"id": block.id, "name": block.name, "input": dict(block.input)} for block in response.content if block.type == "tool_use"]
        text = " ".join(block.text for block in response.content if block.type == "text")[:4000]
        return TurnResult(Step(len(steps) + 1, text, calls, []), native, usage, response.stop_reason, None)

    def _openai(self, history: list[Any], steps: list[Step], usage: Usage) -> TurnResult:
        response = self.client.responses.create(
            model=self.arm.model,
            instructions=SYSTEM,
            input=history,
            max_output_tokens=self.max_output_tokens,
            tools=[{"type": "function", "name": "bash", "description": BASH_DESC, "parameters": BASH_SCHEMA}],
            reasoning={"effort": self.arm.effort},
        )
        ru = response.usage
        details = getattr(ru, "input_tokens_details", None)
        usage.add(input_tokens=int(ru.input_tokens or 0), cached_input_tokens=int(getattr(details, "cached_tokens", 0) or 0), output_tokens=int(ru.output_tokens or 0))
        native = history + [item.model_dump(exclude_none=True) for item in response.output]
        calls: list[dict[str, Any]] = []
        for item in response.output:
            if getattr(item, "type", "") != "function_call":
                continue
            try:
                arguments = json.loads(item.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            calls.append({"id": item.call_id, "name": item.name, "input": arguments})
        return TurnResult(Step(len(steps) + 1, (response.output_text or "")[:4000], calls, []), native, usage, response.status, None)


def _append_tool_results(arm: Arm, native: list[Any], step: Step) -> list[Any]:
    """Return provider-native history after executing the step's tool calls."""
    if not step.calls:
        return native
    if arm.provider == "anthropic":
        return native + [{
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": call["id"], "content": result[:TOOL_OUT_CAP]}
                for call, result in zip(step.calls, step.results, strict=True)
            ],
        }]
    return native + [
        {"type": "function_call_output", "call_id": call["id"], "output": result[:TOOL_OUT_CAP]}
        for call, result in zip(step.calls, step.results, strict=True)
    ]


def _run_episode(
    task: dict[str, Any], strategy: str, policy: dict[str, Any], arms: dict[str, Arm],
    guard: SpendGuard, log_path: Path, max_turns: int, max_output_tokens: int,
) -> tuple[dict[str, Any], str]:
    baseline = str(policy["default_model"])
    clients: dict[str, TurnClient] = {}
    steps: list[Step] = []
    native: list[Any] = []
    incumbent: str | None = None
    routes: list[dict[str, Any]] = []
    usage_by_arm: dict[str, Usage] = collections.defaultdict(Usage)
    started = time.monotonic()
    stop, error = "max_turns", None
    with RemoteContainer(task["image"], f"quality-{strategy}-{task['task_id']}-{uuid.uuid4().hex[:8]}") as box:
        for turn in range(1, max_turns + 1):
            if not guard.may_start_turn():
                stop = "budget"
                break
            route_text = task["prompt"] if strategy != "per-turn" else _transcript_text(task["prompt"], steps)
            if strategy == "baseline":
                chosen, reason, gate = baseline, "static-baseline", "not-applicable"
            else:
                chosen, reason, gate = _route(
                    policy, arms, mode="task" if strategy == "per-task" else "transcript",
                    route_chars=len(route_text), incumbent=incumbent,
                )
            switched = incumbent is not None and chosen != incumbent
            result = clients.setdefault(chosen, TurnClient(arms[chosen], max_output_tokens)).call(
                task["prompt"], steps, native if not switched else [], not switched and incumbent == chosen,
            )
            usage_by_arm[chosen].add(
                input_tokens=result.usage.input_tokens,
                cached_input_tokens=result.usage.cached_input_tokens,
                output_tokens=result.usage.output_tokens,
            )
            turn_cost = result.usage.cost(arms[chosen])
            guard.add(turn_cost)
            routes.append({
                "turn": turn, "model": chosen, "incumbent_before": incumbent, "switched": switched,
                "route_chars": len(route_text), "reason": reason, "gate": gate,
                "input_tokens": result.usage.input_tokens, "cached_input_tokens": result.usage.cached_input_tokens,
                "output_tokens": result.usage.output_tokens, "cost_usd": turn_cost,
            })
            if result.stop == "error":
                stop, error = "error", result.error
                break
            if result.step is None:
                stop, error = "error", "provider returned no step"
                break
            for call in result.step.calls:
                result.step.results.append(box.bash(call["input"].get("command", "")))
            steps.append(result.step)
            native, incumbent = _append_tool_results(arms[chosen], result.native, result.step), chosen
            if not result.step.calls:
                stop = "max_tokens" if result.stop == "incomplete" else "end_turn"
                break
        patch = "" if stop == "error" else box.patch(task["base_commit"])
    log_path.write_text(json.dumps({"task_id": task["task_id"], "strategy": strategy, "routes": routes}, indent=2) + "\n", encoding="utf-8")
    usage_json = {model: dataclasses.asdict(value) | {"cost_usd": value.cost(arms[model])} for model, value in usage_by_arm.items()}
    return ({
        "task_id": task["task_id"], "strategy": strategy, "stop": stop, "error": error,
        "turns": len(steps), "switches": sum(row["switched"] for row in routes),
        "model_mix": dict(collections.Counter(row["model"] for row in routes)),
        "usage_by_model": usage_json, "cost_usd": sum(row["cost_usd"] for row in routes),
        "wall_s": round(time.monotonic() - started, 3), "patch_bytes": len(patch),
    }, patch)


def _grade(task: dict[str, Any], patch: str, root: Path) -> dict[str, Any]:
    tag = f"grade-{task['task_id']}-{uuid.uuid4().hex[:8]}"
    patch_path = root / f"{tag}.patch"
    patch_path.write_text(patch, encoding="utf-8")
    try:
        with RemoteContainer(task["image"], tag) as box:
            box.bash("mkdir -p /tests /logs/artifacts /logs/verifier")
            for name in ("test.sh", "test.patch", "grader.py", "config.json"):
                _docker("cp", str(task["tests_dir"] / name), f"{box.cid}:/tests/{name}", timeout=120.0)
            if patch:
                _docker("cp", str(patch_path), f"{box.cid}:/logs/artifacts/model.patch", timeout=120.0)
            box.bash("chmod +x /tests/test.sh")
            output = box.bash(
                "bash /tests/test.sh 2>&1 | tail -c 4000; "
                "echo '---REWARD---'; cat /logs/verifier/reward.json 2>/dev/null",
                timeout=VERIFY_TIMEOUT_S,
            )
    finally:
        patch_path.unlink(missing_ok=True)
    match = REWARD_RE.search(output.rsplit("---REWARD---", 1)[-1])
    if not match:
        return {"verifier_ok": False, "tail": output[-1500:]}
    result = json.loads(match.group(0))
    result["verifier_ok"] = True
    result["tail"] = output[-1500:]
    if result.get("f2p_total"):
        result["graded"] = result["f2p_passed"] / result["f2p_total"]
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--n-tasks", type=int)
    parser.add_argument("--max-tasks", type=int, default=DEFAULT_MAX_TASKS)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--cap-usd", type=float, default=HARD_CAP_USD)
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS)
    parser.add_argument("--max-output-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    parser.add_argument("--strategy", action="append", choices=("baseline", "per-task", "per-turn"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.max_tasks < 1:
        raise ValueError("--max-tasks must be positive")
    if args.max_turns < 1 or args.max_output_tokens < 1:
        raise ValueError("--max-turns and --max-output-tokens must be positive")
    _load_credentials(args.env_file)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    arms, policy = _arms(args.policy.resolve())
    tasks = _select_tasks(_tasks(args.task_root.resolve()), args)
    if not tasks:
        raise ValueError("no tasks selected")
    strategies = args.strategy or ["baseline", "per-task", "per-turn"]
    guard = SpendGuard(args.cap_usd)
    manifest = {
        "benchmark": "DeepSWE 1.1",
        "task_count": len(tasks),
        "task_ids": [task["task_id"] for task in tasks],
        "baseline": policy["default_model"],
        "policy_path": str(args.policy.resolve()),
        "cache_aware": True,
        "cache_switch_prefill_ratio": PREFILL_SWITCH_RATIO,
        "strategies": strategies,
        "cap_usd": args.cap_usd,
        "docker_host": os.environ.get("DOCKER_HOST", "local-default"),
        "timestamp": time.time(),
    }
    (args.artifact_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results: list[dict[str, Any]] = []
    for strategy in strategies:
        for task in tasks:
            if not guard.may_start_turn():
                break
            print(f"starting strategy={strategy} task={task['task_id']} spent=${guard.spent_usd:.3f}", flush=True)
            route_path = args.artifact_root / f"{strategy}-{task['task_id']}-route.json"
            run, patch = _run_episode(
                task, strategy, policy, arms, guard, route_path,
                args.max_turns, args.max_output_tokens,
            )
            grade = _grade(task, patch, args.artifact_root)
            run.update({
                "f2p_passed": grade.get("f2p_passed"), "f2p_total": grade.get("f2p_total"),
                "graded": grade.get("graded"), "reward_binary": grade.get("reward"),
                "verifier_ok": grade.get("verifier_ok", False), "verifier_tail": grade.get("tail"),
                "route_log": str(route_path), "spent_after": guard.spent_usd,
            })
            results.append(run)
            print(
                f"completed strategy={strategy} task={task['task_id']} graded={run['graded']} "
                f"binary={run['reward_binary']} cost=${run['cost_usd']:.3f} switches={run['switches']}",
                flush=True,
            )
    summary = {"manifest": manifest, "results": results, "spent_usd": guard.spent_usd}
    (args.artifact_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
