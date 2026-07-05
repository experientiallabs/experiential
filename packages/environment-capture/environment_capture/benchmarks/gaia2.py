"""GAIA2 adapter: a stateful multi-app simulated world, graded by a deterministic action match.

Upstream: Gaia2 / Meta Agents Research Environments (ARE),
``meta-agents-research-environments/gaia2`` (CC-BY-4.0). A scenario drops an agent into a simulated
universe of apps (Contacts, Email, Messaging, Calendar, RentAFlat, Shopping, CabApp, ...) populated
with fictional data, and a USER message states a task — e.g. "save every apartment in zip codes
whose violent-crime rate is 5-10". The agent acts by calling the apps' real Python tools against a
LIVE, MUTABLE world (state persists across steps), the ``CommandEnv.execute`` seam this benchmark
exercises. We scope to the ``execution`` and ``search`` capabilities, whose tasks are completed from
the initial universe state by agent tool calls (no time-driven environment events).

The heavy ARE engine (its own dep tree + the scenario universes) lives in a benchmark-local venv,
so THIS module — part of the whole-repo gate — never imports it. It drives the engine OUT OF
PROCESS: :meth:`Gaia2Adapter.open_env` launches ``backend/world_backend.py serve`` under that venv,
which boots one live scenario and speaks a line-delimited JSON protocol on stdio
(:class:`Gaia2Env` is the client). Each agent action is a block of Python run against the live apps;
the backend logs every executed write-action and, on close, dumps that log to a per-task state file.

GRADING IS OUR OWN DETERMINISTIC STRUCTURAL APPROXIMATION, **not** the official Gaia2 score. The
official verifier matches the agent's write-actions to the scenario's oracle actions using
exact-match for structured fields AND an LLM rubric (Llama-3.3-70B) for free-text fields; an LLM
judge is incompatible with the environment_capture contract (``grade`` must be deterministic and
LLM-free so a world model is judged by the same fixed function). :func:`score_actions` therefore
matches agent write-actions to oracle actions by exact/normalized comparison — numeric equality for
numbers, and normalized-string equality (lowercased, whitespace-collapsed) for text, which is
STRICTER than the official LLM rubric on free text and is order-insensitive. Reward is the matched
fraction; treat the numbers as this harness's internal signal, never as official-comparable.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from botocore.exceptions import ClientError, ConnectTimeoutError, ReadTimeoutError

from environment_capture.adapter import AgentRun, CommandEnv, ExecResult
from environment_capture.agent import ConverseClient, make_bedrock_client
from environment_capture.subproc import StderrTail
from environment_capture.trajectory import JsonValue, StepRecord, Task, ToolCall

_READY_TIMEOUT_S = 180.0  # first boot imports ARE + populates the scenario universe
_ERROR_MARKER = "Execution failed"  # how the backend reports a raised/invalid snippet


class Gaia2Error(RuntimeError):
    """The world backend subprocess failed to boot or died mid-episode."""


# --------------------------------------------------------------------------- deterministic grader


@dataclass(frozen=True)
class Action:
    """One structured world action: an app tool call with its argument values."""

    app: str
    function: str
    args: dict[str, JsonValue]


_WS_RE = re.compile(r"\s+")


def _canonical(value: JsonValue) -> JsonValue:
    """Canonicalize an argument value for order-insensitive structural comparison.

    Numbers (and numeric strings) become floats; strings are lowercased and whitespace-collapsed;
    JSON-encoded strings (e.g. ``'["a@b"]'``) are parsed first so a serialized list compares like a
    real list. Lists canonicalize elementwise (order preserved); dicts by sorted keys.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        return [_canonical(v) for v in value]
    if isinstance(value, dict):
        return {k: _canonical(value[k]) for k in sorted(value)}
    if isinstance(value, str):
        text = value.strip()
        if text and text[0] in '[{"':
            try:
                return _canonical(json.loads(text))
            except (json.JSONDecodeError, ValueError):
                pass
        # A numeric-LOOKING string with a leading zero is an identifier (phone number, zip,
        # account id): '007' must not equal '7'. Only canonical numeric text becomes a float.
        stripped = text.lstrip("+-")
        is_id_like = len(stripped) > 1 and stripped[0] == "0" and stripped[1] not in ".eE"
        if not is_id_like:
            try:
                return float(text)
            except ValueError:
                pass
        return _WS_RE.sub(" ", text).lower().strip()
    return value


def _args_match(agent_args: dict[str, JsonValue], oracle_args: dict[str, JsonValue]) -> bool:
    """True if the agent call's args cover every oracle arg with a canonically-equal value."""
    for name, oracle_value in oracle_args.items():
        if name not in agent_args:
            return False
        if _canonical(agent_args[name]) != _canonical(oracle_value):
            return False
    return True


def _actions_match(agent: Action, oracle: Action) -> bool:
    return (
        agent.app == oracle.app
        and agent.function == oracle.function
        and _args_match(agent.args, oracle.args)
    )


def score_actions(agent_actions: list[Action], oracle_actions: list[Action]) -> float:
    """Deterministic structural match: matched fraction of oracle actions (see module docstring).

    Greedy 1:1 matching between agent write-actions and oracle actions on (app, function, args).
    Reward = matches / max(#oracle, #agent) so BOTH missed oracle actions and extra agent actions
    lower the score; 1.0 requires exactly the oracle set with matching args and no extras. An empty
    oracle with no agent actions scores 1.0.
    """
    if not oracle_actions and not agent_actions:
        return 1.0
    remaining = list(agent_actions)
    matched = 0
    for oracle in oracle_actions:
        for index, agent in enumerate(remaining):
            if _actions_match(agent, oracle):
                matched += 1
                remaining.pop(index)
                break
    denominator = max(len(oracle_actions), len(agent_actions))
    return matched / denominator if denominator else 1.0


def _actions_from_records(records: list[JsonValue]) -> list[Action]:
    """Parse the backend's JSON action log / a task's oracle list into Action objects."""
    actions: list[Action] = []
    for record in records:
        assert isinstance(record, dict)
        app = record.get("app", "")
        function = record.get("function", "")
        args = record.get("args", {})
        assert isinstance(app, str) and isinstance(function, str) and isinstance(args, dict)
        actions.append(Action(app=app, function=function, args=args))
    return actions


def _graded_agent_actions(records: list[JsonValue]) -> list[Action]:
    """Keep only the agent's WRITE-operation calls (state-changing / user-facing) for grading."""
    graded: list[JsonValue] = []
    for record in records:
        assert isinstance(record, dict)
        if record.get("write_operation"):
            graded.append(record)
    return _actions_from_records(graded)


# --------------------------------------------------------------------------- out-of-process world


@dataclass(frozen=True)
class _Reply:
    output: str
    error: bool


class Gaia2Env:
    """CommandEnv over one live GAIA2 scenario: ``execute(code)`` runs Python against its apps.

    The ``command`` is a block of Python executed in the scenario's stateful shell (a ``tools`` dict
    of the scenario's app tools is preloaded; world mutations persist across calls). Output is the
    printed/returned text; a non-zero return code flags a raised or invalid snippet. On ``close``
    the backend writes the agent's executed write-action log to ``state_path`` for the grader.
    """

    def __init__(self, command: list[str], *, cwd: Path, timeout_s: int = 120) -> None:
        self.timeout_s = timeout_s
        self._process = subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        # Backends deliberately route all engine chatter to stderr to keep stdout a clean
        # protocol channel — stderr must be drained or the child blocks once the pipe fills.
        self._stderr_tail = StderrTail(self._process.stderr)
        self._await_ready()

    def _await_ready(self) -> None:
        line = self._read_line(timeout_s=int(_READY_TIMEOUT_S))
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            self.close()
            raise Gaia2Error(f"world backend sent no readiness handshake: {line!r}") from error
        if not payload.get("ready"):
            self.close()
            raise Gaia2Error(f"world backend failed to boot: {payload.get('error', line)!r}")

    def _read_line(self, *, timeout_s: int) -> str:
        assert self._process.stdout is not None
        deadline = time.monotonic() + timeout_s
        while True:
            line = self._process.stdout.readline()
            if line:
                return line
            if self._process.poll() is not None:
                stderr = self._stderr_tail.text()
                raise Gaia2Error(
                    f"world backend exited (code {self._process.returncode}): {stderr}"
                )
            if time.monotonic() > deadline:
                raise Gaia2Error(f"world backend timed out after {timeout_s}s")
            time.sleep(0.02)

    def _request(self, payload: dict[str, JsonValue]) -> _Reply:
        assert self._process.stdin is not None
        self._process.stdin.write(json.dumps(payload) + "\n")
        self._process.stdin.flush()
        reply = json.loads(self._read_line(timeout_s=self.timeout_s))
        return _Reply(output=str(reply.get("output", "")), error=bool(reply.get("error", False)))

    def execute(self, command: str) -> ExecResult:
        """Run one block of Python against the live scenario; return its output and error status."""
        if self._process.poll() is not None:
            return ExecResult(output="world backend is no longer running", returncode=1)
        try:
            reply = self._request({"op": "execute", "code": command})
        except (Gaia2Error, json.JSONDecodeError, OSError) as error:
            return ExecResult(output=f"world backend error: {error}", returncode=1)
        return ExecResult(output=reply.output, returncode=1 if reply.error else 0)

    def close(self) -> None:
        if self._process.poll() is None and self._process.stdin is not None:
            try:
                self._process.stdin.write(json.dumps({"op": "close"}) + "\n")
                self._process.stdin.flush()
            except OSError:
                pass
        try:
            self._process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()


class Gaia2Adapter:
    """BenchmarkAdapter over materialized GAIA2 scenarios, driven via the venv backend.

    ``root`` holds ``data/{split}.jsonl`` (task index: prompt + oracle actions), ``datafiles/
    <task_id>.json`` (each scenario's full universe JSON, gitignored — re-fetched), the ``.venv``
    with the ARE engine, and ``backend/world_backend.py``. Every ARE operation runs as a subprocess
    under ``.venv`` so this gate-checked module stays ARE-free. Grading is state-based: the backend
    writes the agent's write-action log to ``runs_state/<experiment_prefix>--<task_id>.json`` on
    close, and :meth:`grade` matches it to the task's oracle actions via :func:`score_actions`.
    """

    name = "gaia2"

    def __init__(
        self,
        root: Path,
        *,
        experiment_prefix: str = "wmh-cap",
        venv_python: Path | None = None,
        backend: Path | None = None,
        timeout_s: int = 120,
    ) -> None:
        self.root = root
        self.experiment_prefix = experiment_prefix
        self.venv_python = venv_python or root / ".venv" / "bin" / "python"
        self.backend = backend or root / "backend" / "world_backend.py"
        self.timeout_s = timeout_s

    def tasks(self, split: str) -> list[Task]:
        path = self.root / "data" / f"{split}.jsonl"
        tasks: list[Task] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            tasks.append(
                Task(task_id=raw["task_id"], prompt=raw["prompt"], data=raw.get("data", {}))
            )
        return tasks

    def _scenario_path(self, task: Task) -> Path:
        return self.root / "datafiles" / f"{task.task_id}.json"

    def _state_path(self, task: Task) -> Path:
        return self.root / "runs_state" / f"{self.experiment_prefix}--{task.task_id}.json"

    def open_env(self, task: Task) -> Gaia2Env:
        state_path = self._state_path(task)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.unlink(missing_ok=True)
        command = [
            str(self.venv_python),
            str(self.backend),
            "serve",
            str(self._scenario_path(task)),
            str(state_path),
        ]
        return Gaia2Env(command, cwd=self.root, timeout_s=self.timeout_s)

    def grade(self, task: Task, submission: str) -> float:
        """Match the agent's logged write-actions to the task's oracle actions (deterministic).

        The submission text is not the grading signal (grading is action-based), so it is ignored;
        the grader reads the write-action log the backend dumped for this task.
        """
        del submission
        state_path = self._state_path(task)
        if not state_path.exists():
            return 0.0
        agent_log = json.loads(state_path.read_text(encoding="utf-8"))
        assert isinstance(agent_log, list)
        oracle_raw = task.data.get("oracle", [])
        assert isinstance(oracle_raw, list)
        return score_actions(_graded_agent_actions(agent_log), _actions_from_records(oracle_raw))


# --------------------------------------------------------------------------- capture agent

_SYSTEM_PROMPT = """You are an autonomous agent operating a simulated world of apps (Contacts,
Email, Messaging, Calendar, RentAFlat, Shopping, CabApp, CityApp and more) pre-populated with the
user's data. You act ONLY by writing Python with the `execute_python` tool: each call runs in a
STATEFUL shell where `tools` (a dict mapping tool names to callables) is preloaded, so variables and
world changes persist across calls. Start by exploring: run `print(describe_tools())` to see the
available `App__function` tools and their signatures, then read data with the list/search/get tools
before acting. Call a tool as `tools["App__function"](arg=value)`. Inspect results before assuming
them, and take one focused step per call. Carry out the user's request exactly (all of it), then, if
the task asks a question, send your answer with `tools["AgentUserInterface__send_message_to_user"]
(content=...)`. When the request is fully done, call the `finish` tool. If you get stuck, still call
`finish` rather than looping."""

_INSTRUCTIONS = (
    "\n\nComplete this request in the world's Python shell using the `execute_python` tool. "
    "Explore the tools with describe_tools(), read before writing, act, then call finish."
)

_TOOL_CONFIG: dict[str, JsonValue] = {
    "tools": [
        {
            "toolSpec": {
                "name": "execute_python",
                "description": "Run one block of Python in the stateful world shell (`tools` and "
                "`describe_tools` are preloaded); returns the real printed output or error.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {"code": {"type": "string"}},
                        "required": ["code"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "finish",
                "description": "End the task once the request is fully carried out.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": [],
                    }
                },
            }
        },
    ]
}

_THROTTLE_CODES = {"ThrottlingException", "TooManyRequestsException", "ServiceUnavailableException"}
_MAX_RETRIES = 6


def task_prompt_with_instructions(task: Task) -> str:
    """The scenario's USER message plus the how-to-act framing."""
    return task.prompt + _INSTRUCTIONS


class Gaia2Agent:
    """CaptureAgent that drives a Gaia2Env through Bedrock converse tool-use.

    The model's only environment action is a block of Python (``execute_python``) run against the
    live world; it ends by calling ``finish``. Throttling is retried with linear backoff; other
    errors propagate so ``run_capture`` can isolate the task.
    """

    def __init__(
        self,
        model_id: str,
        *,
        client: ConverseClient | None = None,
        region: str = "us-east-1",
        max_steps: int = 24,
        max_tokens: int = 2048,
        retry_backoff_s: float = 5.0,
    ) -> None:
        self.model_id = model_id
        self._client = client if client is not None else make_bedrock_client(region)
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.retry_backoff_s = retry_backoff_s

    def _converse(self, messages: list[JsonValue]) -> dict[str, JsonValue]:
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return self._client.converse(
                    modelId=self.model_id,
                    messages=messages,
                    system=[{"text": _SYSTEM_PROMPT}],
                    toolConfig=_TOOL_CONFIG,
                    inferenceConfig={"maxTokens": self.max_tokens},
                )
            except (ReadTimeoutError, ConnectTimeoutError):
                if attempt == _MAX_RETRIES:
                    raise
                time.sleep(self.retry_backoff_s * (attempt + 1))
            except ClientError as error:
                code = error.response.get("Error", {}).get("Code", "")
                if code not in _THROTTLE_CODES or attempt == _MAX_RETRIES:
                    raise
                time.sleep(self.retry_backoff_s * (attempt + 1))
        raise RuntimeError("unreachable")

    def run(self, task: Task, env: CommandEnv) -> AgentRun:
        """Drive the world until the agent finishes, answers in text, or hits max_steps."""
        messages: list[JsonValue] = [
            {"role": "user", "content": [{"text": task_prompt_with_instructions(task)}]}
        ]
        steps: list[StepRecord] = []
        final_answer = ""

        while len(steps) < self.max_steps:
            response = self._converse(messages)
            output = response.get("output")
            assert isinstance(output, dict)
            message = output.get("message")
            assert isinstance(message, dict)
            messages.append(message)

            tool_uses = _tool_uses(message)
            if not tool_uses:
                final_answer = _text_content(message)
                break

            tool_results: list[JsonValue] = []
            finished = False
            for tool_use_id, name, tool_input in tool_uses:
                if name == "finish":
                    answer = tool_input.get("answer", "")
                    final_answer = answer if isinstance(answer, str) else str(answer)
                    finished = True
                    break

                code = tool_input.get("code", "")
                code_text = code if isinstance(code, str) else str(code)
                result = env.execute(code_text)
                steps.append(
                    StepRecord(
                        action=ToolCall(name="execute_python", arguments={"code": code_text}),
                        output=result.output,
                        is_error=result.returncode != 0,
                    )
                )
                tool_results.append(
                    {
                        "toolResult": {
                            "toolUseId": tool_use_id,
                            "content": [{"text": result.output or "(no output)"}],
                            "status": "error" if result.returncode != 0 else "success",
                        }
                    }
                )
                if len(steps) >= self.max_steps:
                    break

            if finished:
                break
            messages.append({"role": "user", "content": tool_results})

        return AgentRun(steps=steps, final_answer=final_answer, model=self.model_id)


def _tool_uses(message: dict[str, JsonValue]) -> list[tuple[str, str, dict[str, JsonValue]]]:
    """Every (id, name, input) tool-use block in the message, in order."""
    content = message.get("content")
    if not isinstance(content, list):
        return []
    uses: list[tuple[str, str, dict[str, JsonValue]]] = []
    for block in content:
        if isinstance(block, dict) and isinstance(block.get("toolUse"), dict):
            tool_use = block["toolUse"]
            assert isinstance(tool_use, dict)
            tool_use_id = tool_use.get("toolUseId", "")
            name = tool_use.get("name", "")
            tool_input = tool_use.get("input", {})
            assert isinstance(tool_use_id, str) and isinstance(name, str)
            assert isinstance(tool_input, dict)
            uses.append((tool_use_id, name, tool_input))
    return uses


def _text_content(message: dict[str, JsonValue]) -> str:
    content = message.get("content")
    if not isinstance(content, list):
        return ""
    parts = [
        block["text"]
        for block in content
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    ]
    return "\n".join(str(p) for p in parts).strip()
