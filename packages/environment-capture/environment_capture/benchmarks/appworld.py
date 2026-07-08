"""AppWorld adapter: a stateful multi-app world driven by real Python-API calls.

Upstream: AppWorld (StonyBrookNLP; github.com/StonyBrookNLP/appworld). AppWorld drops an agent into
a simulated world of nine apps (Amazon, Gmail, Venmo, Spotify, phone, file system, ...) behind 450+
real Python APIs and asks it to complete a multi-step request — e.g. "what is the title of the
most-liked song in my Spotify playlists". The agent acts by writing Python that calls
``apis.<app>.<endpoint>(...)`` against a LIVE, MUTABLE world; it signals completion by calling
``apis.supervisor.complete_task(...)``. This is the first STATEFUL adapter in the harness: world
state carries across steps, which is exactly the world-model dynamics this benchmark exists to
exercise.

The heavy ``appworld`` engine (Python 3.11, its own encrypted data bundles) lives in a
benchmark-local venv, so THIS module — part of the whole-repo gate — never imports it. It drives
the engine OUT OF PROCESS instead: :meth:`AppWorldAdapter.open_env` launches a
``backend/world_backend.py serve`` subprocess under that venv which boots one live ``AppWorld`` for
the task and speaks a line-delimited JSON protocol on stdio (:class:`AppWorldEnv` is the client).
Each agent action is a block of Python executed against that live world — the ``CommandEnv.execute``
seam whose observations a world model reconstructs. Grading shells out to
``backend/world_backend.py grade``, which runs AppWorld's own deterministic evaluation tests over
the final world state; reward is the fraction of those tests that pass (no LLM judging).
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from botocore.exceptions import ClientError, ConnectTimeoutError, ReadTimeoutError

from environment_capture.adapter import AgentRun, CommandEnv, ExecResult
from environment_capture.agent import ConverseClient, make_bedrock_client
from environment_capture.subproc import StderrTail
from environment_capture.trajectory import JsonValue, StepRecord, Task, ToolCall

# The line-delimited JSON stdio protocol spoken with backend/world_backend.py serve. Requests and
# responses are one compact JSON object per line (AppWorld output rides as an escaped string value,
# so a multi-line observation is still a single physical line).
_READY_TIMEOUT_S = 180.0  # first boot loads every app + the task's database
_ERROR_MARKER = "Execution failed"  # how AppWorld's world.execute reports a raised/invalid snippet


class AppWorldError(RuntimeError):
    """The world backend subprocess failed to boot or died mid-episode."""


@dataclass(frozen=True)
class _Reply:
    output: str
    error: bool
    completed: bool


class AppWorldEnv:
    """CommandEnv over one live AppWorld world: ``execute(code)`` runs Python against it.

    The ``command`` a caller passes is a block of Python executed in the world's stateful shell
    (variables and world mutations persist across calls, exactly as in real AppWorld). Output is the
    printed/return text the shell produced; a non-zero return code flags a raised or syntactically
    invalid snippet. ``completed`` tracks whether the agent has called ``complete_task`` yet.
    """

    def __init__(self, command: list[str], *, cwd: Path, timeout_s: int = 120) -> None:
        """Launch the backend subprocess for one task and wait for its readiness handshake."""
        self.timeout_s = timeout_s
        self.completed = False
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
            raise AppWorldError(f"world backend sent no readiness handshake: {line!r}") from error
        if not payload.get("ready"):
            self.close()
            raise AppWorldError(f"world backend failed to boot: {payload.get('error', line)!r}")

    def _read_line(self, *, timeout_s: int) -> str:
        assert self._process.stdout is not None
        deadline = time.monotonic() + timeout_s
        while True:
            line = self._process.stdout.readline()
            if line:
                return line
            if self._process.poll() is not None:
                stderr = self._stderr_tail.text()
                raise AppWorldError(
                    f"world backend exited (code {self._process.returncode}): {stderr}"
                )
            if time.monotonic() > deadline:
                raise AppWorldError(f"world backend timed out after {timeout_s}s")
            time.sleep(0.02)

    def _request(self, payload: dict[str, JsonValue]) -> _Reply:
        assert self._process.stdin is not None
        self._process.stdin.write(json.dumps(payload) + "\n")
        self._process.stdin.flush()
        reply = json.loads(self._read_line(timeout_s=self.timeout_s))
        return _Reply(
            output=str(reply.get("output", "")),
            error=bool(reply.get("error", False)),
            completed=bool(reply.get("completed", False)),
        )

    def execute(self, command: str) -> ExecResult:
        """Run one block of Python against the live world; return its output and error status."""
        if self._process.poll() is not None:
            return ExecResult(output="world backend is no longer running", returncode=1)
        try:
            reply = self._request({"op": "execute", "code": command})
        except (AppWorldError, json.JSONDecodeError, OSError) as error:
            return ExecResult(output=f"world backend error: {error}", returncode=1)
        self.completed = self.completed or reply.completed
        return ExecResult(output=reply.output, returncode=1 if reply.error else 0)

    def close(self) -> None:
        if self._process.poll() is None and self._process.stdin is not None:
            try:
                self._process.stdin.write(json.dumps({"op": "close"}) + "\n")
                self._process.stdin.flush()
            except OSError:
                pass
        try:
            self._process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()


class AppWorldAdapter:
    """BenchmarkAdapter over a materialized AppWorld data directory, driven via the venv backend.

    ``root`` is the benchmark-local directory holding ``data/{split}.jsonl`` (materialized by
    ``backend/fetch_data.py``), the downloaded AppWorld ``data/`` and ``experiments/`` trees, the
    ``.venv`` with the ``appworld`` engine, and ``backend/world_backend.py``. Every AppWorld
    operation runs as a subprocess under ``.venv`` so this gate-checked module stays appworld-free.
    """

    name = "appworld"

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
        # boot serial per task: see _experiment (fresh experiment dir on every open_env)
        self._boot_serials: dict[str, int] = {}
        self._serial_lock = threading.Lock()

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

    def _appworld_id(self, task: Task) -> str:
        appworld_id = task.data.get("appworld_id")
        if not isinstance(appworld_id, str) or not appworld_id:
            raise ValueError(f"task {task.task_id} is missing a string data.appworld_id")
        return appworld_id

    def _experiment(self, task: Task) -> str:
        """A per-(capture-run, task, boot) AppWorld experiment name so states never collide.

        ``experiment_prefix`` carries the capture's run tag (model + pass), so two shards grading
        the same AppWorld task write to disjoint experiment directories. The boot serial makes
        every ``open_env`` (including run_capture's retry of a failed attempt) start from a
        FRESH experiment dir — a retry over the dirty state a crashed attempt left behind would
        grade the wrong world. ``grade`` reads the name of the latest boot for the task.
        """
        serial = self._boot_serials.get(self._appworld_id(task), 0)
        suffix = f"--a{serial}" if serial > 1 else ""
        return f"{self.experiment_prefix}--{self._appworld_id(task)}{suffix}"

    def open_env(self, task: Task) -> AppWorldEnv:
        appworld_id = self._appworld_id(task)
        with self._serial_lock:
            self._boot_serials[appworld_id] = self._boot_serials.get(appworld_id, 0) + 1
        command = [
            str(self.venv_python),
            str(self.backend),
            "serve",
            self._appworld_id(task),
            self._experiment(task),
        ]
        return AppWorldEnv(command, cwd=self.root, timeout_s=self.timeout_s)

    def grade(self, task: Task, submission: str) -> float:
        """Run AppWorld's own evaluation tests over the final world state; reward = pass fraction.

        The submission text is irrelevant to AppWorld (grading is state-based), so it is ignored;
        the grader reads the world the agent left behind under this task's experiment directory.
        """
        del submission
        result = subprocess.run(
            [
                str(self.venv_python),
                str(self.backend),
                "grade",
                self._appworld_id(task),
                self._experiment(task),
            ],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
        )
        if result.returncode != 0:
            raise AppWorldError(f"grade failed for {task.task_id}: {result.stderr.strip()}")
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        return float(payload["reward"])


# --------------------------------------------------------------------------- capture agent

_SYSTEM_PROMPT = """You are an autonomous agent operating AppWorld, a simulated world of nine apps
(Amazon, Gmail, Venmo, Spotify, phone, file system, Splitwise, Todoist, SimpleNote) plus a
`supervisor` app for your account. You act ONLY by writing Python code with the `execute_python`
tool: each call runs your code in a STATEFUL Python shell where `apis` is preloaded, so variables
and world changes persist across calls. Start by exploring: read the task, then discover and read
the relevant API docs with `apis.api_docs.show_api_descriptions(app_name=...)` and
`apis.api_docs.show_api_doc(app_name=..., api_name=...)`. Authenticate via
`apis.supervisor.show_account_passwords()` and each app's login endpoint before calling protected
APIs. Inspect results before assuming them, and take one focused step per call. When — and only
when — the request is fully carried out, call `apis.supervisor.complete_task()` (pass
`answer=<value>` if the task asks a question), then call the `finish` tool to end. If you get
stuck, still call `finish` rather than looping."""

_INSTRUCTIONS = (
    "\n\nComplete this request in the AppWorld Python shell using the `execute_python` tool. "
    "Explore the APIs, authenticate, act, then call apis.supervisor.complete_task(...) and finish."
)

_TOOL_CONFIG: dict[str, JsonValue] = {
    "tools": [
        {
            "toolSpec": {
                "name": "execute_python",
                "description": "Run one block of Python in the stateful AppWorld shell (the `apis` "
                "object is preloaded); returns the real printed output or error traceback.",
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
                "description": "End the task after you have called apis.supervisor.complete_task.",
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
    """The task prompt plus the how-to-act framing (mirrors bird-sql's SQL framing)."""
    return task.prompt + _INSTRUCTIONS


class AppWorldAgent:
    """CaptureAgent that drives an AppWorldEnv through Bedrock converse tool-use.

    Faithful to AppWorld's real action space: the model's only environment action is a block of
    Python (the ``execute_python`` tool) run against the live world; it ends by calling ``finish``
    after signalling completion in-world with ``apis.supervisor.complete_task``. Throttling is
    retried with linear backoff; other errors propagate so ``run_capture`` can isolate the task.
    """

    def __init__(
        self,
        model_id: str,
        *,
        client: ConverseClient | None = None,
        region: str = "us-east-1",
        max_steps: int = 20,
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
