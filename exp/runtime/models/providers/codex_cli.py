"""The ``codex-cli`` provider: Astra (``gpt-6-astra``) through the local Codex CLI, no tools.

A catalog connection ``provider = "codex-cli"`` makes every alias on it (``[models.astra]``)
resolve to an OpenAI-compatible client whose transport is :class:`CodexCliTransport`. The client
builds the usual chat request; the transport turns it into one ``codex exec`` call:

- the system and user messages become the prompt, written to stdin;
- the forced tool's parameters become ``--output-schema`` (made strict: every property required,
  optional ones nullable, closed objects), and the answer is read from ``-o out.json`` and
  returned as that tool's call, so callers see an ordinary forced tool answer;
- ``codex exec`` runs with :data:`CODEX_FLAGS`: read-only sandbox, ephemeral, no user config or
  rules, approvals never, an empty environment for any shell, and every tool feature disabled
  (shell, unified exec, apps, browser, computer use, plugins, MCP-backed skills, image tools, web
  search, multi-agent, hooks), in a fresh empty temporary directory, under a timeout;
- ``--json`` events are parsed. Only the event and item types in :data:`ALLOWED_EVENTS` and
  :data:`ALLOWED_ITEMS` (messages and reasoning) are accepted: any other item, a command, a file
  change, a tool, MCP or web-search call, refuses the whole output (HTTP-style 422,
  ``codex_tool_use_refused``) and nothing of it is used.

Failures map onto the shared retry classifier: a timeout raises :class:`TimeoutError`, a
non-zero exit a status-less :class:`ProviderTransportError` (both transient), and a rate limit is
backed off inside the call, then raised as a 429. Slot files under a shared directory bound
concurrent Codex calls across threads and processes (:func:`set_codex_concurrency`, default 4).

Astra is approved for offline use only. :meth:`CodexCliTransport.post` refuses, before spawning
anything, unless the caller entered :func:`offline_closed_model`; only offline entry points
(the Astra prompt refinement, build-time Astra roles) enter it. The identity of a codex-cli
model is its snapshot: provider ``codex-cli``, the model, the catalog ``revision`` (the pinned
Codex CLI version, checked against ``codex --version`` before the first call) and the
capabilities digest (which holds the reasoning effort).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

from filelock import Timeout

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.common.core.locks import CoordinationLock
from exp.common.models import ModelCapabilities, ModelSnapshot
from exp.runtime.models.providers.admitted_transport import AdmittedJsonTransport
from exp.runtime.models.providers.openai_compatible import OpenAICompatibleClient
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    JsonHttpTransport,
    ProviderTransportError,
    RetryPolicy,
)

CODEX_CLI_PROVIDER = "codex-cli"
"""Catalog provider identifier of the local Codex CLI."""

CODEX_BASE_URL = "https://codex-cli.invalid/v1"
"""Placeholder endpoint the client addresses; nothing is ever sent to it."""

DISABLED_FEATURES: tuple[str, ...] = (
    "shell_tool",
    "unified_exec",
    "unified_exec_tty",
    "shell_snapshot",
    "apps",
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "in_app_browser",
    "computer_use",
    "code_mode_host",
    "multi_agent",
    "plugins",
    "remote_plugin",
    "skill_search",
    "skill_mcp_dependency_install",
    "image_generation",
    "view_image",
    "sleep_tool",
    "tool_suggest",
    "goals",
    "hooks",
    "workspace_dependencies",
)
"""Codex 0.166 features that expose a tool or run code; all are disabled."""

CODEX_FLAGS: tuple[str, ...] = (
    "--sandbox",
    "read-only",
    "--ephemeral",
    "--skip-git-repo-check",
    "--ignore-user-config",
    "--ignore-rules",
    "--json",
    "--color",
    "never",
    "-c",
    'approval_policy="never"',
    "-c",
    'web_search="disabled"',
    "-c",
    "tools.web_search=false",
    "-c",
    "tools.view_image=false",
    "-c",
    'shell_environment_policy.inherit="none"',
    "-c",
    "mcp_servers={}",
    *(flag for name in DISABLED_FEATURES for flag in ("--disable", name)),
)
"""Every ``codex exec`` flag besides the model, effort, schema and output file."""

ALLOWED_EVENTS = frozenset(
    {"thread.started", "turn.started", "turn.completed", "item.started", "item.updated",
     "item.completed", "turn.failed", "error"}
)  # fmt: skip
"""Event types a tool-free answer may emit; any other event refuses the output."""

ALLOWED_ITEMS = frozenset({"agent_message", "reasoning"})
"""Item types a tool-free answer may emit; any other item (a tool) refuses the output."""

NOTICE_ITEMS = frozenset({"error"})
"""Item types that are notices, not actions (Codex reports config or startup warnings as
``error`` items before the turn); they are kept as warnings and never refuse or fail a call."""

RATE_LIMIT_MARKERS = ("rate limit", "rate_limit", "429", "usage limit", "too many requests")
"""Lowercase markers of a rate-limited call in Codex's error events or stderr."""

DROPPED_KEYWORDS = (
    "default", "title", "minLength", "maxLength", "pattern", "format", "minItems", "maxItems",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
)  # fmt: skip
"""Schema keywords a strict output schema leaves out; the caller's host checks still apply."""

_OFFLINE: ContextVar[str | None] = ContextVar("offline_closed_model", default=None)
_SLOTS: dict[str, object] = {
    "limit": 4,
    "directory": Path(tempfile.gettempdir()) / "exp-codex-slots",
}
_VERSIONS: dict[str, str] = {}


class OfflineScopeRequired(RuntimeError):
    """A closed offline-only model was called outside :func:`offline_closed_model`."""


@contextmanager
def offline_closed_model(purpose: str) -> Iterator[None]:
    """Allow codex-cli calls inside this block, naming the offline purpose."""
    token = _OFFLINE.set(purpose)
    try:
        yield
    finally:
        _OFFLINE.reset(token)


def offline_purpose() -> str | None:
    """The current offline purpose, or None on every other path."""
    return _OFFLINE.get()


def set_codex_concurrency(limit: int, directory: Path | None = None) -> None:
    """Set the bound on concurrent Codex calls across every process sharing ``directory``."""
    if limit < 1:
        raise ValueError("the Codex concurrency limit must be at least 1")
    _SLOTS["limit"] = limit
    if directory is not None:
        _SLOTS["directory"] = directory


@contextmanager
def codex_slot(poll_seconds: float = 0.05) -> Iterator[int]:
    """Hold one of the ``limit`` slot locks, waiting until one is free.

    The slots are lock files in a shared directory (:class:`CoordinationLock`), so the bound
    holds across threads and processes of this machine.
    """
    limit, directory = int(str(_SLOTS["limit"])), Path(str(_SLOTS["directory"]))
    while True:
        for index in range(limit):
            lock = CoordinationLock(str(directory / f"slot-{index}.lock"), thread_local=False)
            try:
                lock.acquire(blocking=False)
            except Timeout:
                continue
            try:
                yield index
            finally:
                lock.release()
            return
        time.sleep(poll_seconds)


@dataclass(frozen=True)
class CodexRun:
    """One finished ``codex exec`` process.

    Attributes:
        returncode: Exit status.
        stdout: The JSONL event stream.
        stderr: Diagnostics.
        output: Contents of the ``-o`` file, or None when it was not written.
    """

    returncode: int
    stdout: str
    stderr: str
    output: str | None


Runner = Callable[[Sequence[str], str, Path, float], CodexRun]
"""(argv, stdin, cwd, timeout) -> finished run; raises TimeoutError on timeout."""


def run_codex(argv: Sequence[str], stdin: str, cwd: Path, timeout: float) -> CodexRun:
    """Run one ``codex exec`` process; the ``-o`` file is read from its argv."""
    try:
        done = subprocess.run(
            list(argv), input=stdin, capture_output=True, text=True, cwd=cwd, timeout=timeout,
            check=False,
        )  # fmt: skip
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(f"codex exec exceeded {timeout:.0f}s") from error
    out = Path(argv[list(argv).index("-o") + 1])
    return CodexRun(
        done.returncode, done.stdout, done.stderr, out.read_text() if out.exists() else None
    )


def codex_version(binary: str) -> str:
    """The installed Codex CLI version (``codex --version``), cached per binary."""
    if binary not in _VERSIONS:
        done = subprocess.run(
            [binary, "--version"], capture_output=True, text=True, timeout=30, check=False
        )
        _VERSIONS[binary] = done.stdout.strip().rsplit(" ", 1)[-1]
    return _VERSIONS[binary]


def strict_schema(schema: JsonObject) -> tuple[JsonObject, dict[str, set[str]]]:
    """Inline references and close every object for ``--output-schema``.

    Returns:
        The strict schema and, per object path, the properties that were optional (made
        nullable), whose nulls :func:`drop_optional_nulls` removes from the answer.
    """
    definitions = schema.get("$defs", {})
    optional: dict[str, set[str]] = {}
    dropped = set(DROPPED_KEYWORDS)

    def walk(node: JsonValue, path: str) -> JsonValue:
        if isinstance(node, list):
            return [walk(item, path) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node and isinstance(definitions, dict):
            name = str(node["$ref"]).rsplit("/", 1)[-1]
            return walk(deepcopy(definitions[name]), path)
        out: JsonObject = {}
        for key, value in node.items():
            if key in dropped or key == "$defs":
                continue
            if key == "properties" and isinstance(value, dict):
                out[key] = {name: walk(item, f"{path}/{name}") for name, item in value.items()}
            elif key == "items":
                out[key] = walk(value, f"{path}/*")
            else:
                out[key] = walk(value, path)
        properties = out.get("properties")
        if out.get("type") == "object" and isinstance(properties, dict):
            listed = out.get("required")
            required = {str(n) for n in listed} if isinstance(listed, list) else set()
            missing = set(properties) - required
            optional[path] = missing
            for name in missing:
                properties[name] = {"anyOf": [properties[name], {"type": "null"}]}
            out["required"] = sorted(properties)
            out["additionalProperties"] = False
        return out

    strict = walk(deepcopy(schema), "")
    assert isinstance(strict, dict)
    return strict, optional


def drop_optional_nulls(
    value: JsonValue, optional: Mapping[str, set[str]], path: str = ""
) -> JsonValue:
    """Remove the nulls the strict schema forced onto originally optional properties."""
    if isinstance(value, list):
        return [drop_optional_nulls(item, optional, f"{path}/*") for item in value]
    if not isinstance(value, dict):
        return value
    names = optional.get(path, set())
    return {
        key: drop_optional_nulls(item, optional, f"{path}/{key}")
        for key, item in value.items()
        if not (item is None and key in names)
    }


def _text(content: JsonValue) -> str:
    """A message's text, from a string or a list of text parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return ""


def codex_prompt(payload: JsonObject) -> tuple[str, JsonObject, str]:
    """The stdin prompt, the output schema and the forced tool name of one chat request.

    Raises:
        ProviderTransportError: The request forces no single function tool.
    """
    tools = payload.get("tools")
    if not isinstance(tools, list) or len(tools) != 1 or not isinstance(tools[0], dict):
        raise ProviderTransportError(
            "codex-cli serves forced single-tool answers only", status_code=400,
            error_code="codex_unsupported_request",
        )  # fmt: skip
    function = tools[0].get("function")
    assert isinstance(function, dict)
    name, parameters = str(function["name"]), function.get("parameters") or {}
    assert isinstance(parameters, dict)
    messages = payload.get("messages")
    parts = []
    for message in messages if isinstance(messages, list) else []:
        if isinstance(message, dict):
            parts.append(
                f"<{message.get('role')}>\n{_text(message.get('content'))}\n</{message.get('role')}>"
            )
    prompt = (
        "You are answering one structured request. Do not run commands, read files, browse, "
        "or call any tool: answer from the text below only.\n\n"
        + "\n\n".join(parts)
        + f"\n\nReply with exactly one JSON object: the arguments of `{name}` "
        f"({function.get('description', '')}), matching the output schema."
    )
    return prompt, parameters, name


@dataclass
class EventSummary:
    """What a ``--json`` event stream showed.

    Attributes:
        types: Every event type, with item types as ``item.completed:<type>``, in order.
        tools: Events that show a tool or command; non-empty refuses the output.
        errors: Error and failed-turn messages.
        warnings: Notice items (``error`` items), kept for diagnosis.
        usage: Token usage of the completed turn.
    """

    types: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)


def parse_events(stdout: str) -> EventSummary:
    """Classify every JSONL event; unknown events and non-message items count as tools."""
    summary = EventSummary()
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            summary.tools.append(f"unparseable event {line[:80]!r}")
            continue
        kind = str(event.get("type", "")) if isinstance(event, dict) else ""
        item = event.get("item") if isinstance(event, dict) else None
        item_type = str(item.get("type", "")) if isinstance(item, dict) else None
        summary.types.append(kind if item_type is None else f"{kind}:{item_type}")
        if kind not in ALLOWED_EVENTS:
            summary.tools.append(kind or "untyped event")
        elif item_type in NOTICE_ITEMS:
            summary.warnings.append(json.dumps(item)[:400])
        elif item_type is not None and item_type not in ALLOWED_ITEMS:
            summary.tools.append(f"{kind}:{item_type}")
        if kind in ("error", "turn.failed"):
            summary.errors.append(json.dumps(event)[:400])
        usage = event.get("usage") if isinstance(event, dict) else None
        if kind == "turn.completed" and isinstance(usage, dict):
            summary.usage = {k: int(v) for k, v in usage.items() if isinstance(v, int)}
    return summary


class CodexCliTransport(JsonHttpTransport):
    """Serve forced chat requests with ``codex exec``; refuse any output that used a tool.

    Attributes:
        model: The Codex model (``gpt-6-astra``).
        effort: Reasoning effort passed as ``model_reasoning_effort``.
        version: The pinned Codex CLI version; a different installed version is refused.
        binary: The ``codex`` executable.
        timeout_seconds: Wall-clock bound of one call.
        rate_limit_retries: Backed-off reruns of a rate-limited call before raising a 429.
        backoff_seconds: First backoff; each rerun doubles it.
    """

    def __init__(
        self,
        *,
        model: str,
        effort: str | None,
        version: str | None,
        binary: str = "codex",
        timeout_seconds: float = 900.0,
        rate_limit_retries: int = 3,
        backoff_seconds: float = 10.0,
        runner: Runner | None = None,
        sleep: Callable[[float], None] | None = None,
        version_of: Callable[[str], str] | None = None,
    ) -> None:
        """Bind the model, the pinned version and the process runner.

        The runner, sleep and version reader default to this module's :func:`run_codex`,
        :func:`time.sleep` and :func:`codex_version`, read when the transport is built.
        """
        self.model, self.effort, self.version = model, effort, version
        self.binary = shutil.which(binary) or binary
        self.timeout_seconds, self.rate_limit_retries = timeout_seconds, rate_limit_retries
        self.backoff_seconds = backoff_seconds
        self._runner = runner if runner is not None else run_codex
        self._sleep = sleep if sleep is not None else time.sleep
        self._version_of = version_of if version_of is not None else codex_version

    def get(
        self, url: str, *, headers: Mapping[str, str], timeout_seconds: float
    ) -> JsonHttpResponse:
        """No metadata endpoint exists."""
        raise ProviderTransportError("codex-cli has no metadata endpoint", status_code=404)

    def argv(self, schema_path: Path, out_path: Path) -> list[str]:
        """The exact ``codex exec`` command line."""
        effort = [] if self.effort is None else ["-c", f'model_reasoning_effort="{self.effort}"']
        return [
            self.binary, "exec", "-m", self.model, *effort, *CODEX_FLAGS,
            "--output-schema", str(schema_path), "-o", str(out_path), "-",
        ]  # fmt: skip

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Run one tool-free Codex call and return its answer as the forced tool's call.

        Raises:
            OfflineScopeRequired: Called outside :func:`offline_closed_model`.
            TimeoutError: The call exceeded its timeout (transient).
            ProviderTransportError: A non-zero exit (transient), a rate limit after backoff
                (429), a tool event (422 ``codex_tool_use_refused``) or an unusable answer.
        """
        if offline_purpose() is None:
            raise OfflineScopeRequired(
                "codex-cli serves a closed model approved for offline use only; enter "
                "offline_closed_model() at an offline entry point"
            )
        if self.version is not None and self._version_of(self.binary) != self.version:
            raise ProviderTransportError(
                f"installed codex is not the pinned {self.version}", status_code=412,
                error_code="codex_version_changed",
            )  # fmt: skip
        prompt, parameters, name = codex_prompt(payload)
        strict, optional = strict_schema(parameters)
        attempt = 0
        while True:
            with codex_slot(), tempfile.TemporaryDirectory(prefix="codex-astra-") as scratch:
                root = Path(scratch)
                work = root / "work"
                work.mkdir()
                schema_path, out_path = root / "schema.json", root / "out.json"
                schema_path.write_text(json.dumps(strict))
                run = self._runner(
                    self.argv(schema_path, out_path), prompt, work, self.timeout_seconds
                )
                if out_path.exists():
                    run = CodexRun(run.returncode, run.stdout, run.stderr, out_path.read_text())
            events = parse_events(run.stdout)
            if events.tools:
                raise ProviderTransportError(
                    f"codex used a tool or emitted an unknown event: {events.tools[:5]}",
                    status_code=422, error_code="codex_tool_use_refused",
                )  # fmt: skip
            failure = " ".join([*events.errors, run.stderr[-400:]]).lower()
            if run.returncode != 0 or events.errors:
                if any(marker in failure for marker in RATE_LIMIT_MARKERS):
                    if attempt < self.rate_limit_retries:
                        self._sleep(self.backoff_seconds * 2**attempt)
                        attempt += 1
                        continue
                    raise ProviderTransportError("codex is rate limited", status_code=429)
                raise ProviderTransportError(f"codex exec exited {run.returncode}")
            return self._answer(run, events, name, optional)

    def _answer(
        self, run: CodexRun, events: EventSummary, name: str, optional: Mapping[str, set[str]]
    ) -> JsonHttpResponse:
        """The chat-completion body carrying the answer as the forced tool's call."""
        try:
            answer = json.loads(run.output or "")
        except ValueError as error:
            raise ProviderTransportError(
                "codex wrote no JSON answer", status_code=502, error_code="codex_no_answer"
            ) from error
        if not isinstance(answer, dict):
            raise ProviderTransportError(
                "codex answer is not an object", status_code=502, error_code="codex_no_answer"
            )
        arguments = drop_optional_nulls(answer, optional)
        prompt_tokens = events.usage.get("input_tokens", 0)
        completion = events.usage.get("output_tokens", 0)
        return JsonHttpResponse(
            status_code=200,
            body={
                "id": "codex-cli",
                "model": self.model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "id": "codex-call-0",
                                    "function": {"name": name, "arguments": json.dumps(arguments)},
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion,
                    "total_tokens": prompt_tokens + completion,
                },
            },
        )


def codex_cli_client(
    snapshot: ModelSnapshot, capabilities: ModelCapabilities, bound: object
) -> OpenAICompatibleClient:
    """The client of one codex-cli alias: one attempt per call, Codex as the only transport.

    ``bound`` is the registry's transport for the alias. An admitted transport keeps its
    attempt receipts, attempt cap, deadline and host admission, over Codex instead of HTTP; any
    other transport is ignored, so a codex-cli alias can never reach a network endpoint.
    """
    codex = CodexCliTransport(
        model=snapshot.model_id, effort=capabilities.reasoning_effort, version=snapshot.revision
    )
    transport = bound.over(codex) if isinstance(bound, AdmittedJsonTransport) else codex
    return OpenAICompatibleClient(
        model=snapshot,
        api_key="codex-cli-local",
        base_url=CODEX_BASE_URL,
        transport=transport,
        retry_policy=RetryPolicy(maximum_attempts=1),
        timeout_seconds=codex.timeout_seconds,
        supports_temperature=False,
        supports_reasoning=False,
    )
