"""Out-of-process GAIA2 world, run under the benchmark-local venv (ARE engine).

The gate-checked ``environment_capture.benchmarks.gaia2`` module never imports the heavy ARE engine;
it shells out to this script instead. One subcommand:

* ``serve <scenario_json_path> <state_out_path>`` imports ONE Gaia2 scenario, populates its apps'
  universe, and speaks a line-delimited JSON protocol on stdio — one ``{"op": ...}`` request per
  line in, one response per line out — keeping the world stateful across many ``execute`` calls:
    - ``{"op": "execute", "code": "<python>"}`` -> ``{"output", "error"}``. The code runs in a
      persistent namespace where ``tools`` (name -> app-tool callable) and ``describe_tools()`` are
      preloaded; every tool call is logged (app, function, args, write_operation).
    - ``{"op": "close"}`` -> dumps the agent's action log to ``state_out_path`` (JSON) and exits.

ARE chatters on real stdout during import/boot, so this process redirects the engine's stdout to
stderr and writes the protocol JSON to the original stdout fd, keeping it clean. The gate-module
grader reads ``state_out_path`` and matches the logged WRITE actions against the scenario's oracle
actions deterministically (no LLM). This file imports ARE and is excluded from the repo type gate;
it is exercised end-to-end by ``backend/smoke.py`` under the venv.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import json
import signal
import sys
import traceback
from types import FrameType

_PROTOCOL_OUT = sys.stdout
sys.stdout = sys.stderr

from are.simulation.data_handler.importer import JsonScenarioImporter  # noqa: E402

_ERROR_MARKER = "Execution failed"
_MAX_OUTPUT_CHARS = 6000  # keep a single observation (e.g. a full app dump) from bloating the trace
_EXECUTE_TIMEOUT_S = 30  # a runaway agent snippet (infinite loop / huge compute) must not wedge us


class _ExecTimeout(Exception):
    """Raised by the SIGALRM handler when one execute snippet overruns its wall-clock budget."""


def _on_alarm(signum: int, frame: FrameType | None) -> None:
    raise _ExecTimeout()


signal.signal(signal.SIGALRM, _on_alarm)


def _emit(payload: dict[str, object]) -> None:
    _PROTOCOL_OUT.write(json.dumps(payload, default=str) + "\n")
    _PROTOCOL_OUT.flush()


def _truncate(text: str) -> str:
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    return text[:_MAX_OUTPUT_CHARS] + f"\n... [truncated {len(text) - _MAX_OUTPUT_CHARS} chars]"


def _wrap_tool(tool: object, action_log: list[dict[str, object]]) -> object:
    """Wrap an AppTool callable so every invocation is logged (args bound to their names)."""
    app_name = getattr(tool, "app_name", "")
    func_name = getattr(tool, "func_name", getattr(tool, "name", ""))
    write_operation = bool(getattr(tool, "write_operation", False))
    bound = getattr(tool.class_instance, tool.func_name)  # type: ignore[attr-defined]
    try:
        param_names = [
            p for p in inspect.signature(bound).parameters if p not in ("self", "args", "kwargs")
        ]
    except (TypeError, ValueError):
        param_names = []

    def wrapper(*args: object, **kwargs: object) -> object:
        named = dict(zip(param_names, args, strict=False))
        named.update(kwargs)
        action_log.append(
            {
                "app": app_name,
                "function": func_name,
                "args": named,
                "write_operation": write_operation,
            }
        )
        return bound(*args, **kwargs)

    return wrapper


def _describe_tools(tools: dict[str, object]) -> str:
    lines: list[str] = []
    for name in sorted(tools):
        tool = tools[name]
        arg_names = [str(getattr(a, "name", a)) for a in (getattr(tool, "args", []) or [])]
        description = str(getattr(tool, "function_description", "") or "").split("\n")[0][:100]
        lines.append(f"{name}({', '.join(arg_names)}) - {description}")
    return "\n".join(lines)


def _serve(scenario_json_path: str, state_out_path: str) -> None:
    action_log: list[dict[str, object]] = []
    try:
        with open(scenario_json_path, encoding="utf-8") as handle:
            scenario_json = handle.read()
        scenario, _, _ = JsonScenarioImporter().import_from_json_to_benchmark(
            scenario_json, load_completed_events=False
        )
        scenario.initialize()
        raw_tools = {t.name: t for t in scenario.get_tools()}
        tools = {name: _wrap_tool(tool, action_log) for name, tool in raw_tools.items()}
    except Exception as error:  # noqa: BLE001 - report boot failure over the protocol, then exit
        _emit({"ready": False, "error": f"{type(error).__name__}: {error}"})
        return

    namespace: dict[str, object] = {
        "tools": tools,
        "describe_tools": lambda: _describe_tools(raw_tools),
    }
    _emit({"ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        op = request.get("op")
        if op == "close":
            break
        if op == "execute":
            code = str(request.get("code", ""))
            buffer = io.StringIO()
            error = False
            signal.alarm(_EXECUTE_TIMEOUT_S)
            try:
                with contextlib.redirect_stdout(buffer):
                    try:
                        value = eval(compile(code, "<world>", "eval"), namespace)  # noqa: S307
                        if value is not None:
                            print(repr(value), file=buffer)
                    except SyntaxError:
                        exec(compile(code, "<world>", "exec"), namespace)  # noqa: S102
            except _ExecTimeout:
                error = True
                buffer.write(f"{_ERROR_MARKER}. timed out after {_EXECUTE_TIMEOUT_S}s\n")
            except Exception as exc:  # noqa: BLE001 - a snippet error is a normal observation
                error = True
                buffer.write(f"{_ERROR_MARKER}. {type(exc).__name__}: {exc}\n")
                buffer.write("".join(traceback.format_exception_only(type(exc), exc)))
            finally:
                signal.alarm(0)
            _emit({"output": _truncate(buffer.getvalue().rstrip("\n")), "error": error})
            continue
        _emit({"output": f"unknown op: {op!r}", "error": True})

    with open(state_out_path, "w", encoding="utf-8") as handle:
        json.dump(action_log, handle, default=str)


def main() -> None:
    if len(sys.argv) == 4 and sys.argv[1] == "serve":
        _serve(sys.argv[2], sys.argv[3])
    else:
        sys.stderr.write("usage: world_backend.py serve <scenario_json_path> <state_out_path>\n")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
