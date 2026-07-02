"""Out-of-process AppWorld engine, run under the benchmark-local venv (Python 3.11 + ``appworld``).

The gate-checked ``environment_capture.benchmarks.appworld`` module never imports the heavy
``appworld`` engine; it shells out to this script instead. Two subcommands:

* ``serve <task_id> <experiment_name>`` boots ONE live ``AppWorld`` for the task and speaks a
  line-delimited JSON protocol on stdio — one ``{"op": ...}`` request per line in, one response per
  line out — keeping the world stateful across many ``execute`` calls (the whole point of AppWorld):
    - ``{"op": "execute", "code": "<python>"}`` -> ``{"output", "error", "completed"}``
    - ``{"op": "close"}`` -> shuts the world down and exits.
  AppWorld chatters on real stdout during boot/execution, so this process redirects the engine's
  stdout to stderr and writes the protocol JSON to the original stdout fd, keeping it clean.

* ``grade <task_id> <experiment_name>`` runs AppWorld's own deterministic evaluation tests over the
  world the agent left behind and prints ``{"reward", "success", "num_tests"}`` — reward is the
  fraction of tests that pass (no LLM).

This file imports ``appworld`` and is excluded from the repo's type gate (like other heavy-dep
scripts); it is exercised end-to-end by ``backend/smoke.py`` under the venv.
"""

from __future__ import annotations

import json
import sys

# AppWorld prints setup/usage banners to stdout; move its stdout to stderr and keep the ORIGINAL
# stdout as the clean protocol channel before importing/booting anything.
_PROTOCOL_OUT = sys.stdout
sys.stdout = sys.stderr

from appworld import AppWorld, evaluate_task  # noqa: E402  (after the stdout redirect above)

_EXECUTION_ERROR_MARKER = "Execution failed"


def _emit(payload: dict[str, object]) -> None:
    _PROTOCOL_OUT.write(json.dumps(payload) + "\n")
    _PROTOCOL_OUT.flush()


def _serve(task_id: str, experiment_name: str) -> None:
    try:
        world = AppWorld(task_id=task_id, experiment_name=experiment_name)
        world.__enter__()
    except Exception as error:  # noqa: BLE001 - report boot failure over the protocol, then exit
        _emit({"ready": False, "error": f"{type(error).__name__}: {error}"})
        return
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
            try:
                output = world.execute(code)
            except Exception as error:  # noqa: BLE001 - a shell error is a normal observation
                output = f"{_EXECUTION_ERROR_MARKER}. {type(error).__name__}: {error}"
            _emit(
                {
                    "output": output,
                    "error": output.startswith(_EXECUTION_ERROR_MARKER),
                    "completed": bool(world.task_completed()),
                }
            )
            continue
        _emit({"output": f"unknown op: {op!r}", "error": True, "completed": False})

    world.close()
    world.__exit__(None, None, None)


def _grade(task_id: str, experiment_name: str) -> None:
    tracker = evaluate_task(
        task_id=task_id,
        experiment_name=experiment_name,
        suppress_errors=True,
        save_report=False,
    )
    report = tracker.to_dict()
    num_tests = int(report["num_tests"])
    reward = (len(report["passes"]) / num_tests) if num_tests else 0.0
    _emit({"reward": reward, "success": bool(report["success"]), "num_tests": num_tests})


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "serve" and len(sys.argv) == 4:
        _serve(sys.argv[2], sys.argv[3])
    elif command == "grade" and len(sys.argv) == 4:
        _grade(sys.argv[2], sys.argv[3])
    else:
        sys.stderr.write("usage: world_backend.py {serve|grade} <task_id> <experiment_name>\n")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
