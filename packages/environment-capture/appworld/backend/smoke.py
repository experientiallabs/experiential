"""End-to-end smoke check of the AppWorld adapter against a REAL world (no Bedrock, no model cost).

Runs under the MAIN workspace interpreter (``uv run``): the adapter is appworld-free and launches
the real ``backend/world_backend.py`` under the benchmark venv. A tiny scripted "agent" does a task
by hand — authenticate, answer, ``complete_task`` — then the adapter grades it via AppWorld's own
tests. This validates the serve/execute/grade plumbing before a capture run.

    uv run python packages/environment-capture/appworld/backend/smoke.py
"""

from __future__ import annotations

from pathlib import Path

from environment_capture.benchmarks.appworld import AppWorldAdapter

_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    adapter = AppWorldAdapter(_ROOT, experiment_prefix="smoke")
    task = adapter.tasks("train")[0]
    print(f"task {task.task_id} ({task.data['appworld_id']}): {task.prompt}")

    env = adapter.open_env(task)
    try:
        # A minimal real interaction: read the supervisor's own info, then complete the task.
        info = env.execute("print(apis.supervisor.show_profile())")
        print("profile head:", info.output[:120].replace("\n", " "))
        done = env.execute("apis.supervisor.complete_task(answer='smoke test answer')")
        print("completed flag after complete_task:", env.completed, "| error:", done.returncode)
    finally:
        env.close()

    reward = adapter.grade(task, "smoke test answer")
    print(f"grade reward = {reward}")


if __name__ == "__main__":
    main()
