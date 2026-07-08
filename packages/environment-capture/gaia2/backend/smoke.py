"""End-to-end plumbing check for the GAIA2 world backend (real world, no Bedrock).

Boots the first train scenario through the gate-module client (:class:`Gaia2Adapter` /
:class:`Gaia2Env`), lists its tools, runs a read then a write against the live world, closes, and
grades the logged actions against the oracle. Run from the repo root:

    uv run python packages/environment-capture/gaia2/backend/smoke.py
"""

from __future__ import annotations

from pathlib import Path

from environment_capture.benchmarks.gaia2 import Gaia2Adapter

_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    adapter = Gaia2Adapter(_ROOT, experiment_prefix="smoke")
    task = adapter.tasks("train")[0]
    print(f"task {task.task_id} ({task.data.get('config')}): {task.prompt[:100]}...")
    print(f"oracle actions: {task.data.get('oracle')}")

    oracle = task.data.get("oracle") or []
    env = adapter.open_env(task)
    try:
        docs = env.execute("print(describe_tools())")
        print(f"\n[describe_tools] rc={docs.returncode}, {len(docs.output.splitlines())} tools")
        # Replay the oracle actions directly to validate the log->grade loop end to end.
        for action in oracle:
            call = "tools[{name!r}](**{args!r})".format(
                name=f"{action['app']}__{action['function']}", args=action["args"]
            )
            res = env.execute(f"print({call})")
            print(f"[replay {action['function']}] rc={res.returncode}: {res.output[:120]}")
    finally:
        env.close()

    reward = adapter.grade(task, "")
    print(f"\ngrade after replaying the oracle actions = {reward} (expected 1.0)")


if __name__ == "__main__":
    main()
