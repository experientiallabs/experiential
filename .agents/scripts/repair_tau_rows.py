"""Repair a tau real-episode rows.jsonl in place from persisted records. No episode is re-bought.

Two corrections, both rulings that landed after this grid was measured (DECISIONS, 2026-07-29):

1. STEP UNIT IS BILLED PROVIDER CALLS. Rows recorded `steps` as the count of assistant turns that
   called a tool. That is not the unit the `max_turns` cap enforces and not the unit cost scales
   with: tau2 opens each episode with a scripted greeting nobody paid for, and plenty of billed
   turns only talk to the user. Across this grid the real call volume was 1.48x the recorded
   number, and 3x on the conversational candidates. Recomputed from each episode's tau2
   `results.json` as the number of assistant messages carrying `usage`, which is exactly when a
   completion was purchased.

2. A CANDIDATE-CAUSED FAILURE IS A SCORED ZERO. An episode that died because the candidate
   emitted an assistant message with neither content nor a tool call failed the task by the
   benchmark's own semantics; tau2 would have scored it 0 had it limped as far as scoring. Those
   rows become reward=0.0, keeping their error text. `reward=None` stays reserved for
   INFRASTRUCTURE (429, auth, timeout, provider refusal), which is excluded-and-reported.

Cost is untouched by both: it is computed from token usage summed over every assistant message,
which neither correction changes.

The original file is preserved beside the repaired one as `rows.jsonl.prerepair` so the raw
evidence survives, and every changed row records what changed in `repairs`.

    uv run python .agents/scripts/repair_tau_rows.py --rows <rows.jsonl> --capture-dir <dir>
    uv run python .agents/scripts/repair_tau_rows.py --rows <rows.jsonl> --capture-dir <dir> --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

_RL = Path(__file__).resolve().parents[1].parent / "packages/environment-capture/tau-bench/rl"
sys.path.insert(0, str(_RL))

from real_episodes import (  # noqa: E402  (path shim; the runner is not an installed module)
    save_to_name,
)

logger = logging.getLogger("tau-repair")

# The candidate's own output, not our infrastructure: tau2's own message validation rejected what
# the model returned. Matched on the exception's message because tau2 collapses every failure into
# one termination_reason.
CANDIDATE_CAUSED_SIGNATURES = ("must have either content or tool_calls",)


def is_candidate_caused(error: str | None) -> bool:
    """Whether an unscored episode failed on the candidate's own output."""
    return bool(error) and any(sig in error for sig in CANDIDATE_CAUSED_SIGNATURES)


def billed_calls(capture_dir: Path, model: str, domain: str, episode: int, task_id: str) -> int | None:
    """Assistant messages carrying usage for one episode, across that cell's attempt dirs.

    Args:
        capture_dir: Directory holding `tau2-bench/data/simulations`.
        model: Pool entry name.
        domain: tau2 domain.
        episode: Episode index.
        task_id: tau2 task id.

    Returns:
        The billed-call count from the LATEST attempt that recorded this task, or None when no
        save directory still holds it.
    """
    sims = capture_dir / "tau2-bench" / "data" / "simulations"
    found: int | None = None
    for attempt in range(0, 100):
        results = sims / save_to_name_for(model, domain, episode, attempt) / "results.json"
        if not results.is_file():
            continue
        try:
            payload = json.loads(results.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for sim in payload.get("simulations") or []:
            if sim.get("task_id") != task_id:
                continue
            messages = sim.get("messages") or []
            found = sum(
                1
                for message in messages
                if message.get("role") == "assistant" and message.get("usage") is not None
            )
    return found


def save_to_name_for(model: str, domain: str, episode: int, attempt: int) -> str:
    """The runner's save-directory name, reusing its own slug rule."""

    class _Entry:
        name = model

    return save_to_name(_Entry(), domain, episode, attempt)  # type: ignore[arg-type]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    raw = [json.loads(line) for line in args.rows.read_text(encoding="utf-8").splitlines() if line.strip()]
    steps_fixed = 0
    steps_unrecoverable = 0
    rescored = 0
    for row in raw:
        repairs: list[str] = []
        calls = billed_calls(
            args.capture_dir, row["model"], row["domain"], int(row["episode"]), row["task_id"]
        )
        if calls is None:
            steps_unrecoverable += 1
        elif calls != row.get("steps"):
            repairs.append(f"steps {row.get('steps')} -> {calls} (billed provider calls)")
            row["steps"] = calls
            steps_fixed += 1
        if row.get("reward") is None and is_candidate_caused(row.get("error")):
            repairs.append("reward None -> 0.0 (candidate-caused failure is a scored zero)")
            row["reward"] = 0.0
            rescored += 1
        if repairs:
            row["repairs"] = [*row.get("repairs", []), *repairs]

    logger.info(
        "steps corrected on %d row(s); %d row(s) had no surviving save dir; %d unscored row(s) "
        "rescored to 0 as candidate-caused",
        steps_fixed,
        steps_unrecoverable,
        rescored,
    )
    if args.dry_run:
        logger.info("dry run: nothing written")
        return 0
    backup = args.rows.with_suffix(args.rows.suffix + ".prerepair")
    if not backup.exists():
        shutil.copy2(args.rows, backup)
        logger.info("raw evidence preserved at %s", backup)
    args.rows.write_text(
        "".join(json.dumps(row) + "\n" for row in raw), encoding="utf-8"
    )
    logger.info("rewrote %s (%d rows)", args.rows, len(raw))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
