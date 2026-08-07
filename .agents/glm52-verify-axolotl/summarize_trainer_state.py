"""Create a compact milestone report from a Transformers trainer state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean


def main() -> None:
    """Write selected point metrics and nonoverlapping interval loss means."""
    parser = argparse.ArgumentParser()
    parser.add_argument("trainer_state", type=Path)
    parser.add_argument("--milestones", type=int, nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    state = json.loads(args.trainer_state.read_text(encoding="utf-8"))
    history = {
        int(item["step"]): item
        for item in state["log_history"]
        if "step" in item and "loss" in item
    }
    missing = set(args.milestones) - history.keys()
    if missing:
        raise ValueError(f"missing requested milestones: {sorted(missing)}")

    rows = []
    previous = 0
    for step in args.milestones:
        point = history[step]
        interval = [
            history[current]["loss"]
            for current in sorted(history)
            if previous < current <= step
        ]
        rows.append(
            {
                "step": step,
                "epoch": point["epoch"],
                "learning_rate": point["learning_rate"],
                "point_loss": point["loss"],
                "interval_mean_loss": fmean(interval),
                "grad_norm": point["grad_norm"],
                "ppl": point["ppl"],
                "tokens_total": point["tokens/total"],
                "tokens_supervised": point["tokens/trainable"],
            }
        )
        previous = step

    report = {
        "schema": "axolotl-training-milestones-v1",
        "source_trainer_state": str(args.trainer_state),
        "global_step": state["global_step"],
        "rows": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
