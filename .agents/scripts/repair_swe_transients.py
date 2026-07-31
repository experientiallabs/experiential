"""Rescue cells whose episode SURVIVED a transient provider fault, without re-buying them.

The runner records the first provider exception of a cell and, if one occurred, leaves the cell
unscored: an infrastructure fault is not a verdict. That rule is right when the fault ENDED the
episode and too blunt when it did not. Measured on this cohort: OpenRouter occasionally returns a
truncated JSON body (`JSONDecodeError: Expecting value: line ... column 1`), litellm retries it, and
the episode goes on to reach a terminal harness status anyway. Four cells recorded such a fault; two
then exhausted the step budget with no diff (a candidate failure), one submitted a real 507-char
patch (a scorable outcome nobody scored), and one ended with no status at all (the fault plausibly
did end it).

So the rule this pass applies, per cell rather than per class:

- The episode reached a TERMINAL harness status (`Submitted` or `LimitsExceeded`), which means the
  transport blip was recovered from and did not decide the outcome. Score it: an empty patch is 0.0
  (the candidate submitted nothing), and a real patch is handed to the official verifier here, which
  costs no model spend at all because the patch is already on disk.
- The episode ended with no status (`unknown`). Leave it UNSCORED: the fault is the best available
  explanation for why it stopped, and inventing a verdict would be worse than a reported hole.

Nothing here re-runs an episode and nothing here touches a cell whose reward is already set.

    uv run python .agents/scripts/repair_swe_transients.py <grid-dir>
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_swe_grid import CellKey, verify_patch  # noqa: E402

logger = logging.getLogger("repair_swe_transients")

TRANSIENT_PROVIDER_SIGNS = ("JSONDecodeError",)
"""Provider faults known to be transport-level, which litellm retries transparently."""

TERMINAL_STATUSES = ("Submitted", "LimitsExceeded")
"""Harness exit statuses that prove the episode ran to its own end."""

SWE_DIR = Path(
    "/Users/silen/Desktop/Projects/world-model-harness/packages/environment-capture/swe-bench"
)


def _patch_of(cell_dir: Path, instance_id: str) -> str:
    """The prediction the episode recorded, or "" when it recorded none."""
    preds = cell_dir / "agent" / "preds.json"
    if not preds.exists():
        return ""
    try:
        rows = json.loads(preds.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ""
    row = rows.get(instance_id)
    return (row or {}).get("model_patch") or "" if isinstance(row, dict) else ""


def main(grid_dir: Path) -> int:
    """Rescue what is rescuable in `grid_dir`; report every decision."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rescued = left = 0
    for outcome_path in sorted(grid_dir.glob("cells/*/*/ep*/outcome.json")):
        outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
        error = outcome.get("error") or ""
        if outcome.get("reward") is not None:
            continue
        if not any(sign in error for sign in TRANSIENT_PROVIDER_SIGNS):
            continue
        cell_dir = outcome_path.parent
        status = outcome.get("stop_reason") or "unknown"
        label = f"{outcome['model']} {outcome['scenario_id']} ep{outcome['episode']}"
        if status not in TERMINAL_STATUSES:
            logger.info("%s: LEFT UNSCORED (no terminal status: %s)", label, status)
            left += 1
            continue
        patch = _patch_of(cell_dir, outcome["scenario_id"])
        if not patch.strip():
            outcome["reward"] = 0.0
            outcome["success"] = False
            outcome["critique"] = f"no diff submitted (harness exit: {status}); survived {error}"
            outcome["error"] = None
            logger.info("%s: SCORED 0.0 (terminal %s, empty patch)", label, status)
        else:
            cell = CellKey(
                model=outcome["model"],
                instance_id=outcome["scenario_id"],
                episode=outcome["episode"],
            )
            resolved, detail = verify_patch(
                cell,
                alias=f"repair-{cell.slug}",
                cell_dir=cell_dir,
                patch=patch,
                swe_dir=SWE_DIR,
                timeout_s=1800.0,
            )
            if resolved is None:
                logger.info("%s: LEFT UNSCORED (verifier: %s)", label, detail)
                left += 1
                continue
            outcome["reward"] = 1.0 if resolved else 0.0
            outcome["success"] = resolved
            outcome["critique"] = f"swe-bench test suite: {detail}; survived {error}"
            outcome["error"] = None
            logger.info("%s: SCORED %.1f (%s)", label, outcome["reward"], detail)
        outcome_path.write_text(json.dumps(outcome, indent=2), encoding="utf-8")
        rescued += 1
    logger.info("rescued %d, left unscored %d", rescued, left)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
