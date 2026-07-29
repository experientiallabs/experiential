"""Repair `steps` on already-measured cells from their persisted per-call record.

No episode is re-bought: the exact call count is in each cell's cell.json (call_usage) and in
outcome.json's call_seconds, so the corrected value is derivable offline. This is the repair the
per-call persistence exists for.
"""

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("repair_swe_steps")

logging.basicConfig(level=logging.INFO, format="%(message)s")
base = Path(sys.argv[1])
fixed = unchanged = 0
for outcome_path in sorted(base.glob("cells/*/*/ep*/outcome.json")):
    outcome = json.loads(outcome_path.read_text())
    calls = len(outcome.get("call_seconds") or [])
    if calls and outcome.get("steps") != calls:
        before = outcome["steps"]
        outcome["steps"] = calls
        outcome_path.write_text(json.dumps(outcome, indent=2))
        logger.info("%s %s steps %d -> %d", outcome["model"], outcome["scenario_id"], before, calls)
        fixed += 1
    else:
        unchanged += 1
logger.info("repaired %d, unchanged %d", fixed, unchanged)
