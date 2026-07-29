"""Repair `steps` on already-measured cells from their persisted per-call record.

No episode is re-bought: the exact call count is in each cell's cell.json (call_usage) and in
outcome.json's call_seconds, so the corrected value is derivable offline. This is the repair the
per-call persistence exists for.
"""

import json, sys
from pathlib import Path

base = Path(sys.argv[1])
fixed = unchanged = 0
for outcome_path in sorted(base.glob("cells/*/*/ep*/outcome.json")):
    outcome = json.loads(outcome_path.read_text())
    calls = len(outcome.get("call_seconds") or [])
    if calls and outcome.get("steps") != calls:
        before = outcome["steps"]
        outcome["steps"] = calls
        outcome_path.write_text(json.dumps(outcome, indent=2))
        print(f"  {outcome['model']:16s} {outcome['scenario_id']:26s} steps {before} -> {calls}")
        fixed += 1
    else:
        unchanged += 1
print(f"repaired {fixed}, unchanged {unchanged}")
