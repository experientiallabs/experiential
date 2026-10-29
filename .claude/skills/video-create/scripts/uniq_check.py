#!/usr/bin/env python3
"""The no-repeat rule for style-match edits.

Lists every footage source used by more than one shot in a timeline written as
S('id', f0, f1, 'type', { ... }) rows (one row per line). Pass the shot ids where callbacks
are allowed (the reference's burst montage, PIP grids) with --allow. Every remaining repeat
must be one the reference also makes: continuous lip-synced lines, poster triplets, bookends.

Usage: uniq_check.py engine/film.js [--allow S63,S52]
"""

import collections
import re
import sys
from pathlib import Path

ROW = re.compile(r"S\('(\w+)', (\d+), (\d+), '(\w+)', \{(.*?)\}\);\n")
SOURCE = re.compile(r"\b(?:p|clip|cut): '(\w+)'")


def main() -> None:
    """Print reused sources and a summary line; exit 1 on bad usage."""
    if len(sys.argv) < 2:
        sys.stdout.write(str(__doc__))
        sys.exit(1)
    src = Path(sys.argv[1]).read_text()
    allow = (
        set(sys.argv[sys.argv.index("--allow") + 1].split(",")) if "--allow" in sys.argv else set()
    )
    uses: dict[str, list[str]] = collections.defaultdict(list)
    for m in ROW.finditer(src):
        sid, body = m.group(1), m.group(5)
        if sid in allow:
            continue
        for name in SOURCE.findall(body):
            uses[name].append(sid)
    reused = {k: v for k, v in uses.items() if len(set(v)) > 1}
    for name, shots in sorted(reused.items()):
        sys.stdout.write(f"REUSE {name} {shots}\n")
    sys.stdout.write(f"{len(uses)} sources; {len(reused)} reused\n")


if __name__ == "__main__":
    main()
