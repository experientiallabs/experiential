"""The prior-art map in AGENTS.md rule 16 names owners that exist."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _map_rows() -> list[str]:
    """The table rows of AGENTS.md rule 16."""
    text = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    rule = text[text.index("16. **Follow prior art") :]
    rule = rule[: rule.index("Keep this table current")]
    return [line for line in rule.splitlines() if line.strip().startswith("|") and "`exp/" in line]


def test_every_owner_in_the_prior_art_map_exists() -> None:
    """Moving or renaming an owner fails until the map names its new home."""
    rows = _map_rows()
    assert len(rows) >= 10, "the map lists the shared mechanisms"
    missing = [
        path
        for row in rows
        for path in re.findall(r"`(exp/[\w/]+\.py)`", row.split("|")[2])
        if not (REPO_ROOT / path).is_file()
    ]
    assert not missing, f"prior-art map names missing owners: {missing}"


def test_every_named_entry_point_is_defined_in_its_owner() -> None:
    """Each function or class the map tells callers to use is defined in a module it names."""
    for row in _map_rows():
        cells = row.split("|")
        owners = [REPO_ROOT / p for p in re.findall(r"`(exp/[\w/]+\.py)`", cells[2] + cells[3])]
        source = "\n".join(owner.read_text(encoding="utf-8") for owner in owners)
        named = re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", cells[3])
        for name in named:
            assert re.search(rf"^(def|class) {name}\b|^{name}\b\s*[:=]", source, re.M), (
                f"{name} is not defined in {[o.name for o in owners]}"
            )
