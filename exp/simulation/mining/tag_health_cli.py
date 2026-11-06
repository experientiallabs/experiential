"""Read-only corpus tag-health report over saved tag-matrix files, with no model call.

Run it as soon as a bank finishes tagging::

    python -m exp.simulation.mining.tag_health_cli MATRIX.json [MATRIX.json ...]
        [--excluded EXCLUDED.json] [--json OUT.json] [--top N]

Each ``MATRIX.json`` is one saved :class:`~exp.simulation.mining.tagging.TagMatrix` (for example
``<bank>.<view>.json``). The command opens every input read-only, computes the checkpoint-1
statistics of :mod:`~exp.simulation.mining.tag_health` (fire rates, always-yes and always-no
questions, near-duplicate pairs by answer agreement, abstain and low-confidence rates, per-bank
coverage), prints a summary, and writes the full statistics to ``--json`` only when that path is
given. ``--excluded`` names a JSON record of traces the tagger excluded at the credential
boundary: an object mapping a ``<bank>.<view>`` key or a view to a count or a list of traces, or a
list of entries (objects with a ``view`` are counted per view, anything else under ``*``).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.table import Table

from exp.common.core.artifacts import JsonValue
from exp.simulation.mining.tag_health import CorpusHealth, HealthThresholds, corpus_health
from exp.simulation.mining.tagging import TagMatrix


def read_matrix(path: Path) -> TagMatrix:
    """Read one saved matrix without modifying it."""
    with path.open("rb") as stream:
        return TagMatrix.model_validate_json(stream.read())


def read_excluded(path: Path) -> dict[str, int]:
    """Count excluded traces by ``<bank>.<view>`` key or view from a JSON record."""
    with path.open("rb") as stream:
        payload: JsonValue = json.loads(stream.read())
    counts: dict[str, int] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                counts[key] = value
            elif isinstance(value, list):
                counts[key] = len(value)
    elif isinstance(payload, list):
        for entry in payload:
            view = entry.get("view") if isinstance(entry, dict) else None
            key = view if isinstance(view, str) else "*"
            counts[key] = counts.get(key, 0) + 1
    return counts


def summary_lines(health: CorpusHealth, top: int) -> list[str]:
    """Plain summary lines: per bank, then flagged questions and near-duplicate pairs."""
    lines: list[str] = []
    for bank in health.banks:
        excluded = "unknown" if bank.excluded is None else str(bank.excluded)
        lines.append(
            f"{bank.bank_id}.{bank.view}: {bank.traces} traces x {bank.questions} questions; "
            f"coverage {bank.coverage:.3f}; mean fired {bank.mean_fired:.2f}; flagged "
            f"{bank.flagged} ({bank.flagged_share:.2f}); near-duplicate pairs {bank.duplicates}; "
            f"credential-boundary exclusions {excluded}"
        )
        rows = [q for q in health.questions if q.bank_id == bank.bank_id and q.view == bank.view]
        for flag in ("always_yes", "always_no", "near_constant", "low_confidence"):
            named = [q for q in rows if flag in q.flags]
            if named:
                shown = ", ".join(f"{q.question_id} ({q.fire_rate:.3f})" for q in named[:top])
                more = f" and {len(named) - top} more" if len(named) > top else ""
                lines.append(f"  {flag}: {len(named)}: {shown}{more}")
        pairs = [p for p in health.pairs if p.bank_id == bank.bank_id and p.view == bank.view]
        for pair in pairs[:top]:
            lines.append(
                f"  near_duplicate: {pair.left} ~ {pair.right} agreement {pair.agreement:.3f} "
                f"jaccard {pair.jaccard:.3f} correlation {pair.correlation:.3f}"
            )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, compute the statistics and present them.

    Returns:
        The process exit status.
    """
    parser = argparse.ArgumentParser(description="Read-only corpus tag-health report.")
    parser.add_argument("matrices", nargs="+", type=Path, help="saved tag-matrix JSON files")
    parser.add_argument("--excluded", type=Path, help="credential-boundary exclusion record")
    parser.add_argument("--json", type=Path, dest="output", help="write the full statistics")
    parser.add_argument("--top", type=int, default=10, help="flagged rows shown per flag")
    parser.add_argument("--duplicate-agreement", type=float, default=None)
    arguments = parser.parse_args(argv)
    thresholds = HealthThresholds()
    if arguments.duplicate_agreement is not None:
        thresholds = HealthThresholds(duplicate_agreement=arguments.duplicate_agreement)
    matrices = [read_matrix(path) for path in arguments.matrices]
    excluded = None if arguments.excluded is None else read_excluded(arguments.excluded)
    health = corpus_health(matrices, thresholds, excluded=excluded)
    console = Console()
    table = Table(title="Corpus tag health (checkpoint 1, no model call)")
    for column in ("bank.view", "traces", "questions", "coverage", "flagged", "pairs"):
        table.add_column(column)
    for bank in health.banks:
        table.add_row(
            f"{bank.bank_id}.{bank.view}",
            str(bank.traces),
            str(bank.questions),
            f"{bank.coverage:.3f}",
            f"{bank.flagged} ({bank.flagged_share:.2f})",
            str(bank.duplicates),
        )
    console.print(table)
    for line in summary_lines(health, arguments.top):
        console.print(line, markup=False, highlight=False)
    if arguments.output is not None:
        arguments.output.write_text(health.model_dump_json(indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
