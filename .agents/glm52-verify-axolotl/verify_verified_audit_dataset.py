"""Verify that admitted audit rows preserve complete judgment provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def nonempty_text(value: Any) -> bool:
    """Return whether a value is a nonempty string."""
    return isinstance(value, str) and bool(value.strip())


PROVENANCE_TEXT_FIELDS = (
    "judge_model_id",
    "judge_provider",
    "judge_region",
    "message_log_sha256",
    "prompt_sha256",
    "prompt_version",
    "request_id",
    "requested_at",
    "rollout_id",
    "source_corpus_revision",
    "source_corpus_sha256",
    "source_row_sha256",
    "task_id",
)


def provenance_complete(judgment: dict[str, Any], record: dict[str, Any]) -> bool:
    """Validate the judge schema's flattened provenance fields."""
    return (
        all(nonempty_text(judgment.get(field)) for field in PROVENANCE_TEXT_FIELDS)
        and isinstance(judgment.get("row_index"), int)
        and isinstance(judgment.get("manifest_order"), int)
        and judgment["row_index"] == record["source_row_index"]
        and judgment["source_row_sha256"] == record["source_row_sha256"]
    )


def main() -> None:
    """Validate every admitted record and print only aggregate diagnostics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("audit", type=Path)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.audit.read_text(encoding="utf-8").splitlines()
        if line
    ]
    checks = {
        "rows_match": len(rows) == args.expected_rows,
        "all_selected": all(
            row["admission"]["selected_for_sft"] is True for row in rows
        ),
        "all_source_raw": all(nonempty_text(row["source_raw_json"]) for row in rows),
        "all_primary_raw": all(
            nonempty_text(row["primary_judgment_raw_json"]) for row in rows
        ),
        "all_adjudicator_raw": all(
            nonempty_text(row["adjudicator_judgment_raw_json"]) for row in rows
        ),
        "all_primary_rationale": all(
            nonempty_text(row["primary_judgment"]["decision"].get("rationale"))
            for row in rows
        ),
        "all_adjudicator_rationale": all(
            nonempty_text(row["adjudicator_judgment"]["decision"].get("rationale"))
            for row in rows
        ),
        "all_primary_provenance": all(
            provenance_complete(row["primary_judgment"], row)
            for row in rows
        ),
        "all_adjudicator_provenance": all(
            provenance_complete(row["adjudicator_judgment"], row)
            for row in rows
        ),
    }
    first_primary = rows[0]["primary_judgment"] if rows else {}
    first_adjudicator = rows[0]["adjudicator_judgment"] if rows else {}
    output = {
        "rows": len(rows),
        **checks,
        "all_checks_pass": all(checks.values()),
        "primary_top_level_keys": sorted(first_primary),
        "adjudicator_top_level_keys": sorted(first_adjudicator),
        "primary_decision_keys": sorted(first_primary.get("decision", {})),
        "adjudicator_decision_keys": sorted(first_adjudicator.get("decision", {})),
    }
    rendered = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not output["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
