"""Tests for complete double-judge audit preservation and admission."""

from __future__ import annotations

import unittest

from build_verified_audit_dataset import admission_failures, build_records


def judgment(verdict: str, confidence: int, use_for_sft: bool) -> dict:
    """Return a minimal judgment carrying rationale and provenance fixtures."""
    return {
        "schema": "judgment-v1",
        "row_index": 3,
        "request_id": "request-1",
        "decision": {
            "verdict": verdict,
            "confidence": confidence,
            "use_for_sft": use_for_sft,
            "rationale": "fixture rationale",
        },
    }


class BuildVerifiedAuditDatasetTest(unittest.TestCase):
    """Verify fail-closed admission and lossless nested records."""

    def test_strict_pass_requires_all_three_conditions(self) -> None:
        self.assertEqual(admission_failures(judgment("PASS", 90, True), "p"), [])
        self.assertTrue(admission_failures(judgment("PASS", 89, True), "p"))
        self.assertTrue(admission_failures(judgment("FAIL", 99, True), "p"))
        self.assertTrue(admission_failures(judgment("PASS", 99, False), "p"))

    def test_complete_source_and_judgments_survive(self) -> None:
        source_raw = '{"task_id":"task-1","custom_source_field":{"x":1}}'
        primary = judgment("PASS", 95, True)
        adjudicator = judgment("PASS", 96, True)
        ledger, admitted = build_records(
            sources={3: (source_raw, {"task_id": "task-1", "custom_source_field": {"x": 1}})},
            primary={3: ("primary raw", primary)},
            adjudicator={3: ("adjudicator raw", adjudicator)},
        )
        self.assertEqual(len(ledger), 1)
        self.assertEqual(len(admitted), 1)
        row = admitted[0]
        self.assertEqual(row["source_raw_json"], source_raw)
        self.assertEqual(row["source"]["custom_source_field"], {"x": 1})
        self.assertEqual(row["primary_judgment"], primary)
        self.assertEqual(row["primary_judgment_raw_json"], "primary raw")
        self.assertEqual(row["adjudicator_judgment"], adjudicator)

    def test_missing_or_error_judgment_is_excluded(self) -> None:
        passing = judgment("PASS", 99, True)
        ledger, admitted = build_records(
            sources={3: ('{"task_id":"task-1"}', {"task_id": "task-1"})},
            primary={3: ("primary raw", passing)},
            adjudicator={3: ("error raw", {"row_index": 3, "error": "bad output"})},
        )
        self.assertEqual(admitted, [])
        self.assertIn(
            "adjudicator:no_valid_decision",
            ledger[0]["admission"]["exclusion_reasons"],
        )


if __name__ == "__main__":
    unittest.main()
