"""Tests for the real-verifier SFT bundle filter."""

from __future__ import annotations

import unittest

from build_real_verifier_filtered_sft import build_bundle, merge_replay_recoveries


def audit(task_id: str, source_hash: str) -> dict[str, object]:
    """Return one compact canonical audit record."""
    return {
        "source_row_index": 1,
        "source_row_sha256": source_hash,
        "source": {"task_id": task_id, "rollout_id": f"rollout-{task_id}"},
        "admission": {"selected_for_sft": True},
        "primary_judgment": {"decision": {"rationale": "primary rationale"}},
        "adjudicator_judgment": {
            "decision": {"rationale": "adjudicator rationale"}
        },
    }


def qwen(task_id: str, source_hash: str) -> dict[str, object]:
    """Return one compact materialized training row."""
    return {
        "messages": [{"role": "assistant", "content": "gold"}],
        "provenance": {
            "task_id": task_id,
            "source_row_sha256": source_hash,
            "selected_for_sft": False,
            "real_verifier_admission_pending": True,
        },
    }


def replay(task_id: str, source_hash: str, reward: float) -> dict[str, object]:
    """Return one scored replay result."""
    return {
        "task_id": task_id,
        "source_row_sha256": source_hash,
        "status": "scored",
        "reward": reward,
        "final_verifier": {"reward": reward},
        "finished_at": 123.0,
        "model_judgment_is_official_task_verification": False,
    }


def unscored_replay(
    task_id: str, source_hash: str, *, finished_at: float | None = 123.0
) -> dict[str, object]:
    """Return one terminal or unfinished unscored replay result."""
    return {
        "task_id": task_id,
        "source_row_sha256": source_hash,
        "status": "episode_error",
        "reward": None,
        "finished_at": finished_at,
        "model_judgment_is_official_task_verification": False,
    }


class BuildBundleTest(unittest.TestCase):
    """Verify strict selection, lossless audit augmentation, and alignment."""

    def test_selects_only_perfect_and_preserves_judge_rationales(self) -> None:
        audits = [audit("a", "ha"), audit("b", "hb")]
        qwens = [qwen("a", "ha"), qwen("b", "hb")]
        replays = [replay("a", "ha", 1.0), replay("b", "hb", 0.75)]
        selected_audit, selected_qwen, ledger = build_bundle(
            audit_rows=audits,
            qwen_rows=qwens,
            replay_rows=replays,
            minimum_reward=1.0,
        )
        self.assertEqual([row["source"]["task_id"] for row in selected_audit], ["a"])
        self.assertEqual(selected_audit[0]["primary_judgment"], audits[0]["primary_judgment"])
        self.assertEqual(
            selected_audit[0]["adjudicator_judgment"],
            audits[0]["adjudicator_judgment"],
        )
        self.assertEqual(selected_qwen[0]["messages"], qwens[0]["messages"])
        self.assertTrue(selected_qwen[0]["provenance"]["selected_for_sft"])
        self.assertFalse(
            selected_qwen[0]["provenance"]["real_verifier_admission_pending"]
        )
        self.assertFalse(
            selected_qwen[0]["provenance"]["pre_verifier_selected_for_sft"]
        )
        self.assertTrue(ledger[0]["selected_for_real_verifier_sft"])
        self.assertFalse(ledger[1]["selected_for_real_verifier_sft"])
        self.assertEqual(
            ledger[1]["exclusion_reasons"], ["replay_reward_below_1:0.75"]
        )

    def test_rejects_hash_and_task_set_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "qwen source row hash mismatch"):
            build_bundle(
                audit_rows=[audit("a", "ha")],
                qwen_rows=[qwen("a", "wrong")],
                replay_rows=[replay("a", "ha", 1.0)],
                minimum_reward=1.0,
            )

    def test_recovery_replaces_only_unscored_primary_and_preserves_order(self) -> None:
        primary = [unscored_replay("a", "ha"), replay("b", "hb", 0.5)]
        recovery = replay("a", "ha", 1.0)
        merged, decisions = merge_replay_recoveries(
            primary, [("recovery-one", [recovery])]
        )
        self.assertEqual([row["task_id"] for row in merged], ["a", "b"])
        self.assertIs(merged[0], recovery)
        self.assertIs(merged[1], primary[1])
        self.assertTrue(decisions[0]["selected_as_authoritative"])

    def test_unscored_recovery_is_audited_but_not_selected(self) -> None:
        primary = [unscored_replay("a", "ha")]
        recovery = unscored_replay("a", "ha")
        merged, decisions = merge_replay_recoveries(
            primary, [("recovery-one", [recovery])]
        )
        self.assertIs(merged[0], primary[0])
        self.assertFalse(decisions[0]["selected_as_authoritative"])

    def test_recovery_rejects_scored_primary_hash_and_unfinished_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "refusing to replace scored primary"):
            merge_replay_recoveries(
                [replay("a", "ha", 0.5)], [("recovery", [replay("a", "ha", 1.0)])]
            )
        with self.assertRaisesRegex(ValueError, "source row hash mismatch"):
            merge_replay_recoveries(
                [unscored_replay("a", "ha")],
                [("recovery", [replay("a", "wrong", 1.0)])],
            )
        with self.assertRaisesRegex(ValueError, "recovery is unfinished"):
            merge_replay_recoveries(
                [unscored_replay("a", "ha")],
                [("recovery", [unscored_replay("a", "ha", finished_at=None)])],
            )

    def test_recovery_rejects_duplicate_task_across_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate recovery task"):
            merge_replay_recoveries(
                [unscored_replay("a", "ha")],
                [
                    ("recovery-one", [unscored_replay("a", "ha")]),
                    ("recovery-two", [unscored_replay("a", "ha")]),
                ],
            )
        with self.assertRaisesRegex(ValueError, "replay task set differs"):
            build_bundle(
                audit_rows=[audit("a", "ha")],
                qwen_rows=[qwen("a", "ha")],
                replay_rows=[replay("b", "hb", 1.0)],
                minimum_reward=1.0,
            )


if __name__ == "__main__":
    unittest.main()
