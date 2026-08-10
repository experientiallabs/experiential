"""Tests for strict merging of real-verifier-admitted SFT bundles."""

from __future__ import annotations

import pytest
from merge_realverified_sft_bundles import merge_bundles


def rows(task_id: str, source_index: int) -> tuple[dict[str, object], dict[str, object]]:
    """Return one aligned admitted audit and Qwen row pair."""
    source_hash = f"hash-{source_index}"
    audit = {
        "source_row_index": source_index,
        "source_row_sha256": source_hash,
        "source": {"task_id": task_id},
        "real_verifier_admission": {"selected_for_sft": True},
    }
    qwen = {
        "messages": [{"role": "assistant", "content": "gold"}],
        "provenance": {
            "task_id": task_id,
            "source_row_sha256": source_hash,
            "selected_for_sft": True,
            "real_verifier_admission_pending": False,
            "real_verifier_reward": 1.0,
        },
    }
    return audit, qwen


def test_preserves_bundle_and_row_order() -> None:
    audit_a, qwen_a = rows("a", 1)
    audit_b, qwen_b = rows("b", 2)
    audits, qwens = merge_bundles(
        [("old", [audit_a], [qwen_a]), ("new", [audit_b], [qwen_b])]
    )
    assert audits == [audit_a, audit_b]
    assert [row["messages"] for row in qwens] == [
        qwen_a["messages"],
        qwen_b["messages"],
    ]
    assert all(row["provenance"]["selected_for_sft"] is True for row in qwens)
    assert all(
        row["provenance"]["real_verifier_admission_pending"] is False
        for row in qwens
    )
    assert all(
        row["provenance"]["legacy_real_verifier_admission_upgraded"] is False
        for row in qwens
    )


def test_upgrades_only_complete_legacy_verifier_provenance() -> None:
    audit, qwen = rows("legacy", 7)
    provenance = qwen["provenance"]
    provenance.pop("selected_for_sft")
    provenance.pop("real_verifier_admission_pending")
    provenance["final_environment_verifier_used"] = True
    provenance["model_judgment_is_official_task_verification"] = False
    _audits, qwens = merge_bundles([("legacy", [audit], [qwen])])
    assert qwens[0]["provenance"]["selected_for_sft"] is True
    assert qwens[0]["provenance"]["real_verifier_admission_pending"] is False
    assert qwens[0]["provenance"]["legacy_real_verifier_admission_upgraded"] is True

    audit, qwen = rows("incomplete", 8)
    provenance = qwen["provenance"]
    provenance.pop("selected_for_sft")
    provenance.pop("real_verifier_admission_pending")
    with pytest.raises(ValueError, match="lacks a valid admission boundary"):
        merge_bundles([("incomplete", [audit], [qwen])])


def test_rejects_duplicate_and_alignment_mismatches() -> None:
    audit_a, qwen_a = rows("a", 1)
    audit_b, qwen_b = rows("a", 2)
    with pytest.raises(ValueError, match="duplicate task"):
        merge_bundles(
            [("old", [audit_a], [qwen_a]), ("new", [audit_b], [qwen_b])]
        )

    mismatched_audit, mismatched_qwen = rows("c", 3)
    mismatched_qwen["provenance"]["task_id"] = "wrong"
    with pytest.raises(ValueError, match="task alignment mismatch"):
        merge_bundles([("bad", [mismatched_audit], [mismatched_qwen])])


def test_rejects_pending_or_nonperfect_rows() -> None:
    audit, qwen = rows("a", 1)
    qwen["provenance"]["real_verifier_admission_pending"] = True
    with pytest.raises(ValueError, match="lacks a valid admission boundary"):
        merge_bundles([("pending", [audit], [qwen])])

    audit, qwen = rows("a", 1)
    qwen["provenance"]["real_verifier_reward"] = 0.5
    with pytest.raises(ValueError, match="reward is not perfect"):
        merge_bundles([("nonperfect", [audit], [qwen])])
