"""Tests for strict TMax recovery merging."""

from __future__ import annotations

import unittest

from merge_tmax_recovery import select_rows


def wrapped(task_id: str, status: str, reward: float | None = None):
    """Return the row and placeholder path shape used by the merger."""
    row = {"task_id": task_id, "status": status, "reward": reward}
    return row, None


class RecoveryMergeTest(unittest.TestCase):
    """Ensure recovery replaces only matching nonterminal episodes."""

    def test_replaces_starting_and_keeps_terminal(self) -> None:
        original = {
            "a": wrapped("a", "scored", 1.0),
            "b": wrapped("b", "starting"),
        }
        recovery = {"b": wrapped("b", "scored", 0.5)}
        selected, replaced = select_rows(original, recovery)  # type: ignore[arg-type]
        self.assertEqual(selected["a"][0]["reward"], 1.0)
        self.assertEqual(selected["b"][0]["reward"], 0.5)
        self.assertEqual(replaced, ["b"])

    def test_refuses_terminal_replacement_and_unexpected_task(self) -> None:
        original = {"a": wrapped("a", "scored", 1.0)}
        with self.assertRaisesRegex(ValueError, "refusing to replace terminal"):
            select_rows(  # type: ignore[arg-type]
                original, {"a": wrapped("a", "scored", 0.5)}
            )
        with self.assertRaisesRegex(ValueError, "unexpected tasks"):
            select_rows(  # type: ignore[arg-type]
                original, {"b": wrapped("b", "scored", 0.5)}
            )


if __name__ == "__main__":
    unittest.main()
