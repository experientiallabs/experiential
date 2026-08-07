"""Focused tests for trajectory-judge normalization invariants."""

from __future__ import annotations

import unittest

from judge_trajectories import PROMPT_VERSION, SYSTEM_PROMPT, normalize_decision


def decision_value(**overrides: object) -> dict[str, object]:
    """Build one complete valid raw judge decision."""
    value: dict[str, object] = {
        "verdict": "PASS",
        "confidence": 95,
        "use_for_sft": True,
        "task_completion_summary": "Task completed.",
        "rationale": "Tests passed.",
        "success_evidence": ["Independent task test passed."],
        "failure_evidence": [],
        "unverified_requirements": [],
        "failure_modes": ["NONE"],
        "replay_risk": "LOW",
    }
    value.update(overrides)
    return value


class NormalizeDecisionTest(unittest.TestCase):
    """Exercise strict SFT-admission normalization."""

    def test_clean_high_confidence_pass_is_admitted(self) -> None:
        """A clean high-confidence PASS remains eligible for SFT."""
        decision, warnings = normalize_decision(decision_value())
        self.assertEqual(decision.verdict, "PASS")
        self.assertTrue(decision.use_for_sft)
        self.assertEqual(warnings, [])

    def test_pass_with_unverified_requirement_is_downgraded(self) -> None:
        """Residual uncertainty prevents a PASS from entering SFT."""
        decision, warnings = normalize_decision(
            decision_value(unverified_requirements=["Required report.txt was not inspected."])
        )
        self.assertEqual(decision.verdict, "UNCERTAIN")
        self.assertEqual(decision.confidence, 89)
        self.assertFalse(decision.use_for_sft)
        self.assertTrue(warnings)

    def test_missing_audit_arrays_receive_conservative_defaults(self) -> None:
        """Missing non-core arrays do not destroy the raw judgment record."""
        raw = decision_value()
        raw.pop("unverified_requirements")
        raw.pop("failure_modes")
        raw.pop("replay_risk")
        decision, warnings = normalize_decision(raw)
        self.assertEqual(decision.unverified_requirements, [])
        self.assertEqual(decision.failure_modes, [])
        self.assertEqual(decision.replay_risk, "HIGH")
        self.assertEqual(warnings, [])

    def test_v3_prompt_rejects_vacuous_hidden_grader_caveats(self) -> None:
        """The calibrated prompt reserves uncertainty for explicit requirements."""
        self.assertEqual(PROMPT_VERSION, "terminal-trajectory-quality-v3")
        self.assertIn("hypothetical hidden-grader", SYSTEM_PROMPT)
        self.assertIn("explicit material requirement", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
