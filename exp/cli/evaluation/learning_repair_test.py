"""Resolution selection of the repair-learning payload.

Each resolution runs end to end through ``exp eval --continuations`` in the learning owner's
tests: ``lessons_test`` (screening), ``lesson_evolution_test`` (evolution) and ``repair_test``
(observation retention and trace retirement).
"""

import pytest
from pydantic import ValidationError

from exp.cli.evaluation.continuations import ContinuationCommand
from exp.optimize.evaluation.learning.lessons import LessonEvolutionRequest
from exp.optimize.evaluation.learning.repair import LearningRepairSpec, TraceRetirementRequest
from exp.optimize.evaluation.learning.screening import LessonScreeningSpec

REF = {"artifact_id": "saved", "sha256": "0" * 64}


def test_resolution_selects_the_payload_and_defaults_to_observation_retention() -> None:
    """Existing settlement specs need no resolution; the others name theirs."""
    cases = (
        ({"run": REF, "report": REF, "reason": "operator"}, LearningRepairSpec),
        ({"resolution": "screen-lessons"}, LessonScreeningSpec),
        (
            {"resolution": "evolve-lessons", "lessons": REF, "run_id": "edit"},
            LessonEvolutionRequest,
        ),
        (
            {"resolution": "retire-trace", "trace_id": "t-1", "reason": "refused"},
            TraceRetirementRequest,
        ),
    )
    for payload, kind in cases:
        command = ContinuationCommand.model_validate(
            {"action": "repair-learning", "learning_repair": payload}
        )
        assert isinstance(command.learning_repair, kind)
    with pytest.raises(ValidationError, match="does not match any of the expected tags"):
        ContinuationCommand.model_validate(
            {"action": "repair-learning", "learning_repair": {"resolution": "guess"}}
        )
