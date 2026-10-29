"""Unknown costs never bypass immutable lineage validation."""

from pathlib import Path

import pytest

from exp.common.project import ProjectStore
from exp.simulation.engines.text.errors import SimulationResumeError
from exp.simulation.engines.text.lineage_spend import lineage_spend
from exp.simulation.engines.text.redaction_test import _rollout, _span


@pytest.mark.parametrize("known_first", [False, True])
def test_unknown_spend_still_checks_retry_lineage(tmp_path: Path, known_first: bool) -> None:
    """An unknown result or sibling cannot hide a missing retry binding.

    Args:
        tmp_path: Isolated artifact root for retry-lineage verification.
        known_first: Places the original rollout before the malformed retry when True.
    """
    unknown = _rollout(spans=(_span("done"),), final_output=None)
    invalid = unknown.model_copy(
        update={"rollout_id": "invalid-retry", "retry_attempt": 1, "simulation_binding": None}
    )
    rows = (unknown, invalid) if known_first else (invalid, unknown)
    with pytest.raises(SimulationResumeError, match="complete simulation binding"):
        lineage_spend(ProjectStore(tmp_path, "project-a").artifacts, rows, measure=lambda _: None)
