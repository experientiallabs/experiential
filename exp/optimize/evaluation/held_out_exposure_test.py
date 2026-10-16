"""The held-out exposure decision needs the build's grouping in the ledger it decides with."""

from pathlib import Path

import pytest

from exp.optimize.evaluation.held_out_exposure import held_out_exposure, held_out_members
from exp.optimize.evaluation.held_out_fidelity_test import (
    USERS,
    SimulatedProvider,
    build_project,
    episode,
    held_out_lineages,
)
from exp.simulation.mining.exposure_ledger import ExposureLedger


def test_exposure_is_decided_only_with_a_ledger_that_recorded_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every build records itself; a ledger without that build cannot decide independence."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)])
    _, store, _ = build_project(tmp_path, monkeypatch, provider)
    members = held_out_members(store)
    assert set(members) == held_out_lineages(store) and all(members.values())
    decision = held_out_exposure(store)
    assert decision.exposed == ()
    assert decision.ledger == ExposureLedger(store.paths.root).snapshot().identity
    with pytest.raises(ValueError, match="never recorded its traces and grouping"):
        held_out_exposure(store, ExposureLedger(tmp_path / "elsewhere").snapshot())
