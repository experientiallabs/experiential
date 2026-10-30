"""Fresh-session, budget and saved-reference checks for initial scoring."""

from pathlib import Path

import pytest

from exp.common.project import artifact_input
from exp.optimize.model.sft.initial_scoring import score_initial_training_losses
from exp.optimize.model.sft.tail_sft_test import _ScoringBackend
from exp.optimize.model.sft.training import TinkerSFTError
from exp.optimize.model.sft.training_test import _TIME, _persisted_dataset, _spec


def test_fresh_scoring_replays_without_opening_and_never_updates(tmp_path: Path) -> None:
    """The initial reference cannot accidentally come from a resumed optimizer session."""
    fixture = _persisted_dataset(tmp_path)
    dataset = artifact_input(
        fixture.store.artifacts.read(fixture.artifact.dataset.dataset_id).manifest
    )
    spec = _spec(batch_size=2, maximum_cost_usd=1)
    backend = _ScoringBackend(initial=True)
    reference = score_initial_training_losses(
        fixture.store, dataset, spec, backend, created_at=_TIME, code_revision="test"
    )
    assert backend.open_resume_paths == [None]
    assert backend.train_calls == 0
    assert backend.score_calls == 1
    assert (
        score_initial_training_losses(
            fixture.store, dataset, spec, backend, created_at=_TIME, code_revision="test"
        )
        == reference
    )
    assert backend.open_resume_paths == [None]


def test_initial_budget_refuses_before_open(tmp_path: Path) -> None:
    """Finite call counts do not authorize a scoring pass exceeding its dollar ceiling."""
    fixture = _persisted_dataset(tmp_path)
    dataset = artifact_input(
        fixture.store.artifacts.read(fixture.artifact.dataset.dataset_id).manifest
    )
    backend = _ScoringBackend(initial=True)
    with pytest.raises(TinkerSFTError, match="budget"):
        score_initial_training_losses(
            fixture.store,
            dataset,
            _spec(maximum_cost_usd=0.01),
            backend,
            created_at=_TIME,
            code_revision="test",
        )
    assert backend.open_resume_paths == []
