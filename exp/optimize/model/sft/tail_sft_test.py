"""Loss-filter counterexamples and durable training checks without provider calls."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput, sha256_json
from exp.common.project import artifact_input
from exp.optimize.model.sft.tail_sft import (
    LossDatum,
    TailSFTPolicy,
    TargetLoss,
    load_initial_losses,
    record_initial_losses,
    select_tail,
)
from exp.optimize.model.sft.training import TinkerSFTBudgetExceeded, TinkerSFTError, TinkerSFTSpec
from exp.optimize.model.sft.training_test import (
    _TIME,
    _FakeBackend,
    _FakeDatum,
    _FakeSession,
    _persisted_dataset,
    _PersistedDataset,
    _run,
    _spec,
)


def _loss(name: str, loss: float, tokens: int = 2) -> TargetLoss:
    """Bind a hand-calculated example to a fixed target mask."""
    return TargetLoss(
        example_id=name, datum_sha256=sha256_json(name), target_tokens=tokens, mean_loss=loss
    )


def test_selection_drops_most_improved_not_easiest_or_longest() -> None:
    """A harder current example can be dropped because its improvement is much larger."""
    initial = (_loss("easy", 1), _loss("hard", 8, 200), _loss("worse", 2))
    current = (_loss("easy", 0.8), _loss("hard", 3, 200), _loss("worse", 4))
    selected = select_tail(initial, current, 1 / 3)
    assert selected.retained == ("easy", "worse")
    assert selected.margins == {"easy": -0.19999999999999996, "hard": -5, "worse": 2}
    assert len(selected.current) == 3


def test_ties_use_identity_and_target_mask_drift_refuses() -> None:
    """Equal margins are stable; normalization cannot silently switch target spans."""
    rows = (_loss("b", 1), _loss("a", 1))
    assert select_tail(rows, rows, 0.5).retained == ("b",)
    with pytest.raises(ValueError, match="target mask changed"):
        select_tail(rows, (_loss("b", 1, 3),), 0)
    with pytest.raises(ValueError, match="whole batch"):
        select_tail(rows[:1], rows[:1], 0.9)
    with pytest.raises(ValueError, match="unique"):
        select_tail(rows, (rows[0], rows[0]), 0)


class _ScoringSession(_FakeSession):
    """Separate initial and updated losses while exercising the actual training owner."""

    def datum_identity(self, datum: LossDatum) -> str:
        """Bind both input identity and supervised-token mask."""
        return sha256_json([datum.example_id, datum.supervised_token_count])

    def score_batch(self, datums: Sequence[LossDatum]) -> tuple[TargetLoss, ...]:
        """Make the first example improve more, independently of the shuffled order."""
        backend = self._backend
        assert isinstance(backend, _ScoringBackend)
        backend.score_calls += 1
        if backend.fail_score:
            raise RuntimeError("unknown scoring outcome")
        return tuple(
            TargetLoss(
                example_id=datum.example_id,
                datum_sha256=self.datum_identity(datum),
                target_tokens=datum.supervised_token_count,
                mean_loss=10
                if backend.initial
                else (1 if datum.supervised_token_count == 3 else 5),
            )
            for datum in datums
        )


class _ScoringBackend(_FakeBackend):
    """Fresh sessions expose scoring; no provider or optimizer is hidden in the test."""

    def __init__(self, *, initial: bool = False, fail_score: bool = False) -> None:
        """Configure scoring behavior and the existing optimizer-call journal."""
        super().__init__(conservative_cost_per_batch=0.2)
        self.initial = initial
        self.fail_score = fail_score
        self.score_calls = 0

    def open(self, spec: TinkerSFTSpec, resume_state_path: str | None) -> _ScoringSession:
        """Record session creation and return the scoring implementation."""
        self.open_resume_paths.append(resume_state_path)
        return _ScoringSession(self)


def _reference(
    fixture: _PersistedDataset, spec: TinkerSFTSpec, backend: _ScoringBackend
) -> tuple[ArtifactInput, tuple[_FakeDatum, ...]]:
    """Record exact untrained losses for the fixture's two training examples."""
    session = backend.open(spec, None)
    datums = session.render_examples(
        tuple(row.example for row in fixture.artifact.rows if row.partition == "train")
    )
    reference = record_initial_losses(
        fixture.store.artifacts,
        artifact_input(fixture.store.artifacts.read(fixture.artifact.dataset.dataset_id).manifest),
        spec.initialization_identity(),
        datums,
        session,
        batch_size=2,
        created_at=_TIME,
        code_revision="w13-test",
    )
    return reference, datums


def test_saved_initial_reference_and_filtered_training_replay(tmp_path: Path) -> None:
    """Only retained targets update; replay needs neither scoring nor a training session."""
    fixture = _persisted_dataset(tmp_path)
    spec = _spec(batch_size=2, epochs=2)
    initial_backend = _ScoringBackend(initial=True)
    reference, datums = _reference(fixture, spec, initial_backend)
    assert len(datums) == 2
    assert len(load_initial_losses(fixture.store.artifacts, reference).losses) == 2
    assert _reference(fixture, spec, initial_backend)[0] == reference
    assert initial_backend.score_calls == 1
    training = spec.model_copy(
        update={"tail_sft": TailSFTPolicy(reference=reference, drop_fraction=0.5)}
    )
    backend = _ScoringBackend()
    result = _run(fixture, tmp_path / "run", backend, spec=training)
    assert backend.score_calls == backend.train_calls == 2
    assert backend.trained_example_ids == [datums[1].example_id] * 2
    replay_backend = _ScoringBackend()
    assert _run(fixture, tmp_path / "run", replay_backend, spec=training) == result
    assert replay_backend.open_resume_paths == []


def test_initial_unknown_is_not_silently_scored_again(tmp_path: Path) -> None:
    """A failed scoring request leaves an intent that blocks a hidden second draw."""
    fixture = _persisted_dataset(tmp_path)
    backend = _ScoringBackend(initial=True, fail_score=True)
    with pytest.raises(RuntimeError, match="unknown"):
        _reference(fixture, _spec(), backend)
    backend.fail_score = False
    with pytest.raises(ValueError, match="outcome unknown"):
        _reference(fixture, _spec(), backend)
    assert backend.score_calls == 1


def test_changed_initialization_refuses_before_client_open(tmp_path: Path) -> None:
    """Reference losses from another seed cannot select examples for this model."""
    fixture = _persisted_dataset(tmp_path)
    reference, _ = _reference(fixture, _spec(), _ScoringBackend(initial=True))
    spec = _spec(seed=20).model_copy(
        update={"tail_sft": TailSFTPolicy(reference=reference, drop_fraction=0)}
    )
    backend = _ScoringBackend()
    with pytest.raises(TinkerSFTError, match="initialization"):
        _run(fixture, tmp_path / "run", backend, spec=spec)
    assert backend.open_resume_paths == []


def test_filtering_does_not_hide_provider_cost_overrun(tmp_path: Path) -> None:
    """Accounting the scoring ceiling must not erase an observed update overrun."""
    fixture = _persisted_dataset(tmp_path)
    reference, _ = _reference(fixture, _spec(), _ScoringBackend(initial=True))
    spec = _spec(batch_size=2).model_copy(
        update={"tail_sft": TailSFTPolicy(reference=reference, drop_fraction=0.5)}
    )
    backend = _ScoringBackend()
    backend.cost_per_batch = 10
    with pytest.raises(TinkerSFTBudgetExceeded, match="exceeded"):
        _run(fixture, tmp_path / "run", backend, spec=spec)
    assert backend.train_calls == 1
