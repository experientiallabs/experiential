"""A climb design climbs one actor policy, splits whole lineages and ignores case deadlines."""

from datetime import timedelta
from pathlib import Path

import pytest

from exp.optimize.evaluation.hillclimb.design import ClimbSpec, climb_split
from exp.optimize.evaluation.hillclimb.tests.climb_test import climb_fixture


def test_design_refuses_mixed_policies_odd_repeats_and_short_allowances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every case must share one policy, A/A halves and enough attempts for its repeats."""
    project, spec, _ = climb_fixture(tmp_path, monkeypatch, "records")
    first, *rest = spec.cases
    with pytest.raises(ValueError, match="same actor alias, model and system prompt"):
        ClimbSpec(cases=(first.model_copy(update={"system_prompt": "Other."}), *rest))
    with pytest.raises(ValueError, match="must be even"):
        spec.model_copy(update={"screen_repeats": 3}).model_validate(
            {**spec.model_dump(mode="json"), "screen_repeats": 3}
        )
    with pytest.raises(ValueError, match="maximum_physical_attempts of at least 32"):
        ClimbSpec(
            cases=tuple(
                job.model_copy(update={"maximum_physical_attempts": 8}) for job in spec.cases
            )
        )
    later = spec.model_copy(
        update={
            "cases": tuple(
                job.model_copy(update={"deadline": job.deadline + timedelta(days=3)})
                for job in spec.cases
            )
        }
    )
    assert later.identity() == spec.identity()
    assert spec.model_copy(update={"split_salt": "other"}).identity() != spec.identity()
    split = climb_split(project, spec)
    assert sorted((*split.train, *split.test)) == list(range(len(spec.cases)))
    assert len(set(split.lineages)) == len(spec.cases)
