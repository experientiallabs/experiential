"""Coverage quotas cannot be satisfied by extra draws or relabeled related tasks."""

from itertools import product

import pytest

from exp.common.core.artifacts import sha256_json
from exp.simulation.task_distribution import (
    CoverageStratum,
    FamilyCandidate,
    TaskDistribution,
    select_distribution,
)


def policy(*ids: str) -> TaskDistribution:
    """Require one family per explicitly declared capability stratum."""
    return TaskDistribution(
        strata=tuple(
            CoverageStratum(
                id=key,
                user="operator",
                capability=key,
                category="frontier",
                information="discoverable",
                horizon="multi_step",
                minimum_families=1,
            )
            for key in ids
        )
    )


def candidate(key: str, *strata: str, lineage: str | None = None) -> FamilyCandidate:
    """Create unrelated source evidence unless a test explicitly links it."""
    return FamilyCandidate(
        id=key, strata=strata, lineages=(lineage or key,), workflow=sha256_json(key)
    )


def test_overlapping_coverage_matches_exhaustive_assignment_oracle() -> None:
    """Every three-family incidence graph matches a brute-force slot assignment optimum."""
    rules = policy("a", "b", "c")
    choices = ((), ("a",), ("b",), ("c",), ("a", "b"), ("a", "c"), ("b", "c"), ("a", "b", "c"))
    for edges in product(choices, repeat=3):
        rows = tuple(candidate(str(index), *labels) for index, labels in enumerate(edges) if labels)
        optimum = max(
            sum(value is not None for value in assignment)
            for assignment in product((None, 0, 1, 2), repeat=3)
            if len([value for value in assignment if value is not None])
            == len({value for value in assignment if value is not None})
            and all(
                value is None or label in edges[value]
                for label, value in zip(("a", "b", "c"), assignment, strict=True)
            )
        )
        selected = select_distribution(rules, rows)
        assert sum(map(len, selected.selected.values())) == optimum
        assert select_distribution(rules, tuple(reversed(rows))) == selected


def test_failed_bridge_cannot_restore_independent_family_status() -> None:
    """A and C remain one family through failed B, even when their direct sources differ."""
    a = candidate("a", "coverage", lineage="source-one")
    b = candidate("b", "coverage", lineage="source-one").model_copy(
        update={"traces": ("trace-two",)}
    )
    c = candidate("c", "coverage").model_copy(update={"traces": ("trace-two",)})
    rules = policy("coverage")
    rules = rules.model_copy(
        update={"strata": (rules.strata[0].model_copy(update={"minimum_families": 2}),)}
    )
    report = select_distribution(rules, (a, b, c), eligible=frozenset({"a", "c"}))
    assert report.components == (("a", "b", "c"),)
    assert report.deficits == {"coverage": 1}


def test_same_workflow_with_new_ids_is_one_family() -> None:
    """Export renumbering cannot turn the same workflow into two coverage units."""
    a = candidate("one", "coverage")
    b = candidate("two", "coverage").model_copy(update={"workflow": a.workflow})
    report = select_distribution(policy("coverage"), (a, b))
    assert report.components == (("one", "two"),)
    assert len(report.selected["coverage"]) == 1


def test_unknown_strata_and_eligibility_refuse() -> None:
    """Neither invented axes nor an out-of-population outcome may affect admission."""
    with pytest.raises(ValueError, match="undeclared"):
        select_distribution(policy("known"), (candidate("one", "unknown"),))
    with pytest.raises(ValueError, match="frozen candidate"):
        select_distribution(policy("known"), (), eligible=frozenset({"invented"}))
