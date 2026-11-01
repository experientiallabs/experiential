"""Tests for the shared core-set metric."""

from __future__ import annotations

import math

import pytest

from exp.simulation.mining.geometry import CoreSetGeometry, cosine


def _geometry(**rows: tuple[float, ...]) -> CoreSetGeometry:
    """Build a geometry with unit mass per identity."""
    return CoreSetGeometry.of(dict(rows), dict.fromkeys(rows, 1))


def test_cosine_clamps_drift_into_the_valid_range() -> None:
    """Coverage records a similarity bounded at one, so the metric cannot hand them 1.0000001."""
    almost = 1.0 / math.sqrt(2.0)
    assert cosine((almost, almost), (almost, almost)) <= 1.0
    assert cosine((1.0, 0.0), (-1.0, 0.0)) == -1.0


def test_cosine_refuses_two_widths() -> None:
    """Two geometries have no distance between them, so pairing them must fail rather than pad."""
    with pytest.raises(ValueError):
        cosine((1.0, 0.0), (1.0, 0.0, 0.0))


def test_the_medoid_follows_workload_rather_than_coordinates() -> None:
    """The centre of a group is the case representing its traffic, not its middle coordinate."""
    vectors = {"a": (1.0, 0.0), "b": (0.0, 1.0), "c": (0.8, 0.6)}
    balanced = CoreSetGeometry.of(vectors, {"a": 1, "b": 1, "c": 1})
    skewed = CoreSetGeometry.of(vectors, {"a": 50, "b": 1, "c": 1})

    assert balanced.medoid(("a", "b", "c")) == "c"
    assert skewed.medoid(("a", "b", "c")) == "a"


def test_every_tie_breaks_on_ascending_identity() -> None:
    """Identical positions must not let input order decide, or artifacts stop reproducing."""
    geometry = _geometry(zeta=(1.0, 0.0), alpha=(1.0, 0.0), mu=(1.0, 0.0))

    assert geometry.medoid(("zeta", "mu", "alpha")) == "alpha"
    assert geometry.medoid(("alpha", "mu", "zeta")) == "alpha"
    assert geometry.farthest_first(("alpha",)) == "mu"


def test_the_first_coverage_point_is_a_centre_not_an_outlier() -> None:
    """With nothing selected, farthest-first has no reference, so it starts from the medoid."""
    geometry = _geometry(a=(1.0, 0.0), b=(0.98, 0.199), c=(0.0, 1.0))

    # b sits between a and the outlier c, so it is the weighted centre of the three.
    assert geometry.farthest_first(()) == "b"
    assert geometry.farthest_first(()) == geometry.medoid(geometry.ids)


def test_farthest_first_measures_against_everything_already_selected() -> None:
    """Filling one cluster must not place a point beside a representative chosen elsewhere."""
    geometry = _geometry(near=(1.0, 0.0), far=(0.0, 1.0), middle=(0.7071, 0.7071))

    assert geometry.farthest_first(("near",), within=("far", "middle")) == "far"
    assert geometry.farthest_first(("near", "far"), within=("far", "middle")) == "middle"


def test_an_exhausted_eligible_set_is_a_failure_rather_than_a_repeat() -> None:
    """Returning an already-selected identity would silently shrink the task set."""
    geometry = _geometry(a=(1.0, 0.0))

    with pytest.raises(ValueError, match="no remaining candidate"):
        geometry.farthest_first(("a",))


def test_positions_without_workload_are_refused() -> None:
    """A candidate with no mass would vanish from conservation accounting."""
    with pytest.raises(ValueError, match="one workload mass per positioned identity"):
        CoreSetGeometry.of({"a": (1.0,), "b": (0.0,)}, {"a": 1})


def test_identities_are_sorted_so_iteration_never_depends_on_input_order() -> None:
    """Every backend builds its matrix from this order, so it has to be canonical."""
    assert _geometry(zeta=(1.0,), alpha=(1.0,), mu=(1.0,)).ids == ("alpha", "mu", "zeta")
