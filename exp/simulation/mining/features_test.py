"""Tests for the explicitly chosen core-set selection geometry."""

from __future__ import annotations

import math

import pytest
from pydantic import TypeAdapter, ValidationError

from exp.simulation.mining.descriptors import RoutingDescriptor
from exp.simulation.mining.features import (
    LOGIT_CLAMP,
    DecisionTagSelectionSpec,
    HashingSelectionSpec,
    SelectionGeometryError,
    SelectionSpec,
    selection_vectors,
)
from exp.simulation.mining.questions import Question, QuestionBank, packaged_question_bank
from exp.simulation.mining.tagging import TagMatrix, TagRow

_SPEC_ADAPTER: TypeAdapter[SelectionSpec] = TypeAdapter(SelectionSpec)


def _descriptor(trace_id: str, intent: str = "Review the supplier invoice.") -> RoutingDescriptor:
    """Build one request-visible descriptor for a trace."""
    return RoutingDescriptor(
        trace_id=trace_id, intent=intent, initial_context={}, tools=(), tags=()
    )


def _bank(count: int = 3, bank_id: str = "fixture-v1") -> QuestionBank:
    """Build a situation bank of ``count`` propositions plus one other family."""
    situation = tuple(
        Question(question_id=f"sit.q{index}", family="situation", text=f"Is case {index} so?")
        for index in range(count)
    )
    return QuestionBank(
        bank_id=bank_id,
        version=1,
        questions=(
            *situation,
            Question(question_id="dom.legal", family="domain", text="Is it legal work?"),
        ),
    )


def _matrix(
    rows: dict[str, tuple[float, ...]],
    bank: QuestionBank | None = None,
) -> TagMatrix:
    """Build a matrix whose columns are the bank's, with one row per supplied trace."""
    resolved = bank or _bank()
    return TagMatrix(
        bank_id=resolved.bank_id,
        bank_version=resolved.version,
        question_ids=resolved.question_ids(),
        view="observations",
        rows=tuple(
            TagRow(trace_id=trace_id, unit_hash=f"hash-{trace_id}", values=values)
            for trace_id, values in rows.items()
        ),
    )


def test_a_decision_tag_spec_without_a_tag_matrix_raises() -> None:
    """A silent fall back to hashing still returns tasks, so it would never be noticed."""
    spec = DecisionTagSelectionSpec(bank_id="fixture-v1")
    with pytest.raises(SelectionGeometryError, match="needs the tag matrix"):
        selection_vectors(spec, descriptors=[_descriptor("trace-1")])


def test_the_missing_matrix_error_names_both_ways_out() -> None:
    """A failure a user can hit must say what to do, not only what went wrong."""
    spec = DecisionTagSelectionSpec(bank_id="situation-v1")
    with pytest.raises(SelectionGeometryError) as caught:
        selection_vectors(spec, descriptors=[_descriptor("trace-1")])
    assert "TraceTagger" in str(caught.value)
    assert "HashingSelectionSpec" in str(caught.value)


def test_hashing_geometry_needs_no_tag_matrix() -> None:
    """A provider-free build must still get a usable geometry for every candidate."""
    descriptors = [_descriptor("a", "first case"), _descriptor("b", "second case")]
    vectors = selection_vectors(HashingSelectionSpec(dimensions=16), descriptors=descriptors)
    assert len(vectors) == 2
    assert all(len(vector) == 16 for vector in vectors)
    assert vectors[0] != vectors[1]
    for vector in vectors:
        assert math.isclose(math.sqrt(sum(value**2 for value in vector)), 1.0, abs_tol=1e-9)


def test_hashing_geometry_ignores_a_supplied_tag_matrix() -> None:
    """A hashing spec declares the geometry, so an unused matrix must not change it."""
    descriptors = [_descriptor("a")]
    spec = HashingSelectionSpec(dimensions=16)
    without = selection_vectors(spec, descriptors=descriptors)
    with_matrix = selection_vectors(
        spec, descriptors=descriptors, tag_matrix=_matrix({"a": (1.0, 0.0, 0.0, 0.0)})
    )
    assert without == with_matrix


def test_vectors_follow_descriptor_order_not_matrix_order() -> None:
    """The result is assigned positionally, so a reordered matrix must not shuffle rows."""
    bank = _bank()
    matrix = _matrix({"b": (0.0, 1.0, 0.0, 0.0), "a": (1.0, 0.0, 0.0, 0.0)}, bank)
    spec = DecisionTagSelectionSpec(bank_id=bank.bank_id, transform="none")
    vectors = selection_vectors(
        spec, descriptors=[_descriptor("a"), _descriptor("b")], tag_matrix=matrix, bank=bank
    )
    assert vectors == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


def test_only_the_selected_families_become_coordinates() -> None:
    """A spec names families so the geometry is readable, and other columns must stay out."""
    bank = _bank()
    matrix = _matrix({"a": (1.0, 0.0, 0.0, 1.0)}, bank)
    situation = selection_vectors(
        DecisionTagSelectionSpec(bank_id=bank.bank_id, transform="none"),
        descriptors=[_descriptor("a")],
        tag_matrix=matrix,
        bank=bank,
    )
    both = selection_vectors(
        DecisionTagSelectionSpec(
            bank_id=bank.bank_id, families=("situation", "domain"), transform="none"
        ),
        descriptors=[_descriptor("a")],
        tag_matrix=matrix,
        bank=bank,
    )
    assert situation == ((1.0, 0.0, 0.0),)
    assert both == ((1.0, 0.0, 0.0, 1.0),)


def test_transform_none_passes_probabilities_through() -> None:
    """Keeping the raw values keeps distance sensitive to how many tags a case triggered."""
    bank = _bank()
    vectors = selection_vectors(
        DecisionTagSelectionSpec(bank_id=bank.bank_id, transform="none"),
        descriptors=[_descriptor("a")],
        tag_matrix=_matrix({"a": (0.25, 0.5, 0.0, 1.0)}, bank),
        bank=bank,
    )
    assert vectors == ((0.25, 0.5, 0.0),)


def test_transform_l2_unit_normalizes_the_selected_values() -> None:
    """Unit length makes distance reflect which tags fired rather than how many."""
    bank = _bank()
    vectors = selection_vectors(
        DecisionTagSelectionSpec(bank_id=bank.bank_id, transform="l2"),
        descriptors=[_descriptor("a")],
        tag_matrix=_matrix({"a": (1.0, 1.0, 0.0, 0.0)}, bank),
        bank=bank,
    )
    root = 1.0 / math.sqrt(2.0)
    assert vectors[0] == pytest.approx((root, root, 0.0))


def test_transform_hellinger_takes_the_root_before_normalizing() -> None:
    """The Hellinger metric is the natural distance between probability vectors."""
    bank = _bank()
    vectors = selection_vectors(
        DecisionTagSelectionSpec(bank_id=bank.bank_id, transform="hellinger"),
        descriptors=[_descriptor("a")],
        tag_matrix=_matrix({"a": (0.25, 0.75, 0.0, 0.0)}, bank),
        bank=bank,
    )
    roots = (math.sqrt(0.25), math.sqrt(0.75), 0.0)
    norm = math.sqrt(sum(value**2 for value in roots))
    assert vectors[0] == pytest.approx(tuple(value / norm for value in roots))


def test_transform_logit_clamps_a_certain_answer_to_a_finite_coordinate() -> None:
    """A certain answer has infinite log-odds, and one such column would dominate."""
    bank = _bank()
    vectors = selection_vectors(
        DecisionTagSelectionSpec(bank_id=bank.bank_id, transform="logit"),
        descriptors=[_descriptor("a")],
        tag_matrix=_matrix({"a": (1.0, 0.5, 0.0, 0.0)}, bank),
        bank=bank,
    )
    extreme = math.log((1.0 - LOGIT_CLAMP) / LOGIT_CLAMP)
    odds = (extreme, 0.0, -extreme)
    norm = math.sqrt(sum(value**2 for value in odds))
    assert all(math.isfinite(value) for value in vectors[0])
    assert vectors[0] == pytest.approx(tuple(value / norm for value in odds))


def test_logit_separates_a_confident_case_that_l2_cannot_tell_apart() -> None:
    """Spreading confidence is the reason to pay for probabilities rather than booleans."""
    bank = _bank()
    matrix = _matrix({"confident": (0.9, 0.9, 0.0, 0.0), "unsure": (0.5, 0.5, 0.0, 0.0)}, bank)
    descriptors = [_descriptor("confident"), _descriptor("unsure")]

    def _cosine(transform: str) -> float:
        """Return the cosine between the two cases under one transform."""
        spec = DecisionTagSelectionSpec(bank_id=bank.bank_id, transform=transform)  # ty: ignore[invalid-argument-type]
        left, right = selection_vectors(spec, descriptors=descriptors, tag_matrix=matrix, bank=bank)
        return sum(first * second for first, second in zip(left, right, strict=True))

    assert _cosine("l2") == pytest.approx(1.0)
    assert _cosine("logit") < 0.95


def test_an_all_zero_row_is_refused_rather_than_placed_at_the_origin() -> None:
    """A case at the origin is nearest to nothing meaningful, so it must not pass silently."""
    bank = _bank()
    spec = DecisionTagSelectionSpec(bank_id=bank.bank_id, transform="l2")
    with pytest.raises(SelectionGeometryError, match="no direction") as caught:
        selection_vectors(
            spec,
            descriptors=[_descriptor("clean")],
            tag_matrix=_matrix({"clean": (0.0, 0.0, 0.0, 1.0)}, bank),
            bank=bank,
        )
    assert "clean" in str(caught.value)
    assert "logit" in str(caught.value)


def test_logit_keeps_an_all_zero_row_usable() -> None:
    """A wholly unremarkable case is ordinary evidence, so one transform must handle it."""
    bank = _bank()
    vectors = selection_vectors(
        DecisionTagSelectionSpec(bank_id=bank.bank_id, transform="logit"),
        descriptors=[_descriptor("clean")],
        tag_matrix=_matrix({"clean": (0.0, 0.0, 0.0, 0.0)}, bank),
        bank=bank,
    )
    assert math.isclose(math.sqrt(sum(value**2 for value in vectors[0])), 1.0, abs_tol=1e-9)


def test_an_untagged_descriptor_is_named() -> None:
    """Selection covers every candidate, so a gap in the matrix must say which trace."""
    bank = _bank()
    spec = DecisionTagSelectionSpec(bank_id=bank.bank_id)
    with pytest.raises(SelectionGeometryError, match="no row for trace trace-9"):
        selection_vectors(
            spec,
            descriptors=[_descriptor("trace-9")],
            tag_matrix=_matrix({"a": (1.0, 0.0, 0.0, 0.0)}, bank),
            bank=bank,
        )


def test_a_matrix_from_another_bank_is_refused() -> None:
    """Two banks' columns mean different things, so mixing them is not a geometry."""
    other = _bank(bank_id="other-v1")
    spec = DecisionTagSelectionSpec(bank_id="fixture-v1")
    with pytest.raises(SelectionGeometryError, match="tag matrix carries 'other-v1'"):
        selection_vectors(
            spec,
            descriptors=[_descriptor("a")],
            tag_matrix=_matrix({"a": (1.0, 0.0, 0.0, 0.0)}, other),
            bank=other,
        )


def test_a_matrix_missing_a_selected_column_is_refused() -> None:
    """A partially tagged matrix would quietly build a narrower geometry than asked for."""
    bank = _bank()
    narrow = TagMatrix(
        bank_id=bank.bank_id,
        bank_version=bank.version,
        question_ids=("sit.q0", "sit.q1"),
        view="observations",
        rows=(TagRow(trace_id="a", unit_hash="h", values=(1.0, 0.0)),),
    )
    with pytest.raises(SelectionGeometryError, match="missing columns"):
        selection_vectors(
            DecisionTagSelectionSpec(bank_id=bank.bank_id),
            descriptors=[_descriptor("a")],
            tag_matrix=narrow,
            bank=bank,
        )


def test_a_family_the_bank_does_not_declare_is_refused() -> None:
    """An empty column selection would produce zero-width vectors instead of a failure."""
    bank = _bank()
    with pytest.raises(SelectionGeometryError, match="has no family"):
        selection_vectors(
            DecisionTagSelectionSpec(bank_id=bank.bank_id, families=("risk",)),
            descriptors=[_descriptor("a")],
            tag_matrix=_matrix({"a": (1.0, 0.0, 0.0, 0.0)}, bank),
            bank=bank,
        )


def test_the_packaged_bank_resolves_without_being_passed() -> None:
    """The harness names a packaged bank by id, so families must resolve from that alone."""
    bank = packaged_question_bank()
    matrix = TagMatrix(
        bank_id=bank.bank_id,
        bank_version=bank.version,
        question_ids=bank.question_ids(),
        view="observations",
        rows=(
            TagRow(
                trace_id="a",
                unit_hash="h",
                values=tuple(1.0 if index == 0 else 0.0 for index in range(28)),
            ),
        ),
    )
    vectors = selection_vectors(
        DecisionTagSelectionSpec(bank_id=bank.bank_id),
        descriptors=[_descriptor("a")],
        tag_matrix=matrix,
    )
    assert len(vectors[0]) == 28
    assert vectors[0][0] == 1.0


def test_no_descriptors_produce_no_vectors() -> None:
    """An empty candidate set is a valid corpus state, not a geometry failure."""
    assert selection_vectors(HashingSelectionSpec(), descriptors=[]) == ()
    assert selection_vectors(DecisionTagSelectionSpec(bank_id="x-v1"), descriptors=[]) == ()


def test_the_discriminator_round_trips_through_json() -> None:
    """A spec is persisted and replayed, so its kind must select the same model back."""
    for spec in (
        HashingSelectionSpec(dimensions=32),
        DecisionTagSelectionSpec(bank_id="situation-v1", transform="hellinger"),
    ):
        restored = _SPEC_ADAPTER.validate_python(spec.model_dump(mode="json"))
        assert restored == spec
        assert type(restored) is type(spec)


def test_an_unknown_kind_is_refused_by_the_discriminator() -> None:
    """A spec from a newer revision must fail closed rather than pick a default geometry."""
    with pytest.raises(ValidationError):
        _SPEC_ADAPTER.validate_python({"kind": "embedding", "dimensions": 64})


def test_an_unknown_transform_is_refused() -> None:
    """Transform names are the contract, so a typo must not silently pass values through."""
    with pytest.raises(ValidationError):
        DecisionTagSelectionSpec(bank_id="x-v1", transform="sqrt")  # ty: ignore[invalid-argument-type]
