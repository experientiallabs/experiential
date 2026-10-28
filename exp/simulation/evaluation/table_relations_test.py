"""Task-level relations distinguish wrong effects without prescribing actor tool sequences."""

from itertools import product

import pytest

from exp.common.core.artifacts import JsonObject
from exp.simulation.evaluation.table_relations import TableRelation, assess_table_relation


def test_global_last_occurrence_rejects_adjacency_and_preserves_payloads() -> None:
    """Interleaved duplicates expose an incorrect shortcut; every retained cell matters."""
    relation = TableRelation(
        source="/before", target="/after", operation="last_per_key", key_columns=(1,)
    )
    before = [[1, 1, "old"], [2, 2, "keep"], [3, 1, "new"]]
    for after, expected in (
        ([[2, 2, "keep"], [3, 1, "new"]], "pass"),
        (before, "fail"),
        ([[1, 1, "old"], [2, 2, "keep"]], "fail"),
        ([[2, 2, "keep"], [3, 1, "wrong"]], "fail"),
        ([[3, 1, "new"], [2, 2, "keep"]], "fail"),
    ):
        assert assess_table_relation(relation, {"before": before, "after": after})[0] == expected


@pytest.mark.parametrize(
    "value", [{}, {"before": None, "after": []}, {"before": [[1], [2, 3]], "after": []}]
)
def test_unknown_or_ragged_tables_abstain(value: JsonObject) -> None:
    """Unavailable evidence must not become a false empty-table success."""
    relation = TableRelation(
        source="/before", target="/after", operation="last_per_key", key_columns=(0,)
    )
    assert assess_table_relation(relation, value)[0] == "unassessable"


def test_typed_keys_headers_and_unordered_alternatives() -> None:
    """Headers are retained exactly and equal-looking JSON types remain distinct keys."""
    relation = TableRelation(
        source="/before",
        target="/after",
        operation="first_per_key",
        key_columns=(0,),
        header_rows=1,
        ordering="any",
    )
    value: JsonObject = {
        "before": [["header", "value"], [1, "int"], [True, "bool"], [1.0, "float"], [1, "later"]],
        "after": [["header", "value"], [1.0, "float"], [True, "bool"], [1, "int"]],
    }
    assert assess_table_relation(relation, value)[0] == "pass"
    assert (
        assess_table_relation(relation, {**value, "after": [["header", "value"], [1, "int"]]})[0]
        == "fail"
    )


def test_cell_metadata_cannot_be_discarded() -> None:
    """Value-based grouping still checks entire retained cells, including types and formats."""
    relation = TableRelation(
        source="/before",
        target="/after",
        operation="last_per_key",
        key_columns=(0,),
        key_path="/value",
    )
    cell = {"value": 3, "data_type": "n", "number_format": "0.00"}
    assert assess_table_relation(relation, {"before": [[cell]], "after": [[cell]]})[0] == "pass"
    assert (
        assess_table_relation(
            relation, {"before": [[cell]], "after": [[{**cell, "number_format": "General"}]]}
        )[0]
        == "fail"
    )


def test_all_short_key_sequences_against_direct_membership_specification() -> None:
    """Exhaustive small inputs compare grouping with a separately expressed property."""
    for length in range(6):
        for keys in product((0, 1, 2), repeat=length):
            rows = [[index, key] for index, key in enumerate(keys)]
            for operation in ("first_per_key", "last_per_key"):
                rule = TableRelation.model_validate(
                    {
                        "source": "/before",
                        "target": "/after",
                        "operation": operation,
                        "key_columns": [1],
                    }
                )
                expected = [
                    row
                    for i, row in enumerate(rows)
                    if row[1] not in (keys[:i] if operation == "first_per_key" else keys[i + 1 :])
                ]
                assert assess_table_relation(rule, {"before": rows, "after": expected})[0] == "pass"
                if expected:
                    assert (
                        assess_table_relation(rule, {"before": rows, "after": expected[:-1]})[0]
                        == "fail"
                    )


def test_composite_and_null_keys_are_explicit() -> None:
    """All composite-key components matter, and unknown null semantics cannot yield success."""
    rule = TableRelation(
        source="/before", target="/after", operation="last_per_key", key_columns=(0, 1)
    )
    value: JsonObject = {
        "before": [["a", 1, "old"], ["a", 2, "other"], ["a", 1, "new"]],
        "after": [["a", 2, "other"], ["a", 1, "new"]],
    }
    assert assess_table_relation(rule, value)[0] == "pass"
    nulls: JsonObject = {"before": [[None, 1], [None, 1]], "after": [[None, 1]]}
    assert assess_table_relation(rule, nulls)[0] == "unassessable"
    assert assess_table_relation(rule.model_copy(update={"null_keys": "equal"}), nulls)[0] == "pass"


def test_unordered_identity_keeps_multiplicity_and_headers() -> None:
    """Accept a valid permutation but reject dropped duplicates or a changed header."""
    rule = TableRelation(
        source="/before", target="/after", operation="identity", header_rows=1, ordering="any"
    )
    value: JsonObject = {
        "before": [["header"], [1], [1], [2]],
        "after": [["header"], [2], [1], [1]],
    }
    assert assess_table_relation(rule, value)[0] == "pass"
    assert assess_table_relation(rule, {**value, "after": [["header"], [2], [1]]})[0] == "fail"
    assert (
        assess_table_relation(rule, {**value, "after": [["changed"], [2], [1], [1]]})[0] == "fail"
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"key_columns": [True]},
        {"key_columns": ["1"]},
        {"key_columns": [-1]},
        {"key_columns": [128]},
        {"key_columns": [0, 0]},
        {"source": "/after"},
        {"operation": "identity"},
        {"key_path": "relative"},
    ],
)
def test_ambiguous_rule_definitions_refuse(changes: JsonObject) -> None:
    """Validation rejects lossy configuration coercion and tautological predicates."""
    with pytest.raises(ValueError):
        TableRelation.model_validate(
            {
                "source": "/before",
                "target": "/after",
                "operation": "last_per_key",
                "key_columns": [0],
                **changes,
            }
        )


def test_bounded_table_evidence() -> None:
    """Excessive, non-finite and unavailable key evidence cannot be graded successful."""
    rule = TableRelation(
        source="/before", target="/after", operation="last_per_key", key_columns=(0,)
    )
    for rows in (
        [[1]] * 10001,
        [[1] * 129],
        [[1] * 11] * 10000,
        [[float("nan")]],
        [["x" * (8 * 1024 * 1024)]],
        [[{}]],
    ):
        assert assess_table_relation(rule, {"before": rows, "after": rows})[0] == "unassessable"
