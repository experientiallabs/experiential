"""Bounded relational checks over observed tables, independent of constructed answer files."""

from collections import Counter
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel, JsonObject, JsonValue, canonical_json_bytes
from exp.simulation.reconstruction.models import path_parts, resolve


class TableRelation(ContractModel):
    """Compare whole observed rows under a task operation, not a reference trajectory.

    Attributes:
        source: JSON pointer to the complete original rectangular array of rows.
        target: JSON pointer to the complete resulting rectangular array of rows.
        operation: Preserve all rows or retain the first/last row for each global key.
        key_columns: Zero-based composite key positions, required only for grouped operations.
        key_path: Pointer inside each key cell, such as /value for typed workbook cells.
        header_rows: Exact prefix to preserve, excluded from grouping and reordering.
        ordering: Preserve survivor order, or accept any permutation with multiplicity.
        null_keys: Whether observed null keys form a group or prevent assessment.
            Missing evidence always prevents assessment, independently of this setting.
    """

    source: str
    target: str
    operation: Literal["identity", "first_per_key", "last_per_key"]
    key_columns: tuple[Annotated[int, Field(ge=0, le=127, strict=True)], ...] = Field(
        default=(), max_length=128
    )
    key_path: str = ""
    header_rows: int = Field(default=0, ge=0, le=32, strict=True)
    ordering: Literal["preserve", "any"] = "preserve"
    null_keys: Literal["equal", "unassessable"] = "unassessable"

    @model_validator(mode="after")
    def constrained(self) -> Self:
        """Refuse ambiguous identities and repeated key positions before reading rows."""
        for pointer in (self.source, self.target, self.key_path):
            path_parts(pointer)
        if self.source == self.target:
            raise ValueError("table relation needs distinct source and target observations")
        if self.operation == "identity":
            if self.key_columns or self.key_path or self.null_keys != "unassessable":
                raise ValueError("identity does not accept key settings")
        elif not self.key_columns:
            raise ValueError("grouped table relation requires key_columns")
        if len(set(self.key_columns)) != len(self.key_columns):
            raise ValueError("key_columns must be unique")
        return self


def _rows(value: JsonValue) -> list[list[JsonValue]]:
    """Require complete finite rectangular evidence within fixed cell and byte ceilings."""
    if not isinstance(value, list) or len(value) > 10000:
        raise ValueError("table is unavailable or exceeds 10000 rows")
    result: list[list[JsonValue]] = []
    width = None
    size = 0
    for row in value:
        if not isinstance(row, list) or not 1 <= len(row) <= 128:
            raise ValueError("table row must contain 1 through 128 observed cells")
        if width is None:
            width = len(row)
        if len(row) != width or len(value) * width > 100000:
            raise ValueError("table is ragged or exceeds 100000 cells")
        size += len(canonical_json_bytes(row))
        if size > 8 * 1024 * 1024:
            raise ValueError("table exceeds 8 MiB of observed values")
        result.append(row)
    return result


def _key(rule: TableRelation, row: list[JsonValue]) -> bytes:
    """Use canonical JSON scalar keys without coercion or missing-value imputation."""
    values: list[JsonValue] = []
    for column in rule.key_columns:
        if column >= len(row):
            raise ValueError("key column is absent from observed table")
        value = resolve(row[column], rule.key_path)
        if isinstance(value, (dict, list)):
            raise ValueError("key cell must resolve to an observed scalar")
        if value is None and rule.null_keys == "unassessable":
            raise ValueError("null key has no declared grouping semantics")
        values.append(value)
    return canonical_json_bytes(values)


def assess_table_relation(
    rule: TableRelation, outcome: JsonObject
) -> tuple[Literal["pass", "fail", "unassessable"], str]:
    """Check whole rows, headers, types and multiplicity under canonical JSON equality.

    Evidence must be complete for the declared table; this function cannot discover
    unrecorded source rows. Numeric representations remain distinct.
    """
    try:
        source, target = (_rows(resolve(outcome, path)) for path in (rule.source, rule.target))
        if len(source) < rule.header_rows:
            raise ValueError("source does not contain the declared header prefix")
        header, data = source[: rule.header_rows], source[rule.header_rows :]
        if rule.operation == "identity":
            expected = data
        else:
            chosen: dict[bytes, int] = {}
            for index, row in enumerate(data):
                key = _key(rule, row)
                if rule.operation == "last_per_key" or key not in chosen:
                    chosen[key] = index
            expected = [data[index] for index in sorted(chosen.values())]
        if canonical_json_bytes(target[: rule.header_rows]) != canonical_json_bytes(header):
            return "fail", "table header changed or is absent"
        actual = target[rule.header_rows :]
        if rule.ordering == "preserve":
            equal = canonical_json_bytes(actual) == canonical_json_bytes(expected)
        else:
            equal = Counter(map(canonical_json_bytes, actual)) == Counter(
                map(canonical_json_bytes, expected)
            )
    except (ValueError, TypeError, OverflowError, RecursionError) as error:
        return "unassessable", f"table relation evidence unavailable: {error}"
    return (
        ("pass", "observed table satisfies the declared row relation")
        if equal
        else ("fail", f"observed table violates {rule.operation} with {rule.ordering} order")
    )
