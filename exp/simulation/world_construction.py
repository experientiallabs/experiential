"""Compile bounded declarative tables into coherent, unqualified world assets.

This constructs initial assets only. Runtime events, task verifiers and evidence admission
retain their existing owners. Column order is lexical, independent of JSON object order.
"""

import io
import re
import sys
import zlib
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, StrictBool, StrictFloat, StrictInt, StrictStr, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    SourceIdentity,
    canonical_json_bytes,
    sha256_bytes,
    sha256_json,
    stable_id,
)
from exp.common.project import ArtifactAlreadyExistsError, ArtifactStore, artifact_input

Cell = (
    None
    | StrictBool
    | StrictInt
    | Annotated[StrictFloat, Field(allow_inf_nan=False)]
    | Annotated[StrictStr, Field(max_length=32767)]
)
Name = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,30}$")]


class ValuesColumn(ContractModel):
    """Explicit scalar observations, with a single value broadcast to every row.

    Attributes:
        kind: Literal operator discriminator.
        values: One broadcast value or exactly as many values as the table has rows.
    """

    kind: Literal["values"]
    values: tuple[Cell, ...] = Field(min_length=1, max_length=10000)


class SequenceColumn(ContractModel):
    """An authored integer sequence, not an empirically estimated distribution.

    Attributes:
        kind: Sequence operator discriminator.
        start: First integer, default one.
        step: Increment per row, including zero or negative increments.
    """

    kind: Literal["sequence"]
    start: int = Field(default=1, strict=True, ge=-(2**53), le=2**53)
    step: int = Field(default=1, strict=True, ge=-(2**53), le=2**53)


class LookupColumn(ContractModel):
    """Join a local key to an exact, unique, typed source key without coercion.

    Attributes:
        kind: Lookup operator discriminator.
        table: Source table, which may itself depend on other tables.
        key_column: Unique, non-null source column.
        value_column: Source value returned for each local key.
        key_from: Local column supplying the lookup keys.
    """

    kind: Literal["lookup"]
    table: Name
    key_column: Name
    value_column: Name
    key_from: Name


Column = Annotated[ValuesColumn | SequenceColumn | LookupColumn, Field(discriminator="kind")]


class TablePlan(ContractModel):
    """Finite initial facts with deterministic row and column order.

    Attributes:
        rows: Explicit row count, one through ten thousand.
        columns: Named generators; rendered in lexical name order.
        key: Optional primary key, required to be unique and non-null.
    """

    rows: int = Field(ge=1, le=10000, strict=True)
    columns: dict[Name, Column] = Field(min_length=1, max_length=64)
    key: Name | None = None


class ConstructionPlan(ContractModel):
    """Authored initial assets with retained assumptions and source provenance.

    Attributes:
        name: Human-readable construction name.
        assumptions: Required limitations; generated data is never empirical by declaration.
        tables: At most 32 tables, with at most 100,000 materialized cells in total.
        workbooks: Flat XLSX filenames mapped to ordered table/sheet names, without headers.
            JSON always includes explicit column names. At most 100,000 rendered XLSX cells.
    """

    name: str = Field(min_length=1, max_length=128)
    assumptions: tuple[Annotated[str, Field(min_length=1, max_length=4000)], ...] = Field(
        min_length=1, max_length=64
    )
    tables: dict[Name, TablePlan] = Field(min_length=1, max_length=32)
    workbooks: dict[str, tuple[Name, ...]] = Field(default_factory=dict, max_length=16)

    @model_validator(mode="after")
    def bounded_layout(self) -> Self:
        """Reject unknown targets, collisions and excessive work before allocating rows."""
        if sum(table.rows * len(table.columns) for table in self.tables.values()) > 100000:
            raise ValueError("construction exceeds 100000 cells; split the construction")
        rendered = 0
        seen: set[str] = set()
        for filename, sheets in self.workbooks.items():
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\.xlsx", filename):
                raise ValueError("workbook needs a flat safe .xlsx filename")
            if filename.casefold() in seen:
                raise ValueError("workbook filenames collide case-insensitively")
            seen.add(filename.casefold())
            if not sheets or len({sheet.casefold() for sheet in sheets}) != len(sheets):
                raise ValueError("workbook requires distinct sheet names")
            for sheet in sheets:
                if sheet not in self.tables:
                    raise ValueError(f"unknown workbook table {sheet}; declare it first")
                table = self.tables[sheet]
                rendered += table.rows * len(table.columns)
        if rendered > 100000:
            raise ValueError("workbooks exceed 100000 rendered cells; reduce repeated sheets")
        for name, table in self.tables.items():
            if table.key is not None and table.key not in table.columns:
                raise ValueError(f"unknown primary key {name}.{table.key}")
            for column in table.columns.values():
                if isinstance(column, LookupColumn) and (
                    column.table not in self.tables
                    or column.key_column not in self.tables[column.table].columns
                    or column.value_column not in self.tables[column.table].columns
                    or column.key_from not in table.columns
                ):
                    raise ValueError(f"unknown lookup dependency in {name}; declare its columns")
        return self


class ConstructionSpec(ConstructionPlan):
    """An asset plan bound to coordinator-selected immutable source provenance.

    Attributes:
        sources: Evidence supporting or inspiring construction; generated facts remain proposals.
    """

    sources: tuple[ArtifactInput, ...] = Field(default=(), max_length=128)


class MaterializedTable(ContractModel):
    """Named columns and ordered scalar rows shared by every renderer.

    Attributes:
        columns: Lexically ordered field names.
        rows: Exact rows, with no hidden headers or implicit values.
    """

    columns: tuple[str, ...]
    rows: tuple[tuple[Cell, ...], ...]


def _unique_keys(values: tuple[Cell, ...], location: str) -> dict[bytes, int]:
    """Index typed scalar keys without treating booleans, numbers or strings as synonyms."""
    keys: dict[bytes, int] = {}
    for index, value in enumerate(values):
        if value is None:
            raise ValueError(f"null key in {location}; provide non-null keys")
        key = canonical_json_bytes(value)
        if key in keys:
            raise ValueError(f"duplicate key in {location}; provide unique keys")
        keys[key] = index
    return keys


def materialize_tables(spec: ConstructionSpec) -> dict[str, MaterializedTable]:
    """Resolve the initial dependency graph or refuse the entire construction."""
    spec = ConstructionSpec.model_validate_json(spec.model_dump_json())
    pending = {
        (name, field): column
        for name, table in spec.tables.items()
        for field, column in table.columns.items()
    }
    resolved: dict[tuple[str, str], tuple[Cell, ...]] = {}
    scalar_bytes = 0
    while pending:
        progressed = False
        for (name, field), column in tuple(pending.items()):
            count = spec.tables[name].rows
            if isinstance(column, ValuesColumn):
                if len(column.values) not in (1, count):
                    raise ValueError(f"length of {name}.{field} must be 1 or {count}")
                values = column.values * count if len(column.values) == 1 else column.values
            elif isinstance(column, SequenceColumn):
                values = tuple(column.start + i * column.step for i in range(count))
            else:
                dependencies = (
                    (name, column.key_from),
                    (column.table, column.key_column),
                    (column.table, column.value_column),
                )
                if any(key not in resolved for key in dependencies):
                    continue
                local, source_keys, source_values = (resolved[key] for key in dependencies)
                lookup = _unique_keys(source_keys, f"{column.table}.{column.key_column}")
                selected: list[Cell] = []
                for key in local:
                    encoded = canonical_json_bytes(key)
                    if encoded not in lookup:
                        raise ValueError(
                            f"missing lookup key in {name}.{field}; supply referenced facts"
                        )
                    selected.append(source_values[lookup[encoded]])
                values = tuple(selected)
            for value in values:
                scalar_bytes += len(canonical_json_bytes(value))
                if scalar_bytes > 8 * 1024 * 1024:
                    raise ValueError("construction exceeds 8 MiB of facts; reduce text or rows")
            resolved[(name, field)] = values
            del pending[(name, field)]
            progressed = True
        if not progressed:
            raise ValueError(
                f"construction dependency cycle: {sorted(pending)}; "
                "break the cycle with explicit facts"
            )
    result: dict[str, MaterializedTable] = {}
    for name, table in sorted(spec.tables.items()):
        if table.key is not None:
            _unique_keys(resolved[(name, table.key)], f"{name}.{table.key}")
        columns = tuple(sorted(table.columns))
        result[name] = MaterializedTable(
            columns=columns,
            rows=tuple(zip(*(resolved[(name, column)] for column in columns), strict=True)),
        )
    return result


def _xlsx(tables: dict[str, MaterializedTable], sheets: tuple[str, ...]) -> bytes:
    """Render literal cells and refuse any XLSX representation that loses information."""
    # Importers that only declare construction schemas must not need the artifact-edits extra.
    from openpyxl import Workbook, load_workbook
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

    from exp.runtime.environments import workbook_edits

    book = Workbook()
    book.remove(book.worksheets[0])
    book.properties.created = datetime(2000, 1, 1)
    book.properties.modified = datetime(2000, 1, 1)
    for name in sheets:
        sheet = book.create_sheet(name)
        for row_index, row in enumerate(tables[name].rows, 1):
            for column_index, value in enumerate(row, 1):
                if isinstance(value, str) and (not value or ILLEGAL_CHARACTERS_RE.search(value)):
                    raise ValueError(
                        f"XLSX cannot preserve text in {name} row {row_index}; "
                        "use JSON or revise the asset"
                    )
                cell = sheet.cell(row_index, column_index, value)
                if isinstance(value, str):
                    cell.data_type = "s"
    buffer = io.BytesIO()
    book.save(buffer)
    book.close()
    payload = workbook_edits.canonical_workbook_bytes(buffer.getvalue())
    restored = load_workbook(io.BytesIO(payload))
    try:
        for name in sheets:
            sheet = restored[name]
            if (sheet.max_row, sheet.max_column) != (
                len(tables[name].rows),
                len(tables[name].columns),
            ):
                raise ValueError(f"XLSX loses dimensions of {name}; use JSON or explicit values")
            for row_index, row in enumerate(tables[name].rows, 1):
                for column_index, expected in enumerate(row, 1):
                    actual = sheet.cell(row_index, column_index).value
                    if type(actual) is not type(expected) or actual != expected:
                        raise ValueError(
                            f"XLSX loses value in {name} row {row_index} column {column_index}; "
                            "use JSON or represent it as text"
                        )
    finally:
        restored.close()
    return payload


def compile_construction(spec: ConstructionSpec) -> dict[str, bytes]:
    """Produce bounded public assets without importing generated code or making calls."""
    tables = materialize_tables(spec)
    files = {
        "assets/tables.json": canonical_json_bytes(
            {name: table.model_dump(mode="json") for name, table in tables.items()}
        )
    }
    if len(files["assets/tables.json"]) > 8 * 1024 * 1024:
        raise ValueError("construction exceeds 8 MiB of facts; reduce text or rows")
    files.update(
        {f"assets/{name}": _xlsx(tables, sheets) for name, sheets in sorted(spec.workbooks.items())}
    )
    return files


def _identity(spec: ConstructionSpec, code_revision: str) -> str:
    """Bind source bytes as well as the caller's revision, including dirty local edits."""
    from exp.runtime.environments import workbook_edits

    return stable_id(
        "world-construction",
        {
            "spec": sha256_json(spec),
            "code": code_revision,
            "compiler": sha256_bytes(Path(__file__).read_bytes()),
            "renderer": sha256_bytes(Path(workbook_edits.__file__).read_bytes()),
            "python": sys.version,
            "zlib": zlib.ZLIB_RUNTIME_VERSION,
            "dependencies": {name: version(name) for name in ("openpyxl", "et-xmlfile")},
        },
    )


def _verify_sources(store: ArtifactStore, spec: ConstructionSpec) -> tuple[ArtifactInput, ...]:
    """Refuse stale references before rendering or reusing constructed assets."""
    sources = tuple(sorted(set(spec.sources), key=lambda item: (item.artifact_id, item.sha256)))
    for reference in sources:
        if artifact_input(store.read(reference.artifact_id).manifest) != reference:
            raise ValueError("construction source changed; supply its exact saved reference")
    return sources


def preflight_construction(store: ArtifactStore, spec: ConstructionSpec) -> dict[str, bytes]:
    """Verify immutable sources and rendering without publishing or invoking providers."""
    _verify_sources(store, spec)
    return compile_construction(spec)


def persist_construction(
    store: ArtifactStore, spec: ConstructionSpec, *, created_at: datetime, code_revision: str
) -> ArtifactInput:
    """Publish authored assets with generated authority, never as independent goal evidence."""
    inputs = _verify_sources(store, spec)
    identity = _identity(spec, code_revision)
    if store.exists(identity):
        reference = artifact_input(store.read(identity).manifest)
        load_construction(store, reference)
        return reference
    files = compile_construction(spec)
    files["construction.json"] = canonical_json_bytes(spec)
    try:
        manifest = store.write(
            artifact_id=identity,
            artifact_type="world-construction",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=created_at,
                code_revision=code_revision,
                inputs=inputs,
                source=SourceIdentity(kind="generated", source_id=identity),
            ),
            files=files,
        )
    except ArtifactAlreadyExistsError:
        reference = artifact_input(store.read(identity).manifest)
        load_construction(store, reference)
        return reference
    return artifact_input(manifest)


def load_construction(store: ArtifactStore, reference: ArtifactInput) -> ConstructionSpec:
    """Recompute assets and provenance; a stored completion flag cannot validate this world."""
    manifest = store.read(reference.artifact_id).manifest
    if artifact_input(manifest) != reference or manifest.artifact_type != "world-construction":
        raise ValueError("construction reference differs from its saved artifact")
    spec = ConstructionSpec.model_validate_json(
        store.read_bytes(reference.artifact_id, "construction.json")
    )
    if (
        manifest.inputs != _verify_sources(store, spec)
        or _identity(spec, manifest.code_revision) != reference.artifact_id
    ):
        raise ValueError("construction source or compiler changed; construct the assets again")
    files = compile_construction(spec)
    files["construction.json"] = canonical_json_bytes(spec)
    store.write_or_verify_exact(
        artifact_id=reference.artifact_id,
        artifact_type="world-construction",
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=manifest.created_at,
            code_revision=manifest.code_revision,
            inputs=manifest.inputs,
            source=SourceIdentity(kind="generated", source_id=reference.artifact_id),
        ),
        files=files,
    )
    return spec
