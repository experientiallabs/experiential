"""Bounded XLSX row edits that preserve typed cells and refuse unsupported dependencies."""

import io
import re
import sys
import zlib
from collections.abc import Mapping
from datetime import date, datetime, time
from importlib.metadata import version
from typing import Self
from xml.dom import minidom
from xml.etree import ElementTree
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from openpyxl import load_workbook
from openpyxl.cell.cell import Cell, MergedCell
from openpyxl.workbook.workbook import Workbook
from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ContractModel,
    JsonObject,
    JsonValue,
    sha256_json,
    validate_artifact_file_path,
)
from exp.common.models import ToolCall
from exp.runtime.environments.artifact_edits import MAX_ARTIFACT_BYTES, NativeArtifactOutput
from exp.runtime.environments.world_contracts import WorldAccess

# Identity of the code that decides this artifact's behavior, in place of its source text.
# Bump it when behavior changes. exp/tests/code_protocols_test.py fails when the covered
# source changes until it is bumped or, for a pure refactor, its digest is updated.
XLSX_ROW_EDITS_PROTOCOL = "xlsx-row-edits-v3"

_ALLOWED_PART = re.compile(
    r"(?:\[Content_Types\]\.xml|_rels/\.rels|docProps/(?:app|core)\.xml|"
    r"xl/(?:workbook\.xml|_rels/workbook\.xml\.rels|styles\.xml|sharedStrings\.xml|"
    r"theme/theme\d+\.xml|worksheets/sheet\d+\.xml))"
)


def _require_workbook_source(source: str) -> None:
    """Require an explicit workbook path inside the detached artifact namespace."""
    if validate_artifact_file_path(source).as_posix() != source or not source.endswith(".xlsx"):
        raise ValueError("workbook source must be a canonical relative .xlsx path")


class DeleteWorkbookRows(ContractModel):
    """One explicit row interval, interpreted in the current workbook state.

    Attributes:
        sheet: Exact sheet name.
        start: One-based first row to remove.
        count: Number of existing rows to remove.
    """

    sheet: str = Field(min_length=1)
    start: int = Field(ge=1, strict=True)
    count: int = Field(ge=1, le=100000, strict=True)


class WorkbookEditArguments(ContractModel):
    """An explicit input/output workbook and sequential deletion plan.

    Attributes:
        source: Registered input or earlier output path, never a host path.
        destination: Declared output path.
        deletions: Ordered row intervals applied sequentially, never sorted or guessed.
    """

    source: str
    destination: str
    deletions: tuple[DeleteWorkbookRows, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def safe_paths(self) -> Self:
        """Refuse noncanonical paths and formats outside this adapter's contract."""
        for path in (self.source, self.destination):
            if validate_artifact_file_path(path).as_posix() != path or not path.endswith(".xlsx"):
                raise ValueError("workbook paths must be canonical relative .xlsx paths")
        return self


class WorkbookReadArguments(ContractModel):
    """A bounded rectangular observation of a current native workbook.

    Attributes:
        source: Registered workbook path, never a host path.
        sheet: Exact worksheet name.
        start_row: One-based first row.
        row_count: Requested existing rows, bounded to 128.
        start_column: One-based first column.
        column_count: Requested existing columns, bounded to 64.
    """

    source: str
    sheet: str = Field(min_length=1)
    start_row: int = Field(default=1, ge=1, strict=True)
    row_count: int = Field(default=1, ge=1, le=128, strict=True)
    start_column: int = Field(default=1, ge=1, strict=True)
    column_count: int = Field(default=1, ge=1, le=64, strict=True)

    @model_validator(mode="after")
    def safe_source(self) -> Self:
        """Refuse paths outside the detached artifact namespace."""
        _require_workbook_source(self.source)
        return self


class WorkbookDescriptionArguments(ContractModel):
    """Paginated native worksheet names and dimensions, without reading cell contents.

    Attributes:
        source: Registered workbook path, never a host path.
        offset: Zero-based first worksheet in workbook order.
        limit: Maximum worksheet descriptions, bounded to 128.
    """

    source: str
    offset: int = Field(default=0, ge=0, strict=True)
    limit: int = Field(default=32, ge=1, le=128, strict=True)

    @model_validator(mode="after")
    def safe_source(self) -> Self:
        """Use the same canonical file namespace as explicit cell reads."""
        _require_workbook_source(self.source)
        return self


class WorkbookEditAdapter:
    """Pure native XLSX transformation; no shell, filesystem inputs or reference answers."""

    @property
    def definition(self) -> JsonObject:
        """Bind the adapter protocol, dependencies and supported operation schema."""
        return {
            "name": XLSX_ROW_EDITS_PROTOCOL,
            "python": sys.version,
            "zlib": zlib.ZLIB_RUNTIME_VERSION,
            "dependencies": {name: version(name) for name in ("openpyxl", "et-xmlfile")},
            "tool": "edit_workbook",
            "arguments_schema": WorkbookEditArguments.model_json_schema(),
            "read_tool": "inspect_workbook",
            "read_arguments_schema": WorkbookReadArguments.model_json_schema(),
            "describe_tool": "describe_workbook",
            "describe_arguments_schema": WorkbookDescriptionArguments.model_json_schema(),
            "projection": "ordered sheets and rectangular cells: value, data_type, number_format",
            "normalization": "sorted ZIP members, fixed ZIP timestamps and core modified timestamp",
            "effects": "detached artifact bytes only",
        }

    def access(self, action: ToolCall) -> WorldAccess:
        """Resolve the workbook one edit reads and the workbook it writes, without applying it."""
        if action.name == "inspect_workbook":
            read = WorkbookReadArguments.model_validate(action.arguments)
            return WorldAccess(reads=frozenset({read.source}))
        if action.name == "describe_workbook":
            description = WorkbookDescriptionArguments.model_validate(action.arguments)
            return WorldAccess(reads=frozenset({description.source}))
        if action.name != "edit_workbook":
            raise ValueError(
                "unsupported tool; use edit_workbook, inspect_workbook or describe_workbook"
            )
        arguments = WorkbookEditArguments.model_validate(action.arguments)
        return WorldAccess(
            reads=frozenset({arguments.source}), writes=frozenset({arguments.destination})
        )

    def apply(
        self,
        files: Mapping[str, bytes],
        operations: tuple[ToolCall, ...],
        output_paths: tuple[str, ...],
    ) -> NativeArtifactOutput:
        """Execute declared deletions while preserving native types and explicit ordering."""
        current = dict(files)
        produced: dict[str, bytes] = {}
        projection: JsonObject = {}
        for action in operations:
            if action.name == "describe_workbook":
                description = WorkbookDescriptionArguments.model_validate(action.arguments)
                if description.source not in current:
                    raise ValueError("workbook description source is absent")
                workbook = read_supported_workbook(current[description.source])
                total = len(workbook.worksheets)
                end = min(total, description.offset + description.limit)
                projection[description.source] = {
                    "offset": description.offset,
                    "total_sheets": total,
                    "next_offset": end if end < total else None,
                    "sheets": [
                        {
                            "name": sheet.title,
                            "max_row": sheet.max_row,
                            "max_column": sheet.max_column,
                        }
                        for sheet in workbook.worksheets[description.offset : end]
                    ],
                }
                continue
            if action.name == "inspect_workbook":
                read = WorkbookReadArguments.model_validate(action.arguments)
                if read.source not in current:
                    raise ValueError("workbook probe source is absent")
                workbook = read_supported_workbook(current[read.source])
                if read.sheet not in workbook.sheetnames:
                    raise ValueError("workbook probe names a missing sheet")
                sheet = workbook[read.sheet]
                row_end = read.start_row + read.row_count - 1
                column_end = read.start_column + read.column_count - 1
                if row_end > sheet.max_row or column_end > sheet.max_column:
                    raise ValueError("workbook probe extends beyond existing cells")
                projection[read.source] = {
                    "sheet": read.sheet,
                    "max_row": sheet.max_row,
                    "max_column": sheet.max_column,
                    "start_row": read.start_row,
                    "start_column": read.start_column,
                    "rows": [
                        [_cell_projection(cell) for cell in row]
                        for row in sheet.iter_rows(
                            min_row=read.start_row,
                            max_row=row_end,
                            min_col=read.start_column,
                            max_col=column_end,
                        )
                    ],
                }
                continue
            if action.name != "edit_workbook":
                raise ValueError(
                    "unsupported tool; use edit_workbook, inspect_workbook or describe_workbook"
                )
            arguments = WorkbookEditArguments.model_validate(action.arguments)
            if arguments.destination not in output_paths:
                raise ValueError("workbook edit targets an undeclared output")
            if arguments.source not in current:
                raise ValueError("workbook edit source is not in the registered native checkpoint")
            workbook = read_supported_workbook(current[arguments.source])
            for deletion in arguments.deletions:
                if deletion.sheet not in workbook.sheetnames:
                    raise ValueError("workbook edit names a missing sheet")
                sheet = workbook[deletion.sheet]
                if deletion.start + deletion.count - 1 > sheet.max_row:
                    raise ValueError("workbook deletion extends beyond existing rows")
                sheet.delete_rows(deletion.start, deletion.count)
            intended = workbook_projection(workbook)
            buffer = io.BytesIO()
            workbook.save(buffer)
            payload = canonical_workbook_bytes(buffer.getvalue())
            actual = workbook_projection(read_supported_workbook(payload))
            if sha256_json(actual) != sha256_json(intended):
                raise ValueError(
                    "workbook serialization changes retained cell values, types or formats; "
                    "use an adapter that preserves the declared precision"
                )
            current[arguments.destination] = payload
            produced[arguments.destination] = payload
            projection[arguments.destination] = actual
        return NativeArtifactOutput(produced, projection)


def read_supported_workbook(payload: bytes) -> Workbook:
    """Refuse large archives and workbook features that row deletion cannot safely update."""
    if len(payload) > MAX_ARTIFACT_BYTES:
        raise ValueError("workbook exceeds compressed byte allowance")
    with ZipFile(io.BytesIO(payload)) as archive:
        members = archive.infolist()
        if (
            len(members) > 256
            or len({row.filename for row in members}) != len(members)
            or sum(row.file_size for row in members) > MAX_ARTIFACT_BYTES
        ):
            raise ValueError("workbook archive exceeds entry or expanded byte allowance")
        for member in members:
            if not _ALLOWED_PART.fullmatch(member.filename):
                raise ValueError("unsupported workbook part; use a full sandbox adapter")
            content = archive.read(member)
            if b"<!DOCTYPE" in content or b"<!ENTITY" in content:
                raise ValueError("workbook XML declarations are unsupported")
            if member.filename.endswith(".rels"):
                if any(
                    node.get("TargetMode") == "External" for node in ElementTree.fromstring(content)
                ):
                    raise ValueError("external workbook relationships are unsupported")
    workbook = load_workbook(io.BytesIO(payload), data_only=False, keep_links=False)
    if workbook.defined_names or any(sheet.defined_names for sheet in workbook):
        raise ValueError("defined names require a full workbook dependency adapter")
    if sum(sheet.max_row * sheet.max_column for sheet in workbook) > 100000:
        raise ValueError("workbook exceeds the projected cell allowance")
    for sheet in workbook:
        if (
            sheet.merged_cells.ranges
            or sheet.tables
            or sheet.conditional_formatting
            or sheet.data_validations.count
            or sheet.auto_filter.ref
            or sheet.print_area
            or sheet.print_title_rows
            or sheet.print_title_cols
            or sheet.freeze_panes
            or sheet.row_breaks.count
            or any(
                row.hidden or row.has_style or row.customHeight or row.outlineLevel or row.collapsed
                for row in sheet.row_dimensions.values()
            )
        ):
            raise ValueError(
                "workbook row dependencies are unsupported; use a full sandbox adapter"
            )
        if any(cell.data_type == "f" or cell.comment is not None for row in sheet for cell in row):
            raise ValueError("formulas and comments require a full workbook dependency adapter")
    return workbook


def workbook_projection(workbook: Workbook) -> JsonObject:
    """Read actual ordered cell values, types and number formats from a native workbook."""
    sheets: list[JsonValue] = []
    for sheet in workbook:
        rows: list[JsonValue] = []
        for row in sheet:
            rows.append([_cell_projection(cell) for cell in row])
        sheets.append({"name": sheet.title, "rows": rows})
    return {"sheets": sheets}


def _cell_projection(cell: Cell | MergedCell) -> JsonObject:
    """Preserve the same typed value for full internal checks and bounded public reads."""
    value = cell.value
    if isinstance(value, (datetime, date, time)):
        value = value.isoformat()
    if not isinstance(value, (str, int, float, bool, type(None))):
        raise ValueError("unsupported native cell value")
    return {"value": value, "data_type": cell.data_type, "number_format": cell.number_format}


def canonical_workbook_bytes(payload: bytes) -> bytes:
    """Remove writer-clock variance without changing worksheet values or types."""
    output = io.BytesIO()
    with (
        ZipFile(io.BytesIO(payload)) as source,
        ZipFile(output, "w", compression=ZIP_DEFLATED) as target,
    ):
        for name in sorted(source.namelist()):
            content = source.read(name)
            if name == "docProps/core.xml":
                document = minidom.parseString(content)
                modified = document.getElementsByTagNameNS("http://purl.org/dc/terms/", "modified")
                if len(modified) != 1 or not isinstance(modified[0].firstChild, minidom.Text):
                    raise ValueError("workbook writer omitted its modification timestamp")
                modified[0].firstChild.nodeValue = "2000-01-01T00:00:00Z"
                content = document.toxml(encoding="utf-8")
                document.unlink()
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            target.writestr(info, content)
    return output.getvalue()
