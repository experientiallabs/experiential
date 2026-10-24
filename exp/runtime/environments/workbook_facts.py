"""Workbook instance facts: a typed, lossless, size-bounded encoding of one starting workbook.

The authority is openpyxl at the installed version, loaded the way an agent's program loads a
workbook: ``load_workbook(path)`` for stored values and formulas, and ``data_only=True`` for the
cached results of formulas. The facts state exactly what that reader returns for every stored
cell (its typed value, formula, cached value, stored cell type and number format) plus sheet
order and state, dimensions, merged ranges, comments, hyperlinks, defined names, tables, and the
ranges of data validations, conditional formats and the auto filter.

Every encoding is checked before it is returned. The JSON value is serialized and parsed back,
decoded into typed Python values, and compared cell by cell with both loads by Python type and
exact ``repr``; any difference refuses the workbook. Content the encoding cannot carry is
refused with :class:`WorkbookFactsRefused`, never dropped: unknown cell value types, data-table
formulas, encodings over the byte bound, and archive parts whose content lies outside the
encoding (drawings and images, charts, pivot tables, external links, macros and ActiveX, form
controls, threaded comments, unrecognized parts). A deriver that sets ``allow_unrepresented``
accepts those parts instead, and the facts then list each one by name and kind. Presentation
(fonts, fills, borders, widths, heights, hidden rows, views and print setup) is not instance
content and is outside the encoding by contract.

The encoding is row oriented so that a prompt reads a workbook as rows, and it is explained by
the ``legend`` carried in every value. :func:`slice_workbook_facts` cuts a bounded view (full
inventory, selected cells) from an encoded value for prompts that need only a read set.
"""

from __future__ import annotations

import io
import json
import math
import re
import warnings
from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from hashlib import sha256
from importlib.metadata import version
from typing import cast
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

from openpyxl import load_workbook
from openpyxl.cell.cell import Cell, MergedCell
from openpyxl.chartsheet.chartsheet import Chartsheet
from openpyxl.utils.cell import (
    column_index_from_string,
    coordinate_from_string,
    get_column_letter,
    range_boundaries,
)
from openpyxl.workbook.workbook import Workbook
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.worksheet.formula import ArrayFormula, DataTableFormula
from openpyxl.worksheet.worksheet import Worksheet

from exp.common.core.artifacts import JsonObject, JsonValue, canonical_json_bytes
from exp.common.tasks.facts import (
    FactProvenance,
    FactValidation,
    InstanceFact,
    InstanceFacts,
    InstanceFactsError,
)
from exp.runtime.environments.artifact_edits import MAX_ARTIFACT_BYTES
from exp.runtime.environments.facts_derivers import DEFAULT_MAX_FACT_BYTES

# Identity of the code that decides these facts, in place of its source text. Bump it when the
# encoding or validation changes. exp/tests/code_protocols_test.py fails when this module
# changes until it is bumped or, for a pure refactor, its digest is updated.
WORKBOOK_FACTS_PROTOCOL = "xlsx-instance-facts-v1"

MAX_WORKBOOK_CELLS = 500_000
_MAX_EXPANDED_BYTES = 256 * 1024 * 1024
_MAX_PARTS = 10_000
_MAX_READER_WARNINGS = 20
# More absent cells than this between two stored cells of one row start a new row segment.
_SEGMENT_GAP = 3
_GENERAL = "General"

LEGEND = (
    "sheets[].rows: [row, first column (1 = A), value, value, ...] covering consecutive "
    "columns; a row may have several segments. null = no stored cell. A value is a JSON "
    "string, number or boolean as openpyxl reads it, or a tagged object: {} stored empty cell; "
    '{"empty": type} stored empty cell of another openpyxl data type; {"d": ISO datetime} '
    '(date only means midnight); {"date": ISO date}; {"t": ISO time}; {"td": [days, seconds, '
    'microseconds]} duration; {"e": error text}; {"f": formula, "c": cached value}; '
    '{"af": array formula, "ref": range, "c": cached value}. A formula without "c" has no '
    "cached value. sheets[].formats: [column, first row, last row, index into number_formats] "
    "for stored cells of that column in those rows; every other stored cell is General."
)

# Archive parts by declared content type. Content parts are decoded; metadata parts carry
# nothing an openpyxl program reads as instance content; every other part is unrepresented.
_MAIN = "application/vnd.openxmlformats-officedocument.spreadsheetml."
_CONTENT_TYPES = frozenset(
    (
        _MAIN + "sheet.main+xml",
        _MAIN + "template.main+xml",
        "application/vnd.ms-excel.sheet.macroEnabled.main+xml",
        _MAIN + "worksheet+xml",
        _MAIN + "styles+xml",
        _MAIN + "sharedStrings+xml",
        _MAIN + "comments+xml",
        _MAIN + "table+xml",
    )
)
_METADATA_TYPES = frozenset(
    (
        "application/vnd.openxmlformats-package.relationships+xml",
        "application/vnd.openxmlformats-package.core-properties+xml",
        "application/vnd.openxmlformats-officedocument.extended-properties+xml",
        "application/vnd.openxmlformats-officedocument.custom-properties+xml",
        "application/vnd.openxmlformats-officedocument.theme+xml",
        "application/vnd.openxmlformats-officedocument.customXmlProperties+xml",
        "application/vnd.openxmlformats-officedocument.vmlDrawing",
        _MAIN + "calcChain+xml",
        _MAIN + "sheetMetadata+xml",
        _MAIN + "printerSettings",
        "application/vnd.ms-excel.person+xml",
    )
)
# Package metadata and custom XML data stores, whatever content type they declare.
_METADATA_PART = re.compile(r"\[Content_Types\]\.xml|docProps/[^/]+|customXml/.+")
_UNREPRESENTED_KINDS = (
    ("chart", "chart"),
    ("drawing+xml", "drawing"),
    ("image/", "image"),
    ("pivot", "pivot table"),
    ("externalLink", "external link"),
    ("vbaProject", "macro"),
    ("activeX", "ActiveX control"),
    ("controlproperties", "form control"),
    ("threadedComment", "threaded comment"),
    ("slicer", "slicer"),
    ("timeline", "slicer"),
    ("queryTable", "data connection"),
    ("connections+xml", "data connection"),
    ("oleObject", "embedded object"),
    ("package", "embedded object"),
)
_CONTENT_TYPES_NAMESPACE = "{http://schemas.openxmlformats.org/package/2006/content-types}"

ScalarValue = str | int | float | bool | datetime | date | time | timedelta
CellValue = ScalarValue | ArrayFormula | None


class WorkbookFactsRefused(InstanceFactsError):
    """The workbook has content these facts cannot represent exactly, or exceeds a bound."""


@dataclass(frozen=True)
class DecodedCell:
    """One stored cell as openpyxl reads it, reconstructed from encoded facts.

    Attributes:
        value: The formula-mode value: a formula string, an ``ArrayFormula``, a typed
            scalar, or None for a stored empty cell.
        data_type: openpyxl's stored cell type (``n``, ``s``, ``b``, ``d``, ``e``, ``f``).
        number_format: The cell's number format.
        cached: The ``data_only=True`` value: the cached result of a formula, otherwise the
            same value as ``value``.
        cached_data_type: openpyxl's cell type in the ``data_only=True`` load.
    """

    value: CellValue
    data_type: str
    number_format: str
    cached: CellValue
    cached_data_type: str


@dataclass(frozen=True)
class DecodedSheet:
    """One sheet reconstructed from encoded facts.

    Attributes:
        name: Sheet title.
        chartsheet: Whether the sheet is a chartsheet, which has no cells.
        state: ``visible``, ``hidden`` or ``veryHidden``.
        dimensions: openpyxl's ``Worksheet.dimensions``, including stored empty cells.
        cells: Stored cells keyed by coordinate, such as ``B3``.
        features: Merged ranges, comments, hyperlinks, names, tables and range lists, in
            their encoded JSON form.
    """

    name: str
    chartsheet: bool
    state: str
    dimensions: str | None
    cells: Mapping[str, DecodedCell]
    features: JsonObject


@dataclass(frozen=True)
class DecodedWorkbook:
    """A workbook reconstructed from encoded facts.

    Attributes:
        sheets: Sheets in workbook order.
        names: Workbook-level defined names and their formulas.
    """

    sheets: tuple[DecodedSheet, ...]
    names: JsonObject

    def sheet(self, name: str) -> DecodedSheet:
        """The sheet titled ``name``; raises KeyError when the workbook has none."""
        for sheet in self.sheets:
            if sheet.name == name:
                return sheet
        raise KeyError(name)


def derive_workbook_facts(
    path: str, payload: bytes, *, max_bytes: int, allow_unrepresented: bool
) -> InstanceFacts:
    """Validated facts about one starting workbook, keyed ``<path>/workbook``.

    Args:
        path: The workbook's canonical path in the task's world, such as ``input.xlsx``.
        payload: Exact workbook bytes.
        max_bytes: Bound on the encoded value's canonical JSON size.
        allow_unrepresented: List unrepresented archive parts instead of refusing them.

    Returns:
        One conditioning fact at state version 0 whose value is the encoded workbook.

    Raises:
        WorkbookFactsRefused: The workbook has unrepresented content, exceeds a bound,
            cannot be read, or fails round-trip validation.
    """
    value = encode_workbook(payload, max_bytes=max_bytes, allow_unrepresented=allow_unrepresented)
    reader = f"openpyxl {version('openpyxl')}"
    fact = InstanceFact(
        key=f"{path}/workbook",
        value=value,
        visibility="conditioning",
        provenance=FactProvenance(
            adapter=f"{WORKBOOK_FACTS_PROTOCOL} ({reader})",
            source=path,
            source_sha256=sha256(payload, usedforsecurity=False).hexdigest(),
            validation=FactValidation(
                status="validated",
                method=(
                    f"every stored cell decoded from the encoding equals {reader} (formula "
                    "and data_only loads) by type and exact repr; sheet order, dimensions "
                    "and features round-trip exactly"
                ),
            ),
            state_version=0,
        ),
    )
    return InstanceFacts(facts=(fact,))


def encode_workbook(
    payload: bytes,
    *,
    max_bytes: int = DEFAULT_MAX_FACT_BYTES,
    allow_unrepresented: bool = False,
) -> JsonObject:
    """Encode one workbook exactly as openpyxl reads it, validated by a full round trip.

    Args:
        payload: Exact workbook bytes.
        max_bytes: Bound on the canonical JSON size of the returned value.
        allow_unrepresented: List unrepresented archive parts instead of refusing them.

    Returns:
        The encoded workbook (see ``LEGEND``).

    Raises:
        WorkbookFactsRefused: The workbook has unrepresented content, exceeds a bound,
            cannot be read, or fails round-trip validation.
    """
    unrepresented = _unrepresented_parts(payload)
    if unrepresented and not allow_unrepresented:
        listed = ", ".join(f"{row['part']} ({row['kind']})" for row in unrepresented[:5])
        more = f" and {len(unrepresented) - 5} more" if len(unrepresented) > 5 else ""
        raise WorkbookFactsRefused(
            f"workbook has content the facts cannot represent: {listed}{more}. Remove it from "
            "the input, or set allow_unrepresented to derive cell facts that list these parts"
        )
    formulas, cached, reader_warnings = _load(payload)
    stored = sum(len(sheet._cells) for sheet in formulas.worksheets)
    if stored > MAX_WORKBOOK_CELLS:
        raise WorkbookFactsRefused(
            f"workbook stores {stored:,} cells, over the {MAX_WORKBOOK_CELLS:,}-cell bound for "
            "instance facts; supply a smaller input workbook"
        )
    value = _encode(formulas, cached)
    if unrepresented:
        value["unrepresented"] = cast(JsonValue, unrepresented)
    if reader_warnings:
        value["reader_warnings"] = list(reader_warnings)
    size = len(canonical_json_bytes(value))
    if size > max_bytes:
        raise WorkbookFactsRefused(
            f"encoded workbook facts are {size:,} bytes, over the {max_bytes:,}-byte bound; "
            "raise max_bytes (at most 8 MiB) or supply a smaller input workbook"
        )
    _validate(json.loads(canonical_json_bytes(value)), formulas, cached)
    return value


def decode_workbook_facts(value: JsonValue) -> DecodedWorkbook:
    """Reconstruct typed openpyxl values from an encoded workbook.

    Args:
        value: A value produced by :func:`encode_workbook` (after any JSON round trip).

    Returns:
        Every sheet and stored cell, with formula-mode and cached values.

    Raises:
        ValueError: The value is not a workbook encoding of this protocol.
    """
    root = _object(value, "workbook facts")
    if root.get("codec") != WORKBOOK_FACTS_PROTOCOL:
        raise ValueError(f"workbook facts must use codec {WORKBOOK_FACTS_PROTOCOL}")
    formats = [str(row) for row in _array(root.get("number_formats"), "number_formats")]
    sheets: list[DecodedSheet] = []
    for raw in _array(root.get("sheets"), "sheets"):
        sheet = _object(raw, "sheet")
        name = str(sheet["name"])
        state = str(sheet.get("state", "visible"))
        if sheet.get("chartsheet") is True:
            sheets.append(DecodedSheet(name, True, state, None, {}, {}))
            continue
        cells: dict[str, DecodedCell] = {}
        for (row, column), (encoded, fmt) in _unpack(sheet).items():
            cells[f"{get_column_letter(column)}{row}"] = _decode_cell(encoded, formats[fmt])
        features = {key: item for key, item in sheet.items() if key in _FEATURE_KEYS}
        sheets.append(DecodedSheet(name, False, state, str(sheet["dimensions"]), cells, features))
    names = root.get("names", {})
    return DecodedWorkbook(tuple(sheets), _object(names, "names"))


def slice_workbook_facts(
    value: JsonValue,
    selection: Mapping[str, Sequence[str] | None],
    *,
    max_bytes: int = DEFAULT_MAX_FACT_BYTES,
) -> JsonObject:
    """A bounded view of an encoded workbook: the full inventory, selected cells only.

    Args:
        value: An encoded workbook.
        selection: Sheet names mapped to cell ranges such as ``A1:D20`` (whole rows or
            columns allowed, such as ``3:10`` or ``B:B``), or to None for the whole sheet.
            Unselected sheets keep their inventory entry and no cells.
        max_bytes: Bound on the view's canonical JSON size.

    Returns:
        A value in the same encoding whose ``coverage`` maps each selected sheet to its
        ranges (``*`` for the whole sheet); cells outside coverage are unknown, not empty.

    Raises:
        ValueError: A selected sheet is missing, a range is malformed, or the view exceeds
            ``max_bytes``.
    """
    root = _object(value, "workbook facts")
    if root.get("codec") != WORKBOOK_FACTS_PROTOCOL:
        raise ValueError(f"workbook facts must use codec {WORKBOOK_FACTS_PROTOCOL}")
    names = {str(_object(sheet, "sheet")["name"]) for sheet in _array(root["sheets"], "sheets")}
    missing = sorted(set(selection) - names)
    if missing:
        raise ValueError(f"view selects sheets the workbook does not have: {', '.join(missing)}")
    sheets: list[JsonValue] = []
    coverage: JsonObject = {}
    for raw in _array(root["sheets"], "sheets"):
        sheet = _object(raw, "sheet")
        name = str(sheet["name"])
        if name not in selection or sheet.get("chartsheet") is True:
            continue
        ranges = selection[name]
        bounds = None if ranges is None else [_bounds(text) for text in ranges]
        coverage[name] = ["*"] if ranges is None else list(ranges)
        cells = {
            position: cell
            for position, cell in _unpack(sheet).items()
            if bounds is None or any(_inside(position, box) for box in bounds)
        }
        kept: JsonObject = {
            key: sheet[key] for key in ("name", "state", "dimensions") if key in sheet
        }
        kept.update(_pack(cells))
        for key in _FEATURE_KEYS:
            if key not in sheet:
                continue
            feature = sheet[key]
            if key in ("comments", "hyperlinks") and bounds is not None:
                feature = {
                    coordinate: item
                    for coordinate, item in _object(feature, key).items()
                    if any(_inside(_position(coordinate), box) for box in bounds)
                }
                if not feature:
                    continue
            kept[key] = feature
        sheets.append(kept)
    view: JsonObject = {
        key: item for key, item in root.items() if key not in ("sheets", "coverage")
    }
    view["coverage"] = coverage
    view["sheets"] = sheets
    size = len(canonical_json_bytes(view))
    if size > max_bytes:
        raise ValueError(
            f"workbook view is {size:,} bytes, over the {max_bytes:,}-byte bound; select fewer "
            "cells"
        )
    return view


_FEATURE_KEYS = (
    "merged",
    "comments",
    "hyperlinks",
    "names",
    "tables",
    "validations",
    "conditional_formats",
    "auto_filter",
    "freeze_panes",
)


def _unrepresented_parts(payload: bytes) -> list[JsonObject]:
    """Classify every archive part, refusing unsafe archives and XML declarations."""
    if len(payload) > MAX_ARTIFACT_BYTES:
        raise WorkbookFactsRefused(
            f"workbook is over the {MAX_ARTIFACT_BYTES:,}-byte input bound for instance facts"
        )
    try:
        archive = ZipFile(io.BytesIO(payload))
    except BadZipFile as error:
        raise WorkbookFactsRefused(
            "input is not an XLSX archive (legacy .xls, encrypted or corrupt workbooks are "
            "unsupported); save it as an unencrypted .xlsx"
        ) from error
    with archive:
        members = [row for row in archive.infolist() if not row.filename.endswith("/")]
        if len(members) > _MAX_PARTS or sum(row.file_size for row in members) > (
            _MAX_EXPANDED_BYTES
        ):
            raise WorkbookFactsRefused("workbook archive exceeds the part or expanded-byte bound")
        declared = _content_types(archive)
        unrepresented: list[JsonObject] = []
        for member in members:
            name = member.filename
            if name.endswith((".xml", ".rels", ".vml")):
                content = archive.read(member)
                if b"<!DOCTYPE" in content or b"<!ENTITY" in content:
                    raise WorkbookFactsRefused(f"workbook part {name} declares XML entities")
            content_type = declared.get(name)
            if (
                _METADATA_PART.fullmatch(name)
                or content_type in _CONTENT_TYPES
                or content_type in _METADATA_TYPES
            ):
                continue
            kind = next(
                (
                    label
                    for marker, label in _UNREPRESENTED_KINDS
                    if content_type is not None and marker.lower() in content_type.lower()
                ),
                "unrecognized part",
            )
            unrepresented.append({"part": name, "kind": kind})
    return unrepresented


def _content_types(archive: ZipFile) -> dict[str, str]:
    """Each part's declared content type: its override, else its extension's default."""
    try:
        document = ElementTree.fromstring(archive.read("[Content_Types].xml"))
    except (KeyError, ElementTree.ParseError) as error:
        raise WorkbookFactsRefused(
            "workbook archive has no readable [Content_Types].xml; save it as an .xlsx"
        ) from error
    defaults = {
        str(node.get("Extension", "")).lower(): str(node.get("ContentType", ""))
        for node in document.iter(_CONTENT_TYPES_NAMESPACE + "Default")
    }
    overrides = {
        str(node.get("PartName", "")).lstrip("/"): str(node.get("ContentType", ""))
        for node in document.iter(_CONTENT_TYPES_NAMESPACE + "Override")
    }
    declared: dict[str, str] = {}
    for name in archive.namelist():
        content_type = overrides.get(name, defaults.get(name.rpartition(".")[2].lower()))
        if content_type is not None:
            declared[name] = content_type
    return declared


def _load(payload: bytes) -> tuple[Workbook, Workbook, tuple[str, ...]]:
    """Load the formula and cached-value views the way a default openpyxl program does."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            formulas = load_workbook(io.BytesIO(payload))
            cached = load_workbook(io.BytesIO(payload), data_only=True)
        except Exception as error:  # openpyxl raises many types for unreadable workbooks
            raise WorkbookFactsRefused(
                f"openpyxl cannot read the workbook ({type(error).__name__}: {error}); supply a "
                "workbook openpyxl opens"
            ) from error
    messages = tuple(dict.fromkeys(str(row.message) for row in caught))
    return formulas, cached, messages[:_MAX_READER_WARNINGS]


def _encode(formulas: Workbook, cached: Workbook) -> JsonObject:
    """Encode both loads of one workbook, without bounds or validation."""
    number_formats: dict[str, int] = {_GENERAL: 0}
    inventory: list[JsonValue] = []
    sheets: list[JsonValue] = []
    for name in formulas.sheetnames:
        sheet = formulas[name]
        header: JsonObject = {"name": name}
        if sheet.sheet_state != "visible":
            header["state"] = sheet.sheet_state
        if isinstance(sheet, Chartsheet):
            entry: JsonObject = {**header, "chartsheet": True}
            inventory.append(entry)
            sheets.append(entry)
            continue
        cached_sheet = cached[name]
        assert isinstance(cached_sheet, Worksheet)
        cells: dict[tuple[int, int], tuple[JsonValue, int]] = {}
        formula_count = uncached = 0
        for position, cell in sorted(sheet._cells.items()):
            if isinstance(cell, MergedCell) or _omittable(cell):
                continue
            encoded = _encode_cell(cell, cached_sheet._cells.get(position), f"{name}!")
            if isinstance(encoded, dict) and ("f" in encoded or "af" in encoded):
                formula_count += 1
                uncached += "c" not in encoded
            fmt = number_formats.setdefault(cell.number_format, len(number_formats))
            cells[position] = (encoded, fmt)
        inventory.append(
            {
                **header,
                "dimensions": sheet.dimensions,
                "cells": len(cells),
                "formulas": formula_count,
                "uncached_formulas": uncached,
            }
        )
        sheets.append(
            {**header, "dimensions": sheet.dimensions, **_pack(cells), **_features(sheet)}
        )
    value: JsonObject = {
        "codec": WORKBOOK_FACTS_PROTOCOL,
        "reader": f"openpyxl {version('openpyxl')}",
        "legend": LEGEND,
        "coverage": "complete",
        "inventory": inventory,
        "number_formats": list(number_formats),
        "sheets": sheets,
    }
    names = {name: item.attr_text for name, item in formulas.defined_names.items()}
    if names:
        value["names"] = cast(JsonValue, names)
    return value


def _omittable(cell: Cell) -> bool:
    """Whether a stored cell is indistinguishable from an absent one to an openpyxl program."""
    return (
        cell.value is None
        and cell.data_type == "n"
        and cell.number_format == _GENERAL
        and cell.comment is None
        and cell.hyperlink is None
    )


def _encode_cell(cell: Cell, cached: Cell | MergedCell | None, sheet: str) -> JsonValue:
    """Encode one stored cell's formula-mode value and, for a formula, its cached value."""
    where = sheet + cell.coordinate
    value = cell.value
    if isinstance(value, DataTableFormula):
        raise WorkbookFactsRefused(f"{where} holds a data-table formula, which facts cannot carry")
    if isinstance(value, ArrayFormula) or cell.data_type == "f":
        encoded: JsonObject = (
            {"af": value.text, "ref": value.ref}
            if isinstance(value, ArrayFormula)
            else {"f": _scalar(value, where)}
        )
        if cached is not None and cached.value is not None:
            encoded["c"] = _typed(cached.value, cached.data_type, where)
        elif cached is not None and cached.data_type != "n":
            encoded["c"] = {"empty": cached.data_type}
        return encoded
    if value is None:
        return {} if cell.data_type == "n" else {"empty": cell.data_type}
    return _typed(value, cell.data_type, where)


def _typed(value: object, data_type: str, where: str) -> JsonValue:
    """Encode one non-formula value, tagging every type JSON cannot state by itself."""
    if data_type == "e":
        return {"e": _scalar(value, where)}
    return _scalar(value, where)


def _scalar(value: object, where: str) -> JsonValue:
    """Encode one typed scalar exactly, refusing types the encoding does not define."""
    if isinstance(value, bool | str):
        return value
    if isinstance(value, int | float):
        if isinstance(value, float) and not math.isfinite(value):
            raise WorkbookFactsRefused(f"{where} holds a non-finite number")
        return value
    if isinstance(value, datetime):
        if value.tzinfo is None and value.time() == time():
            return {"d": value.date().isoformat()}
        return {"d": value.isoformat()}
    if isinstance(value, date):
        return {"date": value.isoformat()}
    if isinstance(value, time):
        return {"t": value.isoformat()}
    if isinstance(value, timedelta):
        return {"td": [value.days, value.seconds, value.microseconds]}
    raise WorkbookFactsRefused(
        f"{where} holds a {type(value).__name__} value, which instance facts cannot represent"
    )


def _features(sheet: Worksheet) -> JsonObject:
    """Encode a worksheet's non-cell content, omitting what it does not have."""
    features: JsonObject = {}
    merged = [str(item) for item in cast(Iterable[object], sheet.merged_cells.ranges)]
    comments = {
        cell.coordinate: [cell.comment.author, cell.comment.text]
        for cell in sheet._cells.values()
        if cell.comment is not None
    }
    hyperlinks = {
        cell.coordinate: {
            field: getattr(cell.hyperlink, field)
            for field in ("target", "location", "display", "tooltip")
            if getattr(cell.hyperlink, field) is not None
        }
        for cell in sheet._cells.values()
        if cell.hyperlink is not None
    }
    names = {name: item.attr_text for name, item in sheet.defined_names.items()}
    tables = {name: ref for name, ref in sheet.tables.items()}
    validations = [
        str(item.sqref)
        for item in cast(Iterable[DataValidation], sheet.data_validations.dataValidation)
    ]
    conditional = [str(item.sqref) for item in sheet.conditional_formatting]
    for key, item in (
        ("merged", merged),
        ("comments", comments),
        ("hyperlinks", hyperlinks),
        ("names", names),
        ("tables", tables),
        ("validations", validations),
        ("conditional_formats", conditional),
        ("auto_filter", sheet.auto_filter.ref),
        ("freeze_panes", sheet.freeze_panes),
    ):
        if item:
            features[key] = cast(JsonValue, item)
    return features


def _pack(cells: Mapping[tuple[int, int], tuple[JsonValue, int]]) -> JsonObject:
    """Encode positioned cells as row segments and per-column number-format runs."""
    rows: list[JsonValue] = []
    segment: list[JsonValue] = []
    previous: tuple[int, int] | None = None
    for row, column in sorted(cells):
        if previous is None or previous[0] != row or column - previous[1] - 1 > _SEGMENT_GAP:
            if segment:
                rows.append(segment)
            segment = [row, column]
        else:
            segment.extend([None] * (column - previous[1] - 1))
        segment.append(cells[(row, column)][0])
        previous = (row, column)
    if segment:
        rows.append(segment)
    formats: list[JsonValue] = []
    for column in sorted({column for _, column in cells}):
        run: list[int] | None = None
        for row in sorted(row for row, other in cells if other == column):
            fmt = cells[(row, column)][1]
            if run is not None and run[3] == fmt:
                run[2] = row
                continue
            if run is not None and run[3] != 0:
                formats.append(list(run))
            run = [column, row, row, fmt]
        if run is not None and run[3] != 0:
            formats.append(list(run))
    packed: JsonObject = {"rows": rows}
    if formats:
        packed["formats"] = formats
    return packed


def _unpack(sheet: JsonObject) -> dict[tuple[int, int], tuple[JsonValue, int]]:
    """Positioned cells and number-format indexes from row segments and format runs."""
    cells: dict[tuple[int, int], tuple[JsonValue, int]] = {}
    for raw in _array(sheet.get("rows", []), "rows"):
        segment = _array(raw, "row segment")
        row, first = int(cast(int, segment[0])), int(cast(int, segment[1]))
        for offset, encoded in enumerate(segment[2:]):
            if encoded is not None:
                cells[(row, first + offset)] = (encoded, 0)
    rows_by_column: dict[int, list[int]] = {}
    for row, column in sorted(cells):
        rows_by_column.setdefault(column, []).append(row)
    for raw in _array(sheet.get("formats", []), "formats"):
        column, first, last, fmt = (int(cast(int, item)) for item in _array(raw, "format run"))
        rows = rows_by_column.get(column, [])
        for row in rows[bisect_left(rows, first) : bisect_right(rows, last)]:
            cells[(row, column)] = (cells[(row, column)][0], fmt)
    return cells


def _decode_cell(encoded: JsonValue, number_format: str) -> DecodedCell:
    """Typed formula-mode and cached values of one encoded cell."""
    if isinstance(encoded, dict) and ("f" in encoded or "af" in encoded):
        value: CellValue = (
            ArrayFormula(ref=str(encoded["ref"]), text=str(encoded["af"]))
            if "af" in encoded
            else str(encoded["f"])
        )
        cached = encoded.get("c")
        cached_value: CellValue
        if cached is None:
            cached_value, cached_type = None, "n"
        elif isinstance(cached, dict) and "empty" in cached:
            cached_value, cached_type = None, str(cached["empty"])
        else:
            cached_value, cached_type = _decode_value(cached)
        return DecodedCell(value, "f", number_format, cached_value, cached_type)
    if isinstance(encoded, dict) and not encoded:
        return DecodedCell(None, "n", number_format, None, "n")
    if isinstance(encoded, dict) and "empty" in encoded:
        data_type = str(encoded["empty"])
        return DecodedCell(None, data_type, number_format, None, data_type)
    value, data_type = _decode_value(encoded)
    return DecodedCell(value, data_type, number_format, value, data_type)


def _decode_value(encoded: JsonValue) -> tuple[ScalarValue, str]:
    """One typed scalar and its openpyxl data type."""
    if isinstance(encoded, bool):
        return encoded, "b"
    if isinstance(encoded, int | float):
        return encoded, "n"
    if isinstance(encoded, str):
        return encoded, "s"
    tagged = _object(encoded, "cell value")
    if len(tagged) != 1:
        raise ValueError(f"cell value {tagged!r} is not a single-tag object")
    tag, item = next(iter(tagged.items()))
    if tag == "e":
        return str(item), "e"
    if tag == "d":
        return datetime.fromisoformat(str(item)), "d"
    if tag == "date":
        return date.fromisoformat(str(item)), "d"
    if tag == "t":
        return time.fromisoformat(str(item)), "d"
    if tag == "td":
        days, seconds, microseconds = (int(cast(int, part)) for part in _array(item, "td"))
        return timedelta(days=days, seconds=seconds, microseconds=microseconds), "d"
    raise ValueError(f"unknown cell value tag {tag!r}")


def _validate(value: JsonValue, formulas: Workbook, cached: Workbook) -> None:
    """Refuse unless every stored cell and feature round-trips exactly through the encoding."""
    decoded = decode_workbook_facts(value)
    if [sheet.name for sheet in decoded.sheets] != formulas.sheetnames:
        raise WorkbookFactsRefused("encoded sheet order differs from the workbook")
    expected_names = {name: item.attr_text for name, item in formulas.defined_names.items()}
    if decoded.names != expected_names:
        raise WorkbookFactsRefused("encoded defined names differ from the workbook")
    mismatches: list[str] = []
    for sheet in decoded.sheets:
        source = formulas[sheet.name]
        if sheet.state != source.sheet_state or sheet.chartsheet != isinstance(source, Chartsheet):
            mismatches.append(f"{sheet.name}: sheet kind or state")
            continue
        if isinstance(source, Chartsheet):
            continue
        cached_sheet = cached[sheet.name]
        assert isinstance(cached_sheet, Worksheet)
        if sheet.dimensions != source.dimensions:
            mismatches.append(f"{sheet.name}: dimensions")
        if json.loads(canonical_json_bytes(_features(source))) != sheet.features:
            mismatches.append(f"{sheet.name}: merged ranges, comments, links or names")
        seen: set[str] = set()
        for position, cell in source._cells.items():
            coordinate = f"{get_column_letter(position[1])}{position[0]}"
            got = sheet.cells.get(coordinate)
            if isinstance(cell, MergedCell) or got is None:
                if got is not None or not (isinstance(cell, MergedCell) or _omittable(cell)):
                    mismatches.append(f"{sheet.name}!{coordinate}")
                continue
            seen.add(coordinate)
            other = cached_sheet._cells.get(position)
            if (
                _signature(got.value) != _signature(cell.value)
                or got.data_type != cell.data_type
                or got.number_format != cell.number_format
                or other is None
                or _signature(got.cached) != _signature(other.value)
                or got.cached_data_type != other.data_type
            ):
                mismatches.append(f"{sheet.name}!{coordinate}")
        mismatches.extend(f"{sheet.name}!{name}" for name in sorted(set(sheet.cells) - seen))
    if mismatches:
        listed = ", ".join(mismatches[:5])
        more = f" and {len(mismatches) - 5} more" if len(mismatches) > 5 else ""
        raise WorkbookFactsRefused(
            f"encoded workbook facts differ from openpyxl at {listed}{more}; the workbook has "
            "content this encoding does not reproduce exactly"
        )


def _signature(value: object) -> tuple[str, ...]:
    """Exact comparison key: Python type plus ``repr``, or an array formula's range and text."""
    if isinstance(value, ArrayFormula):
        return ("ArrayFormula", str(value.ref), str(value.text))
    return (type(value).__name__, repr(value))


def _bounds(text: str) -> tuple[int, int, int, int]:
    """Inclusive (min column, min row, max column, max row) of one range; open sides unbounded."""
    try:
        low_column, low_row, high_column, high_row = range_boundaries(text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"view range {text!r} is not a cell range such as A1:D20") from error
    return (
        low_column or 1,
        low_row or 1,
        high_column or 2**31,
        high_row or 2**31,
    )


def _inside(position: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    """Whether a (row, column) position lies in an inclusive range box."""
    row, column = position
    return box[0] <= column <= box[2] and box[1] <= row <= box[3]


def _position(coordinate: str) -> tuple[int, int]:
    """(row, column) of one coordinate such as ``B3``."""
    letters, row = coordinate_from_string(coordinate)
    return row, column_index_from_string(letters)


def _object(value: JsonValue, label: str) -> JsonObject:
    """Require a JSON object."""
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _array(value: JsonValue, label: str) -> list[JsonValue]:
    """Require a JSON array."""
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value
