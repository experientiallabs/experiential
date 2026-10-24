"""Workbook instance facts: typed lossless encoding, round-trip validation and refusals.

Every fixture is a real XLSX written by openpyxl in the test. Where openpyxl cannot write the
case (exact serials, inline strings, cached values, error cells), the saved sheet XML is
patched the way Excel stores it. The three cases the earlier study decoder got wrong (time-only
values, sub-second datetimes, string runs and merged shadows) are pinned against openpyxl.
"""

import io
import json
import re
from collections.abc import Callable
from datetime import date, datetime, time, timedelta
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from openpyxl import Workbook, load_workbook
from openpyxl.comments import Comment
from openpyxl.formatting.rule import CellIsRule
from openpyxl.workbook.defined_name import DefinedName
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.worksheet.formula import ArrayFormula
from openpyxl.worksheet.table import Table

from exp.common.core.artifacts import canonical_json_bytes
from exp.common.tasks.facts import InstanceFacts, InstanceFactsError
from exp.runtime.environments import workbook_facts
from exp.runtime.environments.facts_derivers import WorkbookFactsDeriver
from exp.runtime.environments.workbook_facts import (
    WORKBOOK_FACTS_PROTOCOL,
    WorkbookFactsRefused,
    decode_workbook_facts,
    encode_workbook,
    slice_workbook_facts,
)


def xlsx(build: Callable[[Workbook], None]) -> bytes:
    """Save a workbook built by ``build`` and return its bytes."""
    workbook = Workbook()
    build(workbook)
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def rows_xlsx(*rows: list[object]) -> bytes:
    """A one-sheet workbook holding ``rows``."""

    def build(workbook: Workbook) -> None:
        """Append each row to the active sheet."""
        sheet = workbook.active
        assert sheet is not None
        for row in rows:
            sheet.append(row)

    return xlsx(build)


def patch(payload: bytes, part: str, pattern: str, replacement: str) -> bytes:
    """Rewrite one archive part by a regular-expression substitution that must match once."""
    source = ZipFile(io.BytesIO(payload))
    output = io.BytesIO()
    with source, ZipFile(output, "w", compression=ZIP_DEFLATED) as target:
        for name in source.namelist():
            content = source.read(name)
            if name == part:
                text, count = re.subn(pattern, replacement, content.decode("utf-8"))
                assert count == 1, f"{pattern!r} matched {count} times in {part}"
                content = text.encode("utf-8")
            target.writestr(name, content)
    return output.getvalue()


def add_part(payload: bytes, name: str, content: bytes) -> bytes:
    """Append one archive part."""
    output = io.BytesIO(payload)
    with ZipFile(output, "a") as archive:
        archive.writestr(name, content)
    return output.getvalue()


def roundtrip(payload: bytes, **options: int | bool) -> workbook_facts.DecodedWorkbook:
    """Derive facts, pass them through InstanceFacts JSON, and decode the value."""
    facts = WorkbookFactsDeriver.model_validate(options).derive("input.xlsx", payload)
    restored = InstanceFacts.model_validate_json(facts.model_dump_json())
    return decode_workbook_facts(restored.facts[0].value)


def value_of(payload: bytes) -> dict:
    """The encoded workbook value."""
    return encode_workbook(payload)


def view_of(value: dict, selection: dict, **options: int) -> dict:
    """A sliced view of an encoded workbook."""
    return slice_workbook_facts(value, selection, **options)


def typed(value: object) -> tuple[str, str]:
    """Exact comparison key: type name and repr."""
    return type(value).__name__, repr(value)


def test_every_stored_cell_round_trips_typed_with_formats_and_sheet_structure() -> None:
    """Values, types, formats, formulas, merges, order, state and features survive exactly."""

    def build(workbook: Workbook) -> None:
        """A two-sheet workbook with typed cells and every encoded feature."""
        sheet = workbook.active
        assert sheet is not None
        sheet.title = "Data"
        sheet.append(["name", "amount", "when", "ok"])
        sheet.append(["ΓΙΑΝΝΙΟΥ", 800436319, datetime(2023, 2, 12), True])
        sheet.append(["Bob", 0.1, datetime(2023, 3, 24, 13, 45, 7, 250000), False])
        sheet.append(["=not a formula", 1e16, date(2024, 1, 2), None])
        sheet["A4"].data_type = "s"  # text that looks like a formula
        sheet["B2"].number_format = "#,##0"
        sheet["B3"].number_format = "0.00%"
        sheet["E1"] = "=SUM(B2:B3)"
        sheet["F1"] = ArrayFormula("F1:F2", "=B2:B3*2")
        sheet["H2"].number_format = "0.000"  # stored empty cell with a format
        sheet["Z9"] = "far"
        sheet.merge_cells("A6:C6")
        sheet["A6"] = "merged title"
        sheet["A2"].comment = Comment("checked", "auditor")
        sheet["A3"].hyperlink = "https://example.com/bob"
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = "A1:D4"
        sheet.add_data_validation(DataValidation(type="list", formula1='"a,b"', sqref="D2:D4"))
        sheet.conditional_formatting.add("B2:B4", CellIsRule(operator="greaterThan", formula=["1"]))
        sheet.add_table(Table(displayName="People", ref="A1:D4"))
        sheet.defined_names["local_total"] = DefinedName("local_total", attr_text="Data!$E$1")
        other = workbook.create_sheet("Hidden")
        other.sheet_state = "hidden"
        other["A1"] = 3
        workbook.defined_names["rate"] = DefinedName("rate", attr_text="Hidden!$A$1")

    payload = xlsx(build)
    decoded = roundtrip(payload)
    loaded = load_workbook(io.BytesIO(payload))
    assert [sheet.name for sheet in decoded.sheets] == ["Data", "Hidden"]
    assert decoded.sheet("Hidden").state == "hidden"
    assert decoded.names == {"rate": "Hidden!$A$1"}
    data = decoded.sheet("Data")
    assert data.dimensions == loaded["Data"].dimensions == "A1:Z9"
    for coordinate in ("A2", "B2", "C2", "D2", "A4", "B3", "B4", "C3", "C4", "Z9", "A6", "E1"):
        cell = loaded["Data"][coordinate]
        assert typed(data.cells[coordinate].value) == typed(cell.value), coordinate
        assert data.cells[coordinate].data_type == cell.data_type
        assert data.cells[coordinate].number_format == cell.number_format
    assert data.cells["A4"].value == "=not a formula" and data.cells["A4"].data_type == "s"
    assert data.cells["E1"].value == "=SUM(B2:B3)" and data.cells["E1"].cached is None
    array = data.cells["F1"].value
    assert isinstance(array, ArrayFormula) and (array.ref, array.text) == ("F1:F2", "=B2:B3*2")
    assert data.cells["H2"].value is None and data.cells["H2"].number_format == "0.000"
    assert "B6" not in data.cells  # merged shadow cell
    assert data.features["merged"] == ["A6:C6"]
    assert data.features["comments"] == {"A2": ["auditor", "checked"]}
    assert data.features["hyperlinks"] == {"A3": {"target": "https://example.com/bob"}}
    assert data.features["tables"] == {"People": "A1:D4"}
    assert data.features["names"] == {"local_total": "Data!$E$1"}
    assert data.features["validations"] == ["D2:D4"]
    assert data.features["conditional_formats"] == ["B2:B4"]
    assert data.features["auto_filter"] == "A1:D4"
    assert data.features["freeze_panes"] == "A2"
    inventory = value_of(payload)["inventory"]
    assert inventory[0] == {
        "name": "Data",
        "dimensions": "A1:Z9",
        "cells": 20,
        "formulas": 2,
        "uncached_formulas": 2,
    }


def test_time_only_values_are_times_not_1899_dates() -> None:
    """Known decoder issue 1: a serial below one day with a time format is a time of day."""

    def build(workbook: Workbook) -> None:
        """Times written by openpyxl and one raw serial of 0.8125 with an h:mm format."""
        sheet = workbook.active
        assert sheet is not None
        sheet["A1"] = time(19, 30)
        sheet["A1"].number_format = "h:mm"
        sheet["A2"] = time(9, 54, 12)
        sheet["A2"].number_format = "h:mm:ss"
        sheet["A3"] = 1
        sheet["A3"].number_format = "h:mm"

    payload = patch(
        xlsx(build), "xl/worksheets/sheet1.xml", r"(<c r=\"A3\"[^>]*>)<v>1</v>", r"\1<v>0.8125</v>"
    )
    cells = roundtrip(payload).sheets[0].cells
    assert cells["A1"].value == time(19, 30) and cells["A1"].data_type == "d"
    assert cells["A2"].value == time(9, 54, 12)
    assert cells["A3"].value == time(19, 30)
    rows = value_of(payload)["sheets"][0]["rows"]
    assert rows[0] == [1, 1, {"t": "19:30:00"}]
    assert "1899-12-30" not in json.dumps(value_of(payload))


def test_datetimes_keep_openpyxl_millisecond_rounding_exactly() -> None:
    """Known decoder issue 2: a fractional serial reads as openpyxl's datetime, not a float day."""

    def build(workbook: Workbook) -> None:
        """One datetime cell whose raw serial is replaced below."""
        sheet = workbook.active
        assert sheet is not None
        sheet["A1"] = datetime(2020, 12, 11)
        sheet["A1"].number_format = "m/d/yy h:mm"

    payload = patch(
        xlsx(build),
        "xl/worksheets/sheet1.xml",
        r"(<c r=\"A1\"[^>]*>)<v>[^<]+</v>",
        r"\1<v>44176.4251680556</v>",
    )
    expected = load_workbook(io.BytesIO(payload)).active["A1"].value
    assert expected == datetime(2020, 12, 11, 10, 12, 14, 520000)
    naive = datetime(1899, 12, 30) + timedelta(days=44176.4251680556)
    assert naive != expected  # what a float-day decoder reports
    cell = roundtrip(payload).sheets[0].cells["A1"]
    assert typed(cell.value) == typed(expected)
    assert value_of(payload)["sheets"][0]["rows"][0] == [1, 1, {"d": "2020-12-11T10:12:14.520000"}]


def test_inline_string_runs_and_merged_shadow_values_match_openpyxl() -> None:
    """Known decoder issue 3: inline runs exclude phonetic text; merged shadows are not cells."""

    def build(workbook: Workbook) -> None:
        """An inline-string cell (patched below) and a merged row with a shadow value."""
        sheet = workbook.active
        assert sheet is not None
        sheet["A1"] = "placeholder"
        sheet["A3"] = "C"
        sheet["B3"] = "C"
        sheet["C3"] = 1

    payload = patch(
        xlsx(build),
        "xl/worksheets/sheet1.xml",
        r"<c r=\"A1\" t=\"inlineStr\"><is><t>placeholder</t></is></c>",
        '<c r="A1" t="inlineStr"><is><r><t>Hel</t></r><r><t xml:space="preserve">lo </t></r>'
        '<rPh sb="0" eb="1"><t>ph</t></rPh></is></c>',
    )
    payload = patch(
        payload,
        "xl/worksheets/sheet1.xml",
        r"</sheetData>",
        '</sheetData><mergeCells count="1"><mergeCell ref="A3:C3"/></mergeCells>',
    )
    loaded = load_workbook(io.BytesIO(payload)).active
    assert loaded["A1"].value == "Hello " and loaded["B3"].value is None
    cells = roundtrip(payload).sheets[0].cells
    assert cells["A1"].value == "Hello " and cells["A1"].data_type == "s"
    assert cells["A3"].value == "C"
    assert "B3" not in cells and "C3" not in cells


def test_durations_errors_and_cached_formula_values_are_tagged() -> None:
    """Durations, error cells, and cached results of formulas keep their exact types."""

    def build(workbook: Workbook) -> None:
        """A duration, a formula with a cached result and a literal error."""
        sheet = workbook.active
        assert sheet is not None
        sheet["A1"] = timedelta(hours=1, minutes=18)
        sheet["A1"].number_format = "[h]:mm"
        sheet["B1"] = "=1/0"
        sheet["C1"] = "=2+3"
        sheet["D1"] = "x"

    payload = xlsx(build)
    for pattern, replacement in (
        (r"<c r=\"B1\"><f>1/0</f><v></v></c>", '<c r="B1" t="e"><f>1/0</f><v>#DIV/0!</v></c>'),
        (r"<c r=\"C1\"><f>2\+3</f><v></v></c>", '<c r="C1"><f>2+3</f><v>5</v></c>'),
        (r"<c r=\"D1\" t=\"inlineStr\"><is><t>x</t></is></c>", '<c r="D1" t="e"><v>#N/A</v></c>'),
    ):
        payload = patch(payload, "xl/worksheets/sheet1.xml", pattern, replacement)
    cells = roundtrip(payload).sheets[0].cells
    assert cells["A1"].value == timedelta(seconds=4680) and cells["A1"].data_type == "d"
    assert cells["B1"].value == "=1/0" and cells["B1"].cached == "#DIV/0!"
    assert cells["B1"].cached_data_type == "e"
    assert typed(cells["C1"].cached) == ("int", "5")
    assert cells["D1"].value == "#N/A" and cells["D1"].data_type == "e"
    first = value_of(payload)["sheets"][0]["rows"][0]
    assert first[2:] == [
        {"td": [0, 4680, 0]},
        {"f": "=1/0", "c": {"e": "#DIV/0!"}},
        {"f": "=2+3", "c": 5},
        {"e": "#N/A"},
    ]


def test_sparse_rows_split_into_segments_and_formats_into_column_runs() -> None:
    """Wide gaps start a new segment; equal formats down a column become one run."""

    def build(workbook: Workbook) -> None:
        """Dates down column A and two far-apart cells in row 1."""
        sheet = workbook.active
        assert sheet is not None
        sheet["A1"] = "head"
        sheet["H1"] = "far"
        sheet["B1"] = 7
        for row in range(2, 6):
            sheet.cell(row=row, column=1, value=datetime(2023, 1, row))
        sheet["C1"] = 1.5
        sheet["C1"].number_format = "0.0"

    value = value_of(xlsx(build))
    sheet = value["sheets"][0]
    assert sheet["rows"][:2] == [[1, 1, "head", 7, 1.5], [1, 8, "far"]]
    formats = value["number_formats"]
    runs = {(run[0], run[1], run[2]): formats[run[3]] for run in sheet["formats"]}
    assert runs == {(1, 2, 5): "yyyy-mm-dd h:mm:ss", (3, 1, 1): "0.0"}
    decoded = decode_workbook_facts(json.loads(canonical_json_bytes(value)))
    assert decoded.sheets[0].cells["A5"].value == datetime(2023, 1, 5)
    assert decoded.sheets[0].cells["A1"].number_format == "General"


def test_derived_fact_is_validated_conditioning_state_zero_with_file_provenance() -> None:
    """The deriver names the world path and digest, never a host path, and states its method."""
    payload = rows_xlsx([1, "a"])
    facts = WorkbookFactsDeriver().derive("inputs/book.xlsx", payload)
    (fact,) = facts.facts
    assert fact.key == "inputs/book.xlsx/workbook"
    assert fact.visibility == "conditioning"
    assert fact.provenance.state_version == 0
    assert fact.provenance.source == "inputs/book.xlsx"
    assert fact.provenance.adapter.startswith(WORKBOOK_FACTS_PROTOCOL + " (openpyxl ")
    assert fact.provenance.validation.status == "validated"
    assert "type and exact repr" in str(fact.provenance.validation.method)
    assert facts.conditioning(0) is not None
    assert isinstance(fact.value, dict) and fact.value["coverage"] == "complete"


def test_unrepresented_parts_are_refused_or_listed_never_dropped() -> None:
    """A drawing is refused by default; allowed, it is named in the facts."""
    payload = rows_xlsx([1])
    payload = patch(
        payload,
        "[Content_Types].xml",
        r"</Types>",
        '<Override PartName="/xl/drawings/drawing1.xml" ContentType="application/vnd.'
        'openxmlformats-officedocument.drawing+xml"/></Types>',
    )
    payload = add_part(payload, "xl/drawings/drawing1.xml", b"<wsDr/>")
    with pytest.raises(WorkbookFactsRefused, match=r"xl/drawings/drawing1.xml \(drawing\)"):
        WorkbookFactsDeriver().derive("input.xlsx", payload)
    allowed = encode_workbook(payload, allow_unrepresented=True)
    assert allowed["unrepresented"] == [{"part": "xl/drawings/drawing1.xml", "kind": "drawing"}]
    unknown = add_part(payload, "xl/mystery.bin", b"?")
    with pytest.raises(WorkbookFactsRefused, match=r"xl/mystery.bin \(unrecognized part\)"):
        encode_workbook(unknown)


def test_bounds_unreadable_inputs_and_entities_are_refused_with_a_next_step() -> None:
    """Oversized encodings, non-XLSX bytes and XML entity declarations refuse."""

    def build(workbook: Workbook) -> None:
        """Enough text to exceed a 1 KiB bound."""
        sheet = workbook.active
        assert sheet is not None
        for row in range(1, 60):
            sheet.cell(row=row, column=1, value=f"value number {row}")

    with pytest.raises(WorkbookFactsRefused, match="over the 1,024-byte bound; raise max_bytes"):
        WorkbookFactsDeriver(max_bytes=1024).derive("input.xlsx", xlsx(build))
    with pytest.raises(WorkbookFactsRefused, match="not an XLSX archive"):
        encode_workbook(b"PK not really")
    entity = patch(xlsx(build), "xl/workbook.xml", r"^(<\?xml[^>]*\?>)?", r"\1<!DOCTYPE x>")
    with pytest.raises(WorkbookFactsRefused, match="declares XML entities"):
        encode_workbook(entity)
    assert issubclass(WorkbookFactsRefused, InstanceFactsError)


def test_data_table_formulas_and_unknown_value_types_are_refused() -> None:
    """Formula kinds and values outside the encoding refuse with their coordinate."""
    payload = rows_xlsx([1, 2])
    table = patch(
        payload,
        "xl/worksheets/sheet1.xml",
        r"<c r=\"B1\" t=\"n\"><v>2</v></c>",
        '<c r="B1"><f t="dataTable" ref="B1:B2" dt2D="0" dtr="0" r1="A1"/><v>2</v></c>',
    )
    with pytest.raises(WorkbookFactsRefused, match="!B1 holds a data-table formula"):
        encode_workbook(table)
    with pytest.raises(WorkbookFactsRefused, match="holds a bytes value"):
        workbook_facts._scalar(b"raw", "Sheet!A1")


def test_validation_refuses_an_encoding_that_loses_precision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cell-by-cell check catches a lossy encoder instead of trusting it."""

    def build(workbook: Workbook) -> None:
        """One sub-second datetime."""
        sheet = workbook.active
        assert sheet is not None
        sheet["A1"] = datetime(2023, 3, 24, 13, 45, 7, 250000)

    original = workbook_facts._scalar

    def lossy(value: object, where: str) -> object:
        """Drop microseconds, as a careless decoder would."""
        if isinstance(value, datetime):
            value = value.replace(microsecond=0)
        return original(value, where)

    monkeypatch.setattr(workbook_facts, "_scalar", lossy)
    with pytest.raises(WorkbookFactsRefused, match=r"differ from openpyxl at Sheet!A1"):
        encode_workbook(xlsx(build))


def test_slice_keeps_inventory_and_only_selected_cells_with_coverage() -> None:
    """A view for a read set carries every sheet's inventory and only the selected cells."""

    def build(workbook: Workbook) -> None:
        """A 10 x 4 table on one sheet and a second sheet."""
        sheet = workbook.active
        assert sheet is not None
        sheet.title = "Data"
        for row in range(1, 11):
            sheet.append([row, f"r{row}", row * 1.5, row % 2 == 0])
        sheet["B2"].comment = Comment("note", "me")
        sheet["B9"].comment = Comment("other", "me")
        workbook.create_sheet("Other")["A1"] = "x"

    value = value_of(xlsx(build))
    view = view_of(value, {"Data": ["A1:B3"]})
    assert view["coverage"] == {"Data": ["A1:B3"]}
    assert view["inventory"] == value["inventory"]
    assert [sheet["name"] for sheet in view["sheets"]] == ["Data"]
    assert view["sheets"][0]["rows"] == [[1, 1, 1, "r1"], [2, 1, 2, "r2"], [3, 1, 3, "r3"]]
    assert view["sheets"][0]["comments"] == {"B2": ["me", "note"]}
    sliced = decode_workbook_facts(view).sheet("Data")
    assert sorted(sliced.cells) == ["A1", "A2", "A3", "B1", "B2", "B3"]
    assert sliced.dimensions == "A1:D10"
    whole = view_of(value, {"Other": None})
    assert whole["coverage"] == {"Other": ["*"]}
    assert whole["sheets"][0]["rows"] == [[1, 1, "x"]]
    rows = view_of(value, {"Data": ["9:10"]})["sheets"][0]["rows"]
    assert [row[0] for row in rows] == [9, 10]
    with pytest.raises(ValueError, match="does not have: Missing"):
        view_of(value, {"Missing": None})
    with pytest.raises(ValueError, match="over the 1,024-byte bound"):
        view_of(value, {"Data": None}, max_bytes=1024)
    with pytest.raises(ValueError, match="not a cell range"):
        view_of(value, {"Data": ["not a range"]})
