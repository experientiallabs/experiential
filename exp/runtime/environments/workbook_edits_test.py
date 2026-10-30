"""Native workbook edits preserve types, reject unsafe scope and replay deterministically."""

import io
from datetime import datetime
from xml.etree import ElementTree
from zipfile import ZipFile

import pytest
from openpyxl import Workbook, load_workbook
from openpyxl.workbook.defined_name import DefinedName

from exp.common.core.artifacts import JsonObject
from exp.common.models import ToolCall
from exp.runtime.environments.workbook_edits import WorkbookEditAdapter


def test_workbook_discovery_pages_exact_names_without_cell_contents() -> None:
    """Case-sensitive sheet names are discoverable without reading all worksheet values."""
    workbook = Workbook()
    active = workbook.active
    assert active is not None
    workbook.remove(active)
    for name in ("sheet1", "Sheet1", "سجل"):
        sheet = workbook.create_sheet(name)
        sheet.append(["private cell", 3])
    # Excel disallows names differing only by case; retain the actual written names.
    names = workbook.sheetnames
    stream = io.BytesIO()
    workbook.save(stream)
    files = {"input.xlsx": stream.getvalue()}
    adapter = WorkbookEditAdapter()
    pages = []
    for offset in range(3):
        call = ToolCall(
            call_id=f"describe-{offset}",
            name="describe_workbook",
            arguments={
                "source": "input.xlsx",
                "offset": offset,
                "limit": 1,
            },
        )
        assert adapter.access(call).writes == frozenset()
        result = adapter.apply(files, (call,), ("output.xlsx",))
        assert not result.files
        page = result.projection["input.xlsx"]
        assert isinstance(page, dict)
        assert page["total_sheets"] == 3
        assert page["next_offset"] == (offset + 1 if offset < 2 else None)
        assert page["sheets"] == [{"name": names[offset], "max_row": 1, "max_column": 2}]
        pages.append(page)
    assert "private cell" not in str(pages)


@pytest.mark.parametrize(
    "arguments",
    [
        {"source": "../input.xlsx"},
        {"source": "input.xlsx", "offset": -1},
        {"source": "input.xlsx", "limit": 129},
        {"source": "input.xlsx", "limit": True},
    ],
)
def test_workbook_discovery_refuses_unsafe_or_unbounded_reads(
    arguments: JsonObject,
) -> None:
    """Discovery obeys the same detached namespace and explicit page limit as cell reads."""
    with pytest.raises(ValueError):
        WorkbookEditAdapter().access(
            ToolCall(call_id="describe", name="describe_workbook", arguments=arguments)
        )


def test_rectangular_read_does_not_build_a_full_workbook_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A one-cell query renders one cell, independent of unrelated worksheet contents."""

    def full_projection_forbidden(workbook: Workbook) -> None:
        raise AssertionError("bounded reads must not materialize the whole workbook")

    monkeypatch.setattr(
        "exp.runtime.environments.workbook_edits.workbook_projection", full_projection_forbidden
    )
    result = WorkbookEditAdapter().apply(
        {"input.xlsx": workbook_bytes()},
        (
            ToolCall(
                call_id="read",
                name="inspect_workbook",
                arguments={
                    "source": "input.xlsx",
                    "sheet": "Tickets",
                    "start_row": 3,
                    "start_column": 2,
                },
            ),
        ),
        ("output.xlsx",),
    )
    assert result.projection["input.xlsx"] == {
        "sheet": "Tickets",
        "max_row": 3,
        "max_column": 2,
        "start_row": 3,
        "start_column": 2,
        "rows": [
            [
                {
                    "value": "2022-12-22T00:00:00",
                    "data_type": "d",
                    "number_format": "mm-dd-yy",
                }
            ]
        ],
    }


def workbook_bytes(*, formula: bool = False) -> bytes:
    """Create an initial artifact whose date type cannot survive naive JSON rewriting."""
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = "Tickets"
    sheet.append(["id", "date"])
    sheet.append([1, datetime(2022, 12, 21)])
    sheet.append([2, datetime(2022, 12, 22)])
    sheet["B3"].number_format = "mm-dd-yy"
    if formula:
        sheet["A1"] = "=SUM(A2:A3)"
    stream = io.BytesIO()
    workbook.save(stream)
    return stream.getvalue()


def edit(**changes: str) -> ToolCall:
    """Declare a row edit without a shell or filesystem capability."""
    return ToolCall(
        call_id="delete-row",
        name=changes.get("name", "edit_workbook"),
        arguments={
            "source": "input.xlsx",
            "destination": changes.get("destination", "output.xlsx"),
            "deletions": [{"sheet": "Tickets", "start": 2, "count": 1}],
        },
    )


def test_native_dates_and_repeatable_bytes() -> None:
    """The output is an actual workbook with a native date, not an ISO string cell."""
    initial = workbook_bytes()
    adapter = WorkbookEditAdapter()
    first = adapter.apply({"input.xlsx": initial}, (edit(),), ("output.xlsx",))
    second = adapter.apply({"input.xlsx": initial}, (edit(),), ("output.xlsx",))
    assert first.files == second.files
    output = load_workbook(io.BytesIO(first.files["output.xlsx"]))["Tickets"]
    assert output.max_row == 2
    assert output["A2"].value == 2
    assert output["B2"].value == datetime(2022, 12, 22)
    assert output["B2"].data_type == "d"
    assert output["B2"].number_format == "mm-dd-yy"
    assert load_workbook(io.BytesIO(initial))["Tickets"].max_row == 3


@pytest.mark.parametrize(
    "change", [{"name": "Bash"}, {"destination": "../output.xlsx"}, {"destination": "other.xlsx"}]
)
def test_unadmitted_native_operations_are_refused(change: dict[str, str]) -> None:
    """Unknown effects, host-path escapes and undeclared outputs cannot execute."""
    with pytest.raises(ValueError):
        WorkbookEditAdapter().apply(
            {"input.xlsx": workbook_bytes()}, (edit(**change),), ("output.xlsx",)
        )


def test_formula_dependency_is_not_silently_broken() -> None:
    """The bounded adapter refuses features whose references require a full workbook engine."""
    with pytest.raises(ValueError, match="formulas"):
        WorkbookEditAdapter().apply(
            {"input.xlsx": workbook_bytes(formula=True)}, (edit(),), ("output.xlsx",)
        )


def test_worksheet_scoped_defined_name_is_refused() -> None:
    """A sheet-local named range would not shift with deleted rows, so it is refused."""
    workbook = load_workbook(io.BytesIO(workbook_bytes()))
    sheet = workbook["Tickets"]
    sheet.defined_names["Local"] = DefinedName("Local", attr_text="Tickets!$A$3:$A$3")
    buffer = io.BytesIO()
    workbook.save(buffer)
    assert not load_workbook(io.BytesIO(buffer.getvalue())).defined_names
    with pytest.raises(ValueError, match="defined names"):
        WorkbookEditAdapter().apply({"input.xlsx": buffer.getvalue()}, (edit(),), ("output.xlsx",))


def test_workbook_core_property_type_namespaces_remain_bound() -> None:
    """Canonicalizing metadata must preserve the prefixes used by XML Schema type values."""

    result = WorkbookEditAdapter().apply(
        {"input.xlsx": workbook_bytes()}, (edit(),), ("output.xlsx",)
    )
    with ZipFile(io.BytesIO(result.files["output.xlsx"])) as archive:
        xml = archive.read("docProps/core.xml")
    namespaces = {
        prefix for _, (prefix, uri) in ElementTree.iterparse(io.BytesIO(xml), events=("start-ns",))
    }
    for element in ElementTree.fromstring(xml).iter():
        value = element.get("{http://www.w3.org/2001/XMLSchema-instance}type")
        if value is not None and ":" in value:
            assert value.split(":", 1)[0] in namespaces


def test_deletions_follow_declared_sequence_not_sorted_source_rows() -> None:
    """Two edits address the current workbook after each preceding edit."""
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = "Tickets"
    for value in ("id", 1, 2, 3, 4):
        sheet.append([value])
    buffer = io.BytesIO()
    workbook.save(buffer)
    action = ToolCall(
        call_id="sequential",
        name="edit_workbook",
        arguments={
            "source": "input.xlsx",
            "destination": "output.xlsx",
            "deletions": [
                {"sheet": "Tickets", "start": 2, "count": 1},
                {"sheet": "Tickets", "start": 3, "count": 1},
            ],
        },
    )
    result = WorkbookEditAdapter().apply(
        {"input.xlsx": buffer.getvalue()}, (action,), ("output.xlsx",)
    )
    output = load_workbook(io.BytesIO(result.files["output.xlsx"]))["Tickets"]
    assert [row[0].value for row in output] == ["id", 2, 4]


def test_deletion_past_last_row_is_refused() -> None:
    """Openpyxl's permissive deletion cannot silently turn invalid edits into success."""
    original = edit()
    action = original.model_copy(
        update={
            "arguments": {
                **original.arguments,
                "deletions": [{"sheet": "Tickets", "start": 3, "count": 2}],
            }
        }
    )
    with pytest.raises(ValueError, match="beyond existing rows"):
        WorkbookEditAdapter().apply({"input.xlsx": workbook_bytes()}, (action,), ("output.xlsx",))


def test_access_resolves_source_and_destination_without_applying() -> None:
    """The adapter names the workbook an edit reads and the workbook it writes."""
    access = WorkbookEditAdapter().access(edit(destination="input.xlsx"))
    assert access.reads == frozenset({"input.xlsx"})
    assert access.writes == frozenset({"input.xlsx"})
    with pytest.raises(ValueError, match="edit_workbook"):
        WorkbookEditAdapter().access(edit(name="Bash"))


def test_numeric_precision_loss_is_refused_before_native_output_publication() -> None:
    """An unrelated retained cell must not change value during XLSX reserialization."""
    initial = workbook_bytes()
    rebuilt = io.BytesIO()
    with ZipFile(io.BytesIO(initial)) as source, ZipFile(rebuilt, "w") as target:
        for name in source.namelist():
            payload = source.read(name)
            if name == "xl/worksheets/sheet1.xml":
                assert b"<v>2</v>" in payload
                payload = payload.replace(b"<v>2</v>", b"<v>1092666.6666666667</v>")
            target.writestr(name, payload)
    original = rebuilt.getvalue()
    assert load_workbook(io.BytesIO(original))["Tickets"]["A3"].value == 1092666.6666666667
    with pytest.raises(ValueError, match="serialization changes retained cell"):
        WorkbookEditAdapter().apply({"input.xlsx": original}, (edit(),), ("output.xlsx",))
