"""Native workbook edits preserve types, reject unsafe scope and replay deterministically."""

import io
from datetime import datetime
from xml.etree import ElementTree
from zipfile import ZipFile

import pytest
from openpyxl import Workbook, load_workbook
from openpyxl.workbook.defined_name import DefinedName

from exp.common.models import ToolCall
from exp.runtime.environments.workbook_edits import WorkbookEditAdapter


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
