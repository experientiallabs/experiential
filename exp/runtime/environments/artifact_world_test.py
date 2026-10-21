"""Artifact worlds: exact file images, adapter refusals and declared paths."""

import io
import json

import pytest
from openpyxl import Workbook, load_workbook

from exp.common.models import ToolCall
from exp.common.tasks import TaskCase
from exp.runtime.environments.artifact_world import ArtifactWorldFactory
from exp.runtime.environments.workbook_edits import WorkbookEditAdapter
from exp.runtime.environments.world_contracts import NativeImage
from exp.runtime.environments.worlds import WorldContext


def _workbook() -> bytes:
    """Return a two-row typed workbook."""
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = "Data"
    sheet.append(["id", "value"])
    sheet.append([1, 2.5])
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def _factory() -> ArtifactWorldFactory:
    """Bind one input workbook and one output path."""
    return ArtifactWorldFactory(
        WorkbookEditAdapter(), files={"in.xlsx": _workbook()}, outputs=("out.xlsx",)
    )


def _task() -> TaskCase:
    """Return a task with no tools; the backend is driven directly."""
    return TaskCase(
        task_id="artifacts",
        lineage_group_id="artifacts",
        partition="fit",
        instruction="Fixture.",
        workload_weight=1,
        origin="authored",
    )


def _delete(start: int) -> ToolCall:
    """Delete one row of the input into the output."""
    return ToolCall(
        call_id=f"delete-{start}",
        name="edit_workbook",
        arguments={
            "source": "in.xlsx",
            "destination": "out.xlsx",
            "deletions": [{"sheet": "Data", "start": start, "count": 1}],
        },
    )


def test_image_holds_exact_bytes_and_marks_unwritten_outputs() -> None:
    """Captured parts are the native files; an output exists only after it is written."""
    source = _workbook()
    factory = ArtifactWorldFactory(
        WorkbookEditAdapter(), files={"in.xlsx": source}, outputs=("out.xlsx",)
    )
    world = factory.create(_task())
    image = world.capture()
    assert image.parts == {"in.xlsx": source}
    absent = image.domains["out.xlsx"]
    reply = world.execute(_delete(2), WorldContext.__new__(WorldContext))
    assert not reply.is_error
    written = world.capture()
    assert written.domains["out.xlsx"] != absent
    restored = _factory().restore(written)
    assert restored.capture() == written
    sheet = load_workbook(io.BytesIO(written.parts["out.xlsx"]))["Data"]
    assert sheet.max_row == 1


def test_adapter_refusal_is_an_error_observation_without_change() -> None:
    """An out-of-range edit is reported to the actor and writes nothing."""
    world = _factory().create(_task())
    before = world.capture()
    reply = world.execute(_delete(9), WorldContext.__new__(WorldContext))
    assert reply.is_error and "beyond existing rows" in reply.content
    assert world.capture() == before


def test_paths_and_images_stay_within_declared_files() -> None:
    """Noncanonical paths and images with undeclared files are refused."""
    with pytest.raises(ValueError, match="relative POSIX"):
        ArtifactWorldFactory(WorkbookEditAdapter(), files={}, outputs=("../out.xlsx",))
    with pytest.raises(ValueError, match="at least one output"):
        ArtifactWorldFactory(WorkbookEditAdapter(), files={}, outputs=())
    image = NativeImage(parts={"other.xlsx": b""}, domains={})
    with pytest.raises(ValueError, match="undeclared files"):
        _factory().restore(image)


def test_write_acknowledgement_excludes_full_projection_but_keeps_actual_bytes() -> None:
    """An edit acknowledges committed output; cells are available through explicit reads."""
    world = _factory().create(_task())
    reply = world.execute(_delete(1), WorldContext.__new__(WorldContext))
    value = json.loads(reply.content)
    assert set(value) == {"written"}
    assert value["written"]["out.xlsx"]["sha256"] == world.capture().domains["out.xlsx"]
    assert len(reply.content) < 200
    read = world.execute(
        ToolCall(
            call_id="read",
            name="inspect_workbook",
            arguments={
                "source": "out.xlsx",
                "sheet": "Data",
                "row_count": 1,
                "column_count": 2,
            },
        ),
        WorldContext.__new__(WorldContext),
    )
    assert json.loads(read.content)["projection"]["out.xlsx"]["rows"][0][1]["value"] == 2.5
