"""Native workbook worlds and answer-range verifiers prepared from operator-declared files.

A native workbook capture needs an immutable input artifact holding the exact starting bytes, a
``WorkbookRuntimeDefinition`` whose probe digest binds its views, and a verifier whose paths
resolve in that probe. :func:`native_workbook_capture` publishes the starting files byte for
byte as one ``authored-task`` artifact, builds the definition and capture for one development
task, and, when a reference workbook and answer ranges are declared, compiles the reference
values into an evaluator-only answer-range verifier over private row views. It also reports the
native probe of the untouched world, which a simulated world can mirror so that one verifier
grades both. The reference workbook is read once and never stored in the project.
"""

import io
import re
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Self

from openpyxl import load_workbook
from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    JsonValue,
    sha256_bytes,
    validate_artifact_file_path,
)
from exp.common.project import ProjectStore, artifact_input
from exp.common.tasks import ToolSchema
from exp.optimize.evaluation.continuations.capture import CaptureDefinition
from exp.optimize.evaluation.executable import workbook as workbook_runtime
from exp.optimize.evaluation.executable.loader import load_runtime_definition
from exp.optimize.evaluation.executable.workbook import (
    WorkbookRuntimeDefinition,
    WorkbookStateView,
)
from exp.runtime.environments.workbook_edits import (
    WorkbookDescriptionArguments,
    WorkbookEditArguments,
    WorkbookReadArguments,
    read_supported_workbook,
    workbook_projection,
)
from exp.runtime.environments.world_contracts import StateDomain, ToolEffectPolicy, WorldManifest
from exp.simulation.development_scope import development_task
from exp.simulation.evaluation.tasks import OutcomeRule, TaskVerifier

_CELL = re.compile(r"([A-Z]{1,3})([1-9][0-9]{0,6})")
WORKBOOK_TOOLS = ("edit_workbook", "inspect_workbook", "describe_workbook")


class AnswerRange(ContractModel):
    """One rectangular answer range of the reference workbook.

    Attributes:
        sheet: Exact worksheet name in both the starting and reference workbooks.
        cells: A1 range such as ``A1:D12``, or one cell such as ``B3``.
    """

    sheet: str = Field(min_length=1, max_length=128)
    cells: str = Field(min_length=2, max_length=32)

    @model_validator(mode="after")
    def _a1_range(self) -> Self:
        """Require upper-case A1 corners in reading order."""
        corners = [_CELL.fullmatch(part) for part in self.cells.split(":")]
        if len(corners) > 2 or any(match is None for match in corners):
            raise ValueError(f"answer range {self.cells!r} must be A1 or A1:B2")
        (top, left), (bottom, right) = _corners(self.cells)
        if bottom < top or right < left:
            raise ValueError(f"answer range {self.cells!r} must run top-left to bottom-right")
        return self


class NativeWorkbook(ContractModel):
    """A native workbook world for one development task of the build.

    Attributes:
        task_id: Fit task of the completed build; its instruction and lineage are kept.
        files: World input paths, such as ``input.xlsx``, mapped to host files holding the
            exact starting bytes. Host paths are read once and never recorded.
        output_paths: Native output files the actor may create.
        tools: Public tools the task publishes; empty publishes the adapter's three tools.
        manifest: Native domain and tool policy; None declares every file native and lets
            each tool read every file, with ``edit_workbook`` writing the outputs.
        views: Private worksheet views beside the generated ``answer<i>`` views.
        reference_file: Evaluator-only reference workbook whose answer ranges the verifier
            requires in the first output; requires ``answer_ranges``.
        answer_ranges: Ranges of the reference workbook to require; requires reference_file.
        maximum_model_calls: Actor inference allowance of the capture.
        system_prompt: Optional fixed system prompt of the capture.
    """

    task_id: str = Field(min_length=1)
    files: dict[str, Path] = Field(min_length=1, max_length=32)
    output_paths: tuple[str, ...] = Field(default=("output.xlsx",), min_length=1, max_length=32)
    tools: tuple[ToolSchema, ...] = ()
    manifest: WorldManifest | None = None
    views: dict[str, WorkbookStateView] = Field(default_factory=dict, max_length=16)
    reference_file: Path | None = None
    answer_ranges: tuple[AnswerRange, ...] = Field(default=(), max_length=16)
    maximum_model_calls: int = Field(ge=1, le=1000, strict=True)
    system_prompt: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        """Require canonical world paths and a reference only together with its ranges."""
        for name in self.files:
            if validate_artifact_file_path(name).as_posix() != name or not name.endswith(".xlsx"):
                raise ValueError(f"input path {name!r} must be a canonical relative .xlsx path")
        if (self.reference_file is None) != (not self.answer_ranges):
            raise ValueError("reference_file and answer_ranges go together")
        clashes = [name for name in self.views if re.fullmatch(r"answer[0-9]+", name)]
        if clashes and self.answer_ranges:
            raise ValueError("answer<i> views are generated from answer_ranges; rename them")
        return self


def _corners(cells: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """Zero-based (row, column) of a range's top-left and bottom-right cells."""
    parts = cells.split(":")

    def index(reference: str) -> tuple[int, int]:
        """One A1 reference as a zero-based (row, column)."""
        match = _CELL.fullmatch(reference)
        if match is None:
            raise ValueError(f"invalid cell reference {reference!r}")
        column = 0
        for character in match.group(1):
            column = column * 26 + ord(character) - 64
        return int(match.group(2)) - 1, column - 1

    return index(parts[0]), index(parts[-1])


def default_tools() -> tuple[ToolSchema, ...]:
    """The native workbook adapter's three public tools with their exact argument schemas."""
    return (
        ToolSchema(
            name="edit_workbook",
            description="Delete explicit rows from a workbook, writing the destination file.",
            input_schema=WorkbookEditArguments.model_json_schema(),
        ),
        ToolSchema(
            name="inspect_workbook",
            description="Read a rectangular range of typed cells from input.xlsx or output.xlsx.",
            input_schema=WorkbookReadArguments.model_json_schema(),
        ),
        ToolSchema(
            name="describe_workbook",
            description="List sheet names and dimensions of a workbook.",
            input_schema=WorkbookDescriptionArguments.model_json_schema(),
        ),
    )


def default_manifest(inputs: tuple[str, ...], outputs: tuple[str, ...]) -> WorldManifest:
    """Every file native; reads see every file and only ``edit_workbook`` writes outputs."""
    names = (*inputs, *outputs)
    return WorldManifest(
        domains=tuple(StateDomain(name=name, owner="native") for name in names),
        tools=(
            ToolEffectPolicy(name="edit_workbook", reads=names, writes=outputs),
            ToolEffectPolicy(name="inspect_workbook", reads=names, writes=()),
            ToolEffectPolicy(name="describe_workbook", reads=names, writes=()),
        ),
    )


def _reference_values(payload: bytes, answer: AnswerRange) -> list[list[JsonValue]]:
    """Reference cell values over one range, typed as the native probe types outputs."""
    (top, left), (bottom, right) = _corners(answer.cells)
    try:
        sheets = workbook_projection(read_supported_workbook(payload))["sheets"]
        if not isinstance(sheets, list):
            raise ValueError("malformed reference projection")
        grid = next(
            row["rows"] for row in sheets if isinstance(row, dict) and row["name"] == answer.sheet
        )
        if not isinstance(grid, list):
            raise ValueError("malformed reference projection")

        def at(row: int, column: int) -> JsonValue:
            """One projected reference value, or None outside the stored grid."""
            line = grid[row] if row < len(grid) else []
            cell = line[column] if isinstance(line, list) and column < len(line) else None
            return cell["value"] if isinstance(cell, dict) else None

    except StopIteration as error:
        raise ValueError(f"reference workbook has no sheet {answer.sheet!r}") from error
    except ValueError:
        # Reference workbooks may carry parts the native adapter refuses; their cached values
        # are still plain JSON when openpyxl reads them, and anything else is refused.
        worksheet = load_workbook(io.BytesIO(payload), data_only=True)[answer.sheet]

        def at(row: int, column: int) -> JsonValue:
            """One cached reference value read by openpyxl."""
            value = worksheet.cell(row=row + 1, column=column + 1).value
            if isinstance(value, datetime):
                return value.isoformat()
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise ValueError(f"reference cell value {value!r} is not plain JSON") from None
            return value

    columns = range(left, right + 1)
    return [[at(row, column) for column in columns] for row in range(top, bottom + 1)]


def answer_range_verifier(
    reference: bytes, answers: tuple[AnswerRange, ...], output: str, task_id: str
) -> tuple[TaskVerifier, dict[str, WorkbookStateView]]:
    """Compile reference answer ranges into private views and a strict equality verifier.

    Every reference cell in a range must equal the output's projected value by strict JSON
    type; a blank reference cell requires a blank output cell. Rows and columns before a range
    are unconstrained, and the output must have been written.

    Args:
        reference: Evaluator-only reference workbook bytes.
        answers: Ranges to require.
        output: Output file the views read.
        task_id: Task named in the verifier description.

    Returns:
        The verifier and one ``answer<i>`` row view per range.
    """
    views: dict[str, WorkbookStateView] = {}
    properties: JsonObject = {}
    for index, answer in enumerate(answers):
        name = f"answer{index}"
        (top, left), _ = _corners(answer.cells)
        values = _reference_values(reference, answer)
        rows: list[JsonValue] = [{} for _ in range(top)]
        last_row = -1
        for offset, line in enumerate(values):
            items: list[JsonValue] = [{} for _ in range(left)]
            last_column = -1
            for column, value in enumerate(line):
                if value is None:
                    items.append({"type": "object", "properties": {"value": {"type": "null"}}})
                else:
                    last_column = column
                    items.append(
                        {
                            "type": "object",
                            "required": ["value"],
                            "properties": {"value": {"const": value}},
                        }
                    )
            row: JsonObject = {"type": "array", "prefixItems": items}
            if last_column >= 0:
                row["minItems"] = left + last_column + 1
                last_row = offset
            rows.append(row)
        sheet: JsonObject = {"type": "array", "prefixItems": rows}
        if last_row >= 0:
            sheet["minItems"] = top + last_row + 1
        properties[name] = {
            "type": "object",
            "required": ["source", "sheets"],
            "properties": {
                "source": {"const": output},
                "sheets": {
                    "type": "array",
                    "minItems": 1,
                    "prefixItems": [
                        {
                            "type": "object",
                            "required": ["rows"],
                            "properties": {"name": {"const": answer.sheet}, "rows": sheet},
                        }
                    ],
                },
            },
        }
        views[name] = WorkbookStateView(
            input_file="input.xlsx", output_file=output, sheets=(answer.sheet,), include_rows=True
        )
    ranges = ", ".join(f"{row.sheet}!{row.cells}" for row in answers)
    verifier = TaskVerifier(
        description=(
            f"Reference answer check for {task_id}: {output} ranges {ranges} equal the "
            "evaluator-only reference workbook."
        ),
        outcome_schema={
            "type": "object",
            "required": ["final_state"],
            "properties": {
                "final_state": {
                    "type": "object",
                    "required": ["views"],
                    "properties": {
                        "views": {
                            "type": "object",
                            "required": list(properties),
                            "properties": properties,
                        }
                    },
                }
            },
        },
        rules=tuple(
            OutcomeRule(path=f"/final_state/views/{name}/source", operator="equals", value=output)
            for name in properties
        ),
        allowed_tools=WORKBOOK_TOOLS,
    )
    return verifier, views


def _input_artifact(store: ProjectStore, files: dict[str, bytes], revision: str) -> ArtifactInput:
    """Publish the starting files byte for byte once, keyed by their content."""
    listing = b"".join(
        name.encode() + b"\0" + sha256_bytes(data).encode() for name, data in files.items()
    )
    digest = sha256(listing, usedforsecurity=False).hexdigest()
    identity = f"authored-workbook-{digest[:20]}"
    if store.artifacts.exists(identity):
        saved = store.artifacts.read(identity).manifest
        for name, payload in files.items():
            if store.artifacts.read_bytes(identity, name) != payload:
                raise ValueError(f"saved workbook input {identity} differs from {name}")
        return artifact_input(saved)
    return artifact_input(
        store.artifacts.write(
            artifact_id=identity,
            artifact_type="authored-task",
            envelope=ArtifactEnvelope(
                schema_version=1, created_at=datetime.now(UTC), code_revision=revision
            ),
            files=files,
        )
    )


def native_workbook_capture(
    store: ProjectStore, request: NativeWorkbook, *, revision: str, publish: bool
) -> JsonObject:
    """Prepare a native workbook capture, its answer verifier and its untouched projection.

    Args:
        store: Project with a completed build.
        request: Task, starting files, outputs, views and optional reference answers.
        revision: Installed execution revision recorded on the published input.
        publish: Publish the input artifact; False validates and reports without writing.

    Returns:
        ``capture`` (a ready capture operation payload), ``verifier`` (or None),
        ``initial_projection`` (the native probe before any action, or None when not
        published) and ``input_artifact``.
    """
    task = development_task(store, request.task_id)
    task = task.model_copy(update={"tools": request.tools or default_tools()})
    files = {name: path.read_bytes() for name, path in request.files.items()}
    for name, payload in files.items():
        try:
            read_supported_workbook(payload)
        except ValueError as error:
            raise ValueError(f"the native workbook adapter refuses {name}: {error}") from error
    views = dict(request.views)
    verifier: TaskVerifier | None = None
    if request.reference_file is not None:
        verifier, generated = answer_range_verifier(
            request.reference_file.read_bytes(),
            request.answer_ranges,
            request.output_paths[0],
            task.task_id,
        )
        views.update(generated)
    manifest = request.manifest or default_manifest(tuple(files), request.output_paths)
    if publish:
        reference = _input_artifact(store, files, revision)
    else:
        reference = ArtifactInput(artifact_id="authored-workbook-dry-run", sha256="0" * 64)
    definition = WorkbookRuntimeDefinition(
        input_artifact=reference,
        files={name: name for name in files},
        output_paths=request.output_paths,
        manifest=manifest,
        views=views,
    )
    build = store.load_project().build
    if build is None:
        raise ValueError("a native workbook capture needs a completed build")
    runtime_file = Path(workbook_runtime.__file__).resolve()
    capture = CaptureDefinition(
        task=task,
        sources=(reference, build.trace_dataset),
        runtime_file=runtime_file,
        runtime_sha256=sha256_bytes(runtime_file.read_bytes()),
        configuration=definition.model_dump(mode="json"),
        probe_sha256=definition.probe_sha256,
        maximum_model_calls=request.maximum_model_calls,
        system_prompt=request.system_prompt,
    )
    projection: JsonObject | None = None
    if publish:
        with (
            load_runtime_definition(
                store,
                runtime_file=capture.runtime_file,
                runtime_sha256=capture.runtime_sha256,
                configuration=capture.configuration,
                probe_sha256=capture.probe_sha256,
            ) as runtime,
            runtime.world.open(task) as world,
        ):
            projection = runtime.probe(world.checkpoint())
    return {
        "input_artifact": reference.model_dump(mode="json"),
        "capture": {"action": "capture", "capture": capture.model_dump(mode="json")},
        "verifier": None if verifier is None else verifier.model_dump(mode="json"),
        "initial_projection": projection,
    }
