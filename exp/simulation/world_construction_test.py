"""Construction preserves shared facts, refuses ambiguity and emits executable assets."""

import io
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError
from openpyxl import load_workbook
from pydantic import JsonValue

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput
from exp.common.project import ProjectStore, artifact_input
from exp.simulation.evaluation.admission import verify_label_witness
from exp.simulation.evaluation.admission_test import specification, witnessed
from exp.simulation.world_construction import (
    ConstructionSpec,
    LookupColumn,
    SequenceColumn,
    TablePlan,
    ValuesColumn,
    compile_construction,
    load_construction,
    persist_construction,
)


def linked_spec() -> ConstructionSpec:
    """Author two related tables without copying a customer's name into each order."""
    return ConstructionSpec(
        name="linked-orders",
        assumptions=("Authored examples, no estimated production frequencies.",),
        tables={
            "Customers": TablePlan(
                rows=2,
                key="id",
                columns={
                    "id": SequenceColumn(kind="sequence"),
                    "name": ValuesColumn(kind="values", values=("Ada", "Bo")),
                },
            ),
            "Orders": TablePlan(
                rows=3,
                columns={
                    "customer": ValuesColumn(kind="values", values=(2, 1, 2)),
                    "name": LookupColumn(
                        kind="lookup",
                        table="Customers",
                        key_column="id",
                        value_column="name",
                        key_from="customer",
                    ),
                },
            ),
        },
        workbooks={"orders.xlsx": ("Customers", "Orders")},
    )


def test_linked_facts_and_replay(tmp_path: Path) -> None:
    """Relationships survive both renderings and digest-bound persisted reconstruction."""
    spec = linked_spec()
    assets = compile_construction(spec)
    tables = json.loads(assets["assets/tables.json"])
    assert tables["Orders"] == {
        "columns": ["customer", "name"],
        "rows": [[2, "Bo"], [1, "Ada"], [2, "Bo"]],
    }
    book = load_workbook(io.BytesIO(assets["assets/orders.xlsx"]))
    assert list(book["Orders"].values) == [(2, "Bo"), (1, "Ada"), (2, "Bo")]
    book.close()
    assert (
        compile_construction(ConstructionSpec.model_validate_json(spec.model_dump_json())) == assets
    )
    project = ProjectStore(tmp_path, "construction")
    reference = persist_construction(
        project.artifacts, spec, created_at=datetime.now(UTC), code_revision="test"
    )
    assert load_construction(project.artifacts, reference) == spec
    assert (
        persist_construction(
            project.artifacts, spec, created_at=datetime.now(UTC), code_revision="test"
        )
        == reference
    )


@pytest.mark.parametrize(
    "keys,match", [((1, 1), "duplicate"), ((None, 2), "null"), ((3, 4), "missing lookup")]
)
def test_lookup_rejects_ambiguous_or_missing_evidence(
    keys: tuple[int | None, ...], match: str
) -> None:
    """No missing or ambiguous foreign key silently becomes a guessed value."""
    raw = linked_spec().model_dump(mode="json")
    raw["tables"]["Customers"]["columns"]["id"] = {"kind": "values", "values": keys}
    with pytest.raises(ValueError, match=match):
        compile_construction(ConstructionSpec.model_validate(raw))


def test_cycle_and_incomplete_vector_refuse() -> None:
    """Unresolvable initial dependencies and ragged vectors never produce partial assets."""
    raw = linked_spec().model_dump(mode="json")
    raw["tables"]["Customers"]["columns"]["name"] = {
        "kind": "lookup",
        "table": "Orders",
        "key_column": "customer",
        "value_column": "name",
        "key_from": "id",
    }
    with pytest.raises(ValueError, match="cycle"):
        compile_construction(ConstructionSpec.model_validate(raw))
    raw = linked_spec().model_dump(mode="json")
    raw["tables"]["Orders"]["columns"]["customer"]["values"] = [1, 2]
    with pytest.raises(ValueError, match="length"):
        compile_construction(ConstructionSpec.model_validate(raw))


@pytest.mark.parametrize("values", [[float("nan")], [float("inf")], [{"x": 1}], [[1]]])
def test_non_scalar_and_non_finite_facts_refuse(values: list[JsonValue]) -> None:
    """Tables never flatten nested facts or serialize non-finite numeric observations."""
    with pytest.raises(ValueError):
        ValuesColumn.model_validate({"values": values})


@pytest.mark.parametrize("missing", ["table", "key_column", "value_column", "key_from"])
def test_undefined_dependencies_refuse(missing: str) -> None:
    """Every referenced table and field must be declared before construction."""
    raw = linked_spec().model_dump(mode="json")
    raw["tables"]["Orders"]["columns"]["name"][missing] = "Absent"
    with pytest.raises(ValueError, match="unknown lookup dependency"):
        ConstructionSpec.model_validate(raw)


@pytest.mark.parametrize("value", ["", 1234567890123456789, "bad\x00cell"])
def test_xlsx_refuses_lossy_cells(value: str | int) -> None:
    """A valid JSON scalar must not silently become another observation in XLSX."""
    spec = ConstructionSpec(
        name="lossy",
        assumptions=("test",),
        tables={
            "Data": TablePlan(rows=1, columns={"x": ValuesColumn(kind="values", values=(value,))})
        },
        workbooks={"input.xlsx": ("Data",)},
    )
    with pytest.raises(ValueError, match="XLSX"):
        compile_construction(spec)


def test_formula_shaped_text_is_literal() -> None:
    """Untrusted source text cannot become a spreadsheet formula during rendering."""
    spec = ConstructionSpec(
        name="literal",
        assumptions=("test",),
        tables={
            "Data": TablePlan(rows=1, columns={"x": ValuesColumn(kind="values", values=("=1+2",))})
        },
        workbooks={"input.xlsx": ("Data",)},
    )
    book = load_workbook(io.BytesIO(compile_construction(spec)["assets/input.xlsx"]))
    assert book["Data"]["A1"].value == "=1+2"
    assert book["Data"]["A1"].data_type == "s"
    book.close()


@pytest.mark.parametrize(
    "filename",
    ["../input.xlsx", "/input.xlsx", "nested/input.xlsx", "input.xls", "input.xlsx/extra"],
)
def test_unsafe_paths_refuse(filename: str) -> None:
    """Rendering never accepts a host path or an ambiguous archive path."""
    raw = linked_spec().model_dump(mode="json")
    raw["workbooks"] = {filename: ["Orders"]}
    with pytest.raises(ValueError, match="filename"):
        ConstructionSpec.model_validate(raw)


def test_budget_applies_to_broadcasts_and_repeated_rendering() -> None:
    """Small definitions cannot expand into unbounded tables or duplicate assets."""
    spec = ConstructionSpec(
        name="big",
        assumptions=("test",),
        tables={
            "Data": TablePlan(
                rows=10000, columns={"text": ValuesColumn(kind="values", values=("x" * 32767,))}
            )
        },
    )
    with pytest.raises(ValueError, match="8 MiB"):
        compile_construction(spec)
    raw = spec.model_dump(mode="json")
    raw["workbooks"] = {f"book{i}.xlsx": ["Data"] for i in range(11)}
    with pytest.raises(ValueError, match="rendered cells"):
        ConstructionSpec.model_validate(raw)


def test_generated_world_cannot_certify_its_own_derived_labels(tmp_path: Path) -> None:
    """A native-looking file does not erase the authority of its generated ancestors."""
    project = ProjectStore(tmp_path, "construction")
    reference = persist_construction(
        project.artifacts, linked_spec(), created_at=datetime.now(UTC), code_revision="test"
    )
    derived = specification(project, sources=(reference,))
    label = witnessed(project, derived, "done")
    refusal = verify_label_witness(project.artifacts, label, label="pass", compared=())
    assert refusal is not None and refusal.reason == "self_approval"
    assert reference.artifact_id in refusal.detail
    assert (
        verify_label_witness(
            project.artifacts,
            witnessed(project, specification(project), "done"),
            label="pass",
            compared=(),
        )
        is None
    )


def test_changed_sources_and_saved_payloads_refuse(tmp_path: Path) -> None:
    """Neither a source digest nor a saved report substitutes for actual constructed bytes."""
    project = ProjectStore(tmp_path, "construction")
    source = specification(project)
    spec = linked_spec().model_copy(update={"sources": (source,)})
    reference = persist_construction(
        project.artifacts, spec, created_at=datetime.now(UTC), code_revision="test"
    )
    assert load_construction(project.artifacts, reference) == spec
    bad = source.model_copy(update={"sha256": "0" * 64})
    with pytest.raises(ValueError, match="source changed"):
        persist_construction(
            project.artifacts,
            spec.model_copy(update={"sources": (bad,)}),
            created_at=datetime.now(UTC),
            code_revision="test",
        )
    with pytest.raises(ValueError, match="reference differs"):
        load_construction(project.artifacts, reference.model_copy(update={"sha256": "0" * 64}))
    other = ProjectStore(tmp_path, "tampered")
    clean = linked_spec()
    # A separately hashed artifact still needs to agree with the compiler, not just itself.
    original = persist_construction(
        other.artifacts, clean, created_at=datetime.now(UTC), code_revision="test"
    )
    independent = ProjectStore(tmp_path, "forged")
    original_manifest = other.artifacts.read(original.artifact_id).manifest
    forged_files = {
        row.path: other.artifacts.read_bytes(original.artifact_id, row.path)
        for row in original_manifest.files
    }
    forged_files["assets/tables.json"] = b"{}"
    forged = independent.artifacts.write(
        artifact_id=original.artifact_id,
        artifact_type="world-construction",
        envelope=ArtifactEnvelope.model_validate(
            original_manifest.model_dump(
                include={"schema_version", "created_at", "code_revision", "source", "inputs"}
            )
        ),
        files=forged_files,
    )
    with pytest.raises(ValueError, match="differs"):
        load_construction(independent.artifacts, artifact_input(forged))


def test_compiler_drift_and_concurrent_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent publication adopts one artifact; a changed writer requires reconstruction."""
    project = ProjectStore(tmp_path, "construction")

    def publish(_: int) -> ArtifactInput:
        """Run an independent publisher with its own timestamp."""
        return persist_construction(
            project.artifacts, linked_spec(), created_at=datetime.now(UTC), code_revision="test"
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        references = list(pool.map(publish, range(2)))
    assert references[0] == references[1]
    monkeypatch.setattr("exp.simulation.world_construction.version", lambda _: "changed")
    with pytest.raises(ValueError, match="compiler changed"):
        load_construction(project.artifacts, references[0])


def test_mapping_order_cannot_change_rendered_bytes() -> None:
    """JSON key order is metadata, including when resolving dependent tables."""
    spec = linked_spec()
    raw = spec.model_dump(mode="json")
    raw["tables"] = dict(reversed(raw["tables"].items()))
    for table in raw["tables"].values():
        table["columns"] = dict(reversed(table["columns"].items()))
    assert compile_construction(spec) == compile_construction(ConstructionSpec.model_validate(raw))


@pytest.mark.parametrize("value", [1.0, None])
def test_xlsx_cannot_change_scalar_types_or_dimensions(value: float | None) -> None:
    """Integral floats and trailing blank dimensions are explicitly refused in XLSX."""
    spec = ConstructionSpec(
        name="shape",
        assumptions=("test",),
        tables={
            "Data": TablePlan(rows=2, columns={"x": ValuesColumn(kind="values", values=(value,))})
        },
        workbooks={"input.xlsx": ("Data",)},
    )
    with pytest.raises(ValueError, match="XLSX loses"):
        compile_construction(spec)


def test_native_deletion_distinguishes_interleaved_keys() -> None:
    """Native execution distinguishes global duplicate removal from an adjacency shortcut."""
    from exp.common.models import ToolCall
    from exp.runtime.environments.workbook_edits import WorkbookEditAdapter

    final_rows = []
    for keys in ((1, 1, 1, 2), (1, 2, 1, 1)):
        spec = ConstructionSpec(
            name="duplicate-boundary",
            assumptions=("authored",),
            tables={
                "Data": TablePlan(
                    rows=4,
                    columns={
                        "id": SequenceColumn(kind="sequence"),
                        "key": ValuesColumn(kind="values", values=keys),
                    },
                )
            },
            workbooks={"input.xlsx": ("Data",)},
        )
        payload = compile_construction(spec)["assets/input.xlsx"]
        global_drop = [i for i, key in enumerate(keys, 1) if key in keys[i:]]
        adjacent_drop = [i + 1 for i in range(len(keys) - 1) if keys[i] == keys[i + 1]]
        outputs = []
        for drops in (global_drop, adjacent_drop):
            result = WorkbookEditAdapter().apply(
                {"input.xlsx": payload},
                (
                    ToolCall(
                        name="edit_workbook",
                        call_id="edit",
                        arguments={
                            "source": "input.xlsx",
                            "destination": "output.xlsx",
                            "deletions": [
                                {"sheet": "Data", "start": row, "count": 1}
                                for row in sorted(drops, reverse=True)
                            ],
                        },
                    ),
                ),
                output_paths=("output.xlsx",),
            )
            book = load_workbook(io.BytesIO(result.files["output.xlsx"]))
            outputs.append(list(book["Data"].values))
            book.close()
        final_rows.append(outputs)
    assert final_rows == [
        [[(3, 1), (4, 2)], [(3, 1), (4, 2)]],
        [[(2, 2), (4, 1)], [(1, 1), (2, 2), (4, 1)]],
    ]


def test_schema_requires_every_column_discriminator() -> None:
    """A schema-valid author response must not fail because the union needs an omitted kind."""
    original = linked_spec().model_dump(mode="json")
    validator = Draft202012Validator(ConstructionSpec.model_json_schema())
    validator.validate(original)
    for table in original["tables"].values():
        for column in table["columns"].values():
            kind = column.pop("kind")
            with pytest.raises(ValidationError):
                validator.validate(original)
            column["kind"] = kind
