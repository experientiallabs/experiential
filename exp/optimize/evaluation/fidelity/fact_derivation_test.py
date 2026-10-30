"""Deriving workbook facts for declared task inputs and attaching them to their owners."""

import io
import json
from datetime import time
from pathlib import Path

import pytest
from openpyxl import Workbook
from pydantic import ValidationError

from exp.common.project import ProjectStore
from exp.common.tasks import TaskCase
from exp.common.tasks.facts import InstanceFactsError
from exp.common.traces.facts import TracePackageFacts
from exp.common.traces.ingest.chat_json import CHAT_JSON_SOURCE
from exp.optimize.evaluation.fidelity.fact_derivation import (
    FactsDerivation,
    attach_trace_package,
    derive_facts,
    save_derived_facts,
)
from exp.optimize.evaluation.fidelity.instance_facts import ReplayFacts
from exp.runtime.environments.workbook_facts import decode_workbook_facts


def workbook_file(path: Path, rows: list[list[object]]) -> Path:
    """Write a one-sheet XLSX with ``rows`` and return its path."""
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    for row in rows:
        sheet.append(row)
    buffer = io.BytesIO()
    workbook.save(buffer)
    path.write_bytes(buffer.getvalue())
    return path


def conversation(key: str) -> dict[str, object]:
    """One chat JSON conversation that reads a workbook."""
    return {
        "trace_id": key,
        "messages": [
            {"role": "user", "content": "Sum column B"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "python", "arguments": '{"code": "print(1)"}'},
                    }
                ],
            },
            {"role": "tool", "name": "python", "tool_call_id": "c1", "content": "1"},
            {"role": "assistant", "content": "Done."},
        ],
    }


def spec(tmp_path: Path, **extra: object) -> FactsDerivation:
    """A derivation over two traces' workbooks and a third declared trace's broken file."""
    first = workbook_file(tmp_path / "a.xlsx", [["when", time(19, 30)], ["n", 2]])
    second = workbook_file(tmp_path / "b.xlsx", [["x", 1.5]])
    broken = tmp_path / "broken.xlsx"
    broken.write_bytes(b"not a workbook")
    return FactsDerivation.model_validate(
        {
            "deriver": {"kind": "xlsx-workbook"},
            "inputs": [
                {"trace": "sb-1", "file": str(first)},
                {"trace": "sb-2", "file": str(second), "path": "data/lookup.xlsx"},
                {"trace": "sb-3", "file": str(broken)},
            ],
            **extra,
        }
    )


def test_refusals_stop_derivation_unless_declared(tmp_path: Path) -> None:
    """A refused input stops the action by default and is reported when allowed."""
    with pytest.raises(InstanceFactsError, match="refused 1 of 3 inputs: sb-3 input.xlsx"):
        derive_facts(spec(tmp_path))
    result = derive_facts(spec(tmp_path, allow_refusals=True))
    assert sorted(result.facts) == ["sb-1", "sb-2"]
    assert [row.trace for row in result.refused] == ["sb-3"]
    assert "not an XLSX archive" in str(result.refused[0].refusal)
    (fact,) = result.facts["sb-2"].facts.facts
    assert fact.key == "data/lookup.xlsx/workbook" and fact.provenance.source == "data/lookup.xlsx"
    assert str(tmp_path) not in json.dumps(result.facts["sb-2"].model_dump(mode="json"))
    assert result.identity == derive_facts(spec(tmp_path, allow_refusals=True)).identity


def test_facts_reach_trace_package_facts_after_import(tmp_path: Path) -> None:
    """The annotated package imports with each trace's validated facts for its own replay."""
    package = tmp_path / "traces.json"
    package.write_text(json.dumps([conversation("sb-1"), conversation("sb-2")]))
    output = tmp_path / "traces.facts.json"
    derivation = spec(tmp_path, allow_refusals=True, traces=str(package), output=str(output))
    result = derive_facts(derivation)
    assert attach_trace_package(derivation, result, write=False) == {"sb-1", "sb-2"}
    assert not output.exists()
    attach_trace_package(derivation, result, write=True)
    traces = CHAT_JSON_SOURCE.load(output).traces
    first = next(row for row in traces if row.spans[0].attributes["exp.source.trace.id"] == "sb-1")
    task = TaskCase(
        task_id="task",
        lineage_group_id="lineage",
        partition="fit",
        instruction="Sum column B",
        workload_weight=1,
        source_trace_ids=(first.trace_id,),
    )
    facts = TracePackageFacts(first).initial_facts(task)
    assert facts is not None
    cells = decode_workbook_facts(facts.facts[0].value).sheets[0].cells
    assert cells["B1"].value == time(19, 30)


def test_every_trace_with_facts_must_match_a_conversation(tmp_path: Path) -> None:
    """Facts for a trace the package does not contain refuse before anything is written."""
    package = tmp_path / "traces.json"
    package.write_text(json.dumps([conversation("sb-1")]))
    output = tmp_path / "out.json"
    derivation = spec(tmp_path, allow_refusals=True, traces=str(package), output=str(output))
    with pytest.raises(ValueError, match="has no conversation keyed 'sb-2'"):
        attach_trace_package(derivation, derive_facts(derivation), write=True)
    assert not output.exists()


def test_saved_facts_are_replay_facts_for_runtime_definitions(tmp_path: Path) -> None:
    """``facts.json`` maps each trace to a value the grounded definition's facts accept."""
    store = ProjectStore(tmp_path / "root", "case")
    result = derive_facts(spec(tmp_path, allow_refusals=True))
    path = save_derived_facts(store, result)
    assert path == store.paths.runtime_directory / "facts" / result.identity / "facts.json"
    saved = json.loads(path.read_text())
    assert ReplayFacts.model_validate(saved["sb-1"]) == result.facts["sb-1"]
    assert saved["sb-1"]["first_version"] == 0


def test_spec_refuses_repeated_inputs_paths_and_unpaired_outputs(tmp_path: Path) -> None:
    """Invalid declarations fail at validation with what to change."""
    file = str(tmp_path / "a.xlsx")
    base = {"deriver": {"kind": "xlsx-workbook"}}
    with pytest.raises(ValidationError, match="declares 'input.xlsx' twice"):
        FactsDerivation.model_validate(
            {**base, "inputs": [{"trace": "t", "file": file}, {"trace": "t", "file": file}]}
        )
    with pytest.raises(ValidationError, match="relative POSIX paths"):
        FactsDerivation.model_validate(
            {**base, "inputs": [{"trace": "t", "file": file, "path": "/abs/input.xlsx"}]}
        )
    with pytest.raises(ValidationError, match="traces and output go together"):
        FactsDerivation.model_validate(
            {**base, "inputs": [{"trace": "t", "file": file}], "traces": file}
        )
    with pytest.raises(ValidationError, match="output must be a new file"):
        FactsDerivation.model_validate(
            {**base, "inputs": [{"trace": "t", "file": file}], "traces": file, "output": file}
        )
    with pytest.raises(ValidationError, match="kind"):
        FactsDerivation.model_validate(
            {"deriver": {"kind": "sqlite"}, "inputs": [{"trace": "t", "file": file}]}
        )
