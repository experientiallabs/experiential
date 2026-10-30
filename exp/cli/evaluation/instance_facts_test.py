"""``exp eval --continuations`` derives workbook facts and attaches them to a trace package."""

import json
from pathlib import Path

from typer.testing import CliRunner

from exp.cli.app import app
from exp.common.traces.ingest.chat_json import CHAT_JSON_SOURCE
from exp.optimize.evaluation.fidelity.fact_derivation_test import conversation, workbook_file


def run(root: Path, request: Path, *flags: str) -> tuple[int, str]:
    """Run one continuation action and return its exit code and flattened output."""
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "case",
            "--root",
            str(root),
            "--continuations",
            str(request),
            "--non-interactive",
            *flags,
        ],
    )
    return result.exit_code, " ".join(result.output.split())


def test_derive_facts_dry_run_then_writes_package_facts_and_report(tmp_path: Path) -> None:
    """A dry run writes nothing; the run writes the package, facts.json and the report."""
    package = tmp_path / "traces.json"
    package.write_text(json.dumps([conversation("sb-1")]))
    output = tmp_path / "traces.facts.json"
    workbook = workbook_file(tmp_path / "init.xlsx", [["name", 3]])
    request = tmp_path / "derive.json"
    request.write_text(
        json.dumps(
            {
                "action": "derive-facts",
                "facts": {
                    "deriver": {"kind": "xlsx-workbook"},
                    "inputs": [{"trace": "sb-1", "file": str(workbook)}],
                    "traces": str(package),
                    "output": str(output),
                },
            }
        )
    )
    root = tmp_path / "root"
    code, text = run(root, request, "--dry-run")
    assert code == 0, text
    assert "1 of 1 inputs validated for 1 traces, 0 refused" in text
    assert "1 conversations receive instance_facts" in text
    assert "Dry run: nothing written." in text and not output.exists()

    code, text = run(root, request)
    assert code == 0, text
    assert "No provider calls." in text
    squashed = text.replace(" ", "")
    assert f"--traces{output}--sourcechat-json" in squashed
    assert "Report:" in text and "/facts.json" in squashed
    (trace,) = CHAT_JSON_SOURCE.load(output).traces
    facts = trace.instance_facts
    assert facts is not None and isinstance(facts["facts"], list)
    (fact,) = facts["facts"]
    assert isinstance(fact, dict) and fact["key"] == "input.xlsx/workbook"


def test_derive_facts_refusal_is_a_usage_error_naming_the_input(tmp_path: Path) -> None:
    """A refused workbook stops the action with the reason and the way forward."""
    broken = tmp_path / "broken.xlsx"
    broken.write_bytes(b"not a workbook")
    request = tmp_path / "derive.json"
    request.write_text(
        json.dumps(
            {
                "action": "derive-facts",
                "facts": {
                    "deriver": {"kind": "xlsx-workbook"},
                    "inputs": [{"trace": "sb-9", "file": str(broken)}],
                },
            }
        )
    )
    code, text = run(tmp_path / "root", request)
    assert code == 2, text
    squashed = text.replace(" ", "").replace("│", "")
    assert "sb-9input.xlsx" in squashed and "allow_refusals" in squashed
    assert "Traceback" not in text
