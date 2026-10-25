"""Complete workbook observables need the maintained owner and every declared output pinned."""

from pathlib import Path

import pytest

from exp.common.core.artifacts import sha256_bytes
from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.optimize.evaluation.executable import workbook as workbook_runtime
from exp.optimize.evaluation.executable.qualification import _pins_native_workbook_outputs
from exp.optimize.evaluation.executable.task import ExecutableTask
from exp.optimize.evaluation.executable.workbook import WorkbookRuntimeDefinition, WorkbookStateView
from exp.optimize.evaluation.executable.workbook_test import workbook_definition
from exp.runtime.environments.world_contracts import StateDomain
from exp.simulation.evaluation.tasks import OutcomeRule, TaskVerifier


def bound_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ExecutableTask:
    """Build a typed recognition fixture; this is not a registered feasible execution."""
    _, job, _ = job_fixture(tmp_path / "actor", monkeypatch)
    _, definition, _ = workbook_definition(tmp_path / "native")
    definition = definition.model_copy(
        update={
            "views": {
                "whole": WorkbookStateView(input_file="input.xlsx", output_file="output.xlsx")
            }
        }
    )
    predicate = TaskVerifier(
        description="Complete observable workbook output",
        outcome_schema={"type": "object"},
        allowed_tools=("edit_workbook",),
        rules=(
            OutcomeRule(path="/final_state/views/whole/sha256", operator="equals", value="a" * 64),
        ),
    )
    job = job.model_copy(
        update={
            "runtime_file": Path(workbook_runtime.__file__),
            "runtime_sha256": sha256_bytes(Path(workbook_runtime.__file__).read_bytes()),
            "configuration": definition.model_dump(mode="json"),
            "study": job.study.model_copy(
                update={"verifier": predicate, "probe_sha256": definition.probe_sha256}
            ),
        }
    )
    return ExecutableTask(
        intent="Preserve complete observable output", job=job, output_paths=("output.xlsx",)
    )


def configured(task: ExecutableTask, definition: WorkbookRuntimeDefinition) -> ExecutableTask:
    """Bind any changed private definition to its own correct probe identity."""
    return task.model_copy(
        update={
            "job": task.job.model_copy(
                update={
                    "configuration": definition.model_dump(mode="json"),
                    "study": task.job.study.model_copy(
                        update={"probe_sha256": definition.probe_sha256}
                    ),
                }
            )
        }
    )


def second_output(task: ExecutableTask, *, cover: bool) -> ExecutableTask:
    """Declare another actual native output, optionally with its own complete equality view."""
    definition = WorkbookRuntimeDefinition.model_validate(task.job.configuration)
    manifest = definition.manifest.model_copy(
        update={
            "domains": (
                *definition.manifest.domains,
                StateDomain(name="second.xlsx", owner="native"),
            ),
            "tools": tuple(
                tool.model_copy(update={"writes": ("output.xlsx", "second.xlsx")})
                for tool in definition.manifest.tools
            ),
        }
    )
    views = {
        **definition.views,
        "second": WorkbookStateView(input_file="input.xlsx", output_file="second.xlsx"),
    }
    definition = definition.model_copy(
        update={
            "output_paths": ("output.xlsx", "second.xlsx"),
            "manifest": manifest,
            "views": views,
        }
    )
    task = configured(task, definition).model_copy(
        update={"output_paths": ("output.xlsx", "second.xlsx")}
    )
    if cover:
        predicate = task.job.study.verifier.model_copy(
            update={
                "rules": (
                    *task.job.study.verifier.rules,
                    OutcomeRule(
                        path="/final_state/views/second/sha256", operator="equals", value="b" * 64
                    ),
                )
            }
        )
        task = task.model_copy(
            update={
                "job": task.job.model_copy(
                    update={"study": task.job.study.model_copy(update={"verifier": predicate})}
                )
            }
        )
    return task


def test_native_all_sheet_outputs_are_recognized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every owned output needs an all-sheet, non-fallback complete observable equality."""
    task = bound_task(tmp_path, monkeypatch)
    assert _pins_native_workbook_outputs(task)
    assert _pins_native_workbook_outputs(second_output(task, cover=True))
    assert not _pins_native_workbook_outputs(second_output(task, cover=False))


@pytest.mark.parametrize(
    "kind",
    [
        "partial",
        "fallback",
        "input_only",
        "missing_view",
        "malformed_digest",
        "non_equality",
        "wrong_probe",
        "no_outputs",
        "unknown_task_output",
        "custom_digest",
        "counterfeit_digest",
        "mutable_input",
        "unwritable_output",
        "non_native",
        "malformed_configuration",
    ],
)
def test_incomplete_or_unowned_views_do_not_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Plausible view names and hash strings cannot stand in for verified complete ownership."""
    task = bound_task(tmp_path, monkeypatch)
    definition = WorkbookRuntimeDefinition.model_validate(task.job.configuration)
    view = definition.views["whole"]
    if kind in ("partial", "fallback", "input_only"):
        changes = {
            "partial": {"sheets": ("Data",)},
            "fallback": {"baseline_until_written": True},
            "input_only": {"output_file": None},
        }[kind]
        task = configured(
            task,
            definition.model_copy(update={"views": {"whole": view.model_copy(update=changes)}}),
        )
    elif kind == "missing_view":
        task = configured(task, definition.model_copy(update={"views": {}}))
    elif kind in ("malformed_digest", "non_equality"):
        rule = (
            OutcomeRule(
                path="/final_state/views/whole/sha256", operator="equals", value="not-a-digest"
            )
            if kind == "malformed_digest"
            else OutcomeRule(path="/final_state/views/whole/sha256", operator="exists")
        )
        predicate = task.job.study.verifier.model_copy(update={"rules": (rule,)})
        task = task.model_copy(
            update={
                "job": task.job.model_copy(
                    update={"study": task.job.study.model_copy(update={"verifier": predicate})}
                )
            }
        )
    elif kind == "wrong_probe":
        task = task.model_copy(
            update={
                "job": task.job.model_copy(
                    update={"study": task.job.study.model_copy(update={"probe_sha256": "f" * 64})}
                )
            }
        )
    elif kind == "no_outputs":
        task = task.model_copy(update={"output_paths": ()})
    elif kind == "unknown_task_output":
        task = task.model_copy(update={"output_paths": ("unregistered.xlsx",)})
    elif kind in ("custom_digest", "counterfeit_digest"):
        path = tmp_path / "custom.py"
        path.write_text('def continuation_runtime(*args):\n    raise RuntimeError("custom")\n')
        updates = {"runtime_file": path}
        if kind == "custom_digest":
            updates["runtime_sha256"] = sha256_bytes(path.read_bytes())
        task = task.model_copy(update={"job": task.job.model_copy(update=updates)})
    elif kind == "unwritable_output":
        manifest = definition.manifest.model_copy(
            update={
                "tools": tuple(
                    tool.model_copy(update={"writes": ()}) for tool in definition.manifest.tools
                )
            }
        )
        task = configured(task, definition.model_copy(update={"manifest": manifest}))
    elif kind == "mutable_input":
        manifest = definition.manifest.model_copy(
            update={
                "tools": tuple(
                    x.model_copy(update={"writes": ("input.xlsx", "output.xlsx")})
                    for x in definition.manifest.tools
                )
            }
        )
        task = configured(task, definition.model_copy(update={"manifest": manifest}))
    elif kind == "non_native":
        manifest = definition.manifest.model_copy(
            update={
                "domains": (
                    *definition.manifest.domains,
                    StateDomain(name="simulated", owner="simulated"),
                )
            }
        )
        task = configured(task, definition.model_copy(update={"manifest": manifest}))
    else:
        task = task.model_copy(update={"job": task.job.model_copy(update={"configuration": {}})})
    assert not _pins_native_workbook_outputs(task)


def test_generic_cli_and_foreign_runtime_do_not_import_optional_workbook_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh interpreter without openpyxl handles generic CLI and foreign owners."""
    import json
    import os
    import subprocess
    import sys

    task = bound_task(tmp_path, monkeypatch)
    foreign = tmp_path / "foreign.py"
    foreign.write_text('raise AssertionError("foreign runtime must not be imported")\n')
    cases = {
        "native": task.model_dump(mode="json"),
        "foreign": task.model_copy(
            update={
                "job": task.job.model_copy(
                    update={
                        "runtime_file": foreign,
                        "runtime_sha256": sha256_bytes(foreign.read_bytes()),
                    }
                )
            }
        ).model_dump(mode="json"),
        "spoofed": task.model_copy(
            update={"job": task.job.model_copy(update={"runtime_file": foreign})}
        ).model_dump(mode="json"),
        "missing": task.model_copy(
            update={"job": task.job.model_copy(update={"runtime_file": tmp_path / "absent.py"})}
        ).model_dump(mode="json"),
        "no_outputs": task.model_copy(update={"output_paths": ()}).model_dump(mode="json"),
    }
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases))
    script = """
import importlib.abc,json,sys
class NoWorkbook(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]=='openpyxl':
            raise ModuleNotFoundError('optional workbook dependency unavailable',name='openpyxl')
sys.meta_path.insert(0,NoWorkbook())
from exp.optimize.evaluation.executable.qualification import _pins_native_workbook_outputs
from exp.optimize.evaluation.executable.task import ExecutableTask
from exp.cli.app import app
from typer.testing import CliRunner
result=CliRunner().invoke(app,['optimize','--help'])
assert result.exit_code==0,(result.output,result.exception)
assert 'openpyxl' not in sys.modules
cases=json.load(open(sys.argv[1]))
for name in ('foreign','spoofed','missing','no_outputs'):
    assert not _pins_native_workbook_outputs(ExecutableTask.model_validate(cases[name])),name
assert 'exp.optimize.evaluation.executable.workbook' not in sys.modules
try:
    _pins_native_workbook_outputs(ExecutableTask.model_validate(cases['native']))
except ModuleNotFoundError as error:
    assert error.name=='openpyxl'
else:
    raise AssertionError('selected workbook must still require its optional dependency')
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[5])
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(path)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
