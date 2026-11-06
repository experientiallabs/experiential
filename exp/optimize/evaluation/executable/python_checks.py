"""Native function checks keep expected answers outside the candidate's process."""

from __future__ import annotations

import json
import math
import shlex

from pydantic import Field, field_validator

from exp.common.core.artifacts import (
    ContractModel,
    JsonValue,
    strict_equal,
    validate_artifact_file_path,
)
from exp.common.models import ToolCall
from exp.runtime.environments.docker_workspace import DockerWorkspaceFactory, workspace_projection
from exp.runtime.environments.world_contracts import NativeImage
from exp.runtime.environments.worlds import WorldContext

_WORKER = """
import importlib.util
import json
import sys
from contextlib import redirect_stdout
data = json.load(sys.stdin)
spec = importlib.util.spec_from_file_location("candidate_module", data["module_file"])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
with redirect_stdout(sys.stderr):
    spec.loader.exec_module(module)
function = getattr(module, data["function"])
results = []
for arguments in data["arguments"]:
    try:
        with redirect_stdout(sys.stderr):
            results.append(function(*arguments))
    except BaseException as error:
        results.append({"exception": type(error).__name__})
sys.stdout.write(json.dumps(results, separators=(",", ":"), allow_nan=False))
"""


class PythonFunctionCase(ContractModel):
    """One private input and its evaluator-owned expected value.

    Attributes:
        arguments: Positional arguments sent to the isolated candidate.
        expected: Expected value retained exclusively by the evaluator.
    """

    arguments: tuple[JsonValue, ...]
    expected: JsonValue


class PythonFunctionChecks(ContractModel):
    """A bounded, frozen native acceptance predicate for one Python function.

    Attributes:
        module_file: Canonical relative Python source file in the workspace.
        function: Public callable explicitly named by the task.
        cases: Private native inputs and expected outputs.
        absolute_tolerance: Numeric comparison tolerance; zero means exact equality.
        relative_tolerance: Numeric relative tolerance; zero means exact equality.
    """

    module_file: str
    function: str
    cases: tuple[PythonFunctionCase, ...] = Field(min_length=1, max_length=2048)
    absolute_tolerance: float = Field(default=0, ge=0, allow_inf_nan=False)
    relative_tolerance: float = Field(default=0, ge=0, allow_inf_nan=False)

    @field_validator("module_file")
    @classmethod
    def module_path(cls, value: str) -> str:
        """Refuse source paths outside the selected workspace."""
        if validate_artifact_file_path(value).as_posix() != value or not value.endswith(".py"):
            raise ValueError("function checks require a canonical relative Python file")
        return value

    @field_validator("function")
    @classmethod
    def callable_name(cls, value: str) -> str:
        """Require one directly named public callable."""
        if not value.isidentifier() or value.startswith("_"):
            raise ValueError("function checks require a public Python identifier")
        return value


def _matches(actual: JsonValue, expected: JsonValue, checks: PythonFunctionChecks) -> bool:
    """Compare typed values recursively, applying only explicitly selected numeric tolerances."""
    if (
        isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and isinstance(expected, (int, float))
        and not isinstance(expected, bool)
    ):
        if checks.absolute_tolerance or checks.relative_tolerance:
            return math.isclose(
                actual,
                expected,
                abs_tol=checks.absolute_tolerance,
                rel_tol=checks.relative_tolerance,
            )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _matches(a, e, checks) for a, e in zip(actual, expected, strict=True)
        )
    return strict_equal(actual, expected)


class PythonFunctionResult(ContractModel):
    """Actual outputs checked by the evaluator, with incomplete runs never passing.

    Attributes:
        passed: Every required response arrived and matched its expected value.
        complete: Exactly one well-formed output arrived for every required input.
        matched: Number of matching responses.
        total: Number of required inputs.
        exit_code: Actual native worker exit code.
        implementation_unchanged: Candidate execution preserved its source bytes.
        mismatched_indices: First mismatching input indices for private diagnosis.
    """

    passed: bool
    complete: bool
    matched: int
    total: int
    exit_code: int
    implementation_unchanged: bool
    mismatched_indices: tuple[int, ...]


def check_python_function(
    factory: DockerWorkspaceFactory, image: NativeImage, checks: PythonFunctionChecks
) -> PythonFunctionResult:
    """Execute inputs in a disposable clone; compare all outputs outside the candidate.

    Expected values are never copied into the container. An early successful process exit,
    forged pytest output, malformed output or a truncated result remains incomplete.
    """
    before = workspace_projection(image)["workspace"]
    if not isinstance(before, dict) or not isinstance(before.get("files"), dict):
        raise ValueError("native workspace has no file projection")
    original = before["files"].get(checks.module_file)
    clone = factory.restore(image)
    context = WorldContext.__new__(WorldContext)
    try:
        inputs = {
            "module_file": checks.module_file,
            "function": checks.function,
            "arguments": [list(case.arguments) for case in checks.cases],
        }
        written = clone.execute(
            ToolCall(
                call_id="native-inputs",
                name="Write",
                arguments={"path": ".native_checks/inputs.json", "content": json.dumps(inputs)},
            ),
            context,
        )
        if written.is_error:
            raise ValueError("native check input setup failed")
        observation = clone.execute(
            ToolCall(
                call_id="native-function",
                name="Bash",
                arguments={
                    "command": "python -c "
                    + shlex.quote(_WORKER)
                    + " < .native_checks/inputs.json",
                    "timeout": 60000,
                },
            ),
            context,
        )
        execution = json.loads(observation.content)
        try:
            outputs: JsonValue = json.loads(execution["stdout"])
        except (ValueError, TypeError):
            outputs = None
        complete = (
            execution["exit_code"] == 0
            and not execution["output_truncated"]
            and isinstance(outputs, list)
            and len(outputs) == len(checks.cases)
        )
        mismatches = []
        matched = 0
        if complete and isinstance(outputs, list):
            for index, (actual, case) in enumerate(zip(outputs, checks.cases, strict=True)):
                if _matches(actual, case.expected, checks):
                    matched += 1
                else:
                    mismatches.append(index)
        after = clone.state_projection()["workspace"]
        unchanged = (
            original is not None
            and isinstance(after, dict)
            and isinstance(after.get("files"), dict)
            and after["files"].get(checks.module_file) == original
        )
        return PythonFunctionResult(
            passed=complete and matched == len(checks.cases) and unchanged,
            complete=complete,
            matched=matched,
            total=len(checks.cases),
            exit_code=execution["exit_code"],
            implementation_unchanged=unchanged,
            mismatched_indices=tuple(mismatches[:10]),
        )
    finally:
        clone.close()
