"""Native checks reject metric-only shortcuts and preserve private expected answers."""

import os
import shlex

import pytest

from exp.common.tasks import TaskCase
from exp.optimize.evaluation.executable.python_checks import (
    PythonFunctionCase,
    PythonFunctionChecks,
    check_python_function,
)
from exp.runtime.environments.docker_workspace import DockerWorkspaceConfig, DockerWorkspaceFactory


def test_function_checks_require_public_names_and_workspace_paths() -> None:
    """Unbounded or outside-workspace function definitions fail before execution."""
    case = PythonFunctionCase(arguments=(1,), expected=2)
    with pytest.raises(ValueError):
        PythonFunctionChecks(module_file="../outside.py", function="solve", cases=(case,))
    with pytest.raises(ValueError):
        PythonFunctionChecks(module_file="solution.py", function="__getattr__", cases=(case,))


@pytest.mark.parametrize(
    "source,passed,complete",
    [
        ("def solve(x):\n    return x + 1\n", True, True),
        ("def solve(x):\n    return 0\n", False, True),
        ("import os\nos._exit(0)\n", False, False),
        (
            "import os,sys\nsys.stdout.write('2000 passed')\nsys.stdout.flush()\nos._exit(0)\n",
            False,
            False,
        ),
    ],
)
def test_native_function_checks_reject_incorrect_and_forged_success(
    source: str,
    passed: bool,
    complete: bool,
) -> None:
    """Actual values determine reward; exit zero and a success message are insufficient."""
    image = os.environ.get("EXP_TEST_WORKSPACE_IMAGE")
    if image is None:
        pytest.skip("requires an explicitly selected local Docker workspace image")
    factory = DockerWorkspaceFactory(
        DockerWorkspaceConfig(
            image=image,
            docker=tuple(shlex.split(os.environ.get("EXP_TEST_DOCKER", "docker"))),
        ),
        files={"solution.py": source},
    )
    task = TaskCase(
        task_id="function-check-control",
        lineage_group_id="function-check-control",
        partition="fit",
        instruction="Implement solve.",
        workload_weight=1,
        origin="authored",
    )
    world = factory.create(task)
    try:
        initial = world.capture()
        checks = PythonFunctionChecks(
            module_file="solution.py",
            function="solve",
            cases=(
                PythonFunctionCase(arguments=(1,), expected=2),
                PythonFunctionCase(arguments=(5,), expected=6),
            ),
        )
        result = check_python_function(factory, initial, checks)
        assert result.passed is passed and result.complete is complete
        assert world.capture() == initial
    finally:
        world.close()
