"""Trusted runtime files execute only at their reviewed digest and with a typed factory result."""

from pathlib import Path

import pytest

from exp.common.core.artifacts import sha256_bytes
from exp.common.project import ProjectStore
from exp.optimize.evaluation.executable.loader import load_runtime_definition, load_runtime_module


def test_runtime_module_refuses_changed_source_before_executing_it(tmp_path: Path) -> None:
    """A digest mismatch never executes the edited file."""
    runtime = tmp_path / "runtime.py"
    runtime.write_text("value = 1\n")
    digest = sha256_bytes(runtime.read_bytes())
    assert load_runtime_module(runtime, digest).value == 1
    runtime.write_text("raise AssertionError('must not execute')\n")
    with pytest.raises(ValueError, match="runtime source changed"):
        load_runtime_module(runtime, digest)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("value = 1\n", "must export continuation_runtime"),
        (
            "def continuation_runtime(project, configuration):\n    return 1\n",
            "must return ContinuationRuntime",
        ),
    ],
)
def test_runtime_definition_requires_the_typed_factory(
    tmp_path: Path, source: str, message: str
) -> None:
    """A missing factory or an untyped runtime is refused rather than adapted."""
    runtime = tmp_path / "runtime.py"
    runtime.write_text(source)
    with pytest.raises(ValueError, match=message):
        load_runtime_definition(
            ProjectStore(tmp_path, "case"),
            runtime_file=runtime,
            runtime_sha256=sha256_bytes(runtime.read_bytes()),
            configuration={},
            probe_sha256="0" * 64,
        )
