"""Trusted runtime files execute only at their reviewed digest and with a typed factory result."""

from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject, sha256_bytes
from exp.common.project import ProjectStore
from exp.optimize.evaluation.executable.loader import (
    ContinuationRuntime,
    load_runtime_definition,
    load_runtime_module,
)
from exp.runtime.environments.tests.world_conformance_test import workbook_case
from exp.simulation.continuations import WorldCheckpoint


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


def test_a_refused_runtime_is_closed_before_the_refusal(tmp_path: Path) -> None:
    """A probe mismatch after the factory ran still releases what the factory opened."""
    runtime = tmp_path / "runtime.py"
    released = tmp_path / "released"
    runtime.write_text(
        "from pathlib import Path\n"
        "from exp.optimize.evaluation.executable.loader import ContinuationRuntime\n"
        "def continuation_runtime(project, configuration):\n"
        "    marker = Path(configuration['marker'])\n"
        "    return ContinuationRuntime(None, None, '1' * 64, marker.touch)\n"
    )
    with pytest.raises(ValueError, match="probe differs"):
        load_runtime_definition(
            ProjectStore(tmp_path, "case"),
            runtime_file=runtime,
            runtime_sha256=sha256_bytes(runtime.read_bytes()),
            configuration={"marker": str(released)},
            probe_sha256="0" * 64,
        )
    assert released.exists()


def test_a_runtime_closes_on_interrupt_and_owns_nothing_by_default(tmp_path: Path) -> None:
    """The context manager releases on KeyboardInterrupt; a plain runtime closes as a no-op."""
    released: list[str] = []

    def probe(checkpoint: WorldCheckpoint) -> JsonObject:
        """Project nothing."""
        del checkpoint
        return {}

    world = workbook_case().runtime
    with (
        pytest.raises(KeyboardInterrupt),
        ContinuationRuntime(world, probe, "0" * 64, lambda: released.append("closed")),
    ):
        raise KeyboardInterrupt
    assert released == ["closed"]
    with ContinuationRuntime(world, probe, "0" * 64) as plain:
        assert plain.world is world
    plain.close()
