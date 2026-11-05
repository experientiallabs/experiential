"""Actual native prelude inventory guards, without provider dispatch."""

from pathlib import Path

import pytest

from exp.optimize.evaluation.variance.council_prelude import qualify_council_prelude
from exp.optimize.evaluation.variance.council_prelude_hook import CouncilPreludeCheckpoint
from exp.optimize.evaluation.variance.council_prelude_inventory import prelude_inventory
from exp.optimize.evaluation.variance.council_prelude_test import _prelude_fixture


def test_native_inventory_requires_exact_job_real_group_and_frozen_banks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native quote consumers refuse substituted jobs, absent group owners and changed banks."""
    project, prepared, resume, contract, wire = _prelude_fixture(tmp_path, monkeypatch)
    qualified = qualify_council_prelude(
        project, prepared, resume, contract, code_revision="council-executor"
    )
    job = qualified.jobs[0]
    calls = len(wire.requests)
    checkpoint = CouncilPreludeCheckpoint(
        phase="traceml_group", actor_alias=job.actor_alias, job=job
    )
    quote = prelude_inventory(project, qualified, checkpoint)
    assert quote.attempts > 0 and quote.remaining.attempts > 0
    with pytest.raises(ValueError, match="exact frozen"):
        prelude_inventory(
            project, qualified, checkpoint.model_copy(update={"job": qualified.jobs[1]})
        )
    with pytest.raises(ValueError):
        prelude_inventory(
            project, qualified, checkpoint.model_copy(update={"phase": "traceml_labels"})
        )
    with pytest.raises(ValueError, match="real configured judge"):
        prelude_inventory(
            project, qualified, checkpoint.model_copy(update={"phase": "behavior_reference"})
        )
    with pytest.raises(ValueError, match="sources changed"):
        prelude_inventory(
            project,
            qualified.model_copy(update={"contract": contract.model_copy(update={"banks": {}})}),
            checkpoint,
        )
    assert len(wire.requests) == calls
