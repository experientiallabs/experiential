"""Original-evidence retrieval remains usable without a research-memory source."""

from pathlib import Path

import pytest

from exp.optimize.evaluation import fidelity
from exp.optimize.evaluation.fidelity import (
    FidelityRetrieval,
    FidelitySpec,
    execute_fidelity_evaluation,
    prepare_fidelity_evaluation,
)
from exp.optimize.evaluation.tests.fidelity_memory_test import (
    RoutedTransport,
    judged,
    memory_project,
    predicted,
)
from exp.simulation.retrieval.jev import JEVPolicy
from exp.simulation.world_model.experience_memory import memory_head


def test_retrieval_without_memory_does_not_create_research_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An originals-only fidelity run predicts and judges without an unbound selector."""
    project, base = memory_project(tmp_path)
    spec = FidelitySpec.model_validate(
        {
            **base.model_dump(mode="json"),
            "output_protocol": "json-tool",
            "retrieval": FidelityRetrieval(
                jev_alias="jev", jev=JEVPolicy(candidate_limit=4), memory_top_k=4
            ).model_dump(mode="json"),
        }
    )
    wire = RoutedTransport(
        [predicted("call-0", "reply-0"), predicted("call-1", "reply-1"), judged(spec)]
    )
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    result = execute_fidelity_evaluation(
        project, prepare_fidelity_evaluation(project, spec, code_revision="fixture")
    )
    assert result.mechanical["valid_steps"] == 2
    assert result.judge_status == "completed"
    assert wire.decisions == []
    assert memory_head(project.artifacts, "fixture-memory") is None
