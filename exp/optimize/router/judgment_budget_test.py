"""Durable judgment-budget ledger tests beside its optimizer owner."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from exp.common.judging import Judgment
from exp.common.project import ProjectConfig, ProjectStore
from exp.optimize.router.composition import (
    RouterCompositionBudget,
    RouterCompositionError,
    RouterWorkflowServices,
    compose_router,
)
from exp.optimize.router.composition_test import (
    _COMPACT_MINING_SPEC,
    _TIME,
    _bind_completed_build,
    _Catalog,
    _compact_normalized_traces,
    _Judge,
    _ReviewSupplier,
    _SetupSupplier,
    _SimulatorFactory,
    _snapshot,
)
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.router.runtime_test import _Client


def test_judgment_budget_resumes_interrupted_dispatch_without_widening_budget(
    tmp_path: Path,
) -> None:
    """Resume an interrupted paid dispatch under its reserved slot and stay budget-bound.

    Args:
        tmp_path: Isolated project root for durable judgment reservations.
    """
    project = ProjectStore(tmp_path, "project-a")
    project.initialize(ProjectConfig(project_id="project-a"))
    normalized = _compact_normalized_traces()
    _bind_completed_build(
        project,
        normalized,
        revision="test-revision",
        mining_spec=_COMPACT_MINING_SPEC,
    )
    judge = _Judge()
    judge.fail_on_call = 3
    runtime_client = _Client()
    services = RouterWorkflowServices(
        review_supplier=_ReviewSupplier(),
        setup_supplier=_SetupSupplier(),
        simulator_factory=_SimulatorFactory(),
        judge=judge,
        runtime_catalog=cast(
            RuntimeModelCatalog,
            _Catalog(
                {
                    "candidate-a": _snapshot("candidate-a"),
                    "embedder": _snapshot("embedder"),
                },
                runtime_client,
            ),
        ),
    )
    budget = RouterCompositionBudget(
        maximum_simulation_cost_usd=10.0,
        maximum_judgments=3,
    )

    with pytest.raises(RuntimeError, match="judgment dispatch interruption"):
        compose_router(
            project,
            normalized,
            services=services,
            budget=budget,
            created_at=_TIME,
            code_revision="test-revision",
        )
    assert judge.calls == 3
    assert (
        sum(
            project.artifacts.read(artifact_id).manifest.artifact_type == "judgment-dispatch"
            for artifact_id in project.artifacts.list_ids()
        )
        == budget.maximum_judgments
    )

    with pytest.raises(RouterCompositionError, match="judgment dispatch budget exhausted"):
        compose_router(
            project,
            normalized,
            services=services,
            budget=budget,
            created_at=_TIME,
            code_revision="test-revision",
        )
    assert judge.calls == 4
    assert (
        sum(
            project.artifacts.read(artifact_id).manifest.artifact_type == "judgment-dispatch"
            for artifact_id in project.artifacts.list_ids()
        )
        == budget.maximum_judgments
    )

    first_judgment_id = next(
        artifact_id
        for artifact_id in project.artifacts.list_ids()
        if project.artifacts.read(artifact_id).manifest.artifact_type == "judgment"
    )
    first_judgment = Judgment.model_validate_json(
        project.artifacts.read_bytes(first_judgment_id, "judgment.json")
    )
    forged = first_judgment.model_copy(
        update={
            "judgment_id": "judgment-forged-cross-plan",
            "judge_prompt_sha256": "f" * 64,
        }
    )
    project.artifacts.write_json(
        artifact_id=forged.judgment_id,
        artifact_type="judgment",
        envelope=forged,
        files={"judgment.json": forged},
    )
    with pytest.raises(RouterCompositionError, match="exact plan review pins"):
        compose_router(
            project,
            normalized,
            services=services,
            budget=budget,
            created_at=_TIME,
            code_revision="test-revision",
        )
    assert judge.calls == 4
