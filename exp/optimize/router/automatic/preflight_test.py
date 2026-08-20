"""Aggregate automatic router preflight tests."""

from __future__ import annotations

from pathlib import Path

import pytest

import exp.runtime.models.registry as model_registry
from exp.common.models import (
    BillingSource,
    ConnectionConfig,
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
    ModelSnapshot,
    RouterCandidateSelection,
    write_model_catalog,
)
from exp.common.project import ProjectConfig, ProjectStore
from exp.common.routing import router_embedding_reservation
from exp.optimize.router.automatic.preflight import (
    AutomaticRouterOptions,
    AutomaticRouterPreflightError,
    preflight_automatic_router,
)
from exp.optimize.router.automatic.reservations import remaining_simulation_budget
from exp.runtime.models import RuntimeModelCatalog


def test_preflight_aggregates_missing_inputs_before_credentials_or_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One read-only failure lists build, review, and role prerequisites.

    Args:
        tmp_path: Temporary initialized project root.
        monkeypatch: Credential boundary trap.
    """
    root = tmp_path / ".exp"
    project = ProjectStore(root, "support")
    project.initialize(ProjectConfig(project_id="support"))
    write_model_catalog(project.model_catalog_path, _catalog())
    before_project = project.paths.project_toml.read_bytes()
    before_catalog = project.model_catalog_path.read_bytes()
    before_artifacts = project.artifacts.list_ids()

    def reject_credentials(*args: object, **kwargs: object) -> str:
        """Fail if aggregate local preflight crosses the credential boundary."""
        del args, kwargs
        raise AssertionError("credential access is forbidden during aggregate preflight")

    monkeypatch.setattr(model_registry, "read_connection_api_key", reject_credentials)

    with pytest.raises(AutomaticRouterPreflightError) as error:
        preflight_automatic_router(
            project,
            RouterCandidateSelection(
                candidates=("candidate-a", "candidate-b"), incumbent="candidate-a"
            ),
            options=AutomaticRouterOptions(
                maximum_router_feature_tokens=2_048,
                maximum_retrieval_query_tokens=8_192,
                router_embedding_maximum_attempts=1,
                completion_maximum_attempts=1,
            ),
        )

    message = str(error.value)
    assert "completed build" in message
    assert "frozen model roles" in message
    assert "manual judge" in message
    assert "fidelity" not in message
    assert project.paths.project_toml.read_bytes() == before_project
    assert project.model_catalog_path.read_bytes() == before_catalog
    assert project.artifacts.list_ids() == before_artifacts
    assert project.read_review() is None


def test_router_embedding_reservation_is_admitted_before_other_provider_work() -> None:
    """A router embedding reservation that consumes the ceiling blocks later dispatch."""
    reservation = router_embedding_reservation(
        model=_catalog_model_snapshot(),
        input_usd_per_million_tokens=1_000_000,
        maximum_attempts_per_feature=1,
        maximum_input_tokens_per_feature=2,
        feature_count=1,
    )
    problems: list[str] = []

    remaining = remaining_simulation_budget(
        problems,
        maximum_provider_cost_usd=1,
        router_reservation=reservation,
    )

    assert remaining == 0
    assert problems == [
        "the router embedding reservation consumes the entire provider spend ceiling; "
        "increase --maximum-simulation-cost-usd or lower a request/retry ceiling"
    ]


def test_shared_remainder_excludes_only_the_router_embedding_reservation() -> None:
    """Judge planning estimates take no upfront carve-out from the shared spend pool."""
    reservation = router_embedding_reservation(
        model=_catalog_model_snapshot(),
        input_usd_per_million_tokens=1,
        maximum_attempts_per_feature=1,
        maximum_input_tokens_per_feature=1_000,
        feature_count=1,
    )
    problems: list[str] = []

    remaining = remaining_simulation_budget(
        problems,
        maximum_provider_cost_usd=50,
        router_reservation=reservation,
    )

    assert problems == []
    assert remaining == 50 - reservation.estimated_cost_usd


def _catalog() -> ModelCatalog:
    """Return two complete candidate aliases without project build roles."""
    capabilities = ModelCapabilities(
        supports_completions=True,
        context_window_tokens=32_000,
        maximum_output_tokens=2_000,
        input_cost_per_million_tokens_usd=1,
        output_cost_per_million_tokens_usd=2,
        cached_input_cost_per_million_tokens_usd=0.5,
        cache_write_cost_per_million_tokens_usd=1.5,
    )
    return ModelCatalog(
        connections={"openai": ConnectionConfig(provider="openai", api_key_env="OPENAI_API_KEY")},
        models={
            "candidate-a": ModelRecord(
                billing_source=BillingSource.CUSTOMER_MANAGED,
                connection="openai",
                model="candidate-a",
                capabilities=capabilities,
            ),
            "candidate-b": ModelRecord(
                billing_source=BillingSource.CUSTOMER_MANAGED,
                connection="openai",
                model="candidate-b",
                capabilities=capabilities,
            ),
        },
    )


def _catalog_model_snapshot() -> ModelSnapshot:
    """Return one exact model snapshot from the local fixture catalog."""
    return RuntimeModelCatalog(_catalog(), environment={}).snapshot("candidate-a")[0]
