"""Router candidate selection and persistence tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from wmo.common.models import (
    BillingSource,
    ConnectionConfig,
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
    ModelRoles,
    ProviderConnection,
    ProviderSetup,
    catalog_state_sha256,
    configure_provider_catalog_with_router_candidates,
    load_model_catalog,
    write_model_catalog,
)
from wmo.common.models.router_candidates import (
    RouterCandidateSelection,
    RouterCandidateSetupError,
    completion_candidate_aliases,
    configure_router_candidates,
    router_candidate_capabilities_sha256,
    router_candidate_prices,
)
from wmo.common.models.setup import ProviderModelSelection


def _capabilities(**updates: object) -> ModelCapabilities:
    """Return complete candidate metadata with requested overrides.

    Args:
        **updates: Capability fields replacing the eligible fixture values.

    Returns:
        Candidate capability metadata.
    """
    return ModelCapabilities(
        supports_completions=True,
        context_window_tokens=32_000,
        maximum_output_tokens=4_000,
        input_cost_per_million_tokens_usd=1.0,
        output_cost_per_million_tokens_usd=2.0,
        cached_input_cost_per_million_tokens_usd=0.25,
        cache_write_cost_per_million_tokens_usd=1.25,
    ).model_copy(update=updates)


def _catalog() -> ModelCatalog:
    """Return two eligible candidates and one unrelated project model."""
    return ModelCatalog(
        connections={"provider": ConnectionConfig(provider="openai", api_key_env="OPENAI_API_KEY")},
        models={
            "candidate-a": ModelRecord(
                billing_source=BillingSource.CUSTOMER_MANAGED,
                connection="provider",
                model="a",
                capabilities=_capabilities(),
            ),
            "candidate-b": ModelRecord(
                billing_source=BillingSource.CUSTOMER_MANAGED,
                connection="provider",
                model="b",
                capabilities=_capabilities(),
            ),
            "world": ModelRecord(
                billing_source=BillingSource.CUSTOMER_MANAGED,
                connection="provider",
                model="world",
                capabilities=ModelCapabilities(),
            ),
        },
        roles=ModelRoles(world_model="world"),
    )


def test_candidate_selection_is_atomic_and_preserves_unrelated_roles(tmp_path: Path) -> None:
    """A confirmed selection changes only candidate roles and emits exact prices.

    Args:
        tmp_path: Temporary root containing the shared catalog.
    """
    path = tmp_path / "models.toml"
    write_model_catalog(path, _catalog())
    starting = catalog_state_sha256(path)

    configured = configure_router_candidates(
        path,
        RouterCandidateSelection(
            candidates=("candidate-b", "candidate-a"), incumbent="candidate-a"
        ),
        expected_state_sha256=starting,
    )

    assert configured.roles.world_model == "world"
    assert configured.roles.candidates == ("candidate-b", "candidate-a")
    assert configured.roles.incumbent == "candidate-a"
    assert tuple(price.candidate_alias for price in router_candidate_prices(configured)) == (
        "candidate-b",
        "candidate-a",
    )


def test_incomplete_candidates_list_all_problems_without_writing(tmp_path: Path) -> None:
    """Unknown capability and price gaps fail together before catalog mutation.

    Args:
        tmp_path: Temporary root containing incomplete candidate records.
    """
    path = tmp_path / "models.toml"
    catalog = _catalog()
    catalog = catalog.model_copy(
        update={
            "models": {
                **catalog.models,
                "candidate-a": catalog.models["candidate-a"].model_copy(
                    update={"capabilities": _capabilities(output_cost_per_million_tokens_usd=None)}
                ),
                "candidate-b": catalog.models["candidate-b"].model_copy(
                    update={"capabilities": ModelCapabilities()}
                ),
            }
        }
    )
    write_model_catalog(path, catalog)
    before = path.read_bytes()

    with pytest.raises(RouterCandidateSetupError) as error:
        configure_router_candidates(
            path,
            RouterCandidateSelection(
                candidates=("candidate-a", "candidate-b", "missing"),
                incumbent="candidate-a",
            ),
        )

    message = str(error.value)
    assert "candidate-a" in message and "output" in message
    assert "candidate-b" in message and "supports_completions=true" in message
    assert "missing" in message and "not configured" in message
    assert path.read_bytes() == before


def test_candidate_confirmation_rejects_concurrent_catalog_change(tmp_path: Path) -> None:
    """A stale interactive summary cannot overwrite a newer catalog.

    Args:
        tmp_path: Temporary root whose catalog changes after collection.
    """
    path = tmp_path / "models.toml"
    write_model_catalog(path, _catalog())
    stale = catalog_state_sha256(path)
    changed = _catalog().model_copy(
        update={"roles": ModelRoles(world_model="world", judge="candidate-a")}
    )
    write_model_catalog(path, changed)
    before = path.read_bytes()

    with pytest.raises(RouterCandidateSetupError, match="changed while candidates"):
        configure_router_candidates(
            path,
            RouterCandidateSelection(
                candidates=("candidate-a", "candidate-b"), incumbent="candidate-a"
            ),
            expected_state_sha256=stale,
        )

    assert path.read_bytes() == before


def test_provider_and_router_candidate_setup_is_atomic_on_validation_failure(
    tmp_path: Path,
) -> None:
    """Invalid candidate roles cannot leave newly discovered provider records persisted.

    Args:
        tmp_path: Temporary root containing the shared catalog.
    """
    catalog = _catalog().model_copy(
        update={
            "models": {
                **_catalog().models,
                "judge": ModelRecord(
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    connection="provider",
                    model="judge",
                ),
                "embedder": ModelRecord(
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    connection="provider",
                    model="embedder",
                    capabilities=ModelCapabilities(
                        supports_embeddings=True,
                        input_cost_per_million_tokens_usd=0.1,
                    ),
                ),
            },
            "roles": ModelRoles(world_model="world", judge="judge", embedder="embedder"),
        }
    )
    path = tmp_path / "models.toml"
    write_model_catalog(path, catalog)
    connection = ProviderConnection(
        name="discovered",
        provider="openai",
        api_key_env="DISCOVERED_API_KEY",
    )
    setup = ProviderSetup(
        connections=(connection,),
        models=(
            ProviderModelSelection(
                alias="candidate-new",
                connection=connection.name,
                model="candidate-new",
            ),
        ),
        known_existing_connections=("provider",),
        known_existing_aliases=tuple(catalog.models),
        world_model="world",
        judge="judge",
        embedder="embedder",
    )
    before = path.read_bytes()

    with pytest.raises(RouterCandidateSetupError, match="supports_completions=true"):
        configure_provider_catalog_with_router_candidates(
            path,
            setup,
            RouterCandidateSelection(
                candidates=("candidate-a", "candidate-new"), incumbent="candidate-a"
            ),
        )

    assert path.read_bytes() == before
    assert "discovered" not in load_model_catalog(path).connections


def test_candidate_definition_cannot_retarget_an_existing_alias(tmp_path: Path) -> None:
    """Candidate setup preserves every existing alias identity under the catalog lock.

    Args:
        tmp_path: Temporary root containing an existing build-visible alias.
    """
    path = tmp_path / "models.toml"
    write_model_catalog(path, _catalog())
    before = path.read_bytes()
    replacement = ProviderModelSelection(
        alias="candidate-a",
        connection="provider",
        model="attacker-model",
        capabilities=ModelCapabilities(
            supports_completions=True,
            context_window_tokens=32_000,
            maximum_output_tokens=4_000,
            input_cost_per_million_tokens_usd=1,
            output_cost_per_million_tokens_usd=2,
            cached_input_cost_per_million_tokens_usd=0.25,
            cache_write_cost_per_million_tokens_usd=1.25,
        ),
    )

    with pytest.raises(RouterCandidateSetupError, match="use a new alias"):
        configure_router_candidates(
            path,
            RouterCandidateSelection(
                candidates=("candidate-a", "candidate-b"), incumbent="candidate-a"
            ),
            candidate_models=(replacement,),
        )

    assert path.read_bytes() == before


def test_completion_candidate_listing_requires_every_declared_price() -> None:
    """Interactive choices exclude aliases with unknown completion economics."""
    catalog = _catalog()
    catalog = catalog.model_copy(
        update={
            "models": {
                **catalog.models,
                "candidate-b": catalog.models["candidate-b"].model_copy(
                    update={
                        "capabilities": _capabilities(cached_input_cost_per_million_tokens_usd=None)
                    }
                ),
            }
        }
    )

    assert completion_candidate_aliases(catalog) == ("candidate-a",)


def test_one_candidate_is_rejected_before_catalog_mutation(tmp_path: Path) -> None:
    """Router optimization requires a meaningful choice between two distinct candidates.

    Args:
        tmp_path: Temporary root containing an unchanged eligible catalog.
    """
    path = tmp_path / "models.toml"
    write_model_catalog(path, _catalog())
    before = path.read_bytes()

    with pytest.raises(ValueError, match="at least 2 items"):
        RouterCandidateSelection(candidates=("candidate-a",), incumbent="candidate-a")

    assert path.read_bytes() == before


def test_routing_capability_binding_detects_completion_drift_separately() -> None:
    """Completion drift changes the frozen router contract, not provider model identity."""
    supported = _capabilities(supports_completions=True)
    unsupported = _capabilities(supports_completions=False)

    assert supported.identity_sha256() == unsupported.identity_sha256()
    assert router_candidate_capabilities_sha256(supported) != (
        router_candidate_capabilities_sha256(unsupported)
    )


def test_unknown_tool_support_changes_the_frozen_candidate_capability_digest() -> None:
    """Unknown tool support hashes as its own tri-state value for frozen plans.

    Runtime dispatch treats unknown support permissively and an explicit denial as a hard
    block, so a drift between them must invalidate a frozen candidate digest.
    """
    unknown = _capabilities(supports_tools=None)
    denied = _capabilities(supports_tools=False)
    granted = _capabilities(supports_tools=True)

    assert router_candidate_capabilities_sha256(unknown) != (
        router_candidate_capabilities_sha256(denied)
    )
    assert router_candidate_capabilities_sha256(granted) != (
        router_candidate_capabilities_sha256(unknown)
    )
