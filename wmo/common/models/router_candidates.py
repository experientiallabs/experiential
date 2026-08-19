"""Validated router candidate selection and conflict-safe catalog persistence."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator, model_validator

from wmo.common.core.artifacts import ContractModel, Sha256, sha256_json
from wmo.common.core.locks import file_write_lock
from wmo.common.models.catalog import (
    ModelCatalog,
    load_model_catalog,
    write_model_catalog,
)
from wmo.common.models.model import ModelCapabilities, ReasoningEffort
from wmo.common.models.pricing import CandidateTokenPrice
from wmo.common.models.setup import (
    ProviderModelSelection,
    ProviderSetup,
    _merge_provider_setup,
    catalog_state_sha256,
)


class RouterCandidateSetupError(ValueError):
    """Router candidates are incomplete, ambiguous, or changed during confirmation."""


def router_candidate_capabilities_sha256(capabilities: ModelCapabilities) -> Sha256:
    """Hash the exact non-price candidate execution contract.

    Unknown tool support hashes exactly like an explicit ``False`` so catalogs written before
    tool support became declarable keep the digest frozen into existing plans and policies.

    Args:
        capabilities: Explicit current candidate capability declaration.

    Returns:
        Versioned digest frozen into evaluation plans and runtime policies.
    """
    return sha256_json(
        {
            "version": "router-candidate-capabilities-v1",
            "supports_completions": capabilities.supports_completions,
            "supports_tools": bool(capabilities.supports_tools),
            "supports_structured_output": capabilities.supports_structured_output,
            "context_window_tokens": capabilities.context_window_tokens,
            "maximum_output_tokens": capabilities.maximum_output_tokens,
        }
    )


class RouterCandidateSelection(ContractModel):
    """Explicit ordered router candidates and the quality incumbent among them.

    Each candidate alias may carry its own reasoning-effort choice for candidate-role calls.
    An absent entry means that candidate keeps its catalog capability pin unchanged.
    """

    candidates: tuple[str, ...] = Field(min_length=2)
    incumbent: str = Field(min_length=1, max_length=128)
    candidate_reasoning_efforts: dict[str, ReasoningEffort] = Field(default_factory=dict)

    @field_validator("candidates")
    @classmethod
    def _require_unique_candidates(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject ambiguous repeated aliases while preserving operator order.

        Args:
            value: Candidate aliases in the requested router order.

        Returns:
            The unchanged unique aliases.

        Raises:
            ValueError: One alias appears more than once.
        """
        if len(set(value)) != len(value):
            raise ValueError("router candidate aliases must not repeat")
        return value

    @model_validator(mode="after")
    def _require_incumbent_candidate(self) -> RouterCandidateSelection:
        """Require the incumbent to be one of the explicitly evaluated candidates.

        Returns:
            The validated selection.

        Raises:
            ValueError: The incumbent is absent from the candidate set.
        """
        if self.incumbent not in self.candidates:
            raise ValueError("router incumbent must also be a selected candidate")
        unknown = sorted(set(self.candidate_reasoning_efforts).difference(self.candidates))
        if unknown:
            raise ValueError(
                "candidate_reasoning_efforts name unselected candidates: " + ", ".join(unknown)
            )
        return self


def completion_candidate_aliases(catalog: ModelCatalog) -> tuple[str, ...]:
    """Return aliases with explicit completion support and complete token pricing.

    Args:
        catalog: Local secret-free model catalog.

    Returns:
        Eligible aliases sorted for stable interactive presentation.
    """
    return tuple(
        alias
        for alias, record in sorted(catalog.models.items())
        if record.capabilities is not None
        and record.capabilities.supports_completions is True
        and record.capabilities.context_window_tokens is not None
        and record.capabilities.maximum_output_tokens is not None
        and not _missing_candidate_prices(record.capabilities)
    )


def validate_router_candidate_selection(
    catalog: ModelCatalog,
    selection: RouterCandidateSelection,
) -> tuple[str, ...]:
    """List every selected-alias capability or pricing problem without provider access.

    Args:
        catalog: Local catalog containing candidate metadata.
        selection: Explicit candidate aliases and incumbent.

    Returns:
        Deterministically ordered actionable problems. An empty tuple means the selection is
        ready for immutable evaluation planning.
    """
    problems: list[str] = []
    for alias in selection.candidates:
        record = catalog.models.get(alias)
        if record is None:
            problems.append(f"candidate alias {alias!r} is not configured")
            continue
        capabilities = record.capabilities
        if capabilities is None or capabilities.supports_completions is not True:
            problems.append(f"candidate alias {alias!r} must declare supports_completions=true")
            continue
        missing_capacities = tuple(
            name
            for name, value in (
                ("context_window_tokens", capabilities.context_window_tokens),
                ("maximum_output_tokens", capabilities.maximum_output_tokens),
            )
            if value is None
        )
        if missing_capacities:
            problems.append(
                f"candidate alias {alias!r} is missing explicit " + ", ".join(missing_capacities)
            )
        missing = _missing_candidate_prices(capabilities)
        if missing:
            problems.append(
                f"candidate alias {alias!r} is missing explicit " + ", ".join(missing) + " pricing"
            )
    return tuple(problems)


def configure_router_candidates(
    path: Path,
    selection: RouterCandidateSelection,
    *,
    candidate_models: tuple[ProviderModelSelection, ...] = (),
    expected_state_sha256: str | None = None,
) -> ModelCatalog:
    """Validate and atomically persist only router candidate role assignments.

    Args:
        path: Shared ``models.toml`` path.
        selection: Confirmed ordered candidate aliases and incumbent.
        candidate_models: Confirmed new or replacement candidate metadata.
        expected_state_sha256: Catalog digest shown during interactive confirmation.

    Returns:
        Complete catalog after the candidate roles are committed.

    Raises:
        RouterCandidateSetupError: The catalog changed or selected aliases are incomplete.
        ModelCatalogError: The catalog is absent or malformed.
    """
    with file_write_lock(path, what="router candidate configuration"):
        current_state = catalog_state_sha256(path)
        if expected_state_sha256 is not None and current_state != expected_state_sha256:
            raise RouterCandidateSetupError(
                "models.toml changed while candidates were being confirmed; review and retry"
            )
        catalog = load_model_catalog(path)
        models = dict(catalog.models)
        for candidate in candidate_models:
            if candidate.connection not in catalog.connections:
                raise RouterCandidateSetupError(
                    f"candidate alias {candidate.alias!r} names unknown connection "
                    f"{candidate.connection!r}"
                )
            candidate_record = candidate.catalog_record()
            existing = models.get(candidate.alias)
            if existing is not None and existing != candidate_record:
                raise RouterCandidateSetupError(
                    f"candidate alias {candidate.alias!r} already names different model metadata; "
                    "use a new alias"
                )
            models[candidate.alias] = candidate_record
        catalog = catalog.model_copy(update={"models": models})
        configured = _apply_router_candidate_selection(catalog, selection)
        write_model_catalog(path, configured)
        return configured


def configure_provider_catalog_with_router_candidates(
    path: Path,
    setup: ProviderSetup,
    selection: RouterCandidateSelection,
    *,
    expected_state_sha256: str | None = None,
) -> ModelCatalog:
    """Atomically persist provider records and the selected router roles together.

    Args:
        path: Local ``.wmo/models.toml`` path.
        setup: Newly discovered provider records plus the existing build-role aliases.
        selection: Confirmed ordered router candidates and incumbent.
        expected_state_sha256: Exact catalog state observed during collection.

    Returns:
        Complete catalog after provider records and router roles are committed.

    Raises:
        RouterCandidateSetupError: The catalog changed, provider setup conflicts, or the
            selected aliases are incomplete.
        ModelCatalogError: Existing catalog content is absent or malformed.
    """
    with file_write_lock(path, what="provider model and router candidate configuration"):
        current_state = catalog_state_sha256(path)
        if expected_state_sha256 is not None and current_state != expected_state_sha256:
            raise RouterCandidateSetupError(
                "models.toml changed while candidates were being confirmed; review and retry"
            )
        existing = load_model_catalog(path) if path.exists() else None
        try:
            catalog = _merge_provider_setup(existing, setup, replace=False)
        except ValueError as exc:
            raise RouterCandidateSetupError(str(exc)) from exc
        configured = _apply_router_candidate_selection(catalog, selection)
        write_model_catalog(path, configured)
        return configured


def _apply_router_candidate_selection(
    catalog: ModelCatalog,
    selection: RouterCandidateSelection,
) -> ModelCatalog:
    """Validate and apply router candidate roles to an in-memory catalog.

    Args:
        catalog: Catalog containing all selected provider and model records.
        selection: Confirmed ordered router candidates and incumbent.

    Returns:
        Catalog with the selected router roles assigned.

    Raises:
        RouterCandidateSetupError: The selection is incomplete for the catalog.
    """
    problems = validate_router_candidate_selection(catalog, selection)
    if problems:
        raise RouterCandidateSetupError(
            "router candidate setup is incomplete:\n- " + "\n- ".join(problems)
        )
    retained = {
        alias: effort
        for alias, effort in catalog.roles.candidate_reasoning_efforts.items()
        if alias in selection.candidates
    }
    retained.update(selection.candidate_reasoning_efforts)
    roles = catalog.roles.model_copy(
        update={
            "candidates": selection.candidates,
            "incumbent": selection.incumbent,
            "candidate_reasoning_efforts": retained,
        }
    )
    return catalog.model_copy(update={"roles": roles})


def verify_router_candidate_catalog_state(path: Path, expected_state_sha256: str) -> None:
    """Verify a confirmed candidate plan against the locked current catalog state.

    Args:
        path: Shared ``models.toml`` path confirmed by the operator.
        expected_state_sha256: Exact catalog digest captured during candidate collection.

    Raises:
        RouterCandidateSetupError: The catalog changed after confirmation.
    """
    with file_write_lock(path, what="router candidate configuration"):
        if catalog_state_sha256(path) != expected_state_sha256:
            raise RouterCandidateSetupError(
                "models.toml changed while candidates were being confirmed; review and retry"
            )


def router_candidate_prices(catalog: ModelCatalog) -> tuple[CandidateTokenPrice, ...]:
    """Build exact candidate price rows from validated persisted router roles.

    Args:
        catalog: Catalog with selected candidate and incumbent roles.

    Returns:
        Candidate prices in persisted router order.

    Raises:
        RouterCandidateSetupError: Candidate roles are absent or their metadata is incomplete.
    """
    if not catalog.roles.candidates or catalog.roles.incumbent is None:
        raise RouterCandidateSetupError("router candidates and incumbent are not configured")
    selection = RouterCandidateSelection(
        candidates=catalog.roles.candidates,
        incumbent=catalog.roles.incumbent,
    )
    problems = validate_router_candidate_selection(catalog, selection)
    if problems:
        raise RouterCandidateSetupError(
            "router candidate setup is incomplete:\n- " + "\n- ".join(problems)
        )
    prices = []
    for alias in selection.candidates:
        capabilities = catalog.models[alias].capabilities
        if capabilities is None:
            raise AssertionError("validated candidates have capability metadata")
        input_price = capabilities.input_cost_per_million_tokens_usd
        output_price = capabilities.output_cost_per_million_tokens_usd
        cached_input_price = capabilities.cached_input_cost_per_million_tokens_usd
        cache_write_price = capabilities.cache_write_cost_per_million_tokens_usd
        if (
            input_price is None
            or output_price is None
            or cached_input_price is None
            or cache_write_price is None
        ):
            raise AssertionError("validated candidates have complete price metadata")
        prices.append(
            CandidateTokenPrice(
                candidate_alias=alias,
                input_usd_per_million_tokens=input_price,
                output_usd_per_million_tokens=output_price,
                cached_input_usd_per_million_tokens=cached_input_price,
                cache_write_usd_per_million_tokens=cache_write_price,
            )
        )
    return tuple(prices)


def _missing_candidate_prices(capabilities: ModelCapabilities) -> tuple[str, ...]:
    """Return required candidate price names whose explicit values are absent.

    Args:
        capabilities: Candidate capability record inspected without provider access.

    Returns:
        Human-readable missing price names in billing order.
    """
    return tuple(
        name
        for name, value in (
            ("input", capabilities.input_cost_per_million_tokens_usd),
            ("output", capabilities.output_cost_per_million_tokens_usd),
            ("cached input", capabilities.cached_input_cost_per_million_tokens_usd),
            ("cache write", capabilities.cache_write_cost_per_million_tokens_usd),
        )
        if value is None
    )
