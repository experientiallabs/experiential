"""Account-driven synchronization for hosted provider connections and model identities."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from exp.common.models import (
    BillingSource,
    ModelRecord,
    ProviderConnection,
    derive_model_alias,
    load_model_catalog,
    resolve_discovered_model,
    sync_provider_models,
)
from exp.runtime.models.providers import (
    HttpProviderModelLister,
    ProviderEndpoint,
    ProviderListingError,
    ProviderModelLister,
)


def sync_account_models(
    root: Path,
    *,
    connection: ProviderConnection,
    api_key: str,
    console: Console,
    lister: ProviderModelLister | None = None,
) -> tuple[str, ...]:
    """Register a hosted provider and every model visible to its authenticated account.

    Args:
        root: Local ``.exp`` root receiving the secret-free model catalog.
        connection: Named hosted connection to register.
        api_key: Credential used only for the bounded model-listing request.
        console: Terminal receiving a secret-free synchronization summary.
        lister: Optional model-listing seam used by deterministic tests.

    Returns:
        Stable local aliases for every synchronized model identity.

    Raises:
        ProviderListingError: The account model listing fails or is empty.
    """
    model_lister = lister or HttpProviderModelLister()
    discovered = model_lister.list_models(
        ProviderEndpoint(
            provider=connection.provider,
            api_key=api_key,
            base_url=connection.base_url,
        )
    )
    if not discovered:
        raise ProviderListingError("Experiential Cloud published no models for this account")
    catalog_path = root / "models.toml"
    existing_models = {}
    if catalog_path.is_file():
        existing = load_model_catalog(catalog_path)
        existing_models = {
            record.model: alias
            for alias, record in existing.models.items()
            if record.connection == connection.name
        }
        taken_aliases = set(existing.models)
    else:
        taken_aliases = set()

    records: dict[str, ModelRecord] = {}
    aliases: list[str] = []
    seen_models: set[str] = set()
    for item in discovered:
        if item.model in seen_models:
            continue
        seen_models.add(item.model)
        alias = existing_models.get(item.model)
        if alias is None:
            alias = derive_model_alias(item.provider, item.model, frozenset(taken_aliases))
        taken_aliases.add(alias)
        aliases.append(alias)
        resolved = resolve_discovered_model(item)
        records[alias] = ModelRecord(
            connection=connection.name,
            model=item.model,
            billing_source=BillingSource.HOST_MANAGED,
            capabilities=resolved.capabilities,
        )

    sync_provider_models(
        catalog_path,
        connection=connection,
        models=records,
    )
    console.print(f"[green]Synced Experiential Cloud: {len(aliases)} models.[/green]")
    return tuple(aliases)
