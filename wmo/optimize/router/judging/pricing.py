"""Catalog-backed price resolution for manual judge calibration."""

from __future__ import annotations

from wmo.common.models import (
    ModelCatalog,
    ModelSnapshot,
    PricingSource,
    known_model_metadata,
)
from wmo.optimize.router.judging.contracts import ManualJudgeError
from wmo.runtime.models.registry import RuntimeModelCatalog

_OVERRIDE_TOGETHER = (
    "advanced judge pricing overrides must be supplied together as "
    "--input-usd-per-million and --output-usd-per-million"
)
_MISSING_PRICES = (
    "judge alias {alias!r} has no trustworthy input/output prices; record them with "
    "wmo config providers or pass both advanced --input-usd-per-million and "
    "--output-usd-per-million"
)
_STALE_PRICES = (
    "catalog prices for judge alias {alias!r} are stale relative to WMO known metadata "
    "for {provider}/{model}; re-run wmo config providers or pass both advanced "
    "--input-usd-per-million and --output-usd-per-million"
)
_STALE_IDENTITY = (
    "configured judge identity changed after setup; re-run wmo config judge setup after "
    "restoring the catalog model"
)


def resolve_manual_judge_prices(
    catalog: ModelCatalog,
    *,
    judge_alias: str,
    expected_model: ModelSnapshot,
    input_usd_per_million_tokens: float | None = None,
    output_usd_per_million_tokens: float | None = None,
) -> tuple[float, float, PricingSource]:
    """Resolve judge input and output prices without credentials or provider calls.

    Explicit advanced overrides win. Otherwise persisted catalog prices are used when both
    sides are present and either match WMO known metadata or have no known counterpart.
    Known-model metadata fills a catalog that omitted both prices. Partial catalog prices
    are never mixed with known-model prices.

    Args:
        catalog: Local secret-free model catalog.
        judge_alias: Configured judge alias frozen by setup.
        expected_model: Exact judge snapshot sealed by setup.
        input_usd_per_million_tokens: Optional advanced input-price override.
        output_usd_per_million_tokens: Optional advanced output-price override.

    Returns:
        Input price, output price, and the exact pricing-source provenance.

    Raises:
        ManualJudgeError: Identity drifted, overrides are incomplete, prices are missing,
            or persisted catalog prices disagree with current known metadata.
    """
    if (input_usd_per_million_tokens is None) != (output_usd_per_million_tokens is None):
        raise ManualJudgeError(_OVERRIDE_TOGETHER)
    snapshot, capabilities = RuntimeModelCatalog(catalog).snapshot(judge_alias)
    if snapshot != expected_model:
        raise ManualJudgeError(_STALE_IDENTITY)
    if input_usd_per_million_tokens is not None and output_usd_per_million_tokens is not None:
        return (
            input_usd_per_million_tokens,
            output_usd_per_million_tokens,
            PricingSource.CONFIGURED,
        )
    catalog_input = capabilities.input_cost_per_million_tokens_usd
    catalog_output = capabilities.output_cost_per_million_tokens_usd
    known = known_model_metadata(snapshot.provider, snapshot.model_id)
    known_input = None if known is None else known.input_cost_per_million_tokens_usd
    known_output = None if known is None else known.output_cost_per_million_tokens_usd
    catalog_complete = catalog_input is not None and catalog_output is not None
    catalog_partial = (catalog_input is None) != (catalog_output is None)
    known_complete = known_input is not None and known_output is not None
    if catalog_partial:
        raise ManualJudgeError(_MISSING_PRICES.format(alias=judge_alias))
    if (
        catalog_complete
        and known_complete
        and (catalog_input != known_input or catalog_output != known_output)
    ):
        raise ManualJudgeError(
            _STALE_PRICES.format(
                alias=judge_alias,
                provider=snapshot.provider,
                model=snapshot.model_id,
            )
        )
    if catalog_complete:
        assert catalog_input is not None and catalog_output is not None
        source = PricingSource.WMO_CATALOG if known_complete else PricingSource.CONFIGURED
        return catalog_input, catalog_output, source
    if known_complete:
        assert known_input is not None and known_output is not None
        return known_input, known_output, PricingSource.WMO_CATALOG
    raise ManualJudgeError(_MISSING_PRICES.format(alias=judge_alias))
