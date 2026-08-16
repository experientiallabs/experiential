"""Known, missing, stale, and override tests for judge-calibration pricing."""

from __future__ import annotations

import pytest

from wmo.common.models import (
    ConnectionConfig,
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
    ModelRoles,
    ModelSnapshot,
    PricingSource,
    known_model_metadata,
)
from wmo.optimize.router.judging.contracts import ManualJudgeError
from wmo.optimize.router.judging.pricing import resolve_manual_judge_prices
from wmo.runtime.models.registry import RuntimeModelCatalog

_KNOWN_MODEL = known_model_metadata("openai", "gpt-5.6-luna")
assert _KNOWN_MODEL is not None
_KNOWN_INPUT = _KNOWN_MODEL.input_cost_per_million_tokens_usd
_KNOWN_OUTPUT = _KNOWN_MODEL.output_cost_per_million_tokens_usd
assert _KNOWN_INPUT is not None
assert _KNOWN_OUTPUT is not None


def _catalog(
    *,
    model: str = "custom-judge",
    capabilities: ModelCapabilities | None = None,
) -> ModelCatalog:
    """Return one secret-free catalog with a single judge alias.

    Args:
        model: Provider model ID assigned to the judge alias.
        capabilities: Optional persisted capability and price snapshot.

    Returns:
        Validated local catalog.
    """
    return ModelCatalog(
        connections={
            "openai-main": ConnectionConfig(provider="openai", api_key_env="OPENAI_API_KEY")
        },
        models={
            "judge-main": ModelRecord(
                connection="openai-main",
                model=model,
                capabilities=capabilities,
            )
        },
        roles=ModelRoles(judge="judge-main"),
    )


def _snapshot(catalog: ModelCatalog) -> ModelSnapshot:
    """Return the credential-free snapshot for the fixture judge alias."""
    snapshot, _capabilities = RuntimeModelCatalog(catalog).snapshot("judge-main")
    return snapshot


def test_persisted_catalog_prices_are_configured_when_the_model_is_unknown() -> None:
    """A hand-declared catalog price is trusted when WMO has no known metadata."""
    catalog = _catalog(
        capabilities=ModelCapabilities(
            input_cost_per_million_tokens_usd=1.25,
            output_cost_per_million_tokens_usd=10.0,
        )
    )

    input_price, output_price, source = resolve_manual_judge_prices(
        catalog,
        judge_alias="judge-main",
        expected_model=_snapshot(catalog),
    )

    assert input_price == 1.25
    assert output_price == 10.0
    assert source is PricingSource.CONFIGURED


def test_known_model_metadata_fills_a_catalog_that_omitted_prices() -> None:
    """WMO known prices are used before any credential or provider call."""
    catalog = _catalog(model="gpt-5.6-luna", capabilities=ModelCapabilities())

    input_price, output_price, source = resolve_manual_judge_prices(
        catalog,
        judge_alias="judge-main",
        expected_model=_snapshot(catalog),
    )

    assert input_price == _KNOWN_INPUT
    assert output_price == _KNOWN_OUTPUT
    assert source is PricingSource.WMO_CATALOG


def test_matching_persisted_and_known_prices_keep_wmo_catalog_provenance() -> None:
    """Persisted prices that still match known metadata keep that source."""
    catalog = _catalog(
        model="gpt-5.6-luna",
        capabilities=ModelCapabilities(
            input_cost_per_million_tokens_usd=_KNOWN_INPUT,
            output_cost_per_million_tokens_usd=_KNOWN_OUTPUT,
        ),
    )

    _input_price, _output_price, source = resolve_manual_judge_prices(
        catalog,
        judge_alias="judge-main",
        expected_model=_snapshot(catalog),
    )

    assert source is PricingSource.WMO_CATALOG


def test_missing_pricing_fails_with_an_actionable_repair() -> None:
    """Unknown models without persisted prices fail closed instead of assuming zero."""
    catalog = _catalog(capabilities=ModelCapabilities())

    with pytest.raises(ManualJudgeError, match="no trustworthy input/output prices") as excinfo:
        resolve_manual_judge_prices(
            catalog,
            judge_alias="judge-main",
            expected_model=_snapshot(catalog),
        )

    message = str(excinfo.value)
    assert "wmo config providers" in message
    assert "--input-usd-per-million" in message
    assert "--output-usd-per-million" in message


def test_partial_catalog_prices_are_not_mixed_with_known_metadata() -> None:
    """One persisted side cannot be completed from a neighboring known price."""
    catalog = _catalog(
        model="gpt-5.6-luna",
        capabilities=ModelCapabilities(input_cost_per_million_tokens_usd=1.0),
    )

    with pytest.raises(ManualJudgeError, match="no trustworthy input/output prices"):
        resolve_manual_judge_prices(
            catalog,
            judge_alias="judge-main",
            expected_model=_snapshot(catalog),
        )


def test_stale_catalog_prices_fail_when_known_metadata_disagrees() -> None:
    """Persisted prices that drifted from WMO known metadata are refused."""
    catalog = _catalog(
        model="gpt-5.6-luna",
        capabilities=ModelCapabilities(
            input_cost_per_million_tokens_usd=9.0,
            output_cost_per_million_tokens_usd=99.0,
        ),
    )

    with pytest.raises(ManualJudgeError, match="stale relative to WMO known metadata"):
        resolve_manual_judge_prices(
            catalog,
            judge_alias="judge-main",
            expected_model=_snapshot(catalog),
        )


def test_stale_judge_identity_fails_before_price_lookup() -> None:
    """A replaced catalog model cannot inherit the sealed setup snapshot."""
    catalog = _catalog(
        capabilities=ModelCapabilities(
            input_cost_per_million_tokens_usd=1.0,
            output_cost_per_million_tokens_usd=2.0,
        )
    )
    drifted = _catalog(
        model="other-judge",
        capabilities=ModelCapabilities(
            input_cost_per_million_tokens_usd=1.0,
            output_cost_per_million_tokens_usd=2.0,
        ),
    )

    with pytest.raises(ManualJudgeError, match="judge identity changed after setup"):
        resolve_manual_judge_prices(
            drifted,
            judge_alias="judge-main",
            expected_model=_snapshot(catalog),
        )


def test_advanced_overrides_win_and_record_configured_provenance() -> None:
    """Both advanced flags replace catalog lookup and keep configured provenance."""
    catalog = _catalog(capabilities=ModelCapabilities())

    input_price, output_price, source = resolve_manual_judge_prices(
        catalog,
        judge_alias="judge-main",
        expected_model=_snapshot(catalog),
        input_usd_per_million_tokens=0.0,
        output_usd_per_million_tokens=0.0,
    )

    assert input_price == 0.0
    assert output_price == 0.0
    assert source is PricingSource.CONFIGURED


def test_one_sided_override_is_refused() -> None:
    """A single advanced price flag cannot fabricate the missing side."""
    catalog = _catalog(
        capabilities=ModelCapabilities(
            input_cost_per_million_tokens_usd=1.0,
            output_cost_per_million_tokens_usd=2.0,
        )
    )

    with pytest.raises(ManualJudgeError, match="must be supplied together"):
        resolve_manual_judge_prices(
            catalog,
            judge_alias="judge-main",
            expected_model=_snapshot(catalog),
            input_usd_per_million_tokens=3.0,
        )
