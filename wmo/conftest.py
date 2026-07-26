"""Suite-wide fixtures.

Both fixtures here enforce the same rule: a test must behave identically on a machine that has
developer state (a failover chain, a fetched OpenRouter price catalog) and one that does not,
and no test may reach the network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wmo.providers import openrouter_pricing


@pytest.fixture(autouse=True)
def _no_local_fallback_chain(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the default failover-chain lookup at a nonexistent `.wmo/fallback.toml`.

    The real one is developer-local and gitignored; chain tests pass an explicit `path=`.
    """
    monkeypatch.setattr(
        "wmo.providers.waterfall.FALLBACK_CONFIG_PATH", tmp_path / "no-fallback.toml"
    )


@pytest.fixture(autouse=True)
def _offline_openrouter_catalog(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Cut every test off from the live OpenRouter price catalog and the user's cached copy.

    Pricing tests opt back in by pointing `WMO_OPENROUTER_CATALOG` at their own fixture file or
    by replacing `_fetch_catalog` themselves; a later `monkeypatch.setattr` wins. The
    `_FETCH_ERROR` latch is reset through monkeypatch so the "already failed once" state cannot
    leak from one test into the next.
    """
    monkeypatch.setenv(
        openrouter_pricing.CATALOG_PATH_ENV, str(tmp_path / "no-openrouter-prices.json")
    )
    monkeypatch.setattr(openrouter_pricing, "_FETCH_ERROR", None)
    monkeypatch.setattr(openrouter_pricing, "_fetch_catalog", _refuse_network)


def _refuse_network() -> openrouter_pricing.PriceCatalog:
    """Stand-in for the live catalog fetch: tests never reach openrouter.ai."""
    raise RuntimeError("the test suite does not fetch the OpenRouter catalog")
