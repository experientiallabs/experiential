"""Suite-wide fixtures.

The three autouse fixtures enforce the same rule: a test must behave identically on a machine
that has developer state (a failover chain, a fetched OpenRouter price catalog, a platform
login) and one that does not, and no test may reach the network. `interactive_stdin` is opt-in
and does the opposite job: it supplies the interactive session a prompt test needs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wmo.providers import openrouter_pricing
from wmo.runtime.platform.credentials import ENV_API_URL, ENV_HOME, ENV_ORG, ENV_TOKEN, ENV_WEB_URL


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


@pytest.fixture(autouse=True)
def _no_platform_login(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the user-global platform credential at an empty directory.

    A developer machine is logged in, and `wmo optimize model` now reports its progress to the
    platform whenever a credential resolves. Without this, that developer's suite would push test
    telemetry to a real organization while CI's would not, which is both a network call and the
    kind of machine-dependent behavior the fixtures above exist to remove. Emission tests inject
    their own emitter; credential tests set `WMO_HOME` themselves and a later `monkeypatch` wins.
    """
    monkeypatch.setenv(ENV_HOME, str(tmp_path / "no-wmo-home"))
    # Every credential variable, imported rather than spelled out: `is_complete()`
    # needs only ENV_API_URL and ENV_TOKEN, and the earlier literal list omitted
    # ENV_API_URL, so an exported api url alone still resolved a live credential.
    for name in (ENV_API_URL, ENV_TOKEN, ENV_ORG, ENV_WEB_URL):
        monkeypatch.delenv(name, raising=False)


def _refuse_network() -> openrouter_pricing.PriceCatalog:
    """Stand-in for the live catalog fetch: tests never reach openrouter.ai."""
    raise RuntimeError("the test suite does not fetch the OpenRouter catalog")


@pytest.fixture
def interactive_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Present a terminal stdin, for a test that has to reach an interactive prompt.

    `wmo.cli.consent.can_prompt` requires a TTY on BOTH streams, so forcing a terminal `Console`
    (stdout) is not enough on its own: a spend gate refuses before any prompt is offered. This
    is the input-side counterpart of rich's `force_terminal`, and it patches the seam rather
    than `sys.stdin` because pytest's capture stub and `click.testing.CliRunner` both install a
    non-terminal `sys.stdin` of their own. Tests that need to control what the prompt READS
    replace `sys.stdin` directly instead.
    """
    from wmo.cli import consent

    monkeypatch.setattr(consent, "_stdin_is_terminal", lambda: True)
