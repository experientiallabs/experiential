"""Suite-wide fixtures.

The default failover chain lives in a developer-local, gitignored `.wmh/fallback.toml`; tests must
behave identically whether or not the developer running them has one, so the default lookup path is
pointed at a nonexistent file for every test. Chain tests pass an explicit `path=`.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _no_local_fallback_chain(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "wmh.providers.waterfall.FALLBACK_CONFIG_PATH", tmp_path / "no-fallback.toml"
    )
