"""Tests for the single "artifact dir -> live WorldModel" path shared by the CLI and serving."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from wmo.common.config import HarnessConfig, save_config
from wmo.common.providers import ProviderConfig, ProviderKind
from wmo.common.providers.bedrock import BedrockProvider
from wmo.simulation.model import loader
from wmo.simulation.model.loader import load_world_model
from wmo.simulation.model.world_model import WorldModel

if TYPE_CHECKING:
    from pathlib import Path

    from wmo.common.providers.base import Provider


def _artifact(tmp_path: Path, *, verify: bool = False) -> Path:
    """A built-enough artifact dir: just the config the loader reads."""
    root = tmp_path / ".wmo"
    save_config(
        HarnessConfig(
            providers=[
                ProviderConfig(
                    kind=ProviderKind.BEDROCK,
                    model="us.anthropic.claude-opus-4-8",
                    region="us-east-1",
                )
            ],
            serve_provider=ProviderKind.BEDROCK,
            verify=verify,
        ),
        root=root,
    )
    return root


def test_load_returns_the_model_on_the_provider_its_config_names(tmp_path: Path) -> None:
    root = _artifact(tmp_path)

    world_model, provider = load_world_model(root)

    assert isinstance(world_model, WorldModel)
    assert isinstance(provider, BedrockProvider)
    # The provider is returned so callers that also need it (`wmo demo` runs its agent on the same
    # one) never reconstruct it: it must be the very instance the model serves on.
    assert world_model._provider is provider


def test_a_str_path_loads_the_same_as_a_path(tmp_path: Path) -> None:
    root = _artifact(tmp_path)

    assert isinstance(load_world_model(str(root))[0], WorldModel)


def test_build_flags_in_the_artifact_config_reach_the_loaded_model(tmp_path: Path) -> None:
    # A plain load must honour what the build persisted (here `verify`), or serving would quietly
    # run a different model than the one that was measured.
    root = _artifact(tmp_path, verify=True)

    world_model, _provider = load_world_model(root)

    assert world_model._verify is True


def test_load_forwards_telemetry_root_and_max_fidelity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def _fake_load(
        artifact_dir: str,
        provider: Provider,
        *,
        telemetry_root: str | Path | None = None,
        max_fidelity: bool = False,
    ) -> str:
        captured.update(
            artifact_dir=artifact_dir, telemetry_root=telemetry_root, max_fidelity=max_fidelity
        )
        return "loaded"

    monkeypatch.setattr(loader.WorldModel, "load", _fake_load)
    root = _artifact(tmp_path)
    telemetry = tmp_path / "telemetry"

    load_world_model(root, telemetry_root=telemetry, max_fidelity=True)

    assert captured == {
        "artifact_dir": str(root),
        "telemetry_root": telemetry,
        "max_fidelity": True,
    }


def test_loading_an_unbuilt_directory_says_to_build_first(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="wmo build"):
        load_world_model(tmp_path / "never-built")
