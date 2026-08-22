"""Tests for the pinned LiteLLM comparison helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.runtime.gateway.latency_litellm import (
    LITELLM_PIN,
    write_litellm_config,
)
from exp.runtime.gateway.latency_report import comparison_order


def test_write_litellm_config_pins_openai_compatible_mock(tmp_path: Path) -> None:
    """The generated YAML uses LiteLLM's openai/ custom-endpoint form."""
    path = tmp_path / "litellm.yaml"
    write_litellm_config(
        path,
        api_base="http://127.0.0.1:9/v1",
        api_key="mock-key",
        master_key="sk-test",
    )
    text = path.read_text(encoding="utf-8")
    assert "model_name: latency" in text
    assert "model: openai/latency-mock" in text
    assert "api_base: http://127.0.0.1:9/v1" in text
    assert "api_key: mock-key" in text
    assert "master_key: sk-test" in text
    assert "drop_params: true" in text
    assert LITELLM_PIN == "1.97.0"


def test_comparison_order_rotates_gateways() -> None:
    """Odd runs measure Experiential first; even runs flip LiteLLM first."""
    assert comparison_order(1, compare_litellm=False) == ("experiential",)
    assert comparison_order(1, compare_litellm=True) == ("experiential", "litellm")
    assert comparison_order(2, compare_litellm=True) == ("litellm", "experiential")
    assert comparison_order(3, compare_litellm=True) == ("experiential", "litellm")


def test_resolve_litellm_executable_explains_the_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing CLI names the exact PyPI pin the operator should install."""
    from exp.runtime.gateway import latency_litellm

    monkeypatch.setattr(latency_litellm.sys, "executable", str(tmp_path / "python"))
    monkeypatch.setattr(latency_litellm.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match=r"litellm\[proxy\]==1.97.0"):
        latency_litellm.resolve_litellm_executable()
