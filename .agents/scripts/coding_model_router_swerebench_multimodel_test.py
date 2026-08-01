"""Tests for the frozen SWE-rebench multi-model binding."""

from __future__ import annotations

import coding_model_router_swerebench_execute as runner
import coding_model_router_swerebench_multimodel as multimodel
import pytest


def test_configure_binds_terra_without_smoke_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_validator = runner.REMOTE_VALIDATOR
    original_phase = runner.DEVELOPMENT_PHASE
    original_model = runner.MODEL
    original_prior_spend = runner.DEFAULT_PRIOR_SPEND_USD
    original_reused = runner.REUSED_TASKS
    original_archives = runner.SMOKE_ARCHIVE_SHA256
    try:
        multimodel.configure("gpt-5.6-terra")
        assert runner.MODEL == "gpt-5.6-terra"
        assert runner.DEFAULT_PRIOR_SPEND_USD == multimodel.FROZEN_PRIOR_SPEND_USD
        assert runner.DEVELOPMENT_PHASE.protocol == (
            "coding-router-swerebench-terra-development-v1"
        )
        assert runner.DEVELOPMENT_PHASE.reuse_smoke is False
        assert runner.REUSED_TASKS == set()
        assert '"gpt-5.6-terra"' in runner.REMOTE_VALIDATOR
        assert '"gpt-5.6-luna"' not in runner.REMOTE_VALIDATOR
    finally:
        monkeypatch.setattr(runner, "REMOTE_VALIDATOR", original_validator)
        monkeypatch.setattr(runner, "DEVELOPMENT_PHASE", original_phase)
        monkeypatch.setattr(runner, "MODEL", original_model)
        monkeypatch.setattr(runner, "DEFAULT_PRIOR_SPEND_USD", original_prior_spend)
        monkeypatch.setattr(runner, "REUSED_TASKS", original_reused)
        monkeypatch.setattr(runner, "SMOKE_ARCHIVE_SHA256", original_archives)


def test_model_prices_cover_every_supported_model() -> None:
    assert set(multimodel.MODELS) <= set(runner.MODEL_PRICES_PER_MTOK)
