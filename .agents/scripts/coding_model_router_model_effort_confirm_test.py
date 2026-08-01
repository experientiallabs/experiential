"""Tests for selected model-effort confirmation execution."""

from __future__ import annotations

import coding_model_router_model_effort_confirm as confirm
import coding_model_router_swerebench_execute as runner
import pytest


def test_configure_binds_exact_single_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    originals = {
        "model": runner.MODEL,
        "efforts": runner.EFFORTS,
        "phase": runner.DEVELOPMENT_PHASE,
        "validator": runner.REMOTE_VALIDATOR,
        "reused": runner.REUSED_TASKS,
        "archives": runner.SMOKE_ARCHIVE_SHA256,
        "spend": runner.DEFAULT_PRIOR_SPEND_USD,
    }
    try:
        _, arm = confirm.configure("gpt-5.6-sol", "xhigh", 2_000.0)
        assert arm == "sol-xhigh"
        assert runner.MODEL == "gpt-5.6-sol"
        assert runner.EFFORTS == ("xhigh",)
        assert runner.DEVELOPMENT_PHASE.corpus_sha256 == confirm.CONFIRMATION_CORPUS_SHA256
        assert runner.DEVELOPMENT_PHASE.reuse_smoke is False
        assert runner.DEFAULT_PRIOR_SPEND_USD == 2_000.0
    finally:
        monkeypatch.setattr(runner, "MODEL", originals["model"])
        monkeypatch.setattr(runner, "EFFORTS", originals["efforts"])
        monkeypatch.setattr(runner, "DEVELOPMENT_PHASE", originals["phase"])
        monkeypatch.setattr(runner, "REMOTE_VALIDATOR", originals["validator"])
        monkeypatch.setattr(runner, "REUSED_TASKS", originals["reused"])
        monkeypatch.setattr(runner, "SMOKE_ARCHIVE_SHA256", originals["archives"])
        monkeypatch.setattr(runner, "DEFAULT_PRIOR_SPEND_USD", originals["spend"])
