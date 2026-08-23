"""Tests for optional file-backed guardrail configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.runtime.gateway.guardrails.config import engine_from_document, load_guardrail_engine


def test_missing_file_leaves_the_gateway_unguarded(tmp_path: Path) -> None:
    """Default-off: no configuration file means no engine and no classifiers."""
    assert load_guardrail_engine(tmp_path) is None


def test_document_registers_keyword_adapters_and_identity_policies() -> None:
    """A valid document binds one identity to a local keyword adapter."""
    engine = engine_from_document(
        {
            "adapters": [
                {
                    "adapter_id": "keyword-safety",
                    "kind": "keyword",
                    "needles": ["blocked"],
                }
            ],
            "policies": [
                {
                    "policy_id": "strict-member",
                    "organization_id": "organization-one",
                    "identity_id": "identity-one",
                    "protected": True,
                    "checks": [
                        {
                            "check_id": "input-safety",
                            "capability": "content_safety",
                            "stage": "input",
                            "action": "block",
                            "timeout_ms": 250,
                            "adapter_id": "keyword-safety",
                        }
                    ],
                }
            ],
        }
    )

    assert engine.policy_for("organization-one", "identity-one") is not None
    assert engine.policy_for("organization-one", "identity-two") is None
    assert engine.policy_for("organization-two", "identity-one") is None


def test_unknown_adapter_kind_is_rejected() -> None:
    """Only the local keyword adapter is built in; other kinds are injected in code."""
    with pytest.raises(ValueError, match="keyword adapter kind"):
        engine_from_document(
            {
                "adapters": [{"adapter_id": "hosted", "kind": "hosted", "needles": ["x"]}],
                "policies": [],
            }
        )


def test_invalid_json_file_fails_closed(tmp_path: Path) -> None:
    """A present but unreadable file is a configuration error."""
    path = tmp_path / "gateway"
    path.mkdir()
    (path / "guardrails.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="valid JSON"):
        load_guardrail_engine(tmp_path)
