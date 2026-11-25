"""Tests for the plan-connection vocabulary and the Antigravity connection fields."""

from __future__ import annotations

import pytest

from exp.common.models import ConnectionConfig
from exp.common.models.subscription_connection import (
    MAXIMUM_USER_AGENT_CHARACTERS,
    SUBSCRIPTION_PROVIDERS,
    require_plan_connection_shape,
)

_PROJECT = "proud-sprite-a1b2"


def _antigravity(**fields: object) -> ConnectionConfig:
    """Build an Antigravity connection with ``fields`` overriding the defaults."""
    data: dict[str, object] = {
        "provider": "gemini",
        "subscription": "antigravity",
        "code_assist_project": _PROJECT,
        "code_assist_user_agent": "antigravity/hub/2.34.0 (aidev_client)",
    }
    data.update(fields)
    return ConnectionConfig.model_validate(data)


def test_every_plan_kind_names_its_catalog_provider() -> None:
    """Antigravity is a sign-in for the Gemini catalog provider."""
    assert SUBSCRIPTION_PROVIDERS == {
        "chatgpt": "openai",
        "anthropic": "anthropic",
        "antigravity": "gemini",
    }


def test_an_antigravity_connection_carries_its_project_and_user_agent() -> None:
    """The project and the user agent round-trip through the persisted shape."""
    connection = _antigravity()
    assert connection.code_assist_project == _PROJECT
    assert ConnectionConfig.model_validate(connection.model_dump(mode="json")) == connection


def test_an_antigravity_connection_requires_a_valid_project() -> None:
    """The project is required and must be a Cloud project id."""
    with pytest.raises(ValueError, match="requires code_assist_project"):
        _antigravity(code_assist_project=None)
    for bad in ["Has Spaces", "x", "-leading", "a/b"]:
        with pytest.raises(ValueError, match="Google Cloud project id"):
            _antigravity(code_assist_project=bad)
    with pytest.raises(ValueError, match="sign-in for provider 'gemini'"):
        _antigravity(provider="vertex")


def test_the_user_agent_is_one_bounded_printable_line() -> None:
    """A header-injecting or oversized user agent is refused; an absent one is the default."""
    assert _antigravity(code_assist_user_agent=None).code_assist_user_agent is None
    for bad in ["a\r\nX-Injected: 1", " ", "u" * (MAXIMUM_USER_AGENT_CHARACTERS + 1)]:
        with pytest.raises(ValueError, match="code_assist_user_agent"):
            _antigravity(code_assist_user_agent=bad)


def test_other_connections_refuse_the_antigravity_fields() -> None:
    """The project and the translation exist only on an Antigravity plan."""
    with pytest.raises(ValueError, match="only accepted for subscription='antigravity'"):
        ConnectionConfig(
            provider="gemini", api_key_env="GEMINI_API_KEY", code_assist_project=_PROJECT
        )
    with pytest.raises(ValueError, match="only accepted for subscription='antigravity'"):
        ConnectionConfig(provider="openai", subscription="chatgpt", code_assist_user_agent="ua")
    with pytest.raises(ValueError, match="omit api_key_env"):
        _antigravity(api_key_env="GEMINI_API_KEY")


def test_the_project_is_identity_and_the_user_agent_is_not() -> None:
    """A new project is a different plan account; a new client string is the same endpoint."""
    base = _antigravity()
    assert (
        base.identity_sha256() != _antigravity(code_assist_project="other-proj-1").identity_sha256()
    )
    assert base.identity_sha256() == _antigravity(code_assist_user_agent="ua/2").identity_sha256()
    key = ConnectionConfig(provider="gemini", api_key_env="GEMINI_API_KEY")
    dumped = key.model_dump(mode="json")
    assert "code_assist_project" not in dumped and "code_assist_user_agent" not in dumped


def test_the_shape_check_passes_api_key_connections() -> None:
    """An API-key connection with no plan fields needs nothing."""
    require_plan_connection_shape(
        subscription=None, provider="gemini", code_assist_project=None, code_assist_user_agent=None
    )
