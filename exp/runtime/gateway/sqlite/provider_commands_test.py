"""Tests for translating storage-neutral provider commands into SQLite connections."""

from __future__ import annotations

from pathlib import Path

from exp.common.models import ConnectionConfig
from exp.runtime.gateway.platform import UpsertProviderConnectionCommand
from exp.runtime.gateway.sqlite.platform_test import _platform
from exp.runtime.gateway.sqlite.provider_commands import sqlite_connection_config


def test_a_plan_command_translates_to_a_plan_connection() -> None:
    """The plan kind survives translation, with no secret reference to carry."""
    command = UpsertProviderConnectionCommand(
        organization_id="org-one",
        connection_id="claude-plan",
        revision_id="claude-plan-revision",
        provider="anthropic",
        subscription="anthropic",
    )

    assert sqlite_connection_config(command) == ConnectionConfig(
        provider="anthropic", subscription="anthropic"
    )


def test_a_plan_command_round_trips_through_the_platform_mutation(tmp_path: Path) -> None:
    """``mutate_provider_connection`` authors a plan connection a later read returns intact."""
    platform = _platform(tmp_path)
    command = UpsertProviderConnectionCommand(
        organization_id="org-one",
        connection_id="chatgpt-plan",
        revision_id="chatgpt-plan-revision",
        provider="openai",
        subscription="chatgpt",
    )

    assert platform.mutate_provider_connection(command).changed
    (revision,) = platform.provider_connection_revisions(organization_id="org-one")
    assert revision.subscription == "chatgpt"
    assert not platform.mutate_provider_connection(command).changed
