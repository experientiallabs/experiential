"""Command-level tests for role-free gateway provider connection management."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from click import unstyle
from typer.testing import CliRunner, Result

from exp.cli.app import app
from exp.common.auth import ProviderAuthStore, StoredOAuthTokens
from exp.common.models import ConnectionConfig
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.models.credentials import connection_credential_binding

_runner = CliRunner()


def _initialized_root(tmp_path: Path) -> Path:
    """Initialize one gateway root for provider commands."""
    GatewayManagement(tmp_path).initialize()
    return tmp_path


def _plain_output(output: str) -> str:
    """Flatten styled panel output into one space-separated searchable line."""
    return " ".join(unstyle(output).replace("\u2502", " ").split())


def test_provider_add_rejects_an_unsupported_provider_identifier(tmp_path: Path) -> None:
    """A mistyped provider id fails closed with the supported identifiers listed."""
    root = _initialized_root(tmp_path)

    result = _runner.invoke(
        app,
        [
            "config",
            "gateway",
            "provider",
            "add",
            "mistyped",
            "--provider",
            "openai_compatible",
            "--credential-env",
            "TEST_PROVIDER_KEY",
            "--non-interactive",
            "--json",
            "--root",
            str(root),
        ],
    )

    assert result.exit_code != 0
    plain = _plain_output(result.output)
    assert "unsupported provider 'openai_compatible'" in plain
    assert "openai-compatible" in plain
    assert GatewayManagement(root).provider_connections() == ()


def test_provider_update_rejects_an_unsupported_provider_identifier(tmp_path: Path) -> None:
    """Updates pass through the same fail-closed provider identifier validation."""
    root = _initialized_root(tmp_path)
    add = _runner.invoke(
        app,
        [
            "config",
            "gateway",
            "provider",
            "add",
            "main",
            "--provider",
            "openai-compatible",
            "--base-url",
            "http://127.0.0.1:9/v1",
            "--credential-env",
            "TEST_PROVIDER_KEY",
            "--non-interactive",
            "--json",
            "--root",
            str(root),
        ],
    )
    assert add.exit_code == 0

    update = _runner.invoke(
        app,
        [
            "config",
            "gateway",
            "provider",
            "update",
            "main",
            "--provider",
            "open-ai",
            "--credential-env",
            "TEST_PROVIDER_KEY",
            "--non-interactive",
            "--json",
            "--root",
            str(root),
        ],
    )

    assert update.exit_code != 0
    assert "unsupported provider 'open-ai'" in _plain_output(update.output)
    (authority,) = GatewayManagement(root).provider_connections()
    assert authority.config.provider == "openai-compatible"


def test_provider_add_accepts_a_supported_provider_identifier(tmp_path: Path) -> None:
    """A registry-supported provider id is stored and listed without secret values."""
    root = _initialized_root(tmp_path)

    added = _runner.invoke(
        app,
        [
            "config",
            "gateway",
            "provider",
            "add",
            "main",
            "--provider",
            "openai-compatible",
            "--base-url",
            "http://127.0.0.1:9/v1",
            "--credential-env",
            "TEST_PROVIDER_KEY",
            "--non-interactive",
            "--json",
            "--root",
            str(root),
        ],
    )
    listed = _runner.invoke(
        app,
        ["config", "gateway", "provider", "list", "--json", "--root", str(root)],
    )

    assert added.exit_code == 0
    receipt = json.loads(added.output)
    assert receipt["resource_id"] == "main"
    assert receipt["changed"] is True
    assert listed.exit_code == 0
    listing = json.loads(listed.output)
    assert listing["resource_kind"] == "providers"
    (item,) = listing["items"]
    assert item["provider"] == "openai-compatible"


def test_azure_update_preserves_model_inference_surface_when_omitted(tmp_path: Path) -> None:
    """An endpoint-only update cannot silently switch Foundry back to classic Azure."""
    root = _initialized_root(tmp_path)
    common = [
        "--provider",
        "azure",
        "--base-url",
        "https://resource.services.ai.azure.com/models",
        "--credential-env",
        "AZURE_FOUNDRY_API_KEY",
        "--api-version",
        "2024-05-01-preview",
        "--non-interactive",
        "--json",
        "--root",
        str(root),
    ]
    added = _runner.invoke(
        app,
        [
            "config",
            "gateway",
            "provider",
            "add",
            "foundry",
            *common,
            "--azure-api-surface",
            "model_inference",
        ],
    )
    updated = _runner.invoke(
        app,
        ["config", "gateway", "provider", "update", "foundry", *common],
    )
    listed = _runner.invoke(
        app,
        ["config", "gateway", "provider", "list", "--json", "--root", str(root)],
    )

    assert added.exit_code == 0
    assert updated.exit_code == 0
    (authority,) = GatewayManagement(root).provider_connections()
    assert authority.config.azure_api_surface == "model_inference"
    assert json.loads(listed.output)["items"][0]["azure_api_surface"] == "model_inference"


def test_provider_update_drops_azure_surface_when_changing_provider(tmp_path: Path) -> None:
    """A cross-provider update cannot carry an Azure-only discriminator forward."""
    root = _initialized_root(tmp_path)
    added = _runner.invoke(
        app,
        [
            "config",
            "gateway",
            "provider",
            "add",
            "foundry",
            "--provider",
            "azure",
            "--base-url",
            "https://resource.services.ai.azure.com",
            "--credential-env",
            "AZURE_FOUNDRY_API_KEY",
            "--api-version",
            "2024-05-01-preview",
            "--azure-api-surface",
            "model_inference",
            "--non-interactive",
            "--json",
            "--root",
            str(root),
        ],
    )
    updated = _runner.invoke(
        app,
        [
            "config",
            "gateway",
            "provider",
            "update",
            "foundry",
            "--provider",
            "openai-compatible",
            "--base-url",
            "https://gateway.example.test/v1",
            "--credential-env",
            "CUSTOM_API_KEY",
            "--non-interactive",
            "--json",
            "--root",
            str(root),
        ],
    )

    assert added.exit_code == 0, added.output
    assert updated.exit_code == 0, updated.output
    (authority,) = GatewayManagement(root).provider_connections()
    assert authority.config.provider == "openai-compatible"
    assert authority.config.azure_api_surface is None


def test_provider_update_preserves_explicit_bedrock_credentials(tmp_path: Path) -> None:
    """A metadata-only update does not downgrade an explicit Bedrock pair to ambient auth."""
    root = _initialized_root(tmp_path)
    added = _runner.invoke(
        app,
        [
            "config",
            "gateway",
            "provider",
            "add",
            "bedrock-production",
            "--provider",
            "bedrock",
            "--credential-env",
            "AWS_SECRET_ACCESS_KEY",
            "--access-key-id-env",
            "AWS_ACCESS_KEY_ID",
            "--bedrock-auth-mode",
            "access_key_pair",
            "--region",
            "us-west-2",
            "--non-interactive",
            "--json",
            "--root",
            str(root),
        ],
    )
    updated = _runner.invoke(
        app,
        [
            "config",
            "gateway",
            "provider",
            "update",
            "bedrock-production",
            "--provider",
            "bedrock",
            "--region",
            "us-east-1",
            "--non-interactive",
            "--json",
            "--root",
            str(root),
        ],
    )
    listed = _runner.invoke(
        app,
        ["config", "gateway", "provider", "list", "--json", "--root", str(root)],
    )

    assert added.exit_code == 0
    assert updated.exit_code == 0
    (authority,) = GatewayManagement(root).provider_connections()
    assert authority.config.api_key_env == "AWS_SECRET_ACCESS_KEY"
    assert authority.config.aws_access_key_id_env == "AWS_ACCESS_KEY_ID"
    assert authority.config.bedrock_auth_mode == "access_key_pair"
    assert authority.config.region == "us-east-1"
    (item,) = json.loads(listed.output)["items"]
    assert item["credential_env"] == "AWS_SECRET_ACCESS_KEY"
    assert item["access_key_id_env"] == "AWS_ACCESS_KEY_ID"
    assert item["bedrock_auth_mode"] == "access_key_pair"


def test_provider_update_switches_bedrock_pair_to_api_key_without_stale_access_key(
    tmp_path: Path,
) -> None:
    """Changing auth mode drops the access-key locator from bearer configuration."""
    root = _initialized_root(tmp_path)
    added = _runner.invoke(
        app,
        [
            "config",
            "gateway",
            "provider",
            "add",
            "bedrock-production",
            "--provider",
            "bedrock",
            "--credential-env",
            "BEDROCK_SECRET_ACCESS_KEY",
            "--access-key-id-env",
            "BEDROCK_ACCESS_KEY_ID",
            "--bedrock-auth-mode",
            "access_key_pair",
            "--region",
            "us-west-2",
            "--non-interactive",
            "--json",
            "--root",
            str(root),
        ],
    )
    updated = _runner.invoke(
        app,
        [
            "config",
            "gateway",
            "provider",
            "update",
            "bedrock-production",
            "--provider",
            "bedrock",
            "--credential-env",
            "BEDROCK_API_KEY",
            "--bedrock-auth-mode",
            "api_key",
            "--non-interactive",
            "--json",
            "--root",
            str(root),
        ],
    )

    assert added.exit_code == 0
    assert updated.exit_code == 0, updated.output
    (authority,) = GatewayManagement(root).provider_connections()
    assert authority.config.api_key_env == "BEDROCK_API_KEY"
    assert authority.config.aws_access_key_id_env is None
    assert authority.config.bedrock_auth_mode == "api_key"
    assert authority.config.region == "us-west-2"


def test_provider_update_does_not_reinterpret_a_stale_secret_locator_as_an_api_key(
    tmp_path: Path,
) -> None:
    """A mode switch requires a fresh credential locator instead of changing its meaning."""
    root = _initialized_root(tmp_path)
    added = _runner.invoke(
        app,
        [
            "config",
            "gateway",
            "provider",
            "add",
            "bedrock-production",
            "--provider",
            "bedrock",
            "--credential-env",
            "BEDROCK_SECRET_ACCESS_KEY",
            "--access-key-id-env",
            "BEDROCK_ACCESS_KEY_ID",
            "--bedrock-auth-mode",
            "access_key_pair",
            "--non-interactive",
            "--json",
            "--root",
            str(root),
        ],
    )
    rejected = _runner.invoke(
        app,
        [
            "config",
            "gateway",
            "provider",
            "update",
            "bedrock-production",
            "--provider",
            "bedrock",
            "--bedrock-auth-mode",
            "api_key",
            "--non-interactive",
            "--json",
            "--root",
            str(root),
        ],
    )

    assert added.exit_code == 0
    assert rejected.exit_code != 0
    assert "requires --credential-env" in _plain_output(rejected.output)
    (authority,) = GatewayManagement(root).provider_connections()
    assert authority.config.api_key_env == "BEDROCK_SECRET_ACCESS_KEY"
    assert authority.config.aws_access_key_id_env == "BEDROCK_ACCESS_KEY_ID"
    assert authority.config.bedrock_auth_mode == "access_key_pair"


def test_provider_update_can_clear_bedrock_auth_and_region_to_ambient(tmp_path: Path) -> None:
    """Explicit clear flags remove every credential locator and the pinned region."""
    root = _initialized_root(tmp_path)
    added = _runner.invoke(
        app,
        [
            "config",
            "gateway",
            "provider",
            "add",
            "bedrock-production",
            "--provider",
            "bedrock",
            "--credential-env",
            "BEDROCK_SECRET_ACCESS_KEY",
            "--access-key-id-env",
            "BEDROCK_ACCESS_KEY_ID",
            "--bedrock-auth-mode",
            "access_key_pair",
            "--region",
            "us-west-2",
            "--non-interactive",
            "--json",
            "--root",
            str(root),
        ],
    )
    updated = _runner.invoke(
        app,
        [
            "config",
            "gateway",
            "provider",
            "update",
            "bedrock-production",
            "--provider",
            "bedrock",
            "--clear-credentials",
            "--clear-region",
            "--non-interactive",
            "--json",
            "--root",
            str(root),
        ],
    )

    assert added.exit_code == 0
    assert updated.exit_code == 0, updated.output
    (authority,) = GatewayManagement(root).provider_connections()
    assert authority.config.api_key_env is None
    assert authority.config.aws_access_key_id_env is None
    assert authority.config.bedrock_auth_mode is None
    assert authority.config.region is None


def _codex_auth_file(tmp_path: Path) -> Path:
    """Write a Codex-shaped ``auth.json`` carrying unsigned fixture tokens.

    Args:
        tmp_path: Test directory.

    Returns:
        Path of the fixture file.
    """
    claims = {
        "exp": 4_000_000_000,
        "https://api.openai.com/auth": {
            "chatgpt_account_id": "acct-fixture",
            "chatgpt_plan_type": "team",
        },
    }
    segment = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    token = f"header.{segment}.signature"
    path = tmp_path / "codex-auth.json"
    path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": token,
                    "refresh_token": "refresh-fixture",
                    "id_token": token,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_provider_add_imports_a_codex_sign_in_for_a_chatgpt_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plan connection stores the imported sign-in under the connection and lists its kind."""
    root = _initialized_root(tmp_path / "root")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    auth_file = _codex_auth_file(tmp_path)

    result = _runner.invoke(
        app,
        [
            "config",
            "gateway",
            "provider",
            "add",
            "plan-a",
            "--provider",
            "openai",
            "--subscription",
            "chatgpt",
            "--codex-auth-file",
            str(auth_file),
            "--non-interactive",
            "--json",
            "--root",
            str(root),
        ],
    )

    assert result.exit_code == 0, result.output
    receipt = json.loads(result.output)
    assert receipt["data"] == {
        "subscription": "chatgpt",
        "sign_in": "codex-auth-file",
        "plan_type": "team",
    }
    connection = GatewayManagement(root).provider_connections()[0]
    assert connection.config == ConnectionConfig(provider="openai", subscription="chatgpt")
    stored = ProviderAuthStore().get_oauth(
        "plan-a", binding=connection_credential_binding(connection.config)
    )
    assert stored is not None
    assert stored.account_id == "acct-fixture"
    assert "refresh-fixture" not in result.output

    listed = _runner.invoke(
        app, ["config", "gateway", "provider", "list", "--json", "--root", str(root)]
    )
    assert json.loads(listed.output)["items"][0]["subscription"] == "chatgpt"


def test_provider_add_for_a_plan_needs_a_browser_or_an_import_file(tmp_path: Path) -> None:
    """Without a browser, the command names the import alternative instead of hanging."""
    root = _initialized_root(tmp_path)

    result = _runner.invoke(
        app,
        [
            "config",
            "gateway",
            "provider",
            "add",
            "plan-a",
            "--provider",
            "openai",
            "--subscription",
            "chatgpt",
            "--non-interactive",
            "--json",
            "--root",
            str(root),
        ],
    )

    assert result.exit_code != 0
    assert "--codex-auth-file" in _plain_output(result.output)
    assert GatewayManagement(root).provider_connections() == ()


def test_provider_add_rejects_a_plan_with_a_credential_env(tmp_path: Path) -> None:
    """A plan connection and an API-key locator cannot be combined."""
    root = _initialized_root(tmp_path)

    result = _runner.invoke(
        app,
        [
            "config",
            "gateway",
            "provider",
            "add",
            "plan-a",
            "--provider",
            "openai",
            "--subscription",
            "chatgpt",
            "--credential-env",
            "OPENAI_API_KEY",
            "--codex-auth-file",
            str(tmp_path / "missing.json"),
            "--non-interactive",
            "--root",
            str(root),
        ],
    )

    assert result.exit_code != 0
    assert "omit api_key_env" in _plain_output(result.output)


def test_provider_update_keeps_the_plan_sign_in_and_refuses_a_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Updating a plan connection never re-signs in; adding a key to it is refused."""
    root = _initialized_root(tmp_path / "root")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    added = _runner.invoke(
        app,
        [
            "config",
            "gateway",
            "provider",
            "add",
            "plan-a",
            "--provider",
            "openai",
            "--subscription",
            "chatgpt",
            "--codex-auth-file",
            str(_codex_auth_file(tmp_path)),
            "--non-interactive",
            "--root",
            str(root),
        ],
    )
    assert added.exit_code == 0, added.output

    unchanged = _runner.invoke(
        app,
        [
            "config",
            "gateway",
            "provider",
            "update",
            "plan-a",
            "--provider",
            "openai",
            "--json",
            "--root",
            str(root),
        ],
    )
    assert unchanged.exit_code == 0, unchanged.output
    assert GatewayManagement(root).provider_connections()[0].config.subscription == "chatgpt"

    keyed = _runner.invoke(
        app,
        [
            "config",
            "gateway",
            "provider",
            "update",
            "plan-a",
            "--provider",
            "openai",
            "--credential-env",
            "OPENAI_API_KEY",
            "--root",
            str(root),
        ],
    )
    assert keyed.exit_code != 0
    assert "omit api_key_env" in _plain_output(keyed.output)


def test_a_claude_plan_without_the_operator_app_names_the_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Adding a Claude plan with no Anthropic app configured fails with the variables to set."""
    root = _initialized_root(tmp_path)
    monkeypatch.delenv("EXP_ANTHROPIC_OAUTH_CLIENT_ID", raising=False)

    result = _runner.invoke(
        app,
        [
            "config",
            "gateway",
            "provider",
            "add",
            "claude-plan",
            "--provider",
            "anthropic",
            "--subscription",
            "anthropic",
            "--root",
            str(root),
        ],
    )

    assert result.exit_code != 0
    assert "EXP_ANTHROPIC_OAUTH_CLIENT_ID" in _plain_output(result.output)
    assert GatewayManagement(root).provider_connections() == ()


def _add_imported_plan(root: Path, auth_file: Path) -> Result:
    """Run ``provider add plan-a`` as a ChatGPT plan imported from a Codex file."""
    return _runner.invoke(
        app,
        [
            "config",
            "gateway",
            "provider",
            "add",
            "plan-a",
            "--provider",
            "openai",
            "--subscription",
            "chatgpt",
            "--codex-auth-file",
            str(auth_file),
            "--non-interactive",
            "--json",
            "--root",
            str(root),
        ],
    )


def test_a_plan_whose_sign_in_cannot_be_stored_never_becomes_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unusable credential file fails the add before the connection is written."""
    root = _initialized_root(tmp_path / "root")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    store_path = ProviderAuthStore().path
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text("not json", encoding="utf-8")
    store_path.chmod(0o600)

    result = _add_imported_plan(root, _codex_auth_file(tmp_path))

    assert result.exit_code != 0
    assert GatewayManagement(root).provider_connections() == ()


def test_a_failed_activation_removes_the_new_plan_sign_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A connection that fails to activate leaves no stored sign-in behind."""
    root = _initialized_root(tmp_path / "root")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    def refuse(self: GatewayManagement, **_kwargs: object) -> tuple[bool, object]:
        raise ValueError("authority refused the connection")

    monkeypatch.setattr(GatewayManagement, "upsert_provider_connection", refuse)

    result = _add_imported_plan(root, _codex_auth_file(tmp_path))

    assert result.exit_code != 0
    assert "authority refused the connection" in result.output
    assert ProviderAuthStore().get_oauth("plan-a") is None


def test_a_plan_never_overwrites_an_api_key_stored_under_its_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The credential file is shared with ``exp config providers``; its key survives."""
    root = _initialized_root(tmp_path / "root")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    ProviderAuthStore().put("plan-a", "sk-live-key")

    result = _add_imported_plan(root, _codex_auth_file(tmp_path))

    assert result.exit_code != 0
    assert "API key" in _plain_output(result.output)
    assert ProviderAuthStore().get("plan-a") == "sk-live-key"
    assert GatewayManagement(root).provider_connections() == ()


def test_a_failed_replace_puts_back_the_sign_in_the_connection_had(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A revision that fails to activate leaves the live connection on its own sign-in."""
    root = _initialized_root(tmp_path / "root")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    added = _add_imported_plan(root, _codex_auth_file(tmp_path))
    assert added.exit_code == 0, added.output
    live = GatewayManagement(root).provider_connections()[0].config
    before = ProviderAuthStore().get_oauth("plan-a", binding=connection_credential_binding(live))
    assert before is not None

    def refuse(self: GatewayManagement, **_kwargs: object) -> tuple[bool, object]:
        raise ValueError("authority refused the connection")

    monkeypatch.setattr(GatewayManagement, "upsert_provider_connection", refuse)
    second = tmp_path / "second"
    second.mkdir()
    second_file = _codex_auth_file(second)
    document = json.loads(second_file.read_text(encoding="utf-8"))
    document["tokens"]["refresh_token"] = "refresh-second"
    second_file.write_text(json.dumps(document), encoding="utf-8")
    replaced = _runner.invoke(
        app,
        [
            "config",
            "gateway",
            "provider",
            "add",
            "plan-a",
            "--provider",
            "openai",
            "--subscription",
            "chatgpt",
            "--codex-auth-file",
            str(second_file),
            "--replace",
            "--non-interactive",
            "--root",
            str(root),
        ],
    )

    assert replaced.exit_code != 0
    after = ProviderAuthStore().get_oauth("plan-a", binding=connection_credential_binding(live))
    assert after == before
    assert after is not None and after.refresh_token == "refresh-fixture"


def test_removing_a_plan_connection_forgets_its_sign_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No refresh token outlives the plan connection it was issued for."""
    root = _initialized_root(tmp_path / "root")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    added = _add_imported_plan(root, _codex_auth_file(tmp_path))
    assert added.exit_code == 0, added.output

    removed = _runner.invoke(
        app, ["config", "gateway", "provider", "remove", "plan-a", "--root", str(root)]
    )

    assert removed.exit_code == 0, removed.output
    assert ProviderAuthStore().get_oauth("plan-a") is None


def test_a_rejected_add_leaves_a_concurrent_adds_sign_in_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rollback undoes only its own write, never a sign-in another add stored meanwhile."""
    root = _initialized_root(tmp_path / "root")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    winner = StoredOAuthTokens(access_token="winner", refresh_token="winner-r", expires_at_ms=1)

    def lose_the_race(self: GatewayManagement, **_kwargs: object) -> tuple[bool, object]:
        ProviderAuthStore().put_oauth("plan-a", winner)
        raise ValueError("another add activated this connection first")

    monkeypatch.setattr(GatewayManagement, "upsert_provider_connection", lose_the_race)

    result = _add_imported_plan(root, _codex_auth_file(tmp_path))

    assert result.exit_code != 0
    assert ProviderAuthStore().get_oauth("plan-a") == winner


def test_removal_leaves_a_plan_added_under_the_name_meanwhile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removal deletes exactly the sign-in it read, never one a concurrent add stored."""
    root = _initialized_root(tmp_path / "root")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    assert _add_imported_plan(root, _codex_auth_file(tmp_path)).exit_code == 0
    newcomer = StoredOAuthTokens(access_token="new", refresh_token="new-r", expires_at_ms=1)
    disable = GatewayManagement.disable_provider_connection

    def disable_then_race(self: GatewayManagement, *, connection_id: str) -> bool:
        changed = disable(self, connection_id=connection_id)
        ProviderAuthStore().put_oauth(connection_id, newcomer)
        return changed

    monkeypatch.setattr(GatewayManagement, "disable_provider_connection", disable_then_race)

    removed = _runner.invoke(
        app, ["config", "gateway", "provider", "remove", "plan-a", "--root", str(root)]
    )

    assert removed.exit_code == 0, removed.output
    assert ProviderAuthStore().get_oauth("plan-a") == newcomer


def test_an_unreadable_credential_file_never_blocks_removing_a_key_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A key connection disables even when the shared credential file cannot be parsed."""
    root = _initialized_root(tmp_path / "root")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    added = _runner.invoke(
        app,
        [
            "config",
            "gateway",
            "provider",
            "add",
            "keyed",
            "--provider",
            "openai",
            "--credential-env",
            "OPENAI_API_KEY",
            "--root",
            str(root),
        ],
    )
    assert added.exit_code == 0, added.output
    store_path = ProviderAuthStore().path
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text("not json", encoding="utf-8")
    store_path.chmod(0o600)

    removed = _runner.invoke(
        app, ["config", "gateway", "provider", "remove", "keyed", "--root", str(root)]
    )

    assert removed.exit_code == 0, removed.output
    assert GatewayManagement(root).provider_connections() == ()


def test_an_unreadable_credential_file_fails_removing_a_plan_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plan stays until its sign-in can be deleted, so no refresh token is stranded."""
    root = _initialized_root(tmp_path / "root")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    assert _add_imported_plan(root, _codex_auth_file(tmp_path)).exit_code == 0
    ProviderAuthStore().path.write_text("not json", encoding="utf-8")

    removed = _runner.invoke(
        app, ["config", "gateway", "provider", "remove", "plan-a", "--root", str(root)]
    )

    assert removed.exit_code != 0
    assert "cannot read the sign-in" in " ".join(removed.output.split())
    assert [c.connection_id for c in GatewayManagement(root).provider_connections()] == ["plan-a"]
