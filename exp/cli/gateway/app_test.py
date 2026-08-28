"""Command-level tests for the deferred local gateway management surface."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from click import unstyle
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.gateway import app as gateway_cli_app
from exp.cli.gateway import key_output as gateway_key_output
from exp.common.core.artifacts import sha256_json
from exp.common.models import (
    BillingSource,
    ModelCatalog,
    load_model_catalog,
    normalize_gateway_catalog,
)
from exp.runtime.gateway import catalog_authority as gateway_catalog
from exp.runtime.gateway.auth import IssuedVirtualKey, issue_key_material
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.sqlite import key_delivery
from exp.runtime.gateway.sqlite.alias_activation import AliasActivationOutcomeUnknownError
from exp.runtime.gateway.sqlite.provider_authority import ProviderConnectionBinding
from exp.runtime.gateway.sqlite.store import OperationOutcomeUnknownError, SQLiteGatewayStore

# call, models, and key check are the deliberate caller-side additions for the agent
# core loop against a live gateway: one-shot completion, caller-view discovery, and
# raw-key validation. They widen the locked gateway tree on purpose; see caller.py.
EXPECTED_GATEWAY_COMMANDS = {"call", "init", "models", "status", "usage"}
EXPECTED_GATEWAY_GROUPS = {
    "alias": {"create", "disable", "list", "update"},
    "budget": {"list", "reconcile", "remaining", "set"},
    "grant": {"add", "list", "remove"},
    "identity": {"create", "disable", "list", "update"},
    "key": {"check", "issue", "list", "revoke"},
    "pool": {"certify"},
    "provider": {"add", "disable", "list", "remove", "update"},
}


def test_gateway_help_tree_is_exact_and_every_node_renders() -> None:
    """Lock every public gateway command and prove its installed help path."""
    direct = {command.name for command in gateway_cli_app.gateway_app.registered_commands}
    groups: dict[str | None, set[str | None]] = {}
    for group in gateway_cli_app.gateway_app.registered_groups:
        assert group.typer_instance is not None
        groups[group.name] = {command.name for command in group.typer_instance.registered_commands}
    assert direct == EXPECTED_GATEWAY_COMMANDS
    assert groups == EXPECTED_GATEWAY_GROUPS

    runner = CliRunner()
    paths = [["config", "gateway"]]
    paths.extend(["config", "gateway", name] for name in EXPECTED_GATEWAY_GROUPS)
    paths.extend(["config", "gateway", name] for name in EXPECTED_GATEWAY_COMMANDS)
    paths.extend(
        ["config", "gateway", group, command]
        for group, commands in EXPECTED_GATEWAY_GROUPS.items()
        for command in commands
    )
    for path in paths:
        result = runner.invoke(app, [*path, "--help"])
        assert result.exit_code == 0, result.output


def test_noninteractive_management_story_emits_stable_secret_safe_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An agent can author, grant, issue, inspect, and report without prompts."""
    monkeypatch.setenv("TEST_PROVIDER_KEY", "provider-secret-canary")
    runner = CliRunner()
    commands = (
        ["config", "gateway", "init", "--root", str(tmp_path), "--json"],
        [
            "config",
            "gateway",
            "provider",
            "add",
            "provider-main",
            "--provider",
            "openai-compatible",
            "--credential-env",
            "TEST_PROVIDER_KEY",
            "--base-url",
            "http://127.0.0.1:9/v1",
            "--root",
            str(tmp_path),
            "--non-interactive",
            "--json",
        ],
        [
            "config",
            "gateway",
            "alias",
            "create",
            "coding",
            "--deployment",
            "provider-main:provider-model-exact",
            "--exact-model",
            "model-revision-exact",
            "--root",
            str(tmp_path),
            "--non-interactive",
            "--json",
        ],
        [
            "config",
            "gateway",
            "identity",
            "create",
            "default",
            "--root",
            str(tmp_path),
            "--non-interactive",
            "--json",
        ],
        [
            "config",
            "gateway",
            "grant",
            "add",
            "default",
            "coding",
            "--root",
            str(tmp_path),
            "--non-interactive",
            "--json",
        ],
    )
    for command in commands:
        result = runner.invoke(app, command)
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["schema_version"] == 1

    issue = runner.invoke(
        app,
        [
            "config",
            "gateway",
            "key",
            "issue",
            "default",
            "--key-id",
            "key-one",
            "--root",
            str(tmp_path),
            "--non-interactive",
            "--json",
        ],
    )
    assert issue.exit_code == 0, issue.output
    raw_key = json.loads(issue.stdout)["data"]["raw_key"]
    assert raw_key.startswith("exp_vk_")

    key_list = runner.invoke(
        app,
        ["config", "gateway", "key", "list", "--root", str(tmp_path), "--json"],
    )
    usage = runner.invoke(
        app,
        ["config", "gateway", "usage", "--root", str(tmp_path), "--json"],
    )
    assert key_list.exit_code == 0, key_list.output
    assert usage.exit_code == 0, usage.output
    assert raw_key not in key_list.stdout
    assert raw_key not in usage.stdout
    assert json.loads(usage.stdout)["schema_version"] == 2

    readiness = runner.invoke(
        app,
        ["--root", str(tmp_path), "--check", "--non-interactive", "--json"],
    )
    assert readiness.exit_code == 0, readiness.output
    assert json.loads(readiness.stdout)["status"] == "ready"

    durable = b"".join(
        path.read_bytes() for path in (tmp_path / "gateway").rglob("*") if path.is_file()
    )
    assert raw_key.encode() not in durable


def test_direct_alias_uses_provider_certification_for_tool_streaming(
    tmp_path: Path,
) -> None:
    """Alias authoring never infers raw argument streaming from model tool support alone."""
    runner = CliRunner()
    commands = (
        ["config", "gateway", "init", "--root", str(tmp_path), "--json"],
        [
            "config",
            "gateway",
            "provider",
            "add",
            "oai",
            "--provider",
            "openai",
            "--credential-env",
            "OPENAI_API_KEY",
            "--root",
            str(tmp_path),
            "--non-interactive",
            "--json",
        ],
        [
            "config",
            "gateway",
            "provider",
            "add",
            "google",
            "--provider",
            "gemini",
            "--credential-env",
            "GEMINI_API_KEY",
            "--root",
            str(tmp_path),
            "--non-interactive",
            "--json",
        ],
    )
    for command in commands:
        result = runner.invoke(app, command)
        assert result.exit_code == 0, result.output
    for alias, deployment in (
        ("oai-tools", "oai:gpt-fixture"),
        ("gemini-tools", "google:gemini-fixture"),
    ):
        result = runner.invoke(
            app,
            [
                "config",
                "gateway",
                "alias",
                "create",
                alias,
                "--deployment",
                deployment,
                "--exact-model",
                alias,
                "--supports-tools",
                "--root",
                str(tmp_path),
                "--non-interactive",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output

    catalog = load_model_catalog(tmp_path / "models.toml")
    assert catalog.models["oai-tools"].gateway is not None
    assert catalog.models["gemini-tools"].gateway is not None
    assert catalog.models["oai-tools"].gateway.capabilities.supports_streaming_tool_arguments
    assert not catalog.models["gemini-tools"].gateway.capabilities.supports_streaming_tool_arguments


def test_noninteractive_pool_certification_activates_ordered_alias_with_receipt(
    tmp_path: Path,
) -> None:
    """An agent can certify and activate a digest-guarded ordered deployment pool."""
    runner = CliRunner()
    setup_commands = (
        ["config", "gateway", "init", "--root", str(tmp_path), "--json"],
        [
            "config",
            "gateway",
            "provider",
            "add",
            "provider-main",
            "--provider",
            "openai-compatible",
            "--credential-env",
            "TEST_PROVIDER_KEY",
            "--base-url",
            "http://127.0.0.1:9/v1",
            "--root",
            str(tmp_path),
            "--non-interactive",
            "--json",
        ],
    )
    for command in setup_commands:
        result = runner.invoke(app, command)
        assert result.exit_code == 0, result.output
    catalog_sha256 = ""
    alias_receipts: dict[str, dict[str, object]] = {}
    for deployment_alias in ("primary", "secondary"):
        billing_arguments = (
            ["--billing-source", BillingSource.HOST_MANAGED.value]
            if deployment_alias == "primary"
            else []
        )
        result = runner.invoke(
            app,
            [
                "config",
                "gateway",
                "alias",
                "create",
                deployment_alias,
                "--deployment",
                f"provider-main:{deployment_alias}-model",
                "--exact-model",
                "model-revision-exact",
                *billing_arguments,
                "--root",
                str(tmp_path),
                "--non-interactive",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        receipt = cast(dict[str, object], json.loads(result.stdout))
        data = cast(dict[str, object], receipt["data"])
        alias_receipts[deployment_alias] = data
        catalog_sha256 = cast(str, data["catalog_sha256"])

    assert alias_receipts["primary"]["billing_source"] == "host_managed"
    assert alias_receipts["secondary"]["billing_source"] == "customer_managed"

    certified = runner.invoke(
        app,
        [
            "config",
            "gateway",
            "pool",
            "certify",
            "coding",
            "--deployment-alias",
            "primary",
            "--deployment-alias",
            "secondary",
            "--exact-model",
            "model-revision-exact",
            "--certification-id",
            "certification-one",
            "--provenance",
            "operator-reviewed deployment manifests",
            "--evidence-sha256",
            "a" * 64,
            "--certified-at",
            "2026-08-18T00:00:00Z",
            "--expected-catalog-sha256",
            catalog_sha256,
            "--revision",
            "revision-waterfall-one",
            "--root",
            str(tmp_path),
            "--non-interactive",
            "--json",
        ],
    )

    assert certified.exit_code == 0, certified.output
    receipt = json.loads(certified.stdout)
    assert receipt["schema_version"] == 1
    assert receipt["operation"] == "pool.certify"
    assert receipt["data"]["deployment_aliases"] == ["primary", "secondary"]
    assert receipt["data"]["catalog_sha256"] != catalog_sha256
    assert "TEST_PROVIDER_KEY" not in certified.stdout
    catalog = load_model_catalog(tmp_path / "models.toml")
    assert catalog.gateway_pools["coding"].deployment_aliases == ("primary", "secondary")
    assert catalog.models["primary"].billing_source is BillingSource.HOST_MANAGED
    assert catalog.models["secondary"].billing_source is BillingSource.CUSTOMER_MANAGED
    normalized = normalize_gateway_catalog(catalog)
    pool = next(item for item in normalized.pools if item.pool_id == "coding")
    deployments = {deployment.deployment_id: deployment for deployment in normalized.deployments}
    assert tuple(
        deployments[deployment_id].billing_source for deployment_id in pool.deployment_ids
    ) == (BillingSource.HOST_MANAGED, BillingSource.CUSTOMER_MANAGED)
    alias = next(
        item for item in GatewayManagement(tmp_path).aliases() if item.alias_id == "coding"
    )
    assert alias.pool_id == "coding"
    assert alias.revision_id == "revision-waterfall-one"


def test_project_alias_rejects_deployment_billing_source(tmp_path: Path) -> None:
    """Credential ownership cannot be falsely attributed to a project target."""
    result = CliRunner().invoke(
        app,
        [
            "config",
            "gateway",
            "alias",
            "create",
            "project-model",
            "--project",
            "project-one",
            "--billing-source",
            BillingSource.HOST_MANAGED.value,
            "--root",
            str(tmp_path),
            "--non-interactive",
            "--json",
        ],
    )

    assert result.exit_code == 2
    output_text = " ".join(unstyle(result.output).replace("│", " ").split())
    assert "--billing-source applies only to direct --deployment aliases" in output_text


def test_pool_certification_preflights_revision_conflict_before_catalog_write(
    tmp_path: Path,
) -> None:
    """A reused immutable revision cannot leave a second certified pool behind."""
    runner, catalog_sha256 = _prepare_pool_certification_root(tmp_path)
    first = runner.invoke(
        app,
        _pool_certification_command(
            tmp_path,
            alias="coding",
            revision="revision-waterfall-one",
            expected_catalog_sha256=catalog_sha256,
        ),
    )
    assert first.exit_code == 0, first.output
    current_sha256 = json.loads(first.stdout)["data"]["catalog_sha256"]
    catalog_before = (tmp_path / "models.toml").read_bytes()
    aliases_before = GatewayManagement(tmp_path).aliases()

    conflicting = runner.invoke(
        app,
        _pool_certification_command(
            tmp_path,
            alias="analysis",
            revision="revision-waterfall-one",
            expected_catalog_sha256=current_sha256,
        ),
    )

    assert conflicting.exit_code == 2
    assert "revision ID was reused" in conflicting.output
    assert (tmp_path / "models.toml").read_bytes() == catalog_before
    assert GatewayManagement(tmp_path).aliases() == aliases_before
    assert "analysis" not in load_model_catalog(tmp_path / "models.toml").gateway_pools


def test_pool_certification_rolls_back_activation_failure_and_replays_exact_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed authority write restores the catalog and the exact retry is idempotent."""
    runner, catalog_sha256 = _prepare_pool_certification_root(tmp_path)
    command = _pool_certification_command(
        tmp_path,
        alias="coding",
        revision="revision-waterfall-one",
        expected_catalog_sha256=catalog_sha256,
    )
    catalog_before = (tmp_path / "models.toml").read_bytes()
    aliases_before = GatewayManagement(tmp_path).aliases()
    preflight = GatewayManagement.preflight_direct_alias_activation
    preflight_calls = 0

    def fail_activation(_manager: GatewayManagement, **_kwargs: object) -> bool:
        """Inject a failure after the catalog update but before SQLite activation."""
        raise RuntimeError("injected activation failure")

    def fail_if_reconciliation_is_attempted(
        manager: GatewayManagement,
        **kwargs: object,
    ) -> bool:
        """Allow initial preflight but make any inappropriate recovery read fail."""
        nonlocal preflight_calls
        preflight_calls += 1
        if preflight_calls > 1:
            raise OSError("injected reconciliation read failure")
        return preflight(manager, **kwargs)  # ty: ignore[invalid-argument-type]

    with monkeypatch.context() as patch:
        patch.setattr(GatewayManagement, "activate_direct_alias", fail_activation)
        patch.setattr(
            GatewayManagement,
            "preflight_direct_alias_activation",
            fail_if_reconciliation_is_attempted,
        )
        failed = runner.invoke(app, command)

    assert failed.exit_code == 1
    assert isinstance(failed.exception, RuntimeError)
    assert preflight_calls == 1
    assert (tmp_path / "models.toml").read_bytes() == catalog_before
    assert GatewayManagement(tmp_path).aliases() == aliases_before

    retried = runner.invoke(app, command)
    replayed = runner.invoke(app, command)

    assert retried.exit_code == 0, retried.output
    assert replayed.exit_code == 0, replayed.output
    assert json.loads(retried.stdout)["changed"] is True
    assert json.loads(replayed.stdout)["changed"] is False
    certified_aliases = tuple(
        item for item in GatewayManagement(tmp_path).aliases() if item.alias_id == "coding"
    )
    assert len(certified_aliases) == 1
    assert certified_aliases[0].revision_id == "revision-waterfall-one"


def test_pool_certification_reports_unproven_catalog_compensation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed rollback preserves evidence and emits a content-free recovery receipt."""
    runner, catalog_sha256 = _prepare_pool_certification_root(tmp_path)
    command = _pool_certification_command(
        tmp_path,
        alias="coding",
        revision="revision-waterfall-one",
        expected_catalog_sha256=catalog_sha256,
    )
    original = load_model_catalog(tmp_path / "models.toml")
    write_catalog = gateway_catalog.write_model_catalog

    def fail_activation(_manager: GatewayManagement, **_kwargs: object) -> bool:
        """Inject a definite precommit alias activation failure."""
        raise RuntimeError("injected activation failure")

    def fail_exact_rollback(path: Path, catalog: ModelCatalog) -> None:
        """Leave the desired catalog durable when exact preimage restoration fails."""
        if catalog == original:
            raise OSError("injected rollback failure")
        write_catalog(path, catalog)

    with monkeypatch.context() as patch:
        patch.setattr(GatewayManagement, "activate_direct_alias", fail_activation)
        patch.setattr(gateway_catalog, "write_model_catalog", fail_exact_rollback)
        failed = runner.invoke(app, command)

    assert failed.exit_code == 1
    receipt = json.loads(failed.stdout)
    assert receipt["changed"] is None
    assert receipt["data"]["status"] == "catalog_compensation_outcome_unknown"
    assert receipt["data"]["alias_activation"] == "not_committed"
    assert receipt["data"]["recovery"] == (
        "inspect the catalog digest and alias status before retrying"
    )
    assert "injected" not in failed.stdout
    assert "coding" in load_model_catalog(tmp_path / "models.toml").gateway_pools
    assert all(item.alias_id != "coding" for item in GatewayManagement(tmp_path).aliases())


def test_pool_certification_accepts_acknowledgement_lost_after_exact_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rollback write that landed before raising is reconciled as exact restoration."""
    runner, catalog_sha256 = _prepare_pool_certification_root(tmp_path)
    command = _pool_certification_command(
        tmp_path,
        alias="coding",
        revision="revision-waterfall-one",
        expected_catalog_sha256=catalog_sha256,
    )
    catalog_before = (tmp_path / "models.toml").read_bytes()
    original = load_model_catalog(tmp_path / "models.toml")
    write_catalog = gateway_catalog.write_model_catalog

    def fail_activation(_manager: GatewayManagement, **_kwargs: object) -> bool:
        """Inject a definite precommit alias activation failure."""
        raise RuntimeError("injected activation failure")

    def restore_then_raise(path: Path, catalog: ModelCatalog) -> None:
        """Persist the exact preimage and lose only its acknowledgement."""
        write_catalog(path, catalog)
        if catalog == original:
            raise OSError("injected rollback acknowledgement loss")

    with monkeypatch.context() as patch:
        patch.setattr(GatewayManagement, "activate_direct_alias", fail_activation)
        patch.setattr(gateway_catalog, "write_model_catalog", restore_then_raise)
        failed = runner.invoke(app, command)

    assert failed.exit_code == 1
    assert isinstance(failed.exception, RuntimeError)
    assert (tmp_path / "models.toml").read_bytes() == catalog_before
    assert all(item.alias_id != "coding" for item in GatewayManagement(tmp_path).aliases())


def test_pool_certification_rolls_back_snapshot_failure_before_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A snapshot failure restores the catalog preimage before authority can change."""
    runner, catalog_sha256 = _prepare_pool_certification_root(tmp_path)
    command = _pool_certification_command(
        tmp_path,
        alias="coding",
        revision="revision-waterfall-one",
        expected_catalog_sha256=catalog_sha256,
    )
    catalog_before = (tmp_path / "models.toml").read_bytes()
    aliases_before = GatewayManagement(tmp_path).aliases()

    def fail_snapshot(*_args: object, **_kwargs: object) -> Path:
        """Inject snapshot failure after the authored catalog write lands."""
        raise OSError("injected snapshot persistence failure")

    with monkeypatch.context() as patch:
        patch.setattr(gateway_catalog, "_write_catalog_snapshot", fail_snapshot)
        failed = runner.invoke(app, command)

    assert failed.exit_code == 1
    assert isinstance(failed.exception, OSError)
    assert (tmp_path / "models.toml").read_bytes() == catalog_before
    assert GatewayManagement(tmp_path).aliases() == aliases_before

    retried = runner.invoke(app, command)
    assert retried.exit_code == 0, retried.output
    assert json.loads(retried.stdout)["changed"] is True


def test_pool_certification_preserves_desired_catalog_for_typed_commit_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a typed COMMIT ambiguity preserves desired catalog for manual recovery."""
    runner, catalog_sha256 = _prepare_pool_certification_root(tmp_path)
    command = _pool_certification_command(
        tmp_path,
        alias="coding",
        revision="revision-waterfall-one",
        expected_catalog_sha256=catalog_sha256,
    )
    activate = GatewayManagement.activate_direct_alias

    def commit_then_raise(
        manager: GatewayManagement,
        *,
        alias_id: str,
        alias_name: str,
        revision_id: str,
        pool_id: str,
        snapshot_ref: str,
        catalog_sha256: str,
        provider_connections: tuple[ProviderConnectionBinding, ...] = (),
        refusal_failover: bool = False,
    ) -> bool:
        """Commit exact authority and then simulate a lost acknowledgement."""
        activate(
            manager,
            alias_id=alias_id,
            alias_name=alias_name,
            revision_id=revision_id,
            pool_id=pool_id,
            snapshot_ref=snapshot_ref,
            catalog_sha256=catalog_sha256,
            provider_connections=provider_connections,
            refusal_failover=refusal_failover,
        )
        raise AliasActivationOutcomeUnknownError(
            alias_id=alias_id,
            revision_id=revision_id,
        )

    with monkeypatch.context() as patch:
        patch.setattr(GatewayManagement, "activate_direct_alias", commit_then_raise)
        unknown = runner.invoke(app, command)

    assert unknown.exit_code == 1
    receipt = json.loads(unknown.stdout)
    assert receipt["changed"] is None
    assert receipt["data"]["status"] == "operation_outcome_unknown"
    assert receipt["data"]["recovery"] == "inspect alias status before retrying"
    assert "coding" in load_model_catalog(tmp_path / "models.toml").gateway_pools
    alias = next(
        item for item in GatewayManagement(tmp_path).aliases() if item.alias_id == "coding"
    )
    assert alias.revision_id == "revision-waterfall-one"

    replayed = runner.invoke(app, command)
    assert replayed.exit_code == 0, replayed.output
    assert json.loads(replayed.stdout)["changed"] is False


def test_pool_certification_rejects_a_stale_catalog_digest_before_activation(
    tmp_path: Path,
) -> None:
    """Optimistic catalog authority prevents a stale pool activation."""
    runner = CliRunner()
    initialized = runner.invoke(
        app,
        ["config", "gateway", "init", "--root", str(tmp_path), "--json"],
    )
    assert initialized.exit_code == 0, initialized.output
    provider = runner.invoke(
        app,
        [
            "config",
            "gateway",
            "provider",
            "add",
            "provider-main",
            "--provider",
            "openai-compatible",
            "--credential-env",
            "TEST_PROVIDER_KEY",
            "--base-url",
            "http://127.0.0.1:9/v1",
            "--root",
            str(tmp_path),
            "--non-interactive",
            "--json",
        ],
    )
    assert provider.exit_code == 0, provider.output
    for deployment_alias in ("primary", "secondary"):
        created = runner.invoke(
            app,
            [
                "config",
                "gateway",
                "alias",
                "create",
                deployment_alias,
                "--deployment",
                f"provider-main:{deployment_alias}-model",
                "--exact-model",
                "model-revision-exact",
                "--root",
                str(tmp_path),
                "--non-interactive",
                "--json",
            ],
        )
        assert created.exit_code == 0, created.output

    result = runner.invoke(
        app,
        [
            "config",
            "gateway",
            "pool",
            "certify",
            "coding",
            "--deployment-alias",
            "primary",
            "--deployment-alias",
            "secondary",
            "--exact-model",
            "model-revision-exact",
            "--certification-id",
            "certification-one",
            "--provenance",
            "operator-reviewed deployment manifests",
            "--evidence-sha256",
            "a" * 64,
            "--certified-at",
            "2026-08-18T00:00:00Z",
            "--expected-catalog-sha256",
            "0" * 64,
            "--revision",
            "revision-waterfall-one",
            "--root",
            str(tmp_path),
            "--non-interactive",
            "--json",
        ],
    )

    assert result.exit_code == 2
    assert "refresh its digest" in result.output
    assert "coding" not in load_model_catalog(tmp_path / "models.toml").gateway_pools
    assert all(item.alias_id != "coding" for item in GatewayManagement(tmp_path).aliases())


def _prepare_pool_certification_root(tmp_path: Path) -> tuple[CliRunner, str]:
    """Create two compatible deployments and return their latest catalog digest."""
    runner = CliRunner()
    initialized = runner.invoke(
        app,
        ["config", "gateway", "init", "--root", str(tmp_path), "--json"],
    )
    assert initialized.exit_code == 0, initialized.output
    provider = runner.invoke(
        app,
        [
            "config",
            "gateway",
            "provider",
            "add",
            "provider-main",
            "--provider",
            "openai-compatible",
            "--credential-env",
            "TEST_PROVIDER_KEY",
            "--base-url",
            "http://127.0.0.1:9/v1",
            "--root",
            str(tmp_path),
            "--non-interactive",
            "--json",
        ],
    )
    assert provider.exit_code == 0, provider.output
    catalog_sha256 = ""
    for deployment_alias in ("primary", "secondary"):
        created = runner.invoke(
            app,
            [
                "config",
                "gateway",
                "alias",
                "create",
                deployment_alias,
                "--deployment",
                f"provider-main:{deployment_alias}-model",
                "--exact-model",
                "model-revision-exact",
                "--root",
                str(tmp_path),
                "--non-interactive",
                "--json",
            ],
        )
        assert created.exit_code == 0, created.output
        catalog_sha256 = json.loads(created.stdout)["data"]["catalog_sha256"]
    return runner, catalog_sha256


def _pool_certification_command(
    root: Path,
    *,
    alias: str,
    revision: str,
    expected_catalog_sha256: str,
) -> list[str]:
    """Build one noninteractive certified-pool command with deterministic evidence."""
    return [
        "config",
        "gateway",
        "pool",
        "certify",
        alias,
        "--deployment-alias",
        "primary",
        "--deployment-alias",
        "secondary",
        "--exact-model",
        "model-revision-exact",
        "--certification-id",
        "certification-one",
        "--provenance",
        "operator-reviewed deployment manifests",
        "--evidence-sha256",
        "a" * 64,
        "--certified-at",
        "2026-08-18T00:00:00Z",
        "--expected-catalog-sha256",
        expected_catalog_sha256,
        "--revision",
        revision,
        "--root",
        str(root),
        "--non-interactive",
        "--json",
    ]


def test_key_output_collision_is_rejected_before_key_issuance(tmp_path: Path) -> None:
    """An existing output path cannot consume unrecoverable one-time key material."""
    runner = CliRunner()
    manager = GatewayManagement(tmp_path)
    manager.initialize()
    manager.create_identity(identity_id="default", display_name="Default")
    output = tmp_path / "existing-key"
    output.write_text("keep", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "config",
            "gateway",
            "key",
            "issue",
            "default",
            "--key-id",
            "key-one",
            "--root",
            str(tmp_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 2
    assert output.read_text(encoding="utf-8") == "keep"
    assert manager.keys() == ()


def test_key_output_path_receives_the_only_raw_secret_copy(tmp_path: Path) -> None:
    """Selecting a private file keeps raw key material out of JSON stdout."""
    runner = CliRunner()
    manager = GatewayManagement(tmp_path)
    manager.initialize()
    manager.create_identity(identity_id="default", display_name="Default")
    output = tmp_path / "issued-key"

    result = runner.invoke(
        app,
        [
            "config",
            "gateway",
            "key",
            "issue",
            "default",
            "--key-id",
            "key-one",
            "--root",
            str(tmp_path),
            "--output",
            str(output),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    receipt = json.loads(result.stdout)
    assert "raw_key" not in receipt["data"]
    raw_key = output.read_text(encoding="utf-8").strip()
    assert raw_key.startswith("exp_vk_")
    assert raw_key not in result.stdout
    assert output.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("failure", ["fsync", "link"])
def test_key_output_failure_rolls_back_key_receipt_and_partial_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """Fallible secret publication leaves no key and permits an identical retry."""
    runner = CliRunner()
    manager = GatewayManagement(tmp_path)
    manager.initialize()
    manager.create_identity(identity_id="default", display_name="Default")
    output = tmp_path / "issued-key"
    arguments = [
        "config",
        "gateway",
        "key",
        "issue",
        "default",
        "--key-id",
        "key-one",
        "--operation-id",
        "operation-one",
        "--root",
        str(tmp_path),
        "--output",
        str(output),
        "--json",
    ]

    def fail_operation(*_args: object, **_kwargs: object) -> None:
        """Inject one filesystem durability failure without secret-bearing detail."""
        raise OSError(f"injected {failure} failure")

    with monkeypatch.context() as scoped:
        scoped.setattr(gateway_key_output.os, failure, fail_operation)
        failed = runner.invoke(app, arguments)

    assert failed.exit_code == 2
    assert "exp_vk_" not in failed.stdout
    assert not output.exists()
    assert tuple(tmp_path.glob(".issued-key.*.tmp")) == ()
    assert manager.keys() == ()
    connection = manager.require_initialized().database_path
    with sqlite3.connect(connection) as database:
        assert database.execute("SELECT COUNT(*) FROM operation_receipts").fetchone()[0] == 0

    retried = runner.invoke(app, arguments)
    assert retried.exit_code == 0, retried.output
    assert output.read_text(encoding="utf-8").strip().startswith("exp_vk_")
    assert len(manager.keys()) == 1


def test_duplicate_key_id_is_a_content_free_controlled_cli_error(tmp_path: Path) -> None:
    """A duplicate key ID never exposes a raw SQLite failure or second secret."""
    runner = CliRunner()
    manager = GatewayManagement(tmp_path)
    manager.initialize()
    manager.create_identity(identity_id="default", display_name="Default")
    issued = manager.issue_key(identity_id="default", key_id="key-one")
    output = tmp_path / "duplicate-key"

    result = runner.invoke(
        app,
        [
            "config",
            "gateway",
            "key",
            "issue",
            "default",
            "--key-id",
            "key-one",
            "--root",
            str(tmp_path),
            "--output",
            str(output),
            "--json",
        ],
    )

    assert result.exit_code == 2
    normalized = " ".join(unstyle(result.output).replace("│", " ").split())
    assert "virtual key issuance conflicts with existing gateway authority" in normalized
    assert "Traceback" not in normalized
    assert "IntegrityError" not in normalized
    assert issued.raw_key not in normalized
    assert not output.exists()
    assert not gateway_key_output.key_output_marker_path(output).exists()
    assert tuple(tmp_path.glob(".duplicate-key.*.reserve")) == ()
    assert tuple(item.key_id for item in manager.keys()) == ("key-one",)


@pytest.mark.parametrize(
    ("crash_phase", "return_code"),
    (("reserved", 68), ("linked", 69), ("partial", 70), ("precommit", 71), ("postcommit", 72)),
)
def test_key_output_process_crash_recovers_exact_delivery(
    tmp_path: Path,
    crash_phase: str,
    return_code: int,
) -> None:
    """Restart reconciles each durable publication transition after process death.

    Args:
        tmp_path: Pytest-owned gateway root and output directory.
        crash_phase: Durable key-publication transition that kills the child.
        return_code: Expected child-process exit status.
    """
    manager = GatewayManagement(tmp_path)
    manager.initialize()
    manager.create_identity(identity_id="default", display_name="Default")
    output = tmp_path / "issued-key"
    script = """
import os
import sys
from functools import partial
from pathlib import Path

from exp.cli.gateway import key_output
from exp.runtime.gateway.management import GatewayManagement

root = Path(sys.argv[1])
output = Path(sys.argv[2])
phase = sys.argv[3]
manager = GatewayManagement(root)
store = manager.require_initialized()
original_link = key_output.os.link
original_write = key_output.os.write
if phase in {"reserved", "linked"}:
    def crash_around_target_link(source, target, *args, **kwargs):
        if Path(target) == output and phase == "reserved":
            os._exit(68)
        result = original_link(source, target, *args, **kwargs)
        if Path(target) == output:
            os._exit(69)
        return result
    key_output.os.link = crash_around_target_link
if phase == "partial":
    def crash_mid_output(descriptor, payload):
        if payload.startswith(b"exp_vk_"):
            original_write(descriptor, payload[:8])
            os._exit(70)
        return original_write(descriptor, payload)
    key_output.os.write = crash_mid_output
if phase == "precommit":
    def crash_before_commit(*_args, **_kwargs):
        os._exit(71)
    store._record_operation = crash_before_commit
store.issue_virtual_key(
    organization_id=manager.organization_id,
    identity_id="default",
    key_id="key-one",
    operation_id="operation-one",
    secret_delivery=partial(key_output.deliver_key_output, output),
)
os._exit(72)
"""
    crashed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), str(output), crash_phase],
        cwd=Path(__file__).parents[3],
        check=False,
        capture_output=True,
        text=True,
    )

    assert crashed.returncode == return_code
    marker = gateway_key_output.key_output_marker_path(output)
    assert marker.is_file()
    if crash_phase == "reserved":
        assert not output.exists()
    else:
        assert output.is_file()
    before = output.read_bytes() if output.exists() else b""
    runner = CliRunner()
    retried = runner.invoke(
        app,
        [
            "config",
            "gateway",
            "key",
            "issue",
            "default",
            "--key-id",
            "key-one",
            "--operation-id",
            "operation-one",
            "--root",
            str(tmp_path),
            "--output",
            str(output),
            "--json",
        ],
    )

    assert retried.exit_code == 0, retried.output
    receipt = json.loads(retried.stdout)
    if crash_phase == "postcommit":
        assert receipt["data"]["status"] == "recovered_committed"
        assert receipt["changed"] is False
        assert output.read_bytes() == before
    else:
        assert "status" not in receipt["data"]
        assert receipt["changed"] is True
    assert not marker.exists()
    assert tuple(tmp_path.glob(".issued-key.*.reserve")) == ()
    assert len(manager.keys()) == 1


def test_key_output_crash_recovery_preserves_mismatched_output(tmp_path: Path) -> None:
    """A modified orphan is never deleted even when SQLite proves rollback."""
    manager = GatewayManagement(tmp_path)
    manager.initialize()
    manager.create_identity(identity_id="default", display_name="Default")
    output = tmp_path / "issued-key"
    prefix, raw_key = issue_key_material()
    evidence = key_delivery.KeyDeliveryEvidence(
        organization_id=manager.organization_id,
        identity_id="default",
        key_id="key-one",
        operation_id="operation-one",
        request_sha256=sha256_json(
            {
                "organization_id": manager.organization_id,
                "identity_id": "default",
                "key_id": "key-one",
                "expires_at": None,
            }
        ),
        prefix=prefix,
        fingerprint_version=1,
        fingerprint_sha256="a" * 64,
        expires_at=None,
        created_at="2026-08-19T00:00:00Z",
    )
    gateway_key_output.deliver_key_output(output, raw_key, evidence)
    marker = gateway_key_output.key_output_marker_path(output)
    output.unlink()
    output.write_text("tampered\n", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "config",
            "gateway",
            "key",
            "issue",
            "default",
            "--key-id",
            "key-one",
            "--operation-id",
            "operation-one",
            "--root",
            str(tmp_path),
            "--output",
            str(output),
            "--json",
        ],
    )

    assert result.exit_code == 2
    assert output.read_text(encoding="utf-8") == "tampered\n"
    assert marker.exists()
    assert manager.keys() == ()


def test_key_output_delivery_failure_preserves_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exception cleanup cannot unlink a path swapped after O_EXCL creation."""
    output = tmp_path / "issued-key"
    prefix, raw_key = issue_key_material()
    evidence = key_delivery.KeyDeliveryEvidence(
        organization_id="local",
        identity_id="default",
        key_id="key-one",
        operation_id="operation-one",
        request_sha256="a" * 64,
        prefix=prefix,
        fingerprint_version=1,
        fingerprint_sha256="b" * 64,
        expires_at=None,
        created_at="2026-08-19T00:00:00Z",
    )
    original_write = gateway_key_output.os.write

    def replace_output(descriptor: int, payload: bytes) -> int:
        """Swap the output after writing through the still-owned descriptor."""
        written = original_write(descriptor, payload)
        if output.exists():
            output.unlink()
            output.write_text("unrelated", encoding="utf-8")
            raise OSError("injected post-swap failure")
        return written

    monkeypatch.setattr(gateway_key_output.os, "write", replace_output)
    with pytest.raises(
        gateway_key_output.KeyOutputRecoveryError,
        match="changed during publication",
    ):
        gateway_key_output.deliver_key_output(output, raw_key, evidence)

    assert output.read_text(encoding="utf-8") == "unrelated"
    assert gateway_key_output.key_output_marker_path(output).exists()


def test_unknown_key_commit_emits_content_free_recovery_and_retains_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ambiguous commit preserves the secret file and emits an explicit status."""
    runner = CliRunner()
    manager = GatewayManagement(tmp_path)
    manager.initialize()
    manager.create_identity(identity_id="default", display_name="Default")
    output = tmp_path / "issued-key"
    prefix, raw_key = issue_key_material()
    issued = IssuedVirtualKey(
        key_id="key-one",
        organization_id="local",
        identity_id="default",
        prefix=prefix,
        raw_key=raw_key,
        expires_at=None,
        created_at=datetime.now(UTC),
    )

    def unknown_issue(
        _manager: GatewayManagement,
        *,
        secret_delivery: key_delivery.KeyDeliverySink | None = None,
        **_kwargs: object,
    ) -> IssuedVirtualKey:
        """Publish the secret and inject an inconclusive commit acknowledgement."""
        assert secret_delivery is not None
        secret_delivery(
            raw_key,
            key_delivery.KeyDeliveryEvidence(
                organization_id="local",
                identity_id="default",
                key_id="key-one",
                operation_id="operation-one",
                request_sha256=sha256_json(
                    {
                        "organization_id": "local",
                        "identity_id": "default",
                        "key_id": "key-one",
                        "expires_at": None,
                    }
                ),
                prefix=prefix,
                fingerprint_version=1,
                fingerprint_sha256="a" * 64,
                expires_at=None,
                created_at=issued.created_at.isoformat().replace("+00:00", "Z"),
            ),
        )
        raise OperationOutcomeUnknownError(issued=issued)

    monkeypatch.setattr(GatewayManagement, "issue_key", unknown_issue)
    result = runner.invoke(
        app,
        [
            "config",
            "gateway",
            "key",
            "issue",
            "default",
            "--key-id",
            "key-one",
            "--operation-id",
            "operation-one",
            "--root",
            str(tmp_path),
            "--output",
            str(output),
            "--json",
        ],
    )

    assert result.exit_code == 1
    receipt = json.loads(result.stdout)
    assert receipt["data"]["status"] == "operation_outcome_unknown"
    assert receipt["data"]["output_path"] == str(output)
    assert "raw_key" not in receipt["data"]
    assert raw_key not in result.stdout
    assert output.read_text(encoding="utf-8").strip() == raw_key
    assert output.stat().st_mode & 0o777 == 0o600
    assert gateway_key_output.key_output_marker_path(output).exists()


@pytest.mark.parametrize(
    "commit_error",
    (sqlite3.OperationalError, KeyboardInterrupt, SystemExit),
)
def test_definite_key_commit_failure_is_content_free_and_removes_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    commit_error: type[BaseException],
) -> None:
    """A proven rollback returns a stable CLI error without secret or SQLite detail.

    Args:
        tmp_path: Pytest-owned gateway root and key-output directory.
        monkeypatch: Scoped commit-failure injection.
    """
    runner = CliRunner()
    manager = GatewayManagement(tmp_path)
    manager.initialize()
    manager.create_identity(identity_id="default", display_name="Default")
    output = tmp_path / "issued-key"
    original_connect = SQLiteGatewayStore._connect

    class FailedCommitConnection:
        """Delegate transaction work while rejecting COMMIT before it takes effect."""

        def __init__(self, connection: sqlite3.Connection) -> None:
            """Wrap one configured SQLite connection."""
            self.connection = connection

        def execute(
            self,
            statement: str,
            parameters: tuple[object, ...] = (),
        ) -> sqlite3.Cursor:
            """Raise a secret-bearing SQLite error before the commit is applied."""
            if statement == "COMMIT":
                raise commit_error("injected database detail")
            return self.connection.execute(statement, parameters)

    @contextmanager
    def failed_commit(store: SQLiteGatewayStore) -> Iterator[FailedCommitConnection]:
        """Yield one connection whose transaction is proven absent after close."""
        with original_connect(store) as connection:
            yield FailedCommitConnection(connection)

    arguments = [
        "config",
        "gateway",
        "key",
        "issue",
        "default",
        "--key-id",
        "key-one",
        "--operation-id",
        "operation-one",
        "--root",
        str(tmp_path),
        "--output",
        str(output),
        "--json",
    ]
    with monkeypatch.context() as scoped:
        scoped.setattr(SQLiteGatewayStore, "_connect", failed_commit)
        result = runner.invoke(app, arguments)

    assert result.exit_code == 2
    error_output = result.output + result.stderr
    assert "virtual key issuance did not commit" in error_output
    assert "injected database detail" not in error_output
    assert "Traceback" not in error_output
    assert "exp_vk_" not in error_output
    assert not output.exists()
    assert tuple(tmp_path.glob(".issued-key.*.tmp")) == ()
    assert manager.keys() == ()
    with sqlite3.connect(manager.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM operation_receipts").fetchone()[0] == 0

    retried = runner.invoke(app, arguments)
    assert retried.exit_code == 0, retried.output
    assert output.read_text(encoding="utf-8").strip().startswith("exp_vk_")
