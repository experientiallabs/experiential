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

import pytest
from click import unstyle
from typer.testing import CliRunner

from wmo.cli.app import app
from wmo.cli.gateway import key_output as gateway_key_output
from wmo.common.core.artifacts import sha256_json
from wmo.runtime.gateway.auth import IssuedVirtualKey, issue_key_material
from wmo.runtime.gateway.management import GatewayManagement
from wmo.runtime.gateway.sqlite import key_delivery
from wmo.runtime.gateway.sqlite.store import OperationOutcomeUnknownError, SQLiteGatewayStore


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
    assert raw_key.startswith("wmo_vk_")

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
    assert json.loads(usage.stdout)["schema_version"] == 1

    readiness = runner.invoke(
        app,
        ["run", "--root", str(tmp_path), "--check", "--non-interactive", "--json"],
    )
    assert readiness.exit_code == 0, readiness.output
    assert json.loads(readiness.stdout)["status"] == "ready"

    durable = b"".join(
        path.read_bytes() for path in (tmp_path / "gateway").rglob("*") if path.is_file()
    )
    assert raw_key.encode() not in durable


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
    assert raw_key.startswith("wmo_vk_")
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
    assert "wmo_vk_" not in failed.stdout
    assert not output.exists()
    assert tuple(tmp_path.glob(".issued-key.*.tmp")) == ()
    assert manager.keys() == ()
    connection = manager.require_initialized().database_path
    with sqlite3.connect(connection) as database:
        assert database.execute("SELECT COUNT(*) FROM operation_receipts").fetchone()[0] == 0

    retried = runner.invoke(app, arguments)
    assert retried.exit_code == 0, retried.output
    assert output.read_text(encoding="utf-8").strip().startswith("wmo_vk_")
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

from wmo.cli.gateway import key_output
from wmo.runtime.gateway.management import GatewayManagement

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
        if payload.startswith(b"wmo_vk_"):
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
    assert "wmo_vk_" not in error_output
    assert not output.exists()
    assert tuple(tmp_path.glob(".issued-key.*.tmp")) == ()
    assert manager.keys() == ()
    with sqlite3.connect(manager.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM operation_receipts").fetchone()[0] == 0

    retried = runner.invoke(app, arguments)
    assert retried.exit_code == 0, retried.output
    assert output.read_text(encoding="utf-8").strip().startswith("wmo_vk_")
