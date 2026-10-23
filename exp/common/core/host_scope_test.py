"""Tests for the single-host coordination boundary."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from exp.common.core import host_scope
from exp.common.core.host_scope import (
    MARKER_NAME,
    SINGLE_WRITER_ENV,
    HostScopeError,
    current_host,
    filesystem_type,
    parse_mount_types,
    require_host_scope,
    require_same_host,
)


@pytest.fixture(autouse=True)
def fresh_scope(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Give each test an empty verification cache and an uncached host identity."""
    monkeypatch.setattr(host_scope, "_verified", set())
    monkeypatch.delenv(SINGLE_WRITER_ENV, raising=False)
    current_host.cache_clear()
    yield
    current_host.cache_clear()


def become(monkeypatch: pytest.MonkeyPatch, machine: str, boot: str | None = "boot-a") -> None:
    """Make this process report the given raw machine and boot identifiers."""
    monkeypatch.setattr(host_scope, "_raw_identity", lambda: (machine, boot))
    monkeypatch.setattr(host_scope, "_verified", set())
    current_host.cache_clear()


def test_first_write_binds_the_directory_and_the_same_host_proceeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    become(monkeypatch, "machine-a")
    require_host_scope(tmp_path)
    marker = json.loads((tmp_path / MARKER_NAME).read_text())
    assert marker["machine"] == current_host().machine
    assert "machine-a" not in (tmp_path / MARKER_NAME).read_text(), "raw machine-id is secret"

    monkeypatch.setattr(host_scope, "_verified", set())
    require_host_scope(tmp_path)  # a fresh process on the same machine
    become(monkeypatch, "machine-a", boot="boot-b")
    require_host_scope(tmp_path)  # the same machine after a reboot still owns it


def test_a_foreign_host_is_refused_with_what_to_do(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    become(monkeypatch, "machine-a")
    require_host_scope(tmp_path)
    before = (tmp_path / MARKER_NAME).read_bytes()

    become(monkeypatch, "machine-b")
    with pytest.raises(HostScopeError, match="another machine") as refused:
        require_host_scope(tmp_path)
    assert "remote EnvironmentRuntime" in str(refused.value)
    assert "one machine" in str(refused.value)
    assert (tmp_path / MARKER_NAME).read_bytes() == before, "a refusal never rebinds"


def test_an_unrecognized_or_torn_marker_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    become(monkeypatch, "machine-a")
    monkeypatch.setattr(host_scope, "_CREATION_WINDOW_S", 0.02)
    (tmp_path / MARKER_NAME).write_bytes(b"")
    with pytest.raises(HostScopeError, match="unreadable"):
        require_host_scope(tmp_path)
    (tmp_path / MARKER_NAME).write_text('{"protocol": "other"}')
    with pytest.raises(HostScopeError, match="not an exp-host-scope-v1 record"):
        require_host_scope(tmp_path)


@pytest.mark.parametrize("kind", ["nfs", "nfs4", "cifs", "smb3", "fuse.sshfs", "9p"])
def test_network_filesystems_are_refused_unless_declared_single_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    become(monkeypatch, "machine-a")
    monkeypatch.setattr(host_scope, "filesystem_type", lambda _directory: kind)
    with pytest.raises(HostScopeError, match=f"{kind} network filesystem") as refused:
        require_host_scope(tmp_path)
    assert SINGLE_WRITER_ENV in str(refused.value)
    assert not (tmp_path / MARKER_NAME).exists()

    monkeypatch.setenv(SINGLE_WRITER_ENV, "1")
    require_host_scope(tmp_path)

    # The declaration waives only the filesystem check: another machine is still refused.
    become(monkeypatch, "machine-b")
    with pytest.raises(HostScopeError, match="another machine"):
        require_host_scope(tmp_path)


def test_local_filesystems_and_ambiguous_declarations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    become(monkeypatch, "machine-a")
    monkeypatch.setattr(host_scope, "filesystem_type", lambda _directory: "nfs")
    monkeypatch.setenv(SINGLE_WRITER_ENV, "yes")
    with pytest.raises(HostScopeError, match="must be 1 or unset"):
        require_host_scope(tmp_path)
    monkeypatch.setattr(host_scope, "filesystem_type", lambda _directory: "ext4")
    require_host_scope(tmp_path)


def test_mount_table_parsing_picks_the_innermost_latest_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mountinfo = "\n".join(
        (
            "22 1 8:1 / / rw,relatime shared:1 - ext4 /dev/sda1 rw",
            f"30 22 0:40 / {tmp_path}/share rw shared:2 - tmpfs tmpfs rw",
            f"31 22 0:41 / {tmp_path}/share rw - nfs4 server:/export rw,vers=4.2",
            f"32 22 0:42 / {tmp_path}/with\\040space rw - fuse.sshfs host:/ rw",
            "garbage line",
        )
    )
    assert parse_mount_types(mountinfo)[3] == (f"{tmp_path}/with space", "fuse.sshfs")
    table = tmp_path / "mountinfo"
    table.write_text(mountinfo)
    monkeypatch.setattr(host_scope, "_MOUNTINFO_PATH", table)
    (tmp_path / "share" / "project").mkdir(parents=True)
    (tmp_path / "with space").mkdir()
    (tmp_path / "local").mkdir()
    assert filesystem_type(tmp_path / "share" / "project") == "nfs4"
    assert filesystem_type(tmp_path / "with space") == "fuse.sshfs"
    assert filesystem_type(tmp_path / "local") == "ext4"
    monkeypatch.setattr(host_scope, "_MOUNTINFO_PATH", tmp_path / "absent")
    assert filesystem_type(tmp_path) is None


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux machine-id files")
def test_a_missing_machine_id_is_refused_with_how_to_create_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(host_scope, "_LINUX_MACHINE_ID_PATHS", (tmp_path / "machine-id",))
    with pytest.raises(HostScopeError, match="systemd-machine-id-setup"):
        current_host()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux machine-id files")
def test_linux_identity_reads_machine_and_boot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "machine-id").write_text("abc\n")
    (tmp_path / "boot_id").write_text("def\n")
    monkeypatch.setattr(host_scope, "_LINUX_MACHINE_ID_PATHS", (tmp_path / "machine-id",))
    monkeypatch.setattr(host_scope, "_LINUX_BOOT_ID_PATH", tmp_path / "boot_id")
    host = current_host()
    assert host.machine != "abc" and len(host.machine) == 32
    assert host.boot is not None and host.boot != host.machine


def test_leases_from_another_machine_are_never_judged(monkeypatch: pytest.MonkeyPatch) -> None:
    become(monkeypatch, "machine-a")
    require_same_host(current_host().machine, what="the lease")
    with pytest.raises(HostScopeError, match="the lease is held by a process on another machine"):
        require_same_host("0" * 32, what="the lease")
