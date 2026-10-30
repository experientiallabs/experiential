"""Tests for the single-host coordination boundary."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from exp.common.core import host_scope
from exp.common.core.host_scope import (
    MACHINE_ID_ENV,
    MARKER_NAME,
    SINGLE_WRITER_ENV,
    HostScopeError,
    current_host,
    filesystem_type,
    generated_machine_id_path,
    parse_mount_types,
    require_host_scope,
    require_same_host,
)

ACCOUNT_HOME = host_scope._account_home


@pytest.fixture(autouse=True)
def fresh_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Give each test an empty verification cache, an uncached host identity and own state."""
    monkeypatch.setattr(host_scope, "_account_home", lambda: tmp_path / "account")
    monkeypatch.setattr(host_scope, "_verified", set())
    monkeypatch.delenv(SINGLE_WRITER_ENV, raising=False)
    monkeypatch.delenv(MACHINE_ID_ENV, raising=False)
    current_host.cache_clear()
    yield
    current_host.cache_clear()


def become(
    monkeypatch: pytest.MonkeyPatch, machine: str | None, boot: str | None = "boot-a"
) -> None:
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


def without_machine_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, user: str) -> None:
    """Make this process a Linux host with no machine-id whose account home is ``user``."""
    monkeypatch.setattr(host_scope, "_LINUX_MACHINE_ID_PATHS", (tmp_path / "no-machine-id",))
    monkeypatch.setattr(host_scope, "_account_home", lambda: tmp_path / user)
    monkeypatch.setattr(host_scope, "_verified", set())
    current_host.cache_clear()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux machine-id files")
def test_a_host_without_machine_id_generates_one_stable_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    without_machine_id(monkeypatch, tmp_path, "user-a")
    script = (
        "import sys; from pathlib import Path; from exp.common.core import host_scope; "
        "host_scope._LINUX_MACHINE_ID_PATHS = (Path(sys.argv[1]),); "
        "host_scope._account_home = lambda: Path(sys.argv[2]); "
        "print(host_scope.current_host().machine)"
    )
    other_process = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "no-machine-id"), str(tmp_path / "user-a")],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    machine = current_host().machine
    current_host.cache_clear()
    assert current_host().machine == machine == other_process

    stored = generated_machine_id_path()
    assert stored == tmp_path / "user-a" / ".local" / "state" / "exp" / "machine-id"
    assert stored.stat().st_mode & 0o777 == 0o600
    raw = stored.read_text()
    assert raw not in machine, "the raw identifier is never persisted beside locks"
    assert machine != host_scope._digest("machine", raw), "generated ids have their own kind"
    require_host_scope(tmp_path)


def test_the_machine_id_override_wins_over_platform_and_generated_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    become(monkeypatch, "platform-id")
    platform = current_host().machine
    monkeypatch.setenv(MACHINE_ID_ENV, "pinned-id")
    current_host.cache_clear()
    pinned = current_host().machine
    assert pinned == host_scope._digest("machine", "pinned-id") != platform
    assert "pinned-id" not in pinned

    become(monkeypatch, None)
    assert current_host().machine == pinned
    assert not generated_machine_id_path().exists(), "the override needs no generated id"

    # Pinning a container to its host's machine-id makes the two one machine.
    monkeypatch.setenv(MACHINE_ID_ENV, "platform-id")
    current_host.cache_clear()
    assert current_host().machine == platform


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX account database")
def test_the_generated_identity_ignores_environment_selected_profiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pwd

    monkeypatch.setenv("HOME", str(tmp_path / "another-profile"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "another-state"))
    assert ACCOUNT_HOME() == Path(pwd.getpwuid(os.geteuid()).pw_dir)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux machine-id files")
def test_two_generated_identities_on_one_marker_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    without_machine_id(monkeypatch, tmp_path, "container-1")
    require_host_scope(project)
    before = (project / MARKER_NAME).read_bytes()

    without_machine_id(monkeypatch, tmp_path, "container-2")  # a recreated container
    with pytest.raises(HostScopeError, match="another machine") as refused:
        require_host_scope(project)
    assert MACHINE_ID_ENV in str(refused.value)
    assert (project / MARKER_NAME).read_bytes() == before, "a refusal never rebinds"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux machine-id files")
def test_an_unusable_generated_identity_names_the_override_and_the_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    without_machine_id(monkeypatch, tmp_path, "not-a-directory")
    (tmp_path / "not-a-directory").write_text("")
    with pytest.raises(HostScopeError, match="cannot be created or read") as refused:
        current_host()
    assert MACHINE_ID_ENV in str(refused.value)
    assert str(generated_machine_id_path()) in str(refused.value)

    without_machine_id(monkeypatch, tmp_path, "torn")
    monkeypatch.setattr(host_scope, "_CREATION_WINDOW_S", 0.02)
    generated_machine_id_path().parent.mkdir(parents=True)
    generated_machine_id_path().write_text("")
    with pytest.raises(HostScopeError, match="is empty") as refused:
        current_host()
    assert MACHINE_ID_ENV in str(refused.value)


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
