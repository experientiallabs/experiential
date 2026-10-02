"""Exercise local-backend diagnostics using only synthetic app archives and paths."""

from __future__ import annotations

import io
import os
import plistlib
import select
import stat
import subprocess
import sys
import tarfile
from importlib.metadata import Distribution, PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from exp.runtime.capture import local_backend


def _archive(
    directory: Path, *, app_version: str = "12.0", extension_version: str = "12.0"
) -> Path:
    """Create metadata-only dependency packaging without an executable or real extension."""
    package = directory / "mitmproxy_macos"
    package.mkdir()
    archive = package / "Mitmproxy Redirector.app.tar"
    with tarfile.open(archive, "w:") as bundle:
        for name, version in (
            (local_backend._APP_PLIST, app_version),
            (local_backend._EXTENSION_PLIST, extension_version),
        ):
            payload = plistlib.dumps({"LSMinimumSystemVersion": version})
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            bundle.addfile(member, io.BytesIO(payload))
    return archive


@pytest.fixture
def codesign(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Record signature verification commands without invoking macOS tools in unit tests."""
    command = Mock(return_value=subprocess.CompletedProcess(["codesign"], 0, "", ""))
    monkeypatch.setattr(local_backend.subprocess, "run", command)
    return command


@pytest.fixture
def backend_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, codesign: Mock) -> Path:
    """Limit every preflight path to the test directory and report a normal macOS user."""
    applications = tmp_path / "Applications"
    applications.mkdir()
    monkeypatch.setattr(local_backend, "_APPLICATIONS", applications)
    monkeypatch.setattr(local_backend.sys, "platform", "darwin")
    monkeypatch.setattr(local_backend.os, "geteuid", lambda: 501)
    monkeypatch.setattr(local_backend.platform, "mac_ver", lambda: ("15.0", ("", "", ""), ""))
    monkeypatch.delenv("MITMPROXY_KEEP_REDIRECTOR", raising=False)
    package = Distribution.at(tmp_path / "mitmproxy_macos-0.12.11.dist-info")
    monkeypatch.setattr(local_backend, "distribution", lambda name: package)
    return applications


def test_preflight_does_not_install_or_activate(backend_environment: Path, tmp_path: Path) -> None:
    """A valid packaged dependency passes without extracting an app or changing any file."""
    _archive(tmp_path)
    before = {path: path.stat().st_mtime_ns for path in tmp_path.rglob("*")}
    local_backend.require_local_backend(installation_is_current=lambda: False)
    assert before == {path: path.stat().st_mtime_ns for path in tmp_path.rglob("*")}
    assert list(backend_environment.iterdir()) == []


def test_preflight_rejects_other_platform_before_package_lookup(
    backend_environment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unsupported systems receive the product boundary, not a missing dependency error."""
    monkeypatch.setattr(local_backend.sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="macOS only"):
        local_backend.require_local_backend(installation_is_current=lambda: False)


def test_preflight_rejects_root(backend_environment: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Credential-bearing Capture must remain in the normal user's process."""
    monkeypatch.setattr(local_backend.os, "geteuid", lambda: 0)
    with pytest.raises(RuntimeError, match="normal user, not with sudo"):
        local_backend.require_local_backend(installation_is_current=lambda: False)


def test_missing_dependency_explains_reinstallation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing wheel metadata produces an actionable dependency-specific error."""

    def missing(name: str) -> Distribution:
        """Model an environment where the macOS dependency was not installed."""
        raise PackageNotFoundError(name)

    monkeypatch.setattr(local_backend, "distribution", missing)
    with pytest.raises(RuntimeError, match="Reinstall Experiential.*mitmproxy-macos"):
        local_backend._packaged_archive()


def test_missing_archive_explains_reinstallation(backend_environment: Path) -> None:
    """Installed metadata without the packaged app does not count as readiness."""
    with pytest.raises(RuntimeError, match="redirector package is missing or invalid"):
        local_backend.require_local_backend(installation_is_current=lambda: False)


@pytest.mark.parametrize("content", [b"invalid tar", b"\0" * 10240])
def test_invalid_or_empty_archive_is_rejected(tmp_path: Path, content: bytes) -> None:
    """Corrupt archives and archives without the app metadata fail before installation."""
    archive = tmp_path / "invalid.tar"
    archive.write_bytes(content)
    with pytest.raises(RuntimeError, match="Reinstall Experiential"):
        local_backend._minimum_macos(archive)


@pytest.mark.parametrize("content", [b"invalid plist", b"<?xml version='1.0'?><plist><"])
def test_malformed_plist_explains_reinstallation(tmp_path: Path, content: bytes) -> None:
    """Invalid binary or XML metadata produces the same actionable package diagnostic."""
    archive = tmp_path / "malformed.tar"
    with tarfile.open(archive, "w:") as bundle:
        member = tarfile.TarInfo(local_backend._APP_PLIST)
        member.size = len(content)
        bundle.addfile(member, io.BytesIO(content))
    with pytest.raises(RuntimeError, match="Reinstall Experiential"):
        local_backend._minimum_macos(archive)


@pytest.mark.parametrize("version", ["", "twelve", "12.-1", "99999999999999"])
def test_invalid_deployment_target_is_rejected(tmp_path: Path, version: str) -> None:
    """Malformed deployment metadata is not silently treated as compatible."""
    archive = _archive(tmp_path, extension_version=version)
    with pytest.raises(RuntimeError, match="Reinstall Experiential"):
        local_backend._minimum_macos(archive)


def test_extension_minimum_is_enforced(
    backend_environment: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The newer of app and extension requirements controls platform compatibility."""
    _archive(tmp_path, app_version="12.0", extension_version="13.1")
    monkeypatch.setattr(local_backend.platform, "mac_ver", lambda: ("13.0", ("", "", ""), ""))
    with pytest.raises(RuntimeError, match="macOS 13.1 or newer"):
        local_backend.require_local_backend(installation_is_current=lambda: False)


def test_minimum_os_boundary_passes(
    backend_environment: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact packaged deployment target is accepted."""
    _archive(tmp_path)
    monkeypatch.setattr(local_backend.platform, "mac_ver", lambda: ("12.0", ("", "", ""), ""))
    local_backend.require_local_backend(installation_is_current=lambda: False)


def test_unmanaged_redirector_override_is_explicit(
    backend_environment: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An upstream developer override cannot silently bypass package management checks."""
    _archive(tmp_path)
    monkeypatch.setenv("MITMPROXY_KEEP_REDIRECTOR", "1")
    with pytest.raises(RuntimeError, match="Unset MITMPROXY_KEEP_REDIRECTOR"):
        local_backend.require_local_backend(installation_is_current=lambda: False)


def test_install_permission_denied_is_actionable(
    backend_environment: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A nonwritable Applications directory fails before login or system authorization."""
    _archive(tmp_path)
    monkeypatch.setattr(local_backend, "_writable_directory", lambda path: False)
    with pytest.raises(RuntimeError, match="Ask your administrator.*normal user"):
        local_backend.require_local_backend(installation_is_current=lambda: False)
    assert list(backend_environment.iterdir()) == []


def _installed_app(applications: Path, archive: Path, *, current: bool) -> Path:
    """Create a fake bundle with a chosen installation timestamp."""
    app = applications / local_backend._APP_NAME
    contents = app / "Contents"
    (contents / "MacOS").mkdir(parents=True)
    plist = contents / "Info.plist"
    plist.write_bytes(b"metadata")
    timestamp = archive.stat().st_mtime_ns + (0 if current else 1)
    os.utime(plist, ns=(timestamp, timestamp))
    executable = contents / "MacOS/Mitmproxy Redirector"
    executable.write_bytes(b"not a real executable")
    executable.chmod(0o700)
    return app


def test_current_install_with_different_timestamp_does_not_require_update_permission(
    backend_environment: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exact reusable app can run even when the user cannot replace /Applications apps."""
    archive = _archive(tmp_path)
    app = _installed_app(backend_environment, archive, current=True)
    os.utime(app / "Contents/Info.plist", ns=(1, 1))
    monkeypatch.setattr(local_backend, "_writable_directory", lambda path: False)
    local_backend.require_local_backend(installation_is_current=lambda: True)


def test_outdated_install_requires_app_replacement_access(
    backend_environment: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Writable Applications alone is insufficient to replace a protected old app."""
    archive = _archive(tmp_path)
    _installed_app(backend_environment, archive, current=False)
    monkeypatch.setattr(
        local_backend, "_writable_directory", lambda path: path == backend_environment
    )
    with pytest.raises(RuntimeError, match="permission to install or update"):
        local_backend.require_local_backend(installation_is_current=lambda: False)


def test_current_install_without_executable_fails(
    backend_environment: Path, tmp_path: Path
) -> None:
    """The installed app must be executable before its signed contents can be reused."""
    archive = _archive(tmp_path)
    app = _installed_app(backend_environment, archive, current=True)
    (app / "Contents/MacOS/Mitmproxy Redirector").unlink()
    with pytest.raises(RuntimeError, match="installed Mitmproxy Redirector app is incomplete"):
        local_backend.require_local_backend(installation_is_current=lambda: True)


def test_signature_verification_requires_exact_developer_and_bundles(
    tmp_path: Path, codesign: Mock
) -> None:
    """The verifier checks cryptographic identity, integrity, and both binary architectures."""
    app = tmp_path / local_backend._APP_NAME
    local_backend._verify_bundle_signature(app)
    assert codesign.call_count == 2
    for call, bundle, identifier in zip(
        codesign.call_args_list,
        (app, app / local_backend._EXTENSION_PATH),
        (local_backend._APP_IDENTIFIER, local_backend._EXTENSION_IDENTIFIER),
        strict=True,
    ):
        command = call.args[0]
        assert command[:6] == [
            "/usr/bin/codesign",
            "--verify",
            "--strict",
            "--deep",
            "--all-architectures",
            "--test-requirement",
        ]
        assert command[6] == (
            "=anchor apple generic and certificate 1[field.1.2.840.113635.100.6.2.6] exists "
            "and certificate leaf[field.1.2.840.113635.100.6.1.13] exists "
            'and certificate leaf[subject.OU] = "S8XHQB96PW" '
            f'and identifier "{identifier}"'
        )
        assert command[7] == str(bundle)
        assert call.kwargs["timeout"] == 15
        assert call.kwargs["env"] == {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}
        assert call.kwargs["stdin"] == subprocess.DEVNULL


@pytest.mark.parametrize("exit_code", [1, 3])
def test_tampered_or_wrong_identity_bundle_stops_preflight(
    backend_environment: Path, tmp_path: Path, codesign: Mock, exit_code: int
) -> None:
    """A damaged signature or valid signature from the wrong identity cannot reach install."""
    _archive(tmp_path)
    codesign.return_value = subprocess.CompletedProcess(["codesign"], exit_code, "", "invalid")
    with pytest.raises(RuntimeError, match="rejected.*signature"):
        local_backend.require_local_backend(installation_is_current=lambda: False)
    assert codesign.call_count == 1
    assert list(backend_environment.iterdir()) == []
    assert not Path(codesign.call_args.args[0][-1]).exists()


def test_extension_identity_mismatch_is_rejected(tmp_path: Path, codesign: Mock) -> None:
    """A valid outer app does not excuse a nested extension signed by another developer."""
    codesign.side_effect = [
        subprocess.CompletedProcess(["codesign"], 0, "", ""),
        subprocess.CompletedProcess(["codesign"], 3, "", "requirement failed"),
    ]
    with pytest.raises(RuntimeError, match="rejected.*signature"):
        local_backend._verify_bundle_signature(tmp_path / local_backend._APP_NAME)
    assert codesign.call_count == 2


@pytest.mark.parametrize("failure", [OSError("missing"), subprocess.TimeoutExpired("codesign", 15)])
def test_verifier_unavailable_fails_with_remedy(
    tmp_path: Path, codesign: Mock, failure: Exception
) -> None:
    """Missing or stalled OS verification cannot count as a successful authenticity check."""
    codesign.side_effect = failure
    with pytest.raises(RuntimeError, match="Check that /usr/bin/codesign works"):
        local_backend._verify_bundle_signature(tmp_path / local_backend._APP_NAME)


def test_fresh_install_verifies_private_archive_copy(
    backend_environment: Path, tmp_path: Path, codesign: Mock
) -> None:
    """Verification precedes the native installer without publishing an app ourselves."""
    _archive(tmp_path)
    local_backend.require_local_backend(installation_is_current=lambda: False)
    assert codesign.call_count == 2
    staged = Path(codesign.call_args_list[0].args[0][-1])
    assert staged.name == local_backend._APP_NAME
    assert staged.parent.name.startswith("exp-capture-verify-")
    assert not staged.parent.exists()
    assert list(backend_environment.iterdir()) == []


def test_reused_install_still_requires_valid_signature(
    backend_environment: Path, tmp_path: Path, codesign: Mock
) -> None:
    """Matching packaged contents do not replace checking the installed app's signature."""
    archive = _archive(tmp_path)
    app = _installed_app(backend_environment, archive, current=True)
    codesign.side_effect = [
        subprocess.CompletedProcess(["codesign"], 0, "", ""),
        subprocess.CompletedProcess(["codesign"], 0, "", ""),
        subprocess.CompletedProcess(["codesign"], 1, "", "modified executable"),
    ]
    with pytest.raises(RuntimeError, match="rejected.*signature"):
        local_backend.require_local_backend(installation_is_current=lambda: True)
    assert codesign.call_count == 3
    assert codesign.call_args.args[0][-1] == str(app)
    assert app.exists()


def _member_archive(path: Path, members: list[tarfile.TarInfo]) -> Path:
    """Create a small archive with explicit entries for adversarial extraction tests."""
    with tarfile.open(path, "w:") as bundle:
        for member in members:
            bundle.addfile(member, io.BytesIO(b"x" * member.size))
    return path


@pytest.mark.parametrize(
    "name",
    [
        "/tmp/escape",
        "../escape",
        "Other.app/Contents/file",
        f"{local_backend._APP_NAME}/../escape",
        f"{local_backend._APP_NAME}/./file",
        f"{local_backend._APP_NAME}//file",
        f"{local_backend._APP_NAME}/dir\\file",
        local_backend._APP_NAME,
    ],
)
def test_archive_rejects_paths_outside_single_app(
    tmp_path: Path, codesign: Mock, name: str
) -> None:
    """Traversal, ambiguous paths, and unexpected app roots fail before signature checks."""
    archive = _member_archive(tmp_path / "unsafe.tar", [tarfile.TarInfo(name)])
    with pytest.raises(RuntimeError, match="package is missing or invalid"):
        local_backend._verify_packaged_app(archive)
    codesign.assert_not_called()


@pytest.mark.parametrize(
    "kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE]
)
def test_archive_rejects_links_and_special_files(
    tmp_path: Path, codesign: Mock, kind: bytes
) -> None:
    """Plain-file verification cannot be redirected through filesystem links or devices."""
    member = tarfile.TarInfo(f"{local_backend._APP_NAME}/Contents/file")
    member.type = kind
    member.linkname = "/tmp/escape"
    archive = _member_archive(tmp_path / "unsafe.tar", [member])
    with pytest.raises(RuntimeError, match="package is missing or invalid"):
        local_backend._verify_packaged_app(archive)
    codesign.assert_not_called()


@pytest.mark.parametrize("mode", [0o666, 0o775, 0o4755, 0o2755])
def test_archive_rejects_shared_or_privileged_modes(
    tmp_path: Path, codesign: Mock, mode: int
) -> None:
    """Native installation must not later publish group-writable or set-ID executables."""
    member = tarfile.TarInfo(f"{local_backend._APP_NAME}/Contents/file")
    member.mode = mode
    archive = _member_archive(tmp_path / "unsafe.tar", [member])
    with pytest.raises(RuntimeError, match="package is missing or invalid"):
        local_backend._verify_packaged_app(archive)
    codesign.assert_not_called()


def test_archive_rejects_duplicate_entries(tmp_path: Path, codesign: Mock) -> None:
    """Extraction cannot conceal a later payload behind an earlier file of the same name."""
    member = tarfile.TarInfo(f"{local_backend._APP_NAME}/Contents/file")
    archive = _member_archive(tmp_path / "duplicates.tar", [member, member])
    with pytest.raises(RuntimeError, match="package is missing or invalid"):
        local_backend._verify_packaged_app(archive)
    codesign.assert_not_called()


def test_archive_member_count_is_bounded(
    tmp_path: Path, codesign: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Many tiny entries cannot bypass the extraction work limit."""
    monkeypatch.setattr(local_backend, "_ARCHIVE_MAX_MEMBERS", 2)
    members = [tarfile.TarInfo(f"{local_backend._APP_NAME}/{index}") for index in range(3)]
    archive = _member_archive(tmp_path / "many.tar", members)
    with pytest.raises(RuntimeError, match="package is missing or invalid"):
        local_backend._verify_packaged_app(archive)
    codesign.assert_not_called()


def test_archive_bytes_are_bounded(
    tmp_path: Path, codesign: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Oversized packaged archives fail without unpacking or running the verifier."""
    archive = _archive(tmp_path)
    monkeypatch.setattr(local_backend, "_ARCHIVE_MAX_BYTES", archive.stat().st_size - 1)
    with pytest.raises(RuntimeError, match="package is missing or invalid"):
        local_backend._verify_packaged_app(archive)
    codesign.assert_not_called()


def test_verification_copy_is_private_and_preserves_executable_bits(tmp_path: Path) -> None:
    """Temporary inspection uses owner-only permissions without changing source contents."""
    member = tarfile.TarInfo(f"{local_backend._APP_NAME}/Contents/MacOS/redirector")
    member.mode = 0o755
    member.size = 4
    archive = _member_archive(tmp_path / "app.tar", [member])
    destination = tmp_path / "staging"
    destination.mkdir(mode=0o700)
    local_backend._extract_verification_bundle(archive, destination)
    executable = destination / member.name
    assert executable.read_bytes() == b"xxxx"
    assert stat.S_IMODE(executable.stat().st_mode) == 0o700
    assert stat.S_IMODE(executable.parent.stat().st_mode) == 0o700


@pytest.fixture
def capture_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Replace only the lock's home with a private synthetic directory."""
    home = tmp_path.resolve() / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setattr(local_backend, "_account_home", lambda: home)
    return home


def test_account_home_ignores_environment_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only the effective UID's account database determines the foreground lock home."""
    home = tmp_path / "account-home"
    lookup = Mock(return_value=SimpleNamespace(pw_dir=str(home)))
    monkeypatch.setattr(local_backend, "pwd", SimpleNamespace(getpwuid=lookup))
    monkeypatch.setenv("HOME", str(tmp_path / "environment-home"))
    assert local_backend._account_home() == home
    lookup.assert_called_once_with(os.geteuid())
    assert not home.exists()


def test_missing_account_home_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing account record cannot fall back to an environment-controlled home."""
    lookup = Mock(side_effect=KeyError("account does not exist"))
    monkeypatch.setattr(local_backend, "pwd", SimpleNamespace(getpwuid=lookup))
    with pytest.raises(RuntimeError, match="could not resolve.*account home"):
        local_backend._account_home()


def test_foreground_lock_is_private_and_reused(capture_home: Path) -> None:
    """Release keeps one private inode so later processes cannot lock competing files."""
    path = capture_home / "Library/Application Support/exp/capture/foreground.lock"
    with local_backend.capture_instance():
        metadata = path.stat()
        assert stat.S_IMODE(metadata.st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert path.is_file()
    with local_backend.capture_instance():
        assert path.stat().st_ino == metadata.st_ino


def test_foreground_lock_releases_after_body_exception(capture_home: Path) -> None:
    """Application errors retain their meaning and still release ownership."""
    with pytest.raises(OSError, match="application failed"):
        with local_backend.capture_instance():
            raise OSError("application failed")
    with local_backend.capture_instance():
        pass


_LOCK_CHILD = """
import sys
from pathlib import Path
from unittest.mock import patch
from exp.runtime.capture.local_backend import capture_instance

with patch('exp.runtime.capture.local_backend._account_home', return_value=Path(sys.argv[1])):
    try:
        with capture_instance():
            sys.stdout.write('locked\\n')
            sys.stdout.flush()
            if sys.argv[2] == 'hold':
                sys.stdin.read()
    except RuntimeError as exc:
        sys.stderr.write(str(exc))
        sys.exit(23)
"""


@pytest.mark.parametrize("variable", ["HOME", "XDG_DATA_HOME"])
def test_foreground_lock_rejects_other_process_and_profile(
    capture_home: Path, tmp_path: Path, variable: str
) -> None:
    """HOME and XDG overrides still contend for the same account's actual OS lock."""
    with local_backend.capture_instance():
        child = subprocess.run(
            [sys.executable, "-c", _LOCK_CHILD, str(capture_home), "attempt"],
            cwd=Path(local_backend.__file__).parents[3],
            env={**os.environ, variable: str(tmp_path / "another-profile")},
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    assert child.returncode == 23
    assert "Capture is already running for this macOS user" in child.stderr
    assert not (tmp_path / "another-profile").exists()


def test_foreground_lock_releases_after_process_crash(capture_home: Path) -> None:
    """SIGKILL releases kernel ownership without requiring reset or deleting the file."""
    child = subprocess.Popen(
        [sys.executable, "-c", _LOCK_CHILD, str(capture_home), "hold"],
        cwd=Path(local_backend.__file__).parents[3],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert select.select([child.stdout], [], [], 10)[0], "lock child did not become ready"
        assert child.stdout.readline() == "locked\n"
        with pytest.raises(RuntimeError, match="Capture is already running"):
            with local_backend.capture_instance():
                pytest.fail("the other process already owns Capture")
        path = capture_home / "Library/Application Support/exp/capture/foreground.lock"
        inode = path.stat().st_ino
        child.kill()
        child.wait(timeout=10)
        with local_backend.capture_instance():
            assert path.stat().st_ino == inode
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=10)


@pytest.mark.parametrize("component", ["Library", "Library/Application Support/exp/capture"])
def test_foreground_lock_rejects_symlink_directory(
    capture_home: Path, tmp_path: Path, component: str
) -> None:
    """No file is created through a substituted application or Library directory."""
    target = tmp_path / "redirected"
    target.mkdir()
    link = capture_home / component
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="directory is unsafe"):
        with local_backend.capture_instance():
            pytest.fail("symlink directory was accepted")
    assert list(target.iterdir()) == []


def test_foreground_lock_rejects_symlink_home_ancestor(
    capture_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checking the leaf home alone cannot permit an ancestor that redirects its path."""
    link = tmp_path / "redirected"
    link.symlink_to(capture_home.parent, target_is_directory=True)
    monkeypatch.setattr(local_backend, "_account_home", lambda: link / "home")
    with pytest.raises(RuntimeError, match="unsafe ancestor"):
        with local_backend.capture_instance():
            pytest.fail("symlink ancestor was accepted")
    assert list(capture_home.iterdir()) == []


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "directory", "shared"])
def test_foreground_lock_rejects_unsafe_file(capture_home: Path, tmp_path: Path, kind: str) -> None:
    """Reject aliases and shared files before FileLock can truncate or chmod a target."""
    path = local_backend._foreground_lock_path()
    target = tmp_path / "unrelated"
    target.write_text("preserve")
    if kind == "symlink":
        path.symlink_to(target)
    elif kind == "hardlink":
        path.hardlink_to(target)
    elif kind == "directory":
        path.mkdir()
    else:
        path.write_text("shared")
        path.chmod(0o666)
    with pytest.raises(RuntimeError, match="foreground lock is unsafe"):
        with local_backend.capture_instance():
            pytest.fail("unsafe lock file was accepted")
    assert target.read_text() == "preserve"


def test_foreground_lock_rejects_shared_parent(capture_home: Path) -> None:
    """A writable parent would allow another user to replace the private lock directory."""
    library = capture_home / "Library"
    library.mkdir(mode=0o777)
    library.chmod(0o777)
    with pytest.raises(RuntimeError, match="directory is unsafe"):
        with local_backend.capture_instance():
            pytest.fail("shared parent was accepted")
    assert list(library.iterdir()) == []


def test_foreground_lock_rejects_foreign_home_owner(
    capture_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A different effective user cannot use or normalize this user's capture directory."""
    monkeypatch.setattr(local_backend.os, "geteuid", lambda: capture_home.stat().st_uid + 1)
    with pytest.raises(RuntimeError, match="directory is unsafe"):
        with local_backend.capture_instance():
            pytest.fail("foreign home was accepted")
    assert list(capture_home.iterdir()) == []


def test_foreground_lock_normalizes_only_capture_directory(capture_home: Path) -> None:
    """Legacy readable directories become private without chmodding the user's Library."""
    directory = capture_home / "Library/Application Support/exp/capture"
    directory.mkdir(parents=True)
    directory.chmod(0o755)
    library = capture_home / "Library"
    library.chmod(0o755)
    with local_backend.capture_instance():
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
        assert stat.S_IMODE(library.stat().st_mode) == 0o755
