"""Capture CA persistence never follows links or exposes signing keys."""

import os
import stat
import subprocess
from pathlib import Path

import pytest
from cryptography import x509

from exp.runtime.capture.certificates import (
    certificate_is_trusted,
    prepare_certificate,
    trust_certificate,
)

_DOMAINS = ("api.openai.com", "chatgpt.com", "api.anthropic.com")


def test_certificate_reused_and_private(tmp_path: Path) -> None:
    """Repeated capture reuses one CA while its key stays owner-readable only."""
    directory = tmp_path / "certificates"
    certificate = prepare_certificate(directory)
    before = certificate.read_bytes()
    assert b"BEGIN CERTIFICATE" in before
    assert b"PRIVATE KEY" not in before
    parsed = x509.load_pem_x509_certificate(before)
    assert (
        parsed.subject.get_attributes_for_oid(x509.NameOID.COMMON_NAME)[0].value
        == "Experiential Capture"
    )
    assert (
        parsed.subject.get_attributes_for_oid(x509.NameOID.ORGANIZATION_NAME)[0].value
        == "Experiential Labs"
    )
    assert prepare_certificate(directory).read_bytes() == before
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE((directory / "mitmproxy-ca.pem").stat().st_mode) == 0o600


def test_linked_directory_is_rejected(tmp_path: Path) -> None:
    """A linked CA directory cannot redirect key creation outside its owner."""
    original = tmp_path / "original"
    original.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(original, target_is_directory=True)
    with pytest.raises(ValueError, match="user-owned directory"):
        prepare_certificate(linked)
    assert not list(original.iterdir())


def test_linked_ancestor_is_rejected_before_creating_files(tmp_path: Path) -> None:
    """A symlink above the CA directory is rejected before writing through it."""
    original = tmp_path / "original"
    original.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(original, target_is_directory=True)
    with pytest.raises(ValueError, match="without links"):
        prepare_certificate(linked / "capture" / "ca")
    assert not list(original.iterdir())


def test_linked_key_is_rejected_without_touching_target(tmp_path: Path) -> None:
    """An existing hard-linked key is rejected without changing its target."""
    original = tmp_path / "original"
    original.write_text("untouched")
    directory = tmp_path / "certificates"
    directory.mkdir()
    os.link(original, directory / "mitmproxy-ca.pem")
    with pytest.raises(ValueError, match="unsafe files"):
        prepare_certificate(directory)
    assert original.read_text() == "untouched"


def test_trust_is_one_user_scoped_command_with_every_provider_hostname(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Target only this CA and the user's keychain without broad or administrator trust."""
    certificate = prepare_certificate(tmp_path / "certificates")
    keychain = tmp_path / "custom user.keychain-db"
    keychain.write_text("unrelated-existing-keychain-content")
    calls: list[list[str]] = []
    leaves: list[Path] = []
    ca = x509.load_pem_x509_certificate(certificate.read_bytes())

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        """Inspect trust commands and actual signed leaves without changing macOS trust."""
        calls.append(command)
        assert command[0] == "/usr/bin/security"
        if command[1] == "default-keychain":
            assert command == ["/usr/bin/security", "default-keychain", "-d", "user"]
            return subprocess.CompletedProcess(command, 0, stdout=f'    "{keychain}"\n')
        if command[1] == "add-trusted-cert":
            assert command[-1] == str(certificate)
            assert command[command.index("-k") + 1] == str(keychain)
            assert "-d" not in command
            assert "basic" not in command
            assert command.count("-p") == 1
            assert command[command.index("-p") + 1] == "ssl"
            assert (
                tuple(command[index + 1] for index, value in enumerate(command) if value == "-s")
                == _DOMAINS
            )
            return subprocess.CompletedProcess(command, 0)
        assert command[1] == "verify-cert"
        assert "-r" not in command
        assert "-L" in command
        assert command[command.index("-p") + 1] == "ssl"
        host = command[command.index("-n") + 1]
        leaf = Path(command[command.index("-c") + 1])
        leaves.append(leaf)
        assert command[-2] == str(certificate)
        assert stat.S_IMODE(leaf.stat().st_mode) == 0o600
        assert stat.S_IMODE(leaf.parent.stat().st_mode) == 0o700
        assert b"PRIVATE KEY" not in leaf.read_bytes()
        parsed = x509.load_pem_x509_certificate(leaf.read_bytes())
        assert parsed.extensions.get_extension_for_class(
            x509.SubjectAlternativeName
        ).value.get_values_for_type(x509.DNSName) == [host]
        parsed.verify_directly_issued_by(ca)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("exp.runtime.capture.certificates.subprocess.run", run)
    trust_certificate(certificate, _DOMAINS)
    assert [call[1] for call in calls] == [
        "default-keychain",
        "add-trusted-cert",
        "verify-cert",
        "verify-cert",
        "verify-cert",
    ]
    assert all(not leaf.exists() for leaf in leaves)
    assert keychain.read_text() == "unrelated-existing-keychain-content"


def test_tls_trust_requires_every_domain_and_removes_temporary_leaf_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed hostname check remains untrusted and leaves no temporary certificate."""
    certificate = prepare_certificate(tmp_path / "certificates")
    hosts: list[str] = []
    leaves: list[Path] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        """Reject the second TLS hostname without touching the real trust evaluator."""
        assert command[1] == "verify-cert"
        hosts.append(command[command.index("-n") + 1])
        leaves.append(Path(command[command.index("-c") + 1]))
        return subprocess.CompletedProcess(command, int(len(hosts) == 2))

    monkeypatch.setattr("exp.runtime.capture.certificates.subprocess.run", run)
    assert not certificate_is_trusted(certificate, _DOMAINS)
    assert hosts == list(_DOMAINS[:2])
    assert all(not leaf.exists() for leaf in leaves)
    assert not certificate_is_trusted(certificate, ())


@pytest.mark.parametrize(
    "keychain_output",
    ["", '"relative.keychain"', '"/Library/Keychains/System.keychain"', '"/missing.keychain"'],
)
def test_unsafe_default_keychain_never_installs_trust(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keychain_output: str
) -> None:
    """Do not turn an absent or system keychain preference into a wider trust mutation."""
    certificate = prepare_certificate(tmp_path / "certificates")
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        """Return only the unsafe default-keychain preference under test."""
        calls.append(command)
        assert command[1] == "default-keychain"
        return subprocess.CompletedProcess(command, 0, stdout=keychain_output)

    monkeypatch.setattr("exp.runtime.capture.certificates.subprocess.run", run)
    with pytest.raises(ValueError, match="keychain"):
        trust_certificate(certificate, _DOMAINS)
    assert len(calls) == 1


def test_declined_user_trust_does_not_broaden_or_retry_privileged_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A declined native prompt fails without falling back to administrator trust."""
    certificate = prepare_certificate(tmp_path / "certificates")
    keychain = tmp_path / "login.keychain-db"
    keychain.touch()
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        """Accept the read-only preference lookup, then decline the one trust operation."""
        calls.append(command)
        if command[1] == "default-keychain":
            return subprocess.CompletedProcess(command, 0, stdout=f'"{keychain}"')
        assert command[1] == "add-trusted-cert"
        assert "-d" not in command
        assert "sudo" not in command
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr("exp.runtime.capture.certificates.subprocess.run", run)
    with pytest.raises(ValueError, match="trust was not installed"):
        trust_certificate(certificate, _DOMAINS)
    assert len(calls) == 2
