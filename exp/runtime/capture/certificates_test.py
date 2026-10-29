"""Capture CA persistence never follows links or exposes signing keys."""

import datetime
import ipaddress
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from mitmproxy.certs import CertStore

from exp.runtime.capture import certificates
from exp.runtime.capture.certificates import (
    capture_certificate_directory,
    certificate_is_trusted,
    prepare_certificate,
    trust_certificate,
)

_DOMAINS = ("api.openai.com", "chatgpt.com", "api.anthropic.com")


def test_certificate_reused_and_private(tmp_path: Path) -> None:
    """Repeated capture reuses one CA while its key stays owner-readable only."""
    directory = tmp_path / "certificates"
    certificate = prepare_certificate(directory, _DOMAINS)
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
    assert prepare_certificate(directory, _DOMAINS).read_bytes() == before
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE((directory / "mitmproxy-ca.pem").stat().st_mode) == 0o600


def test_linked_directory_is_rejected(tmp_path: Path) -> None:
    """A linked CA directory cannot redirect key creation outside its owner."""
    original = tmp_path / "original"
    original.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(original, target_is_directory=True)
    with pytest.raises(ValueError, match="user-owned directory"):
        prepare_certificate(linked, _DOMAINS)
    assert not list(original.iterdir())


def test_linked_ancestor_is_rejected_before_creating_files(tmp_path: Path) -> None:
    """A symlink above the CA directory is rejected before writing through it."""
    original = tmp_path / "original"
    original.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(original, target_is_directory=True)
    with pytest.raises(ValueError, match="without links"):
        prepare_certificate(linked / "capture" / "ca", _DOMAINS)
    assert not list(original.iterdir())


def test_linked_key_is_rejected_without_touching_target(tmp_path: Path) -> None:
    """An existing hard-linked key is rejected without changing its target."""
    original = tmp_path / "original"
    original.write_text("untouched")
    directory = tmp_path / "certificates"
    directory.mkdir()
    os.link(original, directory / "mitmproxy-ca.pem")
    with pytest.raises(ValueError, match="unsafe files"):
        prepare_certificate(directory, _DOMAINS)
    assert original.read_text() == "untouched"


def test_trust_is_one_user_ssl_command_with_certificate_enforced_hostname_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trust only this constrained CA in the user keychain using Chromium-compatible SSL policy."""
    certificate = prepare_certificate(tmp_path / "certificates", _DOMAINS)
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
            assert "-s" not in command
            constraints = ca.extensions.get_extension_for_class(x509.NameConstraints)
            assert constraints.critical
            assert constraints.value.permitted_subtrees == [
                x509.DNSName(host) for host in sorted(_DOMAINS)
            ]
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
    certificate = prepare_certificate(tmp_path / "certificates", _DOMAINS)
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
    assert hosts == sorted(_DOMAINS)[:2]
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
    certificate = prepare_certificate(tmp_path / "certificates", _DOMAINS)
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
    certificate = prepare_certificate(tmp_path / "certificates", _DOMAINS)
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


def test_scopes_have_distinct_immutable_keys_and_leave_old_ca_untouched(tmp_path: Path) -> None:
    """Scope order is stable; changing hosts never repurposes another or an old CA key."""
    old = tmp_path / "ca"
    CertStore.create_store(old, "mitmproxy", 2048)
    before = {path.name: path.read_bytes() for path in old.iterdir()}
    directory = capture_certificate_directory(tmp_path, _DOMAINS)
    assert directory.parent == tmp_path / "ca-constrained"
    assert capture_certificate_directory(tmp_path, tuple(reversed(_DOMAINS))) == directory
    assert capture_certificate_directory(tmp_path, (*_DOMAINS, _DOMAINS[0])) == directory
    certificate = prepare_certificate(directory, _DOMAINS)
    single = capture_certificate_directory(tmp_path, (_DOMAINS[0],))
    assert single != directory
    other = prepare_certificate(single, (_DOMAINS[0],))
    parsed = x509.load_pem_x509_certificate(certificate.read_bytes())
    assert parsed.public_key() != x509.load_pem_x509_certificate(other.read_bytes()).public_key()
    assert {path.name: path.read_bytes() for path in old.iterdir()} == before
    initial = {path.name: path.read_bytes() for path in directory.iterdir()}
    with pytest.raises(ValueError, match="constraints"):
        prepare_certificate(directory, (_DOMAINS[0],))
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == initial


def test_nested_hostnames_rejected_before_files_or_trust(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject unrepresentable exact host sets instead of silently trusting extra subdomains."""
    domains = ("openai.com", "api.openai.com")

    def unexpected_run(*args: object, **kwargs: object) -> None:
        """Make a trust lookup or mutation fail loudly for invalid scope input."""
        pytest.fail("Invalid certificate scope must not call security")

    monkeypatch.setattr("exp.runtime.capture.certificates.subprocess.run", unexpected_run)
    with pytest.raises(ValueError, match="Select nonoverlapping hostnames with --domain"):
        capture_certificate_directory(tmp_path, domains)
    with pytest.raises(ValueError, match="subdomain"):
        prepare_certificate(tmp_path / "ca", domains)
    with pytest.raises(ValueError, match="subdomain"):
        trust_certificate(tmp_path / "ca" / "mitmproxy-ca-cert.pem", domains)
    assert not list(tmp_path.iterdir())


def _rewrite_ca(certificate: Path, variant: str) -> None:
    """Create a genuinely signed unsafe CA variant in isolated test-owned files."""
    bundle_path = certificate.parent / "mitmproxy-ca.pem"
    key = serialization.load_pem_private_key(bundle_path.read_bytes(), password=None)
    assert isinstance(key, rsa.RSAPrivateKey)
    ca = x509.load_pem_x509_certificate(certificate.read_bytes())
    builder = (
        x509.CertificateBuilder()
        .subject_name(ca.subject)
        .issuer_name(ca.issuer)
        .public_key(ca.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(ca.not_valid_before_utc)
        .not_valid_after(ca.not_valid_after_utc)
    )
    for extension in ca.extensions:
        value = extension.value
        critical = extension.critical
        if isinstance(value, x509.NameConstraints):
            if variant == "unconstrained":
                continue
            if variant == "noncritical":
                critical = False
            elif variant == "allows_subdomains":
                value = x509.NameConstraints(
                    value.permitted_subtrees,
                    [
                        item
                        for item in value.excluded_subtrees or ()
                        if isinstance(item, x509.IPAddress)
                    ],
                )
            elif variant == "allows_ip":
                value = x509.NameConstraints(
                    value.permitted_subtrees,
                    [
                        item
                        for item in value.excluded_subtrees or ()
                        if isinstance(item, x509.DNSName)
                    ],
                )
        builder = builder.add_extension(value, critical=critical)
    replacement = builder.sign(key, hashes.SHA256()).public_bytes(serialization.Encoding.PEM)
    if variant == "key_mismatch":
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    bundle_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
        + replacement
    )
    if variant != "public_mismatch":
        certificate.write_bytes(replacement)


@pytest.mark.parametrize(
    "variant",
    [
        "unconstrained",
        "noncritical",
        "allows_subdomains",
        "allows_ip",
        "key_mismatch",
        "public_mismatch",
    ],
)
def test_unsafe_existing_ca_never_reused_checked_or_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    """Every entrypoint rejects unsafe scope or mismatched key material before OS trust access."""
    certificate = prepare_certificate(tmp_path / "ca", _DOMAINS)
    _rewrite_ca(certificate, variant)
    before = {path.name: path.read_bytes() for path in certificate.parent.iterdir()}

    def unexpected_run(*args: object, **kwargs: object) -> None:
        """Ensure invalid certificate material never reaches the native trust tool."""
        pytest.fail("Unsafe CA must be rejected before any security subprocess")

    monkeypatch.setattr("exp.runtime.capture.certificates.subprocess.run", unexpected_run)
    with pytest.raises(ValueError, match="Capture"):
        prepare_certificate(certificate.parent, _DOMAINS)
    with pytest.raises(ValueError, match="Capture"):
        certificate_is_trusted(certificate, _DOMAINS)
    with pytest.raises(ValueError, match="Capture"):
        trust_certificate(certificate, _DOMAINS)
    assert {path.name: path.read_bytes() for path in certificate.parent.iterdir()} == before


def test_partial_ca_store_is_never_repaired_with_a_new_key(tmp_path: Path) -> None:
    """A partial initialization fails without replacing any existing signing material."""
    directory = tmp_path / "ca"
    certificate = prepare_certificate(directory, _DOMAINS)
    key = (directory / "mitmproxy-ca.pem").read_bytes()
    certificate.unlink()
    with pytest.raises(ValueError, match="incomplete"):
        prepare_certificate(directory, _DOMAINS)
    assert (directory / "mitmproxy-ca.pem").read_bytes() == key
    assert not certificate.exists()


@pytest.mark.parametrize("filename", ["mitmproxy-ca-cert.pem", "mitmproxy-dhparam.pem"])
@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_failed_ca_initialization_leaves_empty_scope_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    failure: type[BaseException],
) -> None:
    """Failure on the second or third write never publishes an incomplete signing store."""
    directory = capture_certificate_directory(tmp_path, _DOMAINS)
    original_open = os.open

    def fail_write(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        """Interrupt one real certificate file creation after earlier writes have finished."""
        if Path(path).name == filename and flags & os.O_CREAT:
            raise failure("injected certificate write failure")
        return original_open(path, flags, mode, dir_fd=dir_fd)

    with monkeypatch.context() as patch:
        patch.setattr(certificates.os, "open", fail_write)
        with pytest.raises(failure, match="injected certificate write failure"):
            prepare_certificate(directory, _DOMAINS)
    assert directory.is_dir()
    assert not list(directory.iterdir())
    assert list(directory.parent.iterdir()) == [directory]
    certificate = prepare_certificate(directory, _DOMAINS)
    assert certificate.is_file()
    assert prepare_certificate(directory, _DOMAINS) == certificate


def test_abandoned_staging_directory_does_not_block_or_change_next_initialization(
    tmp_path: Path,
) -> None:
    """An unpublished store left by a killed process neither blocks retry nor gets reused."""
    directory = capture_certificate_directory(tmp_path, _DOMAINS)
    directory.mkdir(parents=True, mode=0o700)
    abandoned = directory.parent / f".{directory.name}-interrupted"
    abandoned.mkdir(mode=0o700)
    partial = abandoned / "mitmproxy-ca.pem"
    partial.write_bytes(b"unpublished signing material")
    certificate = prepare_certificate(directory, _DOMAINS)
    assert certificate.is_file()
    assert partial.read_bytes() == b"unpublished signing material"
    assert set(directory.parent.iterdir()) == {directory, abandoned}


def test_staged_ca_must_validate_before_atomic_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fully written but unsafe staging store cannot become the reusable scoped identity."""
    directory = capture_certificate_directory(tmp_path, _DOMAINS)
    original_create = certificates._create_certificate

    def create_unsafe_store(staging: Path, domains: tuple[str, ...]) -> None:
        """Strip constraints from genuinely generated test material before publication."""
        original_create(staging, domains)
        _rewrite_ca(staging / "mitmproxy-ca-cert.pem", "unconstrained")

    with monkeypatch.context() as patch:
        patch.setattr(certificates, "_create_certificate", create_unsafe_store)
        with pytest.raises(ValueError, match="constrained CA"):
            prepare_certificate(directory, _DOMAINS)
    assert not list(directory.iterdir())
    assert list(directory.parent.iterdir()) == [directory]
    assert prepare_certificate(directory, _DOMAINS).is_file()


@pytest.mark.parametrize("complete", [False, True])
def test_atomic_publication_preserves_store_appearing_during_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, complete: bool
) -> None:
    """Atomic directory replacement refuses concurrent complete or partial existing material."""
    directory = capture_certificate_directory(tmp_path, _DOMAINS)
    concurrent = tmp_path / "concurrent"
    prepare_certificate(concurrent, _DOMAINS)
    if not complete:
        (concurrent / "mitmproxy-ca-cert.pem").unlink()
    expected = {path.name: path.read_bytes() for path in concurrent.iterdir()}
    original_replace = os.replace

    def publish_with_race(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        """Publish another store immediately before the real atomic replacement attempt."""
        assert Path(target) == directory
        original_replace(concurrent, directory)
        original_replace(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(certificates.os, "replace", publish_with_race)
        with pytest.raises(OSError):
            prepare_certificate(directory, _DOMAINS)
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == expected
    assert list(directory.parent.iterdir()) == [directory]
    if complete:
        assert prepare_certificate(directory, _DOMAINS).is_file()
    else:
        with pytest.raises(ValueError, match="incomplete"):
            prepare_certificate(directory, _DOMAINS)
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == expected


def _write_leaf(
    certificate: Path, leaf: Path, host: str, sans: list[x509.GeneralName] | None
) -> None:
    """Sign a real TLS certificate, optionally omitting SAN to exercise CN-only clients."""
    key = serialization.load_pem_private_key(
        (certificate.parent / "mitmproxy-ca.pem").read_bytes(), password=None
    )
    assert isinstance(key, rsa.RSAPrivateKey)
    ca = x509.load_pem_x509_certificate(certificate.read_bytes())
    now = datetime.datetime.now(datetime.UTC)
    builder = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(x509.NameOID.COMMON_NAME, host)]))
        .issuer_name(ca.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(hours=1))
        .not_valid_after(now + datetime.timedelta(days=7))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([x509.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(key.public_key()), critical=False
        )
    )
    if sans is not None:
        builder = builder.add_extension(x509.SubjectAlternativeName(sans), critical=False)
    leaf.write_bytes(builder.sign(key, hashes.SHA256()).public_bytes(serialization.Encoding.PEM))


@pytest.mark.parametrize("verifier", ["openssl", "macos"])
@pytest.mark.parametrize(
    ("host", "additional_san", "expected"),
    [
        *((host, "", True) for host in _DOMAINS),
        ("sub.chatgpt.com", "", False),
        ("*.chatgpt.com", "", False),
        ("chatgpt.com.example.com", "", False),
        ("notchatgpt.com", "", False),
        ("openai.com", "", False),
        ("example.com", "", False),
        ("127.0.0.1", "", False),
        ("192.0.2.1", "", False),
        ("::1", "", False),
        ("2001:db8::1", "", False),
        ("chatgpt.com", "example.com", False),
        ("chatgpt.com", "127.0.0.1", False),
        ("chatgpt.com", "::1", False),
        ("example.com", "no_san", False),
    ],
)
def test_real_tls_chains_enforce_exact_dns_and_deny_ip_scope(
    tmp_path: Path, verifier: str, host: str, additional_san: str, expected: bool
) -> None:
    """Real offline verifiers enforce root constraints without importing any Keychain trust."""
    if verifier == "macos" and sys.platform != "darwin":
        pytest.skip("Native certificate verifier requires macOS")
    openssl = shutil.which("openssl")
    if verifier == "openssl" and openssl is None:
        pytest.skip("OpenSSL verifier is not installed")
    certificate = prepare_certificate(tmp_path / "ca", _DOMAINS)
    leaf = tmp_path / "leaf.pem"
    sans: list[x509.GeneralName] = []
    for name in [
        host,
        *([additional_san] if additional_san and additional_san != "no_san" else []),
    ]:
        try:
            sans.append(x509.IPAddress(ipaddress.ip_address(name)))
        except ValueError:
            sans.append(x509.DNSName(name))
    _write_leaf(certificate, leaf, host, None if additional_san == "no_san" else sans)
    if verifier == "macos":
        command = [
            "/usr/bin/security",
            "verify-cert",
            "-p",
            "ssl",
            "-n",
            host,
            "-c",
            str(leaf),
            "-r",
            str(certificate),
            "-L",
        ]
    else:
        assert openssl is not None
        command = [
            openssl,
            "verify",
            "-purpose",
            "sslserver",
            "-CAfile",
            str(certificate),
            str(leaf),
        ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
    assert (result.returncode == 0) == expected, result.stdout + result.stderr
