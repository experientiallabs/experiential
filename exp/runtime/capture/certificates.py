"""Private scope-constrained interception certificates and macOS user trust setup."""

from __future__ import annotations

import datetime
import hashlib
import ipaddress
import os
import shlex
import stat
import subprocess
import tempfile
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from mitmproxy.certs import DEFAULT_DHPARAM, CertStore

from exp.runtime.capture.policy import validate_domains

_CA_FILES = frozenset({"mitmproxy-ca.pem", "mitmproxy-ca-cert.pem", "mitmproxy-dhparam.pem"})


def capture_certificate_directory(base: Path, domains: tuple[str, ...]) -> Path:
    """Select a separate immutable CA identity for each exact hostname scope.

    Args:
        base: Capture's data directory, separate from any previous CA directory.
        domains: Nonoverlapping exact DNS hostnames selected for capture.

    Returns:
        A deterministic scope directory without reading or changing existing CAs.

    Raises:
        ValueError: Hostnames are invalid or contain a parent and its subdomain.
    """
    scope = _certificate_domains(domains)
    digest = hashlib.sha256("\n".join(scope).encode("ascii")).hexdigest()
    return base / "ca-constrained" / digest


def prepare_certificate(directory: Path, domains: tuple[str, ...]) -> Path:
    """Create or verify a private CA whose certificate permits only the selected hosts.

    Args:
        directory: Dedicated scope directory from capture_certificate_directory.
        domains: Exact hostnames encoded in the critical name constraints.

    Returns:
        Public certificate path suitable for the macOS user trust store.

    Raises:
        ValueError: Existing files are unsafe, incomplete, or have a different scope.
    """
    scope = _certificate_domains(domains)
    _check_directory(directory, create=True)
    certificate = directory / "mitmproxy-ca-cert.pem"
    if not any(directory.iterdir()):
        directory.chmod(0o700)
        with tempfile.TemporaryDirectory(
            prefix=f".{directory.name}-", dir=directory.parent
        ) as temporary:
            staging = Path(temporary)
            _create_certificate(staging, scope)
            _validate_certificate(staging / certificate.name, scope)
            # Replacing an empty directory is atomic. A concurrently published
            # nonempty store makes this fail without replacing its signing key.
            os.replace(staging, directory)
    _validate_certificate(certificate, scope)
    directory.chmod(0o700)
    for path in directory.iterdir():
        path.chmod(0o600)
    return certificate


def _certificate_domains(domains: tuple[str, ...]) -> tuple[str, ...]:
    """Reject nested selections that cannot share these exact DNS name constraints."""
    scope = validate_domains(domains)
    for domain in scope:
        for parent in scope:
            if domain.endswith("." + parent):
                raise ValueError(
                    f"Capture cannot combine {parent!r} and its subdomain {domain!r} in one "
                    "certificate. Select nonoverlapping hostnames with --domain."
                )
    return scope


def _name_constraints(domains: tuple[str, ...]) -> x509.NameConstraints:
    """Permit exact DNS names, exclude their subdomains, and prohibit every IP address."""
    return x509.NameConstraints(
        permitted_subtrees=[x509.DNSName(domain) for domain in domains],
        excluded_subtrees=[
            *(x509.DNSName("." + domain) for domain in domains),
            x509.IPAddress(ipaddress.ip_network("0.0.0.0/0")),
            x509.IPAddress(ipaddress.ip_network("::/0")),
        ],
    )


def _check_directory(directory: Path, *, create: bool = False) -> None:
    """Reject linked or foreign certificate paths before reading or creating key material."""
    for ancestor in (directory, *directory.parents):
        if ancestor.is_symlink():
            raise ValueError(
                "Capture certificate directory must be a user-owned directory without links."
            )
    if create:
        directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise ValueError("Capture certificate directory must be a user-owned directory.")
    for path in directory.iterdir():
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
            raise ValueError("Capture certificate directory contains unsafe files.")


def _create_certificate(directory: Path, domains: tuple[str, ...]) -> None:
    """Generate a fresh signing key and constrained CA without replacing existing files."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name(
        [
            x509.NameAttribute(x509.NameOID.COMMON_NAME, "Experiential Capture"),
            x509.NameAttribute(x509.NameOID.ORGANIZATION_NAME, "Experiential Labs"),
        ]
    )
    now = datetime.datetime.now(datetime.UTC)
    ca = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([x509.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .add_extension(_name_constraints(domains), critical=True)
        .sign(key, hashes.SHA256())
    )
    public = ca.public_bytes(serialization.Encoding.PEM)
    bundle = (
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
        + public
    )
    for filename, data in (
        ("mitmproxy-ca.pem", bundle),
        ("mitmproxy-ca-cert.pem", public),
        ("mitmproxy-dhparam.pem", DEFAULT_DHPARAM),
    ):
        descriptor = os.open(directory / filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)


def _validate_certificate(certificate: Path, domains: tuple[str, ...]) -> None:
    """Fail closed before reuse or trust if the CA identity or exact constraints differ."""
    _check_directory(certificate.parent)
    if (
        certificate.name != "mitmproxy-ca-cert.pem"
        or {path.name for path in certificate.parent.iterdir()} != _CA_FILES
    ):
        raise ValueError("Capture CA files are incomplete or unexpected; use a fresh CA directory.")
    bundle = (certificate.parent / "mitmproxy-ca.pem").read_bytes()
    certificates = x509.load_pem_x509_certificates(bundle)
    public = certificate.read_bytes()
    if len(certificates) != 1 or certificates[0].public_bytes(serialization.Encoding.PEM) != public:
        raise ValueError("Capture CA identity does not match its private bundle; use a fresh CA.")
    ca = certificates[0]
    key = serialization.load_pem_private_key(bundle, password=None)
    if not isinstance(key, rsa.RSAPrivateKey) or key.public_key() != ca.public_key():
        raise ValueError("Capture CA signing key does not match its certificate; use a fresh CA.")
    try:
        constraints = ca.extensions.get_extension_for_class(x509.NameConstraints)
        basic = ca.extensions.get_extension_for_class(x509.BasicConstraints)
        usage = ca.extensions.get_extension_for_class(x509.KeyUsage)
        extended = ca.extensions.get_extension_for_class(x509.ExtendedKeyUsage)
    except x509.ExtensionNotFound as error:
        raise ValueError(
            "Capture requires a constrained CA; use a fresh scoped CA directory."
        ) from error
    if (
        not constraints.critical
        or constraints.value != _name_constraints(domains)
        or not basic.critical
        or basic.value != x509.BasicConstraints(ca=True, path_length=0)
        or not usage.critical
        or not usage.value.key_cert_sign
        or extended.value != x509.ExtendedKeyUsage([x509.ExtendedKeyUsageOID.SERVER_AUTH])
    ):
        raise ValueError(
            "Capture CA constraints do not match the selected hostnames; use a fresh CA."
        )
    ca.verify_directly_issued_by(ca)
    now = datetime.datetime.now(datetime.UTC)
    if not ca.not_valid_before_utc <= now < ca.not_valid_after_utc:
        raise ValueError(
            "Capture CA has expired or is not valid yet; use a fresh scoped CA directory."
        )


def certificate_is_trusted(certificate: Path, domains: tuple[str, ...]) -> bool:
    """Verify constrained CA leaves for every host against installed macOS user trust.

    Explicit roots are deliberately omitted so this cannot bypass installed trust.
    Only temporary public leaf certificates touch disk.
    """
    if not domains:
        return False
    scope = _certificate_domains(domains)
    _validate_certificate(certificate, scope)
    store = CertStore.from_files(
        certificate.parent / "mitmproxy-ca.pem", certificate.parent / "mitmproxy-dhparam.pem"
    )
    with tempfile.TemporaryDirectory(prefix="verify-", dir=certificate.parent) as temporary:
        leaf = Path(temporary) / "leaf.pem"
        for domain in scope:
            descriptor = os.open(leaf, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(descriptor, "wb") as output:
                output.write(store.get_cert(domain, [x509.DNSName(domain)]).cert.to_pem())
            result = subprocess.run(
                [
                    "/usr/bin/security",
                    "verify-cert",
                    "-p",
                    "ssl",
                    "-n",
                    domain,
                    "-c",
                    str(leaf),
                    "-c",
                    str(certificate),
                    "-L",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
            )
            if result.returncode:
                return False
    return True


def trust_certificate(certificate: Path, domains: tuple[str, ...]) -> None:
    """Request user SSL trust only after verifying the CA's critical hostname restrictions.

    Chromium ignores macOS hostname-specific trust policies. The certificate itself
    therefore enforces the exact DNS scope; this command deliberately omits `-s`.

    Args:
        certificate: Public constrained certificate generated by prepare_certificate.
        domains: Exact hostnames captured by the local proxy.

    Raises:
        ValueError: CA constraints are unsafe or trust installation cannot be verified.
    """
    scope = _certificate_domains(domains)
    _validate_certificate(certificate, scope)
    keychain = _user_keychain()
    result = subprocess.run(
        [
            "/usr/bin/security",
            "add-trusted-cert",
            "-r",
            "trustRoot",
            "-p",
            "ssl",
            "-k",
            str(keychain),
            str(certificate),
        ],
        check=False,
        timeout=180,
        env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
    )
    if result.returncode or not certificate_is_trusted(certificate, scope):
        raise ValueError("Capture certificate trust was not installed. Run exp capture to retry.")


def _user_keychain() -> Path:
    """Resolve an existing user-owned default keychain without changing preferences."""
    result = subprocess.run(
        ["/usr/bin/security", "default-keychain", "-d", "user"],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    paths = shlex.split(result.stdout)
    if result.returncode or len(paths) != 1:
        raise ValueError("Cannot find your default keychain. Unlock your login keychain and retry.")
    keychain = Path(paths[0])
    if (
        not keychain.is_absolute()
        or not keychain.is_file()
        or keychain.stat().st_uid != os.getuid()
        or keychain.resolve().is_relative_to("/Library/Keychains")
        or keychain.resolve().is_relative_to("/System/Library/Keychains")
    ):
        raise ValueError("Capture trust requires an existing keychain owned by the current user.")
    return keychain
