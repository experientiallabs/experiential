"""Host filters are finite exact public names shared by CLI and interception."""

import pytest

from exp.runtime.capture.policy import DEFAULT_CAPTURE_DOMAINS, validate_domains


def test_default_provider_hosts_are_valid() -> None:
    """The default covers supported OpenAI, Codex, and Anthropic endpoints."""
    assert DEFAULT_CAPTURE_DOMAINS == ("api.openai.com", "chatgpt.com", "api.anthropic.com")
    assert validate_domains(DEFAULT_CAPTURE_DOMAINS) == tuple(sorted(DEFAULT_CAPTURE_DOMAINS))


@pytest.mark.parametrize(
    "domain",
    [
        "",
        "localhost",
        "127.0.0.1",
        "::1",
        "*.openai.com",
        "https://api.openai.com",
        "api.openai.com:443",
        "api.openai.com/path",
        "api.openai.com\nelse",
        "api.localhost",
        "api.local",
        "api.internal",
        "api.test",
        "api.invalid",
        "A.com",
        "api.openai.com.",
        "api..openai.com",
        "-api.openai.com",
        "api-.openai.com",
        "api_host.example.com",
        "café.example.com",
        "a" * 64 + ".example.com",
        ".".join(["a" * 63] * 4),
    ],
)
def test_invalid_domains_are_rejected(domain: str) -> None:
    """The filter cannot be widened by URLs, addresses, wildcards, or malformed names."""
    with pytest.raises(ValueError, match="Invalid provider hostname"):
        validate_domains((domain,))


def test_domains_are_deduplicated_and_sorted() -> None:
    """Equivalent inputs produce stable exact-host policy without duplicate filters."""
    assert validate_domains(("b.example.com", "a.example.com", "b.example.com")) == (
        "a.example.com",
        "b.example.com",
    )


@pytest.mark.parametrize("count", [0, 33])
def test_domain_count_is_bounded(count: int) -> None:
    """A filter cannot accidentally select everything or grow without a limit."""
    with pytest.raises(ValueError, match="between 1 and 32"):
        validate_domains(tuple(f"host{index}.example.com" for index in range(count)))


def test_maximum_valid_hostnames_are_supported() -> None:
    """The validator accepts the exact DNS length and host-count boundaries."""
    longest = ".".join(["a" * 63] * 3 + ["b" * 61])
    assert len(longest) == 253
    domains = (longest, *(f"host{index}.example.com" for index in range(31)))
    assert validate_domains(domains) == tuple(sorted(domains))
