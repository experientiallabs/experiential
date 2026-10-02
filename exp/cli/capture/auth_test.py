"""Capture uses the same endpoint-bound login as other CLI services."""

from collections.abc import Mapping
from pathlib import Path

import pytest
import typer
from rich.console import Console

from exp.cli.capture import auth
from exp.cli.capture.auth import capture_credentials
from exp.cli.providers.experiential_cloud import hosted_credential_binding
from exp.common.auth import (
    ProviderAuthStore,
    ProviderAuthStoreError,
    StoredCredentialEndpointMismatch,
)


def test_saved_login_reused_without_browser(tmp_path: Path) -> None:
    """Capture reuses an endpoint-bound login without initiating authentication."""
    store = ProviderAuthStore(tmp_path / "auth.json")
    store.put("experiential-cloud", "xpl_saved", binding=hosted_credential_binding({}))
    credentials = capture_credentials(console=Console(), environment={}, root=tmp_path, store=store)
    assert credentials.api_key == "xpl_saved"
    assert credentials.api_url == "https://api.experientiallabs.ai"
    assert "xpl_saved" not in repr(credentials)


@pytest.mark.parametrize(
    "api_url",
    ["https://api.experientiallabs.ai/v1/", "https://API.EXPERIENTIALLABS.AI:443/v1"],
)
def test_equivalent_production_endpoint_still_uses_normal_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api_url: str
) -> None:
    """Canonical production URLs must not require a separate preview web origin."""
    store = ProviderAuthStore(tmp_path / "auth.json")
    environment = {"EXP_GATEWAY_URL": api_url}
    calls: list[Mapping[str, str]] = []

    def login(
        *, console: Console, environment: Mapping[str, str], root: Path, store: ProviderAuthStore
    ) -> None:
        """Persist one synthetic login with the canonical production binding."""
        calls.append(environment)
        store.put("experiential-cloud", "xpl_new", binding=hosted_credential_binding({}))

    monkeypatch.setattr(auth, "run_login", login)
    credentials = capture_credentials(
        console=Console(), environment=environment, root=tmp_path, store=store
    )
    assert calls == [environment]
    assert credentials.api_key == "xpl_new"
    assert credentials.web_url == "https://platform.experientiallabs.ai"


@pytest.mark.parametrize("bound", [True, False])
def test_preview_logs_in_instead_of_reusing_an_incompatible_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bound: bool
) -> None:
    """A changed endpoint or unbound saved key requires a fresh normal login."""
    store = ProviderAuthStore(tmp_path / "auth.json")
    store.put(
        "experiential-cloud",
        "xpl_saved",
        binding=hosted_credential_binding({}) if bound else None,
    )
    environment = {
        "EXP_GATEWAY_URL": "https://api.preview.example/v1",
        "EXP_PLATFORM_URL": "https://preview.example",
    }
    calls: list[Mapping[str, str]] = []

    def login(
        *, console: Console, environment: Mapping[str, str], root: Path, store: ProviderAuthStore
    ) -> None:
        """Supply only the new endpoint's credential through the ordinary login seam."""
        assert root == tmp_path
        assert store.get("experiential-cloud") == "xpl_saved"
        calls.append(environment)
        store.put(
            "experiential-cloud", "xpl_preview", binding=hosted_credential_binding(environment)
        )

    monkeypatch.setattr(auth, "run_login", login)
    credentials = capture_credentials(
        console=Console(), environment=environment, root=tmp_path, store=store
    )
    assert calls == [environment]
    assert credentials.api_key == "xpl_preview"
    assert credentials.api_url == "https://api.preview.example"
    assert credentials.web_url == "https://preview.example"


def test_cancelled_endpoint_login_preserves_the_saved_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancelling the new endpoint's login leaves the existing credential untouched."""
    store = ProviderAuthStore(tmp_path / "auth.json")
    store.put("experiential-cloud", "xpl_saved", binding=hosted_credential_binding({}))
    original = store.path.read_bytes()

    def cancel(
        *, console: Console, environment: Mapping[str, str], root: Path, store: ProviderAuthStore
    ) -> None:
        """Abort at the browser-login boundary before any new credential is saved."""
        raise typer.Abort

    monkeypatch.setattr(auth, "run_login", cancel)
    with pytest.raises(typer.Abort):
        capture_credentials(
            console=Console(),
            environment={
                "EXP_GATEWAY_URL": "https://api.preview.example/v1",
                "EXP_PLATFORM_URL": "https://preview.example",
            },
            root=tmp_path,
            store=store,
        )
    assert store.path.read_bytes() == original


def test_login_cannot_return_the_wrong_endpoints_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A login that leaves an incompatible credential still fails closed without a loop."""
    store = ProviderAuthStore(tmp_path / "auth.json")
    store.put("experiential-cloud", "xpl_saved", binding=hosted_credential_binding({}))
    calls: list[str] = []

    def incomplete_login(
        *, console: Console, environment: Mapping[str, str], root: Path, store: ProviderAuthStore
    ) -> None:
        """Record one login without replacing the previous endpoint's key."""
        calls.append("login")

    monkeypatch.setattr(auth, "run_login", incomplete_login)
    with pytest.raises(StoredCredentialEndpointMismatch):
        capture_credentials(
            console=Console(),
            environment={
                "EXP_GATEWAY_URL": "https://api.preview.example/v1",
                "EXP_PLATFORM_URL": "https://preview.example",
            },
            root=tmp_path,
            store=store,
        )
    assert calls == ["login"]


@pytest.mark.parametrize("web_url", ["", "https://platform.experientiallabs.ai/"])
@pytest.mark.parametrize("saved", [True, False])
def test_preview_requires_its_web_origin_before_browser_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, web_url: str, saved: bool
) -> None:
    """A preview API cannot receive a new key minted by the production login page."""
    store = ProviderAuthStore(tmp_path / "auth.json")
    if saved:
        store.put("experiential-cloud", "xpl_saved", binding=hosted_credential_binding({}))

    def unexpected_login(
        *, console: Console, environment: Mapping[str, str], root: Path, store: ProviderAuthStore
    ) -> None:
        """Reject any attempt to open the production browser login for a preview API."""
        pytest.fail("Mismatched login origins must fail before browser login")

    monkeypatch.setattr(auth, "run_login", unexpected_login)
    with pytest.raises(ValueError, match="EXP_PLATFORM_URL"):
        capture_credentials(
            console=Console(),
            environment={
                "EXP_GATEWAY_URL": "https://api.preview.example/v1",
                "EXP_PLATFORM_URL": web_url,
            },
            root=tmp_path,
            store=store,
        )
    assert store.get("experiential-cloud", binding=hosted_credential_binding({})) == (
        "xpl_saved" if saved else None
    )


def test_malformed_credential_store_is_not_replaced_by_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A damaged auth file remains an error instead of being overwritten by login."""
    store = ProviderAuthStore(tmp_path / "auth.json")
    store.path.write_text("malformed auth", encoding="utf-8")

    def unexpected_login(
        *, console: Console, environment: Mapping[str, str], root: Path, store: ProviderAuthStore
    ) -> None:
        """Reject an attempted login after an unrelated credential-store failure."""
        pytest.fail("Malformed credentials must not trigger login")

    monkeypatch.setattr(auth, "run_login", unexpected_login)
    with pytest.raises(ProviderAuthStoreError):
        capture_credentials(console=Console(), environment={}, root=tmp_path, store=store)
    assert store.path.read_text(encoding="utf-8") == "malformed auth"


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://public.example/v1",
        "https://user:password@example.com/v1",
        "https://example.com/v1?key=secret",
        "https://example.com/other",
    ],
)
def test_unsafe_endpoint_rejected_before_login(tmp_path: Path, endpoint: str) -> None:
    """Invalid API origins fail before credentials or login are consulted."""
    with pytest.raises(ValueError, match="EXP_GATEWAY_URL"):
        capture_credentials(
            console=Console(),
            environment={"EXP_GATEWAY_URL": endpoint},
            root=tmp_path,
            store=ProviderAuthStore(tmp_path / "absent.json"),
        )


def test_loopback_preview_environment_key_supported(tmp_path: Path) -> None:
    """An explicit local development endpoint accepts its ordinary environment key."""
    credentials = capture_credentials(
        console=Console(),
        environment={
            "EXP_GATEWAY_URL": "http://127.0.0.1:8000/v1",
            "EXPLABS_API_KEY": "xpl_local",
        },
        root=tmp_path,
        store=ProviderAuthStore(tmp_path / "absent.json"),
    )
    assert credentials.api_url == "http://127.0.0.1:8000"
    assert credentials.api_key == "xpl_local"
