"""Tests for the platform CLI commands (wiring and kind resolution)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import typer
from typer.testing import CliRunner

from wmh.cli.app import app
from wmh.cli.platform_cmds import _pull_harness, _resolve_kind
from wmh.harness.doc import HarnessDoc, Surface, SurfaceKind
from wmh.harness.store import HarnessStore
from wmh.platform.client import HarnessVersionDoc, PlatformError, WhoAmI
from wmh.platform.credentials import ENV_HOME, PlatformCredentials, save_credentials

if TYPE_CHECKING:
    from wmh.platform.client import PlatformClient

runner = CliRunner()

_WHOAMI = WhoAmI.model_validate(
    {
        "actor": {"kind": "api_key", "id": "api-key:org-1"},
        "orgs": [{"id": "org-1", "slug": "acme", "name": "Acme"}],
    }
)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_HOME, str(tmp_path))
    for var in (
        "WMH_PLATFORM_URL",
        "WMH_PLATFORM_API_URL",
        "WMH_PLATFORM_TOKEN",
        "WMH_PLATFORM_ORG",
    ):
        monkeypatch.delenv(var, raising=False)


class _StubClient:
    """PlatformClient stand-in: canned whoami, no network."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def __enter__(self) -> _StubClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        pass

    def whoami(self) -> WhoAmI:
        return _WHOAMI


def test_platform_commands_are_registered() -> None:
    result = runner.invoke(app, ["--help"])
    for command in ("login", "logout", "status", "push", "pull"):
        assert command in result.output


def test_status_without_credentials_points_to_login() -> None:
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 1
    assert "wmh login" in result.output


def test_status_reports_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    save_credentials(
        PlatformCredentials(
            web_url="https://platform.test", api_url="https://api.test", token="xpl_x"
        )
    )
    monkeypatch.setattr("wmh.cli.platform_cmds.PlatformClient", _StubClient)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0, result.output
    assert "Acme" in result.output
    assert "org-1" in result.output


def test_status_surfaces_rejected_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    save_credentials(PlatformCredentials(api_url="https://api.test", token="xpl_bad"))

    class _RejectingClient(_StubClient):
        def whoami(self) -> WhoAmI:
            raise PlatformError("Unauthorized", status_code=401)

    monkeypatch.setattr("wmh.cli.platform_cmds.PlatformClient", _RejectingClient)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 1
    assert "Unauthorized" in result.output


def test_pull_rejects_unknown_kind() -> None:
    """An invalid --kind fails fast instead of dispatching to harness routes."""
    result = runner.invoke(app, ["pull", "anything", "--kind", "typo"])
    assert result.exit_code != 0
    assert "must be 'model' or 'harness'" in result.output


def _pathful_doc() -> HarnessDoc:
    return HarnessDoc(
        name="pi",
        surfaces=[
            Surface(id="prompt:core", kind=SurfaceKind.PROMPT, content="p"),
            Surface(
                id="code:src-agent-ts",
                kind=SurfaceKind.CODE,
                path="src/agent.ts",
                content="// a",
            ),
        ],
    )


class _HarnessVersionClient(_StubClient):
    """Serves one canned harness version payload with a configurable hash."""

    payload_doc_hash = ""

    def get_harness_version(self, org_id: str, name: str, version: int) -> HarnessVersionDoc:
        del org_id, name
        return HarnessVersionDoc(
            version=version,
            doc=_pathful_doc().model_dump(mode="json"),
            doc_hash=type(self).payload_doc_hash,
        )


def test_pull_harness_accepts_the_legacy_doc_hash(tmp_path: Path) -> None:
    """Pathful versions the platform recorded pre-path-hash must stay pullable."""
    doc = _pathful_doc()
    root = str(tmp_path / ".wmh")

    class _LegacyClient(_HarnessVersionClient):
        payload_doc_hash = doc.legacy_doc_hash

    _pull_harness(cast("PlatformClient", _LegacyClient()), "org-1", "pi", root, version=3)

    assert HarnessStore(root).load("pi").doc_hash == doc.doc_hash


def test_pull_harness_still_rejects_a_corrupt_doc_hash(tmp_path: Path) -> None:
    class _CorruptClient(_HarnessVersionClient):
        payload_doc_hash = "0" * 32

    with pytest.raises(typer.Exit):
        _pull_harness(
            cast("PlatformClient", _CorruptClient()),
            "org-1",
            "pi",
            str(tmp_path / ".wmh"),
            version=3,
        )


def test_login_with_token_drops_stale_default_org(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A relogin keeps the default organization only if the new identity sees it."""
    save_credentials(
        PlatformCredentials(
            web_url="https://platform.test",
            api_url="https://api.test",
            token="xpl_old",
            default_org="org-gone",
        )
    )
    monkeypatch.setattr("wmh.cli.platform_cmds.fetch_cli_config", lambda _url: "https://api.test")
    monkeypatch.setattr("wmh.cli.platform_cmds.PlatformClient", _StubClient)

    result = runner.invoke(app, ["login", "--token", "xpl_new"])

    assert result.exit_code == 0, result.output
    from wmh.platform.credentials import load_credentials

    saved = load_credentials()
    assert saved.token == "xpl_new"
    # org-gone is invisible to the new identity; the single visible
    # organization becomes the default instead.
    assert saved.default_org == "org-1"


def test_login_with_explicit_api_url_skips_web_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protected previews can pair browser auth with their direct backend URL."""

    def unexpected_discovery(_url: str) -> str:
        pytest.fail("explicit --api-url must skip web discovery")

    monkeypatch.setattr("wmh.cli.platform_cmds.fetch_cli_config", unexpected_discovery)
    monkeypatch.setattr("wmh.cli.platform_cmds.PlatformClient", _StubClient)

    result = runner.invoke(
        app,
        [
            "login",
            "--url",
            "https://preview.test/",
            "--api-url",
            "https://api-preview.test/",
            "--token",
            "xpl_new",
        ],
    )

    assert result.exit_code == 0, result.output
    from wmh.platform.credentials import load_credentials

    saved = load_credentials()
    assert saved.web_url == "https://preview.test"
    assert saved.api_url == "https://api-preview.test"
    assert saved.token == "xpl_new"


def test_push_requires_login_first(tmp_path: Path) -> None:
    result = runner.invoke(app, ["push", "anything", "--root", str(tmp_path)])
    assert result.exit_code != 0
    assert "no local world model or harness" in result.output


def test_logout_when_not_logged_in() -> None:
    result = runner.invoke(app, ["logout"])
    assert result.exit_code == 0
    assert "nothing to remove" in result.output.lower()


def test_resolve_kind_disambiguates() -> None:
    assert _resolve_kind(None, model=True, harness=False) == "model"
    assert _resolve_kind(None, model=False, harness=True) == "harness"
    assert _resolve_kind("model", model=True, harness=True) == "model"
    with pytest.raises(typer.BadParameter, match="pass --kind"):
        _resolve_kind(None, model=True, harness=True)
    with pytest.raises(typer.BadParameter, match="no local world model or harness"):
        _resolve_kind(None, model=False, harness=False)
    with pytest.raises(typer.BadParameter, match="no local world model"):
        _resolve_kind("model", model=False, harness=True)
    with pytest.raises(typer.BadParameter, match="must be"):
        _resolve_kind("bundle", model=True, harness=False)


def test_bare_login_targets_the_hosted_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """`wmh login` with no --url and no saved platform uses the default."""
    seen: dict[str, str] = {}

    def fake_config(url: str) -> str:
        seen["web_url"] = url
        return "https://api.test"

    monkeypatch.setattr("wmh.cli.platform_cmds.fetch_cli_config", fake_config)
    monkeypatch.setattr("wmh.cli.platform_cmds.PlatformClient", _StubClient)

    result = runner.invoke(app, ["login", "--token", "xpl_new"])

    assert result.exit_code == 0, result.output
    assert seen["web_url"] == "https://platform.experientiallabs.ai"
