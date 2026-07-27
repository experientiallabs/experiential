"""Tests for the platform CLI commands (wiring and kind resolution)."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import typer
from typer.testing import CliRunner, Result

from wmo.cli.app import app
from wmo.cli.platform_cmds import _pull_harness, _resolve_kind
from wmo.harness.doc import HarnessDoc, Surface, SurfaceKind
from wmo.harness.store import HarnessStore
from wmo.platform.client import (
    HarnessVersionDoc,
    PlatformError,
    PlatformUnreachable,
    RemoteWorldModel,
    WhoAmI,
)
from wmo.platform.credentials import ENV_HOME, PlatformCredentials, save_credentials

if TYPE_CHECKING:
    from wmo.platform.client import PlatformClient

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
        "WMO_PLATFORM_URL",
        "WMO_PLATFORM_API_URL",
        "WMO_PLATFORM_TOKEN",
        "WMO_PLATFORM_ORG",
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


def _unreachable() -> PlatformUnreachable:
    return PlatformUnreachable(
        "cannot reach https://api.test/api/whoami: [Errno 61] Connection refused; "
        "check your connection, or re-run `wmo login --url <platform url>`"
    )


def _flatten(output: str) -> str:
    """Rejoin a rich-wrapped message so assertions can match it as one string."""
    return " ".join(output.replace("│", " ").split())


def _assert_clean_failure(result: Result) -> None:
    """The command failed through the CLI instead of letting an exception escape.

    A `PlatformError`/`ConnectError` reaching here is what a user sees as a
    Python traceback, so the exception type is the assertion that matters.
    """
    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit), result.exception


def test_platform_commands_are_registered() -> None:
    result = runner.invoke(app, ["--help"])
    for command in ("login", "logout", "status", "push", "pull"):
        assert command in result.output


def test_status_without_credentials_points_to_login() -> None:
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 1
    assert "wmo login" in result.output


def test_status_reports_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    save_credentials(
        PlatformCredentials(
            web_url="https://platform.test", api_url="https://api.test", token="xpl_x"
        )
    )
    monkeypatch.setattr("wmo.cli.platform_cmds.PlatformClient", _StubClient)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0, result.output
    assert "Acme" in result.output
    assert "org-1" in result.output


def test_status_surfaces_rejected_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    save_credentials(PlatformCredentials(api_url="https://api.test", token="xpl_bad"))

    class _RejectingClient(_StubClient):
        def whoami(self) -> WhoAmI:
            raise PlatformError("Unauthorized", status_code=401)

    monkeypatch.setattr("wmo.cli.platform_cmds.PlatformClient", _RejectingClient)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 1
    assert "Unauthorized" in result.output


def test_status_reports_an_unreachable_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """The command that answers "is my connection healthy?" must survive "no"."""
    save_credentials(PlatformCredentials(api_url="http://127.0.0.1:9", token="xpl_x"))

    class _UnreachableClient(_StubClient):
        def whoami(self) -> WhoAmI:
            raise _unreachable()

    monkeypatch.setattr("wmo.cli.platform_cmds.PlatformClient", _UnreachableClient)

    result = runner.invoke(app, ["status"])

    _assert_clean_failure(result)
    assert "Connection check failed" in result.output
    assert "cannot reach" in result.output


def test_status_falls_back_to_the_api_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env-var credentials carry no web_url; the host still has to be named."""
    save_credentials(PlatformCredentials(api_url="https://api.test", token="xpl_x"))
    monkeypatch.setattr("wmo.cli.platform_cmds.PlatformClient", _StubClient)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0, result.output
    assert "https://api.test" in result.output
    assert "None" not in result.output


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
    root = str(tmp_path / ".wmo")

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
            str(tmp_path / ".wmo"),
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
    monkeypatch.setattr("wmo.cli.platform_cmds.fetch_cli_config", lambda _url: "https://api.test")
    monkeypatch.setattr("wmo.cli.platform_cmds.PlatformClient", _StubClient)

    result = runner.invoke(app, ["login", "--token", "xpl_new"])

    assert result.exit_code == 0, result.output
    from wmo.platform.credentials import load_credentials

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

    monkeypatch.setattr("wmo.cli.platform_cmds.fetch_cli_config", unexpected_discovery)
    monkeypatch.setattr("wmo.cli.platform_cmds.PlatformClient", _StubClient)

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
    from wmo.platform.credentials import load_credentials

    saved = load_credentials()
    assert saved.web_url == "https://preview.test"
    assert saved.api_url == "https://api-preview.test"
    assert saved.token == "xpl_new"


def test_login_with_api_url_only_records_no_web_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """--api-url alone cannot know the web app, so it must not claim the hosted one."""
    monkeypatch.setattr("wmo.cli.platform_cmds.PlatformClient", _StubClient)

    result = runner.invoke(
        app, ["login", "--api-url", "https://api-preview.test/", "--token", "xpl_new"]
    )

    assert result.exit_code == 0, result.output
    from wmo.platform.credentials import load_credentials

    saved = load_credentials()
    assert saved.api_url == "https://api-preview.test"
    assert saved.web_url is None


def test_login_reports_an_unreachable_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(_url: str) -> str:
        raise _unreachable()

    monkeypatch.setattr("wmo.cli.platform_cmds.fetch_cli_config", refuse)

    result = runner.invoke(app, ["login", "--url", "http://127.0.0.1:9"])

    _assert_clean_failure(result)
    assert "cannot reach" in result.output
    assert "wmo login --url" in result.output


def test_login_reports_a_url_that_is_not_a_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 200 of HTML (login wall, SPA rewrite) is a verdict, not a decode crash."""

    def not_json(_url: str) -> str:
        raise PlatformError("answered HTTP 200 with text/html, not JSON", status_code=200)

    monkeypatch.setattr("wmo.cli.platform_cmds.fetch_cli_config", not_json)

    result = runner.invoke(app, ["login", "--url", "https://preview.test"])

    _assert_clean_failure(result)
    assert "does not look like a platform" in result.output
    assert "not JSON" in result.output


def test_login_does_not_blame_the_key_for_an_unreachable_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _UnreachableClient(_StubClient):
        def whoami(self) -> WhoAmI:
            raise _unreachable()

    monkeypatch.setattr("wmo.cli.platform_cmds.PlatformClient", _UnreachableClient)

    result = runner.invoke(app, ["login", "--api-url", "http://127.0.0.1:9", "--token", "xpl_new"])

    _assert_clean_failure(result)
    assert "Connection failed" in result.output
    assert "rejected" not in result.output


def test_push_requires_login_first(tmp_path: Path) -> None:
    result = runner.invoke(app, ["push", "anything", "--root", str(tmp_path)])
    assert result.exit_code != 0
    assert "no local world model or harness" in result.output


def _connected() -> None:
    save_credentials(
        PlatformCredentials(
            web_url="https://platform.test",
            api_url="https://api.test",
            token="xpl_x",
            default_org="org-1",
        )
    )


def _write_harness(root: str, name: str = "demo") -> None:
    doc = HarnessDoc(
        name=name, surfaces=[Surface(id="prompt:core", kind=SurfaceKind.PROMPT, content="p")]
    )
    HarnessStore(root).save_version(doc)


def test_push_reports_an_unreachable_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = str(tmp_path / ".wmo")
    _write_harness(root)
    _connected()

    class _UnreachableClient(_StubClient):
        def push_harness_version(self, *_args: object, **_kwargs: object) -> object:
            raise _unreachable()

    monkeypatch.setattr("wmo.cli.platform_cmds.PlatformClient", _UnreachableClient)

    result = runner.invoke(app, ["push", "demo", "--root", root])

    _assert_clean_failure(result)
    assert "Push failed" in result.output
    assert "cannot reach" in result.output


def test_push_reports_a_platform_http_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = str(tmp_path / ".wmo")
    _write_harness(root)
    _connected()

    class _MissingOrgClient(_StubClient):
        def push_harness_version(self, *_args: object, **_kwargs: object) -> object:
            raise PlatformError("not found", status_code=404)

    monkeypatch.setattr("wmo.cli.platform_cmds.PlatformClient", _MissingOrgClient)

    result = runner.invoke(app, ["push", "demo", "--root", root])

    _assert_clean_failure(result)
    assert "Push failed: not found" in _flatten(result.output)


def test_pull_surfaces_the_hint_inside_a_platform_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A revoked key already carries the remedy; it must not be buried in a traceback."""
    _connected()

    class _RevokedKeyClient(_StubClient):
        def list_world_models(self, _org_id: str) -> list[RemoteWorldModel]:
            raise PlatformError(
                "invalid API key — run `wmo login` (or check WMO_PLATFORM_TOKEN)", status_code=401
            )

    monkeypatch.setattr("wmo.cli.platform_cmds.PlatformClient", _RevokedKeyClient)

    result = runner.invoke(app, ["pull", "demo", "--root", str(tmp_path / ".wmo")])

    _assert_clean_failure(result)
    normalized = _flatten(result.output)
    assert "Pull failed: invalid API key" in normalized
    assert "run `wmo login`" in normalized


def test_pull_reports_an_unreachable_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _connected()

    class _UnreachableClient(_StubClient):
        def list_world_models(self, _org_id: str) -> list[RemoteWorldModel]:
            raise _unreachable()

    monkeypatch.setattr("wmo.cli.platform_cmds.PlatformClient", _UnreachableClient)

    result = runner.invoke(app, ["pull", "demo", "--root", str(tmp_path / ".wmo")])

    _assert_clean_failure(result)
    assert "Pull failed" in result.output
    assert "cannot reach" in result.output


def test_push_rejects_an_unknown_ref(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo'd --ref is a usage error naming where the versions are listed."""
    root = str(tmp_path / ".wmo")
    _write_harness(root)
    _connected()
    monkeypatch.setattr("wmo.cli.platform_cmds.PlatformClient", _StubClient)

    result = runner.invoke(app, ["push", "demo", "--ref", "v99", "--root", root])

    _assert_clean_failure(result)
    normalized = _flatten(result.output)
    assert "no version v99" in normalized
    assert "wmo harness list" in normalized


def test_push_unknown_name_names_the_root_and_the_next_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Run against the default relative --root so the message's root is readable
    # in the wrapped error box, exactly as a user in the wrong directory sees it.
    monkeypatch.chdir(tmp_path)
    _write_harness(".wmo")

    result = runner.invoke(app, ["push", "nosuchmodel"])

    _assert_clean_failure(result)
    normalized = _flatten(result.output)
    assert "no local world model or harness named 'nosuchmodel' under .wmo" in normalized
    assert "have: demo" in normalized
    assert "wmo list" in normalized


def test_logout_when_not_logged_in() -> None:
    result = runner.invoke(app, ["logout"])
    assert result.exit_code == 0
    assert "nothing to remove" in result.output.lower()


def test_resolve_kind_disambiguates(tmp_path: Path) -> None:
    root = str(tmp_path / ".wmo")
    resolve = partial(_resolve_kind, name="x", root=root)
    assert resolve(None, model=True, harness=False) == "model"
    assert resolve(None, model=False, harness=True) == "harness"
    assert resolve("model", model=True, harness=True) == "model"
    with pytest.raises(typer.BadParameter, match="pass --kind"):
        resolve(None, model=True, harness=True)
    with pytest.raises(typer.BadParameter, match="no local world model or harness"):
        resolve(None, model=False, harness=False)
    with pytest.raises(typer.BadParameter, match="no local world model"):
        resolve("model", model=False, harness=True)
    with pytest.raises(typer.BadParameter, match="must be"):
        resolve("bundle", model=True, harness=False)


def test_bare_login_targets_the_hosted_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """`wmo login` with no --url and no saved platform uses the default."""
    seen: dict[str, str] = {}

    def fake_config(url: str) -> str:
        seen["web_url"] = url
        return "https://api.test"

    monkeypatch.setattr("wmo.cli.platform_cmds.fetch_cli_config", fake_config)
    monkeypatch.setattr("wmo.cli.platform_cmds.PlatformClient", _StubClient)

    result = runner.invoke(app, ["login", "--token", "xpl_new"])

    assert result.exit_code == 0, result.output
    assert seen["web_url"] == "https://platform.experientiallabs.ai"
