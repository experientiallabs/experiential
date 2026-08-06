"""Tests for the platform CLI commands (wiring and kind resolution)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from typer.testing import CliRunner, Result

from wmo.cli.app import app
from wmo.runtime.harness.doc import HarnessDoc, Surface, SurfaceKind
from wmo.runtime.harness.store import HarnessStore
from wmo.runtime.platform.client import (
    PlatformError,
    PlatformUnreachable,
    RemoteWorldModel,
    WhoAmI,
)
from wmo.runtime.platform.credentials import ENV_HOME, PlatformCredentials, save_credentials
from wmo.runtime.runs.client import PushAck
from wmo.runtime.runs.schema import pipeline_external_id

if TYPE_CHECKING:
    pass

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
    # Equality on the host, not a substring: the bug printed "None" here, so
    # what matters is exactly which host got named.
    connected = next(line for line in result.output.splitlines() if "Connected to" in line)
    assert connected.split("Connected to ", 1)[1].strip() == "https://api.test"


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
    from wmo.runtime.platform.credentials import load_credentials

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
    from wmo.runtime.platform.credentials import load_credentials

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
    from wmo.runtime.platform.credentials import load_credentials

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


def test_logout_when_not_logged_in() -> None:
    result = runner.invoke(app, ["logout"])
    assert result.exit_code == 0
    assert "nothing to remove" in result.output.lower()


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


def _write_model_dir(root: str, name: str = "demo-model") -> Path:
    """A minimal local world model: the config file is what makes it one."""
    model_dir = Path(root) / "models" / name
    model_dir.mkdir(parents=True)
    (model_dir / "config.toml").write_text("embed_dim = 64\n", encoding="utf-8")
    return model_dir


_PUSHED = RemoteWorldModel(id="wm-99", name="demo-model", status="ready")


class _RecordingPushClient(_StubClient):
    """Records the push legs; the bundle upload itself is short-circuited."""

    calls: list[tuple[str, object]] = []

    def push_model_bundle(self, *_args: object, **_kwargs: object) -> RemoteWorldModel:
        type(self).calls.append(("bundle", None))
        return _PUSHED

    def get_endpoint(self, org_id: str, name: str) -> dict[str, object] | None:
        del org_id, name
        return None

    def create_endpoint(self, org_id: str, name: str, **kwargs: object) -> dict[str, object]:
        del org_id
        type(self).calls.append(("create", {"name": name, **kwargs}))
        return {"name": name}

    def install_endpoint_artifacts(
        self, org_id: str, name: str, *, policy: object, report: object
    ) -> dict[str, object]:
        del org_id
        type(self).calls.append(("artifacts", {"name": name, "policy": policy, "report": report}))
        return {"name": name, "status": "ready"}


def test_push_model_installs_endpoint_artifacts_beside_the_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One push carries the simulation AND the measured endpoint artifacts."""
    root = str(tmp_path / ".wmo")
    model_dir = _write_model_dir(root)
    (model_dir / "policy.json").write_text(
        '{"kind": "static", "default_model": "m1", "pool": []}', encoding="utf-8"
    )
    (model_dir / "report.json").write_text('{"headline": {"accuracy": 1.0}}', encoding="utf-8")
    _connected()
    _RecordingPushClient.calls = []
    monkeypatch.setattr("wmo.cli.platform_cmds.PlatformClient", _RecordingPushClient)

    result = runner.invoke(app, ["push", "demo-model", "--root", root])

    assert result.exit_code == 0, result.output
    kinds = [kind for kind, _ in _RecordingPushClient.calls]
    assert kinds == ["bundle", "create", "artifacts"]
    _, create = _RecordingPushClient.calls[1]
    assert create == {"name": "demo-model", "world_model_id": "wm-99", "model": None}
    _, artifacts = _RecordingPushClient.calls[2]
    assert isinstance(artifacts, dict)
    policy = cast("dict[str, object]", artifacts)["policy"]
    assert isinstance(policy, dict)
    assert cast("dict[str, object]", policy)["kind"] == "static"
    # The pipeline leg reports its own absence instead of failing the push.
    assert "skipping the pipeline leg" in _flatten(result.output)


def test_push_model_without_artifacts_still_pushes_the_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bundle-only directory pushes exactly as before, with skip notes."""
    root = str(tmp_path / ".wmo")
    _write_model_dir(root)
    _connected()
    _RecordingPushClient.calls = []
    monkeypatch.setattr("wmo.cli.platform_cmds.PlatformClient", _RecordingPushClient)

    result = runner.invoke(app, ["push", "demo-model", "--root", root])

    assert result.exit_code == 0, result.output
    assert [kind for kind, _ in _RecordingPushClient.calls] == ["bundle"]
    flat = _flatten(result.output)
    assert "skipping the endpoint leg" in flat
    assert "skipping the pipeline leg" in flat


class _FakeReader:
    """RunsReader stand-in whose event count is set by the test."""

    count = 0

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def event_count(self, _external_id: str) -> int:
        return type(self).count


class _FakeSink:
    """RunsSink stand-in recording what was pushed."""

    pushes: list[tuple[str, int]] = []

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def push(self, external_id: str, events: list[object]) -> PushAck:
        type(self).pushes.append((external_id, len(events)))
        return PushAck(accepted=len(events), last_seq=len(events))


def _pipeline_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    root = str(tmp_path / ".wmo")
    model_dir = _write_model_dir(root)
    manifest = model_dir / "optimize" / "optimize-run.json"
    manifest.parent.mkdir()
    manifest.write_text("{}", encoding="utf-8")
    _connected()
    _RecordingPushClient.calls = []
    _FakeSink.pushes = []
    monkeypatch.setattr("wmo.cli.platform_cmds.PlatformClient", _RecordingPushClient)
    monkeypatch.setattr("wmo.cli.platform_cmds.RunsReader", _FakeReader)
    monkeypatch.setattr("wmo.cli.platform_cmds.RunsSink", _FakeSink)
    # The derivation itself is backfill_test.py's contract; here only the
    # composition is under test.
    monkeypatch.setattr(
        "wmo.cli.platform_cmds.optimize_events", lambda *_a, **_k: [object(), object()]
    )
    return root


def test_push_model_replays_the_pipeline_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The optimize manifest replays into the platform's run history on push."""
    _FakeReader.count = 0
    root = _pipeline_fixture(tmp_path, monkeypatch)

    result = runner.invoke(app, ["push", "demo-model", "--root", root])

    assert result.exit_code == 0, result.output
    assert _FakeSink.pushes == [(pipeline_external_id("demo-model"), 2)]
    assert "Pushed pipeline run" in _flatten(result.output)


def test_push_model_skips_an_already_recorded_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run the platform already holds is left alone rather than double-counted."""
    _FakeReader.count = 7
    root = _pipeline_fixture(tmp_path, monkeypatch)

    result = runner.invoke(app, ["push", "demo-model", "--root", root])

    assert result.exit_code == 0, result.output
    assert _FakeSink.pushes == []
    assert "already has 7 recorded" in _flatten(result.output)


def test_push_reports_malformed_artifact_json_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrupt policy.json fails through the CLI, not with a raw traceback."""
    root = str(tmp_path / ".wmo")
    model_dir = _write_model_dir(root)
    (model_dir / "policy.json").write_text("{not json", encoding="utf-8")
    (model_dir / "report.json").write_text("{}", encoding="utf-8")
    _connected()
    _RecordingPushClient.calls = []
    monkeypatch.setattr("wmo.cli.platform_cmds.PlatformClient", _RecordingPushClient)

    result = runner.invoke(app, ["push", "demo-model", "--root", root])

    _assert_clean_failure(result)
    assert "not valid JSON" in _flatten(result.output)


def test_push_warns_when_the_existing_endpoint_links_elsewhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A same-named endpoint tied to another simulation must not silently lie."""
    root = str(tmp_path / ".wmo")
    model_dir = _write_model_dir(root)
    (model_dir / "policy.json").write_text(
        '{"kind": "static", "default_model": "m1", "pool": []}', encoding="utf-8"
    )
    (model_dir / "report.json").write_text('{"headline": {}}', encoding="utf-8")
    _connected()

    class _LinkedElsewhereClient(_RecordingPushClient):
        def get_endpoint(self, org_id: str, name: str) -> dict[str, object] | None:
            del org_id
            return {"name": name, "world_model_id": "wm-other"}

    _LinkedElsewhereClient.calls = []
    monkeypatch.setattr("wmo.cli.platform_cmds.PlatformClient", _LinkedElsewhereClient)

    result = runner.invoke(app, ["push", "demo-model", "--root", root])

    assert result.exit_code == 0, result.output
    kinds = [kind for kind, _ in _LinkedElsewhereClient.calls]
    assert kinds == ["bundle", "artifacts"]
    assert "linked to a different simulation" in _flatten(result.output)


def test_push_skip_message_names_the_partial_replay_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted earlier push is recoverable from the skip message itself."""
    _FakeReader.count = 1  # fewer than the derivation's 2: possibly partial
    root = _pipeline_fixture(tmp_path, monkeypatch)

    result = runner.invoke(app, ["push", "demo-model", "--root", root])

    assert result.exit_code == 0, result.output
    assert _FakeSink.pushes == []
    flat = _flatten(result.output)
    assert "wmo runs backfill" in flat
    assert "--force" in flat


class _PullStateClient(_StubClient):
    """Client double for the current-state pull: canned registry and policy reads."""

    model_names: tuple[str, ...] = ()
    endpoint_names: tuple[str, ...] = ()
    policy_payload: dict[str, object] = {}
    bank_bytes: bytes | None = None

    def list_world_models(self, _org_id: str) -> list[RemoteWorldModel]:
        return [
            RemoteWorldModel(id=f"wm-{name}", name=name, status="ready")
            for name in type(self).model_names
        ]

    def get_endpoint(self, _org_id: str, name: str) -> dict[str, object] | None:
        return {"name": name} if name in type(self).endpoint_names else None

    def download_endpoint_policy(
        self, _org_id: str, name: str, *, bank_dest: Path | None = None
    ) -> dict[str, object] | None:
        if name not in type(self).endpoint_names:
            return None
        payload = dict(type(self).policy_payload)
        bank_bytes = type(self).bank_bytes
        if bank_dest is not None and bank_bytes is not None:
            bank_dest.write_bytes(bank_bytes)
        return payload


def test_pull_model_overwrites_artifacts_with_the_endpoint_current_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pull hands back what the endpoint serves NOW, not what was pushed."""
    root = str(tmp_path / ".wmo")
    _connected()
    _PullStateClient.model_names = ("demo-model",)
    _PullStateClient.endpoint_names = ("demo-model",)
    _PullStateClient.policy_payload = {
        "policy": {"kind": "static", "default_model": "current-pick", "pool": []},
        "report": {"headline": {"accuracy": 0.9}},
        "bank": None,
    }
    _PullStateClient.bank_bytes = None
    monkeypatch.setattr("wmo.cli.platform_cmds.PlatformClient", _PullStateClient)

    def fake_pull_model(
        _client: object, _org_id: str, name: str, root_dir: str, *, force: bool
    ) -> None:
        del force
        model_dir = Path(root_dir) / "models" / name
        model_dir.mkdir(parents=True)
        # The bundle carries the AS-PUSHED policy; the artifacts leg must replace it.
        (model_dir / "policy.json").write_text(
            '{"kind": "static", "default_model": "as-pushed"}', encoding="utf-8"
        )

    monkeypatch.setattr("wmo.cli.platform_cmds._pull_model", fake_pull_model)

    result = runner.invoke(app, ["pull", "demo-model", "--root", root])

    assert result.exit_code == 0, result.output
    model_dir = Path(root) / "models" / "demo-model"
    policy = json.loads((model_dir / "policy.json").read_text(encoding="utf-8"))
    assert policy["default_model"] == "current-pick"
    report = json.loads((model_dir / "report.json").read_text(encoding="utf-8"))
    assert report["headline"] == {"accuracy": 0.9}
    assert "current" in _flatten(result.output)


def test_pull_reaches_an_endpoint_that_was_never_pushed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A platform-created endpoint with no simulation still pulls its artifacts."""
    root = str(tmp_path / ".wmo")
    _connected()
    _PullStateClient.model_names = ()
    _PullStateClient.endpoint_names = ("hosted-only",)
    _PullStateClient.policy_payload = {
        "policy": {"kind": "knn", "default_model": "m1", "pool": []},
        "report": {"headline": {"accuracy": 0.7}},
        "bank": {"sha256": "abc", "byte_size": 3},
    }
    _PullStateClient.bank_bytes = b"npz"
    monkeypatch.setattr("wmo.cli.platform_cmds.PlatformClient", _PullStateClient)

    result = runner.invoke(app, ["pull", "hosted-only", "--root", root])

    assert result.exit_code == 0, result.output
    model_dir = Path(root) / "models" / "hosted-only"
    assert json.loads((model_dir / "policy.json").read_text(encoding="utf-8"))["kind"] == "knn"
    assert (model_dir / "policy.json.bank.npz").read_bytes() == b"npz"
    assert (model_dir / "report.json").is_file()
    assert "no simulation to pull" in _flatten(result.output)


def test_pull_removes_a_stale_bank_when_the_current_policy_has_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bankless current policy must not keep an older fit's sidecar beside it."""
    root = str(tmp_path / ".wmo")
    _connected()
    _PullStateClient.model_names = ()
    _PullStateClient.endpoint_names = ("hosted-only",)
    _PullStateClient.policy_payload = {
        "policy": {"kind": "static", "default_model": "m1", "pool": []},
        "report": None,
        "bank": None,
    }
    _PullStateClient.bank_bytes = None
    monkeypatch.setattr("wmo.cli.platform_cmds.PlatformClient", _PullStateClient)
    stale = Path(root) / "models" / "hosted-only" / "policy.json.bank.npz"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old fit")

    result = runner.invoke(app, ["pull", "hosted-only", "--root", root])

    assert result.exit_code == 0, result.output
    assert not stale.exists()
    assert "stale policy.json.bank.npz" in _flatten(result.output)


def test_pull_removes_a_stale_report_when_the_endpoint_has_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reportless current endpoint must not keep old evidence reading as current."""
    root = str(tmp_path / ".wmo")
    _connected()
    _PullStateClient.model_names = ()
    _PullStateClient.endpoint_names = ("hosted-only",)
    _PullStateClient.policy_payload = {
        "policy": {"kind": "static", "default_model": "m1", "pool": []},
        "report": None,
        "bank": None,
    }
    _PullStateClient.bank_bytes = None
    monkeypatch.setattr("wmo.cli.platform_cmds.PlatformClient", _PullStateClient)
    stale = Path(root) / "models" / "hosted-only" / "report.json"
    stale.parent.mkdir(parents=True)
    stale.write_text('{"headline": {"accuracy": 0.99}}', encoding="utf-8")

    result = runner.invoke(app, ["pull", "hosted-only", "--root", root])

    assert result.exit_code == 0, result.output
    assert not stale.exists()
    assert "stale report.json" in _flatten(result.output)
