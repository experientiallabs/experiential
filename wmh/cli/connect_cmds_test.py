"""Tests for `wmh connect` / `wmh context`: a fake in-test connector, no network.

Every test points `WMH_CONNECTORS_PATH` at a tmp file and swaps the connector registry for a
fake, so the CLI surface is exercised end to end (connect, pull, list, show, attach) without a
real service, a real credential file, or a real model build.
"""

from __future__ import annotations

import httpx
import pytest
from typer.testing import CliRunner

import wmh.connect.connector as connector_registry
from wmh.cli import app
from wmh.config import save_config
from wmh.config.config import HarnessConfig
from wmh.connect import ConnectorAuth, ConnectUI, ContextItem, ItemKind, PullQuery
from wmh.connect.brave import BraveConnector
from wmh.connect.credentials import (
    ENV_CONNECTORS_PATH,
    connectors_path,
    load_connector_auth,
    save_connector_auth,
)

runner = CliRunner()


class FakeConnector:
    """A canned connector: token-prompt auth, two-item pulls, call recording."""

    name = "fake"
    label = "Fake Service"

    def __init__(self) -> None:
        self.pulled: list[PullQuery] = []

    def connect(self, ui: ConnectUI) -> ConnectorAuth:
        ui.info("starting fake auth")
        secret = ui.prompt_secret("paste the fake token")
        return ConnectorAuth(kind="token", access_token=secret or "tok")

    def verify(self, auth: ConnectorAuth) -> str:
        return "fake-user @ fake-org"

    def pull(self, auth: ConnectorAuth, query: PullQuery) -> list[ContextItem]:
        self.pulled.append(query)
        items = [
            ContextItem(
                id="1",
                source=self.name,
                kind=ItemKind.DOCUMENT,
                title="Doc One",
                body="hello world",
                created_at="2026-07-01T00:00:00+00:00",
                updated_at="2026-07-02T00:00:00+00:00",
            ),
            ContextItem(
                id="2", source=self.name, kind=ItemKind.MESSAGE, title="Msg Two", body="hi"
            ),
        ]
        return items[: query.limit]


@pytest.fixture
def fake_connector(monkeypatch, tmp_path) -> FakeConnector:  # noqa: ANN001 - pytest fixtures
    """Tmp credential file + a registry holding only the fake connector."""
    monkeypatch.setenv(ENV_CONNECTORS_PATH, str(tmp_path / "connectors.toml"))
    monkeypatch.delenv("WMH_FAKE_TOKEN", raising=False)
    fake = FakeConnector()
    monkeypatch.setattr(connector_registry, "_CONNECTORS", {"fake": fake})
    return fake


def _pull(tmp_path, name: str = "bundle-a") -> None:  # noqa: ANN001 - pytest fixture path
    result = runner.invoke(app, ["context", "pull", "fake", "--dir", str(tmp_path), "--name", name])
    assert result.exit_code == 0, result.output


def _connected() -> None:
    save_connector_auth(
        "fake", ConnectorAuth(kind="token", access_token="tok", account="fake-user @ fake-org")
    )


# -- wmh connect ----------------------------------------------------------------------------


def test_connect_bare_lists_connectors_with_status(fake_connector, monkeypatch) -> None:  # noqa: ANN001
    other = FakeConnector()
    other.name = "other"
    monkeypatch.setattr(connector_registry, "_CONNECTORS", {"fake": fake_connector, "other": other})
    _connected()

    result = runner.invoke(app, ["connect"])
    assert result.exit_code == 0, result.output
    assert "Fake Service" in result.output
    assert "fake-user @ fake-org" in result.output  # stored credential status
    assert "not connected" in result.output  # the other connector has none
    assert "wmh connect <name>" in result.output


def test_connect_bare_flags_an_incomplete_credential(fake_connector) -> None:  # noqa: ANN001
    """An aborted OAuth flow can leave an empty-token placeholder; the table must say so."""
    save_connector_auth("fake", ConnectorAuth(kind="oauth", access_token=""))
    result = runner.invoke(app, ["connect"])
    assert result.exit_code == 0, result.output
    assert "incomplete" in result.output
    assert "wmh connect fake" in result.output


def test_connect_service_runs_flow_verifies_and_saves(fake_connector) -> None:  # noqa: ANN001
    result = runner.invoke(app, ["connect", "fake"], input="s3cret\n")
    assert result.exit_code == 0, result.output
    assert "fake-user @ fake-org" in result.output

    stored = load_connector_auth("fake")
    assert stored is not None
    assert stored.access_token == "s3cret"
    assert stored.account == "fake-user @ fake-org"  # verify() identity stamped on save


def test_connect_env_token_is_never_written_to_disk(fake_connector, monkeypatch) -> None:  # noqa: ANN001
    """An env-injected token must stay off disk (the documented CI/headless contract)."""
    monkeypatch.setenv("WMH_FAKE_TOKEN", "env-tok")
    result = runner.invoke(app, ["connect", "fake"], input="env-tok\n")
    assert result.exit_code == 0, result.output
    flattened = " ".join(result.output.split())
    assert "fake-user @ fake-org" in flattened
    assert "$WMH_FAKE_TOKEN" in flattened
    assert "nothing written to disk" in flattened
    assert not connectors_path().exists()


@pytest.fixture
def brave_registry(monkeypatch, tmp_path):  # noqa: ANN001, ANN201 - pytest fixtures
    """Tmp credential file + a registry holding a real BraveConnector on a mock transport."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Subscription-Token"] == "brv-env"
        return httpx.Response(200, json={"web": {"results": []}})

    monkeypatch.setenv(ENV_CONNECTORS_PATH, str(tmp_path / "connectors.toml"))
    monkeypatch.delenv("WMH_BRAVE_TOKEN", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    brave = BraveConnector(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(connector_registry, "_CONNECTORS", {"brave": brave})
    return brave


def test_connect_brave_env_key_is_never_written_to_disk(brave_registry, monkeypatch) -> None:  # noqa: ANN001
    """BRAVE_SEARCH_API_KEY follows the same no-write contract as WMH_<NAME>_TOKEN."""
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brv-env")

    result = runner.invoke(app, ["connect", "brave"])
    assert result.exit_code == 0, result.output
    flattened = " ".join(result.output.split())
    assert "$BRAVE_SEARCH_API_KEY" in flattened
    assert "nothing written to disk" in flattened
    assert not connectors_path().exists()


def test_connect_table_shows_the_brave_env_key_status(brave_registry, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brv-env")

    result = runner.invoke(app, ["connect"])
    assert result.exit_code == 0, result.output
    joined = "".join(result.output.split())  # rich may wrap the table cell
    assert "($BRAVE_SEARCH_API_KEY)" in joined


def test_context_pull_brave_without_auth_names_both_env_vars(brave_registry, tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(app, ["context", "pull", "brave", "--dir", str(tmp_path)])
    assert result.exit_code == 2, result.output
    joined = "".join(result.output.split())
    assert "wmhconnectbrave" in joined
    assert "WMH_BRAVE_TOKEN" in joined
    assert "BRAVE_SEARCH_API_KEY" in joined


def test_context_pull_brave_uses_the_env_key(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    """A pull authorized purely by the deployed env key works end to end (mock transport)."""
    monkeypatch.setenv(ENV_CONNECTORS_PATH, str(tmp_path / "connectors.toml"))
    monkeypatch.delenv("WMH_BRAVE_TOKEN", raising=False)
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brv-env")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Subscription-Token"] == "brv-env"
        row = {"title": "Hit", "url": "https://example.com/hit", "description": "snippet"}
        return httpx.Response(200, json={"web": {"results": [row]}})

    def fetch(url: str, headers: dict[str, str]) -> str:
        del url, headers
        return "<p>fetched body</p>"

    brave = BraveConnector(transport=httpx.MockTransport(handler), fetch=fetch)
    monkeypatch.setattr(connector_registry, "_CONNECTORS", {"brave": brave})

    result = runner.invoke(
        app,
        ["context", "pull", "brave", "--dir", str(tmp_path), "--query", "wmh", "--name", "b"],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / ".wmh" / "context" / "b" / "items.jsonl").exists()
    assert not connectors_path().exists()


def test_connect_unknown_service_is_a_usage_error(fake_connector) -> None:  # noqa: ANN001
    result = runner.invoke(app, ["connect", "nope"])
    assert result.exit_code == 2, result.output
    assert "no connector named 'nope'" in result.output
    assert "fake" in result.output  # lists what is available


def test_connect_remove_deletes_the_stored_credential(fake_connector) -> None:  # noqa: ANN001
    _connected()
    removed = runner.invoke(app, ["connect", "fake", "--remove"])
    assert removed.exit_code == 0, removed.output
    assert "removed" in removed.output.lower()
    assert load_connector_auth("fake") is None

    again = runner.invoke(app, ["connect", "fake", "--remove"])
    assert again.exit_code == 0, again.output
    assert "nothing stored" in again.output.lower()


# -- wmh context pull -----------------------------------------------------------------------


def test_context_pull_saves_bundle_and_prints_summary(fake_connector, tmp_path) -> None:  # noqa: ANN001
    _connected()
    result = runner.invoke(
        app,
        [
            "context",
            "pull",
            "fake",
            "--dir",
            str(tmp_path),
            "--name",
            "bundle-a",
            "--target",
            "some/where",
            "--limit",
            "5",
        ],
    )
    assert result.exit_code == 0, result.output

    bundle_dir = tmp_path / ".wmh" / "context" / "bundle-a"
    assert (bundle_dir / "manifest.json").exists()
    assert (bundle_dir / "items.jsonl").exists()
    assert "document" in result.output and "message" in result.output  # per-kind summary
    assert str(bundle_dir) in result.output.replace("\n", "")  # rich wraps long paths
    assert fake_connector.pulled[0].target == "some/where"
    assert fake_connector.pulled[0].limit == 5


def test_context_pull_default_bundle_name_carries_the_service(fake_connector, tmp_path) -> None:  # noqa: ANN001
    _connected()
    result = runner.invoke(app, ["context", "pull", "fake", "--dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    names = [p.name for p in (tmp_path / ".wmh" / "context").iterdir()]
    assert len(names) == 1 and names[0].startswith("fake-")


def test_context_pull_without_auth_says_how_to_connect(fake_connector, tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(app, ["context", "pull", "fake", "--dir", str(tmp_path)])
    assert result.exit_code == 2, result.output
    assert "wmh connect fake" in result.output
    assert "WMH_FAKE_TOKEN" in result.output  # the env-token alternative


def test_context_pull_existing_bundle_needs_overwrite(fake_connector, tmp_path) -> None:  # noqa: ANN001
    _connected()
    _pull(tmp_path)
    clash = runner.invoke(
        app, ["context", "pull", "fake", "--dir", str(tmp_path), "--name", "bundle-a"]
    )
    assert clash.exit_code == 2, clash.output
    assert "--overwrite" in clash.output

    replaced = runner.invoke(
        app,
        ["context", "pull", "fake", "--dir", str(tmp_path), "--name", "bundle-a", "--overwrite"],
    )
    assert replaced.exit_code == 0, replaced.output


# -- wmh context list / show ------------------------------------------------------------------


def test_context_list_empty_state_says_how_to_pull(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(app, ["context", "list", "--dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "no context bundles" in result.output
    assert "wmh context pull" in result.output


def test_context_list_shows_manifests(fake_connector, tmp_path) -> None:  # noqa: ANN001
    _connected()
    _pull(tmp_path)
    result = runner.invoke(app, ["context", "list", "--dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "bundle-a" in result.output
    assert "fake" in result.output
    assert "2" in result.output  # item count


def test_context_show_renders_items_and_respects_limit(fake_connector, tmp_path) -> None:  # noqa: ANN001
    _connected()
    _pull(tmp_path)
    result = runner.invoke(app, ["context", "show", "bundle-a", "--dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "Doc One" in result.output and "Msg Two" in result.output
    assert "document" in result.output and "message" in result.output
    assert "2026-07-01" in result.output

    limited = runner.invoke(
        app, ["context", "show", "bundle-a", "--dir", str(tmp_path), "--limit", "1"]
    )
    assert limited.exit_code == 0, limited.output
    assert "Doc One" in limited.output and "Msg Two" not in limited.output
    assert "1 more" in limited.output  # the cut is visible


def test_context_show_missing_bundle_is_a_usage_error(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(app, ["context", "show", "nope", "--dir", str(tmp_path)])
    assert result.exit_code == 2, result.output
    assert "no context bundle named" in result.output


# -- wmh context attach -----------------------------------------------------------------------


def _model_with_knowledge(root) -> None:  # noqa: ANN001 - pytest fixture path
    save_config(HarnessConfig(knowledge=True), root=root / "models" / "m")


def test_context_attach_writes_a_knowledge_file(fake_connector, tmp_path) -> None:  # noqa: ANN001
    _connected()
    _pull(tmp_path)
    root = tmp_path / ".wmh-models"
    _model_with_knowledge(root)

    result = runner.invoke(
        app,
        [
            "context",
            "attach",
            "bundle-a",
            "--model",
            "m",
            "--root",
            str(root),
            "--dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output

    written = root / "models" / "m" / "knowledge" / "context-bundle-a.md"
    assert written.exists()
    text = written.read_text(encoding="utf-8")
    assert "# Context bundle: bundle-a" in text
    assert "Doc One" in text and "hello world" in text
    assert str(written) in result.output.replace("\n", "")  # rich wraps long paths
    assert "budget" in result.output  # says how the render budget treats the knowledge base


def test_context_attach_max_chars_drops_items_visibly(fake_connector, tmp_path) -> None:  # noqa: ANN001
    _connected()
    _pull(tmp_path)
    root = tmp_path / ".wmh-models"
    _model_with_knowledge(root)

    result = runner.invoke(
        app,
        [
            "context",
            "attach",
            "bundle-a",
            "--model",
            "m",
            "--root",
            str(root),
            "--dir",
            str(tmp_path),
            "--max-chars",
            "220",
        ],
    )
    assert result.exit_code == 0, result.output
    text = (root / "models" / "m" / "knowledge" / "context-bundle-a.md").read_text(encoding="utf-8")
    assert len(text) <= 220
    assert "items omitted" in text


def test_context_attach_refuses_a_model_without_knowledge_support(
    fake_connector,  # noqa: ANN001
    tmp_path,  # noqa: ANN001
) -> None:
    _connected()
    _pull(tmp_path)
    root = tmp_path / ".wmh-models"
    save_config(HarnessConfig(), root=root / "models" / "m")  # knowledge off, no knowledge/ dir

    result = runner.invoke(
        app,
        [
            "context",
            "attach",
            "bundle-a",
            "--model",
            "m",
            "--root",
            str(root),
            "--dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1, result.output
    assert "no knowledge base" in result.output
    assert "wmh build --knowledge" in result.output
    assert not (root / "models" / "m" / "knowledge").exists()


def test_context_attach_accepts_a_bare_knowledge_dir(fake_connector, tmp_path) -> None:  # noqa: ANN001
    """A knowledge/ dir someone created by hand counts as knowledge support."""
    _connected()
    _pull(tmp_path)
    root = tmp_path / ".wmh-models"
    model_dir = root / "models" / "m"
    save_config(HarnessConfig(), root=model_dir)
    (model_dir / "knowledge").mkdir()

    result = runner.invoke(
        app,
        [
            "context",
            "attach",
            "bundle-a",
            "--model",
            "m",
            "--root",
            str(root),
            "--dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (model_dir / "knowledge" / "context-bundle-a.md").exists()


def test_context_attach_unknown_model_is_a_usage_error(fake_connector, tmp_path) -> None:  # noqa: ANN001
    _connected()
    _pull(tmp_path)
    result = runner.invoke(
        app,
        [
            "context",
            "attach",
            "bundle-a",
            "--model",
            "ghost",
            "--root",
            str(tmp_path / ".wmh-models"),
            "--dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 2, result.output
    assert "no world model named" in result.output
