"""Tag-rules authoring end to end: router tags, alias create, a gateway, a live classifier."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest
from typer.testing import CliRunner, Result

from exp.cli.app import app
from exp.cli.gateway import router as gateway_router
from exp.common.core.artifacts import JsonObject
from exp.common.models import (
    ConnectionConfig,
    GatewayDeploymentCapabilities,
    GatewayTokenPrices,
    ModelCapabilities,
)
from exp.common.project import ProjectConfig, ProjectStore
from exp.runtime.gateway.catalog_authority import upsert_connection, upsert_singleton_deployment
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.project_activation import LocalArtifactProjectActivationRepository
from exp.runtime.gateway.tag_router import (
    NO_RULE_MATCHED,
    TAG_ROUTER_ROUTE_REASON,
    TAG_RULES_POLICY_ARTIFACT_TYPE,
    TagRulesRouterPolicy,
)
from exp.runtime.openai_protocol.requests import decode_chat

_ENVIRONMENT = {"PRIMARY_PROVIDER_KEY": "available", "SECONDARY_PROVIDER_KEY": "available"}


class _FakeSystemOne(BaseHTTPRequestHandler):
    """Answer every asked question with ``probability``."""

    probability: ClassVar[float] = 0.0
    states: ClassVar[list[str]] = []

    def do_POST(self) -> None:
        """Answer one decision request."""
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", "0"))))
        type(self).states.append(body["state"])
        answers = {
            key: {"type": "noul", "noul": type(self).probability} for key in body["questions"]
        }
        payload = json.dumps({"answers": answers}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib signature.
        """Keep test output quiet."""
        del format, args


@contextmanager
def _systemone(probability: float) -> Iterator[str]:
    """Serve the fake decision model and yield its SystemOne URL."""
    _FakeSystemOne.probability = probability
    _FakeSystemOne.states = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeSystemOne)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/systemone"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _rules(url: str) -> JsonObject:
    """Return a rules file routing coding requests to ``secondary``."""
    return {
        "candidates": ["primary", "secondary"],
        "default": "primary",
        "classifier": {
            "adapter_id": "task-tags",
            "kind": "decision_model",
            "url": url,
            "model": "test-decision-model",
            "categories": {
                "coding": {
                    "threshold": 0.7,
                    "questions": {"coding_work": {"instructions": "Is this coding work?"}},
                }
            },
        },
        "rules": [{"when_all": ["coding"], "route": "secondary"}],
    }


def _author(root: Path, rules: JsonObject) -> str:
    """Freeze the rules into project ``project-one`` through the CLI; return the policy ID."""
    ProjectStore(root, "project-one").initialize(ProjectConfig(project_id="project-one"))
    path = root / "rules.json"
    path.write_text(json.dumps(rules))
    command = ["config", "gateway", "router", "tags", str(path), "--project", "project-one"]
    result = CliRunner().invoke(app, [*command, "--root", str(root), "--json"])
    assert result.exit_code == 0, result.output
    receipt = json.loads(result.output)
    assert receipt["operation"] == "router.tags"
    assert receipt["changed"] is True
    again = CliRunner().invoke(app, [*command, "--root", str(root), "--json"])
    assert again.exit_code == 0, again.output
    assert json.loads(again.output)["resource_id"] == receipt["resource_id"]
    assert json.loads(again.output)["changed"] is False
    return str(receipt["resource_id"])


def _two_model_gateway(root: Path) -> GatewayManagement:
    """Initialize a gateway with one singleton deployment of each of two exact models."""
    manager = GatewayManagement(root)
    manager.initialize()
    for name in ("primary", "secondary"):
        upsert_connection(
            root,
            name=f"{name}-provider",
            connection=ConnectionConfig(
                provider="openai-compatible",
                base_url="http://127.0.0.1:9/v1",
                api_key_env=f"{name.upper()}_PROVIDER_KEY",
            ),
            replace=False,
        )
        upsert_singleton_deployment(
            root,
            deployment_alias=name,
            connection_name=f"{name}-provider",
            provider_model=f"{name}-model",
            exact_model_id=f"model-{name}",
            revision=None,
            capabilities=ModelCapabilities(),
            gateway_capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
            prices=GatewayTokenPrices(),
            pricing_source=None,
            replace=False,
        )
    manager.create_identity(identity_id="default", display_name="Default")
    return manager


def _alias_create(root: Path, project: str, *policy: str) -> Result:
    """Run ``alias create tagged --project PROJECT [--policy ID]`` and return the result."""
    command = ["config", "gateway", "alias", "create", "tagged", "--project", project]
    if policy:
        command.extend(["--policy", *policy])
    return CliRunner().invoke(app, [*command, "--root", str(root), "--non-interactive", "--json"])


@pytest.mark.parametrize(
    ("probability", "exact_model", "reason"),
    [(0.9, "model-secondary", None), (0.1, "model-primary", NO_RULE_MATCHED)],
)
def test_a_tag_rules_alias_routes_by_the_live_classifier(
    tmp_path: Path, probability: float, exact_model: str, reason: str | None
) -> None:
    """Authored, activated and served for real: the live answer picks the exact model."""
    manager = _two_model_gateway(tmp_path)
    with _systemone(probability) as url:
        policy_id = _author(tmp_path, _rules(url))
        created = _alias_create(tmp_path, "project-one", policy_id)
        assert created.exit_code == 0, created.output
        manager.add_grant(identity_id="default", alias_id="tagged")
        raw_key = manager.issue_key(identity_id="default", key_id="key-one").raw_key
        components = load_gateway_components(
            tmp_path,
            environment=_ENVIRONMENT,
            project_repository=LocalArtifactProjectActivationRepository(tmp_path),
        )
        assert components.unavailable_aliases == ()
        request = decode_chat(
            {"model": "tagged", "messages": [{"role": "user", "content": "Fix the failing test."}]}
        ).request
        authorization = components.store.authorize_request(
            raw_key=raw_key,
            alias="tagged",
            request=request,
            deadline_monotonic=time.monotonic() + 30.0,
        )
        route = components.routes.resolve_project_blocking(
            authorization=authorization,
            request=request,
            episode_namespace=("org", "identity", authorization.alias_revision_id, "episode"),
        )
    assert route.snapshot.exact_model_id == exact_model
    assert (route.route_reason, route.fallback_reason) == (TAG_ROUTER_ROUTE_REASON, reason)
    assert _FakeSystemOne.states == ["[latest user]\nFix the failing test."]


def test_alias_create_reports_a_missing_project_as_a_usage_error(tmp_path: Path) -> None:
    """A project that does not exist is a usage error, not a crash."""
    _two_model_gateway(tmp_path)
    result = _alias_create(tmp_path, "missing-project")
    assert result.exit_code == 2, result.output


def test_refreezing_after_an_upgrade_adopts_the_stored_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same routing content under a newer release is the same policy, unchanged."""
    _two_model_gateway(tmp_path)
    policy_id = _author(tmp_path, _rules("https://decisions.example.invalid/systemone"))
    monkeypatch.setattr(gateway_router, "installed_release_revision", lambda: "next-release")
    command = ["config", "gateway", "router", "tags", str(tmp_path / "rules.json")]
    result = CliRunner().invoke(
        app, [*command, "--project", "project-one", "--root", str(tmp_path), "--json"]
    )
    assert result.exit_code == 0, result.output
    receipt = json.loads(result.output)
    assert (receipt["resource_id"], receipt["changed"]) == (policy_id, False)


def test_refreezing_refuses_a_stored_policy_the_gateway_would_not_activate(
    tmp_path: Path,
) -> None:
    """A stored policy whose manifest disagrees with its payload is refused, not adopted."""
    _two_model_gateway(tmp_path)
    policy_id = _author(tmp_path, _rules("https://decisions.example.invalid/systemone"))
    authored = ProjectStore(tmp_path, "project-one")
    authored.load_project()
    payload = authored.artifacts.read_bytes(policy_id, "policy.json")
    policy = TagRulesRouterPolicy.model_validate_json(payload)
    planted = ProjectStore(tmp_path, "project-two")
    planted.initialize(ProjectConfig(project_id="project-two"))
    planted.artifacts.write(
        artifact_id=policy_id,
        artifact_type=TAG_RULES_POLICY_ARTIFACT_TYPE,
        envelope=policy.model_copy(update={"code_revision": "other-release"}),
        files={"policy.json": payload},
    )
    command = ["config", "gateway", "router", "tags", str(tmp_path / "rules.json")]
    result = CliRunner().invoke(
        app, [*command, "--project", "project-two", "--root", str(tmp_path), "--json"]
    )
    assert result.exit_code == 2, result.output
    assert policy_id in result.output
