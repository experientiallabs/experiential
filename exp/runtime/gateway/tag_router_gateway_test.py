"""End-to-end tag routing: authoring, alias activation, a loaded gateway, a live classifier."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar, cast

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.common.models import load_model_catalog
from exp.common.project import ProjectConfig, ProjectStore
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.lifecycle_test import (
    _configured_project_pool,
    _granted_authorities,
    _ReadinessProjectRepository,
)
from exp.runtime.gateway.project_activation import ProjectActivation, load_project_activation
from exp.runtime.gateway.tag_router import (
    NO_RULE_MATCHED,
    TAG_ROUTER_ROUTE_REASON,
    TagRulesActivation,
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


def _rules(url: str) -> dict[str, object]:
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


def _author(root: Path, rules: dict[str, object]) -> str:
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


@pytest.mark.parametrize(("probability", "reason"), [(0.9, None), (0.1, NO_RULE_MATCHED)])
def test_a_tag_rules_alias_routes_by_the_live_classifier(
    tmp_path: Path, probability: float, reason: str | None
) -> None:
    """A coding request matches the rule; any other takes the default, and says so."""
    _manager, raw_key = _configured_project_pool(tmp_path)
    with _systemone(probability) as url:
        policy_id = _author(tmp_path, _rules(url))
        activation = load_project_activation(
            ProjectStore(tmp_path, "project-one").artifacts,
            project_ref="project-one",
            activation_ref=policy_id,
        )
        assert isinstance(activation, TagRulesActivation)
        # The fixture alias names activation-one; serve this policy under that reference.
        policy = activation.policy.model_copy(update={"policy_id": "activation-one"})
        served = TagRulesActivation(
            project_ref="project-one", activation_ref="activation-one", policy=policy
        )
        components = load_gateway_components(
            tmp_path,
            environment=_ENVIRONMENT,
            project_repository=_ReadinessProjectRepository(cast("ProjectActivation", served)),
        )
        assert components.unavailable_aliases == ()
        assert _granted_authorities(components, raw_key) == {"coding": "revision-project-one"}
        request = decode_chat(
            {"model": "coding", "messages": [{"role": "user", "content": "Fix the failing test."}]}
        ).request
        authorization = components.store.authorize_request(
            raw_key=raw_key,
            alias="coding",
            request=request,
            deadline_monotonic=time.monotonic() + 30.0,
        )
        route = components.routes.resolve_project_blocking(
            authorization=authorization,
            request=request,
            episode_namespace=("org", "identity", "revision-project-one", "episode"),
        )
    assert (route.route_reason, route.fallback_reason) == (TAG_ROUTER_ROUTE_REASON, reason)
    assert _FakeSystemOne.states == ["[latest user]\nFix the failing test."]


def test_alias_create_activates_a_stored_tag_rules_policy(tmp_path: Path) -> None:
    """The authored policy activates as a project alias; its candidates must be served."""
    manager, _raw_key = _configured_project_pool(tmp_path)
    policy_id = _author(tmp_path, _rules("https://decisions.example.invalid/systemone"))
    result = CliRunner().invoke(
        app,
        [
            "config",
            "gateway",
            "alias",
            "create",
            "tagged",
            "--project",
            "project-one",
            "--policy",
            policy_id,
            "--root",
            str(tmp_path),
            "--non-interactive",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    alias = next(item for item in manager.aliases() if item.alias_name == "tagged")
    assert (alias.target_kind, alias.activation_ref) == ("project", policy_id)
    catalog = load_model_catalog(tmp_path / "models.toml")
    assert {"primary", "secondary"} <= set(catalog.models)
