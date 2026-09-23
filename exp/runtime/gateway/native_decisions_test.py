"""SQLite-backed native decision admission tests without provider calls."""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from exp.common.core.artifacts import JsonObject, sha256_json
from exp.common.models import (
    ConnectionConfig,
    GatewayDeploymentCapabilities,
    GatewayEquivalenceCertification,
    GatewayTokenPrices,
    ModelCapabilities,
    load_model_catalog,
    write_model_catalog,
)
from exp.common.models.gateway_chains import (
    GatewayDeploymentRung,
    GatewayModelChain,
    GatewayModelReferenceRung,
)
from exp.runtime.gateway.catalog_authority import (
    upsert_certified_pool,
    upsert_connection,
    upsert_singleton_deployment,
)
from exp.runtime.gateway.contracts import ProjectTarget
from exp.runtime.gateway.decisions_contracts import (
    MAX_DECISION_INPUT_BYTES,
    DecisionRequest,
    decode_decision_request,
)
from exp.runtime.gateway.ledger import SQLiteAttemptLedger
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.native_accounting import NativeBridgeError
from exp.runtime.gateway.native_bridge import NativeControlPlane
from exp.runtime.gateway.native_decisions import _admit_accepted
from exp.runtime.gateway.tests.chain_authority_fixture_test import (
    chain_components,
    publish_authored_chain_fixture,
)
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.typesafe import TypeSafeClient

_PRICES = GatewayTokenPrices(
    input_nano_usd_per_million_tokens=50_000_000,
    output_nano_usd_per_million_tokens=0,
)


def _control_plane(
    root: Path,
    *,
    supports_decisions: bool = True,
    prices: GatewayTokenPrices = _PRICES,
    rungs: int = 1,
) -> tuple[NativeControlPlane, str]:
    """Create real authority and ledger state with a certified native decision pool."""
    manager = GatewayManagement(root)
    manager.initialize()
    upsert_connection(
        root,
        name="provider-main",
        connection=ConnectionConfig(provider="typesafe", api_key_env="TEST_PROVIDER_KEY"),
        replace=False,
    )
    for index in range(rungs):
        normalized, snapshot, _changed = upsert_singleton_deployment(
            root,
            deployment_alias=f"decision-{index}",
            connection_name="provider-main",
            provider_model="systemone",
            exact_model_id="systemone-exact",
            revision=None,
            capabilities=ModelCapabilities(supports_completions=False, supports_embeddings=False),
            gateway_capabilities=GatewayDeploymentCapabilities(
                supports_decisions=supports_decisions
            ),
            prices=prices,
            pricing_source="test-fixture",
            replace=False,
        )
    assert rungs > 0
    pool_id = "decision-0"
    if rungs > 1:
        pool_id = "decision-pool"
        normalized, snapshot, _changed = upsert_certified_pool(
            root,
            pool_id=pool_id,
            exact_model_id="systemone-exact",
            deployment_aliases=tuple(f"decision-{index}" for index in range(rungs)),
            certification=GatewayEquivalenceCertification(
                certification_id="fixture-certification",
                provenance="test-only same-model fixture",
                evidence_sha256=sha256_json({"fixture": "same-model"}),
                certified_at=datetime(2026, 9, 16, tzinfo=UTC),
            ),
            expected_catalog_sha256=normalized.identity_sha256(),
            replace=False,
        )
    manager.activate_direct_alias(
        alias_id="decisions",
        alias_name="decisions",
        revision_id="revision-one",
        pool_id=pool_id,
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.create_identity(identity_id="default", display_name="Default")
    manager.add_grant(identity_id="default", alias_id="decisions")
    issued = manager.issue_key(identity_id="default", key_id="key-one")
    components = load_gateway_components(
        root, environment={"TEST_PROVIDER_KEY": "provider-secret-canary"}
    )
    return NativeControlPlane(components), issued.raw_key


def _body() -> JsonObject:
    """Return all supported question forms, including content-leak canaries."""
    return {
        "model": "decisions",
        "state": {"description": "private-state-canary"},
        "questions": {
            "safe": {
                "type": "noul",
                "instructions": "private-instructions-canary",
                "criteria": {"true": "allowed", "false": "blocked"},
            },
            "class": {
                "type": "choice",
                "instructions": ["Classify the state"],
                "criteria": {"yes": "appropriate", "no": None},
            },
            "quality": {
                "type": "score",
                "instructions": {"task": "Rate the state"},
                "criteria": ["poor", "good"],
            },
        },
    }


def _admit(control: NativeControlPlane, raw_key: str, body: JsonObject | None = None) -> JsonObject:
    """Call only the Python admission seam, with no provider HTTP work."""
    return json.loads(
        control.admit_decisions(
            json.dumps({"raw_key": raw_key, "body": json.dumps(_body() if body is None else body)})
        )
    )


def test_decisions_refuse_selected_model_chain_before_acceptance(tmp_path: Path) -> None:
    """A host receipt cannot authorize unsupported decision semantics for a selected chain root."""
    _control, key = _control_plane(tmp_path)
    catalog = load_model_catalog(tmp_path / "models.toml")
    root = catalog.models["decision-0"]
    assert root.gateway is not None
    child = root.model_copy(
        update={"gateway": root.gateway.model_copy(update={"exact_model_id": "other-decision"})}
    )
    chain = GatewayModelChain(
        model_id="systemone-exact",
        pool_id="decision-0",
        revision="chain-test",
        rungs=(
            GatewayDeploymentRung(deployment_id="decision-0"),
            GatewayModelReferenceRung(model_id="other-decision"),
        ),
    )
    write_model_catalog(
        tmp_path / "models.toml",
        catalog.model_copy(
            update={
                "models": {**catalog.models, "decision-child": child},
                "gateway_model_chains": {"systemone-exact": chain},
            }
        ),
    )

    publish_authored_chain_fixture(
        tmp_path, alias_id="decisions", revision_id="revision-chain", pool_id="decision-0"
    )
    control = NativeControlPlane(
        chain_components(tmp_path, environment={"TEST_PROVIDER_KEY": "provider-secret-canary"})
    )
    with pytest.raises(NativeBridgeError) as error:
        _admit(control, key)
    assert _public_error(error.value)["code"] == "model_chain_authority_unavailable"
    assert _rows(control) == []
    ledger = cast(SQLiteAttemptLedger, control._components.ledger)
    with sqlite3.connect(ledger.database_path) as connection:
        assert connection.execute("SELECT count(*) FROM gateway_attempts").fetchone()[0] == 0


def _rows(control: NativeControlPlane) -> list[tuple[str, str | None]]:
    """Read durable API surfaces and terminal states, not content-bearing traffic."""
    ledger = cast("SQLiteAttemptLedger", control._components.ledger)  # noqa: SLF001
    with sqlite3.connect(ledger.database_path) as connection:
        return connection.execute(
            "select api_surface, terminal_state from gateway_requests order by rowid"
        ).fetchall()


def _public_error(error: NativeBridgeError) -> JsonObject:
    """Decode one sanitized boundary error."""
    return json.loads(error.public_error_json)


def test_admit_decisions_preserves_questions_native_wire_and_reservation(tmp_path: Path) -> None:
    """Every dispatch uses SystemOne and keeps state separate from chat or continuation."""
    control, raw_key = _control_plane(tmp_path)
    admitted = _admit(control, raw_key)
    route = admitted["route"]
    assert isinstance(route, list) and len(route) == 1
    wire = route[0]
    assert isinstance(wire, dict)
    assert wire["dialect"] == "typesafe_systemone"
    assert wire["url"] == "https://api.typesafe.ai/v1/systemone"
    assert wire["headers"] == {
        "Authorization": "Bearer provider-secret-canary",
        "Content-Type": "application/json",
    }
    expected = _body()
    expected["model"] = "systemone"
    assert wire["upstream_payload"] == expected
    assert admitted["questions"] == expected["questions"]
    assert wire["upstream_body"] is None
    assert wire["throttle_redial_budget"] == 0
    assert admitted["maximum_same_deployment_attempts"] == 1
    assert admitted["maximum_total_attempts"] == 1
    assert "stream" not in admitted
    assert _rows(control) == [("decisions", None)]
    request_id = str(admitted["request_id"])
    entry = control._accounting.entry(request_id)  # noqa: SLF001
    assert entry is not None and isinstance(entry.request, DecisionRequest)
    assert entry.request.input_token_reservation > 0
    assert entry.request.output_token_reservation > 0
    assert entry.continuation is None
    assert entry.policy is None
    assert entry.throttle_redial_budgets == (0,)
    assert raw_key not in json.dumps(admitted)


def test_decision_attempt_reservation_and_settlement_use_reported_tokens(tmp_path: Path) -> None:
    """Native accounting settles a decision without accessing conversational fields."""
    control, raw_key = _control_plane(tmp_path)
    admitted = _admit(control, raw_key)
    started = json.loads(
        control.start_attempt(
            json.dumps(
                {
                    "request_id": admitted["request_id"],
                    "attempt_ordinal": 0,
                }
            )
        )
    )
    assert started["route_depth"] == 0
    assert (
        control.settle(
            json.dumps(
                {
                    "request_id": admitted["request_id"],
                    "attempt_id": started["attempt_id"],
                    "outcome": "completed",
                    "usage": {"input_tokens": 83, "output_tokens": 9},
                    "tool_names": [],
                    "failure": None,
                }
            )
        )
        == "{}"
    )
    assert _rows(control) == [("decisions", "completed")]
    report = json.loads(control.usage_json("{}"))
    assert report["totals"]["input_tokens"] == 83
    assert report["totals"]["output_tokens"] == 9


def test_admit_authenticates_before_decoding_and_never_echoes_state(tmp_path: Path) -> None:
    """Invalid credentials outrank malformed content; neither failure accepts a request."""
    control, raw_key = _control_plane(tmp_path)
    with pytest.raises(NativeBridgeError) as unauthenticated:
        control.admit_decisions(json.dumps({"raw_key": "not-a-key", "body": "{"}))
    assert _public_error(unauthenticated.value)["status_code"] == 401
    invalid = _body()
    invalid["questions"] = {"private-question-canary": {"type": "invalid"}}
    with pytest.raises(NativeBridgeError) as malformed:
        _admit(control, raw_key, invalid)
    public = malformed.value.public_error_json
    assert _public_error(malformed.value)["status_code"] == 400
    for canary in ("private-state-canary", "private-question-canary", raw_key):
        assert canary not in public
    assert _rows(control) == []


@pytest.mark.parametrize("field", ["messages", "stream", "tools", "temperature", "max_tokens"])
def test_admit_decisions_rejects_chat_controls_before_accept(tmp_path: Path, field: str) -> None:
    """Unsupported conversational controls cannot silently disappear into a decision call."""
    control, raw_key = _control_plane(tmp_path)
    body = _body()
    body[field] = True
    with pytest.raises(NativeBridgeError) as raised:
        _admit(control, raw_key, body)
    assert _public_error(raised.value)["status_code"] == 400
    assert _rows(control) == []


def test_admit_decisions_rejects_oversized_state_before_accept(tmp_path: Path) -> None:
    """The decoder's byte limit bounds native admission before any ledger write."""
    control, raw_key = _control_plane(tmp_path)
    body = _body()
    body["state"] = "x" * MAX_DECISION_INPUT_BYTES
    with pytest.raises(NativeBridgeError) as raised:
        _admit(control, raw_key, body)
    assert _public_error(raised.value)["status_code"] == 400
    assert _rows(control) == []


def test_admit_requires_deployment_decision_capability(tmp_path: Path) -> None:
    """A native URL does not substitute for the positive gateway-only capability flag."""
    control, raw_key = _control_plane(tmp_path, supports_decisions=False)
    with pytest.raises(NativeBridgeError) as raised:
        _admit(control, raw_key)
    assert _public_error(raised.value)["code"] == "unsupported_capability"
    assert _public_error(raised.value)["param"] == "model"
    assert _rows(control) == [("decisions", "failed")]


def test_admit_requires_native_decision_dialect_not_chat_wire(tmp_path: Path) -> None:
    """Even a declared decision deployment cannot dispatch through a chat profile."""
    control, raw_key = _control_plane(tmp_path)
    with patch.object(
        TypeSafeClient,
        "gateway_wire_profile",
        return_value=GatewayWireProfile(
            dialect="openai_compatible",
            url="https://example.test/v1/chat/completions",
            decisions_url="https://example.test/v1/systemone",
            model_id="systemone",
        ),
    ):
        with pytest.raises(NativeBridgeError) as raised:
            _admit(control, raw_key)
    assert _public_error(raised.value)["code"] == "unsupported_capability"
    assert _rows(control) == [("decisions", "failed")]


@pytest.mark.parametrize(
    "prices",
    [
        GatewayTokenPrices(),
        GatewayTokenPrices(input_nano_usd_per_million_tokens=50_000_000),
        GatewayTokenPrices(output_nano_usd_per_million_tokens=0),
        GatewayTokenPrices(
            input_nano_usd_per_million_tokens=50_000_000, output_nano_usd_per_million_tokens=1
        ),
    ],
)
def test_admit_requires_known_input_and_zero_output_price(
    tmp_path: Path,
    prices: GatewayTokenPrices,
) -> None:
    """Unknown or incompatible pricing is not silently interpreted as free usage."""
    control, raw_key = _control_plane(tmp_path, prices=prices)
    with pytest.raises(NativeBridgeError) as raised:
        _admit(control, raw_key)
    assert _public_error(raised.value)["status_code"] == 503
    assert _rows(control) == [("decisions", "failed")]


@pytest.mark.parametrize("rungs", [2, 9])
def test_admission_bounds_one_dispatch_per_certified_rung(tmp_path: Path, rungs: int) -> None:
    """Large pools are narrowed to eight attempts and no same-rung or throttle redials."""
    control, raw_key = _control_plane(tmp_path, rungs=rungs)
    admitted = _admit(control, raw_key)
    assert admitted["maximum_total_attempts"] == min(rungs, 8)
    assert admitted["maximum_same_deployment_attempts"] == 1
    route = admitted["route"]
    assert isinstance(route, list) and len(route) == min(rungs, 8)


def test_unexpected_post_accept_failure_is_terminal_and_content_free(tmp_path: Path) -> None:
    """An unexpected packaging failure cannot leave an accepted request open."""
    control, raw_key = _control_plane(tmp_path)
    with patch.object(
        DecisionRequest, "provider_body", side_effect=RuntimeError("private-state-canary")
    ):
        with pytest.raises(NativeBridgeError) as raised:
            _admit(control, raw_key)
    assert _public_error(raised.value)["status_code"] == 500
    assert "private-state-canary" not in raised.value.public_error_json
    assert _rows(control) == [("decisions", "failed")]


def test_project_targets_fail_closed_without_prompt_selection(tmp_path: Path) -> None:
    """An accepted project alias cannot run learned chat selection for a decision."""
    control, raw_key = _control_plane(tmp_path)
    request = decode_decision_request(json.dumps(_body())).request
    deadline = time.monotonic() + 120
    authorization = control._components.store.authorize_request(  # noqa: SLF001
        raw_key=raw_key, alias="decisions", request=request, deadline_monotonic=deadline
    )
    authorization = authorization.model_copy(
        update={
            "target": ProjectTarget(
                project_ref="fixture-project",
                activation_ref="fixture-activation",
                catalog_sha256=authorization.catalog_sha256,
            )
        }
    )
    control._write_ledger.accept_request(authorization=authorization)  # noqa: SLF001
    with pytest.raises(NativeBridgeError) as raised:
        _admit_accepted(control, authorization, request, deadline)
    assert _public_error(raised.value)["code"] == "unsupported_capability"
    assert _rows(control) == [("decisions", "failed")]


def test_keyed_decision_admissions_are_independent_not_replayed(tmp_path: Path) -> None:
    """An idempotency key creates no false replay promise for decision work."""
    control, raw_key = _control_plane(tmp_path)
    argument = json.dumps(
        {
            "raw_key": raw_key,
            "body": json.dumps(_body()),
            "idempotency_key": "same-caller-key",
        }
    )
    first = json.loads(control.admit_decisions(argument))
    second = json.loads(control.admit_decisions(argument))
    assert first["request_id"] != second["request_id"]
    assert _rows(control) == [("decisions", None), ("decisions", None)]
    assert "same-caller-key" not in json.dumps(first)
    assert "same-caller-key" not in json.dumps(second)


def test_decision_content_does_not_enter_durable_ledger(tmp_path: Path) -> None:
    """The real SQLite ledger stores decision usage and authority, never state or criteria."""
    control, raw_key = _control_plane(tmp_path)
    _admit(control, raw_key)
    ledger = cast("SQLiteAttemptLedger", control._components.ledger)  # noqa: SLF001
    with sqlite3.connect(ledger.database_path) as connection:
        persisted = "\\n".join(connection.iterdump())
    for canary in (
        "private-state-canary",
        "private-instructions-canary",
        "provider-secret-canary",
        raw_key,
    ):
        assert canary not in persisted


def test_decision_alias_rejects_chat_before_provider_payload(tmp_path: Path) -> None:
    """The shared conversational admission cannot convert SystemOne into chat."""
    control, raw_key = _control_plane(tmp_path)
    body = {"model": "decisions", "messages": [{"role": "user", "content": "state-canary"}]}
    with pytest.raises(NativeBridgeError) as raised:
        control.admit(json.dumps({"raw_key": raw_key, "body": json.dumps(body)}))
    assert _public_error(raised.value)["status_code"] == 400
    assert _public_error(raised.value)["code"] == "unsupported_capability"
    assert _rows(control) == [("chat_completions", "failed")]
