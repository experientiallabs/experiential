"""Real HTTP wire admission, shared index reuse and fail-closed embedding spend."""

import asyncio
import json
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import httpx
import pytest

from exp.cli.build.app import (
    GroundedBuildCompletion,
    _complete_grounded_build,
    _embedding_cost_ceiling,
)
from exp.common.core.artifacts import JsonObject, sha256_bytes, sha256_json
from exp.common.models import ModelCapabilities, write_model_catalog
from exp.common.project import ProjectConfig, ProjectModelConfiguration, ProjectStore
from exp.common.traces.ingest.sources import load_trace_source
from exp.optimize.evaluation.spending import operation_spend_ceiling
from exp.runtime.models.providers.admitted_transport import AdmittedJsonTransport
from exp.runtime.models.providers.async_transport import HttpxAsyncJsonTransport
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    JsonHttpTransport,
    ProviderTransportError,
)
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.runtime.models.registry_test import _catalog
from exp.runtime.models.spend_ceiling import (
    SpendCeilingReached,
    install_spend_ceiling,
    operation_spend_scope,
)
from exp.runtime.sessions.dispatch_ledger import DispatchLedger
from exp.simulation.build import ProjectBuild, build_project
from exp.simulation.mining.service import MiningSpec
from exp.simulation.retrieval import load_rag_index


class Embeddings(JsonHttpTransport):
    """Return actual embeddings-wire responses while preserving sent request bodies."""

    def __init__(self, failures: int = 0) -> None:
        self.requests: list[JsonObject] = []
        self.failures = failures

    def post(
        self, url: str, *, headers: Mapping[str, str], payload: JsonObject, timeout_seconds: float
    ) -> JsonHttpResponse:
        self.requests.append(payload)
        if len(self.requests) <= self.failures:
            raise ProviderTransportError("uncertain transport result")
        texts = payload["input"]
        assert isinstance(texts, list)
        return JsonHttpResponse(
            200,
            {
                "data": [{"index": i, "embedding": [1.0, 0.0]} for i in range(len(texts))],
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            },
        )


def prepared(tmp_path: Path) -> tuple[ProjectStore, ProjectBuild, RuntimeModelCatalog, Embeddings]:
    """One actual native-style call produces one fit task and multiple unique key chunks."""
    catalog = _catalog(
        capabilities=ModelCapabilities(
            supports_embeddings=True,
            supports_tools=True,
            context_window_tokens=128000,
            maximum_output_tokens=64,
            input_cost_per_million_tokens_usd=1,
            output_cost_per_million_tokens_usd=1,
        )
    )
    write_model_catalog(tmp_path / "models.toml", catalog)
    store = ProjectStore(tmp_path, "embedding")
    store.initialize(ProjectConfig(project_id="embedding"))
    source = tmp_path / "traces.json"
    source.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Read the initial workbook."},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "inspect_workbook", "arguments": "{}"},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "c1", "content": "native result"},
                    {"role": "assistant", "content": "Done"},
                ]
            }
        )
    )
    completed = build_project(
        load_trace_source("chat-json", source),
        store,
        created_at=datetime.now(UTC),
        code_revision="fixture",
        mining_spec=MiningSpec(fit_task_budget=1, held_out_task_budget=0),
    )
    inner = Embeddings()
    runtime = RuntimeModelCatalog(
        catalog, environment={"FIXTURE_API_KEY": "secret"}, transport_factory=lambda: inner
    )
    return store, completed, runtime, inner


def run(
    store: ProjectStore, completed: ProjectBuild, runtime: RuntimeModelCatalog
) -> GroundedBuildCompletion:
    """Invoke the exact CLI index seam with the exact retry-inclusive quote."""
    snapshot, caps = runtime.snapshot("fixture-model")
    estimate = _embedding_cost_ceiling(completed, caps)
    assert estimate is not None
    return _complete_grounded_build(
        store,
        completed,
        selected=ProjectModelConfiguration(
            world_model="fixture-model", judge="fixture-model", embedder="fixture-model"
        ),
        runtime_catalog=runtime,
        world_snapshot=snapshot,
        embedder_snapshot=snapshot,
        top_k=5,
        estimate=estimate,
        maximum_build_cost_usd=estimate,
        provider_spend_authorized=True,
    )


def test_index_wire_receipt_spending_and_both_indexes_replay(tmp_path: Path) -> None:
    """One real wire batch accounts input-only usage; fit/serving and replay share vectors."""
    store, completed, runtime, inner = prepared(tmp_path)
    result = run(store, completed, runtime)
    assert len(inner.requests) == 1
    assert not result.reused
    for reference in (result.artifacts.fit_rag, result.artifacts.serving_rag):
        assert load_rag_index(store.artifacts, reference.artifact_id).index.transition_count
    receipt_path = next(store.paths.runtime_directory.glob("spend/build-embeddings-*/receipt.json"))
    receipt = json.loads(receipt_path.read_text())
    assert receipt["roles"][0]["role"] == "fixture-model"
    assert receipt["roles"][0]["attempts"] == 1
    assert receipt["settled_usd"] == pytest.approx(5 / 1_000_000)
    intents = [
        i for i in store.artifacts.list_ids() if i.startswith("provider-") and i.endswith("-intent")
    ]
    assert len(intents) == 1
    physical = json.loads(store.artifacts.read_bytes(intents[0], "receipt.json"))
    assert physical["request_sha256"] == sha256_json(inner.requests[0])
    sent = cast(list[str], inner.requests[0]["input"])
    assert physical["request_body"]["input"] == [
        {"text_sha256": sha256_bytes(text.encode("utf-8")), "characters": len(text)}
        for text in sent
    ]
    assert physical["canonical_request"] is not None
    assert "secret" not in json.dumps(physical)
    replay = run(store, completed, runtime)
    assert replay.reused and len(inner.requests) == 1


def test_cap_refuses_before_any_http_send(tmp_path: Path) -> None:
    """The actual CLI phase inherits an existing zero operation cap before its first POST."""
    store, completed, runtime, inner = prepared(tmp_path)
    ceiling = operation_spend_ceiling(store, "zero")
    ceiling.authorize("zero", ceiling_usd=0, unpriced_attempts=0)
    with operation_spend_scope():
        install_spend_ceiling(ceiling)
        with pytest.raises(SpendCeilingReached):
            run(store, completed, runtime)
    assert not inner.requests
    assert not [i for i in store.artifacts.list_ids() if i.startswith("provider-")]
    assert not ceiling.summary().roles


def test_default_async_wire_is_admitted(tmp_path: Path) -> None:
    """The production async transport still crosses admission inside its compatibility worker."""
    store, completed, runtime, _ = prepared(tmp_path)
    sent: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "data": [{"index": i, "embedding": [1.0, 0.0]} for i in range(len(body["input"]))],
                "usage": {"prompt_tokens": 7, "total_tokens": 7},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    wrapped = runtime.with_transport_wrapper(lambda _: HttpxAsyncJsonTransport(client))
    try:
        run(store, completed, wrapped)
    finally:
        asyncio.run(client.aclose())
    assert len(sent) == 1
    receipt = json.loads(
        next(
            store.paths.runtime_directory.glob("spend/build-embeddings-*/receipt.json")
        ).read_text()
    )
    assert receipt["roles"][0]["attempts"] == 1 and receipt["settled_usd"] == pytest.approx(
        7 / 1_000_000
    )


def test_unknown_retry_retains_liability_and_counts_every_physical_send(tmp_path: Path) -> None:
    """A retry is another admitted attempt; the unknown first charge is never erased."""
    store, completed, runtime, inner = prepared(tmp_path)
    inner.failures = 1
    run(store, completed, runtime)
    assert len(inner.requests) == 2
    receipt = json.loads(
        next(
            store.paths.runtime_directory.glob("spend/build-embeddings-*/receipt.json")
        ).read_text()
    )
    assert receipt["roles"][0]["attempts"] == 2
    assert receipt["unknown_usd"] > 0
    assert receipt["settled_usd"] > 0
    assert receipt["accounted_usd"] <= receipt["ceiling_usd"]


def test_partial_index_restart_reuses_vectors_without_another_send(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure after both indexes retains admitted vectors and resumes under the same phase."""
    from exp.cli.build import app as command

    store, completed, runtime, inner = prepared(tmp_path)
    persist = command.persist_grounded_world_model

    def fail(*_args: object, **_kwargs: object) -> None:
        raise ValueError("interrupted after indexed vectors")

    monkeypatch.setattr(command, "persist_grounded_world_model", fail)
    with pytest.raises(ValueError, match="interrupted"):
        run(store, completed, runtime)
    assert len(inner.requests) == 1 and store.load_project().build is None
    monkeypatch.setattr(command, "persist_grounded_world_model", persist)
    result = run(store, completed, runtime)
    assert not result.reused and len(inner.requests) == 1
    receipt = json.loads(
        next(
            store.paths.runtime_directory.glob("spend/build-embeddings-*/receipt.json")
        ).read_text()
    )
    assert receipt["roles"][0]["attempts"] == 1


def test_exhausted_unknown_attempts_cannot_redraw_on_restart(tmp_path: Path) -> None:
    """Unknown outcomes consume the same durable physical allowance across process-style resumes."""
    from exp.common.models.client import RequestAdmissionRefused

    store, completed, runtime, inner = prepared(tmp_path)
    inner.failures = 10
    # The dollar bound may stop even before all physical retry slots are consumed.
    with pytest.raises((ProviderTransportError, RequestAdmissionRefused)):
        run(store, completed, runtime)
    attempts = len(inner.requests)
    assert 1 <= attempts <= 3 and store.load_project().build is None
    with pytest.raises(RequestAdmissionRefused):
        run(store, completed, runtime)
    assert len(inner.requests) == attempts
    receipt = json.loads(
        next(
            store.paths.runtime_directory.glob("spend/build-embeddings-*/receipt.json")
        ).read_text()
    )
    assert receipt["roles"][0]["attempts"] == attempts
    assert receipt["unknown_usd"] > 0 and receipt["settled_usd"] == 0
    assert receipt["accounted_usd"] <= receipt["ceiling_usd"]


def test_inherited_operation_keeps_one_ledger_and_truthful_owner_receipt(tmp_path: Path) -> None:
    """Embedding sends settle the enclosing operation, without inventing another ledger owner."""
    from exp.cli.evaluation.spend_ceiling import enforced_ceiling

    store, completed, runtime, inner = prepared(tmp_path)
    with enforced_ceiling(
        store, "outer", stage="embedding", ceiling_usd=0.001, unpriced_attempts=0, command="fixture"
    ) as ceiling:
        run(store, completed, runtime)
        assert ceiling.summary().settled_usd == pytest.approx(5 / 1_000_000)
    assert len(inner.requests) == 1
    receipt = json.loads((store.paths.runtime_directory / "spend/outer/receipt.json").read_text())
    assert receipt["roles"][0]["attempts"] == 1
    assert not list(store.paths.runtime_directory.glob("spend/build-embeddings-*"))


def test_already_admitted_injection_refuses_before_either_owner_claims(tmp_path: Path) -> None:
    """An opaque existing physical owner cannot be nested or silently adopted by the build."""
    store, completed, runtime, inner = prepared(tmp_path)
    existing_ledger = DispatchLedger(tmp_path / "existing.sqlite", 3, "a" * 64)
    existing_ceiling = operation_spend_ceiling(store, "existing")
    existing_ceiling.authorize("existing", ceiling_usd=0.001, unpriced_attempts=0)
    existing = AdmittedJsonTransport(
        inner,
        ledger=existing_ledger,
        artifacts=store.artifacts,
        code_revision="fixture",
        deadline=datetime.now(UTC) + timedelta(minutes=5),
        spend_ceiling=existing_ceiling,
    )
    injected = runtime.with_transport_wrapper(lambda _: existing)
    with pytest.raises(ValueError, match="requires an unadmitted inner transport"):
        run(store, completed, injected)
    assert not inner.requests and existing_ledger.used() == 0
    assert not existing_ceiling.summary().roles
    outer_path = next(store.paths.runtime_directory.glob("build-embeddings-*/physical.sqlite"))
    with sqlite3.connect(outer_path.as_uri() + "?mode=ro", uri=True) as connection:
        assert connection.execute("SELECT COUNT(*) FROM invocation_slots").fetchone()[0] == 0
    receipt = json.loads(
        next(
            store.paths.runtime_directory.glob("spend/build-embeddings-*/receipt.json")
        ).read_text()
    )
    assert not receipt["roles"] and receipt["accounted_usd"] == 0
    assert not [i for i in store.artifacts.list_ids() if i.startswith("provider-")]


def unknown_runtime(store: ProjectStore, runtime: RuntimeModelCatalog) -> RuntimeModelCatalog:
    """Remove only tariff metadata; keep the exact provider and injected HTTP transport."""
    from exp.common.models import load_model_catalog

    catalog = load_model_catalog(store.model_catalog_path)
    model = catalog.models["fixture-model"]
    assert model.capabilities is not None
    unknown = model.model_copy(
        update={
            "capabilities": model.capabilities.model_copy(
                update={
                    "input_cost_per_million_tokens_usd": None,
                    "output_cost_per_million_tokens_usd": None,
                }
            )
        }
    )
    catalog = catalog.model_copy(update={"models": {"fixture-model": unknown}})
    write_model_catalog(store.model_catalog_path, catalog)
    return runtime.with_catalog(catalog)


def test_explicit_unknown_consent_authorizes_only_finite_physical_schedule(tmp_path: Path) -> None:
    """Unknown prices retain manual consent while actual unpriced sends are bounded and counted."""
    store, completed, runtime, inner = prepared(tmp_path)
    runtime = unknown_runtime(store, runtime)
    snapshot, caps = runtime.snapshot("fixture-model")
    assert _embedding_cost_ceiling(completed, caps) is None
    result = _complete_grounded_build(
        store,
        completed,
        selected=ProjectModelConfiguration(
            world_model="fixture-model", judge="fixture-model", embedder="fixture-model"
        ),
        runtime_catalog=runtime,
        world_snapshot=snapshot,
        embedder_snapshot=snapshot,
        top_k=5,
        estimate=None,
        maximum_build_cost_usd=5,
        provider_spend_authorized=True,
    )
    assert result.artifacts.world_model and len(inner.requests) == 1
    receipt = json.loads(
        next(
            store.paths.runtime_directory.glob("spend/build-embeddings-*/receipt.json")
        ).read_text()
    )
    assert receipt["ceiling_usd"] == 0 and receipt["unpriced_attempt_ceiling"] == 3
    assert receipt["unpriced_attempts"] == 1 and receipt["roles"][0]["unpriced_attempts"] == 1


def test_unknown_without_consent_and_inherited_unpriced_zero_both_refuse(tmp_path: Path) -> None:
    """A local manual grant cannot bypass the caller's false flag or an enclosing zero cap."""
    from exp.runtime.models.spend_ceiling import UnpricedDispatchRefused

    store, completed, runtime, inner = prepared(tmp_path)
    runtime = unknown_runtime(store, runtime)
    snapshot, _ = runtime.snapshot("fixture-model")

    def invoke(authorized: bool) -> GroundedBuildCompletion:
        return _complete_grounded_build(
            store,
            completed,
            selected=ProjectModelConfiguration(
                world_model="fixture-model", judge="fixture-model", embedder="fixture-model"
            ),
            runtime_catalog=runtime,
            world_snapshot=snapshot,
            embedder_snapshot=snapshot,
            top_k=5,
            estimate=None,
            maximum_build_cost_usd=5,
            provider_spend_authorized=authorized,
        )

    with pytest.raises(ValueError, match="explicit authorization"):
        invoke(False)
    assert not inner.requests
    outer = operation_spend_ceiling(store, "unknown-zero")
    outer.authorize("unknown-zero", ceiling_usd=0, unpriced_attempts=0)
    with operation_spend_scope():
        install_spend_ceiling(outer)
        with pytest.raises(UnpricedDispatchRefused):
            invoke(True)
    assert not inner.requests and outer.summary().unpriced_attempt_ceiling == 0
    assert not outer.summary().roles


@pytest.mark.parametrize("yes", [False, True])
def test_actual_cli_unknown_quote_requires_manual_yes_before_wire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, yes: bool
) -> None:
    """The real command retains unknown-price consent and then uses finite admitted HTTP."""
    from typer.testing import CliRunner

    from exp.cli.app import app
    from exp.cli.build import app as command
    from exp.common.models import load_model_catalog

    store, _, runtime, inner = prepared(tmp_path)
    unknown_runtime(store, runtime)
    catalog = load_model_catalog(store.model_catalog_path)
    from exp.common.models import ModelRoles

    catalog = catalog.model_copy(
        update={
            "roles": ModelRoles(
                world_model="fixture-model", judge="fixture-model", embedder="fixture-model"
            )
        }
    )
    model = catalog.models["fixture-model"]
    assert model.capabilities is not None
    catalog = catalog.model_copy(
        update={
            "models": {
                "fixture-model": model.model_copy(
                    update={
                        "capabilities": model.capabilities.model_copy(
                            update={"supports_structured_output": True}
                        )
                    }
                )
            }
        }
    )
    write_model_catalog(store.model_catalog_path, catalog)
    monkeypatch.setattr(
        command,
        "RuntimeModelCatalog",
        lambda _catalog: RuntimeModelCatalog(
            catalog, environment={"FIXTURE_API_KEY": "secret"}, transport_factory=lambda: inner
        ),
    )
    monkeypatch.setattr(command, "capture_build_completed", lambda **_kwargs: None)
    args = [
        "build",
        "cli-unknown",
        "--root",
        str(tmp_path),
        "--traces",
        str(tmp_path / "traces.json"),
        "--source",
        "chat-json",
        "--world-model",
        "fixture-model",
        "--judge",
        "fixture-model",
        "--embedder",
        "fixture-model",
        "--non-interactive",
    ]
    if yes:
        args.append("--yes")
    result = CliRunner().invoke(app, args)
    if not yes:
        assert result.exit_code != 0 and not inner.requests
        return
    assert result.exit_code == 0, result.output
    assert len(inner.requests) == 1
    actual = ProjectStore(tmp_path, "cli-unknown")
    assert actual.load_project().build is not None
    receipt = json.loads(
        next(
            actual.paths.runtime_directory.glob("spend/build-embeddings-*/receipt.json")
        ).read_text()
    )
    assert receipt["unpriced_attempt_ceiling"] == 3 and receipt["unpriced_attempts"] == 1
