"""Build from an OTEL GenAI trace package, learn, measure held-out fidelity, derive a council.

Everything runs offline: one routed fake answers the OpenAI-compatible embedding and chat wire
and the native decision wire for ``exp build``, the learning pass, and the fidelity runs, so
every provider request is counted.
"""

import functools
import hashlib
import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from click import unstyle
from typer.testing import CliRunner

from exp.cli.app import app
from exp.common.core.artifacts import JsonObject, JsonValue
from exp.common.models import (
    BillingSource,
    ConnectionConfig,
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
    load_model_catalog,
    write_model_catalog,
)
from exp.common.models.catalog_roles import ModelRoles
from exp.common.project import ProjectStore
from exp.common.tasks.store import load_task_set
from exp.optimize.evaluation import fidelity
from exp.optimize.evaluation.fidelity import execute_fidelity_evaluation
from exp.optimize.evaluation.held_out_fidelity import (
    register_held_out_fidelity,
    saved_held_out_results,
    source_rag_inputs,
)
from exp.optimize.evaluation.trace_learning import learning_curve, learning_plan
from exp.optimize.evaluation.trace_replay import DEADLINE, aggregate
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    RecordedRequest,
    ScriptedJsonTransport,
)
from exp.simulation.retrieval import load_rag_index
from exp.simulation.retrieval.embedding import HashingRAGEmbedder
from exp.simulation.world_model.experience_memory import (
    consumed_task_lineages,
    is_reserved,
    memory_head,
)
from exp.simulation.world_model.runtime import load_grounded_world_model_artifact

PROJECT = "retail"
USERS = 12
REVISION = "c" * 40


def episode(index: int) -> JsonObject:
    """One synthetic retail support episode: a lookup prefix, then order reads and a cancel."""
    user = f"user_{index:02d}"
    order = f"#W{1000 + index}"
    status = "pending" if index % 3 else "delivered"
    return {
        "user": user,
        "instruction": f"I am {user}. Please cancel order {order} if it has not shipped.",
        "prefix": 1,
        "exchanges": [
            (
                "find_user_id_by_email",
                {"email": f"{user}@example.test"},
                {"user_id": user},
            ),
            (
                "get_order_details",
                {"order_id": order},
                {"order_id": order, "status": status, "total": 10.5 + index},
            ),
            (
                "get_user_details",
                {"user_id": user},
                {"user_id": user, "orders": [order], "payment": "gift_card"},
            ),
            (
                "cancel_pending_order",
                {"order_id": order, "reason": "no longer needed"},
                {"order_id": order, "status": "cancelled" if status == "pending" else status},
            ),
        ],
    }


def package_rows(episodes: list[JsonObject]) -> list[JsonObject]:
    """Encode episodes as flat OTEL GenAI span records, the documented package format.

    Each trace has a ``chat`` span with the customer request and the request context
    ``{"task_lineage": USER, "replay_from": N}``, then one call and one result span per
    recorded tool exchange, paired by ``gen_ai.tool.call.id``.
    """
    rows: list[JsonObject] = []
    start = datetime(2026, 10, 15, tzinfo=UTC)
    for item in episodes:
        user = str(item["user"])
        # A repeat of the same task (same user) is a different trace with its own identity.
        key = user + str(item.get("repeat", ""))
        trace_id = hashlib.sha256(key.encode()).hexdigest()[:32]
        context = {"task_lineage": user, "replay_from": item["prefix"]}
        rows.append(
            {
                "trace_id": trace_id,
                "span_id": hashlib.sha256((key + "chat").encode()).hexdigest()[:16],
                "name": "chat",
                "start_time": start.isoformat(),
                "end_time": start.isoformat(),
                "attributes": {
                    "gen_ai.operation.name": "chat",
                    "gen_ai.provider.name": "recording",
                    "gen_ai.request.model": "recorded-agent",
                    "gen_ai.prompt": item["instruction"],
                    "exp.conversation.id": user,
                    "exp.request.context": json.dumps(context),
                },
            }
        )
        exchanges = item["exchanges"]
        assert isinstance(exchanges, list)
        for position, (name, arguments, reply) in enumerate(exchanges):
            at = start + timedelta(seconds=2 * position + 1)
            call_id = f"{key}-call-{position}"
            for kind, offset, attributes in (
                (
                    "tool.call",
                    timedelta(),
                    {
                        "gen_ai.operation.name": "chat",
                        "gen_ai.provider.name": "recording",
                        "gen_ai.request.model": "recorded-agent",
                        "gen_ai.tool.call.arguments": json.dumps(arguments),
                    },
                ),
                (
                    "tool.result",
                    timedelta(milliseconds=500),
                    {
                        "gen_ai.operation.name": "execute_tool",
                        "gen_ai.tool.message": json.dumps(reply, sort_keys=True),
                    },
                ),
            ):
                rows.append(
                    {
                        "trace_id": trace_id,
                        "span_id": hashlib.sha256((call_id + kind).encode()).hexdigest()[:16],
                        "name": kind,
                        "start_time": (at + offset).isoformat(),
                        "end_time": (at + offset).isoformat(),
                        "attributes": {
                            **attributes,
                            "gen_ai.tool.name": name,
                            "gen_ai.tool.call.id": call_id,
                        },
                    }
                )
    return rows


def write_catalog(root: Path, *, recall: bool = True) -> None:
    """Configure an OpenAI-compatible simulator, judge and embedder, and a memory-recall alias.

    Without ``recall`` the catalog has no native decision model, so a build skips learning.
    """
    write_model_catalog(
        root / "models.toml",
        ModelCatalog(
            connections={
                "http": ConnectionConfig(
                    provider="openai-compatible",
                    base_url="https://fixture.invalid/v1",
                    api_key_env="HELD_OUT_TEST_KEY",
                ),
                "decisions": ConnectionConfig(
                    provider="typesafe",
                    base_url="https://fixture.invalid/v1",
                    api_key_env="HELD_OUT_TEST_KEY",
                    trusted_custom_origin=True,
                ),
            },
            models={
                "world": ModelRecord(
                    connection="http",
                    model="world-v1",
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    capabilities=ModelCapabilities(
                        supports_tools=True,
                        supports_structured_output=True,
                        context_window_tokens=200_000,
                        maximum_output_tokens=8_192,
                    ),
                ),
                "judge": ModelRecord(
                    connection="http",
                    model="judge-v1",
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    capabilities=ModelCapabilities(
                        supports_tools=True,
                        supports_structured_output=True,
                        context_window_tokens=200_000,
                        maximum_output_tokens=16_384,
                    ),
                ),
                "embed": ModelRecord(
                    connection="http",
                    model="embed-v1",
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    capabilities=ModelCapabilities(
                        supports_embeddings=True,
                        context_window_tokens=8_192,
                        input_cost_per_million_tokens_usd=0.02,
                    ),
                ),
                **(
                    {
                        "jev": ModelRecord(
                            connection="decisions",
                            model="jev-v1",
                            billing_source=BillingSource.CUSTOMER_MANAGED,
                            capabilities=ModelCapabilities(supports_completions=False),
                        )
                    }
                    if recall
                    else {}
                ),
            },
            roles=ModelRoles(world_model="world", judge="judge", embedder="embed"),
        ),
    )


LEARNABLE = "get_user_details"


class SimulatedProvider(ScriptedJsonTransport):
    """Answer every wire from recorded truth, with one learnable tool.

    Embeddings are deterministic token hashes, so retrieval is genuinely semantic. The
    simulator proposes each recorded reply exactly, except for tools in ``wrong`` (always
    wrong) and :data:`LEARNABLE` while ``learnable`` is set: that tool is wrong until memory
    delivered to the prediction mentions it, so learning visibly improves predictions.
    Investigations name the mismatched tool, compressions join their inputs, the judge scores
    every criterion 4, and native memory-recall decisions score every candidate.
    """

    def __init__(
        self,
        episodes: list[JsonObject],
        wrong: frozenset[str] = frozenset(),
        *,
        learnable: bool = False,
    ) -> None:
        """Index recorded replies by tool call."""
        super().__init__()
        self.truth: dict[str, JsonValue] = {}
        for item in episodes:
            exchanges = item["exchanges"]
            assert isinstance(exchanges, list)
            for name, arguments, reply in exchanges:
                self.truth[_call_key(name, arguments)] = reply
        self.wrong = wrong
        self.learnable = learnable
        self.embedder = HashingRAGEmbedder(64)

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Route one request by endpoint and requested structure."""
        del timeout_seconds
        self.requests.append(RecordedRequest(url, dict(headers), payload))
        if url.endswith("/embeddings"):
            texts = payload["input"]
            assert isinstance(texts, list)
            vectors = self.embedder.embed([str(text) for text in texts])
            return JsonHttpResponse(
                status_code=200,
                body={
                    "model": "embed-v1",
                    "data": [
                        {"index": i, "embedding": list(vector.values)}
                        for i, vector in enumerate(vectors)
                    ],
                    "usage": {"prompt_tokens": sum(len(str(text)) for text in texts)},
                },
            )
        if url.endswith("/systemone"):
            questions = payload["questions"]
            assert isinstance(questions, dict)
            answers: JsonObject = {
                key: {"type": "choice", "choice": "entity-state"}
                if key == "dependency"
                else {"type": "score", "score": 2}
                for key in questions
            }
            return JsonHttpResponse(status_code=200, body={"answers": answers})
        name = str(_at(payload, "tool_choice", "function", "name"))
        if name == "submit_transition":
            return _tool_reply("world-v1", name, self._transition(payload))
        schema = _obj(_at(payload, "tools", 0, "function", "parameters", "properties"))
        if "axes" in schema:
            return _tool_reply("judge-v1", name, _judgement(schema))
        if "summary" in schema:
            return _tool_reply("world-v1", name, {"summary": _compressed(payload)})
        return _tool_reply("world-v1", name, _investigation(payload))

    def _transition(self, payload: JsonObject) -> JsonObject:
        """Propose the recorded reply for every requested call, unless it is not known yet."""
        request = _user_content(payload)
        memory = json.dumps(request.get("research_history") or {})
        observations: list[JsonValue] = []
        for item in _arr(_at(request, "candidate_response", "tool_calls")):
            call = _obj(item)
            tool = str(call["name"])
            reply = self.truth[_call_key(tool, call["arguments"])]
            unknown = self.learnable and tool == LEARNABLE and LEARNABLE not in memory
            if tool in self.wrong or unknown:
                reply = {"error": "simulated mismatch"}
            observations.append({"call_id": call["call_id"], "reply": reply, "encoding": "json"})
        return {
            "observations": observations,
            "effect": {
                "kind": "edits",
                "edits": [{"op": "set", "path": ["calls"], "value": len(observations)}],
            },
        }

    def predictions(self) -> list[JsonObject]:
        """Every simulator request body sent so far."""
        return [
            _user_content(request.payload)
            for request in self.requests
            if request.url.endswith("/chat/completions")
            and _at(request.payload, "tool_choice", "function", "name") == "submit_transition"
        ]


def _at(value: JsonValue, *path: str | int) -> JsonValue:
    """Walk a JSON value along object keys and array indexes, asserting each step exists."""
    for key in path:
        if isinstance(key, int):
            value = _arr(value)[key]
        else:
            value = _obj(value)[key]
    return value


def _obj(value: JsonValue) -> JsonObject:
    """Narrow a JSON value to an object."""
    assert isinstance(value, dict)
    return value


def _arr(value: JsonValue) -> list[JsonValue]:
    """Narrow a JSON value to an array."""
    assert isinstance(value, list)
    return value


def _user_content(payload: JsonObject) -> JsonObject:
    """The JSON user message of a framed model request."""
    return _obj(json.loads(str(_at(payload, "messages", 1, "content"))))


def _investigation(payload: JsonObject) -> JsonObject:
    """Attribute every mismatch to the learnable tool's reply format."""
    return {
        "discrepancy": f"{LEARNABLE} reply differed from the prediction",
        "hypothesis": f"{LEARNABLE} returns user_id, orders and payment",
        "scope": f"{LEARNABLE} reads",
        "uncertainty": "One observation",
        "supersedes": [],
        "transferable": True,
        "observable_cue": "the same tool and arguments",
    }


def _compressed(payload: JsonObject) -> str:
    """Join the texts a compression request supplies, within the line limit."""
    content = _user_content(payload)
    rows = _arr(content.get("lines") or content.get("halves") or [])
    return ("S: " + " | ".join(str(_obj(row)["text"]) for row in rows))[:200]


def _judgement(schema: JsonObject) -> JsonObject:
    """Score every criterion, and every step when the rubric asks per turn."""
    criteria = list(_obj(_at(schema, "axes", "properties")))
    answer: JsonObject = {
        "axes": {
            key: {
                "status": "assessed",
                "score": 4,
                "rationale": "Fixture",
                "evidence_refs": ["step-0"],
                "critical_flags": [],
            }
            for key in criteria
        }
    }
    if "turns" in schema:
        answer["turns"] = {
            step: {"scores": dict.fromkeys(criteria, 4), "rationale": "Fixture"}
            for step in _obj(_at(schema, "turns", "properties"))
        }
    return answer


def _tool_reply(model: str, name: str, arguments: JsonObject) -> JsonHttpResponse:
    """A Chat Completions response carrying one forced tool call."""
    return JsonHttpResponse(
        status_code=200,
        body={
            "model": model,
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call",
                                "type": "function",
                                "function": {"name": name, "arguments": json.dumps(arguments)},
                            }
                        ],
                    },
                }
            ],
        },
    )


def _call_key(name: object, arguments: object) -> str:
    """Canonical identity of one recorded call."""
    return json.dumps({"name": name, "arguments": arguments}, sort_keys=True)


def build_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: SimulatedProvider,
    *flags: str,
    episodes: list[JsonObject] | None = None,
    learn: bool = False,
) -> tuple[Path, ProjectStore, str]:
    """Run ``exp build`` on the package through the routed fake and return its output.

    With ``learn`` the catalog has a memory-recall model, so the build learns every trace and
    retests its misses. Without it the build stops at the bare simulator; the recall alias is
    added afterwards so a test can drive learning itself.
    """
    root = tmp_path / ".exp"
    root.mkdir()
    write_catalog(root, recall=learn)
    episodes = episodes or [episode(i) for i in range(USERS)]
    package = tmp_path / "package.jsonl"
    package.write_text("".join(json.dumps(row) + "\n" for row in package_rows(episodes)))
    monkeypatch.setenv("HELD_OUT_TEST_KEY", "fixture-only")
    monkeypatch.setenv("EXP_RELEASE_REVISION", REVISION)
    monkeypatch.setattr(
        "exp.cli.build.app.RuntimeModelCatalog",
        functools.partial(RuntimeModelCatalog, transport_factory=lambda: provider),
    )
    monkeypatch.setattr("exp.cli.build.app.capture_build_completed", lambda **_kwargs: None)
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: provider)
    result = CliRunner().invoke(
        app,
        [
            "build",
            PROJECT,
            "--traces",
            str(package),
            "--source",
            "otel-genai",
            "--root",
            str(root),
            "--yes",
            "--non-interactive",
            *flags,
        ],
    )
    assert result.exit_code == 0, result.output
    if not learn:
        write_catalog(root)
    return root, ProjectStore(root, PROJECT), " ".join(result.output.split())


def build_report(store: ProjectStore) -> str:
    """The full learning and retest output ``exp build`` saved beside its memory."""
    lineage = learning_plan(store, scope="all").lineage
    report = store.paths.runtime_directory / "learning" / lineage / "report.txt"
    return " ".join(report.read_text(encoding="utf-8").split())


def audit_report(store: ProjectStore) -> str:
    """The full learning and panel output ``exp eval --audit`` saved beside its memory."""
    lineage = learning_plan(store, scope="fit").lineage
    report = store.paths.runtime_directory / "audits" / lineage / "report.txt"
    return " ".join(report.read_text(encoding="utf-8").split())


def held_out_command(root: Path, count: int, *flags: str) -> str:
    """Invoke ``exp eval PROJECT --audit N`` and return its output."""
    result = CliRunner().invoke(
        app,
        ["eval", PROJECT, "--root", str(root), "--audit", str(count)]
        + ["--yes", "--non-interactive", *flags],
    )
    assert result.exit_code == 0, result.output
    return " ".join(result.output.split())


def held_out_lineages(store: ProjectStore) -> set[str]:
    """Leakage lineages the build assigned to the held-out partition."""
    build = store.load_project().build
    assert build is not None
    return {
        task.lineage_group_id
        for task in load_task_set(store.artifacts, build.task_set.artifact_id).tasks
        if task.partition == "held_out"
    }


def held_out_customers(store: ProjectStore) -> set[str]:
    """Customers whose traces the build held out."""
    build = store.load_project().build
    assert build is not None
    return {
        str(task.initial_context["task_lineage"])
        for task in load_task_set(store.artifacts, build.task_set.artifact_id).tasks
        if task.partition == "held_out"
    }


def test_audit_learns_fit_traces_then_measures_held_out_on_final_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The audit: learn fit traces only with held-out reserved, then replay held-out traces."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)], learnable=True)
    root, store, _ = build_project(tmp_path, monkeypatch, provider)
    output = held_out_command(root, 3)
    plan = learning_plan(store, scope="fit")
    head = memory_head(store.artifacts, plan.lineage)
    assert head is not None
    fit = USERS - len(held_out_lineages(store))
    assert f"learned {fit}/{fit} fit traces · memory v{fit}" in output
    report = audit_report(store)
    assert "Learning curve exact replies by trace order:" in report
    assert "Report:" in output and "report.txt" in output
    assert re.search(r"Complete in (under 1s|\d+(s|m\d\ds|h\d\dm))", output), "time taken"
    # Held-out lineages were reserved before learning and never consumed.
    held = held_out_lineages(store)
    assert all(is_reserved(store.artifacts, plan.lineage, lineage) for lineage in held)
    assert not consumed_task_lineages(store.artifacts, head) & held
    # The first trace mispredicts the learnable tool; once memory names it, predictions match.
    curve = learning_curve(store, "fit")
    assert curve is not None and len(curve.points) == fit
    assert curve.points[0].exact == 2 and all(p.exact == 3 for p in curve.points[1:])
    assert curve.halves[0].exact < curve.halves[1].exact or fit < 4
    # Hybrid: predictions carry recorded fit facts, but never from their own trace's lineage
    # or from any held-out lineage.
    customers = held_out_customers(store)
    for prediction in provider.predictions():
        own = _at(prediction, "task", "initial_context", "task_lineage")
        contexts = _obj(prediction["grounding_contexts"]).values()
        lineages = {_at(context, "initial_context", "task_lineage") for context in contexts}
        assert own not in lineages and lineages.isdisjoint(customers)
    assert any(prediction["grounded_examples"] for prediction in provider.predictions())
    # The panel reads the final fit-only checkpoint, frozen.
    assert f"memory v{head.version}" in output
    assert "9/9 replies exact (100%)" in output and "judge 4.0/5" in output
    assert "judge agrees with exact replies on 9/9 turns (100%)" in output
    assert memory_head(store.artifacts, plan.lineage) == head

    purchased = len(provider.requests)
    replay = held_out_command(root, 3)
    assert "no new model calls (saved results)" in replay and "9/9 replies exact" in replay
    assert audit_report(store).count("Saved results. No provider calls.") == 2
    assert len(provider.requests) == purchased, "the learned audit replays without calls"


def test_identical_repeat_fit_traces_complete_the_audit_without_a_relearn_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeats are learned traces that publish no version, so learning is reported complete."""
    episodes = [
        *(episode(i) for i in range(USERS)),
        *({**episode(i), "repeat": "-again"} for i in range(USERS)),
    ]
    provider = SimulatedProvider(episodes, learnable=True)
    root, store, _ = build_project(tmp_path, monkeypatch, provider, episodes=episodes)
    output = held_out_command(root, 3)
    fit = USERS - len(held_out_lineages(store))
    assert f"learned {2 * fit}/{2 * fit} fit traces · memory v{fit}" in output
    assert "run the audit again" not in output


def _import_id(store: ProjectStore) -> str:
    """The stored trace import the project was built from."""
    import_id = store.load_project().trace_import_id
    assert import_id is not None
    return import_id


def test_source_rag_profile_measures_held_out_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The source-RAG profile: fit-only semantic retrieval and metered embeddings."""
    episodes = [episode(i) for i in range(USERS)]
    provider = SimulatedProvider(episodes, wrong=frozenset({"get_user_details"}))
    _, store, _ = build_project(tmp_path, monkeypatch, provider)
    built = len(provider.requests)
    customers = held_out_customers(store)
    panel = register_held_out_fidelity(
        store, 3, profile="source-rag", code_revision=REVISION, clock=lambda: datetime.now(UTC)
    )
    for prepared in panel.prepared:
        execute_fidelity_evaluation(store, prepared)
    total = aggregate(saved_held_out_results(store, panel))
    # Three traces, each replaying 3 calls after a 1-call public prefix; one tool is wrong.
    assert (total.exact, total.steps, total.judge_mean) == (6, 9, 4.0)
    # The fixture judge scores every turn 4, so it disagrees on the three wrong turns.
    assert (total.judge_agreements, total.judge_turns) == (6, 9)
    fidelity_requests = provider.requests[built:]
    embeddings = [r for r in fidelity_requests if r.url.endswith("/embeddings")]
    assert len(embeddings) == 9, "one budgeted query embedding per simulated turn"
    attempts = store.artifacts.list_ids(artifact_type="provider-physical-attempt")
    assert len([a for a in attempts if a.endswith("-intent")]) == len(fidelity_requests)
    for prediction in provider.predictions():
        contexts = _obj(prediction["grounding_contexts"]).values()
        lineages = {_at(context, "initial_context", "task_lineage") for context in contexts}
        assert lineages.isdisjoint(customers)
        assert {_at(tool, "name") for tool in _arr(_at(prediction, "task", "tools"))} == {
            "cancel_pending_order",
            "find_user_id_by_email",
            "get_order_details",
            "get_user_details",
        }


def test_fit_only_simulator_cannot_retrieve_any_held_out_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The source-RAG simulator's index holds every fit trace and no held-out trace."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)])
    _, store, _ = build_project(tmp_path, monkeypatch, provider)
    build = store.load_project().build
    assert build is not None
    world_model, cases = source_rag_inputs(store, USERS)
    artifact = load_grounded_world_model_artifact(store.artifacts, world_model)
    assert artifact.serving_rag == build.fit_rag
    indexed = load_rag_index(store.artifacts, artifact.serving_rag.artifact_id).transitions
    served = load_rag_index(store.artifacts, build.serving_rag.artifact_id).transitions
    held = {case.trace_id for case in cases}
    assert held and not held & {t.trace_id for t in indexed}
    assert {t.trace_id for t in served} - {t.trace_id for t in indexed} == held
    assert {t.lineage_id for t in indexed}.isdisjoint(case.task.lineage_group_id for case in cases)


def test_run_ids_follow_content_and_retire_attempts_that_cannot_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unchanged inputs replay; changed inputs or dead attempts get a new run automatically."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)])
    _, store, _ = build_project(tmp_path, monkeypatch, provider)
    now = datetime.now(UTC)

    def register(at: datetime):  # noqa: ANN202 - local shorthand
        """Register one source-RAG held-out run at a given time."""
        return register_held_out_fidelity(
            store, 1, profile="source-rag", code_revision=REVISION, clock=lambda: at
        )

    first = register(now)
    again = register(now + timedelta(minutes=5))
    run_id = first.prepared[0].spec.run_id
    assert again.prepared == first.prepared and not again.retired
    assert re.fullmatch(r"held-out-[a-z0-9-]+-[0-9a-f]{12}", run_id)

    # A refused attempt that never dispatched expires; the rerun registers generation two.
    later = now + DEADLINE + timedelta(seconds=1)
    expired = register(later)
    assert expired.retired == (run_id,)
    assert expired.prepared[0].spec.run_id == run_id + "-r2"

    # An execution that stopped with no live owner is retired instead of blocking forever.
    def interrupted(*args: object, **kwargs: object) -> None:
        """Stop after the execution intent is written."""
        raise KeyboardInterrupt

    monkeypatch.setattr(fidelity, "evaluate_workflow", interrupted)
    with pytest.raises(KeyboardInterrupt):
        execute_fidelity_evaluation(store, expired.prepared[0])
    resumed = register(later)
    assert resumed.retired == (run_id, run_id + "-r2")
    assert resumed.prepared[0].spec.run_id == run_id + "-r3"

    # Changed inputs (here the judge's output allowance) address a new run, never a refusal.
    catalog = load_model_catalog(store.model_catalog_path)
    judge = catalog.models["judge"]
    assert judge.capabilities is not None
    smaller = judge.capabilities.model_copy(update={"maximum_output_tokens": 8_192})
    write_model_catalog(
        store.model_catalog_path,
        catalog.model_copy(
            update={
                "models": {
                    **catalog.models,
                    "judge": judge.model_copy(update={"capabilities": smaller}),
                }
            }
        ),
    )
    changed = register(later)
    assert not changed.retired
    assert changed.prepared[0].spec.run_id.split("-")[-1] != run_id.split("-")[-1]


class Outage(SimulatedProvider):
    """The routed fake, with simulator predictions rejected while ``down`` (a revoked key)."""

    down = False

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Answer 401 to predictions while down; route everything else normally."""
        choice = payload.get("tool_choice")
        function = choice.get("function") if isinstance(choice, dict) else None
        named = function.get("name") if isinstance(function, dict) else None
        if self.down and named == "submit_transition":
            self.requests.append(RecordedRequest(url, dict(headers), payload))
            return JsonHttpResponse(status_code=401, body={"error": {"message": "bad key"}})
        return super().post(url, headers=headers, payload=payload, timeout_seconds=timeout_seconds)


@pytest.mark.parametrize("color", [False, True])
def test_a_provider_outage_stops_visibly_and_never_becomes_a_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, color: bool
) -> None:
    """A rejected key stops the command non-zero; no score is saved; a retry measures for real."""
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr("typer.rich_utils.FORCE_TERMINAL", color)
    provider = Outage([episode(i) for i in range(USERS)], learnable=True)
    root, store, _ = build_project(tmp_path, monkeypatch, provider)
    provider.down = True
    command = ["eval", PROJECT, "--root", str(root), "--audit", "3", "--yes", "--non-interactive"]
    failed = CliRunner().invoke(app, command, color=color)
    output = " ".join(unstyle(failed.output).split())
    assert failed.exit_code == 1, failed.output
    assert "a provider request failed" in output and "--retry-incomplete" in output
    assert "Held-out fidelity" not in output and "learned" not in output
    assert learning_curve(store, "fit") is None, "a failed prediction is never a measurement"

    provider.down = False
    stopped = CliRunner().invoke(app, command, color=color)
    assert stopped.exit_code != 0, "an interrupted trace is never bought again silently"
    assert "--retry-incomplete" in " ".join(unstyle(stopped.output).split())
    output = held_out_command(root, 3, "--retry-incomplete")
    assert "9/9 replies exact (100%)" in output
