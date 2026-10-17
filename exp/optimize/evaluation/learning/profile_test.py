"""Actor evaluation on the build's learned simulator: memory delivery, hiding and identity.

Everything runs offline through the routed fake of ``held_out_fidelity_test``, extended to
answer two worker models, the text world model and the actor judge.
"""

import functools
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.common.core.artifacts import JsonObject, JsonValue, canonical_json_bytes
from exp.common.models import (
    BillingSource,
    ModelCapabilities,
    ModelRecord,
    load_model_catalog,
    write_model_catalog,
)
from exp.common.project import ProjectStore
from exp.common.tasks.store import load_task_set
from exp.optimize.evaluation.audit import panel_test as held_out_fidelity_test
from exp.optimize.evaluation.audit.panel_test import (
    PROJECT,
    REVISION,
    USERS,
    SimulatedProvider,
    _arr,
    _at,
    _call_key,
    _obj,
    build_project,
    episode,
)
from exp.optimize.evaluation.contracts import EvaluationBudget
from exp.optimize.evaluation.learning import profile as learned_memory
from exp.optimize.evaluation.learning.profile import NO_MEMORY_NOTE
from exp.optimize.evaluation.learning.traces import learning_plan
from exp.optimize.evaluation.prepare import ModelEvaluationOptions
from exp.optimize.evaluation.runs import (
    EvaluationDefaults,
    _run_records,
    execute_run,
    load_run,
    prepare_run,
    run_directory,
    save_defaults,
)
from exp.optimize.evaluation.runtime import run_prepared_model_evaluation
from exp.optimize.router.automatic.provisional import prepare_hosted_provisional_judge
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.models.providers.transport import JsonHttpResponse
from exp.simulation.world_model.experience_memory import memory_head

WORKERS = ("worker", "helper")
TOOLS = ("find_user_id_by_email", "get_order_details", "get_user_details", "cancel_pending_order")
PRICES = {
    "supports_completions": True,
    "input_cost_per_million_tokens_usd": 1.0,
    "output_cost_per_million_tokens_usd": 2.0,
    "cached_input_cost_per_million_tokens_usd": 0.5,
    "cache_write_cost_per_million_tokens_usd": 1.5,
}
OPTIONS = ModelEvaluationOptions(maximum_steps=4, maximum_concurrency=1)


class ActorProvider(SimulatedProvider):
    """Also answer the worker models, the text world model and the actor judge.

    A worker reads its customer's details once, then answers. The text simulator replies with
    the recorded reply of every call and keeps every request body it saw.
    """

    def __init__(self, episodes: list[JsonObject]) -> None:
        """Index recorded replies and start an empty simulator request log."""
        super().__init__(episodes)
        self.world_requests: list[JsonObject] = []

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Route actor-evaluation chat requests; defer every other wire to the parent."""
        if url.endswith("/chat/completions"):
            model = str(payload.get("model"))
            choice = payload.get("tool_choice")
            structured = isinstance(choice, dict) and "function" in choice
            if model in {f"{alias}-v1" for alias in WORKERS}:
                return self._worker(payload)
            if model == "world-v1" and not structured:
                return self._world(payload)
            if model == "judge-v1" and not structured:
                return self._judge(payload)
        return super().post(url, headers=headers, payload=payload, timeout_seconds=timeout_seconds)

    def _worker(self, payload: JsonObject) -> JsonHttpResponse:
        """Read the customer's details on the first turn, then finish."""
        messages = [_obj(row) for row in _arr(payload["messages"])]
        if any(row.get("role") == "tool" for row in messages):
            return _chat(str(payload["model"]), {"role": "assistant", "content": "Done."}, "stop")
        user = next(str(row["content"]) for row in messages if row["role"] == "user")
        arguments: JsonValue = {"user_id": user.split()[2].rstrip(".")}
        call = {
            "id": "w-1",
            "type": "function",
            "function": {"name": "get_user_details", "arguments": json.dumps(arguments)},
        }
        message: JsonObject = {"role": "assistant", "content": None, "tool_calls": [call]}
        return _chat(str(payload["model"]), message, "tool_calls")

    def _world(self, payload: JsonObject) -> JsonHttpResponse:
        """Reply to each call with its recorded reply, or end the conversation."""
        content = _obj(json.loads(str(_at(payload, "messages", 1, "content"))))
        self.world_requests.append(content)
        calls = [_obj(row) for row in _arr(_at(content, "candidate_response", "tool_calls"))]
        results = [
            {
                "call_id": call["call_id"],
                "content": json.dumps(self.truth.get(_call_key(call["name"], call["arguments"]))),
            }
            for call in calls
        ]
        transition = (
            {"tool_results": results} if calls else {"message": "Thanks.", "terminal": True}
        )
        message: JsonObject = {"role": "assistant", "content": json.dumps(transition)}
        return _chat("world-v1", message, "stop")

    def _judge(self, payload: JsonObject) -> JsonHttpResponse:
        """Score every rubric dimension at its maximum."""
        content = str(_at(payload, "messages", 1, "content"))
        rubric = json.loads(content.split("RUBRIC:\n", 1)[1].split("\n\nROLLOUT:", 1)[0])
        dimensions = [
            {"dimension_id": _obj(row)["dimension_id"], "raw_score": _obj(row)["max_score"]}
            for row in _arr(rubric)
        ]
        message: JsonObject = {
            "role": "assistant",
            "content": json.dumps({"dimensions": dimensions}),
        }
        return _chat("judge-v1", message, "stop")

    def recall_decisions(self) -> int:
        """Native decision requests sent so far."""
        return sum(request.url.endswith("/systemone") for request in self.requests)


def _chat(model: str, message: JsonObject, finish: str) -> JsonHttpResponse:
    """One Chat Completions response with usage."""
    return JsonHttpResponse(
        status_code=200,
        body={
            "model": model,
            "choices": [{"finish_reason": finish, "message": message}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
        },
    )


def _tools(user: str) -> list[JsonObject]:
    """Tool definitions whose descriptions name the customer, so requests stay distinct.

    Identical definitions on every trace would make every request a semantic duplicate of the
    others and merge all tasks into one leakage lineage.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": " ".join(f"{user}-{name}-{k}" for k in range(8)),
                "parameters": {"type": "object"},
            },
        }
        for name in TOOLS
    ]


def built_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, learn: bool
) -> tuple[Path, ProjectStore, ActorProvider]:
    """Build the fixture package with declared tools, a judge and two priced workers.

    With ``learn`` the build learns every trace into the simulator's memory.
    """
    original = held_out_fidelity_test.package_rows

    def rows(episodes: list[JsonObject]) -> list[JsonObject]:
        """The fixture package with tool definitions on each request span."""
        out = original(episodes)
        for row in out:
            if row["name"] == "chat":
                attributes = _obj(row["attributes"])
                attributes["gen_ai.tool.definitions"] = json.dumps(
                    _tools(str(attributes["exp.conversation.id"]))
                )
        return out

    monkeypatch.setattr(held_out_fidelity_test, "package_rows", rows)
    provider = ActorProvider([episode(i) for i in range(USERS)])
    root, store, _ = build_project(tmp_path, monkeypatch, provider, learn=learn)
    monkeypatch.setattr(learned_memory, "HttpxJsonTransport", lambda: provider)
    monkeypatch.setattr(
        "exp.cli.evaluation.flow.RuntimeModelCatalog",
        functools.partial(RuntimeModelCatalog, transport_factory=lambda: provider),
    )
    path = root / "models.toml"
    catalog = load_model_catalog(path)
    models = dict(catalog.models)
    for alias in ("world", "judge"):
        capabilities = models[alias].capabilities
        assert capabilities is not None
        models[alias] = models[alias].model_copy(
            update={"capabilities": capabilities.model_copy(update=PRICES)}
        )
    for alias in WORKERS:
        models[alias] = ModelRecord(
            connection="http",
            model=f"{alias}-v1",
            billing_source=BillingSource.CUSTOMER_MANAGED,
            capabilities=ModelCapabilities(
                supports_tools=True,
                context_window_tokens=200_000,
                maximum_output_tokens=4_096,
            ).model_copy(update=PRICES),
        )
    catalog = catalog.model_copy(update={"models": models})
    write_model_catalog(path, catalog)
    prepare_hosted_provisional_judge(
        store,
        catalog,
        maximum_input_tokens=None,
        maximum_output_tokens=8_192,
        maximum_attempts=3,
        created_at=datetime.now(UTC),
        code_revision=REVISION,
    )
    save_defaults(store, EvaluationDefaults(models=WORKERS, minimum_scenarios=1, options=OPTIONS))
    return root, store, provider


def eval_command(root: Path) -> str:
    """Run ``exp eval PROJECT --models worker,helper`` and return its output."""
    result = CliRunner().invoke(
        app,
        ["eval", PROJECT, "--root", str(root), "--models", ",".join(WORKERS)]
        + ["--yes", "--non-interactive"],
    )
    assert result.exit_code == 0, result.output
    return " ".join(result.output.split())


def lineage_by_task(store: ProjectStore) -> dict[str, str]:
    """Task lineage of every built task."""
    build = store.load_project().build
    assert build is not None
    tasks = load_task_set(store.artifacts, build.task_set.artifact_id).tasks
    return {task.task_id: task.lineage_group_id for task in tasks}


def delivered(request: JsonObject) -> list[JsonObject]:
    """Every memory row one simulator request carried: observations, notes and overview."""
    research = _obj(request.get("research_history") or {})
    return [
        _obj(row)
        for row in (
            *_arr(request.get("memory_observations") or []),
            *_arr(research.get("prior_trajectories") or []),
            *_arr(research.get("memory_overview") or []),
        )
    ]


def test_actor_evaluation_reads_learned_memory_without_its_own_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``exp eval --models`` delivers the build's memory, frozen, minus each task's own traces."""
    root, store, provider = built_project(tmp_path, monkeypatch, learn=True)
    lineage = learning_plan(store, None, "all").lineage
    head = memory_head(store.artifacts, lineage)
    assert head is not None

    output = eval_command(root)

    review, saved = output.split("Estimated $", 1)
    line = f"Simulator memory: v{head.version} ({lineage}"
    assert line in review and line in saved, "the header and the saved report name the version"
    assert provider.world_requests and provider.recall_decisions() > 0
    lineages = lineage_by_task(store)
    observed = 0
    for request in provider.world_requests:
        assert "research_history" in request, "every simulated turn is served with memory"
        rows = delivered(request)
        task = _obj(request["task"])
        own = lineages[str(task["task_id"])]
        customer = str(_obj(task["initial_context"])["task_lineage"])
        assert all(own not in json.dumps(row) for row in rows)
        assert customer not in json.dumps(rows), "the task's own recorded replies are hidden"
        observed += len(_arr(request.get("memory_observations") or []))
    assert observed > 0, "recalled observations from other traces reach predictions"
    head_after = memory_head(store.artifacts, lineage)
    assert head_after is not None and head_after.version == head.version, "never published"
    run = next(path for path in (store.paths.runtime_directory / "evaluations").iterdir())
    report = json.loads((run / "report.json").read_text())
    assert report["memory"]["version"] == head.version
    assert report["simulator_memory"].startswith(f"Simulator memory: v{head.version}")


def test_memory_version_is_part_of_the_plan_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Identical memory replays saved results; another version is a different plan."""
    root, store, provider = built_project(tmp_path, monkeypatch, learn=True)
    catalog = load_model_catalog(store.model_catalog_path)
    run = prepare_run(
        store,
        catalog,
        EvaluationDefaults(models=WORKERS, minimum_scenarios=1, options=OPTIONS),
        code_revision=REVISION,
    )
    assert run.prepared.memory is not None and run.prepared.memory.version > 1
    runtime = RuntimeModelCatalog(catalog, transport_factory=lambda: provider)

    def evaluate(memory_version: int) -> str:
        """Run the prepared matrix on one memory version; return its plan identity."""
        assert run.prepared.memory is not None
        memory = run.prepared.memory.model_copy(update={"version": memory_version})
        return run_prepared_model_evaluation(
            store,
            run.prepared.model_copy(update={"memory": memory}),
            runtime,
            budget=EvaluationBudget(maximum_judgments=run.prepared.cost.judgment_count),
            provider_spend_consented=True,
            created_at=run.created_at,
            code_revision=REVISION,
        ).plan.plan_id

    latest = run.prepared.memory.version
    first = evaluate(latest)
    purchased = len(provider.requests)
    assert evaluate(latest) == first
    assert len(provider.requests) == purchased, "identical memory replays without calls"
    assert evaluate(latest - 1) != first
    assert len(provider.requests) > purchased, "another memory version is evaluated afresh"


def test_a_project_without_learned_memory_evaluates_as_before_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A build that learned nothing evaluates on the bare simulator and prints one note."""
    root, store, provider = built_project(tmp_path, monkeypatch, learn=False)

    output = eval_command(root)

    review, saved = output.split("Estimated $", 1)
    assert NO_MEMORY_NOTE in review, "the review header says the simulator has no memory"
    assert NO_MEMORY_NOTE in saved, "so does the saved report"
    assert provider.world_requests
    assert all("research_history" not in request for request in provider.world_requests)
    assert provider.recall_decisions() == 0
    run_id = next(iter(_run_records(store).list_ids()))
    assert load_run(store, run_id).prepared.memory is None
    report = json.loads((run_directory(store, run_id) / "report.json").read_text())
    assert report["memory"] is None and report["simulator_memory"] == NO_MEMORY_NOTE


def test_an_evaluation_saved_without_memory_resumes_without_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run saved before evaluations read memory keeps its identity and stays memory-free."""
    _, store, provider = built_project(tmp_path, monkeypatch, learn=True)
    catalog = load_model_catalog(store.model_catalog_path)
    run = prepare_run(
        store,
        catalog,
        EvaluationDefaults(models=WORKERS, minimum_scenarios=1, options=OPTIONS),
        code_revision=REVISION,
    )
    record = json.loads(canonical_json_bytes(run))
    assert "memory" in record["prepared"]
    del record["prepared"]["memory"]
    old = canonical_json_bytes(record)
    _run_records(store).write(run.run_id, old)

    saved = load_run(store, run.run_id)
    assert saved.prepared.memory is None
    assert canonical_json_bytes(saved) == old, "an old plan keeps its exact identity"
    decisions = provider.recall_decisions()
    execute_run(
        store,
        saved,
        RuntimeModelCatalog(catalog, transport_factory=lambda: provider),
        provider_spend_consented=True,
    )
    assert provider.world_requests
    assert all("research_history" not in request for request in provider.world_requests)
    assert provider.recall_decisions() == decisions, "no memory is recalled"
