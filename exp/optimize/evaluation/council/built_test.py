"""A fresh normal build composes calibration and actor analysis without exposing references."""

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.evaluations.actor_rubric import ACTOR_JUDGE_PROMPT, actor_judge
from exp.common.models import load_model_catalog, write_model_catalog
from exp.optimize.evaluation.audit.panel import source_rag_inputs
from exp.optimize.evaluation.audit.panel_test import (
    USERS,
    SimulatedProvider,
    _judgement,
    _tool_reply,
    build_project,
    episode,
)
from exp.optimize.evaluation.council import built as built_council
from exp.optimize.evaluation.council import runner as trace_council
from exp.optimize.evaluation.council.built import DEFAULT_ACTOR_MODELS, prepare_built_council
from exp.optimize.evaluation.fidelity import runner as fidelity
from exp.optimize.evaluation.fidelity.runner import execute_fidelity_evaluation
from exp.runtime.models.providers.transport import JsonHttpResponse, RecordedRequest


def test_fresh_build_prepares_disjoint_calibration_and_actor_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The normal build, fit-only evidence, five aliases and saved deadlines are connected."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)])
    _, project, _ = build_project(tmp_path, monkeypatch, provider)
    catalog = load_model_catalog(project.model_catalog_path)
    actors = tuple(f"actor-{i}" for i in range(5))
    record = catalog.models["world"]
    catalog = catalog.model_copy(
        update={"models": {**catalog.models, **{alias: record for alias in actors}}}
    )
    write_model_catalog(project.model_catalog_path, catalog)
    before = len(provider.requests)
    prepared = prepare_built_council(project, actors, code_revision="fixture")
    assert len(provider.requests) == before
    assert len(prepared.council.actors) == 5
    assert prepared.calibration.spec.judge_output_protocol == "json"
    assert prepared.council.judge_output_protocol == "json"
    actor_rubric = actor_judge(prepared.calibration.spec.judge.model_identity)
    assert prepared.council.judge.definition == actor_rubric.definition
    assert prepared.council.judge.prompt == ACTOR_JUDGE_PROMPT
    assert prepared.calibration.spec.judge.definition != actor_rubric.definition
    assert prepared.calibration.spec.maximum_physical_attempts >= 3 * (
        5 * len(prepared.calibration.spec.workload.actions) + 1
    )
    assert prepared.council.maximum_physical_attempts >= 5 * (3 + 2 * 3 + 1) * 3 + 3
    assert (
        prepared.calibration.spec.task.lineage_group_id
        != prepared.council.tasks[0].lineage_group_id
    )
    assert prepared.council.world_model == prepared.calibration.spec.world_model
    assert prepared.council.memory is None
    assert prepared.council.analyst_fidelity_plans[0].artifact_id == prepared.calibration.id
    assert prepared.council.tasks[0].reference_answer is None
    native = json.loads(
        project.artifacts.read_bytes(
            prepared.council.analyst_native_episodes[0].artifact_id, "episode.json"
        )
    )
    assert native["initial_context"] == prepared.council.tasks[0].initial_context
    assert native["steps"]
    replay = prepare_built_council(
        project, actors, code_revision="fixture", clock=lambda: datetime(2040, 1, 1, tzinfo=UTC)
    )
    assert replay == prepared
    assert len(provider.requests) == before


def test_execution_uses_existing_calibration_then_council_owners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The product composition dispatches no custom loop and retains owner ordering."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)])
    _, project, _ = build_project(tmp_path, monkeypatch, provider)
    prepared = prepare_built_council(project, ("world", "judge"), code_revision="fixture")
    stages: list[str] = []
    monkeypatch.setattr(
        built_council,
        "execute_fidelity_evaluation",
        lambda *args, **kwargs: stages.append("calibration"),
    )
    monkeypatch.setattr(
        built_council,
        "prepare_trace_council",
        lambda *args, **kwargs: stages.append("prepare-council") or "registered",
    )
    monkeypatch.setattr(
        built_council,
        "execute_trace_council",
        lambda *args, **kwargs: stages.append("execute-council") or {"result": "saved"},
    )
    result = built_council.execute_built_council(project, prepared)
    assert stages == ["calibration", "prepare-council", "execute-council"]
    assert result == {"result": "saved"}


def test_one_heldout_lineage_is_refused_without_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Several cases from one customer cannot calibrate their own actor findings."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)])
    _, project, _ = build_project(tmp_path, monkeypatch, provider)
    world, cases = source_rag_inputs(project, USERS)
    same_lineage = cases[1].model_copy(
        update={
            "task": cases[1].task.model_copy(
                update={"lineage_group_id": cases[0].task.lineage_group_id}
            )
        }
    )
    monkeypatch.setattr(
        built_council, "source_rag_inputs", lambda *args: (world, (cases[0], same_lineage))
    )
    before = len(provider.requests)
    with pytest.raises(ValueError, match="distinct held-out task lineages"):
        prepare_built_council(project, ("world", "judge"), code_revision="fixture")
    assert len(provider.requests) == before


class CouncilProvider(SimulatedProvider):
    """Answer actual actor, simulator, semantic embedding, judge and analyst wire calls."""

    action: JsonObject

    def post(
        self, url: str, *, headers: Mapping[str, str], payload: JsonObject, timeout_seconds: float
    ) -> JsonHttpResponse:
        """Keep embedding execution real while supplying bounded deterministic completions."""
        model = str(payload.get("model", ""))
        messages = payload.get("messages", [])
        if model == "judge-v1" and not payload.get("tools"):
            self.requests.append(RecordedRequest(url, dict(headers), payload))
            assert isinstance(messages, list)
            user = next(
                row for row in messages if isinstance(row, dict) and row.get("role") == "user"
            )
            body = json.loads(str(user["content"]))
            judgement = _judgement(body["output_schema"]["properties"])
            return JsonHttpResponse(
                status_code=200,
                body={
                    "model": model,
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": json.dumps(judgement)},
                        }
                    ],
                },
            )
        if model.startswith("actor-"):
            self.requests.append(RecordedRequest(url, dict(headers), payload))
            assert isinstance(messages, list)
            done = any(isinstance(row, dict) and row.get("role") == "tool" for row in messages)
            message: JsonObject = {"role": "assistant", "content": "Done" if done else None}
            if not done:
                message["tool_calls"] = [
                    {
                        "id": "actor-read",
                        "type": "function",
                        "function": {
                            "name": self.action["name"],
                            "arguments": json.dumps(self.action["arguments"]),
                        },
                    }
                ]
            return JsonHttpResponse(
                status_code=200,
                body={
                    "model": model,
                    "choices": [
                        {"finish_reason": "stop" if done else "tool_calls", "message": message}
                    ],
                },
            )
        if "You analyze an evaluation conducted entirely" in json.dumps(messages):
            self.requests.append(RecordedRequest(url, dict(headers), payload))
            return _tool_reply(
                "judge-v1",
                "submit_proposal",
                {
                    "summary": "Bounded simulator integration.",
                    "findings": [],
                    "limitations": ["Scripted responses; not behavioral evidence."],
                },
            )
        return super().post(url, headers=headers, payload=payload, timeout_seconds=timeout_seconds)


@pytest.mark.parametrize("use_defaults", [False, True])
def test_semantic_build_council_binds_exact_embedder_and_meters_query_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_defaults: bool
) -> None:
    """A real semantic build must query its own embedder through the council ledger."""
    provider = CouncilProvider([episode(i) for i in range(USERS)])
    _, project, _ = build_project(tmp_path, monkeypatch, provider)
    catalog = load_model_catalog(project.model_catalog_path)
    actors = DEFAULT_ACTOR_MODELS if use_defaults else ("actor-0", "actor-1")
    record = catalog.models["world"]
    write_model_catalog(
        project.model_catalog_path,
        catalog.model_copy(
            update={
                "models": {
                    **catalog.models,
                    **{alias: record.model_copy(update={"model": alias}) for alias in actors},
                }
            }
        ),
    )
    prepared = prepare_built_council(
        project, None if use_defaults else actors, code_revision="semantic-council"
    )
    _, cases = source_rag_inputs(project, USERS)
    case = next(row for row in cases if row.task.task_id == prepared.council.tasks[0].task_id)
    provider.action = {"name": case.steps[0].name, "arguments": case.steps[0].arguments}
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: provider)
    monkeypatch.setattr(trace_council, "HttpxJsonTransport", lambda: provider)
    execute_fidelity_evaluation(project, prepared.calibration)
    before = len(provider.requests)
    result = built_council.execute_built_council(project, prepared)
    sent = provider.requests[before:]
    assert any(row.url.endswith("/embeddings") for row in sent)
    assert all(
        row.payload["model"] == "embed-v1" for row in sent if row.url.endswith("/embeddings")
    )
    assert result["physical_attempts"] == len(sent)
    assert result["analysis_failures"] == {}

    traces = result["traces"]
    assert isinstance(traces, list) and len(traces) == len(actors)
    dispatched = {str(row.payload.get("model")) for row in sent}
    assert set(actors) <= dispatched
    for row in traces:
        assert isinstance(row, dict)
        evaluation = row["evaluation"]
        assert isinstance(evaluation, dict)
        assert evaluation["judge_status"] == "completed"
    assert all(isinstance(row, dict) and row["error_type"] is None for row in traces)


@pytest.mark.parametrize("models", [None, (), ("world", "world")])
def test_missing_or_invalid_actor_registration_never_reduces_the_council(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, models: tuple[str, ...] | None
) -> None:
    """Missing defaults and empty or duplicate overrides fail before plans or calls."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)])
    _, project, _ = build_project(tmp_path, monkeypatch, provider)
    if models is None:
        catalog = load_model_catalog(project.model_catalog_path)
        record = catalog.models["world"]
        for alias in DEFAULT_ACTOR_MODELS[:-1]:
            catalog.models[alias] = record
        write_model_catalog(project.model_catalog_path, catalog)
    before = len(provider.requests)
    message = "actor-kimi.*capabilities" if models is None else "two distinct"
    with pytest.raises(ValueError, match=message):
        prepare_built_council(project, models, code_revision="fixture")
    assert not project.artifacts.list_ids(artifact_type="built-trace-council-plan")
    assert len(provider.requests) == before


def test_invalid_analyst_is_refused_before_fidelity_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-tool-capable analyst must not consume the preceding fidelity budget."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)])
    _, project, _ = build_project(tmp_path, monkeypatch, provider)
    catalog = load_model_catalog(project.model_catalog_path)
    record = catalog.models["judge"]
    assert record.capabilities is not None
    catalog.models["actor"] = catalog.models["world"]
    catalog.models["judge"] = record.model_copy(
        update={"capabilities": record.capabilities.model_copy(update={"supports_tools": False})}
    )
    write_model_catalog(project.model_catalog_path, catalog)
    before = len(provider.requests)
    with pytest.raises(ValueError, match="judge.*tool capabilities"):
        prepare_built_council(project, ("world", "actor"), code_revision="fixture")
    assert len(provider.requests) == before
    assert not project.artifacts.list_ids(artifact_type="built-trace-council-plan")
