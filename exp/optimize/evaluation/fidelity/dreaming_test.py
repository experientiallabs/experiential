"""The real build CLI publishes cited consolidation inside its pinned memory versions."""

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock

import pytest

from exp.common.core.artifacts import JsonObject
from exp.optimize.evaluation.audit.panel_test import (
    REVISION,
    SimulatedProvider,
    _at,
    _obj,
    _tool_reply,
    build_project,
    episode,
)
from exp.optimize.evaluation.fidelity.replay import content_run_id
from exp.optimize.evaluation.learning.traces import (
    learn_from_traces,
    learning_cases,
    learning_curve,
    learning_plan,
    learning_spec,
)
from exp.runtime.models.providers.transport import JsonHttpResponse, RecordedRequest
from exp.simulation.world_model.dreaming import ClaimDraft
from exp.simulation.world_model.experience_memory import (
    Eligibility,
    memory_head,
    memory_view,
    read_memory_version,
)


class DreamProvider(SimulatedProvider):
    """Script controlled draft and evidence-check wires, preserving normal build behavior."""

    cue: str | None = None
    transferable: bool = True

    def post(
        self, url: str, *, headers: Mapping[str, str], payload: JsonObject, timeout_seconds: float
    ) -> JsonHttpResponse:
        """Handle only the two new protocol messages through their actual HTTP shapes."""
        questions = payload.get("questions")
        if isinstance(questions, dict) and "relation" in questions:
            self.requests.append(RecordedRequest(url, dict(headers), payload))
            original = _obj(_at(payload, "state", "original"))
            action = original.get("action")
            name = action.get("name") if isinstance(action, dict) else None
            cited_tool = _at(payload, "state", "claim", "page")
            relation = "support" if name == cited_tool else "unknown"
            return JsonHttpResponse(
                status_code=200,
                body={"answers": {"relation": {"type": "choice", "choice": relation}}},
            )
        if not url.endswith(("/embeddings", "/systemone")):
            schema = _obj(_at(payload, "tools", 0, "function", "parameters", "properties"))
            if "claims" in schema:
                self.requests.append(RecordedRequest(url, dict(headers), payload))
                body = json.loads(str(_at(payload, "messages", 1, "content")))
                key, original = next(
                    (key, row)
                    for key, row in body["originals"].items()
                    if row.get("kind") == "original-observation"
                )
                tool = original["action"]["name"]
                claim = ClaimDraft(
                    page=tool,
                    claim=f"{tool} returns a recorded native reply.",
                    cue=self.cue or f"when calling {tool}",
                    scope="this recorded tool contract",
                    uncertainty="A hypothesis over the supplied instances only.",
                    citations=(key,),
                    transferable=self.transferable,
                )
                name = str(_at(payload, "tool_choice", "function", "name"))
                return _tool_reply("world-v1", name, {"claims": [claim.model_dump(mode="json")]})
        return super().post(url, headers=headers, payload=payload, timeout_seconds=timeout_seconds)


@pytest.mark.parametrize("long_cue", [False, True])
def test_build_consolidation_is_budgeted_pinned_hidden_and_replayed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    long_cue: bool,
) -> None:
    """Two completed traces produce two memory versions; their dream claims obey source hiding."""
    episodes = [episode(index) for index in range(2)]
    provider = DreamProvider(episodes, learnable=True)
    if long_cue:
        provider.cue = "calling this tool with the recorded inputs " * 15
    _, project, output = build_project(
        tmp_path, monkeypatch, provider, "--dream-every", "1", learn=True, episodes=episodes
    )
    plan = learning_plan(project)
    assert plan.dreaming is not None and plan.dreaming.interval == 1
    report_path = project.paths.runtime_directory / "learning" / plan.lineage / "report.txt"
    assert "Report:" in output and report_path.is_file()
    assert "dreaming" in report_path.read_text()
    assert len(plan.dream_trace_ids) == 2
    cases = learning_cases(project, "all")
    spec = learning_spec(project, plan, cases[0], deadline=datetime.now(UTC) + timedelta(hours=1))
    ordinary = learning_spec(
        project, plan.model_copy(update={"dreaming": None}), cases[0], deadline=spec.deadline
    )
    assert spec.maximum_physical_attempts > ordinary.maximum_physical_attempts
    trace = cases[0].trace_id
    assert content_run_id(spec, "learn", trace, code_revision=None) != content_run_id(
        ordinary, "learn", trace, code_revision=None
    )
    head = memory_head(project.artifacts, plan.lineage)
    assert head is not None and head.version == len(head.consumed) == 2
    first = read_memory_version(project.artifacts, plan.lineage, 1)
    if long_cue:
        assert first.appended is not None
        checked = [line.text for line in first.appended if "[checked hypothesis]" in line.text]
        assert len(checked) == 1 and "returns a recorded native reply." in checked[0]
        assert "preview" in checked[0] and "read record" in checked[0]
    initial_view = memory_view(project.artifacts, first, Eligibility(strict=True))
    initial_claims = {
        key for key, row in initial_view.records.items() if row.get("kind") == "wiki-claim"
    }
    assert len(initial_claims) == 1
    final = memory_view(project.artifacts, head, Eligibility(strict=True))
    claims = {key for key, row in final.records.items() if row.get("kind") == "wiki-claim"}
    assert len(claims) == 2
    hidden = memory_view(
        project.artifacts,
        head,
        Eligibility(strict=True, hidden_task_lineage=cases[0].task.lineage_group_id),
    )
    assert not (claims & hidden.records.keys()), "all context dependencies must hide derived claims"
    before = len(provider.requests)
    outcome = learn_from_traces(
        project,
        recall_alias="jev",
        maximum_attempts=10_000,
        code_revision=REVISION,
        clock=lambda: datetime.now(UTC),
    )
    assert outcome.incomplete == 0
    assert len(provider.requests) == before
    with pytest.raises(ValueError, match="dreaming policy differs"):
        learning_plan(project, "jev", "all", dream_every=2)


def test_interrupted_consolidation_resumes_without_redrawing_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lost evidence reader needs explicit recovery; committed predictions and drafts replay."""
    episodes = [episode(0)]

    class InterruptedProvider(DreamProvider):
        """Lose exactly one returned evidence reader after its dispatch was claimed."""

        lost = False
        lock = Lock()

        def post(
            self,
            url: str,
            *,
            headers: Mapping[str, str],
            payload: JsonObject,
            timeout_seconds: float,
        ) -> JsonHttpResponse:
            """Simulate a process stopping after receiving its first support response."""
            result = super().post(
                url, headers=headers, payload=payload, timeout_seconds=timeout_seconds
            )
            questions = payload.get("questions")
            if isinstance(questions, dict) and "relation" in questions:
                with self.lock:
                    if not self.lost:
                        self.lost = True
                        raise KeyboardInterrupt
            return result

    provider = InterruptedProvider(episodes, learnable=True)
    _, project, _ = build_project(tmp_path, monkeypatch, provider, episodes=episodes)

    def learn(*, retry: bool = False) -> None:
        """Drive the real learning orchestration with its existing recovery authority."""
        learn_from_traces(
            project,
            recall_alias="jev",
            maximum_attempts=10_000,
            code_revision=REVISION,
            clock=lambda: datetime.now(UTC),
            dream_every=1,
            retry_incomplete=retry,
        )

    with pytest.raises(KeyboardInterrupt):
        learn()
    plan = learning_plan(project)
    assert memory_head(project.artifacts, plan.lineage) is None
    before = len(provider.requests)
    with pytest.raises(ValueError, match="retry-incomplete"):
        learn()
    assert len(provider.requests) == before

    learn(retry=True)
    retried = provider.requests[before:]
    checks = [row for row in retried if "questions" in row.payload]
    assert len(checks) == 1 and "relation" in _obj(checks[0].payload["questions"])
    # First-time compression follows successful publication; it is not a repeated prediction.
    for row in retried:
        if "questions" not in row.payload:
            assert "summary" in _obj(
                _at(row.payload, "tools", 0, "function", "parameters", "properties")
            )
    head = memory_head(project.artifacts, plan.lineage)
    assert head is not None and head.version == 1
    curve = learning_curve(project)
    assert curve is not None and curve.qualification is not None
    assert curve.qualification.investigation == 1
    assert curve.qualification.prediction == 0
    before = len(provider.requests)
    learn()
    assert len(provider.requests) == before


def test_build_retains_historical_claim_without_promoting_an_overview_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance as a cited record is distinct from publication as a reusable rule line."""
    episodes = [episode(0)]
    provider = DreamProvider(episodes, learnable=True)
    provider.transferable = False
    _, project, _ = build_project(
        tmp_path, monkeypatch, provider, "--dream-every", "1", learn=True, episodes=episodes
    )
    plan = learning_plan(project)
    head = memory_head(project.artifacts, plan.lineage)
    assert head is not None and head.version == 1
    view = memory_view(project.artifacts, head, Eligibility(strict=True))
    claims = [row for row in view.records.values() if row.get("kind") == "wiki-claim"]
    assert len(claims) == 1
    assert _at(claims[0], "investigation", "transferable") is False
    assert head.appended is not None
    assert all("[checked hypothesis]" not in line.text for line in head.appended)
