"""Offline tests for System One: parallel Jev judgments over trace regions, adaptive reading."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.project import ProjectStore
from exp.common.traces import Trace
from exp.optimize.evaluation.rules.research_test import agent_setup
from exp.optimize.evaluation.rules.system_one import (
    FINDINGS_TYPE,
    DecisionBoundary,
    Objective,
    ReadingPolicy,
    RegionCorpus,
    SystemOne,
)
from exp.runtime.gateway.decisions_contracts import DecisionRequest
from exp.simulation.mining.tagging import DecisionCache
from exp.simulation.rules.contracts_test import refund_trace

Answer = Mapping[str, object]
"""One region's scripted answer: shows, relevance, reading, lead, lead_kind."""

QUIET: Answer = {
    "shows": 0.1,
    "relevance": 0.0,
    "reading": "more",
    "lead": 0.0,
    "lead_kind": "none",
}


class ScriptedJev:
    """Thread-safe scripted System One decisions keyed by region locator.

    Attributes:
        answers: Answer per region locator; unlisted regions get ``default``.
        calls: Region locators asked, in arrival order.
        most_in_flight: The most decisions answered at once.
    """

    def __init__(
        self, answers: Mapping[str, Answer], *, default: Answer = QUIET, delay: float = 0.0
    ) -> None:
        """Bind the answers."""
        self.answers, self.default, self.delay = dict(answers), default, delay
        self.calls: list[str] = []
        self.lock = threading.Lock()
        self.in_flight = 0
        self.most_in_flight = 0

    def __call__(self, request: DecisionRequest) -> JsonObject:
        """Answer one typed region request."""
        state = request.state
        assert isinstance(state, dict) and isinstance(state["region"], dict)
        locator = str(state["region"]["locator"])
        with self.lock:
            self.calls.append(locator)
            self.in_flight += 1
            self.most_in_flight = max(self.most_in_flight, self.in_flight)
        try:
            if self.delay:
                time.sleep(self.delay)
            row = self.answers.get(locator, self.default)
            return {
                "answers": {
                    "shows": {"type": "noul", "noul": row["shows"]},
                    "relevance": {"type": "score", "score": row["relevance"]},
                    "reading": {"type": "choice", "choice": row["reading"]},
                    "lead": {"type": "noul", "noul": row["lead"]},
                    "lead_kind": {"type": "choice", "choice": row["lead_kind"]},
                },
                "usage": {"cost": 0.001, "input_tokens": 10},
            }
        finally:
            with self.lock:
                self.in_flight -= 1


def hit(reading: str = "stop", **extra: object) -> Answer:
    """A region that shows the objective."""
    return {**QUIET, "shows": 0.9, "relevance": 2.0, "reading": reading, **extra}


def refund_traces() -> tuple[Trace, ...]:
    """Two world-learning traces with a refused and a successful refund each."""
    return (refund_trace("t1"), refund_trace("t2"))


def system_one(
    project: ProjectStore,
    jev: ScriptedJev,
    *,
    policy: ReadingPolicy | None = None,
    cache: bool = False,
    run: str = "run",
) -> SystemOne:
    """System One over the refund corpus with a scripted decision boundary."""
    boundary = DecisionBoundary(
        decide=jev,
        model_identity={"model": "jev-fixture"},
        cache=DecisionCache(project.paths) if cache else None,
    )
    return SystemOne(
        project,
        boundary,
        RegionCorpus(refund_traces()),
        policy or ReadingPolicy(first_batch=2, more_batch=2, max_rounds=3, enough_passages=2),
        run_identity=run,
        code_revision="test",
    )


REFUND = Objective(topic="refund", tool="refund", question="What precondition gates refund?")


def test_objective_returns_original_passages_with_coordinates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Passages are the exact recorded steps with locator, trace and step; saved as findings."""
    project, _ = agent_setup(tmp_path, monkeypatch)
    jev = ScriptedJev({"trace:t1:span:c2": hit(), "trace:t2:span:c2": hit()})
    reader = system_one(project, jev, policy=ReadingPolicy(first_batch=3, enough_passages=2))
    findings, reference = reader.answer(REFUND)

    corpus = reader.corpus
    assert [item.passage.locator for item in findings.passages] == [
        "trace:t1:span:c2",
        "trace:t2:span:c2",
    ]
    first = findings.passages[0].passage
    region = corpus.by_locator["trace:t1:span:c2"]
    assert (first.trace_id, first.step, first.quote) == ("t1", region.step, region.text)
    assert '"amount": 400' in first.quote and first.source_sha256 == region.sha256
    assert corpus.by_locator["trace:t1:span:r2"] is region
    assert findings.stop == "enough" and findings.bought == len(findings.judgments) == 3
    assert project.artifacts.read(reference.artifact_id).manifest.artifact_type == FINDINGS_TYPE
    passages = findings.brief()["passages"]
    assert isinstance(passages, list) and isinstance(passages[0], dict)
    assert passages[0]["quote"] == region.text


def test_reading_stops_when_judgments_vote_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A round whose relevance-weighted vote is stop ends reading before the round limit."""
    project, _ = agent_setup(tmp_path, monkeypatch)
    jev = ScriptedJev({"trace:t1:span:c2": hit("stop")})
    findings, _ = system_one(
        project, jev, policy=ReadingPolicy(first_batch=2, enough_passages=4)
    ).answer(REFUND)

    assert findings.stop == "judged_stop" and findings.rounds == 1 and findings.votes == ("stop",)
    assert len(jev.calls) == 2


def test_context_vote_reads_neighbors_then_more_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A context vote judges the steps around the best region; a later more vote moves on."""
    project, _ = agent_setup(tmp_path, monkeypatch)
    jev = ScriptedJev(
        {
            "trace:t1:span:c2": {**hit("context"), "shows": 0.5},
            "trace:t1:span:c1": {**QUIET, "relevance": 1.0, "reading": "more"},
        }
    )
    policy = ReadingPolicy(first_batch=1, more_batch=1, max_rounds=3, enough_passages=4)
    findings, _ = system_one(project, jev, policy=policy).answer(REFUND)

    assert findings.votes[0] == "context"
    second_round = [j.locator for j in findings.judgments if j.round == 1]
    assert set(second_round) == {"trace:t1:span:c1", "trace:t1:span:c3"}
    assert findings.rounds == 3 and findings.stop == "max_rounds"
    assert [j.locator for j in findings.judgments if j.round == 2] == ["trace:t2:span:c2"]


def test_discovery_adds_a_lead_objective(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A region revealing a different behavior returns a new objective tied to its parent."""
    project, _ = agent_setup(tmp_path, monkeypatch)
    jev = ScriptedJev(
        {"trace:t1:span:c3": hit("stop", lead=0.9, lead_kind="effect")},
    )
    findings, _ = system_one(project, jev).answer(REFUND)

    (lead,) = findings.leads
    assert lead.origin == "discovery" and lead.parent == REFUND.objective_id
    assert lead.tool == "refund" and "effect" in lead.question
    assert "trace:t1:span:c3" in lead.question


def test_no_candidates_and_bounded_concurrency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regions are judged in parallel within the bound; an empty corpus stops at once."""
    project, _ = agent_setup(tmp_path, monkeypatch)
    jev = ScriptedJev({}, delay=0.05)
    policy = ReadingPolicy(first_batch=6, max_rounds=1, concurrency=3)
    findings, _ = system_one(project, jev, policy=policy).answer(REFUND)
    assert len(findings.judgments) == 6 and 1 < jev.most_in_flight <= 3

    empty = SystemOne(
        project,
        DecisionBoundary(decide=jev, model_identity={"model": "jev"}),
        RegionCorpus(()),
        ReadingPolicy(),
        run_identity="empty",
        code_revision="test",
    )
    assert empty.answer(REFUND)[0].stop == "no_candidates"


def test_findings_and_decisions_replay_without_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A saved objective returns as is, and a new run replays every decision from the cache."""
    project, _ = agent_setup(tmp_path, monkeypatch)
    jev = ScriptedJev({"trace:t1:span:c2": hit(), "trace:t2:span:c2": hit()})
    first, _ = system_one(project, jev, cache=True).answer(REFUND)
    calls = len(jev.calls)

    again, _ = system_one(project, jev, cache=True).answer(REFUND)
    assert again == first and len(jev.calls) == calls

    rerun, _ = system_one(project, jev, cache=True, run="other").answer(REFUND)
    assert len(jev.calls) == calls and rerun.bought == 0 and rerun.replayed == calls
    assert rerun.passages == first.passages


def test_malformed_answers_fail_visibly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A choice outside the declared options is refused, never coerced."""
    project, _ = agent_setup(tmp_path, monkeypatch)
    jev = ScriptedJev({}, default={**QUIET, "reading": "maybe"})
    with pytest.raises(ValueError, match="declared option"):
        system_one(project, jev).answer(REFUND)


def test_an_over_long_step_becomes_a_clipped_passage_with_the_original_digest() -> None:
    """A step longer than a passage allows keeps its head, tail and source digest."""
    from exp.common.core.artifacts import sha256_json as sha256_text
    from exp.optimize.evaluation.rules.system_one import QUOTE_LIMIT, Region

    text = "HEAD " + "x" * 40_000 + " TAIL"
    region = Region(
        locator="trace:t:span:s", trace_id="t", step=3, text=text, sha256=sha256_text(text)
    )
    passage = region.passage()
    assert len(passage.quote) <= QUOTE_LIMIT
    assert passage.quote.startswith("HEAD ") and passage.quote.endswith(" TAIL")
    assert "clipped" in passage.quote and passage.source_sha256 == sha256_text(text)
    short = Region(
        locator="trace:t:span:u", trace_id="t", step=1, text="ok", sha256=sha256_text("ok")
    )
    assert short.passage().quote == "ok"
