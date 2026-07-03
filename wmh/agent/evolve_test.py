"""Tests for the evolutionary manager: archive selection, mutation parsing, and the evolve loop."""

from __future__ import annotations

from wmh.agent.closed_loop import ClosedLoopReport, TaskOutcome
from wmh.agent.evolve import (
    ArchiveEntry,
    HarnessArchive,
    _parse_mutation,
    evolve,
)
from wmh.agent.gold import GoldJudge
from wmh.agent.spec import HarnessSpec
from wmh.agent.tasks import TaskSpec
from wmh.engine.world_model import WorldModel
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmh.retrieval import EmbeddingRetriever, HashingEmbedder


def _report(name: str, rate: float, per_task: dict[str, float] | None = None) -> ClosedLoopReport:
    tasks = per_task or {"t": rate}
    return ClosedLoopReport(
        harness=name,
        success_rate=rate,
        per_task={
            tid: TaskOutcome(task_id=tid, success_rate=r, passes=3) for tid, r in tasks.items()
        },
    )


def test_archive_best_picks_highest_fitness() -> None:
    archive = HarnessArchive()
    archive.add(ArchiveEntry(spec=HarnessSpec(name="a"), report=_report("a", 0.2)))
    archive.add(ArchiveEntry(spec=HarnessSpec(name="b"), report=_report("b", 0.8)))
    assert archive.best().spec.name == "b"


def test_select_parent_is_deterministic_for_a_seed() -> None:
    archive = HarnessArchive()
    archive.add(ArchiveEntry(spec=HarnessSpec(name="a"), report=_report("a", 0.5)))
    archive.add(ArchiveEntry(spec=HarnessSpec(name="b"), report=_report("b", 0.5)))
    picks = {archive.select_parent(seed=7).spec.name for _ in range(5)}
    assert len(picks) == 1  # same seed -> same parent every time


def test_pareto_keeps_per_task_winners() -> None:
    archive = HarnessArchive()
    archive.add(
        ArchiveEntry(spec=HarnessSpec(name="a"), report=_report("a", 0.5, {"t1": 1.0, "t2": 0.0}))
    )
    archive.add(
        ArchiveEntry(spec=HarnessSpec(name="b"), report=_report("b", 0.5, {"t1": 0.0, "t2": 1.0}))
    )
    assert archive.pareto_names() == ["a", "b"]  # each wins one task


def test_parse_mutation_drops_invalid_tools_and_reparents() -> None:
    parent = HarnessSpec(name="parent")
    child = _parse_mutation(
        '{"name": "child", "motivation": "add grep", "system_prompt": "do it", '
        '"tools": ["bash", "submit", "not_real"], "seed_skills": [], '
        '"max_turns": 12, "temperature": 0.5}',
        parent,
    )
    assert child.name == "child"
    assert "not_real" not in child.tools  # invented tool dropped
    assert child.parent == "parent"


def test_parse_mutation_falls_back_to_parent_on_garbage() -> None:
    parent = HarnessSpec(name="parent")
    child = _parse_mutation("not json at all", parent)
    assert child.system_prompt == parent.system_prompt
    assert child.parent == "parent"


class EvolveProvider:
    """Agent+judge+world-model+meta all-in-one; the meta role proposes a valid child once."""

    def __init__(self) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        if "meta-agent that improves" in system:
            return Completion(
                text='{"name": "child", "motivation": "tighter prompt", '
                '"system_prompt": "be careful and verify", "tools": ["bash", "submit"], '
                '"seed_skills": [], "max_turns": 10, "temperature": 0.3}'
            )
        if "grade whether an agent completed a task" in system:
            return Completion(
                text='{"assertions": [{"assertion": "did it", "passed": true, "why": "x"}], '
                '"passed": true}'
            )
        if system.startswith("be careful") or system.startswith("You are a capable"):
            return Completion(text='{"tool": "submit", "arguments": {"answer": "done"}}')
        return Completion(text='{"output": "ok", "is_error": false}')

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


def test_evolve_runs_and_archives_variants() -> None:
    provider = EvolveProvider()
    wm = WorldModel(provider, EmbeddingRetriever(HashingEmbedder(dim=16)))
    tasks = [TaskSpec(task_id="q", instruction="do it", gold=["did it"])]
    result = evolve(
        HarnessSpec(), tasks, wm, provider, provider, GoldJudge(provider), generations=2, k=1
    )
    assert result.generations == 2
    assert len(result.archive.entries) == 3  # seed + 2 generations
    assert result.best_score == 1.0
