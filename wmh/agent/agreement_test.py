"""Tests for the pure sim-real agreement metric (confusion, outcome agreement, rank correlation)."""

from __future__ import annotations

import pytest

from wmh.agent.agreement import _ranks, _spearman, compute_agreement, sim_real_agreement
from wmh.agent.closed_loop import ClosedLoopReport, TaskOutcome
from wmh.agent.gold import GoldJudge
from wmh.agent.spec import HarnessSpec
from wmh.agent.tasks import TaskSpec
from wmh.core.types import Action, Observation
from wmh.engine.world_model import WorldModel
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmh.retrieval import EmbeddingRetriever, HashingEmbedder


def _report(name: str, per_task: dict[str, float]) -> ClosedLoopReport:
    outcomes = {
        tid: TaskOutcome(task_id=tid, success_rate=r, passes=3) for tid, r in per_task.items()
    }
    rates = list(per_task.values())
    return ClosedLoopReport(
        harness=name,
        success_rate=sum(rates) / len(rates) if rates else 0.0,
        per_task=outcomes,
    )


def test_perfect_agreement() -> None:
    sim = [_report("a", {"t1": 1.0, "t2": 0.0}), _report("b", {"t1": 1.0, "t2": 1.0})]
    real = [_report("a", {"t1": 1.0, "t2": 0.0}), _report("b", {"t1": 1.0, "t2": 1.0})]
    rep = compute_agreement(sim, real, k=3)
    assert rep.outcome_agreement == 1.0
    assert rep.confusion.total == 4
    assert rep.confusion.sim_pass_real_fail == 0
    assert rep.mean_abs_gap == 0.0
    # a<b in both worlds -> perfect positive rank correlation.
    assert rep.rank_correlation == pytest.approx(1.0)


def test_sim_over_credits_is_the_dangerous_cell() -> None:
    # Sim says variant passes t2; reality says it fails. That is the mirage evolution would chase.
    sim = [_report("a", {"t1": 1.0, "t2": 1.0})]
    real = [_report("a", {"t1": 1.0, "t2": 0.0})]
    rep = compute_agreement(sim, real, k=3)
    assert rep.confusion.sim_pass_real_fail == 1
    assert rep.confusion.sim_pass_real_pass == 1
    assert rep.outcome_agreement == 0.5


def test_rank_correlation_detects_inverted_ranking() -> None:
    # Sim ranks a>b>c; reality ranks a<b<c. Rank correlation should be strongly negative.
    sim = [_report("a", {"t": 0.9}), _report("b", {"t": 0.5}), _report("c", {"t": 0.1})]
    real = [_report("a", {"t": 0.1}), _report("b", {"t": 0.5}), _report("c", {"t": 0.9})]
    rep = compute_agreement(sim, real, k=3)
    assert rep.rank_correlation == pytest.approx(-1.0)


def test_unmatched_variants_and_tasks_are_skipped() -> None:
    sim = [_report("a", {"t1": 1.0, "extra": 1.0}), _report("only_sim", {"t1": 1.0})]
    real = [_report("a", {"t1": 1.0})]  # no 'extra' task, no 'only_sim' variant
    rep = compute_agreement(sim, real, k=3)
    # Only variant 'a', only task 't1' is shared -> one cell.
    assert [v.harness for v in rep.per_variant] == ["a"]
    assert rep.confusion.total == 1


def test_pass_threshold_binarizes() -> None:
    sim = [_report("a", {"t": 0.66})]  # 2/3 passes
    real = [_report("a", {"t": 0.33})]  # 1/3 passes
    # At threshold 0.5: sim passes, real fails.
    rep = compute_agreement(sim, real, k=3, pass_threshold=0.5)
    assert rep.confusion.sim_pass_real_fail == 1


def test_spearman_none_when_constant_or_singleton() -> None:
    assert _spearman([1.0], [1.0]) is None  # too few points
    assert _spearman([0.5, 0.5, 0.5], [0.1, 0.2, 0.3]) is None  # constant x -> undefined


def test_ranks_average_ties() -> None:
    # Values [10, 10, 20] -> the two 10s share rank (0+1)/2 = 0.5; 20 gets rank 2.
    assert _ranks([10.0, 10.0, 20.0]) == [0.5, 0.5, 2.0]


# --- end-to-end orchestration (offline: fake providers + an injected fake real env) ---------------


class _RoleProvider:
    """One provider playing agent + world-model + gold-judge, by inspecting the prompt.

    - agent: run one `bash` then, once it has seen an observation, `submit`.
    - world model: always reports RESULT=good (the simulator is optimistic here).
    - gold judge: pass iff the transcript shows 'good'.
    """

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
        if "grade whether an agent completed a task" in system:
            # Judge on the transcript's observation token, not the gold-assertion text (which also
            # contains the word "good"): reality passes only when the run actually saw RESULT=good.
            passed = "true" if "RESULT=good" in messages[-1].content else "false"
            return Completion(
                text='{"assertions": [{"assertion": "result is good", "passed": '
                + passed
                + ', "why": ""}], "passed": '
                + passed
                + "}"
            )
        if system.startswith("You are a capable command-line agent"):
            saw_obs = any("RESULT=" in m.content for m in messages)
            if saw_obs:
                return Completion(text='{"tool": "submit", "arguments": {"answer": "done"}}')
            return Completion(text='{"tool": "bash", "arguments": {"command": "check"}}')
        # world model: optimistic observation
        return Completion(text='{"output": "RESULT=good", "is_error": false}')

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN202
        raise NotImplementedError


class _FakeRealEnv:
    """A fake 'real' environment whose bash result is fixed per task (good or bad)."""

    def __init__(self, good: bool) -> None:
        self._good = good

    def execute(self, action: Action) -> Observation:
        return Observation(content="RESULT=good" if self._good else "RESULT=bad")

    def close(self) -> None:
        pass


def test_sim_real_agreement_end_to_end_offline() -> None:
    # Two tasks: reality passes 'good_task' and fails 'bad_task'; the simulator is optimistic about
    # both. So the agreement harness must surface exactly one sim-pass & real-FAIL cell.
    provider = _RoleProvider()
    world_model = WorldModel(provider, EmbeddingRetriever(HashingEmbedder(dim=16)))
    tasks = [
        TaskSpec(task_id="good_task", instruction="do it", gold=["result is good"]),
        TaskSpec(task_id="bad_task", instruction="do it", gold=["result is good"]),
    ]
    real_success = {"good_task": True, "bad_task": False}

    report = sim_real_agreement(
        [HarnessSpec()],
        tasks,
        world_model,
        provider,
        GoldJudge(provider),
        k=1,
        make_real_env=lambda task: _FakeRealEnv(real_success[task.task_id]),
    )

    assert len(report.per_variant) == 1
    v = report.per_variant[0]
    assert v.sim_success == 1.0  # simulator says both tasks pass
    assert v.real_success == 0.5  # reality: only good_task passes
    assert report.confusion.sim_pass_real_pass == 1
    assert report.confusion.sim_pass_real_fail == 1  # the mirage cell
    assert report.outcome_agreement == 0.5


def test_zero_overlapping_cells_reports_none_not_zero() -> None:
    # No shared task_ids -> no cells. outcome_agreement must be None ("no data"), not 0.0.
    sim = [_report("a", {"t_sim": 1.0})]
    real = [_report("a", {"t_real": 1.0})]
    rep = compute_agreement(sim, real, k=3)
    assert rep.confusion.total == 0
    assert rep.outcome_agreement is None


def test_compute_agreement_rejects_duplicate_names() -> None:
    sim = [_report("dup", {"t": 1.0}), _report("dup", {"t": 0.0})]
    real = [_report("dup", {"t": 1.0})]
    with pytest.raises(ValueError, match="duplicate harness name"):
        compute_agreement(sim, real, k=3)


def test_sim_real_agreement_rejects_duplicate_spec_names() -> None:
    provider = _RoleProvider()
    wm = WorldModel(provider, EmbeddingRetriever(HashingEmbedder(dim=16)))
    tasks = [TaskSpec(task_id="t", instruction="do it", gold=["result is good"])]
    with pytest.raises(ValueError, match="duplicate harness name"):
        sim_real_agreement(
            [HarnessSpec(name="base"), HarnessSpec(name="base")],
            tasks,
            wm,
            provider,
            GoldJudge(provider),
            k=1,
            make_real_env=lambda task: _FakeRealEnv(True),
        )


def test_sim_real_agreement_isolates_a_failing_variant() -> None:
    # One variant's real env raises; it must be skipped (recorded in failed_variants), not fatal.
    provider = _RoleProvider()
    wm = WorldModel(provider, EmbeddingRetriever(HashingEmbedder(dim=16)))
    tasks = [TaskSpec(task_id="t", instruction="do it", gold=["result is good"])]

    def make_real(task: TaskSpec):  # noqa: ANN202 - AgentEnvironment
        raise RuntimeError("sandbox exploded")  # every real leg fails; must be caught, not fatal

    report = sim_real_agreement(
        [HarnessSpec(name="ok"), HarnessSpec(name="boom")],
        tasks,
        wm,
        provider,
        GoldJudge(provider),
        k=1,
        make_real_env=make_real,
    )
    # Both variants' real leg raises -> both skipped, no crash, both recorded.
    assert set(report.failed_variants) == {"ok", "boom"}
    assert report.per_variant == []
    assert report.outcome_agreement is None


def test_agreement_report_json_roundtrips() -> None:
    # The `wmh agent verify --out` artifact must reload cleanly (incl. rank_correlation=None) so the
    # experiments tracker can consume it.
    from wmh.agent.agreement import AgreementReport, Confusion, VariantAgreement

    report = AgreementReport(
        k=3,
        pass_threshold=0.5,
        per_variant=[VariantAgreement(harness="base", sim_success=1.0, real_success=0.5)],
        confusion=Confusion(sim_pass_real_pass=1, sim_pass_real_fail=1),
        outcome_agreement=0.5,
        rank_correlation=None,
        mean_abs_gap=0.5,
    )
    assert AgreementReport.model_validate_json(report.model_dump_json()) == report
