"""End-to-end create-loop tests: one provider plays all four roles, no network, no entropy.

Extends the closed-loop test pattern (`closed_loop_test.RoleProvider`) with a fourth role: the
meta-agent, keyed on `MUTATE_SYSTEM`'s distinctive phrase. The agent role is the FALLBACK after
the judge/meta/world-model markers, because a variant's system prompt is exactly what the search
rewrites — no marker on it is stable across generations. The fake judge echoes the gold assertions
verbatim from its prompt (the real judge is scored by text-matching those echoes) and passes or
fails a run based on the submitted answer, so seed and child scores can genuinely differ.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from wmh.engine.world_model import WorldModel
from wmh.evals.closed_loop import ClosedLoopReport, TaskOutcome
from wmh.evals.gold import AssertionResult, GoldJudge, GoldVerdict
from wmh.evals.tasks import TaskSpec
from wmh.harness.create import (
    CreateResult,
    PoolEntry,
    cluster_failures,
    create_harness,
    select_parent,
)
from wmh.harness.doc import HarnessDoc
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmh.retrieval import EmbeddingRetriever, HashingEmbedder

_CAREFUL_PROMPT = "You are a careful agent. Verify the state of the system before submitting."


def _meta_reply(parent: HarnessDoc, new_prompt: str) -> str:
    """A well-formed delta against `parent`, preconditioned on its actual prompt hash."""
    core = parent.surface("prompt:core")
    assert core is not None
    return json.dumps(
        {
            "expected_effect": "the failing tasks flip to pass",
            "preconditions": {"prompt:core": core.content_hash},
            "ops": [
                {
                    "op": "replace",
                    "surface_id": "prompt:core",
                    "content": new_prompt,
                    "rationale": "make the agent verify before submitting",
                }
            ],
        }
    )


class RoleProvider:
    """Plays agent, world model, gold judge, and meta-agent, keyed off the system prompt."""

    def __init__(
        self,
        *,
        meta_reply: str = "not json at all",
        judge_fn: Callable[[str], bool] | None = None,
    ) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")
        self._meta_reply = meta_reply
        self.meta_users: list[str] = []  # every proposer prompt, for history assertions
        # Default: a run passes iff the agent submitted the verified answer.
        self._judge_fn = judge_fn if judge_fn is not None else lambda u: "done-verified" in u

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        user = messages[-1].content
        if "grade whether an agent completed a task" in system:
            passed = self._judge_fn(user)
            results = [
                {"assertion": a, "passed": passed, "why": "x"} for a in _gold_assertions(user)
            ]
            return Completion(text=json.dumps({"assertions": results, "passed": passed}))
        if "meta-agent improving an agent harness" in system:
            self.meta_users.append(user)
            return Completion(text=self._meta_reply)
        if "You ARE the environment" in system:
            return Completion(text='{"output": "ok", "is_error": false}')
        # Fallback: the agent role. What it submits depends on the prompt the variant carries.
        if "careful agent" in system:
            answer = "done-verified"
        elif "broken agent" in system:
            answer = "done-broken"
        else:
            answer = "done"
        return Completion(text=json.dumps({"tool": "submit", "arguments": {"answer": answer}}))

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201 - test fake never calls it
        raise NotImplementedError


def _gold_assertions(user: str) -> list[str]:
    """The gold list the judge prompt carries, echoed back verbatim."""
    _, _, tail = user.partition("GOLD ASSERTIONS")
    return [line[2:] for line in tail.splitlines() if line.startswith("- ")]


def _wm(provider: RoleProvider) -> WorldModel:
    return WorldModel(provider, EmbeddingRetriever(HashingEmbedder(dim=16)))


def _tasks() -> list[TaskSpec]:
    return [TaskSpec(task_id="t1", instruction="answer it", gold=["the work was verified"])]


def _run(
    provider: RoleProvider,
    *,
    iterations: int = 1,
    k: int = 3,
    holdout: list[TaskSpec] | None = None,
    on_progress: Callable[[int, str, float, bool], None] | None = None,
) -> CreateResult:
    return create_harness(
        "winner",
        HarnessDoc.baseline("seed"),
        _tasks(),
        _wm(provider),
        provider,
        provider,
        GoldJudge(provider),
        iterations=iterations,
        k=k,
        holdout=holdout,
        on_progress=on_progress,
    )


def test_create_accepts_improving_delta_and_promotes_suite() -> None:
    seed = HarnessDoc.baseline("seed")
    provider = RoleProvider(meta_reply=_meta_reply(seed, _CAREFUL_PROMPT))
    progress: list[tuple[int, str, float, bool]] = []
    result = _run(provider, on_progress=lambda i, n, r, a: progress.append((i, n, r, a)))

    assert result.skipped == 0
    assert result.best_score == 1.0
    assert result.best.name == "winner"
    assert result.best.system_prompt() == _CAREFUL_PROMPT
    assert progress == [(0, "seed", 0.0, True), (1, "winner-g1", 1.0, True)]

    [delta] = result.archive.deltas
    assert delta.verdict is not None and delta.verdict.accepted
    assert delta.verdict.full_delta == 1.0
    assert delta.verdict.holdout_delta is None
    assert "1/1 tasks now pass" in delta.verdict.reason
    # The trigger came from deterministic clustering of the seed's failures.
    assert delta.trigger.mechanism == "the work was verified"
    assert delta.trigger.task_ids == ["t1"]
    # The newly-passing task promoted into the regression suite.
    assert result.suite == ["t1"]
    # Reports are keyed by content: seed and child doc hashes, k=3 passes each.
    assert set(result.reports) == {seed.doc_hash, delta.child_doc_hash}
    assert all(r.k == 3 for r in result.reports.values())


def test_archive_reconstructs_children_by_folding_deltas() -> None:
    seed = HarnessDoc.baseline("seed")
    provider = RoleProvider(meta_reply=_meta_reply(seed, _CAREFUL_PROMPT))
    result = _run(provider)
    [delta] = result.archive.deltas
    assert delta.child_doc_hash is not None
    rebuilt = result.archive.reconstruct(delta.child_doc_hash)
    assert rebuilt.surfaces == result.best.surfaces
    with pytest.raises(ValueError, match="not in this archive"):
        result.archive.reconstruct("0" * 32)


def test_create_rejects_regressing_delta_and_keeps_champion() -> None:
    # The seed already passes; the proposed prompt makes the agent submit a broken answer.
    seed = HarnessDoc.baseline("seed")
    provider = RoleProvider(
        meta_reply=_meta_reply(seed, "You are a broken agent."),
        judge_fn=lambda user: "done-broken" not in user,
    )
    result = _run(provider)

    assert result.skipped == 0
    [delta] = result.archive.deltas
    assert delta.verdict is not None and not delta.verdict.accepted
    assert "full split" in delta.verdict.reason
    assert result.archive.accepted() == []
    # The champion never moved: the winner is the (renamed) seed at its original score.
    assert result.best_score == 1.0
    assert result.best.system_prompt() == seed.system_prompt()
    assert result.suite == ["t1"]  # and the suite kept the seed's win
    # An all-pass parent gets the generalization trigger, not a fabricated failure.
    assert delta.trigger.mechanism == "none: all tasks pass"


def test_create_skips_unusable_proposals() -> None:
    provider = RoleProvider(meta_reply="not json at all")
    result = _run(provider, iterations=2)
    assert result.skipped == 2
    assert result.archive.deltas == []  # nothing to audit: no delta object ever existed
    assert result.best.name == "winner"  # even a search with no children yields the renamed seed


def test_create_audits_invalid_delta_without_spending_eval() -> None:
    stale = json.dumps(
        {
            "expected_effect": "x",
            "preconditions": {"prompt:core": "0" * 32},
            "ops": [
                {
                    "op": "replace",
                    "surface_id": "prompt:core",
                    "content": _CAREFUL_PROMPT,
                    "rationale": "r",
                }
            ],
        }
    )
    provider = RoleProvider(meta_reply=stale)
    result = _run(provider)
    assert result.skipped == 1
    [delta] = result.archive.deltas
    assert delta.verdict is not None and not delta.verdict.accepted
    assert delta.verdict.reason.startswith("invalid before eval")
    assert delta.child_doc_hash is None  # it never applied, so it never produced a doc
    assert len(result.reports) == 1  # only the seed was ever scored


def test_holdout_regression_rejects_a_full_split_win() -> None:
    # The delta fixes the main task but breaks the held-out one: tiers 1-2 pass, tier 3 rejects.
    seed = HarnessDoc.baseline("seed")

    def judge(user: str) -> bool:
        if "the holdout task" in user:
            return "done-verified" not in user  # holdout passes only for the seed's plain answer
        return "done-verified" in user

    provider = RoleProvider(meta_reply=_meta_reply(seed, _CAREFUL_PROMPT), judge_fn=judge)
    holdout = [TaskSpec(task_id="h1", instruction="the holdout task", gold=["the base flow works"])]
    result = _run(provider, holdout=holdout)

    [delta] = result.archive.deltas
    assert delta.verdict is not None and not delta.verdict.accepted
    assert delta.verdict.full_delta == 1.0  # it really did win the full split...
    assert delta.verdict.holdout_delta == -1.0  # ...and really did regress held-out
    assert "held-out regressed" in delta.verdict.reason
    assert result.best_score == 0.0  # champion stays the seed
    assert set(result.holdout_reports) == {seed.doc_hash, delta.child_doc_hash}


# -- deterministic failure clustering ---------------------------------------------------------


def _failing(task_id: str, unmet: list[str]) -> TaskOutcome:
    verdict = GoldVerdict(
        passed=False,
        fraction=0.0,
        assertions=[AssertionResult(assertion=a, passed=False, why="w") for a in unmet],
    )
    return TaskOutcome(
        task_id=task_id, success_rate=0.0, mean_fraction=0.0, passes=2, verdicts=[verdict, verdict]
    )


def test_cluster_failures_groups_by_shared_assertions() -> None:
    report = ClosedLoopReport(
        per_task={
            "t1": _failing("t1", ["a", "b"]),
            "t2": _failing("t2", ["b", "c"]),
            "t3": _failing("t3", ["z"]),
            "t4": TaskOutcome(task_id="t4", success_rate=1.0, mean_fraction=1.0, passes=2),
            "t5": _failing("t5", []),  # unparseable judge: no per-assertion detail
        }
    )
    tasks = [TaskSpec(task_id=t, instruction=t) for t in ("t1", "t2", "t3", "t4", "t5")]
    clusters = cluster_failures(report, tasks)
    assert [c.task_ids for c in clusters] == [["t1", "t2"], ["t5"], ["t3"]]
    # t1+t2 connect through shared assertion "b", which also labels the mechanism.
    assert clusters[0].mechanism == "b"
    assert clusters[0].unmet_assertions == ["a", "b", "c"]
    assert clusters[1].mechanism == "run failed without per-assertion verdicts"
    assert clusters[2].mechanism == "z"


def test_cluster_failures_empty_when_everything_passes() -> None:
    report = ClosedLoopReport(
        per_task={"t1": TaskOutcome(task_id="t1", success_rate=1.0, mean_fraction=1.0, passes=3)}
    )
    assert cluster_failures(report, [TaskSpec(task_id="t1", instruction="i")]) == []


# -- parent selection --------------------------------------------------------------------------


def _entry(doc_hash: str, success_rate: float) -> PoolEntry:
    return PoolEntry(doc_hash=doc_hash, name=doc_hash, success_rate=success_rate)


def test_select_parent_is_deterministic_and_discounts_expanded_parents() -> None:
    entries = [_entry("a", 0.5), _entry("b", 1.0)]
    assert select_parent(entries, {}, seed=7) == select_parent(entries, {}, seed=7)
    picks_fresh = [select_parent(entries, {}, seed=s).doc_hash for s in range(50)]
    picks_worn = [select_parent(entries, {"b": 9}, seed=s).doc_hash for s in range(50)]
    # Both variants are reachable, and expanding a parent shrinks its share of selections.
    assert set(picks_fresh) == {"a", "b"}
    assert picks_worn.count("b") < picks_fresh.count("b")


def test_select_parent_rejects_empty_pool() -> None:
    with pytest.raises(ValueError, match="empty"):
        select_parent([], {}, seed=1)


# -- staged verification: screening + history ---------------------------------------------------


def _useless_meta_reply(parent: HarnessDoc) -> str:
    """A well-formed delta that changes wording but cannot fix the failing task."""
    return _meta_reply(parent, "You are an agent. Do the task.")


def test_screen_rejects_delta_that_does_not_improve_its_trigger() -> None:
    seed = HarnessDoc.baseline("seed")
    provider = RoleProvider(meta_reply=_useless_meta_reply(seed))
    progress: list[tuple[int, str, float, bool]] = []
    result = _run(provider, on_progress=lambda i, n, r, a: progress.append((i, n, r, a)))

    assert result.screened == 1 and result.skipped == 0
    [delta] = result.archive.deltas
    assert delta.verdict is not None and not delta.verdict.accepted
    assert delta.verdict.reason.startswith("screened out")
    # The cheap screen replaced the full eval: only the seed has a full-split report,
    # and no child progress event ever fired.
    assert len(result.reports) == 1
    assert [e[0] for e in progress] == [0]


def test_rejected_history_reaches_the_next_proposal() -> None:
    seed = HarnessDoc.baseline("seed")
    provider = RoleProvider(meta_reply=_useless_meta_reply(seed))
    _run(provider, iterations=2)
    assert len(provider.meta_users) == 2
    assert "Previous attempts" not in provider.meta_users[0]
    assert "Previous attempts" in provider.meta_users[1]
    assert "screened out" in provider.meta_users[1]  # the verdict itself is the lesson


# -- code deltas end to end ----------------------------------------------------------------------


def _code_meta_reply(parent: HarnessDoc) -> str:
    from wmh.harness.doc import CODE_RUNTIME_ID

    code_surface = parent.surface(CODE_RUNTIME_ID)
    assert code_surface is not None
    new_code = (
        "def run(kit):\n"
        '    kit.execute("bash", {"command": "verify the work"})\n'
        '    return "done-verified"\n'
    )
    return json.dumps(
        {
            "expected_effect": "the failing task flips to pass",
            "preconditions": {CODE_RUNTIME_ID: code_surface.content_hash},
            "ops": [
                {
                    "op": "replace",
                    "surface_id": CODE_RUNTIME_ID,
                    "content": new_code,
                    "rationale": "verify via the environment before submitting",
                }
            ],
        }
    )


def test_code_delta_passes_screen_and_gate_end_to_end() -> None:
    from wmh.harness.doc import CODE_RUNTIME_ID, code_baseline

    seed = code_baseline("seed")
    provider = RoleProvider(meta_reply=_code_meta_reply(seed))
    result = create_harness(
        "winner",
        seed,
        _tasks(),
        _wm(provider),
        provider,
        provider,
        GoldJudge(provider),
        iterations=1,
        k=3,
    )
    assert result.screened == 0 and result.skipped == 0
    [delta] = result.archive.deltas
    assert delta.verdict is not None and delta.verdict.accepted
    assert [op.surface_id for op in delta.ops] == [CODE_RUNTIME_ID]
    assert result.best_score == 1.0
    winner_code = result.best.surface(CODE_RUNTIME_ID)
    assert winner_code is not None and "done-verified" in winner_code.content


def test_broken_code_delta_is_rejected_before_any_eval() -> None:
    from wmh.harness.doc import CODE_RUNTIME_ID, code_baseline

    seed = code_baseline("seed")
    code_surface = seed.surface(CODE_RUNTIME_ID)
    assert code_surface is not None
    broken = json.dumps(
        {
            "expected_effect": "x",
            "preconditions": {CODE_RUNTIME_ID: code_surface.content_hash},
            "ops": [
                {
                    "op": "replace",
                    "surface_id": CODE_RUNTIME_ID,
                    "content": "def run(kit:\n",
                    "rationale": "r",
                }
            ],
        }
    )
    provider = RoleProvider(meta_reply=broken)
    result = create_harness(
        "winner",
        seed,
        _tasks(),
        _wm(provider),
        provider,
        provider,
        GoldJudge(provider),
        iterations=1,
        k=2,
    )
    assert result.skipped == 1
    [delta] = result.archive.deltas
    assert delta.verdict is not None and "does not compile" in delta.verdict.reason
    assert len(result.reports) == 1  # only the seed was ever scored


# -- confirmation re-runs -----------------------------------------------------------------------


def test_narrow_failing_tiers_eligibility() -> None:
    from wmh.harness.create import narrow_failing_tiers
    from wmh.harness.delta import GateRecord

    def record(**kw) -> GateRecord:  # noqa: ANN003
        return GateRecord(accepted=False, reason="r", **kw)

    # Narrow holdout veto on a full-split win -> retry that tier.
    narrow = record(full_delta=0.05, holdout_delta=-0.1)
    assert narrow_failing_tiers(narrow, k=5, n_suite=4, n_holdout=4) == ["holdout"]
    # A wide veto is a real regression, not noise: ineligible.
    wide = record(full_delta=0.05, holdout_delta=-1.0)
    assert narrow_failing_tiers(wide, k=5, n_suite=4, n_holdout=4) is None
    # No full-split win: nothing to confirm.
    no_win = record(full_delta=0.0, holdout_delta=-0.1)
    assert narrow_failing_tiers(no_win, k=5, n_suite=4, n_holdout=4) is None
    # Both tiers narrowly failing -> both retried.
    both = record(full_delta=0.05, suite_delta=-0.05, holdout_delta=-0.1)
    assert narrow_failing_tiers(both, k=5, n_suite=8, n_holdout=4) == ["suite", "holdout"]
    # Accepted verdicts are never retried.
    ok = GateRecord(accepted=True, reason="r", full_delta=0.05)
    assert narrow_failing_tiers(ok, k=5, n_suite=4, n_holdout=4) is None


def test_flaky_holdout_veto_is_overturned_by_confirmation() -> None:
    # The child genuinely fixes the train task; the holdout task fails exactly ONE child
    # attempt (judge flakiness). The initial k-pass gate vetoes; the 2k re-measurement of
    # child AND champion overturns it.
    seed = HarnessDoc.baseline("seed")
    child_h1_calls = {"n": 0}

    def judge(user: str) -> bool:
        if "the holdout task" in user:
            if "done-verified" in user:  # the child's answer style
                child_h1_calls["n"] += 1
                return child_h1_calls["n"] != 1  # fail only the first child attempt
            return True  # the seed always passes holdout
        return "done-verified" in user  # train task needs the careful child

    provider = RoleProvider(meta_reply=_meta_reply(seed, _CAREFUL_PROMPT), judge_fn=judge)
    holdout = [TaskSpec(task_id="h1", instruction="the holdout task", gold=["the base flow works"])]
    result = _run(provider, holdout=holdout, k=5)

    assert result.confirmations == 1
    [delta] = result.archive.deltas
    assert delta.verdict is not None and delta.verdict.accepted
    assert "veto overturned" in delta.verdict.reason
    assert "initially: rejected" in delta.verdict.reason
    assert result.best_score == 1.0  # the win was kept


def test_wide_holdout_regression_skips_confirmation() -> None:
    # Same setup as the round-2 holdout test: the child ALWAYS fails held-out. -1.0 is far
    # beyond the narrow margin, so no re-measurement is spent and the plain rejection stands.
    seed = HarnessDoc.baseline("seed")

    def judge(user: str) -> bool:
        if "the holdout task" in user:
            return "done-verified" not in user
        return "done-verified" in user

    provider = RoleProvider(meta_reply=_meta_reply(seed, _CAREFUL_PROMPT), judge_fn=judge)
    holdout = [TaskSpec(task_id="h1", instruction="the holdout task", gold=["the base flow works"])]
    result = _run(provider, holdout=holdout)
    assert result.confirmations == 0
    [delta] = result.archive.deltas
    assert delta.verdict is not None and not delta.verdict.accepted
    assert "held-out regressed" in delta.verdict.reason


def test_confirmed_suite_overturn_still_faces_the_holdout_tier() -> None:
    # A suite veto narrow enough to overturn must NOT smuggle the child past held-out
    # verification: here the suite flake clears on re-measurement but the child genuinely
    # regresses held-out, so the final verdict is a holdout rejection.
    seed = HarnessDoc.baseline("seed")
    suite_flake = {"n": 0}

    def judge(user: str) -> bool:
        if "the holdout task" in user:
            return "done-verified" not in user  # child ALWAYS fails held-out (wide, real)
        if "suite task" in user:
            if "done-verified" in user:
                suite_flake["n"] += 1
                return suite_flake["n"] != 1  # one flaky failure for the child
            return True  # seed masters the suite task
        return "done-verified" in user  # the trigger task needs the careful child

    tasks = [
        TaskSpec(task_id="t1", instruction="answer it", gold=["the work was verified"]),
        TaskSpec(task_id="s1", instruction="the suite task", gold=["steady state holds"]),
    ]
    holdout = [TaskSpec(task_id="h1", instruction="the holdout task", gold=["the base flow works"])]
    provider = RoleProvider(meta_reply=_meta_reply(seed, _CAREFUL_PROMPT), judge_fn=judge)
    result = create_harness(
        "winner",
        seed,
        tasks,
        _wm(provider),
        provider,
        provider,
        GoldJudge(provider),
        iterations=1,
        k=5,
        holdout=holdout,
    )
    [delta] = result.archive.deltas
    assert delta.verdict is not None and not delta.verdict.accepted
    # The holdout tier was measured (not bypassed) and its regression is the rejection.
    assert delta.verdict.holdout_delta is not None and delta.verdict.holdout_delta < 0
    assert result.best_score == pytest.approx(0.5)  # champion stayed the seed
