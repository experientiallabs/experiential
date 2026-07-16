"""Tests for the max-fidelity auto-configuration search (fakes, no network)."""

from __future__ import annotations

from wmh.core.types import Action, ActionKind, Observation, Step, Trace
from wmh.engine.autoconfig import (
    DEFAULT_CANDIDATES,
    FETCH_CANDIDATE,
    CandidateConfig,
    CorpusSignature,
    search_max_fidelity,
    select_candidates,
)
from wmh.optimize.judge import JudgeResult
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind


class _FakeProvider:
    def __init__(self) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        if "KNOWLEDGE BASE for a simulated" in system:  # KB seeding extraction
            return Completion(text='{"rules": "- gate: x", "entities": "", "schemas": ""}')
        return Completion(text='{"output": "ok", "is_error": false}')

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN202
        raise NotImplementedError


class _ModeAwareJudge:
    """Scores by which agentic features were active — lets tests pick a designed winner."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores
        self.active_label = "base"

    def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult:
        return JudgeResult(score=self._scores.get(self.active_label, 0.5), critique="ok")


def _traces(n: int) -> list[Trace]:
    return [
        Trace(
            trace_id=f"t{i:03d}",
            steps=[
                Step(
                    action=Action(kind=ActionKind.TOOL_CALL, name="get", arguments={"i": i}),
                    observation=Observation(content=f"obs {i}"),
                )
            ],
        )
        for i in range(n)
    ]


def test_default_candidates_start_from_plain_rag() -> None:
    # The default stays "just RAG": base is the first candidate and the tie-break favors it.
    assert DEFAULT_CANDIDATES[0] == CandidateConfig(label="base")
    labels = [c.label for c in DEFAULT_CANDIDATES]
    assert labels == ["base", "reason", "reason+kb", "reason+verify"]


def test_search_picks_the_highest_scoring_candidate() -> None:
    judge = _ModeAwareJudge({"base": 0.5, "reason": 0.7, "reason+kb": 0.6, "reason+verify": 0.65})
    report = search_max_fidelity(
        "BASE",
        _traces(20),
        _traces(4),
        _FakeProvider(),
        judge,
        None,
        val_cap=2,
        candidates=DEFAULT_CANDIDATES,
        on_candidate_start=lambda label: setattr(judge, "active_label", label),
    )
    assert report.winner.label == "reason"
    assert report.scores["reason"] == 0.7
    assert set(report.scores) == {"base", "reason", "reason+kb", "reason+verify"}
    assert report.considered == ["base", "reason", "reason+kb", "reason+verify"]


def test_search_ties_go_to_the_cheaper_candidate() -> None:
    judge = _ModeAwareJudge({})  # every candidate scores the identical default 0.5
    report = search_max_fidelity(
        "BASE", _traces(20), _traces(4), _FakeProvider(), judge, None, val_cap=2
    )
    assert report.winner.label == "base"  # first (cheapest) wins on a tie


def test_search_respects_val_cap_and_reports_it() -> None:
    judge = _ModeAwareJudge({})
    report = search_max_fidelity(
        "BASE", _traces(20), _traces(10), _FakeProvider(), judge, None, val_cap=3
    )
    assert report.val_traces == 3


def test_auto_selection_prunes_by_corpus_signature() -> None:
    # A tool-call corpus (tau-like): only the cheap candidates are worth scoring.
    judge = _ModeAwareJudge({})
    report = search_max_fidelity(
        "BASE", _traces(20), _traces(4), _FakeProvider(), judge, None, val_cap=2
    )
    assert report.considered == ["base", "reason", "rag-deep"]  # kb/verify/fetch pruned


def test_select_candidates_is_a_price_ordered_frontier() -> None:
    # Ladder order is SERVE PRICE, cheapest first (measured $/run: base < reason < grounding
    # class < kb < verify) — a cheap grounding config that ties an expensive deliberation
    # config must win the tie-break. Price never shrinks the MENU: a candidate drops out only
    # when the signature says it cannot matter on this corpus.
    tau = CorpusSignature(curl_get_share=0.0, mean_obs_chars=414, tool_call_share=1.0)
    terminal = CorpusSignature(curl_get_share=0.43, mean_obs_chars=1236, tool_call_share=0.0)
    swe = CorpusSignature(curl_get_share=0.0, mean_obs_chars=889, tool_call_share=0.0)
    # rag-deep is never signature-gated: it never hurt anywhere measured and wins on
    # record-heavy corpora (tau full-slice 0.939 -> 0.955 at k=20+cap).
    assert [c.label for c in select_candidates(tau)] == ["base", "reason", "rag-deep"]
    assert [c.label for c in select_candidates(terminal)] == [
        "base",
        "reason",
        "reason+fetch",  # grounding BEFORE kb: cheaper, tried first
        "rag-deep",
        "reason+kb",
        "reason+verify",  # still on the menu — a cheaper lever existing must not evict it
    ]
    # A pinnable code corpus leads with workspace; verify stays admitted (content-heavy) —
    # fidelity, not price, decides between them at scoring time.
    assert [c.label for c in select_candidates(swe, has_pins=True)] == [
        "base",
        "reason",
        "reason+workspace",
        "rag-deep",
        "reason+kb",
        "reason+verify",
    ]
    # Unpinnable code corpus: the old deliberation ladder remains.
    assert [c.label for c in select_candidates(swe)] == [
        "base",
        "reason",
        "rag-deep",
        "reason+kb",
        "reason+verify",
    ]
    # medium tier (cheap_only): the ladder stops before the expensive deliberation levers,
    # but the cheap grounding class is always searchable — cheap wins must be discoverable.
    assert [c.label for c in select_candidates(terminal, cheap_only=True)] == [
        "base",
        "reason",
        "reason+fetch",
    ]
    assert [c.label for c in select_candidates(swe, has_pins=True, cheap_only=True)] == [
        "base",
        "reason",
        "reason+workspace",
    ]
    # max tier: full ladder, grounding candidates still corpus-gated.
    full = [c.label for c in select_candidates(terminal, full_ladder=True)]
    assert full.index("reason+fetch") < full.index("reason+kb")
    assert FETCH_CANDIDATE not in select_candidates(swe, full_ladder=True)
    assert "reason+workspace" in [
        c.label for c in select_candidates(swe, full_ladder=True, has_pins=True)
    ]
