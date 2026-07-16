"""Max-fidelity auto-configuration: find the agentic config that best fits THIS corpus.

The lever matrix is empirical and task-dependent (measured across tau/terminal/swe: reasoning
wins on tool-call APIs, live fetch on web-heavy shells, the verify self-check on hard content
prediction — and no blanket setting wins everywhere). `wmh build --max-fidelity` automates that
search: each candidate configuration is replay-scored on the build's held-out split (leak-free,
same judge and demos as `wmh eval`), the winner's flags are persisted to the artifact's
config.toml, and serving picks them up automatically. The default build stays plain RAG — the
search is strictly opt-in, and `--fidelity-budget` chooses how deep it goes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import fmean

from pydantic import BaseModel, Field

from wmh.core.types import Trace
from wmh.engine.grounding import FetchGrounder, Grounder, SourceResolver, extract_get_url
from wmh.engine.knowledge import seeded_knowledge_text
from wmh.engine.replay import replay
from wmh.engine.workspace import RepoTreeResolver
from wmh.optimize.judge import Judge
from wmh.providers.base import Embedder, Provider
from wmh.retrieval import EmbeddingRetriever

# Held-out traces scored per candidate by default: small enough that the search costs a fraction
# of the GEPA build, large enough to separate candidates beyond judge noise on most corpora.
DEFAULT_VAL_CAP = 8


@dataclass(frozen=True)
class CandidateConfig:
    """One agentic configuration the search can select (maps 1:1 onto HarnessConfig flags)."""

    label: str
    reasoning: bool = False
    knowledge: bool = False
    verify: bool = False
    grounder: str = "none"
    # Workspace grounding (pinned source files + repo tree). Research-measurable today; enters
    # build's search only once serve-side activation lands (a winner the runtime can't serve
    # would be a lie in auto_fidelity.json).
    workspace: bool = False
    # Retrieval-depth overrides (None = the engine defaults: top_k 5, no demo cap). The measured
    # "rag-deep" config: on record-heavy corpora more demos put more of the database in context
    # (tau full-slice: base 0.939 -> 0.955 at k=20+cap2000, +0.016, replicating PR #72's +0.015);
    # the cap keeps verbose corpora affordable.
    top_k: int | None = None
    demo_obs_cap: int | None = None


# Ordered by MEASURED serve cost, cheapest first (swe $/run: base 5.82, reason 6.20, workspace
# 6.48, kb 11.50, verify 12.73) — ties go to the earlier candidate, so the price-performance
# frontier wins: a cheap grounding config beats an expensive deliberation config that only
# matches it. Grounding-class candidates join right after `reason` because test-time ground
# truth is nearly free and measured as the largest lift class (fetch +0.040, workspace +0.065).
DEFAULT_CANDIDATES: tuple[CandidateConfig, ...] = (
    CandidateConfig(label="base"),
    CandidateConfig(label="reason", reasoning=True),
    CandidateConfig(label="reason+kb", reasoning=True, knowledge=True),
    CandidateConfig(label="reason+verify", reasoning=True, verify=True),
)
# Grounding-class candidates (cheap, corpus-gated). workspace needs instance pins (auto-detected
# next to the traces file); fetch is non-hermetic (hits the real web during the search) and is
# considered only when the corpus actually contains fetchable curl GETs.
WORKSPACE_CANDIDATE = CandidateConfig(label="reason+workspace", reasoning=True, workspace=True)
FETCH_CANDIDATE = CandidateConfig(label="reason+fetch", reasoning=True, grounder="fetch")
# Deep retrieval: 4x the demos, each observation capped (PR #72's optimized RAG, replicated on
# this protocol: tau +0.016 full-slice, terminal +0.004, swe +0.001 per #72). ~2x serve cost —
# it sits in the expensive tail, not the cheap frontier.
RAG_DEEP_CANDIDATE = CandidateConfig(label="rag-deep", top_k=20, demo_obs_cap=2000)
# The ladder's expensive tail: levers that ~2x the serve bill (kb rebuilds context, verify
# doubles completions). The medium tier's cheap-frontier search stops before these.
_EXPENSIVE_LABELS = frozenset({"rag-deep", "reason+kb", "reason+verify"})


@dataclass(frozen=True)
class CorpusSignature:
    """Zero-token corpus features that predict which levers can pay off.

    Measured reference points (healthy corpora, 2026-07-02): tau-bench curl=0.00/obs=414/
    tool=1.00 (winner: reason), terminal-tasks 0.43/1236/0.00 (winner: reason+fetch),
    swe-bench 0.00/889/0.00 (winner: reason+verify).
    """

    curl_get_share: float  # steps whose action is a read-only curl GET
    mean_obs_chars: float  # content-heaviness of observations
    tool_call_share: float  # structured tool-call API vs free-form bash

    @classmethod
    def from_traces(cls, traces: list[Trace]) -> CorpusSignature:
        steps = [s for t in traces for s in t.steps]
        if not steps:
            return cls(curl_get_share=0.0, mean_obs_chars=0.0, tool_call_share=0.0)
        return cls(
            curl_get_share=fmean(1.0 if extract_get_url(s.action) else 0.0 for s in steps),
            mean_obs_chars=fmean(len(s.observation.content) for s in steps),
            tool_call_share=fmean(0.0 if s.action.name == "bash" else 1.0 for s in steps),
        )


def select_candidates(
    signature: CorpusSignature,
    *,
    full_ladder: bool = False,
    has_pins: bool = False,
    cheap_only: bool = False,
) -> tuple[CandidateConfig, ...]:
    """Choose which candidates are worth spending tokens on for THIS corpus.

    Price sets the ORDER, never the menu: candidates are laddered cheapest-first (so a
    truncated budget spends on cheap tricks first, and the winner tie-break favors the cheaper
    config), but a candidate is dropped only when the corpus signature says it CANNOT matter
    here — never because a cheaper lever is also available. Fidelity picks the winner; the
    tier (cheap search vs `full_ladder`) only decides how hard we look.

    Signature gates (from the measured lever matrix, not intuition):
    - knowledge/verify: free-form (bash-like) environments; verify additionally wants
      content-heavy observations (it only ever paid off where content prediction is hardest).
    - fetch: a meaningful share of read-only curl GETs (nothing to prefetch = byte-identical
      to `reason`); workspace: instance pins exist (same no-op logic).
    `full_ladder` (the max tier) keeps only the no-op gates. `cheap_only` (the medium tier)
    truncates the ladder before the expensive deliberation levers — grounding serves at ~base
    cost, so even a budget tier can afford to discover a workspace/fetch win.
    """
    bash_like = signature.tool_call_share < 0.5
    fetchable = signature.curl_get_share >= 0.10
    # PRICE ORDER: base -> reason -> grounding class (workspace/fetch, ~free) -> kb -> verify.
    if full_ladder:
        chosen = [DEFAULT_CANDIDATES[0], DEFAULT_CANDIDATES[1]]
        if has_pins:
            chosen.append(WORKSPACE_CANDIDATE)
        if fetchable:
            chosen.append(FETCH_CANDIDATE)
        chosen.append(RAG_DEEP_CANDIDATE)
        chosen.extend([DEFAULT_CANDIDATES[2], DEFAULT_CANDIDATES[3]])
        return _maybe_cheap(tuple(chosen), cheap_only)
    chosen = [DEFAULT_CANDIDATES[0], DEFAULT_CANDIDATES[1]]  # base, reason
    if has_pins:
        chosen.append(WORKSPACE_CANDIDATE)  # cheapest strong lever when a repo pin exists
    if fetchable:
        chosen.append(FETCH_CANDIDATE)
    chosen.append(RAG_DEEP_CANDIDATE)  # never signature-gated: it never hurt anywhere measured
    if bash_like:
        chosen.append(DEFAULT_CANDIDATES[2])  # reason+kb
        if signature.mean_obs_chars >= 600:
            chosen.append(DEFAULT_CANDIDATES[3])  # reason+verify
    return _maybe_cheap(tuple(chosen), cheap_only)


def _maybe_cheap(
    candidates: tuple[CandidateConfig, ...], cheap_only: bool
) -> tuple[CandidateConfig, ...]:
    if not cheap_only:
        return candidates
    return tuple(c for c in candidates if c.label not in _EXPENSIVE_LABELS)


class WinnerSpec(BaseModel):
    """The winning candidate's resolved flags, persisted so old artifacts stay self-describing.

    Without this, `winner` is a foreign key into the in-code candidate tuple — and the ladder
    churns (this PR alone added three candidates), so a rename would break `--max-fidelity`
    loads of every previously built artifact.
    """

    label: str
    reasoning: bool = False
    knowledge: bool = False
    verify: bool = False
    grounder: str = "none"
    workspace: bool = False
    top_k: int | None = None
    demo_obs_cap: int | None = None

    @classmethod
    def from_candidate(cls, candidate: CandidateConfig) -> WinnerSpec:
        return cls(
            label=candidate.label,
            reasoning=candidate.reasoning,
            knowledge=candidate.knowledge,
            verify=candidate.verify,
            grounder=candidate.grounder,
            workspace=candidate.workspace,
            top_k=candidate.top_k,
            demo_obs_cap=candidate.demo_obs_cap,
        )

    def to_candidate(self) -> CandidateConfig:
        return CandidateConfig(
            label=self.label,
            reasoning=self.reasoning,
            knowledge=self.knowledge,
            verify=self.verify,
            grounder=self.grounder,
            workspace=self.workspace,
            top_k=self.top_k,
            demo_obs_cap=self.demo_obs_cap,
        )


class AutoFidelityReport(BaseModel):
    """The search's outcome, persisted into the artifact for provenance."""

    winner_label: str
    scores: dict[str, float] = Field(default_factory=dict)
    val_traces: int = 0
    considered: list[str] = Field(default_factory=list)  # candidate labels after pruning
    # The winner's resolved flags (None only in pre-WinnerSpec artifacts, which fall back to
    # the in-code label lookup).
    winner_spec: WinnerSpec | None = None

    @property
    def winner(self) -> CandidateConfig:
        if self.winner_spec is not None:
            return self.winner_spec.to_candidate()
        for candidate in (
            *DEFAULT_CANDIDATES,
            FETCH_CANDIDATE,
            WORKSPACE_CANDIDATE,
            RAG_DEEP_CANDIDATE,
        ):
            if candidate.label == self.winner_label:
                return candidate
        raise ValueError(f"unknown winner label {self.winner_label!r}")


def search_max_fidelity(
    prompt: str,
    train: list[Trace],
    val: list[Trace],
    provider: Provider,
    judge: Judge,
    embedder: Embedder | None,
    *,
    val_cap: int = DEFAULT_VAL_CAP,
    top_k: int = 5,
    seed: int = 0,
    concurrency: int = 4,
    candidates: Sequence[CandidateConfig] | None = None,
    full_ladder: bool = False,
    cheap_only: bool = False,
    knowledge_text: str | None = None,
    source_pins: str | None = None,
    on_candidate_start: Callable[[str], None] | None = None,
    on_candidate_done: Callable[[str, float], None] | None = None,
) -> AutoFidelityReport:
    """Replay-score the candidate configs on (a cap of) the held-out split; return the winner.

    `candidates=None` computes the corpus signature (zero tokens) and prunes the ladder to the
    levers that can matter for this corpus (`full_ladder=True` skips the pruning — the max
    tier's "be certain" mode). Leak-free by construction: demos and the candidate knowledge
    base both come from `train` only, and the scored `val` traces are the build's held-out
    split. The winner is the highest mean fidelity, ties resolved toward the earlier (cheaper)
    candidate — so `base` (plain RAG) stays the answer unless an agentic config measurably
    beats it.
    """
    scored_val = val[:val_cap]
    if candidates is None:
        signature = CorpusSignature.from_traces(train)
        candidates = select_candidates(
            signature,
            full_ladder=full_ladder,
            has_pins=source_pins is not None,
            cheap_only=cheap_only,
        )
    source = SourceResolver.from_file(source_pins) if source_pins is not None else None
    tree = RepoTreeResolver(source.pins) if source is not None else None
    # The candidate KB, seeded once (train-only) and reused for every knowledge candidate.
    # `knowledge_text` lets the caller supply the EXACT text the artifact will serve (build
    # seeds into the artifact dir first), so the winning score was measured on the KB that
    # ships — a second independent extraction would be a different nondeterministic text.
    kb_text = knowledge_text
    if kb_text is None and any(c.knowledge for c in candidates):
        kb_text = seeded_knowledge_text(train, provider)

    scores: dict[str, float] = {}
    best: CandidateConfig = candidates[0]
    best_score = -1.0
    for candidate in candidates:
        if on_candidate_start is not None:
            on_candidate_start(candidate.label)
        grounder: Grounder | None = FetchGrounder() if candidate.grounder == "fetch" else None
        use_ws = candidate.workspace and source is not None
        report = replay(
            prompt,
            scored_val,
            provider,
            judge,
            retriever=EmbeddingRetriever(embedder) if embedder is not None else None,
            train=train if embedder is not None else None,
            top_k=candidate.top_k if candidate.top_k is not None else top_k,
            max_retrieved_observation_chars=candidate.demo_obs_cap,
            sample_turns="sampled",
            seed=seed,
            concurrency=concurrency,
            knowledge=kb_text if candidate.knowledge else None,
            reasoning=candidate.reasoning,
            verify=candidate.verify,
            grounder=grounder,
            source=source if use_ws else None,
            source_annotate_stale=use_ws,
            tree=tree if use_ws else None,
        )
        scores[candidate.label] = report.mean_score
        if on_candidate_done is not None:
            on_candidate_done(candidate.label, report.mean_score)
        if report.mean_score > best_score:
            best, best_score = candidate, report.mean_score

    return AutoFidelityReport(
        winner_label=best.label,
        scores=scores,
        val_traces=len(scored_val),
        considered=[c.label for c in candidates],
        winner_spec=WinnerSpec.from_candidate(best),
    )
