"""Open-loop evaluation: reconstruction fidelity over trace files (the default `wmh eval` mode).

`replay` (in `wmh.engine.replay`) scores one corpus of held-out steps teacher-forced. This
orchestration layer is what `wmh eval` calls: it loads one or more OTel trace files, splits each
into train/holdout, replays the holdout through a world-model prompt with leak-free RAG, and
aggregates a per-file + overall scorecard. Its closed-loop counterpart
(`wmh eval --mode closed-loop`, `wmh.evals.closed_loop`) runs a live agent instead of replaying;
both implement the `Evaluation` interface in `wmh.evals.base`.
"""

from __future__ import annotations

from pathlib import Path
from statistics import fmean, pstdev

from pydantic import BaseModel, Field

from wmh.engine.build import split_traces
from wmh.engine.knowledge import seeded_knowledge_text
from wmh.engine.replay import ReplayReport, replay, valid_scores
from wmh.ingest import get_adapter
from wmh.optimize.judge import Judge
from wmh.providers.base import Embedder, Provider
from wmh.retrieval import EmbeddingRetriever


class EvalReport(BaseModel):
    """Per-file fidelity reports plus the step-weighted overall mean ± std.

    `per_file` maps a trace file's clean name to its `ReplayReport` (per-step `StepResult`s), and
    `overall_fidelity`/`overall_std` are the step-weighted aggregates across files.
    """

    per_file: dict[str, ReplayReport] = Field(default_factory=dict)
    overall_fidelity: float = 0.0  # step-weighted mean of valid per-step scores across all files
    overall_std: float = 0.0  # std of valid per-step scores across all files
    total_steps: int = 0  # all steps attempted, including judge-invalid ones
    total_invalid: int = 0  # judge failures across files; excluded from fidelity/std

    @property
    def headline(self) -> float:
        """The `EvalResult` headline: per-step reconstruction fidelity."""
        return self.overall_fidelity

    @property
    def total_valid(self) -> int:
        """Steps that actually back the fidelity mean (judge-invalid ones excluded)."""
        return self.total_steps - self.total_invalid

    def summary(self) -> str:
        invalid = f", {self.total_invalid} judge-invalid excluded" if self.total_invalid else ""
        return (
            f"fidelity={self.overall_fidelity:.3f}±{self.overall_std:.3f} "
            f"({self.total_steps} steps, {len(self.per_file)} file(s){invalid})"
        )


def evaluate_files(
    files: list[Path],
    prompt: str,
    provider: Provider,
    judge: Judge,
    *,
    embedder: Embedder | None = None,
    train_split: float = 0.7,
    top_k: int = 5,
    sample_turns: str = "all",
    seed: int = 0,
    adapter_name: str = "otel-genai",
    knowledge: bool = False,
    reasoning: bool = False,
) -> EvalReport:
    """Replay-score each trace file's held-out split. `embedder=None` -> zero-shot (no retrieval).

    Each file is split deterministically; tiny corpora with no held-out trace fall back to scoring
    every trace. RAG, when enabled, retrieves from that file's own train split only (leak-free).
    `sample_turns`/`seed` are forwarded to `replay` (see its docstring).

    `knowledge` seeds an ephemeral knowledge base from each file's TRAIN split (never the holdout —
    the same leak-free discipline as RAG) and renders it into every prediction; `reasoning`
    switches predictions to the deliberate-then-answer contract. Both mirror the serving engine's
    agentic mode. Closed-loop evals get agentic mode from the ARTIFACT instead (the served
    WorldModel's config / --max-fidelity winner), not from these flags.
    """
    adapter = get_adapter(adapter_name)
    per_file: dict[str, ReplayReport] = {}
    for path in files:
        traces = adapter.from_file(str(path))
        if not traces:
            continue
        train, holdout = split_traces(traces, train_split)
        if not holdout:  # tiny corpus: evaluate on everything
            train, holdout = traces, traces
        retriever = EmbeddingRetriever(embedder) if embedder is not None else None
        # Ephemeral, per-file, train-only KB: rendered text only — nothing under models/ is read
        # or written, so eval can never leak a serve-time learned.md into scoring.
        knowledge_text = seeded_knowledge_text(train, provider) if knowledge else None
        name = _display_name(path)
        per_file[name] = replay(
            prompt,
            holdout,
            provider,
            judge,
            retriever=retriever,
            train=train if embedder is not None else None,
            top_k=top_k,
            sample_turns=sample_turns,
            seed=seed,
            knowledge=knowledge_text,
            reasoning=reasoning,
        )

    # Step-weighted aggregate over every validly-judged step across files (judge failures are
    # counted in total_invalid, never as spurious zeros — see replay.valid_scores).
    step_scores = valid_scores(r for rep in per_file.values() for r in rep.results)
    overall = fmean(step_scores) if step_scores else 0.0
    overall_std = pstdev(step_scores) if len(step_scores) > 1 else 0.0
    return EvalReport(
        per_file=per_file,
        overall_fidelity=overall,
        overall_std=overall_std,
        total_steps=sum(rep.n_steps for rep in per_file.values()),
        total_invalid=sum(rep.n_invalid for rep in per_file.values()),
    )


def _display_name(path: Path) -> str:
    """Human label for a corpus, using the example folder name for `traces.otel.jsonl`."""
    name = path.name.removesuffix(".jsonl").removesuffix(".otel")
    return path.parent.name if name == "traces" else name


class OpenLoopEval:
    """The open-loop `Evaluation`: teacher-forced replay of held-out trace steps."""

    def __init__(
        self,
        files: list[Path],
        prompt: str,
        provider: Provider,
        judge: Judge,
        *,
        embedder: Embedder | None = None,
        train_split: float = 0.7,
        top_k: int = 5,
        sample_turns: str = "all",
        seed: int = 0,
        adapter_name: str = "otel-genai",
        knowledge: bool = False,
        reasoning: bool = False,
    ) -> None:
        self._files = files
        self._prompt = prompt
        self._provider = provider
        self._judge = judge
        self._embedder = embedder
        self._train_split = train_split
        self._top_k = top_k
        self._sample_turns = sample_turns
        self._seed = seed
        self._adapter_name = adapter_name
        self._knowledge = knowledge
        self._reasoning = reasoning

    def run(self) -> EvalReport:
        return evaluate_files(
            self._files,
            self._prompt,
            self._provider,
            self._judge,
            embedder=self._embedder,
            train_split=self._train_split,
            top_k=self._top_k,
            sample_turns=self._sample_turns,
            seed=self._seed,
            adapter_name=self._adapter_name,
            knowledge=self._knowledge,
            reasoning=self._reasoning,
        )
