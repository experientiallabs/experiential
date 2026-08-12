"""Open-loop evaluation: reconstruction fidelity over trace files (the default `wmo eval` mode).

`replay` (in `wmo.simulation.model.replay`) scores one corpus of held-out steps teacher-forced. This
orchestration layer is what `wmo eval` calls: it loads one or more OTel trace files, splits each
into train/holdout, replays the holdout through a world-model prompt with leak-free RAG, and
aggregates a per-file + overall scorecard. Its closed-loop counterpart
(`wmo eval --mode closed-loop`, `wmo.simulation.evaluation.closed_loop`) runs a live agent instead
of replaying. Both implement the `Evaluation` interface in `wmo.simulation.evaluation.base`.
"""

from __future__ import annotations

from pathlib import Path
from statistics import fmean, pstdev

from pydantic import BaseModel, Field

from wmo.common.core.artifacts import JsonObject
from wmo.common.core.types import Action, ActionKind, Observation, Step, Trace
from wmo.common.judging.fidelity import FidelityJudge
from wmo.common.providers.base import Embedder, Provider
from wmo.common.traces import Trace as CanonicalTrace
from wmo.simulation.ingest.otlp import OtlpTraceFormatError, load_otlp_file
from wmo.simulation.model import DEFAULT_TRAIN_SPLIT, split_holdout
from wmo.simulation.model.knowledge import seeded_knowledge_text
from wmo.simulation.model.replay import ReplayReport, replay, valid_scores
from wmo.simulation.retrieval import EmbeddingRetriever


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
    judge: FidelityJudge,
    *,
    embedder: Embedder | None = None,
    train_split: float = DEFAULT_TRAIN_SPLIT,
    val_frac: float | None = None,
    top_k: int = 5,
    sample_turns: str = "all",
    seed: int = 0,
    max_holdout_traces: int | None = None,
    knowledge: bool = False,
    reasoning: bool = False,
) -> EvalReport:
    """Replay-score each trace file's held-out split. `embedder=None` -> zero-shot (no retrieval).

    Each file is split deterministically; tiny corpora with no held-out trace fall back to scoring
    every trace. `train_split` defaults to `DEFAULT_TRAIN_SPLIT`, the same cut `wmo build` makes on
    the same hash line: override it only to match a model that was built with a different ratio,
    because a smaller value here hands GEPA's training traces back as scored "held-out" steps.
    RAG, when enabled, retrieves from that file's own train split only (leak-free).
    `sample_turns`/`seed` are forwarded to `replay` (see its docstring). `max_holdout_traces` caps
    how many held-out traces are scored per file (a deterministic prefix by trace_id) - for cheap
    dry-runs; the train side stays full so retrieval is unaffected.

    `val_frac` makes the split leak-free against a GEPA-evolved prompt: when it is a positive
    fraction, the traces are cut 3-way (`train`/`val`/`test`) on the same hash line GEPA used, and
    only the reserved `test` band is scored, so a prompt selected on `val` is never graded on those
    same traces. Retrieval still draws from `train` only. `val_frac` of `None` or `0` keeps the
    plain 2-way `train`/held-out split (`split_traces_3way` requires a strictly positive band).

    `knowledge` seeds an ephemeral knowledge base from each file's TRAIN split (never the holdout -
    the same leak-free discipline as RAG) and renders it into every prediction; `reasoning`
    switches predictions to the deliberate-then-answer contract. Both mirror the serving engine's
    agentic mode. Closed-loop evals get agentic mode from the ARTIFACT instead (the served
    WorldModel's config / --max-fidelity winner), not from these flags.
    """
    per_file: dict[str, ReplayReport] = {}
    for path in files:
        traces = _load_replay_traces(path)
        if not traces:
            continue
        # 3-way when `val_frac` is a positive band, 2-way otherwise; a tiny corpus with no
        # held-out trace falls back to scoring everything (see `split_holdout`).
        train, holdout, _tiny_corpus = split_holdout(traces, train_split, val_frac)
        if max_holdout_traces is not None:
            holdout = sorted(holdout, key=lambda t: t.trace_id)[:max_holdout_traces]
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
        judge: FidelityJudge,
        *,
        embedder: Embedder | None = None,
        train_split: float = DEFAULT_TRAIN_SPLIT,
        top_k: int = 5,
        sample_turns: str = "all",
        seed: int = 0,
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
            knowledge=self._knowledge,
            reasoning=self._reasoning,
        )


def _load_replay_traces(path: Path) -> list[Trace]:
    """Read OTLP once and project its canonical evidence into legacy replay transitions.

    Replay is an older evaluation surface that still consumes in-memory ``common.core`` traces.
    It is intentionally fed only by the strict canonical OTLP loader, never by a generic source
    adapter or format detector.

    Args:
        path: Local OTLP JSON or JSONL evidence file.

    Returns:
        Legacy replay records derived from valid canonical traces in deterministic order.
    """
    try:
        normalized = load_otlp_file(path)
    except OtlpTraceFormatError as exc:
        if str(exc) == "OTLP JSONL file contains no records":
            return []
        raise ValueError(f"cannot normalize OTLP trace evidence {path}: {exc}") from exc
    return [_to_replay_trace(trace) for trace in normalized.traces]


def _to_replay_trace(canonical: CanonicalTrace) -> Trace:
    """Project one canonical trace into replay transitions without reopening source evidence."""
    pending = Action(kind=ActionKind.MESSAGE, content=canonical.task)
    steps: list[Step] = []
    for span in canonical.spans:
        operation = span.attributes.get("gen_ai.operation.name")
        if operation != "execute_tool":
            pending = _action_from_span(span.attributes, canonical.task)
            continue
        steps.append(
            Step(
                action=pending,
                observation=Observation(
                    content=_observation_from_span(span.attributes),
                    is_error=span.failure is not None,
                ),
                task=canonical.task,
                raw_span_ids=[span.span_id],
            )
        )
    if not steps:
        first = canonical.spans[0]
        steps.append(
            Step(
                action=_action_from_span(first.attributes, canonical.task),
                observation=Observation(
                    content=_observation_from_span(first.attributes),
                    is_error=first.failure is not None,
                ),
                task=canonical.task,
                raw_span_ids=[first.span_id],
            )
        )
    return Trace(
        trace_id=canonical.trace_id,
        steps=steps,
        source=canonical.source.identity.source_id,
        metadata={"canonical_trace_id": canonical.trace_id},
    )


def _action_from_span(attributes: JsonObject, task: str) -> Action:
    """Map visible canonical tool attributes to one legacy replay action."""
    name = attributes.get("gen_ai.tool.name")
    if isinstance(name, str) and name:
        return Action(kind=ActionKind.TOOL_CALL, name=name)
    return Action(kind=ActionKind.MESSAGE, content=task)


def _observation_from_span(attributes: JsonObject) -> str:
    """Choose the recorded output text available to the legacy replay scorer."""
    for key in (
        "gen_ai.tool.message",
        "gen_ai.tool.result",
        "gen_ai.completion",
        "gen_ai.response.text",
        "error.message",
    ):
        value = attributes.get(key)
        if isinstance(value, str):
            return value
    return ""
