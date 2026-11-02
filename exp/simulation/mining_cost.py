"""What a mining run will buy from a provider, counted before it buys any of it.

A tagged mining run is paid work: it asks a decision model about every distinct case it selects.
Nothing may ask for that spend after it has happened, so the count has to come from the part of
mining that is free. This module runs exactly that part, which is the prefix of
:func:`~exp.simulation.mining.service.mine_tasks` up to duplicate collapse, and reports what the
tagging that follows it would cost in calls and input tokens. It reads no credential, constructs
no client and makes no call, so a command can quote the price and take consent for it first.

It lives beside the build rather than inside the mining package because it is the build's
question, not the mining service's: mining is handed a spec and runs it, while a caller that has
to authorize the spend needs the same numbers before mining starts. Turning those counts into
dollars belongs to the command that owns the catalog and the budget, so it is not here either.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil

from exp.common.traces import Trace
from exp.runtime.gateway.decisions_contracts import MAX_DECISION_QUESTIONS
from exp.runtime.gateway.reservation_tokenizer import count_reservation_tokens
from exp.simulation.mining.descriptors import DescriptorEmbedder
from exp.simulation.mining.features import DecisionTagSelectionSpec
from exp.simulation.mining.questions import resolve_question_bank
from exp.simulation.mining.service import MiningSpec, selection_candidates
from exp.simulation.mining.views import build_units


@dataclass(frozen=True)
class TaggingWork:
    """The decision work a tagged mining run will buy, counted without buying any of it.

    A caller prices this to ask for spend consent before the first call, so every number here
    comes from the free part of the pipeline: normalization, request descriptors, duplicate
    collapse, the bank file, and the rendered views. None of it asks a model anything.

    Attributes:
        bank_id: Identity of the bank whose questions will be asked.
        candidates: Distinct cases left after duplicate collapse. Each one is tagged once, which
            is why a corpus pays per distinct request rather than per occurrence.
        questions: Questions the selected bank families declare, which is the width of a row.
        batches: Requests one case takes. The decisions surface carries at most
            ``MAX_DECISION_QUESTIONS`` questions per request, and the rendered view is re-sent
            with every one of them.
        view_tokens: Rendered-view input tokens summed over every case, counted once each.
        question_tokens: Question-text input tokens over one whole pass of the bank.
    """

    bank_id: str
    candidates: int
    questions: int
    batches: int
    view_tokens: int
    question_tokens: int

    @property
    def calls(self) -> int:
        """Decision requests one complete pass dispatches when nothing is cached."""
        return self.candidates * self.batches

    @property
    def input_tokens(self) -> int:
        """Input tokens that pass sends, with each case's view repeated in each of its batches."""
        return self.batches * self.view_tokens + self.candidates * self.question_tokens


def tagging_work(
    traces: Sequence[Trace],
    mining_spec: MiningSpec,
    *,
    embedder: DescriptorEmbedder | None = None,
) -> TaggingWork | None:
    """Count the decision work one mining spec will buy over one corpus, calling nothing.

    This runs the free prefix of :func:`~exp.simulation.mining.service.mine_tasks`: lineages,
    request descriptors, their vectors and duplicate collapse. Those phases decide how many
    distinct cases exist, and a distinct case is exactly one unit of tagging, so the count is
    the real one rather than an upper bound over every trace. The prefix needs no credential and
    makes no call, which is what lets a command quote the price before asking to spend it.

    Args:
        traces: Valid canonical traces the build will mine, in build order.
        mining_spec: The mining controls the build will run.
        embedder: The request-descriptor embedder the build will use. The deterministic local
            hashing embedder is used when omitted, exactly as the build does, because duplicate
            collapse reads these vectors and a different embedder collapses differently.

    Returns:
        What tagging will ask of the decision model, or None when the geometry asks it nothing.

    Raises:
        QuestionBankError: The named bank is absent or declares no selected family.
        ValueError: The traces are empty or inconsistent with duplicate analysis.
    """
    selection = mining_spec.selection
    if not isinstance(selection, DecisionTagSelectionSpec):
        return None
    analysis = selection_candidates(traces, mining_spec, embedder=embedder).analysis
    bank = resolve_question_bank(selection.bank_id).families(*selection.families)
    by_trace_id = {trace.trace_id: trace for trace in traces}
    view_tokens = sum(
        build_units(
            by_trace_id[candidate.representative_trace_id],
            (selection.view,),
            selection.view_spec,
        )[selection.view].tokens
        for candidate in analysis.candidates
    )
    return TaggingWork(
        bank_id=bank.bank_id,
        candidates=len(analysis.candidates),
        questions=len(bank.questions),
        batches=ceil(len(bank.questions) / MAX_DECISION_QUESTIONS),
        view_tokens=view_tokens,
        question_tokens=sum(count_reservation_tokens(item.text) for item in bank.questions),
    )
