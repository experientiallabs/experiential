"""Tests for the credential-free count of what a tagged mining run will buy."""

from __future__ import annotations

from collections.abc import Sequence

from exp.simulation.build_test import _trace
from exp.simulation.mining.descriptors import HashingDescriptorEmbedder
from exp.simulation.mining.features import DecisionTagSelectionSpec
from exp.simulation.mining.questions import EXAMPLE_BANK_ID, QuestionBank, resolve_question_bank
from exp.simulation.mining.service import MiningSpec, mine_tasks
from exp.simulation.mining.tagging import TagMatrix, TagRow
from exp.simulation.mining.views import Unit
from exp.simulation.mining_cost import tagging_work


def test_tagging_work_counts_the_distinct_cases_and_calls_nothing() -> None:
    """The tag route's price comes from the free prefix, so it is quotable before any call.

    Duplicate collapse decides how many decisions a corpus needs, because one distinct case is
    tagged once however many times it occurred. The two duplicated traces below therefore cost
    one call between them, and the count arrives with no tagger in sight.
    """
    repeated = _trace(2).model_copy(
        update={"trace_id": "trace-2-again", "conversation_id": "conversation-2-again"}
    )
    traces = (_trace(1), _trace(2), repeated, _trace(3))
    spec = MiningSpec(selection=DecisionTagSelectionSpec(bank_id=EXAMPLE_BANK_ID))

    work = tagging_work(traces, spec)

    assert work is not None
    assert work.candidates == 3
    assert work.questions == 4
    assert work.batches == 1
    assert work.calls == 3
    assert work.view_tokens > 0
    assert work.input_tokens == work.view_tokens + 3 * work.question_tokens
    assert tagging_work(traces, MiningSpec()) is None


class _RecordingTagger:
    """Answers every question with the same probability and records what it was asked about.

    Attributes:
        units: One entry per tagged unit, in the order the mining service supplied them.
    """

    def __init__(self) -> None:
        """Start with nothing recorded."""
        self.units: list[Unit] = []

    def tag(self, units: Sequence[Unit], bank: QuestionBank) -> TagMatrix:
        """Record the units and answer the whole bank about each of them.

        Args:
            units: Rendered views the mining service built for the selection candidates.
            bank: Questions to answer, in column order.

        Returns:
            One row per unit, in unit order.
        """
        self.units.extend(units)
        return TagMatrix(
            bank_id=bank.bank_id,
            bank_version=bank.version,
            question_ids=bank.question_ids(),
            view=units[0].view,
            rows=tuple(
                TagRow(
                    trace_id=unit.trace_id,
                    unit_hash=unit.unit_hash,
                    values=tuple(
                        0.1 + 0.8 * ((index + position) % 3) / 2
                        for position in range(len(bank.questions))
                    ),
                )
                for index, unit in enumerate(units)
            ),
        )


def test_the_count_is_what_the_mining_service_actually_tags() -> None:
    """The quote runs its own copy of mining's free prefix, so it has to agree with mining.

    Nothing else can check that: a quote that drifted from the pipeline it describes would still
    print, still be authorized, and then buy a different number of calls than it quoted. This
    pins the two together over a corpus with a duplicate, where the two counts differ unless both
    collapse it.
    """
    repeated = _trace(2).model_copy(
        update={"trace_id": "trace-2-again", "conversation_id": "conversation-2-again"}
    )
    traces = (_trace(1), _trace(2), repeated, _trace(3))
    spec = MiningSpec(
        fit_task_budget=2,
        held_out_task_budget=1,
        selection=DecisionTagSelectionSpec(bank_id=EXAMPLE_BANK_ID),
    )
    tagger = _RecordingTagger()

    work = tagging_work(traces, spec)
    mine_tasks(traces, spec, embedder=HashingDescriptorEmbedder(), tagger=tagger)

    assert work is not None
    assert work.candidates == len(tagger.units)
    assert work.questions == len(resolve_question_bank(EXAMPLE_BANK_ID).questions)
    assert work.view_tokens == sum(unit.tokens for unit in tagger.units)
