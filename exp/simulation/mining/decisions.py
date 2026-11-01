"""Tags bought from a decision model over its own typed wire.

A decision model answers propositions. It does not necessarily hold a conversation, and the one
configured here does not: its chat-completions route refuses with an unsupported-capability error,
so the portable chat contract is not available and the typed decisions surface is the only way to
reach it. That surface is also the better fit. A proposition is sent as a question object and comes
back as a probability, with no answer format to agree on and nothing to parse out of prose, which
removes the entire class of failure where a reply drifts out of alignment with the questions.

What arrives is a genuine probability rather than a self-reported confidence digit. The model
returns values strictly inside the open interval, so the all-zero row that a normalizing transform
cannot place does not occur on this wire, and a transform may use the resolution instead of
rounding it away.

The endpoint is never named here. A tagger receives the resolved alias's own wire facts and an
injected transport, so pointing tagging at a different deployment means resolving a different alias,
not editing this module. Answers are bought through the shared decision cache, so a rerun over one
corpus pays only for the units it has not tagged and the questions whose text changed.

A pass keeps a bounded number of those calls in flight at once. One call answers one episode and
spends seconds waiting on the endpoint, so a corpus of thousands of episodes is hours of almost
pure waiting when the calls are made one at a time. Concurrency is the one knob that changes that,
and it changes nothing else: the matrix, the row order and what a pass pays are the same at every
value, which is why it is an argument here rather than part of any persisted spec.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from exp.common.core.artifacts import JsonObject
from exp.common.models import ModelSnapshot
from exp.runtime.gateway.decisions_contracts import (
    MAX_DECISION_QUESTIONS,
    DecisionRequest,
    NoulQuestion,
)
from exp.runtime.models.providers.transport import (
    HttpxJsonTransport,
    JsonHttpTransport,
    ProviderTransportError,
    RetryPolicy,
    run_with_retry,
)
from exp.runtime.models.registry import ResolvedModel
from exp.simulation.mining.dispatch import DEFAULT_TAGGING_CONCURRENCY, dispatch_tagging
from exp.simulation.mining.questions import Question, QuestionBank
from exp.simulation.mining.tagging import (
    DECISION_RECORDS_NAMESPACE,
    DecisionCache,
    TagMatrix,
    TagRow,
    TraceTaggingError,
    decision_key,
    distinct_units,
    question_chunks,
    tagged_view,
)
from exp.simulation.mining.views import Unit

logger = logging.getLogger(__name__)

NATIVE_DECISION_DIALECT = "typesafe_systemone"
"""The one wire dialect that serves typed decisions."""

DEFAULT_DECISION_TIMEOUT_SECONDS = 300.0
"""Per-attempt ceiling for one decision call.

A rendered episode runs to tens of thousands of tokens and a call answers a whole question bank
about it, so this is generous on purpose: a timeout here costs the call twice, once in money and
once in the retained intent a future run has to resolve by hand.
"""


@dataclass(frozen=True)
class _UnitAnswers:
    """One unit's answers to a whole bank, with what this run paid for them.

    Attributes:
        values: One probability per bank question, in column order.
        bought: Calls this run dispatched for this unit, which is zero for a fully replayed unit.
        spend: USD this run paid for this unit, taken only from the calls it dispatched.
        tokens: Input tokens this run paid for, on the same basis as ``spend``.
    """

    values: tuple[float, ...]
    bought: int
    spend: float
    tokens: int


class DecisionsTraceTagger:
    """Buys tags from a decision model over the typed decisions surface.

    Attributes:
        model_identity: The model's JSON identity, part of every cache key, so answers from one
            model are never replayed as another's.
        cache: Project-scoped decision cache, or None to buy every call in this process.
        batch_limit: Largest number of questions asked in one call.
        maximum_concurrency: Calls in flight at once. It changes only how long a pass takes: the
            matrix, the rows, their order and what the pass pays are all the same at any value,
            which is why it is a constructor argument rather than part of any persisted spec.
    """

    def __init__(
        self,
        decide: Callable[[DecisionRequest], JsonObject],
        model: ModelSnapshot,
        *,
        cache: DecisionCache | None = None,
        batch_limit: int = MAX_DECISION_QUESTIONS,
        maximum_concurrency: int = DEFAULT_TAGGING_CONCURRENCY,
    ) -> None:
        """Bind one decision boundary to the model identity behind it.

        Args:
            decide: Provider boundary taking one typed request and returning its response body.
                Injected rather than built, so a test drives the tagger with no transport and a
                caller can route calls through its own accounting. Above one call in flight it is
                called from several threads at once, so an injected boundary has to tolerate that.
            model: Resolved identity of the model that boundary reaches.
            cache: Project-scoped decision cache. Without one, nothing is saved and a rerun buys
                every call again.
            batch_limit: Questions per call, capped by the surface's own per-request limit.
            maximum_concurrency: Calls in flight at once. One is a strictly serial pass.

        Raises:
            TraceTaggingError: The batch limit is outside what one request may carry, or the
                concurrency allowance is not positive.
        """
        if not 1 <= batch_limit <= MAX_DECISION_QUESTIONS:
            raise TraceTaggingError(
                f"one decision request carries 1 through {MAX_DECISION_QUESTIONS} questions, "
                f"so a batch limit of {batch_limit} cannot be dispatched"
            )
        if maximum_concurrency < 1:
            raise TraceTaggingError(
                f"tagging needs at least one call in flight, so a maximum concurrency of "
                f"{maximum_concurrency} would answer nothing"
            )
        self.model_identity: JsonObject = model.model_dump(mode="json")
        self.cache = cache
        self.batch_limit = batch_limit
        self.maximum_concurrency = maximum_concurrency
        self._decide = decide
        self._model_id = model.model_id

    @classmethod
    def for_alias(
        cls,
        resolved: ResolvedModel,
        *,
        transport: JsonHttpTransport | None = None,
        cache: DecisionCache | None = None,
        timeout_seconds: float = DEFAULT_DECISION_TIMEOUT_SECONDS,
        retry: RetryPolicy | None = None,
        batch_limit: int = MAX_DECISION_QUESTIONS,
        maximum_concurrency: int = DEFAULT_TAGGING_CONCURRENCY,
    ) -> DecisionsTraceTagger:
        """Bind a catalog alias, taking its endpoint and credential from its own wire facts.

        Args:
            resolved: Catalog alias resolved to its client, snapshot and capabilities.
            transport: JSON transport for the call. The shared HTTP transport is used when
                omitted. One transport serves every call in flight, so an injected one has to be
                safe to post from several threads at once.
            cache: Project-scoped decision cache.
            timeout_seconds: Per-attempt ceiling.
            retry: Bounded same-endpoint retry policy. The shared default is used when omitted.
            batch_limit: Questions per call.
            maximum_concurrency: Calls in flight at once.

        Returns:
            A tagger that calls that alias's typed decisions endpoint.

        Raises:
            TraceTaggingError: The alias does not resolve to a native decision wire, which is what
                happens when it names a chat-only connection.
        """
        profile_of = getattr(resolved.client, "gateway_wire_profile", None)
        if profile_of is None:
            raise TraceTaggingError(
                f"alias {resolved.alias!r} does not resolve to a native decision client; "
                "configure it on a typesafe connection to tag with a decision model"
            )
        profile = profile_of()
        if profile.dialect != NATIVE_DECISION_DIALECT or not profile.decisions_url:
            raise TraceTaggingError(
                f"alias {resolved.alias!r} speaks {profile.dialect!r} and exposes no typed "
                "decision endpoint; tagging needs an alias on a typesafe connection"
            )
        sender = transport or HttpxJsonTransport()
        policy = retry or RetryPolicy()
        decisions_url = profile.decisions_url

        def decide(request: DecisionRequest) -> JsonObject:
            """Send one decision, retrying only what the shared classifier calls retryable."""

            def attempt() -> JsonObject:
                """Make one physical call and require an object response."""
                response = sender.post(
                    decisions_url,
                    headers=profile.headers,
                    payload=request.provider_body(profile.model_id),
                    timeout_seconds=timeout_seconds,
                )
                if response.status_code != 200 or not isinstance(response.body, dict):
                    raise ProviderTransportError(
                        f"tagging decision returned HTTP {response.status_code}",
                        status_code=response.status_code,
                    )
                return response.body

            return run_with_retry(attempt, policy=policy)

        return cls(
            decide,
            resolved.snapshot,
            cache=cache,
            batch_limit=batch_limit,
            maximum_concurrency=maximum_concurrency,
        )

    def tag(self, units: Sequence[Unit], bank: QuestionBank) -> TagMatrix:
        """Answer the whole bank about every unit, replaying what this project already bought.

        Units are decided at the configured concurrency, which is the difference between hours
        and minutes on a large corpus: a call is almost entirely time spent waiting on the
        endpoint. Nothing about the result depends on it. Rows follow the supplied units, every
        unit's answers are assembled from its own calls, and the totals below are added up in
        unit order on this thread, so a pass at any concurrency returns the same matrix and
        reports the same spend.

        Args:
            units: Rendered views of one view, in any order.
            bank: Questions to ask, in column order.

        Returns:
            One row per distinct trace, in the order the units were supplied.

        Raises:
            TraceTaggingError: The units render more than one view, an answer is missing or is not
                a probability, or a retained intent blocks a repurchase. When several units fail,
                the first of them in unit order is the one raised, on every run.
        """
        view = tagged_view(units)
        chunks = question_chunks(bank.questions, self.batch_limit)
        decided = distinct_units(units)

        def answer(unit: Unit) -> _UnitAnswers:
            """Answer the whole bank about one unit, on whichever thread runs it."""
            return self._unit_answers(unit, chunks)

        results = dispatch_tagging(decided, answer, maximum_concurrency=self.maximum_concurrency)
        answered = {
            unit.unit_hash: result.values for unit, result in zip(decided, results, strict=True)
        }
        spend = 0.0
        tokens = 0
        bought = 0
        for result in results:
            bought += result.bought
            spend += result.spend
            tokens += result.tokens
        rows = tuple(
            TagRow(
                trace_id=unit.trace_id,
                unit_hash=unit.unit_hash,
                values=answered[unit.unit_hash],
            )
            for unit in _first_per_trace(units)
        )
        logger.info(
            "tagged %d traces with %d questions over %d decisions on model %s; "
            "bought %d of them (%d input tokens, %.4f USD) and replayed the rest",
            len(rows),
            len(bank.questions),
            len(chunks) * len(answered),
            self._model_id,
            bought,
            tokens,
            spend,
        )
        return TagMatrix(
            bank_id=bank.bank_id,
            bank_version=bank.version,
            question_ids=bank.question_ids(),
            view=view,
            rows=rows,
        )

    def _unit_answers(self, unit: Unit, chunks: Sequence[Sequence[Question]]) -> _UnitAnswers:
        """Answer every batch about one unit and report what this run paid for them.

        Each worker returns its own totals instead of adding into shared counters, so the
        accounting needs no lock and lands in unit order rather than completion order.

        Args:
            unit: Rendered view to ask about.
            chunks: Question batches covering the bank, in column order.

        Returns:
            This unit's answers in column order, with the calls, tokens and spend this run paid.

        Raises:
            TraceTaggingError: An answer is missing or is not a probability, or a retained intent
                blocks a repurchase.
        """
        values: list[float] = []
        spend = 0.0
        tokens = 0
        bought = 0
        for chunk in chunks:
            body, dispatched = self._body(unit, chunk)
            values.extend(parse_decision_answers(body, chunk, unit.trace_id))
            usage = body.get("usage")
            if dispatched and isinstance(usage, dict):
                bought += 1
                spend += _number(usage.get("cost")) or 0.0
                tokens += int(_number(usage.get("input_tokens")) or 0)
        return _UnitAnswers(values=tuple(values), bought=bought, spend=spend, tokens=tokens)

    def _body(self, unit: Unit, chunk: Sequence[Question]) -> tuple[JsonObject, bool]:
        """Return one batch's response and whether this run paid for it.

        A replayed answer carries the usage of the call that originally bought it, so reporting
        spend without distinguishing the two would bill every rerun again in the log.

        Args:
            unit: Rendered view to ask about.
            chunk: Questions asked in this request.

        Returns:
            The response body, and True when a provider call was actually dispatched.
        """
        request = DecisionRequest(
            state=unit.text,
            questions={
                question.question_id: NoulQuestion(instructions=question.text) for question in chunk
            },
        )
        if self.cache is None:
            return self._decide(request), True
        dispatched = False

        def dispatch() -> JsonObject:
            """Buy one decision and record that this run, not an earlier one, paid for it."""
            nonlocal dispatched
            dispatched = True
            return self._decide(request)

        key = decision_key(unit, chunk, self.model_identity)
        return self.cache.purchase(key, request, dispatch), dispatched


def parse_decision_answers(
    body: Mapping[str, object], chunk: Sequence[Question], trace_id: str
) -> tuple[float, ...]:
    """Read one probability per asked question out of a typed decision response.

    Reading is strict. A missing answer, an answer of another question type and an answer that is
    not a probability all raise, because a dropped answer and a confident no are different
    evidence: coercing the first into the second puts a fabricated column into the geometry, where
    it is indistinguishable from a real one and silently moves which tasks get selected.

    Args:
        body: Decoded response body.
        chunk: The exact questions asked in this request, in order.
        trace_id: Trace the questions were asked about, named in a failure.

    Returns:
        One probability per question, in the order asked.

    Raises:
        TraceTaggingError: The body carries no answer object, or an answer is absent, of the wrong
            type, or not a finite probability.
    """
    answers = body.get("answers")
    if not isinstance(answers, Mapping):
        raise TraceTaggingError(
            f"the decision response about trace {trace_id} carries no answers object; "
            f"remove its saved record from this project's {DECISION_RECORDS_NAMESPACE} records "
            "before retrying"
        )
    values: list[float] = []
    for question in chunk:
        answer = answers.get(question.question_id)
        if not isinstance(answer, Mapping):
            raise TraceTaggingError(
                f"the decision response about trace {trace_id} has no answer for question "
                f"{question.question_id}; a missing answer is not a confident no, so it cannot be "
                "read as zero. Remove the saved record from this project's "
                f"{DECISION_RECORDS_NAMESPACE} records and ask again"
            )
        if answer.get("type") != "noul":
            raise TraceTaggingError(
                f"question {question.question_id} about trace {trace_id} was asked as a "
                f"proposition but answered as {answer.get('type')!r}"
            )
        value = _number(answer.get("noul"))
        if value is None or not 0.0 <= value <= 1.0:
            raise TraceTaggingError(
                f"the answer to {question.question_id} about trace {trace_id} is "
                f"{answer.get('noul')!r}, which is not a probability in [0, 1]"
            )
        values.append(value)
    return tuple(values)


def _number(value: object) -> float | None:
    """Return a finite float for a real JSON number, and None for anything else.

    A bool is an int in Python, so it would otherwise pass as 0.0 or 1.0 and turn a malformed
    answer into a maximally confident one.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number == number and abs(number) != float("inf") else None


def _first_per_trace(units: Sequence[Unit]) -> tuple[Unit, ...]:
    """Return one unit per trace, in first-seen order, so rows follow input order."""
    seen: dict[str, Unit] = {}
    for unit in units:
        seen.setdefault(unit.trace_id, unit)
    return tuple(seen.values())
