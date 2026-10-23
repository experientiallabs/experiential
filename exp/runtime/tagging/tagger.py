"""Ask a tag bank's questions about captured exchanges through a SystemOne decision endpoint.

A ``Tagger`` renders an exchange into its conversation opening window and its turn windows,
sends one decision request per window with that scope's tags as ``noul`` questions, and returns
every probability in bank scope order as ``ExchangeTags``. A window whose decision fails fails
the whole exchange with a ``TaggingError``; no score is ever filled in. ``tag_many`` streams a
batch with a bounded number of decisions in flight, asks each conversation opening once, and
turns any failure confined to one exchange into that exchange's ``TaggingError``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Collection, Iterable, Sequence
from datetime import UTC, datetime

from exp.common.core.artifacts import ContractModel
from exp.common.tags.bank import DEFAULT_TAG_BANK, TagBank, TagDefinition, TagScope
from exp.common.tags.exchange import TagExchange, opening_key, tag_facts
from exp.common.tags.records import ExchangeTags, WindowScores
from exp.common.tags.windows import TagWindow, TagWindowBudget, opening_window, turn_windows
from exp.runtime.gateway.decisions_contracts import DecisionUsage, NoulCriteria, NoulQuestion
from exp.runtime.models.providers.systemone import (
    SystemOneClient,
    SystemOneError,
    SystemOneResponseError,
)

# Longest served model name an ExchangeTags record holds.
_MAX_MODEL_CHARS = 256


class TaggingError(Exception):
    """One exchange was not tagged: a window decision failed, or preparing or tagging it raised.

    Attributes:
        exchange_id: The exchange that was not tagged.
        scope: Scope of the failed window, or None when no window decision failed.
        window_index: Index of the failed window within its scope, or None likewise.
        cause: The failure, also chained as ``__cause__``.
    """

    def __init__(self, exchange_id: str, window: TagWindow | None, cause: Exception) -> None:
        """Describe the failure and keep its cause.

        Args:
            exchange_id: The exchange that was not tagged.
            window: The window whose decision failed, or None for a failure outside one
                window's decision, such as a window that could not be built.
            cause: The failure.
        """
        where = (
            f"the decision for its {window.scope.value} window {window.index} failed"
            if window is not None
            else f"{type(cause).__name__} while preparing or tagging it"
        )
        super().__init__(f"exchange {exchange_id} was not tagged: {where} ({cause})")
        self.__cause__ = cause
        self.exchange_id = exchange_id
        self.scope = None if window is None else window.scope
        self.window_index = None if window is None else window.index
        self.cause = cause


class WindowDecision(ContractModel):
    """Scores for one window plus what the endpoint reported while deciding it.

    Attributes:
        scores: The window's probabilities in bank scope order.
        served_model: Model identity the endpoint reported.
        usage: Tokens the endpoint reported for the request.
    """

    scores: WindowScores
    served_model: str
    usage: DecisionUsage


def conversation_key(exchange: TagExchange) -> str:
    """Return the exchange's conversation key, or its ``opening_key`` when it carries none."""
    return exchange.conversation_key or opening_key(exchange)


def _question(tag: TagDefinition) -> NoulQuestion:
    """Phrase one tag as a ``noul`` question: its wording plus true and false criteria."""
    return NoulQuestion(
        instructions=tag.question, criteria=NoulCriteria(true=tag.true_when, false=tag.false_when)
    )


class Tagger:
    """Tag exchanges with one bank through one SystemOne client.

    Example:
        ``tagger = Tagger(SystemOneClient(url, "clef", max_concurrency=32), max_in_flight=32)``
        then ``tags = await tagger.tag(exchange, opening_already_tagged=False)``.

    At most ``max_in_flight`` window decisions run at once across every call on this tagger.
    """

    def __init__(
        self,
        client: SystemOneClient,
        *,
        bank: TagBank = DEFAULT_TAG_BANK,
        budget: TagWindowBudget | None = None,
        max_in_flight: int,
    ) -> None:
        """Bind a client, bank and window budget without network work.

        Args:
            client: Decision client; its ``max_concurrency`` must allow ``max_in_flight``.
            bank: Tags to ask; both scopes need at least one tag.
            budget: Window size limits; the default budget when None. Size it for the decision
                server's input limit with ``TagWindowBudget.for_input_tokens``.
            max_in_flight: Most window decisions in flight at once; ``sweep_concurrency``
                measures the best value for an endpoint.

        Raises:
            ValueError: ``max_in_flight`` is outside ``1..client.max_concurrency`` or a bank
                scope has no tags.
        """
        if not 1 <= max_in_flight <= client.max_concurrency:
            raise ValueError(
                f"max_in_flight is {max_in_flight} but must be between 1 and the client's "
                f"max_concurrency ({client.max_concurrency}); build the SystemOneClient with "
                "max_concurrency at least as large"
            )
        for scope in TagScope:
            if not bank.for_scope(scope):
                raise ValueError(
                    f"bank {bank.bank_id} v{bank.version} has no {scope.value} tags; a tagger "
                    "asks both scopes, so add at least one tag to each"
                )
        self._client = client
        self._bank = bank
        self._digest = bank.digest
        self._budget = budget or TagWindowBudget()
        self._max_in_flight = max_in_flight
        self._questions = {
            scope: {tag.id: _question(tag) for tag in bank.for_scope(scope)} for scope in TagScope
        }
        self._slots = asyncio.Semaphore(max_in_flight)

    @property
    def client(self) -> SystemOneClient:
        """Return the decision client."""
        return self._client

    @property
    def bank(self) -> TagBank:
        """Return the bank whose tags this tagger asks."""
        return self._bank

    @property
    def budget(self) -> TagWindowBudget:
        """Return the window size limits."""
        return self._budget

    @property
    def max_in_flight(self) -> int:
        """Return the bound on concurrent window decisions."""
        return self._max_in_flight

    def windows(
        self, exchange: TagExchange, *, opening_already_tagged: bool
    ) -> tuple[TagWindow, ...]:
        """Render the windows ``tag`` sends for one exchange.

        Args:
            exchange: The exchange.
            opening_already_tagged: Leave out the conversation opening window.

        Returns:
            The opening window (unless left out) followed by the turn windows.
        """
        turns = turn_windows(exchange, self._budget)
        if opening_already_tagged:
            return turns
        return (opening_window(exchange, self._budget), *turns)

    async def decide_window(self, window: TagWindow) -> WindowDecision:
        """Ask one window's scope tags in a single decision request.

        Args:
            window: A rendered window.

        Returns:
            The window's scores in bank scope order with the served model and usage.

        Raises:
            SystemOneError: The decision failed; see the client for the typed cases.
        """
        questions = self._questions[window.scope]
        async with self._slots:
            result = await self._client.decide(window.text, questions)
        return WindowDecision(
            scores=WindowScores(
                window_index=window.index,
                start_message=window.start_message,
                end_message=window.end_message,
                scores=tuple(result.noul[tag_id] for tag_id in questions),
            ),
            served_model=result.served_model,
            usage=result.usage,
        )

    async def tag(self, exchange: TagExchange, *, opening_already_tagged: bool) -> ExchangeTags:
        """Tag one exchange: its opening (unless already tagged) and every turn window.

        Args:
            exchange: The exchange.
            opening_already_tagged: Skip the conversation opening because another exchange of
                the same conversation already answered it; the record's ``conversation`` is
                then None.

        Returns:
            The exchange's scores, facts and bank and model stamps.

        Raises:
            TaggingError: A window decision failed, or windows were answered by different or
                unrecordable models. Nothing is returned for a partly tagged exchange.
        """
        windows = self.windows(exchange, opening_already_tagged=opening_already_tagged)
        decisions = await self._decide_all(exchange.exchange_id, windows)
        model = decisions[0].served_model
        for window, decision in zip(windows, decisions, strict=True):
            if decision.served_model != model or len(model) > _MAX_MODEL_CHARS:
                cause = SystemOneResponseError(
                    f"windows were answered by {model!r} and {decision.served_model!r}; a "
                    f"record names one served model of at most {_MAX_MODEL_CHARS} characters"
                )
                raise TaggingError(exchange.exchange_id, window, cause) from cause
        return ExchangeTags(
            exchange_id=exchange.exchange_id,
            conversation_key=conversation_key(exchange),
            bank_id=self._bank.bank_id,
            bank_version=self._bank.version,
            bank_digest=self._digest,
            model=model,
            tagged_at=datetime.now(UTC),
            conversation=None if opening_already_tagged else decisions[0].scores,
            turn=tuple(d.scores for d in decisions[0 if opening_already_tagged else 1 :]),
            facts=tag_facts(exchange),
        )

    async def _decide_all(
        self, exchange_id: str, windows: Sequence[TagWindow]
    ) -> list[WindowDecision]:
        """Decide windows concurrently, stopping the rest as soon as one fails.

        Args:
            exchange_id: The exchange the windows belong to.
            windows: Windows in order.

        Returns:
            One decision per window, in window order.

        Raises:
            TaggingError: A window's decision failed; the windows still running are cancelled.
        """
        tasks = [asyncio.create_task(self.decide_window(window)) for window in windows]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        decisions: list[WindowDecision] = []
        for window, task in zip(windows, tasks, strict=True):
            if task.cancelled():
                continue
            error = task.exception()
            if isinstance(error, SystemOneError):
                raise TaggingError(exchange_id, window, error) from error
            if error is not None:
                raise error
            decisions.append(task.result())
        return decisions

    async def tag_many(
        self,
        exchanges: Iterable[TagExchange],
        *,
        opening_tagged_keys: Collection[str] = frozenset(),
    ) -> AsyncGenerator[ExchangeTags | TaggingError, None]:
        """Tag a stream of exchanges, yielding each result as soon as it is ready.

        Exchanges are read lazily; at most ``max_in_flight`` are being tagged or held at once.
        The first exchange of each conversation key (``conversation_key``) not in
        ``opening_tagged_keys`` also tags the opening. Later exchanges with that key are held
        until it finishes, then skip the opening; if it failed, the first held exchange asks
        the opening instead, so a conversation keeps its opening whenever any of its exchanges
        is tagged.

        Args:
            exchanges: Exchanges to tag, in order.
            opening_tagged_keys: Conversation keys whose opening is already tagged, for example
                from an earlier run.

        Yields:
            ``ExchangeTags`` for each tagged exchange and ``TaggingError`` for each exchange that
            was not tagged, in completion order. Any ``Exception`` raised while preparing or
            tagging one exchange (its opening key, its windows, a decision) fails only that
            exchange, so one unexpected input never stops the batch.

        Raises:
            Exception: A failure outside any one exchange, such as reading the exchanges.
                ``KeyboardInterrupt``, ``SystemExit`` and cancellation always propagate.
        """
        tagged = set(opening_tagged_keys)
        # Keys whose opening is being asked, with the exchanges held until it finishes.
        held: dict[str, list[TagExchange]] = {}
        pending: dict[asyncio.Task[ExchangeTags], tuple[str, str | None]] = {}
        source = iter(exchanges)

        def start(exchange: TagExchange, key: str) -> None:
            """Hold the exchange behind an opening in flight, or start tagging it."""
            if key in held:
                held[key].append(exchange)
                return
            claimed = key not in tagged
            if claimed:
                tagged.add(key)
                held[key] = []
            task = asyncio.create_task(self.tag(exchange, opening_already_tagged=not claimed))
            pending[task] = (exchange.exchange_id, key if claimed else None)

        try:
            while True:
                while len(pending) + sum(map(len, held.values())) < self._max_in_flight:
                    exchange = next(source, None)
                    if exchange is None:
                        break
                    try:
                        key = conversation_key(exchange)
                    except Exception as error:  # noqa: BLE001 - fails only this exchange
                        yield TaggingError(exchange.exchange_id, None, error)
                        continue
                    start(exchange, key)
                if not pending:
                    return
                done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in [task for task in pending if task in done]:
                    exchange_id, claimed_key = pending.pop(task)
                    error = task.exception()
                    if error is not None and not isinstance(error, TaggingError):
                        if not isinstance(error, Exception):
                            raise error
                        error = TaggingError(exchange_id, None, error)
                    if claimed_key is not None:
                        if error is not None:
                            tagged.discard(claimed_key)
                        for exchange in held.pop(claimed_key):
                            start(exchange, claimed_key)
                    yield error if isinstance(error, TaggingError) else task.result()
        finally:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)


__all__ = ["Tagger", "TaggingError", "WindowDecision", "conversation_key"]
