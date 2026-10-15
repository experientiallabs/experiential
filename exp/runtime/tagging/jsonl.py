"""Tag a JSONL file of exchanges into a JSONL file of tag records, resumably.

The input holds one ``TagExchange`` per line. Each tagged exchange is appended to the output as
one ``ExchangeTags`` line and each failed exchange to the failures file as one ``TagFailure``
line, flushed as soon as it is ready. Running again with the same output skips exchanges the
output already holds and retries everything else, including earlier failures; the failures file
is an append-only log of every failed attempt. A last line an interrupted write left without its
newline is cut off before anything is appended, so its exchange is tagged again.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import aclosing
from pathlib import Path
from typing import TextIO

from pydantic import Field, ValidationError

from exp.common.core.artifacts import ContractModel
from exp.common.tags.bank import TagScope
from exp.common.tags.exchange import TagExchange
from exp.common.tags.records import ExchangeTags
from exp.runtime.tagging.tagger import Tagger, TaggingError

logger = logging.getLogger(__name__)


class TagFailure(ContractModel):
    """One exchange a failed decision stopped, as written to the failures file.

    Attributes:
        exchange_id: The exchange that was not tagged.
        scope: Scope of the failed window.
        window_index: Index of the failed window within its scope.
        error: Class name of the decision error.
        message: What went wrong and what to do about it.
    """

    exchange_id: str = Field(min_length=1, max_length=512)
    scope: TagScope
    window_index: int = Field(ge=0)
    error: str
    message: str


class TagJsonlReport(ContractModel):
    """Counts from one ``tag_jsonl`` run.

    Attributes:
        tagged: Exchanges tagged and written to the output in this run.
        failed: Exchanges written to the failures file in this run.
        skipped: Input lines skipped because the output already held their exchange id or the
            id repeated an earlier input line.
    """

    tagged: int = Field(ge=0)
    failed: int = Field(ge=0)
    skipped: int = Field(ge=0)


def _invalid(path: Path, number: int, kind: str, error: ValidationError) -> ValueError:
    """Describe line ``number`` of ``path`` that is not a valid ``kind``, without its content."""
    first = error.errors(include_input=False, include_url=False)[0]
    where = ".".join(str(part) for part in first["loc"]) or "line"
    return ValueError(
        f"{path}:{number} is not a valid {kind} ({error.error_count()} errors, first at "
        f"{where}: {first['msg']}); fix or remove that line and run again"
    )


def _drop_unterminated_tail(path: Path) -> None:
    """Cut a last line an interrupted write left without its newline, so appends stay valid.

    Args:
        path: A JSONL file, which may not exist yet.
    """
    if not path.exists():
        return
    with path.open("r+b") as handle:
        end = cut = handle.seek(0, os.SEEK_END)
        while cut > 0:
            start = max(cut - 65_536, 0)
            handle.seek(start)
            newline = handle.read(cut - start).rfind(b"\n")
            if newline >= 0:
                cut = start + newline + 1
                break
            cut = start
        if cut != end:
            logger.warning("dropping an unterminated last line of %s", path)
            handle.truncate(cut)


def _completed(output: Path, digest: str) -> tuple[set[str], set[str]]:
    """Read the exchange ids and tagged openings an existing output already holds.

    Args:
        output: The output file, which may not exist yet.
        digest: Digest of the bank the tagger asks.

    Returns:
        Exchange ids in the output and conversation keys whose opening it holds.

    Raises:
        ValueError: A complete line is not an ``ExchangeTags`` record, or a record was tagged
            with a different bank.
    """
    ids: set[str] = set()
    opened: set[str] = set()
    if not output.exists():
        return ids, opened
    with output.open("rb") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = ExchangeTags.model_validate_json(line)
            except ValidationError as error:
                raise _invalid(output, number, "ExchangeTags record", error) from None
            if record.bank_digest != digest:
                raise ValueError(
                    f"{output}:{number} was tagged with bank {record.bank_id} "
                    f"v{record.bank_version} ({record.bank_digest}), not this tagger's bank; "
                    "write to a new output file"
                )
            ids.add(record.exchange_id)
            if record.conversation is not None:
                opened.add(record.conversation_key)
    return ids, opened


def _append(handle: TextIO, record: ContractModel) -> None:
    """Write one record as a JSON line and flush it."""
    handle.write(record.model_dump_json() + "\n")
    handle.flush()


async def tag_jsonl(
    tagger: Tagger, exchanges: Path, *, output: Path, failures: Path
) -> TagJsonlReport:
    """Tag every exchange of a JSONL file that the output does not hold yet.

    Args:
        tagger: The tagger to use.
        exchanges: Input file with one ``TagExchange`` per line; blank lines are ignored.
        output: File that ``ExchangeTags`` lines are appended to; created when missing.
        failures: File that ``TagFailure`` lines are appended to; created when missing.

    Returns:
        How many exchanges were tagged, failed and skipped.

    Raises:
        ValueError: An input or output line is invalid, or the output holds records from a
            different bank. Records written before the error stay in the output.
    """
    for path in (output, failures):
        _drop_unterminated_tail(path)
    done, opened = _completed(output, tagger.bank.digest)
    skipped = 0

    def pending() -> Iterator[TagExchange]:
        """Yield input exchanges that are not tagged yet, each id once."""
        nonlocal skipped
        with exchanges.open(encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    exchange = TagExchange.model_validate_json(line)
                except ValidationError as error:
                    raise _invalid(exchanges, number, "TagExchange", error) from None
                if exchange.exchange_id in done:
                    skipped += 1
                    continue
                done.add(exchange.exchange_id)
                yield exchange

    tagged = failed = 0
    with (
        output.open("a", encoding="utf-8") as results,
        failures.open("a", encoding="utf-8") as errors,
    ):
        stream = aclosing(tagger.tag_many(pending(), opening_tagged_keys=opened))
        async with stream as outcomes:
            async for outcome in outcomes:
                if isinstance(outcome, TaggingError):
                    logger.warning("%s", outcome)
                    failure = TagFailure(
                        exchange_id=outcome.exchange_id,
                        scope=outcome.scope,
                        window_index=outcome.window_index,
                        error=type(outcome.cause).__name__,
                        message=str(outcome),
                    )
                    _append(errors, failure)
                    failed += 1
                else:
                    _append(results, outcome)
                    tagged += 1
    return TagJsonlReport(tagged=tagged, failed=failed, skipped=skipped)


__all__ = ["TagFailure", "TagJsonlReport", "tag_jsonl"]
