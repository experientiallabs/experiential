"""The batch engine: submit validation, reservation, polling, and settlement.

One engine instance owns job lifecycle over host-installed seams. Submit is
fail-closed: every line must name a batch-callable model explicitly, one
provider serves the whole job, and every accepted line reserves its estimated
cost before the provider sees anything. Settlement is idempotent per line and
survives process restarts because all state lives behind the host's stores.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.batch.contracts import (
    BATCH_SURFACES,
    COMPLETION_WINDOW_SECONDS,
    TERMINAL_STATUSES,
    BatchCounts,
    BatchDeployment,
    BatchFile,
    BatchJob,
    BatchJobPage,
    BatchLine,
    BatchLineError,
    BatchLineResult,
    BatchStatus,
    BatchSubmitError,
    BatchSurface,
    parse_input_jsonl,
)
from exp.runtime.gateway.batch.interfaces import (
    BatchCatalog,
    BatchFileStore,
    BatchLedger,
    BatchSecretResolver,
    BatchStore,
)
from exp.runtime.gateway.batch.providers import (
    PROVIDER_CLIENTS,
    AmbiguousProviderResponse,
    ProviderBatchClient,
)

_LOGGER = logging.getLogger(__name__)

_DEFAULT_POLL_INTERVAL_SECONDS = 30.0
_LIST_LIMIT_MAXIMUM = 100
# The output-ceiling field of each batchable surface, in the order the
# reservation reads them: Chat Completions (legacy and current names), then
# Responses. The first positive integer present is the line's ceiling.
_OUTPUT_CEILING_KEYS = ("max_tokens", "max_completion_tokens", "max_output_tokens")


def _now() -> datetime:
    """Return the current timezone-aware wall-clock time."""
    return datetime.now(UTC)


def _new_id(prefix: str) -> str:
    """Mint one URL-safe public identifier with the given prefix."""
    return f"{prefix}_{secrets.token_hex(12)}"


def _approximate_tokens(body: JsonObject) -> int:
    """Approximate one line's input tokens as serialized bytes over four."""
    return max(1, len(json.dumps(body).encode("utf-8")) // 4)


class BatchEngine:
    """Job lifecycle over host seams; the only writer of batch state.

    Exactly one poller may run per job store. The dispatch-intent guard makes
    an interrupted dispatch fail closed, but it cannot arbitrate two live
    pollers racing the same job: a host that runs multiple workers must lease
    the poller role (or partition jobs) so one engine advances a given job at
    a time. Settlement re-runs are absorbed by the host ledger's contractual
    idempotency, and every public read is owner-scoped.
    """

    def __init__(
        self,
        *,
        store: BatchStore,
        files: BatchFileStore,
        catalog: BatchCatalog,
        ledger: BatchLedger,
        secrets_resolver: BatchSecretResolver,
        clients: dict[str, ProviderBatchClient] | None = None,
        poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
    ) -> None:
        """Bind the host seams and the provider client registry."""
        self._store = store
        self._files = files
        self._catalog = catalog
        self._ledger = ledger
        self._secrets = secrets_resolver
        self._clients = clients if clients is not None else PROVIDER_CLIENTS
        self._poll_interval = max(1.0, poll_interval_seconds)

    def is_batch_model(self, *, model: str) -> bool:
        """Return whether the catalog resolves one explicit batch model."""
        return self._catalog.batch_deployment(model=model) is not None

    def upload_file(
        self, *, organization_id: str, filename: str, purpose: str, content: bytes
    ) -> BatchFile:
        """Store one batch input file after validating its JSONL shape.

        Raises:
            BatchSubmitError: On a non-batch purpose or invalid JSONL content.
        """
        if purpose != "batch":
            raise BatchSubmitError("files uploaded to this gateway must use purpose 'batch'")
        parse_input_jsonl(content)
        record = BatchFile(
            file_id=_new_id("file"),
            organization_id=organization_id,
            filename=filename or "batch.jsonl",
            purpose="batch",
            size_bytes=len(content),
            created_at=_now(),
        )
        self._files.store(file=record, content=content)
        return record

    def file_metadata(self, *, organization_id: str, file_id: str) -> BatchFile | None:
        """Return one owned file's metadata, or None when absent."""
        return self._files.load_metadata(file_id=file_id, organization_id=organization_id)

    def file_content(self, *, organization_id: str, file_id: str) -> bytes | None:
        """Return one owned file's raw content, or None when absent."""
        return self._files.load_content(file_id=file_id, organization_id=organization_id)

    def submit(
        self,
        *,
        organization_id: str,
        identity_id: str,
        input_file_id: str,
        endpoint: str,
        metadata: dict[str, str] | None = None,
    ) -> BatchJob:
        """Validate, reserve, and persist one job ready for provider dispatch.

        The provider submit itself happens on the poller's next pass, so a
        caller-visible job always exists before any provider state does.

        Raises:
            BatchSubmitError: On an unknown surface, missing file, invalid
                lines, mixed providers, or a rejected reservation.
        """
        if endpoint not in BATCH_SURFACES:
            raise BatchSubmitError(
                f"endpoint {endpoint!r} is not batchable; use one of {', '.join(BATCH_SURFACES)}"
            )
        surface = next(candidate for candidate in BATCH_SURFACES if candidate == endpoint)
        content = self._files.load_content(file_id=input_file_id, organization_id=organization_id)
        if content is None:
            raise BatchSubmitError(f"input file {input_file_id!r} does not exist")
        raw_lines = parse_input_jsonl(content)
        lines, line_errors, binding = self._validate_lines(raw_lines, surface)
        if not lines or binding is None:
            raise BatchSubmitError(
                "every input line was rejected: "
                + "; ".join(error.message for error in line_errors[:5])
            )
        created = _now()
        job = BatchJob(
            batch_id=_new_id("batch"),
            organization_id=organization_id,
            identity_id=identity_id,
            surface=surface,
            provider=binding.provider,
            credential_reference=binding.credential_reference,
            input_file_id=input_file_id,
            counts=BatchCounts(total=len(lines)),
            lines=tuple(lines),
            line_errors=tuple(line_errors),
            metadata=dict(metadata or {}),
            created_at=created,
            expires_at=created + timedelta(seconds=COMPLETION_WINDOW_SECONDS),
        )
        job = self._reserve(job)
        self._store.create_job(job=job)
        return job

    def _validate_lines(
        self,
        raw_lines: list[tuple[int, JsonObject]],
        surface: BatchSurface,
    ) -> tuple[list[BatchLine], list[BatchLineError], BatchDeployment | None]:
        """Validate every raw line against the explicit-request contract.

        Returns:
            The accepted lines, the per-line rejections, and the deployment
            binding of the single provider that serves the job, captured from
            the same catalog resolution the validation used so a concurrent
            catalog edit can never split validation from binding.

        Raises:
            BatchSubmitError: When accepted lines span providers, repeat a
                custom id, or violate a provider's uniform-model requirement.
        """
        lines: list[BatchLine] = []
        errors: list[BatchLineError] = []
        seen_ids: set[str] = set()
        providers: set[str] = set()
        binding: BatchDeployment | None = None
        for line_number, raw in raw_lines:
            custom_id = raw.get("custom_id")
            body = raw.get("body")
            url = raw.get("url", surface)
            if not isinstance(custom_id, str) or not custom_id:
                errors.append(
                    BatchLineError(
                        line_number=line_number,
                        code="missing_custom_id",
                        message=f"line {line_number} carries no custom_id",
                    )
                )
                continue
            if custom_id in seen_ids:
                raise BatchSubmitError(f"custom_id {custom_id!r} appears more than once")
            seen_ids.add(custom_id)
            if url != surface:
                errors.append(
                    BatchLineError(
                        line_number=line_number,
                        custom_id=custom_id,
                        code="surface_mismatch",
                        message=f"line url {url!r} differs from the batch endpoint {surface}",
                    )
                )
                continue
            if not isinstance(body, dict):
                errors.append(
                    BatchLineError(
                        line_number=line_number,
                        custom_id=custom_id,
                        code="missing_body",
                        message=f"line {line_number} carries no request body object",
                    )
                )
                continue
            model = body.get("model")
            if not isinstance(model, str) or not model:
                errors.append(
                    BatchLineError(
                        line_number=line_number,
                        custom_id=custom_id,
                        code="missing_model",
                        message=f"line {line_number} names no model",
                    )
                )
                continue
            deployment = self._catalog.batch_deployment(model=model)
            if deployment is None:
                errors.append(
                    BatchLineError(
                        line_number=line_number,
                        custom_id=custom_id,
                        code="not_batch_callable",
                        message=(
                            f"model {model!r} is not batch-callable; batch jobs accept only "
                            "explicit batch models, and batch models accept only batch jobs"
                        ),
                    )
                )
                continue
            if surface not in deployment.surfaces:
                errors.append(
                    BatchLineError(
                        line_number=line_number,
                        custom_id=custom_id,
                        code="surface_unsupported",
                        message=f"model {model!r} does not serve {surface} in batch",
                    )
                )
                continue
            client = self._clients.get(deployment.provider)
            if client is None:
                raise BatchSubmitError(
                    f"provider {deployment.provider} has no batch client enabled"
                )
            # The catalog says the model serves this surface; the client is
            # the engine's truth of what its provider wire can carry.
            if surface not in client.surfaces:
                errors.append(
                    BatchLineError(
                        line_number=line_number,
                        custom_id=custom_id,
                        code="surface_unsupported",
                        message=(
                            f"{deployment.provider} batches do not serve {surface}; "
                            f"this client serves {', '.join(client.surfaces)}"
                        ),
                    )
                )
                continue
            maximum_output = deployment.default_maximum_output_tokens
            for key in _OUTPUT_CEILING_KEYS:
                value = body.get(key)
                if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                    maximum_output = value
                    break
            clean_body = {key: value for key, value in body.items() if key != "model"}
            line = BatchLine(
                custom_id=custom_id,
                surface=surface,
                model=model,
                provider_model=deployment.provider_model,
                body=clean_body,
                estimated_input_tokens=_approximate_tokens(clean_body),
                maximum_output_tokens=maximum_output,
            )
            # Shaping the line now catches every body the engine itself can
            # refuse for this wire, as a per-line 400 at submit rather than a
            # whole failed batch at dispatch; the provider's own per-model
            # admission still applies when the batch runs.
            try:
                client.line_request(line)
            except BatchSubmitError as rejection:
                errors.append(
                    BatchLineError(
                        line_number=line_number,
                        custom_id=custom_id,
                        code="invalid_request",
                        message=rejection.message,
                    )
                )
                continue
            # Only a line that survived every per-line check joins the job's
            # provider set and binding, so a rejected line on another
            # provider stays a per-line error instead of turning the valid
            # remainder into a mixed-provider refusal.
            providers.add(deployment.provider)
            if binding is None:
                binding = deployment
            elif deployment.credential_reference != binding.credential_reference:
                errors.append(
                    BatchLineError(
                        line_number=line_number,
                        custom_id=custom_id,
                        code="connection_mismatch",
                        message=(
                            f"model {model!r} is served by a different provider connection "
                            "than this batch; split lines by connection"
                        ),
                    )
                )
                continue
            lines.append(line)
        if len(providers) > 1:
            raise BatchSubmitError(
                "one batch is served by exactly one provider; split lines by provider "
                f"(saw {', '.join(sorted(providers))})"
            )
        if binding is not None and self._clients[binding.provider].requires_uniform_model:
            models = {line.provider_model for line in lines}
            if len(models) > 1:
                raise BatchSubmitError(
                    f"provider {binding.provider} requires one model per batch; "
                    f"saw {', '.join(sorted(models))}"
                )
        return lines, errors, binding

    def _reserve(self, job: BatchJob) -> BatchJob:
        """Reserve every line's estimated cost, releasing all on any rejection.

        Raises:
            BatchSubmitError: When the host rejects any line's reservation.
        """
        reserved: list[BatchLine] = []
        total = 0
        for line in job.lines:
            try:
                amount = self._ledger.reserve_line(job=job, line=line)
            except Exception as exc:
                for done in reserved:
                    self._ledger.release_line(job=job, line=done, reason="submit_rejected")
                raise BatchSubmitError(
                    f"reservation rejected at line {line.custom_id!r}: {exc}",
                    code="insufficient_quota",
                ) from exc
            reserved.append(line.model_copy(update={"reserved_micro_usd": amount}))
            total += amount
        return job.model_copy(update={"lines": tuple(reserved), "reserved_micro_usd": total})

    def retrieve(self, *, organization_id: str, batch_id: str) -> BatchJob | None:
        """Return one owned job, or None when absent."""
        return self._store.load_job(batch_id=batch_id, organization_id=organization_id)

    def list_jobs(
        self, *, organization_id: str, limit: int = 20, after: str | None = None
    ) -> BatchJobPage:
        """Return one page of the organization's jobs, newest first.

        The store is asked for one job beyond the page, so ``has_more`` is
        the truth of whether a further page exists rather than a guess from
        a full page; the extra job is never rendered.
        """
        bounded = max(1, min(limit, _LIST_LIMIT_MAXIMUM))
        fetched = self._store.list_jobs(
            organization_id=organization_id, limit=bounded + 1, after=after
        )
        return BatchJobPage(jobs=tuple(fetched[:bounded]), has_more=len(fetched) > bounded)

    async def cancel(self, *, organization_id: str, batch_id: str) -> BatchJob:
        """Request cancellation of one owned, non-terminal job.

        Raises:
            BatchSubmitError: When the job is unknown, terminal, or served by
                a provider without cancellation.
        """
        job = self._store.load_job(batch_id=batch_id, organization_id=organization_id)
        if job is None:
            raise BatchSubmitError(f"batch {batch_id!r} does not exist", code="not_found")
        if job.status in TERMINAL_STATUSES:
            return job
        client = self._clients[job.provider]
        if job.provider_batch_id is not None:
            if not client.supports_cancel:
                # An honest refusal beats a silent no-op: the caller learns
                # the provider limitation and the job state stays untouched.
                raise BatchSubmitError(
                    f"{job.provider} batches cannot be cancelled; the job runs to completion",
                    code="cancel_unsupported",
                )
            # The intent persists before the provider call, so a failed call
            # cannot lose it: the poller re-requests cancellation until the
            # provider confirms a terminal state.
            job = job.model_copy(update={"status": BatchStatus.CANCELLING})
            self._store.save_job(job=job)
            await client.cancel(job=job, api_key=self._api_key(job))
        elif self._store.begin_dispatch(batch_id=job.batch_id):
            # Winning the one-time dispatch claim proves no submission ever
            # ran or will run, so the lines release safely right now.
            for line in job.lines:
                self._ledger.release_line(job=job, line=line, reason="cancelled")
            job = job.model_copy(
                update={
                    "status": BatchStatus.CANCELLED,
                    "dispatch_started": True,
                    "finalized_at": _now(),
                    "settled": True,
                }
            )
        else:
            # A dispatch claim exists: a submission may be in flight or was
            # interrupted. Mark the intent and let the poller resolve it
            # without releasing money that may already be spending.
            job = job.model_copy(update={"status": BatchStatus.CANCELLING})
        self._store.save_job(job=job)
        return job

    def _api_key(self, job: BatchJob) -> str:
        """Resolve the job's submit-time credential reference at call time.

        The reference is frozen on the job at submit, so catalog edits while
        the job runs can never repoint an open job at a different connection;
        only the secret value itself is late-bound through the resolver.
        """
        return self._secrets.resolve(job.credential_reference)

    def _cancel_requested(self, job: BatchJob) -> bool:
        """Whether the caller's cancellation intent is on record for this job.

        The poller works from a job read before its provider call, so a
        cancel persisted while that call was in flight is visible only by
        re-reading the store; a snapshot already marked CANCELLING needs no
        read.
        """
        if job.status is BatchStatus.CANCELLING:
            return True
        current = self._store.load_job(batch_id=job.batch_id, organization_id=job.organization_id)
        return current is not None and current.status is BatchStatus.CANCELLING

    async def poll_once(self) -> int:
        """Advance every open job one step; returns how many jobs progressed."""
        advanced = 0
        for job in self._store.open_jobs():
            try:
                await self._advance(job)
                advanced += 1
            except Exception:
                _LOGGER.exception("batch %s poll failed; will retry", job.batch_id)
        return advanced

    async def run_poller(self, *, stop: asyncio.Event | None = None) -> None:
        """Poll open jobs forever, until the optional stop event is set.

        Run exactly one poller per job store; see the class contract.
        """
        while stop is None or not stop.is_set():
            await self.poll_once()
            if stop is None:
                await asyncio.sleep(self._poll_interval)
            else:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=self._poll_interval)
                except TimeoutError:
                    continue

    async def _advance(self, job: BatchJob) -> None:
        """Move one open job forward: submit, poll, settle, or expire."""
        if job.status in TERMINAL_STATUSES:
            if not job.settled:
                await self._settle(job)
            return
        if _now() >= job.expires_at:
            await self._finalize(job, BatchStatus.EXPIRED, "completion window elapsed")
            return
        client = self._clients[job.provider]
        api_key = self._api_key(job)
        if job.provider_batch_id is None:
            if job.status is BatchStatus.CANCELLING:
                # Cancellation won the dispatch race: nothing was or will be
                # submitted, so the job terminalizes without provider work.
                await self._finalize(job, BatchStatus.CANCELLED, None)
                return
            if not self._store.begin_dispatch(batch_id=job.batch_id):
                # Another actor holds or held the one-time dispatch claim.
                # This snapshot is stale by definition, so re-load before
                # concluding anything: the claim holder may have already
                # resolved the job (a cancel that won the claim, a dispatch
                # that persisted its id, or a completed settlement).
                current = self._store.load_job(
                    batch_id=job.batch_id, organization_id=job.organization_id
                )
                if (
                    current is None
                    or current.settled
                    or current.provider_batch_id is not None
                    or current.status in TERMINAL_STATUSES
                    or current.status is BatchStatus.CANCELLING
                ):
                    return
                # The persisted job still shows a claim without a provider
                # id: that dispatch was interrupted, and submitting again
                # would duplicate paid provider work against an unknown
                # provider-side batch, so the job fails closed.
                await self._finalize(
                    current,
                    BatchStatus.FAILED,
                    "dispatch was interrupted before the provider batch id "
                    "was persisted; resubmit the input file as a new batch",
                )
                return
            try:
                provider_batch_id = await client.submit(
                    job=job.model_copy(update={"dispatch_started": True}), api_key=api_key
                )
            except BatchSubmitError as rejection:
                # A raised BatchSubmitError means a provider response was
                # received, so nothing was accepted: fail now with the reason.
                # Ambiguous responses and transport losses raise other errors
                # and take the fail-closed interrupted path on the next poll.
                _LOGGER.warning(
                    "batch %s: %s rejected the submission: %s",
                    job.batch_id,
                    job.provider,
                    rejection.message,
                )
                await self._finalize(
                    job.model_copy(update={"dispatch_started": True}),
                    BatchStatus.FAILED,
                    f"provider rejected the batch submission: {rejection.message}",
                )
                return
            job = job.model_copy(
                update={
                    "dispatch_started": True,
                    "provider_batch_id": provider_batch_id,
                    "status": BatchStatus.IN_PROGRESS,
                }
            )
            if self._cancel_requested(job):
                # Cancellation arrived while the submit was in flight: keep
                # the provider id and the CANCELLING intent. Providers with
                # cancellation get the request; without it the job runs to
                # its provider-terminal state and settles normally, which is
                # the strongest cancellation the provider offers.
                job = job.model_copy(update={"status": BatchStatus.CANCELLING})
                self._store.save_job(job=job)
                if client.supports_cancel:
                    await client.cancel(job=job, api_key=api_key)
                return
            self._store.save_job(job=job)
            return
        if (
            job.status is BatchStatus.CANCELLING
            and job.provider_batch_id is not None
            and client.supports_cancel
        ):
            try:
                await client.cancel(job=job, api_key=api_key)
            except (BatchSubmitError, AmbiguousProviderResponse):
                _LOGGER.warning(
                    "batch %s: provider cancellation re-request failed; will retry",
                    job.batch_id,
                )
        snapshot = await client.poll(job=job, api_key=api_key)
        counts = job.counts.model_copy(
            update={"completed": snapshot.completed, "failed": snapshot.failed}
        )
        # The caller may have persisted a cancellation between this poll's
        # job read and the provider's answer; the intent is re-read after the
        # poll so a terminal snapshot never overwrites CANCELLING with
        # COMPLETED for a batch the caller cancelled.
        next_status = snapshot.status
        if self._cancel_requested(job):
            if next_status is BatchStatus.COMPLETED and snapshot.cancelled_lines > 0:
                # A provider that ends a cancelled batch as "completed"
                # reports the cut lines; with the caller's intent on record
                # that job is CANCELLED. When every line had already run,
                # nothing was cancelled and the job completes, every line
                # billed.
                next_status = BatchStatus.CANCELLED
            elif next_status not in TERMINAL_STATUSES:
                # The caller's cancellation intent survives provider
                # snapshots that have not yet observed it (or providers
                # without cancel).
                next_status = BatchStatus.CANCELLING
        job = job.model_copy(update={"status": next_status, "counts": counts})
        if next_status in TERMINAL_STATUSES:
            await self._finalize(job, next_status, snapshot.failure_message)
        else:
            self._store.save_job(job=job)

    async def _finalize(
        self, job: BatchJob, status: BatchStatus, failure_message: str | None
    ) -> None:
        """Persist a terminal status and settle every line exactly once."""
        job = job.model_copy(
            update={
                "status": status,
                "failure_message": failure_message,
                "finalized_at": _now(),
            }
        )
        self._store.save_job(job=job)
        await self._settle(job)

    async def _settle(self, job: BatchJob) -> None:
        """Settle results per line idempotently, then mark the job settled."""
        current = self._store.load_job(batch_id=job.batch_id, organization_id=job.organization_id)
        if current is not None and current.settled:
            return
        if job.settled:
            return
        results: dict[str, BatchLineResult] = {}
        if job.provider_batch_id is not None:
            # Cancelled, expired, and failed provider batches can still carry
            # billable partial results; settle whatever the provider reports
            # and release only the lines that produced nothing.
            client = self._clients[job.provider]
            try:
                fetched = await client.results(job=job, api_key=self._api_key(job))
            except BatchSubmitError:
                if job.status is BatchStatus.COMPLETED or job.counts.completed > 0:
                    # Results exist: a completed job, or a cancelled, expired,
                    # or failed one whose provider counted served lines. A
                    # fetch failure is retryable, so settlement stays open
                    # for a later poll instead of releasing lines whose work
                    # already ran.
                    raise
                # A definitive provider response on a failed, expired, or
                # cancelled job that served nothing means no per-line results
                # are available.
                _LOGGER.warning(
                    "batch %s: no results retrievable for %s job; releasing all lines",
                    job.batch_id,
                    job.status.value,
                )
                fetched = []
            results = {result.custom_id: result for result in fetched}
        output_lines: list[str] = []
        error_lines: list[str] = []
        completed = 0
        failed = 0
        for index, line in enumerate(job.lines):
            result = results.get(line.custom_id)
            if result is None:
                self._ledger.release_line(job=job, line=line, reason=job.status.value)
                failed += 1
                error_lines.append(
                    json.dumps(
                        BatchLineResult(
                            custom_id=line.custom_id,
                            status_code=500,
                            error={
                                "code": job.status.value,
                                "message": job.failure_message
                                or f"the batch ended {job.status.value} before this line ran",
                            },
                        ).output_jsonl_object(line_id=f"{job.batch_id}_line_{index}")
                    )
                )
                continue
            self._ledger.settle_line(job=job, line=line, result=result)
            rendered = json.dumps(
                result.output_jsonl_object(line_id=f"{job.batch_id}_line_{index}")
            )
            if result.error is None:
                completed += 1
                output_lines.append(rendered)
            else:
                failed += 1
                error_lines.append(rendered)
        output_file_id: str | None = None
        error_file_id: str | None = None
        created = _now()
        if output_lines:
            output_file_id = _new_id("file")
            self._files.store(
                file=BatchFile(
                    file_id=output_file_id,
                    organization_id=job.organization_id,
                    filename=f"{job.batch_id}_output.jsonl",
                    purpose="batch_output",
                    size_bytes=len("\n".join(output_lines).encode("utf-8")),
                    created_at=created,
                ),
                content="\n".join(output_lines).encode("utf-8"),
            )
        if error_lines:
            error_file_id = _new_id("file")
            self._files.store(
                file=BatchFile(
                    file_id=error_file_id,
                    organization_id=job.organization_id,
                    filename=f"{job.batch_id}_errors.jsonl",
                    purpose="batch_output",
                    size_bytes=len("\n".join(error_lines).encode("utf-8")),
                    created_at=created,
                ),
                content="\n".join(error_lines).encode("utf-8"),
            )
        self._store.save_job(
            job=job.model_copy(
                update={
                    "counts": job.counts.model_copy(
                        update={"completed": completed, "failed": failed}
                    ),
                    "output_file_id": output_file_id,
                    "error_file_id": error_file_id,
                    "settled": True,
                }
            )
        )


PollerFactory = Callable[[BatchEngine], asyncio.Task[None]]
