"""Engine lifecycle tests over in-memory host seams and a scripted provider."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from exp.runtime.gateway.batch.contracts import (
    BatchDeployment,
    BatchFile,
    BatchJob,
    BatchLine,
    BatchLineResult,
    BatchStatus,
    BatchSubmitError,
)
from exp.runtime.gateway.batch.engine import BatchEngine
from exp.runtime.gateway.batch.providers import ProviderBatchSnapshot


class MemoryStore:
    """In-memory BatchStore."""

    def __init__(self) -> None:
        """Start empty."""
        self.jobs: dict[str, BatchJob] = {}

    def create_job(self, *, job: BatchJob) -> None:
        """Persist one new job."""
        self.jobs[job.batch_id] = job

    def load_job(self, *, batch_id: str, organization_id: str) -> BatchJob | None:
        """Return one owned job."""
        job = self.jobs.get(batch_id)
        return job if job is not None and job.organization_id == organization_id else None

    def save_job(self, *, job: BatchJob) -> None:
        """Overwrite one job."""
        self.jobs[job.batch_id] = job

    def list_jobs(self, *, organization_id: str, limit: int, after: str | None) -> list[BatchJob]:
        """Return owned jobs newest first."""
        owned = [job for job in self.jobs.values() if job.organization_id == organization_id]
        owned.sort(key=lambda job: job.created_at, reverse=True)
        return owned[:limit]

    def open_jobs(self) -> list[BatchJob]:
        """Return jobs that still need the poller."""
        return [job for job in self.jobs.values() if not job.settled]

    def begin_dispatch(self, *, batch_id: str) -> bool:
        """Claim the one-time dispatch: first caller wins."""
        job = self.jobs[batch_id]
        if job.dispatch_started:
            return False
        self.jobs[batch_id] = job.model_copy(update={"dispatch_started": True})
        return True


class MemoryFiles:
    """In-memory BatchFileStore."""

    def __init__(self) -> None:
        """Start empty."""
        self.records: dict[str, tuple[BatchFile, bytes]] = {}

    def store(self, *, file: BatchFile, content: bytes) -> None:
        """Persist one file."""
        self.records[file.file_id] = (file, content)

    def load_metadata(self, *, file_id: str, organization_id: str) -> BatchFile | None:
        """Return one owned file's metadata."""
        entry = self.records.get(file_id)
        if entry is None or entry[0].organization_id != organization_id:
            return None
        return entry[0]

    def load_content(self, *, file_id: str, organization_id: str) -> bytes | None:
        """Return one owned file's content."""
        entry = self.records.get(file_id)
        if entry is None or entry[0].organization_id != organization_id:
            return None
        return entry[1]


class MemoryCatalog:
    """BatchCatalog with two batch models on different providers."""

    def __init__(self) -> None:
        """Author the fixture deployments."""
        self.deployments = {
            "gpt-oss-120b-batch": BatchDeployment(
                model="gpt-oss-120b-batch",
                provider="openrouter",
                provider_model="openai/gpt-oss-120b:batch",
                credential_reference="secret://openrouter",
                surfaces=("/v1/chat/completions",),
                input_micro_usd_per_million_tokens=40_000,
                output_micro_usd_per_million_tokens=80_000,
            ),
            "kimi-k3-batch": BatchDeployment(
                model="kimi-k3-batch",
                provider="openai",
                provider_model="kimi-k3-batch",
                credential_reference="secret://openai",
                surfaces=("/v1/chat/completions", "/v1/responses"),
                input_micro_usd_per_million_tokens=100_000,
                output_micro_usd_per_million_tokens=200_000,
            ),
        }

    def batch_deployment(self, *, model: str) -> BatchDeployment | None:
        """Resolve one explicit batch model."""
        return self.deployments.get(model)


class MemoryLedger:
    """BatchLedger recording every verb; optionally rejecting reservations."""

    def __init__(self, *, reject_after: int | None = None) -> None:
        """Optionally reject the Nth reservation onward."""
        self.reserved: list[str] = []
        self.settled: list[tuple[str, int]] = []
        self.released: list[tuple[str, str]] = []
        self._reject_after = reject_after

    def reserve_line(self, *, job: BatchJob, line: BatchLine) -> int:
        """Reserve a deterministic estimate or reject when scripted to."""
        if self._reject_after is not None and len(self.reserved) >= self._reject_after:
            raise RuntimeError("insufficient credit")
        self.reserved.append(line.custom_id)
        return 1_000

    def settle_line(self, *, job: BatchJob, line: BatchLine, result: BatchLineResult) -> None:
        """Record one settlement."""
        self.settled.append((line.custom_id, result.output_tokens))

    def release_line(self, *, job: BatchJob, line: BatchLine, reason: str) -> None:
        """Record one release."""
        self.released.append((line.custom_id, reason))


class MemorySecrets:
    """BatchSecretResolver returning a fixed key per reference."""

    def resolve(self, reference: str) -> str:
        """Resolve deterministically."""
        return f"key-for-{reference}"


class ScriptedClient:
    """Provider client driven by a scripted status sequence."""

    provider = "openrouter"
    supports_cancel = False
    requires_uniform_model = True

    def __init__(
        self, snapshots: list[ProviderBatchSnapshot], results: list[BatchLineResult]
    ) -> None:
        """Bind the scripted poll snapshots and final results."""
        self._snapshots = snapshots
        self._results = results
        self.submitted: list[str] = []
        self.cancelled = 0

    async def submit(self, *, job: BatchJob, api_key: str) -> str:
        """Record the submit and mint a provider id."""
        self.submitted.append(api_key)
        return "prov_batch_1"

    async def poll(self, *, job: BatchJob, api_key: str) -> ProviderBatchSnapshot:
        """Pop the next scripted snapshot, holding the last one."""
        if len(self._snapshots) > 1:
            return self._snapshots.pop(0)
        return self._snapshots[0]

    async def results(self, *, job: BatchJob, api_key: str) -> list[BatchLineResult]:
        """Return the scripted results."""
        return list(self._results)

    async def cancel(self, *, job: BatchJob, api_key: str) -> None:
        """Record the cancellation request."""
        self.cancelled += 1


def _engine(
    *,
    ledger: MemoryLedger | None = None,
    client: ScriptedClient | None = None,
) -> tuple[BatchEngine, MemoryStore, MemoryFiles, MemoryLedger, ScriptedClient]:
    """Compose one engine over fresh in-memory seams."""
    store = MemoryStore()
    files = MemoryFiles()
    bound_ledger = ledger if ledger is not None else MemoryLedger()
    bound_client = (
        client
        if client is not None
        else ScriptedClient(
            [ProviderBatchSnapshot(status=BatchStatus.COMPLETED, results_ready=True)], []
        )
    )
    engine = BatchEngine(
        store=store,
        files=files,
        catalog=MemoryCatalog(),
        ledger=bound_ledger,
        secrets_resolver=MemorySecrets(),
        clients={"openrouter": bound_client, "openai": bound_client},
    )
    return engine, store, files, bound_ledger, bound_client


def _upload(engine: BatchEngine, lines: list[str]) -> str:
    """Upload one JSONL input built from raw line strings."""
    record = engine.upload_file(
        organization_id="org_a",
        filename="input.jsonl",
        purpose="batch",
        content="\n".join(lines).encode("utf-8"),
    )
    return record.file_id


def _chat_line(custom_id: str, model: str = "gpt-oss-120b-batch") -> str:
    """Render one valid chat batch line."""
    return (
        f'{{"custom_id": "{custom_id}", "method": "POST", "url": "/v1/chat/completions",'
        f' "body": {{"model": "{model}", "messages": [], "max_tokens": 16}}}}'
    )


def test_submit_accepts_valid_lines_and_reserves_each() -> None:
    """A valid two-line job persists validating with per-line reservations."""
    engine, store, _, ledger, _ = _engine()
    file_id = _upload(engine, [_chat_line("a"), _chat_line("b")])
    job = engine.submit(
        organization_id="org_a",
        identity_id="id_a",
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
    )
    assert job.status is BatchStatus.VALIDATING
    assert job.counts.total == 2
    assert job.reserved_micro_usd == 2_000
    assert ledger.reserved == ["a", "b"]
    assert store.jobs[job.batch_id].provider == "openrouter"


def test_submit_refuses_unknown_endpoint_and_missing_file() -> None:
    """Non-batchable surfaces and unknown files are whole-job refusals."""
    engine, _, _, _, _ = _engine()
    with pytest.raises(BatchSubmitError, match="not batchable"):
        engine.submit(
            organization_id="org_a",
            identity_id="id_a",
            input_file_id="file_x",
            endpoint="/v1/embeddings",
        )
    with pytest.raises(BatchSubmitError, match="does not exist"):
        engine.submit(
            organization_id="org_a",
            identity_id="id_a",
            input_file_id="file_x",
            endpoint="/v1/chat/completions",
        )


def test_submit_quarantines_sync_models_per_line() -> None:
    """A synchronous model name is a per-line explicit-request violation."""
    engine, _, _, _, _ = _engine()
    file_id = _upload(engine, [_chat_line("a"), _chat_line("b", model="gpt-oss-120b")])
    job = engine.submit(
        organization_id="org_a",
        identity_id="id_a",
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
    )
    assert job.counts.total == 1
    assert job.line_errors[0].code == "not_batch_callable"
    assert "explicit batch models" in job.line_errors[0].message


def test_submit_refuses_mixed_providers_and_duplicate_ids() -> None:
    """Cross-provider jobs and repeated custom ids refuse the whole job."""
    engine, _, _, _, _ = _engine()
    mixed = _upload(engine, [_chat_line("a"), _chat_line("b", model="kimi-k3-batch")])
    with pytest.raises(BatchSubmitError, match="exactly one provider"):
        engine.submit(
            organization_id="org_a",
            identity_id="id_a",
            input_file_id=mixed,
            endpoint="/v1/chat/completions",
        )
    duplicated = _upload(engine, [_chat_line("a"), _chat_line("a")])
    with pytest.raises(BatchSubmitError, match="more than once"):
        engine.submit(
            organization_id="org_a",
            identity_id="id_a",
            input_file_id=duplicated,
            endpoint="/v1/chat/completions",
        )


def test_submit_rolls_back_reservations_on_rejection() -> None:
    """A mid-job budget rejection releases every prior reservation."""
    engine, _, _, ledger, _ = _engine(ledger=MemoryLedger(reject_after=1))
    file_id = _upload(engine, [_chat_line("a"), _chat_line("b")])
    with pytest.raises(BatchSubmitError, match="reservation rejected"):
        engine.submit(
            organization_id="org_a",
            identity_id="id_a",
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
        )
    assert ledger.released == [("a", "submit_rejected")]


def test_poller_submits_polls_and_settles_idempotently() -> None:
    """The full lifecycle settles once per line and renders output files."""
    results = [
        BatchLineResult(
            custom_id="a",
            status_code=200,
            response={"usage": {"prompt_tokens": 3, "completion_tokens": 5}},
            input_tokens=3,
            output_tokens=5,
        )
    ]
    client = ScriptedClient(
        [
            ProviderBatchSnapshot(status=BatchStatus.IN_PROGRESS),
            ProviderBatchSnapshot(status=BatchStatus.COMPLETED, results_ready=True),
        ],
        results,
    )
    engine, store, files, ledger, _ = _engine(client=client)
    file_id = _upload(engine, [_chat_line("a"), _chat_line("b")])
    job = engine.submit(
        organization_id="org_a",
        identity_id="id_a",
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
    )
    asyncio.run(engine.poll_once())
    assert store.jobs[job.batch_id].provider_batch_id == "prov_batch_1"
    asyncio.run(engine.poll_once())
    asyncio.run(engine.poll_once())
    settled = store.jobs[job.batch_id]
    assert settled.status is BatchStatus.COMPLETED
    assert settled.settled is True
    assert ledger.settled == [("a", 5)]
    assert ledger.released == [("b", "completed")]
    assert settled.output_file_id is not None
    assert settled.error_file_id is not None
    output = files.load_content(file_id=settled.output_file_id, organization_id="org_a")
    assert output is not None and b'"custom_id": "a"' in output
    before = (len(ledger.settled), len(ledger.released))
    asyncio.run(engine.poll_once())
    assert (len(ledger.settled), len(ledger.released)) == before


def test_expiry_releases_every_line() -> None:
    """A job past its window expires and releases all reservations."""
    engine, store, _, ledger, _ = _engine()
    file_id = _upload(engine, [_chat_line("a")])
    job = engine.submit(
        organization_id="org_a",
        identity_id="id_a",
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
    )
    expired = store.jobs[job.batch_id].model_copy(
        update={"expires_at": datetime.now(UTC) - timedelta(seconds=1)}
    )
    store.save_job(job=expired)
    asyncio.run(engine.poll_once())
    final = store.jobs[job.batch_id]
    assert final.status is BatchStatus.EXPIRED
    assert ledger.released == [("a", "expired")]


def test_cancel_before_dispatch_terminalizes_and_releases() -> None:
    """Cancelling an unsubmitted job needs no provider and releases lines."""
    engine, store, _, ledger, client = _engine()
    file_id = _upload(engine, [_chat_line("a")])
    job = engine.submit(
        organization_id="org_a",
        identity_id="id_a",
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
    )
    cancelled = asyncio.run(engine.cancel(organization_id="org_a", batch_id=job.batch_id))
    assert cancelled.status is BatchStatus.CANCELLED
    assert ledger.released == [("a", "cancelled")]
    assert client.cancelled == 0
    assert store.jobs[job.batch_id].settled is True


def test_cancel_of_unknown_job_is_not_found() -> None:
    """An unknown batch id maps to the not_found code."""
    engine, _, _, _, _ = _engine()
    with pytest.raises(BatchSubmitError, match="does not exist"):
        asyncio.run(engine.cancel(organization_id="org_a", batch_id="batch_missing"))


def test_list_jobs_is_owner_scoped() -> None:
    """Another organization's listing never sees the job."""
    engine, _, _, _, _ = _engine()
    file_id = _upload(engine, [_chat_line("a")])
    engine.submit(
        organization_id="org_a",
        identity_id="id_a",
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
    )
    assert len(engine.list_jobs(organization_id="org_a")) == 1
    assert engine.list_jobs(organization_id="org_b") == []


def test_interrupted_dispatch_fails_closed_without_resubmitting() -> None:
    """A job with dispatch started but no provider id never submits again."""
    client = ScriptedClient(
        [ProviderBatchSnapshot(status=BatchStatus.COMPLETED, results_ready=True)], []
    )
    engine, store, _, ledger, _ = _engine(client=client)
    file_id = _upload(engine, [_chat_line("a")])
    job = engine.submit(
        organization_id="org_a",
        identity_id="id_a",
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
    )
    interrupted = store.jobs[job.batch_id].model_copy(update={"dispatch_started": True})
    store.save_job(job=interrupted)
    asyncio.run(engine.poll_once())
    final = store.jobs[job.batch_id]
    assert final.status is BatchStatus.FAILED
    assert final.failure_message is not None and "interrupted" in final.failure_message
    assert client.submitted == []
    assert ledger.released == [("a", "failed")]


def test_open_jobs_use_the_submit_time_credential_reference() -> None:
    """Repointing the catalog mid-job never changes the credential in use."""
    client = ScriptedClient([ProviderBatchSnapshot(status=BatchStatus.IN_PROGRESS)], [])
    catalog = MemoryCatalog()
    store = MemoryStore()
    engine = BatchEngine(
        store=store,
        files=MemoryFiles(),
        catalog=catalog,
        ledger=MemoryLedger(),
        secrets_resolver=MemorySecrets(),
        clients={"openrouter": client, "openai": client},
    )
    file_id = _upload(engine, [_chat_line("a")])
    job = engine.submit(
        organization_id="org_a",
        identity_id="id_a",
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
    )
    assert store.jobs[job.batch_id].credential_reference == "secret://openrouter"
    repointed = catalog.deployments["gpt-oss-120b-batch"].model_copy(
        update={"credential_reference": "secret://other-connection"}
    )
    catalog.deployments["gpt-oss-120b-batch"] = repointed
    asyncio.run(engine.poll_once())
    assert client.submitted == ["key-for-secret://openrouter"]


def test_definitive_submit_rejection_fails_immediately_with_the_reason() -> None:
    """A provider response rejecting the submit terminalizes the job at once."""

    class RejectingClient(ScriptedClient):
        """Client whose submit receives a definitive provider rejection."""

        async def submit(self, *, job: BatchJob, api_key: str) -> str:
            """Raise the response-backed rejection."""
            raise BatchSubmitError(
                "provider batch create failed with status 401", code="provider_error"
            )

    client = RejectingClient([ProviderBatchSnapshot(status=BatchStatus.IN_PROGRESS)], [])
    engine, store, _, ledger, _ = _engine(client=client)
    file_id = _upload(engine, [_chat_line("a")])
    job = engine.submit(
        organization_id="org_a",
        identity_id="id_a",
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
    )
    asyncio.run(engine.poll_once())
    final = store.jobs[job.batch_id]
    assert final.status is BatchStatus.FAILED
    assert final.failure_message is not None
    assert "provider rejected the batch submission" in final.failure_message
    assert "status 401" in final.failure_message
    assert ledger.released == [("a", "failed")]


def test_validation_and_binding_share_one_catalog_resolution() -> None:
    """The job binds the deployment captured during line validation."""

    class CountingCatalog(MemoryCatalog):
        """Catalog counting resolutions and repointing after the first."""

        def __init__(self) -> None:
            """Track lookups."""
            super().__init__()
            self.lookups = 0

        def batch_deployment(self, *, model: str) -> BatchDeployment | None:
            """Repoint the credential after the first resolution."""
            self.lookups += 1
            deployment = super().batch_deployment(model=model)
            if deployment is not None and self.lookups > 1:
                return deployment.model_copy(update={"credential_reference": "secret://repointed"})
            return deployment

    catalog = CountingCatalog()
    engine = BatchEngine(
        store=MemoryStore(),
        files=MemoryFiles(),
        catalog=catalog,
        ledger=MemoryLedger(),
        secrets_resolver=MemorySecrets(),
    )
    file_id = _upload(engine, [_chat_line("a")])
    job = engine.submit(
        organization_id="org_a",
        identity_id="id_a",
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
    )
    assert job.credential_reference == "secret://openrouter"
    assert catalog.lookups == 1


def test_same_provider_different_connections_split_per_line() -> None:
    """Lines on another connection of the same provider are rejected per line."""
    catalog = MemoryCatalog()
    catalog.deployments["gpt-oss-20b-batch"] = catalog.deployments["gpt-oss-120b-batch"].model_copy(
        update={
            "model": "gpt-oss-20b-batch",
            "credential_reference": "secret://openrouter-second-account",
        }
    )
    engine = BatchEngine(
        store=MemoryStore(),
        files=MemoryFiles(),
        catalog=catalog,
        ledger=MemoryLedger(),
        secrets_resolver=MemorySecrets(),
    )
    file_id = _upload(engine, [_chat_line("a"), _chat_line("b", model="gpt-oss-20b-batch")])
    job = engine.submit(
        organization_id="org_a",
        identity_id="id_a",
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
    )
    assert job.counts.total == 1
    assert job.line_errors[0].code == "connection_mismatch"


def test_ambiguous_submit_response_takes_the_fail_closed_path() -> None:
    """A 2xx submit response that cannot parse never counts as a rejection."""
    from exp.runtime.gateway.batch.providers import AmbiguousProviderResponse

    class AmbiguousClient(ScriptedClient):
        """Client whose submit response is unparseable."""

        async def submit(self, *, job: BatchJob, api_key: str) -> str:
            """Raise the ambiguous outcome."""
            raise AmbiguousProviderResponse("provider batch create returned invalid JSON")

    client = AmbiguousClient([ProviderBatchSnapshot(status=BatchStatus.IN_PROGRESS)], [])
    engine, store, _, ledger, _ = _engine(client=client)
    file_id = _upload(engine, [_chat_line("a")])
    job = engine.submit(
        organization_id="org_a",
        identity_id="id_a",
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
    )
    asyncio.run(engine.poll_once())
    assert store.jobs[job.batch_id].status is BatchStatus.VALIDATING
    asyncio.run(engine.poll_once())
    final = store.jobs[job.batch_id]
    assert final.status is BatchStatus.FAILED
    assert final.failure_message is not None and "interrupted" in final.failure_message
    assert ledger.released == [("a", "failed")]


def test_cancel_during_inflight_dispatch_keeps_reservations() -> None:
    """Losing the dispatch claim marks CANCELLING and releases nothing."""
    engine, store, _, ledger, client = _engine()
    file_id = _upload(engine, [_chat_line("a")])
    job = engine.submit(
        organization_id="org_a",
        identity_id="id_a",
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
    )
    assert store.begin_dispatch(batch_id=job.batch_id)
    cancelled = asyncio.run(engine.cancel(organization_id="org_a", batch_id=job.batch_id))
    assert cancelled.status is BatchStatus.CANCELLING
    assert ledger.released == []
    assert client.cancelled == 0


def test_terminal_jobs_settle_partial_provider_results() -> None:
    """A cancelled provider batch still settles the lines that ran."""
    partial = [
        BatchLineResult(
            custom_id="a",
            status_code=200,
            response={"usage": {"prompt_tokens": 1, "completion_tokens": 4}},
            input_tokens=1,
            output_tokens=4,
        )
    ]
    client = ScriptedClient(
        [
            ProviderBatchSnapshot(status=BatchStatus.IN_PROGRESS),
            ProviderBatchSnapshot(status=BatchStatus.CANCELLED),
        ],
        partial,
    )
    engine, store, _, ledger, _ = _engine(client=client)
    file_id = _upload(engine, [_chat_line("a"), _chat_line("b")])
    job = engine.submit(
        organization_id="org_a",
        identity_id="id_a",
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
    )
    asyncio.run(engine.poll_once())
    asyncio.run(engine.poll_once())
    asyncio.run(engine.poll_once())
    final = store.jobs[job.batch_id]
    assert final.status is BatchStatus.CANCELLED and final.settled
    assert ledger.settled == [("a", 4)]
    assert ledger.released == [("b", "cancelled")]


def test_inflight_cancel_without_provider_support_runs_to_terminal() -> None:
    """CANCELLING survives non-terminal polls and settles at provider end."""
    client = ScriptedClient(
        [
            ProviderBatchSnapshot(status=BatchStatus.IN_PROGRESS),
            ProviderBatchSnapshot(status=BatchStatus.COMPLETED, results_ready=True),
        ],
        [],
    )
    engine, store, _, ledger, _ = _engine(client=client)
    file_id = _upload(engine, [_chat_line("a")])
    job = engine.submit(
        organization_id="org_a",
        identity_id="id_a",
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
    )
    asyncio.run(engine.poll_once())
    store.save_job(
        job=store.jobs[job.batch_id].model_copy(update={"status": BatchStatus.CANCELLING})
    )
    asyncio.run(engine.poll_once())
    assert store.jobs[job.batch_id].status is BatchStatus.CANCELLING
    assert client.cancelled == 0
    asyncio.run(engine.poll_once())
    asyncio.run(engine.poll_once())
    final = store.jobs[job.batch_id]
    assert final.status is BatchStatus.COMPLETED and final.settled
    assert ledger.released == [("a", "completed")]


def test_interrupted_terminal_settlement_resumes_from_open_jobs() -> None:
    """A terminal job whose settlement never ran settles on a later poll."""
    engine, store, _, ledger, _ = _engine()
    file_id = _upload(engine, [_chat_line("a")])
    job = engine.submit(
        organization_id="org_a",
        identity_id="id_a",
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
    )
    store.save_job(
        job=store.jobs[job.batch_id].model_copy(
            update={"status": BatchStatus.FAILED, "failure_message": "crashed mid-finalize"}
        )
    )
    assert not store.jobs[job.batch_id].settled
    asyncio.run(engine.poll_once())
    final = store.jobs[job.batch_id]
    assert final.settled
    assert ledger.released == [("a", "failed")]


def test_completed_job_settlement_retries_on_fetch_failure() -> None:
    """A transient results-fetch error keeps a completed job unsettled."""
    from exp.runtime.gateway.batch.contracts import BatchSubmitError as SubmitError

    class FlakyResultsClient(ScriptedClient):
        """Client whose first results fetch fails, then succeeds."""

        def __init__(self) -> None:
            """Script one completed snapshot and one flaky fetch."""
            super().__init__(
                [ProviderBatchSnapshot(status=BatchStatus.COMPLETED, results_ready=True)],
                [
                    BatchLineResult(
                        custom_id="a",
                        status_code=200,
                        response={"usage": {"prompt_tokens": 1, "completion_tokens": 2}},
                        input_tokens=1,
                        output_tokens=2,
                    )
                ],
            )
            self.fetches = 0

        async def results(self, *, job: BatchJob, api_key: str) -> list[BatchLineResult]:
            """Fail the first fetch definitively, succeed afterwards."""
            self.fetches += 1
            if self.fetches == 1:
                raise SubmitError(
                    "provider result download failed with status 500", code="provider_error"
                )
            return list(self._results)

    client = FlakyResultsClient()
    engine, store, _, ledger, _ = _engine(client=client)
    file_id = _upload(engine, [_chat_line("a")])
    job = engine.submit(
        organization_id="org_a",
        identity_id="id_a",
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
    )
    asyncio.run(engine.poll_once())
    asyncio.run(engine.poll_once())
    assert store.jobs[job.batch_id].settled is False
    assert ledger.released == []
    asyncio.run(engine.poll_once())
    final = store.jobs[job.batch_id]
    assert final.settled is True
    assert ledger.settled == [("a", 2)]
    assert ledger.released == []


def test_poller_with_stale_snapshot_never_overwrites_a_cancelled_job() -> None:
    """A cancel that wins the claim is final; a racing poller changes nothing."""
    engine, store, _, ledger, client = _engine()
    file_id = _upload(engine, [_chat_line("a")])
    job = engine.submit(
        organization_id="org_a",
        identity_id="id_a",
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
    )
    stale_snapshot = store.jobs[job.batch_id]
    cancelled = asyncio.run(engine.cancel(organization_id="org_a", batch_id=job.batch_id))
    assert cancelled.status is BatchStatus.CANCELLED and cancelled.settled
    releases_after_cancel = list(ledger.released)
    asyncio.run(engine._advance(stale_snapshot))
    final = store.jobs[job.batch_id]
    assert final.status is BatchStatus.CANCELLED
    assert final.settled is True
    assert ledger.released == releases_after_cancel
    assert client.submitted == []
