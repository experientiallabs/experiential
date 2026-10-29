"""The asynchronous /v1/batches serving lane: engine, contracts, and seams.

Hosts compose ``BatchEngine`` over their own stores, catalog, ledger, and
secret resolver, expose ``BatchControlPlane`` beside the synchronous control
plane, and own the poller task's lifecycle. Batch models are explicit-request
only: they are refused on synchronous routes, and synchronous models are
refused inside batch jobs.
"""

from exp.runtime.gateway.batch.contracts import (
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
)
from exp.runtime.gateway.batch.engine import BatchEngine
from exp.runtime.gateway.batch.interfaces import (
    BatchCatalog,
    BatchFileStore,
    BatchLedger,
    BatchSecretResolver,
    BatchStore,
)
from exp.runtime.gateway.batch.plane import BatchControlPlane
from exp.runtime.gateway.batch.providers import (
    AnthropicBatchClient,
    OpenAIBatchClient,
    OpenRouterBatchClient,
    ProviderBatchClient,
    ProviderBatchSnapshot,
)

__all__ = [
    "AnthropicBatchClient",
    "BatchCatalog",
    "BatchControlPlane",
    "BatchCounts",
    "BatchDeployment",
    "BatchEngine",
    "BatchFile",
    "BatchFileStore",
    "BatchJob",
    "BatchJobPage",
    "BatchLedger",
    "BatchLine",
    "BatchLineError",
    "BatchLineResult",
    "BatchSecretResolver",
    "BatchStatus",
    "BatchStore",
    "BatchSubmitError",
    "BatchSurface",
    "OpenAIBatchClient",
    "OpenRouterBatchClient",
    "ProviderBatchClient",
    "ProviderBatchSnapshot",
]
