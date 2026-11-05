"""Frozen world-learning retrieval inputs and the exact native query-embedding catalog."""

from pydantic import Field

from exp.common.core.artifacts import ArtifactInput, ContractModel
from exp.optimize.evaluation.executable.catalog_binding import RuntimeCatalogBinding
from exp.simulation.retrieval.scope import CorpusScope
from exp.simulation.retrieval.tags import TagRetrievalBinding


class TraceSearchSpec(ContractModel):
    """A registered tag-aware retrieval over world-learning traces.

    Attributes:
        index: The saved fit-only RAG index, scoped to world-learning traces.
        scope: The world-learning retrieval scope the index was built with.
        tags: Tag ranking policy and task query tags, or None for cosine ranking alone.
        task_id: Task identity whose query tags rank results.
        maximum_top_k: Most matches one search may return.
    """

    index: ArtifactInput
    scope: CorpusScope
    tags: TagRetrievalBinding | None = None
    task_id: str = Field(min_length=1)
    maximum_top_k: int = Field(default=5, ge=1, le=20, strict=True)
    catalog_binding: RuntimeCatalogBinding | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
