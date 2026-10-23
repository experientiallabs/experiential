"""Project canonical trace evidence into construction without upgrading its authority."""

from collections.abc import Sequence

from exp.common.core.artifacts import JsonObject
from exp.common.traces import Trace
from exp.simulation.reconstruction.models import RawEpisode, RawInputs


def from_canonical_traces(
    traces: Sequence[Trace],
    *,
    description: str | None = None,
    published_contracts: JsonObject | None = None,
    prior_context: JsonObject | None = None,
) -> RawInputs:
    """Retain original span ordering, provenance and call/result identities.

    Args:
        traces: Caller-admitted canonical traces from an existing source loader.
        description: Optional environment description, independent of observed outcomes.
        published_contracts: Explicit schema treatment. Trace-level tool inventories
            are not silently promoted into the traces-only condition.
        prior_context: Resolved prior records, kept distinct from published descriptions.

    Returns:
        Raw construction input. Canonicalization does not imply native qualification.
    """
    return RawInputs(
        episodes=tuple(
            RawEpisode(
                id=trace.trace_id,
                records=(
                    {
                        "task": trace.task,
                        "initial_context": trace.initial_context,
                        "source": trace.source.model_dump(mode="json"),
                        "conversation_id": trace.conversation_id,
                    },
                    *(span.model_dump(mode="json") for span in trace.spans),
                ),
            )
            for trace in traces
        ),
        description=description,
        published_contracts=published_contracts,
        mode="supplied" if published_contracts is not None else "induced",
        prior_context=prior_context or {},
    )
