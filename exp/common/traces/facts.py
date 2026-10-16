"""The generic instance-facts adapter for facts a trace package supplies with a trace.

A trace package supplies a trace's instance facts in its request context under
``instance_facts`` (for OTLP, the ``exp.request.context`` attribute). Normalization moves them
out of the request context into :attr:`Trace.instance_facts`; this adapter validates them when
a replay of that trace asks for its starting world. Domain decoding happens before export, in
whatever adapter wrote the package.
"""

from exp.common.tasks import TaskCase
from exp.common.tasks.facts import InstanceFacts, InstanceFactsError, parse_instance_facts
from exp.common.traces.trace import Trace


class TracePackageFacts:
    """Instance facts supplied in one trace's package, validated on read.

    Attributes:
        trace: The trace whose package may carry facts.
    """

    def __init__(self, trace: Trace) -> None:
        """Bind the adapter to one trace."""
        self.trace = trace

    def initial_facts(self, task: TaskCase) -> InstanceFacts | None:
        """The trace's own instance facts, for a task that this trace is a source of.

        Args:
            task: Task replaying this trace.

        Returns:
            Validated facts, or None when the package supplies none.

        Raises:
            InstanceFactsError: The task does not come from this trace, or the supplied facts
                are malformed or not validated.
        """
        if self.trace.trace_id not in task.source_trace_ids:
            raise InstanceFactsError(
                f"trace {self.trace.trace_id} supplies facts about its own instance only; task "
                f"{task.task_id} does not come from it"
            )
        if self.trace.instance_facts is None:
            return None
        return parse_instance_facts(self.trace.instance_facts, f"trace {self.trace.trace_id}")
