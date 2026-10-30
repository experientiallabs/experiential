"""The already-learned guard of the trace learning pass.

A trace whose trajectory the memory lineage already consumed must not be predicted again on
memory that has seen every later trace. An identical repeat earlier in the same pass is the one
exception: it is predicted like any trace and then publishes nothing.
"""

from exp.common.project import ProjectStore
from exp.optimize.evaluation.fidelity.memory import trajectory_record
from exp.optimize.evaluation.fidelity.runner import FidelitySpec
from exp.simulation.world_model.experience_memory import consumed_trajectory_digest, memory_head


def require_unlearned(
    project: ProjectStore, lineage: str, spec: FidelitySpec, earlier: set[str]
) -> None:
    """Refuse to predict a trace this memory already learned, on memory that has seen later traces.

    A trajectory that an earlier trace of the pass consumed belongs to an identical repeat, which
    is predicted like any trace and then publishes nothing.

    Args:
        project: Project holding the memory lineage.
        lineage: Memory lineage of the learning plan being run.
        spec: This trace's learning run.
        earlier: Trajectory digests of the traces before this one in the pass.

    Raises:
        ValueError: The trace is already in the lineage under different run inputs.
    """
    head = memory_head(project.artifacts, lineage)
    digest = consumed_trajectory_digest(trajectory_record(spec))
    if head is not None and digest in head.consumed and digest not in earlier:
        raise ValueError(
            f"trace {spec.task.task_id} is already learned into memory {lineage} "
            f"v{head.version}, but its learning run inputs changed; predicting it again would "
            "read memory that has seen every later trace. The memory is already learned: use "
            "it as is, or change the simulator's knowledge or a learning input to start a fresh "
            "memory"
        )
