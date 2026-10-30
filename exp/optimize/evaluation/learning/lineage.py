"""Identity of a trace-learning plan: its hybrid simulator, memory lineage and dream schedule.

A learning plan's identity binds the simulator it learns into and the memory lineage it
writes. The lineage is a knowledge-compatible bank keyed by the simulator's knowledge, the
scope, the test-then-train protocol and every learning run input besides the trace, simulator
and memory, so changing any of them starts a fresh memory. These helpers are pure over the
artifact store and the replay cases a caller supplies; the learning pass owns case loading.
"""

from collections.abc import Sequence

from exp.common.core.artifacts import ArtifactInput, JsonObject, Sha256, sha256_json, stable_id
from exp.common.project import ArtifactStore, ProjectBuildArtifacts
from exp.optimize.evaluation.fidelity.replay import (
    INVESTIGATION_TOKENS,
    PREDICTION_TOKENS,
    PROPOSAL_REASKS,
    REQUEST_RESERVATION_USD,
    ReplayCase,
    protocol_prompts_sha256,
)
from exp.runtime.models.providers.transport import RetryPolicy
from exp.simulation.engines.text.prompt import INSTANCE_FACTS_GUIDANCE
from exp.simulation.retrieval.evidence_reading import ReadingPolicy
from exp.simulation.world_model.dreaming import DreamPolicy
from exp.simulation.world_model.experience_memory import knowledge_identity
from exp.simulation.world_model.runtime import (
    load_grounded_world_model_artifact,
    persist_fit_only_world_model,
)

PLAN_TYPE = "trace-learning-plan"
FEEDBACK_SOURCE = "recorded-trace-reply"


def plan_world_model(
    store: ArtifactStore, build: ProjectBuildArtifacts, scope: str
) -> ArtifactInput:
    """The hybrid simulator a plan learns into.

    Learned memory supplies behavioral rules; the build's index supplies recorded facts. Each
    task's own lineage is excluded from retrieval, so a trace never retrieves its own recorded
    replies while it is learned. Scope ``fit`` uses the fit-only index, so held-out traces stay
    out of retrieval as well as out of memory.

    Args:
        store: Project artifact owner.
        build: Completed build artifacts.
        scope: The plan's scope, ``all`` or ``fit``.

    Returns:
        The build's world model for ``all``, or its persisted fit-only variant for ``fit``.
    """
    if scope == "all":
        return build.world_model
    return persist_fit_only_world_model(store, build.world_model, build.fit_rag)


def plan_identity(world_model: ArtifactInput, lineage: str) -> str:
    """Saved plan identity for one simulator and memory lineage."""
    plan = {"lineage": lineage, "world_model": world_model.model_dump(mode="json")}
    return stable_id(PLAN_TYPE, plan)


def memory_lineage(
    store: ArtifactStore, world_model: ArtifactInput, scope: str, cases: Sequence[ReplayCase]
) -> str:
    """Memory lineage of one learning simulator under the test-then-train protocol.

    The lineage is a knowledge-compatible bank, separate from exact serving and execution
    identity: a serving change (such as retrieval top-k) keeps the learned memory. It also
    covers every learning run input besides the trace, simulator and memory, so changing any
    of them (a prompt, an allowance, the instance facts delivered to predictions) starts a
    fresh memory that is learned trace by trace again, instead of predicting already-learned
    traces on a memory that has seen every later trace.

    Args:
        store: Project artifact owner.
        world_model: The plan's hybrid simulator.
        scope: The plan's scope, ``all`` or ``fit``.
        cases: The pass's replay cases in learning order.

    Returns:
        The lineage identifier.
    """
    artifact = load_grounded_world_model_artifact(store, world_model)
    key = {"knowledge": knowledge_identity(artifact), "scope": scope}
    delivered = delivered_facts(cases)
    # Only a pass that delivers facts keys on them; fact-free lineages keep their identity.
    facts: JsonObject = {} if delivered is None else {"instance_facts": delivered}
    protocol = learning_protocol()
    return "learn-" + sha256_json({**key, "protocol": "test-then-train", **protocol, **facts})[:16]


def delivered_facts(cases: Sequence[ReplayCase]) -> Sha256 | None:
    """Identity of the instance facts a learning pass delivers to its predictions.

    Facts condition every prediction of their trace, so they shape what each trace teaches:
    adding, changing or removing any trace's conditioning facts starts a fresh memory, as a
    prompt edit does. Verifier-private facts never reach a simulator request and are left out.

    Args:
        cases: The pass's replay cases in learning order.

    Returns:
        A digest over each trace's conditioning facts, or None when no trace delivers any.
    """
    delivered = {
        case.trace_id: [
            fact.model_dump(mode="json")
            for fact in case.facts.facts
            if fact.visibility == "conditioning"
        ]
        for case in cases
        if case.facts is not None
    }
    delivered = {trace: facts for trace, facts in delivered.items() if facts}
    return sha256_json(delivered) if delivered else None


def learning_protocol() -> JsonObject:
    """The learning run inputs shared by every trace of a lineage.

    The prompts include the native-artifact, instance-facts and evidence-reading protocols even
    though only some traces use them: a lineage spans every trace, so editing any of them starts
    a fresh memory.
    """
    return {
        "prompts": protocol_prompts_sha256(native_artifacts=True),
        "facts_prompt": INSTANCE_FACTS_GUIDANCE,
        "reading_protocol": ReadingPolicy().identity,
        "proposal_reasks": PROPOSAL_REASKS,
        "output_tokens": {"prediction": PREDICTION_TOKENS, "investigation": INVESTIGATION_TOKENS},
        "attempts_per_request": RetryPolicy().maximum_attempts,
        "request_reservation_usd": REQUEST_RESERVATION_USD,
        "feedback_source": FEEDBACK_SOURCE,
    }


def dream_schedule(cases: Sequence[ReplayCase], interval: int) -> tuple[str, ...]:
    """Pin periodic and final consolidation boundaries to the imported trace order.

    Args:
        cases: The pass's replay cases in learning order.
        interval: Traces between consolidations.

    Returns:
        Trace IDs after which memory consolidates, including the final trace.
    """
    policy = DreamPolicy(interval=interval)
    return tuple(
        case.trace_id
        for index, case in enumerate(cases, 1)
        if index % policy.interval == 0 or index == len(cases)
    )
