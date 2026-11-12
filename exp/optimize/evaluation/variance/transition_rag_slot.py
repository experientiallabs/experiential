"""The ``state-rag`` slot's transition-tag mode: one sample-evaluate-prune round per family.

For one admitted family this reads the family's fit RAG index and its traces, turns every fit
transition into a :class:`~exp.simulation.retrieval.transition_tags.TransitionStep` (pre-turn
state, action, true next observation), and runs
:func:`~exp.simulation.retrieval.transition_tags.run_round` against the family's current bank: the
saved parent bank when one is given (growth across runs and environments), else an empty bank.
The appended bank version and the grown index are saved as content-addressed artifacts; older
bank versions stay saved, so a checkpoint bound to one still restores. The round record (drafts,
candidates and development steps) is saved beside them for audit.

The whole round replays without a call once its decisions are in the project decision cache and
its authoring completions are saved.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from exp.common.core.artifacts import ArtifactInput, JsonObject
from exp.common.project import ProjectStore
from exp.common.traces import Trace
from exp.runtime.gateway.decisions_contracts import DecisionRequest
from exp.simulation.mining.tag_clusters import write_record
from exp.simulation.retrieval.state_tags import trace_order
from exp.simulation.retrieval.store import load_rag_index
from exp.simulation.retrieval.transition_tags import (
    RoundSpec,
    TransitionAuthor,
    TransitionTagBank,
    TransitionTagIndex,
    load_transition_bank,
    run_round,
    save_transition_bank,
    save_transition_index,
    transition_steps,
)

ROUND_RECORD_TYPE = "transition-tag-round"
"""Artifact type of one round's audit record (``round.json``)."""


def run_transition_family(
    project: ProjectStore,
    family_id: str,
    rag_index: ArtifactInput,
    traces: Mapping[str, Trace],
    author: TransitionAuthor,
    decide: Callable[[DecisionRequest], JsonObject],
    decision_model: JsonObject,
    spec: RoundSpec,
    *,
    parent: ArtifactInput | None,
    code_revision: str,
    concurrency: int = 1,
) -> tuple[ArtifactInput, ArtifactInput]:
    """Run one round over a family's fit steps and save the bank version and index.

    Returns:
        The saved bank version and the saved transition-tag index.

    Raises:
        ValueError: The round kept no tag, so there is nothing to retrieve by.
    """
    store = project.artifacts
    loaded = load_rag_index(store, rag_index.artifact_id)
    trace_ids = sorted({row.trace_id for row in loaded.transitions})
    ordered = trace_order([traces[t] for t in trace_ids], loaded.transitions)
    steps = [
        step
        for rows in ordered.values()
        for step in transition_steps(
            rows,
            history_steps=spec.policy.history_steps,
            observation_characters=spec.policy.observation_characters,
        )
    ]
    bank = (
        TransitionTagBank(scope=family_id)
        if parent is None
        else load_transition_bank(store, parent)
    )
    result = run_round(
        bank.scope,
        steps,
        bank,
        author,
        decide,
        spec,
        environments=(family_id,),
        concurrency=concurrency,
    )
    if not result.bank.active():
        raise ValueError(
            f"the transition-tag round over {family_id} kept no tag; read the round record's "
            "evidence and revise the authoring or the usefulness bar"
        )
    inputs = (rag_index, *(() if parent is None else (parent,)))
    bank_ref = save_transition_bank(store, result.bank, code_revision=code_revision, inputs=inputs)
    index = TransitionTagIndex(
        bank_sha256=result.bank.identity,
        bank_version=result.bank.version,
        decision_model=decision_model,
        rows=result.rows,
    )
    index_ref = save_transition_index(
        store, index, code_revision=code_revision, inputs=(bank_ref, rag_index)
    )
    identity = "transition-tag-round-" + result.bank.identity[:20]
    if not store.exists(identity):
        write_record(
            store,
            identity,
            ROUND_RECORD_TYPE,
            "round.json",
            result.model_copy(update={"rows": ()}),
            code_revision,
            (bank_ref, index_ref),
        )
    return bank_ref, index_ref
