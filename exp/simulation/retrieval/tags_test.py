"""Tests for index tag records and tag-aware retrieval ranking."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import canonical_json_bytes
from exp.common.project import ArtifactCorruptionError, ArtifactStore
from exp.simulation.retrieval import (
    RAGAction,
    RAGQuery,
    TraceRAGRetriever,
    load_rag_index,
    persist_trace_rag,
)
from exp.simulation.retrieval.build import PersistedRAGIndex
from exp.simulation.retrieval.contracts import RAG_INDEX_PATH
from exp.simulation.retrieval.tags import (
    RAGIndexTags,
    RAGTagQuestion,
    RAGTagView,
    RAGTraceTags,
    TagRetrievalBinding,
    TagRetrievalEvidence,
    TagRetrievalPolicy,
    TaskQueryTags,
)
from exp.simulation.retrieval.tests.retrieval_test import (
    _CREATED_AT,
    _bindings,
    _constant_binding,
    _persist_traces,
    _store,
)

_MATRIX = "e" * 64
_QUERY_MATRIX = "f" * 64
_QUESTIONS = (
    RAGTagQuestion(question_id="req.export", family="requirement"),
    RAGTagQuestion(question_id="req.login", family="requirement"),
    RAGTagQuestion(question_id="tone.urgent", family="tone"),
)
_TASK_FIRED: dict[str, tuple[str, ...]] = {
    "trace-0": (),
    "trace-1": ("tone.urgent",),
    "trace-2": ("req.export", "tone.urgent"),
    "trace-3": ("req.login",),
}


def _view(view: str = "task", *, fired: dict[str, tuple[str, ...]] | None = None) -> RAGTagView:
    """Return one tag view record over the four fixture traces.

    Args:
        view: Rendered view the record describes.
        fired: Fired propositions per trace; the default task fixture when omitted.

    Returns:
        The view record.
    """
    rows = _TASK_FIRED if fired is None else fired
    return RAGTagView.model_validate(
        {
            "bank_id": "bank",
            "bank_version": 2,
            "view": view,
            "threshold": 0.5,
            "questions": [item.model_dump() for item in _QUESTIONS],
            "matrix_sha256": _MATRIX,
            "traces": [{"trace_id": key, "fired": list(value)} for key, value in rows.items()],
        }
    )


def _tags(*, observations: bool = False) -> RAGIndexTags:
    """Return the fixture index tag record, optionally with an observations annotation."""
    views = [_view()]
    if observations:
        views.insert(0, _view("observations", fired={key: ("req.login",) for key in _TASK_FIRED}))
    return RAGIndexTags(views=tuple(views))


def _query_tags(*fired: str, view: str = "task") -> TaskQueryTags:
    """Return current-task query tags with the fixture bank and threshold."""
    return TaskQueryTags(
        bank_id="bank",
        bank_version=2,
        view=view,
        threshold=0.5,
        matrix_sha256=_QUERY_MATRIX,
        fired=tuple(sorted(fired)),
    )


def _policy(**update: object) -> TagRetrievalPolicy:
    """Return a policy over the fixture bank, with optional field overrides."""
    payload: dict[str, object] = {
        "bank_id": "bank",
        "bank_version": 2,
        "threshold": 0.5,
        "weight": 1.0,
        **update,
    }
    return TagRetrievalPolicy.model_validate(payload)


def _build(store: ArtifactStore, tags: RAGIndexTags | None) -> PersistedRAGIndex:
    """Persist four fixture traces and build a constant-embedding index, optionally tagged."""
    source, traces = _persist_traces(store, count=4)
    return persist_trace_rag(
        store,
        (source,),
        _bindings(traces),
        created_at=_CREATED_AT,
        code_revision="revision-a",
        embedder=_constant_binding(),
        trace_tags=tags,
    )


def _query() -> RAGQuery:
    """Return a fixed retrieval query over the whole fixture corpus."""
    return RAGQuery(
        task="reset password",
        action=RAGAction(kind="message", content="Which email is on the account?"),
        top_k=4,
    )


def _retriever(store: ArtifactStore, built: PersistedRAGIndex) -> TraceRAGRetriever:
    """Load a constant-embedding retriever, so cosine ties and tags alone decide order."""
    return TraceRAGRetriever(
        load_rag_index(store, built.index.rag_id), embedder=_constant_binding()
    )


def test_untagged_index_keeps_its_bytes_and_identity(tmp_path: Path) -> None:
    """An index built without tags records no tag key and keeps the untagged identity."""
    store = _store(tmp_path, "untagged")
    built = _build(store, None)
    again = _build(_store(tmp_path, "untagged-again"), None)

    assert built.index.trace_tags is None
    assert b"trace_tags" not in store.read_bytes(built.index.rag_id, RAG_INDEX_PATH)
    assert built.index.rag_id == again.index.rag_id


def test_tag_record_round_trips_and_changes_identity(tmp_path: Path) -> None:
    """A tagged index persists its record, reloads it intact and gets a new identity."""
    store = _store(tmp_path, "tagged")
    untagged = _build(store, None)
    tagged = _build(store, _tags(observations=True))

    assert tagged.index.rag_id != untagged.index.rag_id
    loaded = load_rag_index(store, tagged.index.rag_id)
    assert loaded.index.trace_tags == _tags(observations=True)
    record = loaded.index.trace_tags
    assert record is not None
    task = record.view("task")
    assert task is not None
    assert task.fired_by_trace()["trace-2"] == {"req.export", "tone.urgent"}


def test_build_restricts_tags_to_indexed_traces_and_refuses_gaps(tmp_path: Path) -> None:
    """Rows for traces outside the index are dropped; an untagged indexed trace is refused."""
    store = _store(tmp_path, "coverage")
    extra = _view(fired={**_TASK_FIRED, "trace-9": ("req.login",)})
    built = _build(store, RAGIndexTags(views=(extra,)))
    assert built.index.trace_tags == _tags()

    partial = _view(fired={key: value for key, value in _TASK_FIRED.items() if key != "trace-3"})
    with pytest.raises(ValueError, match="no row for 1 indexed traces"):
        _build(_store(tmp_path, "gap"), RAGIndexTags(views=(partial,)))


def test_loader_refuses_a_tag_record_that_misdescribes_the_index(tmp_path: Path) -> None:
    """A tampered record covering other traces than the index holds fails closed on load."""
    store = _store(tmp_path, "tamper")
    built = _build(store, _tags())
    shrunk = _view(fired={key: value for key, value in _TASK_FIRED.items() if key != "trace-0"})
    tampered = built.index.model_copy(
        update={"rag_id": "trace-rag-tampered", "trace_tags": RAGIndexTags(views=(shrunk,))}
    )
    store.write(
        artifact_id=tampered.rag_id,
        artifact_type="trace-rag-index",
        envelope=tampered,
        files={
            RAG_INDEX_PATH: canonical_json_bytes(tampered),
            "transitions.jsonl": store.read_bytes(built.index.rag_id, "transitions.jsonl"),
            "vectors.jsonl": store.read_bytes(built.index.rag_id, "vectors.jsonl"),
        },
    )
    with pytest.raises(ArtifactCorruptionError, match="tag record is inconsistent"):
        load_rag_index(store, tampered.rag_id)


def test_record_contracts_refuse_undeclared_or_repeated_entries() -> None:
    """Fired tags must be declared columns and views must not repeat."""
    with pytest.raises(ValidationError, match="undeclared questions"):
        _view(fired={"trace-0": ("req.unknown",)})
    with pytest.raises(ValidationError, match="sorted and unique"):
        RAGTraceTags(trace_id="t", fired=("b", "a"))
    with pytest.raises(ValidationError, match="at most one"):
        RAGIndexTags(views=(_view(), _view()))


def test_default_retrieval_is_unchanged_on_a_tagged_index(tmp_path: Path) -> None:
    """Without a policy a tagged index ranks exactly as the untagged one."""
    store = _store(tmp_path, "default")
    untagged = _retriever(store, _build(store, None)).retrieve(_query())
    tagged = _retriever(store, _build(store, _tags())).retrieve(_query())

    assert [m.transition.transition_id for m in tagged] == [
        m.transition.transition_id for m in untagged
    ]
    assert [m.score for m in tagged] == [m.score for m in untagged]
    assert all(m.tag_overlap is None for m in tagged)
    assert b"tag_overlap" not in canonical_json_bytes(tagged[0])


def test_tags_rerank_by_cosine_plus_weighted_jaccard(tmp_path: Path) -> None:
    """Score is cosine plus weight times the Jaccard overlap of fired task tags."""
    store = _store(tmp_path, "rank")
    retriever = _retriever(store, _build(store, _tags()))
    baseline = retriever.retrieve(_query())

    matches = retriever.retrieve(
        _query(), tag_policy=_policy(weight=0.5), query_tags=_query_tags("req.export")
    )

    order = [m.transition.trace_id for m in matches]
    assert order[0] == "trace-2"
    assert order != [m.transition.trace_id for m in baseline]
    top = matches[0]
    assert top.tag_overlap == pytest.approx(0.5)
    assert top.score == pytest.approx(1.0 + 0.5 * 0.5)
    assert [m.tag_overlap for m in matches[1:]] == [0.0, 0.0, 0.0]
    assert [m.score for m in matches[1:]] == pytest.approx([1.0, 1.0, 1.0])


def test_zero_weight_boost_keeps_cosine_order(tmp_path: Path) -> None:
    """A zero-weight boost records overlaps but ranks by cosine alone."""
    store = _store(tmp_path, "zero")
    retriever = _retriever(store, _build(store, _tags()))
    baseline = retriever.retrieve(_query())
    matches = retriever.retrieve(
        _query(), tag_policy=_policy(weight=0.0), query_tags=_query_tags("tone.urgent")
    )
    assert [m.transition.transition_id for m in matches] == [
        m.transition.transition_id for m in baseline
    ]


def test_hard_filter_requires_a_shared_family_tag(tmp_path: Path) -> None:
    """The filter keeps only candidates sharing a fired tag from a named family."""
    store = _store(tmp_path, "filter")
    retriever = _retriever(store, _build(store, _tags()))
    policy = _policy(mode="require_shared_family", families=("tone",), weight=0.25)

    matches = retriever.retrieve(
        _query(), tag_policy=policy, query_tags=_query_tags("tone.urgent", "req.login")
    )
    assert {m.transition.trace_id for m in matches} == {"trace-1", "trace-2"}

    nothing = retriever.retrieve(_query(), tag_policy=policy, query_tags=_query_tags("req.login"))
    assert nothing == ()


def test_observations_view_tags_are_refused_at_query_time() -> None:
    """A query can never carry observations-view tags; they would leak the outcome."""
    with pytest.raises(ValidationError, match="must come from the task view"):
        _query_tags("req.login", view="observations")


def test_observations_annotation_never_informs_ranking(tmp_path: Path) -> None:
    """An index carrying only observations tags cannot serve a tag policy at all."""
    store = _store(tmp_path, "obs-only")
    obs = _view("observations")
    retriever = _retriever(store, _build(store, RAGIndexTags(views=(obs,))))
    with pytest.raises(ValueError, match="needs an index built with task-view tags"):
        retriever.retrieve(_query(), tag_policy=_policy(), query_tags=_query_tags("req.login"))


def test_policy_and_query_must_match_the_index(tmp_path: Path) -> None:
    """Bank, revision, threshold, columns and families must agree, and both inputs are needed."""
    store = _store(tmp_path, "mismatch")
    retriever = _retriever(store, _build(store, _tags()))
    query = _query_tags("req.login")
    with pytest.raises(ValueError, match="both a tag policy"):
        retriever.retrieve(_query(), query_tags=query)
    with pytest.raises(ValueError, match="index task tags answer"):
        retriever.retrieve(_query(), tag_policy=_policy(bank_version=3), query_tags=query)
    with pytest.raises(ValueError, match="query tags answer"):
        retriever.retrieve(
            _query(),
            tag_policy=_policy(),
            query_tags=query.model_copy(update={"threshold": 0.6}),
        )
    with pytest.raises(ValueError, match="does not declare"):
        retriever.retrieve(_query(), tag_policy=_policy(), query_tags=_query_tags("req.other"))
    with pytest.raises(ValueError, match="have no question"):
        retriever.retrieve(
            _query(),
            tag_policy=_policy(mode="require_shared_family", families=("absent",)),
            query_tags=query,
        )


def test_policy_contract_is_frozen_and_hashed() -> None:
    """Families follow the mode, the weight is finite and the identity tracks every field."""
    with pytest.raises(ValidationError, match="needs at least one named family"):
        _policy(mode="require_shared_family")
    with pytest.raises(ValidationError, match="name families only for a filter"):
        _policy(families=("tone",))
    with pytest.raises(ValidationError, match="finite"):
        _policy(weight=float("inf"))
    assert _policy().identity == _policy().identity
    assert _policy().identity != _policy(weight=0.5).identity


def test_evidence_records_the_policy_and_query(tmp_path: Path) -> None:
    """Evidence names the policy, its digest, the query tags and the index matrix digest."""
    store = _store(tmp_path, "evidence")
    retriever = _retriever(store, _build(store, _tags()))
    policy = _policy()
    evidence = retriever.tag_evidence(policy, _query_tags("req.login"))

    assert evidence.policy == policy
    assert evidence.policy_sha256 == policy.identity
    assert evidence.query == _query_tags("req.login")
    assert evidence.index_matrix_sha256 == _MATRIX
    with pytest.raises(ValidationError, match="digest differs"):
        TagRetrievalEvidence.model_validate(
            {**evidence.model_dump(mode="json"), "policy_sha256": "0" * 64}
        )


def test_binding_identity_covers_policy_and_only_that_task() -> None:
    """A session binding's identity changes with its policy and its own task's tags only."""
    binding = TagRetrievalBinding(
        policy=_policy(), tasks={"a": _query_tags("req.login"), "b": _query_tags()}
    )
    assert binding.for_task("missing") is None
    other_task_changed = binding.model_copy(
        update={"tasks": {"a": _query_tags("req.login"), "b": _query_tags("tone.urgent")}}
    )
    assert binding.identity_for("a") == other_task_changed.identity_for("a")
    assert binding.identity_for("b") != other_task_changed.identity_for("b")
    reweighted = binding.model_copy(update={"policy": _policy(weight=0.5)})
    assert binding.identity_for("a") != reweighted.identity_for("a")
