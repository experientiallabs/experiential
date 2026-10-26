"""Agent memory stages with injected completions: merging, failures, plans and windows.

The provider path (admitted transport, consent and replay) runs end to end in
``exp/cli/evaluation/agent_memory_test.py``.
"""

import json
import operator
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput, JsonObject, JsonValue
from exp.common.models.catalog import (
    ConnectionConfig,
    ModelCatalog,
    ModelRecord,
    load_model_catalog,
    write_model_catalog,
)
from exp.common.models.discovery import DiscoveredModel, resolve_discovered_model
from exp.common.models.model import BillingSource, ModelCapabilities
from exp.common.models.unknown_outcome import DispatchOutcomeUnknown
from exp.common.project import ProjectStore
from exp.optimize.evaluation.learning import agent_memory
from exp.optimize.evaluation.learning.agent_memory import (
    Completions,
    PreparedMemory,
    consolidate,
    consolidation_cost,
    extract_candidates,
    extraction_cost,
    prepare_agent_memory,
    stage_completions,
)
from exp.optimize.evaluation.learning.agent_memory_contracts import (
    AgentMemorySpec,
    CandidateSet,
    FailedCall,
    Lesson,
)
from exp.optimize.evaluation.spending import operation_spend_ceiling
from exp.runtime.models.spend_ceiling import install_spend_ceiling, operation_spend_scope
from exp.runtime.models.structured_completion import ReturnedOutputRejected

NOW = datetime(2026, 10, 24, 12, tzinfo=UTC)


# The OSS route the project runs: an OpenAI-compatible gateway with the catalog's own prices.
GATEWAY = ConnectionConfig(
    provider="openai-compatible",
    base_url="https://api.experientiallabs.ai/v1",
    api_key_env="EXPERIENTIAL_API_KEY",
)
GATEWAY_MODEL = ModelCapabilities(
    supports_tools=True,
    supports_structured_output=True,
    supports_completions=True,
    context_window_tokens=1_048_576,
    maximum_output_tokens=131_072,
    input_cost_per_million_tokens_usd=1.4,
    output_cost_per_million_tokens_usd=4.4,
)


def project_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    priced: bool = True,
    model: str = "glm-5.3",
) -> Path:
    """An EXP root whose catalog holds one learning alias, ``learner``.

    ``glm-5.3`` is served by the OpenAI-compatible gateway with declared prices; an Anthropic
    model is resolved by model discovery from EXP's maintained metadata.
    """
    if model.startswith("claude-"):
        connection = ConnectionConfig(provider="anthropic", api_key_env="ANTHROPIC_API_KEY")
        capabilities = resolve_discovered_model(
            DiscoveredModel(provider="anthropic", model=model)
        ).capabilities
    else:
        connection, capabilities = GATEWAY, GATEWAY_MODEL
    if not priced:
        capabilities = capabilities.model_copy(update={"input_cost_per_million_tokens_usd": None})
    root = tmp_path / ".exp"
    root.mkdir()
    write_model_catalog(
        root / "models.toml",
        ModelCatalog(
            connections={"provider": connection},
            models={
                "learner": ModelRecord(
                    connection="provider",
                    model=model,
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    capabilities=capabilities,
                )
            },
        ),
    )
    assert connection.api_key_env is not None
    monkeypatch.setenv(connection.api_key_env, "fixture-key")
    return root


def at(value: JsonValue, *keys: str) -> JsonValue:
    """The value at a path of object keys, asserting every step is an object."""
    for key in keys:
        assert isinstance(value, dict), key
        value = value[key]
    return value


def clock() -> datetime:
    """A fixed aware clock."""
    return NOW


def export(tmp_path: Path, rows: list[JsonObject]) -> Path:
    """A chat JSON export of the given conversations."""
    path = tmp_path / "traces.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def passed(trace_id: str, text: str = "report") -> JsonObject:
    """One passing conversation."""
    return {
        "trace_id": trace_id,
        "exp.outcome.status": "success",
        "messages": [
            {"role": "user", "content": f"Write the {text}."},
            {"role": "assistant", "content": "Done."},
        ],
    }


def prepared_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    count: int = 3,
    *,
    batch_bytes: int = 150_000,
    maximum_trace_bytes: int = 40_000,
) -> tuple[ProjectStore, PreparedMemory]:
    """A registered plan over ``count`` passing traces."""
    root = project_root(tmp_path, monkeypatch)
    store = ProjectStore(root, "memory")
    spec = AgentMemorySpec(
        traces=export(tmp_path, [passed(f"t{index}") for index in range(count)]),
        output=tmp_path / "out",
        alias="learner",
        batch_bytes=batch_bytes,
        maximum_trace_bytes=maximum_trace_bytes,
    )
    return store, prepare_agent_memory(store, spec, code_revision="test", clock=clock)


def scripted(
    answers: dict[str, Callable[[JsonObject], JsonObject]],
) -> Callable[[], Completions]:
    """Builds stage completions that answer from the request body, recording nothing."""

    def stage(name: str) -> Callable[[JsonObject, str], JsonObject]:
        """One stage's completion."""
        return lambda body, _logical_id: answers[name](body)

    return lambda: Completions(
        extract=stage("extract"), merge=stage("merge"), screen=stage("screen"), concurrency=2
    )


def shown_labels(body: JsonObject) -> list[str]:
    """The trace labels an extraction request may cite."""
    enum = at(body, "response_schema", "properties", "lessons", "items", "properties", "traces")
    enum = at(enum, "items", "enum")
    assert isinstance(enum, list)
    return [str(label) for label in enum]


def lesson(statement: str, traces: list[str], kind: str = "procedure") -> JsonObject:
    """One extracted lesson."""
    return {
        "kind": kind,
        "statement": statement,
        "applies_when": "always",
        "evidence": f"seen in {traces}",
        "traces": traces,
    }


def candidates(*rows: Lesson) -> CandidateSet:
    """An unsaved candidate set holding the given lessons."""
    return CandidateSet(
        schema_version=1,
        created_at=NOW,
        code_revision="test",
        candidates_id="candidates",
        plan=ArtifactInput(artifact_id="plan", sha256="0" * 64),
        candidates=rows,
        failed=(),
    )


def test_merging_unites_citations_and_evidence_and_never_adds_a_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two batches cite different traces; their merged lesson cites both, once."""
    store, prepared = prepared_memory(
        tmp_path, monkeypatch, batch_bytes=10_000, maximum_trace_bytes=5_000
    )
    prepared = PreparedMemory(
        plan=prepared.plan,
        reference=prepared.reference,
        traces=prepared.traces,
        batches=((0,), (1, 2)),
        alias_sha256=prepared.alias_sha256,
    )
    completions = scripted(
        {
            "extract": lambda body: {"lessons": [lesson("Use make test.", shown_labels(body))]},
            "merge": lambda _body: {
                "lessons": [{"statement": "Run make test.", "applies_when": "a", "sources": [0, 1]}]
            },
            "screen": lambda _body: {
                "screenings": [{"lesson": 0, "general": 1, "covered": 0, "supported": 0.6}]
            },
        }
    )
    found = extract_candidates(store, prepared, completions, code_revision="t", clock=clock)
    assert [len(row.traces) for row in found[1].candidates] == [1, 2]
    _, memory = consolidate(store, prepared, found, completions, code_revision="t", clock=clock)
    (merged,) = memory.lessons
    assert merged.statement == "Run make test."
    assert merged.traces == tuple(sorted(trace.trace_id for trace in prepared.traces))
    assert merged.evidence == ("seen in ['T1']", "seen in ['T2', 'T3']")
    assert merged.score == pytest.approx(0.6) and merged.accepted
    assert memory.outcomes == {"PASS": 3, "FAIL": 0, "unknown": 0}


def test_unusable_answers_are_recorded_by_kind_and_leave_their_inputs_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unknown, refused and malformed answers are kept apart; none is retried or invented."""
    store, prepared = prepared_memory(tmp_path, monkeypatch)

    def unknown(_body: JsonObject) -> JsonObject:
        """A dispatch whose answer was never saved."""
        raise DispatchOutcomeUnknown("lost")

    def refused(_body: JsonObject) -> JsonObject:
        """A returned answer the completion owner refused."""
        raise ReturnedOutputRejected("not a proposal")

    lost = scripted({"extract": unknown})
    _, nothing = extract_candidates(store, prepared, lost, code_revision="t", clock=clock)
    assert nothing.candidates == ()
    assert [(row.stage, row.status) for row in nothing.failed] == [("extract", "unknown")]

    other = ProjectStore(store.paths.root, "second")
    prepared = prepare_agent_memory(
        other,
        AgentMemorySpec(traces=tmp_path / "traces.jsonl", output=tmp_path / "o", alias="learner"),
        code_revision="t",
        clock=clock,
    )
    completions = scripted(
        {
            "extract": lambda body: {
                "lessons": [lesson(f"s{label}", [label], "pitfall") for label in shown_labels(body)]
            },
            "merge": refused,
            "screen": lambda _body: {"screenings": []},
        }
    )
    found = extract_candidates(other, prepared, completions, code_revision="t", clock=clock)
    _, memory = consolidate(other, prepared, found, completions, code_revision="t", clock=clock)
    assert [lesson.statement for lesson in memory.lessons] == ["sT1", "sT2", "sT3"]
    assert all(lesson.screening is None and not lesson.accepted for lesson in memory.lessons)
    assert [(row.stage, row.status) for row in memory.failed] == [
        ("merge", "rejected"),
        ("screen", "rejected"),
    ]
    assert "answer refused" in memory.failed[1].reason


def test_any_other_failure_stops_the_stage_without_saving_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A spend stop or transport error propagates, so a rerun resumes the same requests."""
    store, prepared = prepared_memory(tmp_path, monkeypatch)

    def stopped(_body: JsonObject) -> JsonObject:
        """An infrastructure stop."""
        raise TimeoutError("completion deadline reached before dispatch")

    with pytest.raises(TimeoutError):
        extract_candidates(
            store,
            prepared,
            scripted({"extract": stopped}),
            code_revision="t",
            clock=clock,
        )
    assert store.artifacts.list_ids(artifact_type="agent-memory-candidates") == ()


def long_candidates(*names: str) -> list[Lesson]:
    """Procedure candidates of about 4,000 shown bytes each, two to a 10,000-byte chunk."""
    return [
        Lesson(
            kind="procedure",
            statement=f"{name} " + "x" * 4_000,
            applies_when="w",
            evidence=("e",),
            traces=("t",),
        )
        for name in names
    ]


@pytest.mark.parametrize(
    ("failure", "status"),
    [
        (DispatchOutcomeUnknown("lost"), "unknown"),
        (ReturnedOutputRejected("not a proposal"), "rejected"),
    ],
)
def test_candidates_of_a_failed_merge_are_never_asked_about_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception, status: str
) -> None:
    """A lost or refused merge beside a shrinking one keeps its candidates out of later rounds."""
    _, prepared = prepared_memory(
        tmp_path, monkeypatch, batch_bytes=10_000, maximum_trace_bytes=5_000
    )
    rows = long_candidates("a1", "a2", "b1", "b2")
    asked: list[str] = []

    def merge(body: JsonObject) -> JsonObject:
        """Merge the a candidates into one lesson; fail every request showing a b candidate."""
        text = json.dumps(body)
        asked.append(text)
        if "b1 " in text or "b2 " in text:
            raise failure
        return {"lessons": [{"statement": "m", "applies_when": "w", "sources": [0, 1]}]}

    failed: list[FailedCall] = []
    merged = agent_memory._merge_kind(prepared, rows, scripted({"merge": merge})(), failed)
    assert [row.statement for row in merged] == ["m", *(row.statement for row in rows[2:])]
    assert sum("b1 " in text for text in asked) == 1
    assert [row.status for row in failed] == [status]


def test_a_resumed_merge_never_asks_about_a_failed_chunk_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run stopped in round two resumes with the same requests, none over the lost chunk."""
    _, prepared = prepared_memory(
        tmp_path, monkeypatch, batch_bytes=10_000, maximum_trace_bytes=5_000
    )
    rows = long_candidates("a1", "a2", "a3", "a4", "b1", "b2")
    asked: list[str] = []
    stopped: list[str] = []

    def merge(body: JsonObject) -> JsonObject:
        """Merge a pairs, lose the b pair, and stop once at the first round-two request."""
        text = json.dumps(body)
        asked.append(text)
        if "b1 " in text or "b2 " in text:
            raise DispatchOutcomeUnknown("lost")
        later = "merged-a1" in text
        if later and not stopped:
            stopped.append(text)
            raise TimeoutError("completion deadline reached before dispatch")
        statement = "merged-a1" if later or "a1 " in text else "merged-a3"
        return {"lessons": [{"statement": statement, "applies_when": "w", "sources": [0, 1]}]}

    with pytest.raises(TimeoutError):
        agent_memory._merge_kind(prepared, rows, scripted({"merge": merge})(), [])
    failed: list[FailedCall] = []
    merged = agent_memory._merge_kind(prepared, rows, scripted({"merge": merge})(), failed)
    assert [row.statement for row in merged] == [
        "merged-a1",
        *(row.statement for row in rows[4:]),
    ]
    assert len({text for text in asked if "b1 " in text}) == 1
    assert [row.status for row in failed] == ["unknown"]


def test_the_plan_replays_and_the_window_is_fixed_by_the_first_paid_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preparing twice registers one plan; a later window never moves the deadline."""
    store, first = prepared_memory(tmp_path, monkeypatch)
    spec = AgentMemorySpec(
        traces=tmp_path / "traces.jsonl", output=tmp_path / "out", alias="learner"
    )
    again = prepare_agent_memory(
        store, spec, code_revision="test", clock=lambda: NOW + timedelta(days=3)
    )
    assert again.reference == first.reference
    assert store.artifacts.list_ids(artifact_type="agent-memory-window") == ()
    with operation_spend_scope():
        install_spend_ceiling(operation_spend_ceiling(store, "operation"))
        for hours_later in (0, 9):
            stage_completions(
                store,
                first,
                hours=2,
                concurrency=1,
                code_revision="t",
                clock=partial(operator.add, NOW, timedelta(hours=hours_later)),
            )
    (window_id,) = store.artifacts.list_ids(artifact_type="agent-memory-window")
    saved = json.loads(store.artifacts.read_bytes(window_id, "window.json"))
    assert datetime.fromisoformat(saved["deadline"]) == NOW + timedelta(hours=2)


def test_stage_completions_are_refused_before_consent_installs_the_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Completions built before consent would dispatch unreserved, so building refuses."""
    store, prepared = prepared_memory(tmp_path, monkeypatch)
    with operation_spend_scope(), pytest.raises(ValueError, match="after the stage's spend"):
        stage_completions(store, prepared, hours=2, concurrency=1, code_revision="t", clock=clock)


def edited_model(catalog: ModelCatalog) -> ModelCatalog:
    """The learning alias pointed at another model."""
    record = catalog.models["learner"].model_copy(update={"model": "glm-6"})
    return catalog.model_copy(update={"models": {"learner": record}})


def edited_capabilities(catalog: ModelCatalog) -> ModelCatalog:
    """The learning alias with another output price and nothing else changed."""
    record = catalog.models["learner"]
    assert record.capabilities is not None
    capabilities = record.capabilities.model_copy(
        update={"output_cost_per_million_tokens_usd": 9.9}
    )
    edited = record.model_copy(update={"capabilities": capabilities})
    return catalog.model_copy(update={"models": {"learner": edited}})


def edited_connection(catalog: ModelCatalog) -> ModelCatalog:
    """The learning alias's connection pointed at another endpoint."""
    connection = catalog.connections["provider"].model_copy(
        update={"base_url": "https://other.experientiallabs.ai/v1"}
    )
    return catalog.model_copy(update={"connections": {"provider": connection}})


@pytest.mark.parametrize("edit", [edited_model, edited_capabilities, edited_connection])
def test_stage_completions_refuse_a_catalog_edited_after_planning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    edit: Callable[[ModelCatalog], ModelCatalog],
) -> None:
    """A changed model, capability or connection would dispatch under the plan, so it refuses."""
    store, prepared = prepared_memory(tmp_path, monkeypatch)
    write_model_catalog(
        store.model_catalog_path, edit(load_model_catalog(store.model_catalog_path))
    )
    with operation_spend_scope():
        install_spend_ceiling(operation_spend_ceiling(store, "operation"))
        with pytest.raises(ValueError, match="changed in the model catalog after the plan"):
            stage_completions(
                store, prepared, hours=2, concurrency=1, code_revision="t", clock=clock
            )


def test_exclusions_are_refused_unless_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record the source cannot normalize stops learning until the operator allows it."""
    root = project_root(tmp_path, monkeypatch)
    store = ProjectStore(root, "memory")
    broken = {**passed("bad"), "exp.outcome.status": "failure"}
    traces = export(tmp_path, [passed("good"), broken])
    spec = AgentMemorySpec(traces=traces, output=tmp_path / "out", alias="learner")
    with pytest.raises(ValueError, match="1 records .* cannot be normalized"):
        prepare_agent_memory(store, spec, code_revision="t", clock=clock)
    allowed = spec.model_copy(update={"allow_exclusions": True})
    prepared = prepare_agent_memory(store, allowed, code_revision="t", clock=clock)
    assert len(prepared.traces) == 1


def test_costs_price_every_request_at_the_spend_ceilings_whole_generation_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Consent shows what admission will reserve, so it can never stop a consented run."""
    store, prepared = prepared_memory(
        tmp_path, monkeypatch, count=4, batch_bytes=10_000, maximum_trace_bytes=5_000
    )
    extraction = extraction_cost(store, prepared)
    assert extraction.attempts == len(prepared.batches)
    window, price = (
        GATEWAY_MODEL.context_window_tokens,
        GATEWAY_MODEL.output_cost_per_million_tokens_usd,
    )
    assert window is not None and price is not None
    assert extraction.usd is not None
    assert extraction.usd > extraction.attempts * window * price / 1_000_000
    rows = tuple(
        Lesson(kind=kind, statement=f"s{index}", applies_when="w", evidence=("e",), traces=("t",))
        for index in range(25)
        for kind in ("procedure", "pitfall")
    )
    merge, screen = consolidation_cost(store, prepared, candidates(*rows))
    assert merge.attempts == 2
    assert screen.attempts == 4
    assert consolidation_cost(store, prepared, candidates())[0].attempts == 0
    large = tuple(
        Lesson(
            kind="procedure",
            statement="s" * 4_000,
            applies_when="w",
            evidence=("e",),
            traces=("t",),
        )
        for _ in range(10)
    )
    merge, _ = consolidation_cost(store, prepared, candidates(*large))
    assert merge.attempts == 5 + (agent_memory.MERGE_ROUNDS - 1) * 5


def test_the_merge_estimate_covers_later_rounds_that_lengthen_lessons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Answers longer than their sources pack into more chunks than the first round had."""
    store, prepared = prepared_memory(
        tmp_path, monkeypatch, batch_bytes=10_000, maximum_trace_bytes=5_000
    )
    rows = [
        Lesson(
            kind="procedure",
            statement=f"c{index} " + "x" * 3_200,
            applies_when="w",
            evidence=("e",),
            traces=("t",),
        )
        for index in range(6)
    ]
    sent: list[int] = []

    def lengthen(body: JsonObject) -> JsonObject:
        """Keep every candidate apart, each rewritten longer than it was shown."""
        last = at(body, "response_schema", "properties", "lessons", "items", "properties")
        last = at(last, "sources", "items", "maximum")
        assert isinstance(last, int)
        sent.append(last + 1)
        return {
            "lessons": [
                {"statement": f"m{index} " + "y" * 4_800, "applies_when": "w", "sources": [index]}
                for index in range(last + 1)
            ]
        }

    agent_memory._merge_kind(prepared, rows, scripted({"merge": lengthen})(), [])
    merge, _ = consolidation_cost(store, prepared, candidates(*rows))
    assert sent[:2] == [3, 3] and len(sent) > 2 * agent_memory.MERGE_ROUNDS
    assert merge.attempts >= len(sent)
