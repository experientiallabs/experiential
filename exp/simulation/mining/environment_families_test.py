"""Environment families from tag clusters, reviewed offline by scripted agentic reviewers."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject, SourceIdentity
from exp.common.models import (
    AssistantAction,
    BillingSource,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    OperationEconomics,
    ToolCall,
)
from exp.common.project import ProjectStore
from exp.common.tasks import TaskCase, ToolSchema
from exp.common.traces import Trace
from exp.common.traces.trace import TraceOutcome, TraceSource
from exp.simulation.mining.environment_families import (
    FAMILIES_TYPE,
    EnvironmentFamilies,
    EnvironmentFamilySpec,
    EnvironmentReviewWindow,
    admitted_families,
    environment_families,
    review_environment_families,
)
from exp.simulation.mining.environment_review import (
    READ_TRACE,
    RECONCILE_PROMPT,
    REVIEW_TOOL,
    REVIEW_VARIANTS,
    CompatibilityReviewer,
    CompatibilitySpec,
    ReviewerError,
)
from exp.simulation.mining.tag_clusters import (
    ClusterCountScore,
    ClusteringSpec,
    TagCluster,
    TagClustering,
    assign_clusters,
)
from exp.simulation.mining.tagging import TagMatrix, TagRow
from exp.simulation.rules.contracts_test import call, result

SNAPSHOT = ModelSnapshot(
    provider="fixture",
    model_id="reviewer-fixture",
    billing_source=BillingSource.CUSTOMER_MANAGED,
    capabilities_sha256="a" * 64,
    connection_sha256="b" * 64,
)

Decide = Callable[[frozenset[str], str], JsonObject]
"""Scripted answer for a group of members under a variant (or ``reconcile``)."""


def order_trace(trace_id: str) -> Trace:
    """An order-service episode: look up then refund."""
    return _trace(
        trace_id,
        "Refund an order",
        (
            call(f"{trace_id}c1", 0, "lookup", {"order_id": "7"}, "k1"),
            result(f"{trace_id}r1", 1, "lookup", {"status": "paid", "amount": 40}, "k1"),
            call(f"{trace_id}c2", 2, "refund", {"order_id": "7"}, "k2"),
            result(f"{trace_id}r2", 3, "refund", {"ok": True}, "k2"),
        ),
        ("lookup", "refund"),
    )


def file_trace(trace_id: str) -> Trace:
    """A file-system episode: read then write a file."""
    return _trace(
        trace_id,
        "Fix the config file",
        (
            call(f"{trace_id}c1", 0, "read_file", {"path": "a.toml"}, "k1"),
            result(f"{trace_id}r1", 1, "read_file", "x = 1", "k1"),
            call(f"{trace_id}c2", 2, "write_file", {"path": "a.toml", "text": "x=2"}, "k2"),
            result(f"{trace_id}r2", 3, "write_file", {"written": 3}, "k2"),
        ),
        ("read_file", "write_file"),
    )


def _trace(trace_id: str, task: str, spans: tuple, tools: tuple[str, ...]) -> Trace:
    """One canonical trace declaring its tools."""
    return Trace(
        trace_id=trace_id,
        task=task,
        tools=tuple(
            ToolSchema(name=name, description=name, input_schema={"type": "object"})
            for name in tools
        ),
        spans=spans,
        outcome=TraceOutcome(status="success"),
        source=TraceSource(
            identity=SourceIdentity(kind="otlp", source_id="fixture.otlp", sha256="a" * 64),
            semantic_convention_version="1.37.0",
        ),
    )


def clustering(groups: dict[str, tuple[str, ...]]) -> TagClustering:
    """A clustering with the given clusters over one tagged column."""
    clusters = tuple(
        TagCluster(
            cluster_id=cid,
            size=len(members),
            weight=len(members) / sum(len(m) for m in groups.values()),
            propositions={"task:q0": 0.9 if position == 0 else 0.1},
            members=members,
        )
        for position, (cid, members) in enumerate(groups.items())
    )
    traces = tuple(sorted(t for members in groups.values() for t in members))
    return TagClustering(
        spec=ClusteringSpec(),
        sources=(),
        columns=("task:q0",),
        trace_ids=traces,
        scores=(
            ClusterCountScore(
                clusters=len(groups),
                log_likelihood=0.0,
                parameters=1,
                bic=0.0,
                occupied=len(groups),
                held_out=0.0,
                stability=1.0,
                stable=True,
                selected=True,
            ),
        ),
        clusters=clusters,
        assignments={t: cid for cid, members in groups.items() for t in members},
        confidence=dict.fromkeys(traces, 1.0),
        reason="fixture",
    )


def answer(verdict: str, members: frozenset[str], **extra: JsonValue) -> JsonObject:
    """A well-cited answer of one verdict about one group."""
    first = sorted(members)[0]
    body: JsonObject = {
        "verdict": verdict,
        "rationale": f"{verdict} by script",
        "evidence": [f"trace:{first}"],
    }
    if verdict == "compatible":
        body["state_domains"] = [
            {"name": "records", "description": "shared records", "evidence": [f"trace:{first}"]}
        ]
        body["world_semantics"] = "the same calls have the same effects"
    if verdict == "uncertain":
        body["open_questions"] = [{"question": "Do refunds settle the same way?"}]
    body.update(extra)
    return body


def split(*parts: tuple[str, ...]) -> list[JsonValue]:
    """Sub-groups of a split, each citing its first member."""
    return [
        {"members": list(part), "rationale": "different tools", "evidence": [f"trace:{part[0]}"]}
        for part in parts
    ]


@dataclass
class ScriptedReviewer:
    """Reads one member, then answers from the script; saves responses by identity.

    Attributes:
        decide: The scripted answer per group and variant.
        refuse: Raise for every draw (a replay must not draw).
        requests: Every request answered, in order.
        saved: Responses by identity.
    """

    decide: Decide
    refuse: bool = False
    requests: list[ModelRequest] = field(default_factory=list)
    saved: dict[str, ModelResponse] = field(default_factory=dict)

    def __call__(self, request: ModelRequest, *, identity: str) -> ModelResponse:
        """Answer one draw."""
        if identity in self.saved:
            return self.saved[identity]
        if self.refuse:
            raise ReviewerError("scripted outage")
        self.requests.append(request)
        system = request.messages[0].content or ""
        variant = (
            "reconcile"
            if system == RECONCILE_PROMPT
            else next(name for name, text in REVIEW_VARIANTS.items() if text in system)
        )
        packet = json.loads(request.messages[1].content or "{}")
        members = frozenset(row["trace_id"] for row in packet["members"])
        read = any(message.role == "tool" for message in request.messages)
        names = {tool.name for tool in request.tools}
        if READ_TRACE in names and not read:
            tool_call = ToolCall(
                call_id="r", name=READ_TRACE, arguments={"trace_id": sorted(members)[0]}
            )
        else:
            tool_call = ToolCall(
                call_id="s", name=REVIEW_TOOL, arguments=self.decide(members, variant)
            )
        response = ModelResponse(
            output=AssistantAction(tool_calls=(tool_call,)),
            model=SNAPSHOT,
            economics=OperationEconomics(),
        )
        self.saved[identity] = response
        return response


def reviewer(
    decide: Decide, *, refuse: bool = False
) -> tuple[ScriptedReviewer, CompatibilityReviewer]:
    """A scripted reviewer and its injected binding."""
    script = ScriptedReviewer(decide, refuse=refuse)
    return script, CompatibilityReviewer(draw=script, model=SNAPSHOT)


ORDERS = ("o1", "o2", "o3")
FILES = ("f1", "f2")
TRACES = {t: order_trace(t) for t in ORDERS} | {t: file_trace(t) for t in FILES}


def by_tools(members: frozenset[str], variant: str) -> JsonObject:
    """Split a mixed group by environment; call a pure group compatible."""
    del variant
    orders = tuple(sorted(members & set(ORDERS)))
    files = tuple(sorted(members & set(FILES)))
    if orders and files:
        return answer("split", members, subgroups=split(orders, files))
    return answer("compatible", members)


def task(task_id: str, sources: tuple[str, ...], *, partition: str = "fit") -> TaskCase:
    """A mined task over source traces."""
    return TaskCase.model_validate(
        {
            "task_id": task_id,
            "lineage_group_id": f"lineage-{task_id}",
            "partition": partition,
            "instruction": "do it",
            "initial_context": {"account": task_id},
            "workload_weight": 1.0,
            "source_trace_ids": sources,
        }
    )


def test_a_compatible_cluster_becomes_one_family_with_contracts_tasks_and_evidence() -> None:
    """Three variant loops read before answering, a reconcile decides, tasks attach."""
    script, bound = reviewer(by_tools)

    record = review_environment_families(
        clustering({"c00": ORDERS}),
        bound,
        traces=TRACES,
        eligible=ORDERS,
        held_out=(),
        tasks=(task("t1", ("o1",)), task("t2", ("o2", "o3"))),
    )

    (family,) = record.families
    assert not record.uncertain
    assert family.status == "compatible" and family.members == ORDERS
    assert set(family.tool_contracts) == {"lookup", "refund"}
    assert family.tool_contracts["lookup"].calls == 3
    assert [d.name for d in family.state_domains] == ["records"]
    assert family.task_ids() == ("t1", "t2")
    assert family.tasks[0].start.initial_context_sha256
    assert len(family.evidence.reviews) == 3
    assert all(row.reads == 1 and row.turns == 2 for row in family.evidence.reviews)
    assert family.evidence.reconcile is not None and family.evidence.reconcile.answer
    assert len(script.requests) == 3 * 2 + 1


def test_a_mixed_cluster_is_split_and_each_part_rereviewed_into_its_own_family() -> None:
    """The mixed cluster splits; both parts are reviewed again and become families."""
    script, bound = reviewer(by_tools)
    members = ORDERS + FILES

    record = review_environment_families(
        clustering({"c00": members}), bound, traces=TRACES, eligible=members, held_out=()
    )

    assert [f.path for f in record.families] == ["c00/0", "c00/1"]
    assert [f.members for f in record.families] == [ORDERS, FILES]
    assert set(record.families[1].tool_contracts) == {"read_file", "write_file"}
    assert len({f.family_id for f in record.families}) == 2
    assert not record.uncertain
    assert len(script.requests) == 3 * 7


def test_an_uncertain_cluster_is_kept_with_its_questions_and_admitted_only_when_named() -> None:
    """An uncertain verdict is never forced into a family."""
    _, bound = reviewer(lambda members, variant: answer("uncertain", members))

    record = review_environment_families(
        clustering({"c00": ORDERS}), bound, traces=TRACES, eligible=ORDERS, held_out=()
    )

    assert not record.families
    (group,) = record.uncertain
    assert group.open_questions[0].question.startswith("Do refunds")
    assert admitted_families(record) == ()
    (admitted,) = admitted_families(record, (group.group_id,), traces=TRACES)
    assert admitted.status == "admitted_uncertain" and admitted.members == ORDERS
    assert set(admitted.tool_contracts) == {"lookup", "refund"}
    with pytest.raises(ValueError, match="no uncertain group"):
        admitted_families(record, ("unknown",))


def test_a_reconcile_must_address_dissent_and_cite_only_what_was_shown() -> None:
    """A reconcile overriding a split review without saying why is repaired once."""
    drafts: list[str] = []

    def decide(members: frozenset[str], variant: str) -> JsonObject:
        """One variant wants a split; the reconcile first ignores it, then addresses it."""
        if variant == "contract_first":
            return answer("split", members, subgroups=split(("o1",), ("o2", "o3")))
        if variant != "reconcile":
            return answer("compatible", members)
        drafts.append(variant)
        if len(drafts) == 1:
            return answer("compatible", members, evidence=["trace:never-shown"])
        dissent: list[JsonValue] = [{"variant": "contract_first", "resolution": "shapes agree"}]
        return answer("compatible", members, dissent=dissent)

    _, bound = reviewer(decide)
    record = review_environment_families(
        clustering({"c00": ORDERS}), bound, traces=TRACES, eligible=ORDERS, held_out=()
    )

    (family,) = record.families
    final = family.evidence.reconcile
    assert final is not None and final.repairs == 1 and final.answer is not None
    assert final.answer.dissent[0].variant == "contract_first"


def test_failed_reviews_and_the_depth_bound_keep_groups_uncertain() -> None:
    """Too few usable reviews, or sub-groups past the bound, stay uncertain."""
    _, failing = reviewer(by_tools, refuse=True)
    record = review_environment_families(
        clustering({"c00": ORDERS}), failing, traces=TRACES, eligible=ORDERS, held_out=()
    )
    assert not record.families and "usable reviews" in record.uncertain[0].reason

    _, bound = reviewer(by_tools)
    members = ORDERS + FILES
    record = review_environment_families(
        clustering({"c00": members}),
        bound,
        traces=TRACES,
        eligible=members,
        held_out=(),
        spec=CompatibilitySpec(max_depth=0),
    )
    assert not record.families
    assert [g.path for g in record.uncertain] == ["c00/0", "c00/1"]
    assert all("depth bound" in g.reason for g in record.uncertain)


def test_held_out_traces_and_tasks_are_refused_and_ineligible_members_skipped() -> None:
    """A clustering holding a held-out trace, or a held-out task, is refused."""
    _, bound = reviewer(by_tools)
    with pytest.raises(ValueError, match="held-out"):
        review_environment_families(
            clustering({"c00": ORDERS}), bound, traces=TRACES, eligible=ORDERS, held_out=("o1",)
        )
    with pytest.raises(ValueError, match="held out"):
        review_environment_families(
            clustering({"c00": ORDERS}),
            bound,
            traces=TRACES,
            eligible=ORDERS,
            held_out=(),
            tasks=(task("t9", ("o1",), partition="held_out"),),
        )
    record = review_environment_families(
        clustering({"c00": ORDERS, "c01": FILES}),
        bound,
        traces=TRACES,
        eligible=ORDERS,
        held_out=(),
    )
    assert record.skipped == {"c01": "no eligible member"}


def test_saved_families_replay_without_any_call(tmp_path: Path) -> None:
    """The saved record replays for the same clustering, reviewer and evidence."""
    store = ProjectStore(tmp_path, "case").artifacts
    fitted = clustering({"c00": ORDERS + FILES})
    _, bound = reviewer(by_tools)
    record, reference = environment_families(
        store,
        fitted,
        bound,
        traces=TRACES,
        eligible=ORDERS + FILES,
        held_out=(),
        code_revision="test",
    )
    assert store.read(reference.artifact_id).manifest.artifact_type == FAMILIES_TYPE

    _, refusing = reviewer(by_tools, refuse=True)
    replay, again = environment_families(
        store,
        fitted,
        refusing,
        traces=TRACES,
        eligible=ORDERS + FILES,
        held_out=(),
        code_revision="test",
    )
    assert (replay, again) == (record, reference)


def test_traces_outside_the_fit_are_placed_by_the_mixture_over_their_answered_columns() -> None:
    """A probe trace answering one column is placed by its posterior, others marginalized."""
    fitted = clustering({"c00": ORDERS, "c01": FILES})
    fitted = fitted.model_copy(
        update={
            "columns": ("task:q0", "observations:q0"),
            "clusters": tuple(
                c.model_copy(update={"propositions": {**c.propositions, "observations:q0": 0.5}})
                for c in fitted.clusters
            ),
        }
    )
    matrix = TagMatrix(
        bank_id="b",
        bank_version=1,
        question_ids=("q0",),
        view="task",
        rows=(
            TagRow(trace_id="p1", unit_hash="u1", values=(0.95,)),
            TagRow(trace_id="p2", unit_hash="u2", values=(0.05,)),
        ),
    )

    placed = assign_clusters(fitted, [matrix], ("p1", "p2", "p3"))

    assert placed["p1"][0] == "c00" and placed["p2"][0] == "c01"
    assert placed["p1"][1] > 0.5 and "p3" not in placed


@pytest.mark.parametrize("workers", [None, 1, 2, 4])
def test_root_review_keeps_serial_results_and_complete_split_trees(
    workers: int | None,
) -> None:
    """Root scheduling preserves native draw IDs, private collectors and original order."""
    groups = {"mixed": ("o1", "f1"), "uncertain": ("o2",), "orders": ("o3",)}
    selected = {key: TRACES[key] for members in groups.values() for key in members}

    def decide(members: frozenset[str], variant: str) -> JsonObject:
        """Leave one root uncertain and recursively review the mixed root."""
        if members == frozenset(("o2",)):
            return answer("uncertain", members)
        return by_tools(members, variant)

    def run(cap: int | None) -> tuple[EnvironmentFamilies, ScriptedReviewer]:
        """Run the exact evidence with only the registered execution cap changed."""
        script, bound = reviewer(decide)
        record = review_environment_families(
            clustering(groups),
            bound,
            traces=selected,
            eligible=tuple(selected),
            held_out=(),
            spec=CompatibilitySpec(maximum_parallel_groups=cap),
        )
        return record, script

    serial, serial_script = run(None)
    actual, script = run(workers)
    assert actual.model_copy(update={"spec": serial.spec}) == serial
    assert script.saved == serial_script.saved
    assert len(script.requests) == len(serial_script.requests)
    assert [family.members for family in actual.families] == [("o1",), ("f1",), ("o3",)]
    assert [row.members for row in actual.uncertain] == [("o2",)]
    members = [member for row in (*actual.families, *actual.uncertain) for member in row.members]
    assert sorted(members) == sorted(selected)
    assert len(members) == len(set(members))


@pytest.mark.parametrize("start,count", [(0, 2), (2, 2), (4, 3), (10, 1)])
def test_registered_window_quotes_native_roots_and_retains_all_queued_evidence(
    start: int,
    count: int,
) -> None:
    """A finite wave executes its source-order slice and keeps every unselected root."""
    traces = {f"o{i}": order_trace(f"o{i}") for i in range(5)}
    dropped = order_trace("not-tagged")
    script, bound = reviewer(lambda members, _variant: answer("compatible", members))
    spec = EnvironmentFamilySpec(
        review=CompatibilitySpec(
            variants=("contract_first",),
            min_reviews=1,
            max_turns=1,
            repairs=0,
            max_depth=0,
            maximum_parallel_groups=2,
            maximum_candidate_members=64,
            maximum_candidate_packet_bytes=48_000,
        ),
        review_window=EnvironmentReviewWindow(start=start, count=count),
    )
    fitted = clustering({f"c{i}": (key,) for i, key in enumerate(traces)})
    record = review_environment_families(
        fitted,
        bound,
        traces={**traces, dropped.trace_id: dropped},
        eligible=(*traces, dropped.trace_id),
        held_out=(),
        spec=spec.review,
        review_window=spec.review_window,
    )
    selected = list(traces)[start : start + count]
    assert [row.members for row in record.families] == [(key,) for key in selected]
    assert len(script.requests) == len(selected) * 2
    quote = spec.candidate_calls(tuple(traces.values()), clusters=5)
    assert quote == min(count, max(0, 5 - start)) * 2
    assert len(script.requests) <= quote
    represented = [member for row in (*record.families, *record.queued) for member in row.members]
    assert sorted(represented) == sorted(traces)
    assert len(set(represented)) == len(represented)
    assert record.unclustered == ("not-tagged",)
    assert not record.uncertain
    assert [root.cluster_id for root in record.queued] == [
        f"c{i}" for i, key in enumerate(traces) if key not in selected
    ]
    assert all(root.group_id for root in record.queued)
    assert spec.review_window == record.review_window


def test_unset_window_preserves_native_aggregate_identity_and_serialized_records() -> None:
    """Absent window and empty queued inventory retain the original canonical contract."""
    from exp.simulation.mining.environment_families import families_identity

    traces = {"o1": TRACES["o1"]}
    _, bound = reviewer(lambda members, _variant: answer("compatible", members))
    fitted = clustering({"orders": ("o1",)})
    spec = CompatibilitySpec()
    assert "review_window" not in EnvironmentFamilySpec().model_dump(mode="json")
    original = families_identity(fitted, bound, spec, traces, ())
    assert families_identity(fitted, bound, spec, traces, (), review_window=None) == original
    assert (
        families_identity(
            fitted,
            bound,
            spec,
            traces,
            (),
            review_window=EnvironmentReviewWindow(count=1),
        )
        != original
    )
    record = review_environment_families(
        fitted, bound, traces=traces, eligible=traces, held_out=(), spec=spec
    )
    payload = record.model_dump(mode="json")
    assert not {"review_window", "queued", "unclustered"} & payload.keys()
    for values in ({"start": -1, "count": 1}, {"start": 0, "count": 0}):
        with pytest.raises(ValueError):
            EnvironmentReviewWindow.model_validate(values)


def test_windowed_replay_keys_the_exact_eligible_unclustered_inventory(tmp_path: Path) -> None:
    """Two excluded tag inventories cannot silently share a saved windowed report."""
    project = ProjectStore(tmp_path, "queued")
    traces = {"o1": TRACES["o1"]}
    script, bound = reviewer(lambda members, _variant: answer("compatible", members))
    fitted = clustering({"orders": ("o1",)})
    first, before = environment_families(
        project.artifacts,
        fitted,
        bound,
        traces=traces,
        eligible=(*traces, "untagged-a"),
        held_out=(),
        review_window=EnvironmentReviewWindow(count=1),
        code_revision="test",
    )
    calls = len(script.requests)
    second, after = environment_families(
        project.artifacts,
        fitted,
        bound,
        traces=traces,
        eligible=(*traces, "untagged-b"),
        held_out=(),
        review_window=EnvironmentReviewWindow(count=1),
        code_revision="test",
    )
    assert first.unclustered == ("untagged-a",)
    assert second.unclustered == ("untagged-b",)
    assert before != after
    assert len(script.requests) == calls
