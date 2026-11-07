"""Environment families from tag clusters, reviewed offline by scripted agentic reviewers."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic import JsonValue

from exp.common.core.artifacts import ArtifactInput, JsonObject, SourceIdentity
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
from exp.runtime.models.providers.transport import ProviderTransportError
from exp.simulation.mining.environment_families import (
    FAMILIES_TYPE,
    GROUP_TYPE,
    EnvironmentFamilies,
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
from exp.simulation.mining.review_states import FamiliesPending
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


def test_a_transient_503_leaves_the_group_pending_and_the_next_run_reviews_it(
    tmp_path: Path,
) -> None:
    """A gateway 503 is never a verdict: the group stays pending and the record unwritten;
    the next run reviews it and saves its verdict, replaying the reviewed group with no call."""
    store = ProjectStore(tmp_path, "case").artifacts
    fitted = clustering({"c00": ORDERS, "c01": FILES})
    script, bound = reviewer(by_tools)
    failing = {"f1"}

    def flaky(request: ModelRequest, *, identity: str) -> ModelResponse:
        """Fail every draw about the file group with a 503 on the first run."""
        packet = json.loads(request.messages[1].content or "{}")
        if failing & {row["trace_id"] for row in packet["members"]}:
            raise ProviderTransportError("gateway busy", status_code=503)
        return script(request, identity=identity)

    def run(bound: CompatibilityReviewer) -> tuple[EnvironmentFamilies, ArtifactInput]:
        """Review, or replay, both clusters."""
        return environment_families(
            store,
            fitted,
            bound,
            traces=TRACES,
            eligible=ORDERS + FILES,
            held_out=(),
            code_revision="test",
        )

    with pytest.raises(FamiliesPending) as caught:
        run(CompatibilityReviewer(draw=flaky, model=SNAPSHOT))
    assert set(caught.value.pending) == {"c01"} and caught.value.pending["c01"] == "http_503"
    saved = {store.read(i).manifest.artifact_type for i in store.list_ids()}
    assert FAMILIES_TYPE not in saved and GROUP_TYPE in saved
    first = len(script.requests)

    record, reference = run(bound)
    assert store.read(reference.artifact_id).manifest.artifact_type == FAMILIES_TYPE
    assert {f.cluster_id for f in record.families} == {"c00", "c01"} and not record.uncertain
    reviewed_files = [r for r in script.requests[first:] if "f1" in (r.messages[1].content or "")]
    assert reviewed_files and len(script.requests[first:]) == len(reviewed_files)

    _, refusing = reviewer(by_tools, refuse=True)
    assert run(refusing) == (record, reference)


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
