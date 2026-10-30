"""Consolidation preserves source authority, counterexamples, reservations and first draws."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject, canonical_json_bytes
from exp.common.models.unknown_outcome import DispatchOutcomeUnknown
from exp.common.project import ArtifactStore, ProjectPaths
from exp.runtime.gateway.decisions_contracts import DecisionRequest
from exp.runtime.models.structured_completion import ReturnedOutputRejected
from exp.simulation.world_model.dreaming import (
    ClaimDraft,
    DreamPolicy,
    DreamResult,
    DreamSource,
    consolidate_memory,
)
from exp.simulation.world_model.experience_memory import Eligibility
from exp.simulation.world_model.memory_tree_test import source
from exp.simulation.world_model.resumption import Resumption

NOW = datetime(2026, 10, 17, tzinfo=UTC)


def draft(**changes: str | tuple[str, ...] | bool) -> ClaimDraft:
    """One scoped hypothesis with an exact original citation."""
    value = {
        "page": "cancel",
        "claim": "Delivered orders cannot be cancelled.",
        "cue": "cancel on a delivered order",
        "scope": "recorded retail cancellation tool",
        "uncertainty": "Only the observed tool version was inspected.",
        "citations": ("a",),
        "transferable": True,
    }
    return ClaimDraft.model_validate({**value, **changes})


def catalog(store: ArtifactStore) -> dict[str, DreamSource]:
    """Contrasting original sources, without expected reader judgments in the records."""
    return {
        "a": DreamSource(
            original={"kind": "original-observation", "reply": "delivered"},
            sources=(source(store, "a"),),
        ),
        "b": DreamSource(
            original={"kind": "original-observation", "reply": "cancelled"},
            sources=(source(store, "b"),),
        ),
    }


def answer(relation: str) -> JsonObject:
    """Return the real typed decision response shape."""
    return {"answers": {"relation": {"type": "choice", "choice": relation}}}


def test_uncited_contradiction_rejects_claim_and_replays_without_calls(tmp_path: Path) -> None:
    """A proposer cannot hide contrary originals by citing only favorable examples."""
    store = ArtifactStore(ProjectPaths(tmp_path, "dream"))
    sources = catalog(store)
    calls = []

    def complete(body: JsonObject, logical: str) -> JsonObject:
        """Propose a claim supported by a while leaving the contrary b uncited."""
        calls.append(logical)
        return {"claims": [draft().model_dump(mode="json")]}

    def decide(request: DecisionRequest) -> JsonObject:
        """Represent independent reader judgments, with a withheld-looking counterexample."""
        assert isinstance(request.state, dict)
        key = request.state["source_id"]
        calls.append(key)
        return answer("support" if key == "a" else "contradict")

    result = consolidate_memory(
        store,
        "pass",
        sources,
        policy=DreamPolicy(),
        eligibility=Eligibility(),
        complete=complete,
        decide=decide,
        clock=lambda: NOW,
        code_revision="test",
    )
    assert not result.records
    assert result.decisions[0].relations == {"a": "support", "b": "contradict"}
    assert result.decisions[0].reason == "contrary evidence in inspected records"
    first = list(calls)
    assert (
        consolidate_memory(
            store,
            "pass",
            sources,
            policy=DreamPolicy(),
            eligibility=Eligibility(),
            complete=complete,
            decide=decide,
            clock=lambda: NOW,
            code_revision="test",
        )
        == result
    )
    assert calls == first


def test_published_claim_depends_on_all_seen_sources_and_keeps_uncertainty(tmp_path: Path) -> None:
    """Uncited context still influenced the proposer and must remain in the source closure."""
    store = ArtifactStore(ProjectPaths(tmp_path, "dream"))
    sources = catalog(store)
    result = consolidate_memory(
        store,
        "pass",
        sources,
        policy=DreamPolicy(),
        eligibility=Eligibility(),
        complete=lambda *_: {"claims": [draft().model_dump(mode="json")]},
        decide=lambda _: answer("support"),
        clock=lambda: NOW,
        code_revision="test",
    )
    record = json.loads(store.read_bytes(result.records[0], "record.json"))
    assert record["kind"] == "wiki-claim"
    assert "not native fact" in record["authority"]
    assert record["claim"]["uncertainty"] == draft().uncertainty
    assert {row["artifact_id"] for row in record["depends_on"]} == {
        sources["a"].sources[0].artifact_id,
        sources["b"].sources[0].artifact_id,
    }
    changed = {**sources, "b": sources["b"].model_copy(update={"sources": sources["a"].sources})}
    with pytest.raises(ValueError, match="sources or policy changed"):
        consolidate_memory(
            store,
            "pass",
            changed,
            policy=DreamPolicy(),
            eligibility=Eligibility(),
            complete=lambda *_: {},
            decide=lambda _: {},
            clock=lambda: NOW,
            code_revision="test",
        )


def test_reserved_and_unresolvable_sources_are_never_sent_to_models(tmp_path: Path) -> None:
    """Strict provenance filtering precedes proposal, reading and publication."""
    store = ArtifactStore(ProjectPaths(tmp_path, "dream"))
    sources = catalog(store)
    seen = []

    def complete(body: JsonObject, logical: str) -> JsonObject:
        """Capture exactly what the proposer received."""
        seen.append(body)
        return {"claims": [draft(citations=("b",)).model_dump(mode="json")]}

    result = consolidate_memory(
        store,
        "pass",
        sources,
        policy=DreamPolicy(),
        eligibility=Eligibility(reserved_task_lineages=("b",)),
        complete=complete,
        decide=lambda _: pytest.fail("invented citation must refuse before readers"),
        clock=lambda: NOW,
        code_revision="test",
    )
    assert set(result.inspected) == {"a"}
    assert result.omitted["b"] == "unknown or excluded source provenance"
    assert result.decisions[0].reason == "unknown or repeated citation"
    assert not result.records
    assert '"cancelled"' not in json.dumps(seen)


def test_resume_reuses_saved_checks_and_qualifies_only_lost_check(tmp_path: Path) -> None:
    """An interrupted second reader never redraws the proposal or the saved first reader."""
    store = ArtifactStore(ProjectPaths(tmp_path, "dream"))
    sources = catalog(store)
    counts = {"propose": 0, "a": 0, "b": 0}
    retries = []

    def complete(body: JsonObject, logical: str) -> JsonObject:
        """Count actual proposer draws."""
        counts["propose"] += 1
        return {"claims": [draft().model_dump(mode="json")]}

    def decide(request: DecisionRequest) -> JsonObject:
        """Lose b's first response after its durable intent."""
        assert isinstance(request.state, dict)
        key = str(request.state["source_id"])
        counts[key] += 1
        if key == "b" and counts[key] == 1:
            raise TimeoutError("response lost")
        return answer("support")

    policy = DreamPolicy(workers=1)
    with pytest.raises(TimeoutError):
        consolidate_memory(
            store,
            "pass",
            sources,
            policy=policy,
            eligibility=Eligibility(),
            complete=complete,
            decide=decide,
            clock=lambda: NOW,
            code_revision="test",
        )
    with pytest.raises(DispatchOutcomeUnknown):
        consolidate_memory(
            store,
            "pass",
            sources,
            policy=policy,
            eligibility=Eligibility(),
            complete=complete,
            decide=decide,
            clock=lambda: NOW,
            code_revision="test",
        )
    result = consolidate_memory(
        store,
        "pass",
        sources,
        policy=policy,
        eligibility=Eligibility(),
        complete=complete,
        decide=decide,
        clock=lambda: NOW,
        code_revision="test",
        resumption=Resumption(window=NOW, retried=lambda *row: retries.append(row)),
    )
    assert result.records
    assert counts == {"propose": 1, "a": 1, "b": 2}
    assert len(retries) == 1 and retries[0][0] == "investigation"
    assert retries[0][1].startswith("dream:pass-claim-")


def test_invalid_returned_proposal_is_logged_not_redrawn(tmp_path: Path) -> None:
    """A rejected provider response is a seen draw, not an unknown outcome."""
    store = ArtifactStore(ProjectPaths(tmp_path, "dream"))
    sources = catalog(store)
    calls = []

    def complete(body: JsonObject, logical: str) -> JsonObject:
        """Simulate structured completion rejecting actual returned model output."""
        calls.append(logical)
        raise ReturnedOutputRejected("bad output")

    for _ in range(2):
        result = consolidate_memory(
            store,
            "pass",
            sources,
            policy=DreamPolicy(),
            eligibility=Eligibility(),
            complete=complete,
            decide=lambda _: {},
            clock=lambda: NOW,
            code_revision="test",
        )
        assert result.invalid_proposal == "ValidationError"
        assert not result.records
    assert len(calls) == 1


def test_claim_cannot_use_another_hypothesis_as_observational_support(tmp_path: Path) -> None:
    """Even a reader that agrees cannot turn repeated model text into fresh evidence."""
    store = ArtifactStore(ProjectPaths(tmp_path, "dream"))
    sources = catalog(store)
    sources["old"] = DreamSource(
        original={"kind": "wiki-claim", "claim": draft().model_dump(mode="json")},
        sources=sources["a"].sources,
    )
    result = consolidate_memory(
        store,
        "self-support",
        sources,
        policy=DreamPolicy(),
        eligibility=Eligibility(),
        complete=lambda *_: {"claims": [draft(citations=("old",)).model_dump(mode="json")]},
        decide=lambda _: answer("support"),
        clock=lambda: NOW,
        code_revision="test",
    )
    assert not result.records
    assert result.decisions[0].reason == "citations must name original observations"


def test_old_hypothesis_cannot_veto_observed_correction_and_withdrawal(tmp_path: Path) -> None:
    """Old prose supplies context and provenance, while only originals can confirm or contradict."""
    store = ArtifactStore(ProjectPaths(tmp_path, "dream"))
    sources = catalog(store)
    sources["old"] = DreamSource(
        original={"kind": "wiki-claim", "claim": draft().model_dump(mode="json")},
        sources=sources["a"].sources,
    )
    seen = []

    def check(request: DecisionRequest) -> JsonObject:
        """Original observations support the correction; the old prose says the opposite."""
        assert isinstance(request.state, dict)
        key = request.state["source_id"]
        seen.append(key)
        return answer("contradict" if key == "old" else "support")

    corrected = draft(claim="Cancellation was observed in this tool version.", supersedes=("old",))
    result = consolidate_memory(
        store,
        "correction",
        sources,
        policy=DreamPolicy(),
        eligibility=Eligibility(),
        complete=lambda *_: {"claims": [corrected.model_dump(mode="json")]},
        decide=check,
        clock=lambda: NOW,
        code_revision="test",
    )
    assert result.records and set(seen) == {"a", "b"}
    record = json.loads(store.read_bytes(result.records[0], "record.json"))
    assert record["investigation"]["supersedes"] == ["old"]
    assert record["authority"].startswith("Checked historical hypothesis")
    withdrawn = consolidate_memory(
        store,
        "withdrawal",
        sources,
        policy=DreamPolicy(),
        eligibility=Eligibility(),
        complete=lambda *_: {
            "claims": [
                draft(retire=True, supersedes=("old",), citations=("b",)).model_dump(mode="json")
            ]
        },
        decide=lambda _: answer("contradict"),
        clock=lambda: NOW,
        code_revision="test",
    )
    retired = json.loads(store.read_bytes(withdrawn.records[0], "record.json"))
    assert retired["kind"] == "wiki-retirement"
    assert retired["investigation"]["transferable"] is False
    assert retired["investigation"]["supersedes"] == ["old"]


def test_proposer_keeps_hypotheses_out_of_originals_and_citation_schema(tmp_path: Path) -> None:
    """A live proposer cited research notes when they were grouped under the originals label."""
    from jsonschema import Draft202012Validator

    from exp.simulation.world_model.dreaming import _request

    store = ArtifactStore(ProjectPaths(tmp_path, "dream"))
    sources = catalog(store)
    sources["research"] = DreamSource(
        original={"kind": "research", "hypothesis": "A guessed behavioral rule."},
        sources=sources["a"].sources,
    )
    request = _request(sources, DreamPolicy())
    messages = request["messages"]
    assert isinstance(messages, list) and isinstance(messages[1], dict)
    payload = json.loads(str(messages[1]["content"]))
    assert set(payload["originals"]) == {"a", "b"}
    assert payload["research_context"]["research"] == sources["research"].original
    validator = Draft202012Validator(request["response_schema"])
    assert validator.is_valid({"claims": [draft().model_dump(mode="json")]})
    assert not validator.is_valid(
        {"claims": [draft(citations=("research",)).model_dump(mode="json")]}
    )
    assert not validator.is_valid(
        {"claims": [draft(citations=("invented",)).model_dump(mode="json")]}
    )
    assert validator.is_valid({"claims": []})


def test_unlabeled_claim_still_requires_full_observed_support(tmp_path: Path) -> None:
    """Missing organizational metadata is explicit, while semantic/evidence fields stay required."""
    store = ArtifactStore(ProjectPaths(tmp_path, "dream"))
    sources = catalog(store)
    raw = draft().model_dump(mode="json")
    del raw["page"]
    assert ClaimDraft.model_validate(raw).page is None
    for required in ("claim", "cue", "scope", "uncertainty", "citations", "transferable"):
        with pytest.raises(ValueError):
            ClaimDraft.model_validate({key: value for key, value in raw.items() if key != required})
    result = consolidate_memory(
        store,
        "unlabeled",
        sources,
        policy=DreamPolicy(),
        eligibility=Eligibility(),
        complete=lambda *_: {"claims": [raw]},
        decide=lambda _: answer("support"),
        clock=lambda: NOW,
        code_revision="test",
    )
    assert result.records and result.decisions[0].proposal.page is None
    assert result.decisions[0].relations == {"a": "support", "b": "support"}
    record = json.loads(store.read_bytes(result.records[0], "record.json"))
    assert record["claim"]["page"] is None
    rejected = consolidate_memory(
        store,
        "unlabeled-unsupported",
        sources,
        policy=DreamPolicy(),
        eligibility=Eligibility(),
        complete=lambda *_: {"claims": [raw]},
        decide=lambda _: answer("unknown"),
        clock=lambda: NOW,
        code_revision="test",
    )
    assert not rejected.records and "does not support" in rejected.decisions[0].reason


def test_relational_claim_checks_complete_cited_originals_together(tmp_path: Path) -> None:
    """Two observations establish a comparison which neither observation alone can establish."""
    store = ArtifactStore(ProjectPaths(tmp_path, "joint"))
    sources = catalog(store)
    sources = {
        key: row.model_copy(
            update={
                "original": {
                    "kind": "original-observation",
                    "file": "input.xlsx",
                    "A1": value,
                }
            }
        )
        for (key, row), value in zip(sources.items(), ("ITEM", "Count"), strict=True)
    }
    claim = draft(
        claim="Two observed input.xlsx files had different A1 values.",
        scope="These two recorded instances only.",
        citations=("a", "b"),
    )
    requests: list[DecisionRequest] = []

    def decide(request: DecisionRequest) -> JsonObject:
        """Compare the actual paired values only when both original records are visible."""
        requests.append(request)
        assert isinstance(request.state, dict)
        originals = request.state.get("originals")
        if originals is None:
            return answer("unknown")
        assert originals == {key: row.original for key, row in sources.items()}
        assert "research_context" not in request.state
        return answer("support")

    result = consolidate_memory(
        store,
        "joint",
        sources,
        policy=DreamPolicy(workers=1),
        eligibility=Eligibility(),
        complete=lambda *_: {"claims": [claim.model_dump(mode="json")]},
        decide=decide,
        clock=lambda: NOW,
        code_revision="test",
    )
    assert len(result.records) == 1
    assert result.decisions[0].relations == {"a": "unknown", "b": "unknown"}
    assert result.decisions[0].joint_relation == "support"
    assert len(requests) == 3
    record = json.loads(store.read_bytes(result.records[0], "record.json"))
    assert set(record["citations"]) == {"a", "b"}
    assert "not native fact" in record["authority"]
    assert record["support_check"] == "joint cited observations"
    again = consolidate_memory(
        store,
        "joint",
        sources,
        policy=DreamPolicy(workers=1),
        eligibility=Eligibility(),
        complete=lambda *_: pytest.fail("saved proposal redrawn"),
        decide=lambda _: pytest.fail("saved checks redrawn"),
        clock=lambda: NOW,
        code_revision="test",
    )
    assert again == result


@pytest.mark.parametrize("joint", ["unknown", "both", "contradict", "garbage"])
def test_joint_support_must_be_positive_and_well_formed(tmp_path: Path, joint: str) -> None:
    """Ambiguous, contrary and malformed combined judgments cannot publish a hypothesis."""
    store = ArtifactStore(ProjectPaths(tmp_path, "joint"))
    claim = draft(citations=("a", "b"))

    def decide(request: DecisionRequest) -> JsonObject:
        """Return the selected joint answer only for a complete multi-record request."""
        assert isinstance(request.state, dict)
        return answer(joint if "originals" in request.state else "unknown")

    result = consolidate_memory(
        store,
        "joint",
        catalog(store),
        policy=DreamPolicy(),
        eligibility=Eligibility(),
        complete=lambda *_: {"claims": [claim.model_dump(mode="json")]},
        decide=decide,
        clock=lambda: NOW,
        code_revision="test",
    )
    assert not result.records
    assert result.decisions[0].joint_relation == (joint if joint != "garbage" else "invalid")


def test_joint_support_cannot_overrule_an_uncited_contradiction(tmp_path: Path) -> None:
    """The relational check is not even dispatched after a local counterexample is observed."""
    store = ArtifactStore(ProjectPaths(tmp_path, "joint"))
    sources = catalog(store)
    sources["c"] = DreamSource(
        original={"kind": "original-observation", "reply": "contrary"},
        sources=(source(store, "c"),),
    )
    claim = draft(citations=("a", "b"))

    def decide(request: DecisionRequest) -> JsonObject:
        """Both citations are individually inconclusive, but a third original contradicts them."""
        assert isinstance(request.state, dict) and "originals" not in request.state
        return answer("contradict" if request.state["source_id"] == "c" else "unknown")

    result = consolidate_memory(
        store,
        "joint",
        sources,
        policy=DreamPolicy(),
        eligibility=Eligibility(),
        complete=lambda *_: {"claims": [claim.model_dump(mode="json")]},
        decide=decide,
        clock=lambda: NOW,
        code_revision="test",
    )
    assert not result.records and result.decisions[0].joint_relation is None
    assert result.decisions[0].reason == "contrary evidence in inspected records"


def test_lost_joint_judgment_never_redraws_implicitly(tmp_path: Path) -> None:
    """A lost combined response preserves the proposal and every completed local judgment."""
    store = ArtifactStore(ProjectPaths(tmp_path, "joint"))
    sources = catalog(store)
    calls: list[str] = []

    def decide(request: DecisionRequest) -> JsonObject:
        """Lose only the combined judgment after its intent has been saved."""
        assert isinstance(request.state, dict)
        if "originals" in request.state:
            calls.append("joint")
            raise TimeoutError("lost joint response")
        calls.append(str(request.state["source_id"]))
        return answer("unknown")

    def run() -> DreamResult:
        """Re-enter precisely the same frozen pass after interruption."""
        return consolidate_memory(
            store,
            "joint",
            sources,
            policy=DreamPolicy(workers=1),
            eligibility=Eligibility(),
            complete=lambda *_: {"claims": [draft(citations=("a", "b")).model_dump(mode="json")]},
            decide=decide,
            clock=lambda: NOW,
            code_revision="test",
        )

    with pytest.raises(TimeoutError, match="lost joint"):
        run()
    with pytest.raises(DispatchOutcomeUnknown):
        run()
    assert calls == ["a", "b", "joint"]


def test_joint_bundle_over_budget_is_not_truncated_or_dispatched(tmp_path: Path) -> None:
    """Complete originals fit the proposer, but a long claim makes their joint read too large."""
    store = ArtifactStore(ProjectPaths(tmp_path, "joint"))
    sources = {
        key: row.model_copy(
            update={
                "original": {
                    "kind": "original-observation",
                    "reply": key * 1400,
                }
            }
        )
        for key, row in catalog(store).items()
    }
    claim = draft(
        claim="c" * 2000,
        cue="q" * 1000,
        scope="s" * 1000,
        uncertainty="u" * 1000,
        citations=("a", "b"),
    )
    seen = []

    def decide(request: DecisionRequest) -> JsonObject:
        """Only complete individual records can be sent under this byte ceiling."""
        assert isinstance(request.state, dict) and "originals" not in request.state
        seen.append(request.state["source_id"])
        return answer("unknown")

    result = consolidate_memory(
        store,
        "joint",
        sources,
        policy=DreamPolicy(input_bytes=8000),
        eligibility=Eligibility(),
        complete=lambda *_: {"claims": [claim.model_dump(mode="json")]},
        decide=decide,
        clock=lambda: NOW,
        code_revision="test",
    )
    assert set(seen) == {"a", "b"}
    assert not result.records and result.decisions[0].joint_relation == "over_budget"
    assert result.decisions[0].reason == "complete joint evidence exceeds reader budget"
    assert DreamPolicy(maximum_records=2, maximum_claims=1).calls == 4


def test_historical_support_does_not_automatically_make_a_claim_transferable(
    tmp_path: Path,
) -> None:
    """A supported instance-specific fact remains a record, not a reusable behavioral rule."""
    store = ArtifactStore(ProjectPaths(tmp_path, "historical"))
    proposal = {**draft().model_dump(mode="json"), "transferable": False}
    result = consolidate_memory(
        store,
        "historical",
        catalog(store),
        policy=DreamPolicy(),
        eligibility=Eligibility(),
        complete=lambda *_: {"claims": [proposal]},
        decide=lambda _: answer("support"),
        clock=lambda: NOW,
        code_revision="test",
    )
    assert len(result.records) == 1
    record = json.loads(store.read_bytes(result.records[0], "record.json"))
    assert record["investigation"]["transferable"] is False


def test_originals_cannot_be_crowded_out_by_research_context(tmp_path: Path) -> None:
    """Interpretations arriving first cannot consume all slots for observable evidence."""
    store = ArtifactStore(ProjectPaths(tmp_path, "selection"))
    originals = catalog(store)
    sources = {
        **{
            key: DreamSource(
                original={"kind": "research", "hypothesis": "interpretation"},
                sources=originals["a"].sources,
            )
            for key in ("research-1", "research-2")
        },
        **originals,
    }
    seen = []

    def complete(body: JsonObject, _: str) -> JsonObject:
        """Capture the actual proposer request, without assuming its selection algorithm."""
        messages = body["messages"]
        assert isinstance(messages, list) and isinstance(messages[1], dict)
        seen.append(json.loads(str(messages[1]["content"])))
        return {"claims": []}

    result = consolidate_memory(
        store,
        "selection",
        sources,
        policy=DreamPolicy(maximum_records=2),
        eligibility=Eligibility(),
        complete=complete,
        decide=lambda _: pytest.fail("empty proposal must not dispatch readers"),
        clock=lambda: NOW,
        code_revision="test",
    )
    assert set(seen[0]["originals"]) == {"a", "b"}
    assert set(result.omitted) == {"research-1", "research-2"}


@pytest.mark.parametrize("citation", ["a", "b"])
def test_individual_reader_never_dispatches_an_oversized_complete_request(
    tmp_path: Path, citation: str
) -> None:
    """Adding a long claim to a fitting original can exceed the reader's own byte budget."""
    store = ArtifactStore(ProjectPaths(tmp_path, "reader-budget"))
    sources = catalog(store)
    small = sources["b"]
    sources = {
        "a": sources["a"].model_copy(
            update={
                "original": {
                    "kind": "original-observation",
                    "reply": "a" * 3400,
                }
            }
        ),
        "b": small,
    }
    claim = draft(
        claim="c" * 2000,
        cue="q" * 1000,
        scope="s" * 1000,
        uncertainty="u" * 1000,
        citations=(citation,),
    )
    seen = []

    def decide(request: DecisionRequest) -> JsonObject:
        """Never allow a model call whose complete request exceeds the registered bound."""
        size = len(canonical_json_bytes(request.model_dump(mode="json")))
        assert size <= 9000, f"oversized request dispatched: {size}"
        assert isinstance(request.state, dict)
        seen.append(request.state["source_id"])
        return answer("support")

    result = consolidate_memory(
        store,
        "reader-budget",
        sources,
        policy=DreamPolicy(input_bytes=9000),
        eligibility=Eligibility(),
        complete=lambda *_: {"claims": [claim.model_dump(mode="json")]},
        decide=decide,
        clock=lambda: NOW,
        code_revision="test",
    )
    assert {"a", "b"} <= result.inspected.keys()
    assert seen == ["b"] and not result.records
    assert result.decisions[0].relations == {"a": "over_budget", "b": "support"}


@pytest.mark.parametrize("transferable", [True, False])
def test_historical_claim_cannot_silently_retire_a_reusable_rule(
    tmp_path: Path, transferable: bool
) -> None:
    """Historical revisions may replace historical records, but cannot erase reusable rules."""
    store = ArtifactStore(ProjectPaths(tmp_path, "supersession"))
    sources = catalog(store)
    sources["old"] = DreamSource(
        original={
            "kind": "wiki-claim",
            "claim": draft(transferable=transferable).model_dump(mode="json"),
        },
        sources=sources["a"].sources,
    )
    result = consolidate_memory(
        store,
        "supersession",
        sources,
        policy=DreamPolicy(),
        eligibility=Eligibility(),
        complete=lambda *_: {
            "claims": [draft(transferable=False, supersedes=("old",)).model_dump(mode="json")]
        },
        decide=lambda _: answer("support"),
        clock=lambda: NOW,
        code_revision="test",
    )
    assert result.decisions[0].accepted is not transferable
    if transferable:
        assert "cannot supersede a reusable rule" in result.decisions[0].reason
        assert result.decisions[0].relations == {}


@pytest.mark.parametrize("invalid", [None, "true", "false", 0, 1])
def test_transferability_is_an_explicit_boolean(invalid: object) -> None:
    """Neither missing intent nor coercible model prose grants reusable-rule authority."""
    value = {**draft().model_dump(mode="json"), "transferable": invalid}
    with pytest.raises(ValueError):
        ClaimDraft.model_validate(value)


def test_large_episode_leaves_room_to_revise_an_eligible_wiki_claim(tmp_path: Path) -> None:
    """New originals cannot indefinitely starve the old claims that consolidation revises."""
    store = ArtifactStore(ProjectPaths(tmp_path, "crowding"))
    originals = {
        f"o{i}": DreamSource(
            original={"kind": "original-observation", "reply": i},
            sources=(source(store, f"original-{i}"),),
        )
        for i in range(12)
    }
    originals["old-rule"] = DreamSource(
        original={"kind": "wiki-claim", "claim": draft().model_dump(mode="json")},
        sources=(source(store, "old"),),
    )
    seen: list[JsonObject] = []

    def complete(body: JsonObject, logical: str) -> JsonObject:
        """Retain the actual complete proposer input and make no new claims."""
        seen.append(body)
        return {"claims": []}

    result = consolidate_memory(
        store,
        "crowding",
        originals,
        policy=DreamPolicy(),
        eligibility=Eligibility(),
        complete=complete,
        decide=lambda _: answer("support"),
        clock=lambda: NOW,
        code_revision="test",
    )
    assert "old-rule" not in result.omitted
    assert "old-rule" in json.dumps(seen)
    assert len(result.omitted) == 1
