"""Frozen knowledge identities preserve the distinction between rules and rejected research."""

from exp.common.core.artifacts import JsonObject, sha256_json
from exp.simulation.engines.text.knowledge import SimulationKnowledge
from exp.simulation.reconstruction.revision import ResearchNote


def test_rejected_research_changes_identity_without_becoming_a_rule() -> None:
    """An investigation remains a separate input channel with its rejection status."""
    note = ResearchNote(
        id="investigation",
        content="Unconditional cancellation failed",
        investigation_sha256="a" * 64,
        source_trace_ids=("trace-a",),
        disposition="rejected",
    )
    parent = SimulationKnowledge(source_trace_ids=("trace-a",), rules="Respect status")
    child = parent.model_copy(update={"research_history": (note,)})
    assert child.identity != parent.identity
    assert child.behavioral_context() == parent.behavioral_context()
    assert child.research_history[0].disposition == "rejected"


def test_prior_originals_and_rejected_research_cannot_leak_into_the_rule_channel() -> None:
    """Four-store rendering keeps retrieved originals and investigations independently visible."""
    context: JsonObject = {
        "library_id": "library",
        "selected": [
            {
                "entry": {"kind": "trajectory", "disposition": "source"},
                "payload": {"value": "historical-user-value"},
            },
            {
                "entry": {"kind": "research", "disposition": "rejected"},
                "payload": {"hypothesis": "rejected-claim"},
            },
            {
                "entry": {"kind": "rule", "disposition": "accepted"},
                "payload": {
                    "knowledge": {
                        "rules": "supported-abstraction",
                        "transfer_context": {"raw_secret": "nested-original"},
                        "research_history": [{"claim": "nested-research"}],
                    },
                    "decision": {"active": {"raw": "decision-original"}},
                },
            },
            {
                "entry": {"kind": "rule", "disposition": "rejected"},
                "payload": {"rule": "never-active"},
            },
        ],
    }
    knowledge = SimulationKnowledge(transfer_context=context)
    behavioral = str(knowledge.behavioral_context())
    assert "supported-abstraction" in behavioral
    assert all(
        value not in behavioral
        for value in (
            "historical-user-value",
            "rejected-claim",
            "never-active",
            "nested-original",
            "nested-research",
            "decision-original",
        )
    )
    assert len(knowledge.original_library_evidence()) == 1
    assert "rejected-claim" in str(knowledge.research_context())
    assert knowledge.transfer_context == context


def test_legacy_empty_claims_keep_saved_knowledge_identity() -> None:
    """Old artifacts retain their exact serialized identity and request context."""
    legacy: JsonObject = {
        "source_trace_ids": [],
        "rules": "",
        "contracts": {},
        "unknowns": [],
        "research_history": [],
        "prediction_guidance": "",
        "transfer_context": {},
    }
    knowledge = SimulationKnowledge.model_validate(legacy)
    assert knowledge.model_dump(mode="json") == legacy
    assert knowledge.identity == sha256_json(legacy)
    assert "claims" not in knowledge.behavioral_context()
