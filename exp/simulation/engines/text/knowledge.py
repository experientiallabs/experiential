"""Versioned behavioral knowledge and investigation records for text simulation."""

from operator import not_

from pydantic import Field

from exp.common.core.artifacts import ContractModel, JsonObject, Sha256, sha256_json
from exp.simulation.reconstruction.context import prior_records
from exp.simulation.reconstruction.models import Claim
from exp.simulation.reconstruction.revision import ResearchNote


class SimulationKnowledge(ContractModel):
    """Immutable knowledge pinned by the simulator artifact for an entire session.

    Attributes:
        source_trace_ids: Whole source episodes exposed during construction.
        rules: Source-backed behavioral abstractions, not current entity values.
        contracts: Inferred or supplied tool interfaces with their authority labels.
        claims: Scoped generated propositions with original support and authority labels.
        unknowns: Behavior the source episodes do not establish.
        research_history: Investigations limited to the permitted source pool.
        prediction_guidance: Tunable interpretation guidance under the fixed wire protocol.
        transfer_context: Explicit prior library selection, separate from target episode count.
        authored: Declares rules, contracts, claims, unknowns and guidance hand-written rather
            than derived from recorded traces, so held-out evaluation accepts them without
            source traces. Research notes and library entries always need provenance.
    """

    source_trace_ids: tuple[str, ...] = ()
    rules: str = ""
    contracts: JsonObject = Field(default_factory=dict)
    claims: tuple[Claim, ...] = Field(default=(), exclude_if=not_)
    unknowns: tuple[str, ...] = ()
    research_history: tuple[ResearchNote, ...] = ()
    prediction_guidance: str = ""
    transfer_context: JsonObject = Field(default_factory=dict)
    authored: bool = Field(default=False, strict=True, exclude_if=not_)

    @property
    def identity(self) -> Sha256:
        """Bind all knowledge channels, including rejected investigations, to their bytes."""
        return sha256_json(self)

    def behavioral_context(self) -> JsonObject:
        """Render behavior without promoting research conclusions or historical values."""
        context: JsonObject = {
            "source_trace_ids": list(self.source_trace_ids),
            "rules": self.rules,
            "contracts": self.contracts,
            "unknowns": list(self.unknowns),
            "prediction_guidance": self.prediction_guidance,
            "transfer_context": {
                **{
                    key: self.transfer_context[key]
                    for key in ("library_id", "query_sha256", "authority")
                    if key in self.transfer_context
                },
                "selected": prior_records(self.transfer_context, ("rule", "skill")),
            }
            if self.transfer_context
            else {},
        }

        if self.claims:
            context["claims"] = [claim.model_dump(mode="json") for claim in self.claims]
        return context

    def original_library_evidence(self) -> list[JsonObject]:
        """Render prior originals in the evidence channel, separate from generated rules."""
        return prior_records(self.transfer_context, ("trajectory",))

    def research_context(self) -> JsonObject:
        """Keep target and prior investigations labelled, including rejected attempts."""
        return {
            "target": [note.model_dump(mode="json") for note in self.research_history],
            "prior_library": prior_records(self.transfer_context, ("research",)),
        }
