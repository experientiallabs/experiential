"""Admitted construction and durable receipts for the cascade's cheap next-action deciders.

A decider's provider attempts share the caller's admitted transport, physical-attempt ledger and
deadline. Every decision writes its exact request as an intent before dispatch and its answer or
failure afterwards, so an interrupted decision stays unknown and is never redrawn.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Self

from pydantic import model_validator

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, ContractModel
from exp.common.models import ModelCatalog, ModelRequest
from exp.common.project import ArtifactStore, artifact_input
from exp.optimize.evaluation.fidelity.decision_access import decision_caller
from exp.runtime.agents.cascade import DeciderDefinition, NextActionDecider, NextActionDecision
from exp.runtime.agents.cascade_deciders import SampledDraftDecider, SystemOneDecider
from exp.runtime.models.call_evidence import call_consumer, canonical_call
from exp.runtime.models.providers.admitted_transport import (
    AdmittedJsonTransport,
    require_admitted_transport,
)
from exp.runtime.models.registry import RuntimeModelCatalog


def preflight_decider(catalog: ModelCatalog, definition: DeciderDefinition) -> None:
    """Check identity, admitted transport and decider capability without credentials.

    Raises:
        ValueError: The alias resolves to another model, sends requests outside the admitted
            transport, or cannot serve the declared decider kind.
    """
    model, capabilities = RuntimeModelCatalog(catalog, environment={}).snapshot(definition.alias)
    if model != definition.model:
        raise ValueError(
            "cascade decider alias differs from its frozen snapshot; refit with the current model"
        )
    provider = catalog.connections[catalog.models[definition.alias].connection].provider
    if definition.kind == "systemone":
        # Decisions are posted through the admitted transport by ``decision_caller`` itself.
        if provider != "typesafe":
            raise ValueError("a systemone cascade decider needs a TypeSafe decision alias")
        return
    require_admitted_transport(catalog, definition.alias, consumer="cascade decider")
    if capabilities.supports_tools is False or (
        capabilities.maximum_output_tokens is not None
        and definition.maximum_output_tokens > capabilities.maximum_output_tokens
    ):
        raise ValueError(
            "a sampled_drafts cascade decider needs a tool-capable chat model whose output "
            "capacity covers the draft allowance"
        )


def admitted_decider(
    catalog: RuntimeModelCatalog,
    definition: DeciderDefinition,
    transport: AdmittedJsonTransport,
    deadline: datetime,
) -> NextActionDecider:
    """Resolve the frozen decider through the caller's admitted catalog and transport.

    Raises:
        ValueError: The resolved model differs from the frozen snapshot.
    """
    resolved = catalog.resolve(definition.alias)
    if resolved.snapshot != definition.model:
        raise ValueError("cascade decider alias differs from its frozen snapshot")
    if definition.kind == "systemone":
        return SystemOneDecider(
            definition, decision_caller(resolved, transport, deadline, lambda: datetime.now(UTC))
        )
    return SampledDraftDecider(definition, resolved.client)


class DecisionReceipt(ContractModel):
    """The saved answer to one decision intent, or its sanitized failure.

    Attributes:
        decision: Decider answer, when one was returned.
        failure: Exception type or ``UnfinishedDispatchIntent``, when none was.
    """

    decision: NextActionDecision | None = None
    failure: str | None = None

    @model_validator(mode="after")
    def exactly_one(self) -> Self:
        """Require either an answer or a failure."""
        if (self.decision is None) == (self.failure is None):
            raise ValueError("a decision receipt holds an answer or a failure")
        return self


def saved_decision(store: ArtifactStore, key: str) -> DecisionReceipt | None:
    """Return a saved receipt, an unknown outcome for an orphaned intent, or None if unstarted."""
    if store.exists(key):
        return DecisionReceipt.model_validate_json(store.read_bytes(key, "decision.json"))
    if store.exists(key + "-intent"):
        return DecisionReceipt(failure="UnfinishedDispatchIntent")
    return None


class RecordedDecider:
    """Persist each decision's intent before dispatch and its receipt afterwards."""

    def __init__(
        self,
        decider: NextActionDecider,
        *,
        store: ArtifactStore,
        prefix: str,
        inputs: tuple[ArtifactInput, ...],
        code_revision: str,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        """Bind the inner decider, the receipt prefix and the provenance of every receipt."""
        self._decider, self._store, self._prefix = decider, store, prefix
        self._inputs, self._revision, self._clock = inputs, code_revision, clock
        self._count = 0

    @property
    def definition(self) -> DeciderDefinition:
        """Return the inner decider's frozen definition."""
        return self._decider.definition

    def decide(self, request: ModelRequest, candidates: tuple[str, ...]) -> NextActionDecision:
        """Decide under the next sequential receipt key."""
        key = f"{self._prefix}-decision-{self._count}"
        self._count += 1
        return self.decide_at(key, request, candidates)

    def decide_at(
        self, key: str, request: ModelRequest, candidates: tuple[str, ...]
    ) -> NextActionDecision:
        """Write the intent, dispatch once, then retain the answer or the failure.

        Raises:
            ValueError: The key already holds an intent; a saved slot is never redrawn.
        """
        if self._store.exists(key + "-intent"):
            raise ValueError("decision slot already dispatched; read its saved receipt")
        envelope = ArtifactEnvelope(
            schema_version=1,
            created_at=self._clock(),
            code_revision=self._revision,
            inputs=tuple(sorted(self._inputs, key=lambda row: row.artifact_id)),
        )
        intent = artifact_input(
            self._store.write_json(
                artifact_id=key + "-intent",
                artifact_type="cascade-decision-intent",
                envelope=envelope,
                files={
                    "request.json": request,
                    "decider.json": self._decider.definition,
                    "candidates.json": list(candidates),
                },
            )
        )
        try:
            with canonical_call(intent), call_consumer("actor"):
                decision = self._decider.decide(request, candidates)
        except (ValueError, RuntimeError, OSError) as error:
            self._receipt(key, intent, DecisionReceipt(failure=type(error).__name__))
            raise
        self._receipt(key, intent, DecisionReceipt(decision=decision))
        return decision

    def _receipt(self, key: str, intent: ArtifactInput, receipt: DecisionReceipt) -> None:
        """Publish the immutable receipt bound to its intent."""
        self._store.write_json(
            artifact_id=key,
            artifact_type="cascade-decision",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=self._clock(),
                code_revision=self._revision,
                inputs=(intent,),
            ),
            files={"decision.json": receipt},
        )
