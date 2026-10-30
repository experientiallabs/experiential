"""Trace-only actor sensitivity probes with a repeated-reference sampling control.

No environment executes here. A fresh actor decision is sampled after a completed tool turn,
changing only one observation. The repeated reference measures ordinary sampling variation.
Exact decision agreement is descriptive, never a semantic-equivalence or correctness verdict.
"""

from collections import Counter
from datetime import UTC, datetime
from typing import Literal, Self

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import AwareDatetime, Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.core.json_schema import expand_local_references
from exp.common.core.locks import CoordinationLock
from exp.common.models import ModelRequest, ModelResponse, ModelSnapshot, load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.runtime.models.call_evidence import call_consumer, canonical_call
from exp.runtime.models.providers.admitted_transport import (
    AdmittedJsonTransport,
    require_admitted_transport,
)
from exp.runtime.models.providers.transport import HttpxJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.runtime.sessions.dispatch_ledger import DispatchLedger

Arm = Literal["reference", "reference_repeat", "simulated"]
_ARMS: tuple[Arm, ...] = ("reference", "reference_repeat", "simulated")
_FAILURE_GUIDANCE = {
    "CatalogChanged": "Inspect the frozen catalog; configuration changes require a new study.",
    "ModelChanged": "Restore the frozen actor model; a different model requires a new study.",
    "ConfigurationUnavailable": "Check catalog and credentials; saved draws stay fixed.",
    "TransportSetupFailed": "Check local transport and ledger access; no automatic redraw.",
    "ProviderFailure": "Inspect saved dispatch receipts; raw provider details withheld.",
    "UnfinishedDispatchIntent": "Saved intent lacks a response; outcome unknown, no redraw.",
    "DeadlineBeforeDispatch": "Deadline elapsed before dispatch; this slot was not purchased.",
    "IncompleteResponse": "Output truncated; a different token allowance needs a new study.",
    "UnavailableTool": "Undeclared tool; inspect the published interface and saved response.",
    "InvalidToolArguments": "Arguments violate the schema; inspect the saved response.",
    "ModelMismatch": "Response model differs from the frozen snapshot; inspect model resolution.",
}


class ActorDecisionCase(ContractModel):
    """Caller-admitted observed prefix and one saved simulator reply.

    Attributes:
        name: Unique diagnostic case label, not an exposure or lineage identity.
        request: Full public actor prefix ending at a completed tool turn, with a token bound.
        observation: Index of the tool reply replaced in the simulated arm.
        simulated_reply: Candidate reply, never disclosed to the reference arms.
        sources: Exact retained evidence dependencies. Their contents are not sent to the actor.
        admission: How the caller reconstructed the prefix and excluded future/private evidence.
            This declaration is not an independently checked completeness or independence claim.
    """

    name: str = Field(min_length=1, max_length=200)
    request: ModelRequest
    observation: int = Field(ge=0, strict=True)
    simulated_reply: str = Field(max_length=262144)
    sources: tuple[ArtifactInput, ...] = Field(min_length=1)
    admission: str = Field(min_length=1, max_length=4000)

    @model_validator(mode="after")
    def complete_turn(self) -> Self:
        """Require complete unambiguous call/reply pairing and refuse lossy media persistence."""
        messages = self.request.messages
        if self.observation >= len(messages) or messages[self.observation].role != "tool":
            raise ValueError("actor probe observation must select a tool reply")
        if messages[-1].role != "tool" or self.request.maximum_output_tokens is None:
            raise ValueError("actor probe needs a completed tool turn and an output token bound")
        pending: set[str] = set()
        seen: set[str] = set()
        for message in messages:
            if message.content_parts:
                raise ValueError("actor probes require text-only canonical prefixes; media refused")
            if message.role == "tool":
                if message.tool_call_id not in pending:
                    raise ValueError("actor probe has an ambiguous or unmatched tool reply")
                pending.remove(message.tool_call_id)
            else:
                if pending:
                    raise ValueError("actor probe has an unfinished parallel tool turn")
                if message.assistant_action is not None:
                    for call in message.assistant_action.tool_calls:
                        if call.call_id in seen:
                            raise ValueError("actor probe has duplicate call IDs")
                        seen.add(call.call_id)
                        pending.add(call.call_id)
        if pending:
            raise ValueError("actor probe has an unfinished parallel tool turn")
        # Earlier interventions would leave subsequent recorded decisions in the context.
        if any(message.role != "tool" for message in messages[self.observation :]):
            raise ValueError("actor probe may substitute only the final completed tool turn")
        return self

    def for_arm(self, arm: Arm) -> ModelRequest:
        """Change only one reply; retain every instruction, tool, call ID and sampling setting."""
        if arm != "simulated":
            return self.request
        messages = list(self.request.messages)
        messages[self.observation] = messages[self.observation].model_copy(
            update={"content": self.simulated_reply}
        )
        return self.request.model_copy(update={"messages": tuple(messages)})


class ActorDecisionSpec(ContractModel):
    """Finite registered actor panel over caller-admitted trace prefixes.

    Attributes:
        cases: Frozen population; failures remain in the report.
        actor_alias: Configured actor selected for this experiment.
        actor_model: Exact model snapshot, checked on every response.
        catalog_sha256: Frozen credential-free provider configuration.
        repeats: Independent triples per case; three arm orders rotate across triples.
        maximum_input_bytes: Complete canonical request limit, with no truncation.
        deadline: Aware deadline shared by every physical dispatch.
    """

    cases: tuple[ActorDecisionCase, ...] = Field(min_length=1, max_length=100)
    actor_alias: str = Field(min_length=1)
    actor_model: ModelSnapshot
    catalog_sha256: Sha256
    repeats: int = Field(default=2, ge=1, le=25, strict=True)
    maximum_input_bytes: int = Field(default=128000, ge=512, le=512000, strict=True)
    deadline: AwareDatetime

    @model_validator(mode="after")
    def distinct(self) -> Self:
        """Refuse duplicate labels or lossy serialization before deriving the run identity."""
        if len({case.name for case in self.cases}) != len(self.cases):
            raise ValueError("actor probe case names must be unique")
        return self

    def identity(self, revision: str) -> str:
        """Freeze population, requests, controls, provider configuration and implementation."""
        return stable_id("actor-decisions", {"spec": sha256_json(self), "code": revision})


class ActorDecisionCell(ContractModel):
    """One immutable logical draw, including malformed or lost results.

    Attributes:
        case: Registered case index.
        repeat: Registered replicate index.
        arm: Observation condition, never supplied as an actor instruction.
        response: Original parsed model response when saved, including truncated outputs.
        failure: Sanitized failure or unknown intent; never silently retried.
    """

    case: int
    repeat: int
    arm: Arm
    response: ModelResponse | None = None
    failure: str | None = None


class ActorDecisionReport(ContractModel):
    """All raw next decisions, without pretending to execute their proposed tool calls.

    Attributes:
        spec: Registered inputs and limits.
        cells: Every planned draw, including unknowns.
        scope: Fixed interpretation and information limits.
    """

    spec: ActorDecisionSpec
    cells: tuple[ActorDecisionCell, ...]
    scope: Literal[
        "development next-decision sensitivity; caller-admitted prefixes; "
        "exact agreement is not semantic equivalence, correctness or full-episode fidelity; "
        "no tools executed; no initial workbook required"
    ] = (
        "development next-decision sensitivity; caller-admitted prefixes; "
        "exact agreement is not semantic equivalence, correctness or full-episode fidelity; "
        "no tools executed; no initial workbook required"
    )

    @model_validator(mode="after")
    def complete_inventory(self) -> Self:
        """Require each registered draw once; missing responses never count as agreement."""
        if len(self.cells) != len(_planned(self.spec)):
            raise ValueError("actor probe report needs every registered draw")
        for cell, planned in zip(self.cells, _planned(self.spec), strict=True):
            if cell.model_copy(update={"response": None, "failure": None}) != planned:
                raise ValueError("actor probe report changed its registered draw order")
            if cell.response is None:
                if not cell.failure:
                    raise ValueError("missing actor decision requires an explicit failure")
            elif cell.failure != _failure(cell.response, self.spec, self.spec.cases[cell.case]):
                raise ValueError("actor decision status differs from its saved response")
        return self

    def metrics(self) -> JsonObject:
        """Compare the simulation treatment against the repeated-reference noise control."""
        output: JsonObject = {
            "scope": self.scope,
            "decision_projection": (
                "ordered tool names and parsed arguments, ignoring generated IDs and accompanying "
                "prose; exact final text when no tools are called; "
                "agreement can include shared errors"
            ),
            "interface_diagnostics": actor_decision_interfaces(self.spec),
            "failures": [
                {
                    "code": code,
                    "count": count,
                    "guidance": _FAILURE_GUIDANCE.get(
                        code, "Inspect the saved draw and dispatch receipts."
                    ),
                    "draws": [index for index, row in enumerate(self.cells) if row.failure == code],
                }
                for code, count in sorted(
                    Counter(row.failure for row in self.cells if row.failure).items()
                )
            ],
            "empty_argument_tool_calls": {
                arm: sum(
                    not call.arguments
                    for cell in self.cells
                    if cell.arm == arm and cell.response is not None
                    for call in cell.response.output.tool_calls
                )
                for arm in _ARMS
            },
        }
        for other in ("reference_repeat", "simulated"):
            comparisons: list[bool | None] = []
            for case in range(len(self.spec.cases)):
                for repeat in range(self.spec.repeats):
                    rows = {
                        row.arm: row
                        for row in self.cells
                        if row.case == case and row.repeat == repeat
                    }
                    left, right = rows["reference"], rows[other]
                    comparisons.append(
                        None
                        if left.failure or right.failure
                        else _decision(left.response) == _decision(right.response)
                    )
            known = [value for value in comparisons if value is not None]
            matches, unknown = sum(known), len(comparisons) - len(known)
            output[other] = {
                "planned": len(comparisons),
                "assessable": len(known),
                "exact_matches": matches,
                "unknown": unknown,
                "exact_agreement_bounds": [
                    matches / len(comparisons),
                    (matches + unknown) / len(comparisons),
                ],
            }
        output["bound_meaning"] = "missing-outcome bounds, not confidence intervals"
        return output


def _decision(response: ModelResponse | None) -> str:
    """Compare ordered names and parsed arguments or final text; ignore generated call IDs."""
    assert response is not None
    action = response.output
    return sha256_json(
        {
            "content": action.content if not action.tool_calls else None,
            "calls": [
                {"name": call.name, "arguments": call.arguments} for call in action.tool_calls
            ],
        }
    )


def _interface_gaps(case: ActorDecisionCase) -> JsonObject:
    """Identify unnamed observed arguments without inferring required fields or invalidity."""
    schemas = {tool.name: expand_local_references(tool.input_schema) for tool in case.request.tools}
    gaps: dict[str, set[str]] = {}
    for message in case.request.messages:
        if message.assistant_action is None:
            continue
        for call in message.assistant_action.tool_calls:
            properties = schemas.get(call.name, {}).get("properties", {})
            named = set(properties) if isinstance(properties, dict) else set()
            missing = set(call.arguments) - named
            if missing:
                gaps.setdefault(call.name, set()).update(missing)
    return {name: sorted(keys) for name, keys in sorted(gaps.items())}


def actor_decision_interfaces(spec: ActorDecisionSpec) -> list[JsonObject]:
    """Expose recorded argument names without named schema properties before any paid draw."""
    return [
        {"case": case.name, "observed_arguments_without_named_properties": _interface_gaps(case)}
        for case in spec.cases
    ]


def _planned(spec: ActorDecisionSpec) -> tuple[ActorDecisionCell, ...]:
    """Rotate arm dispatch order without changing the registered population."""
    return tuple(
        ActorDecisionCell(case=case, repeat=repeat, arm=_ARMS[(offset + case + repeat) % 3])
        for case in range(len(spec.cases))
        for repeat in range(spec.repeats)
        for offset in range(3)
    )


def _sources(project: ProjectStore, spec: ActorDecisionSpec) -> tuple[ArtifactInput, ...]:
    """Verify all declared source digests without exposing source bodies to actors."""
    refs = {source for case in spec.cases for source in case.sources}
    for source in refs:
        if artifact_input(project.artifacts.read(source.artifact_id).manifest) != source:
            raise ValueError("actor probe evidence source changed")
    return tuple(sorted(refs, key=lambda row: row.artifact_id))


def preflight_actor_decisions(project: ProjectStore, spec: ActorDecisionSpec) -> int:
    """Validate complete requests and provider capacity before credentials or dispatch."""
    _sources(project, spec)
    catalog = load_model_catalog(project.model_catalog_path)
    if sha256_json(catalog.model_dump(mode="json")) != spec.catalog_sha256:
        raise ValueError("actor probe catalog changed; register its exact configuration")
    model, capability = RuntimeModelCatalog(catalog, environment={}).snapshot(spec.actor_alias)
    if model != spec.actor_model:
        raise ValueError("actor probe model differs from its frozen snapshot")
    require_admitted_transport(catalog, spec.actor_alias, consumer="actor probe")
    for case in spec.cases:
        for tool in case.request.tools:
            try:
                Draft202012Validator.check_schema(expand_local_references(tool.input_schema))
            except SchemaError as error:
                raise ValueError("actor probe has an invalid tool schema") from error
        for arm in _ARMS:
            request = case.for_arm(arm)
            size = len(request.model_dump_json().encode())
            tokens = request.maximum_output_tokens
            assert tokens is not None
            if size > spec.maximum_input_bytes:
                raise ValueError("actor probe exceeds complete request byte allowance")
            if request.tools and capability.supports_tools is not True:
                raise ValueError("actor probe model must support its published tools")
            if (
                capability.maximum_output_tokens is not None
                and tokens > capability.maximum_output_tokens
            ):
                raise ValueError("actor probe output exceeds model capacity")
            if capability.context_window_tokens is not None and (
                size + 4 * len(request.messages) + tokens > capability.context_window_tokens
            ):
                raise ValueError("actor probe complete input and output exceed model capacity")
    return len(_planned(spec))


def _failure(
    response: ModelResponse, spec: ActorDecisionSpec, case: ActorDecisionCase
) -> str | None:
    """Truncated, foreign-model and unavailable-tool actions cannot count as agreement."""
    if response.model != spec.actor_model:
        return "ModelMismatch"
    if response.finish_reason.value != "completed":
        return "IncompleteResponse"
    tools = {tool.name: tool for tool in case.request.tools}
    if any(call.name not in tools for call in response.output.tool_calls):
        return "UnavailableTool"
    if any(
        not Draft202012Validator(expand_local_references(tools[call.name].input_schema)).is_valid(
            call.arguments
        )
        for call in response.output.tool_calls
    ):
        return "InvalidToolArguments"
    return None


def saved_actor_decisions(
    project: ProjectStore, spec: ActorDecisionSpec, *, code_revision: str
) -> ActorDecisionReport | None:
    """Reconstruct saved results and every exact dependency without provider configuration."""
    store, identity = project.artifacts, spec.identity(code_revision)
    if not store.exists(identity + "-result"):
        return None
    report = ActorDecisionReport.model_validate_json(
        store.read_bytes(identity + "-result", "report.json")
    )
    registration = store.read(identity).manifest
    if (
        report.spec != spec
        or registration.inputs != _sources(project, spec)
        or ActorDecisionSpec.model_validate_json(store.read_bytes(identity, "spec.json")) != spec
        or len(report.cells) != len(_planned(spec))
    ):
        raise ValueError("actor probe report differs from registration")
    inputs = {artifact_input(registration)}
    for index, (cell, planned) in enumerate(zip(report.cells, _planned(spec), strict=True)):
        manifest = store.read(f"{identity}-cell-{index}").manifest
        suffix = "stopped" if cell.failure == "DeadlineBeforeDispatch" else "intent"
        intent = store.read(f"{identity}-cell-{index}-{suffix}").manifest
        if (
            cell.model_copy(update={"response": None, "failure": None}) != planned
            or ActorDecisionCell.model_validate_json(
                store.read_bytes(manifest.artifact_id, "cell.json")
            )
            != cell
            or manifest.inputs != (artifact_input(intent),)
            or intent.inputs != (artifact_input(registration),)
            or ModelRequest.model_validate_json(
                store.read_bytes(intent.artifact_id, "request.json")
            )
            != spec.cases[cell.case].for_arm(cell.arm)
            or (cell.response is None and not cell.failure)
            or (
                cell.response is not None
                and cell.failure != _failure(cell.response, spec, spec.cases[cell.case])
            )
        ):
            raise ValueError("actor probe cell differs from its registered draw")
        inputs.add(artifact_input(manifest))
    if set(store.read(identity + "-result").manifest.inputs) != inputs:
        raise ValueError("actor probe report lost a planned cell")
    return report


def actor_decisions_need_dispatch(
    project: ProjectStore, spec: ActorDecisionSpec, *, code_revision: str
) -> bool:
    """Separate new inference from credential-free recovery of saved or deadline-stopped draws."""
    if datetime.now(UTC) >= spec.deadline:
        return False
    identity = spec.identity(code_revision)
    return any(
        not project.artifacts.exists(f"{identity}-cell-{index}")
        and not project.artifacts.exists(f"{identity}-cell-{index}-intent")
        and not project.artifacts.exists(f"{identity}-cell-{index}-stopped")
        for index in range(len(_planned(spec)))
    )


def run_actor_decisions(
    project: ProjectStore, spec: ActorDecisionSpec, *, code_revision: str
) -> ActorDecisionReport:
    """Sample each next action once; interrupted intents stay unknown and never redraw."""
    if any(message.content_parts for case in spec.cases for message in case.request.messages):
        raise ValueError("actor probes require text-only canonical prefixes; media refused")
    spec = ActorDecisionSpec.model_validate_json(spec.model_dump_json())
    store, identity = project.artifacts, spec.identity(code_revision)
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        saved = saved_actor_decisions(project, spec, code_revision=code_revision)
        if saved is not None:
            return saved
        if actor_decisions_need_dispatch(project, spec, code_revision=code_revision):
            preflight_actor_decisions(project, spec)
        if datetime.now(UTC) >= spec.deadline and not store.exists(identity):
            raise ValueError("actor probe deadline expired; completed reports can still replay")
        envelope = ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            inputs=_sources(project, spec),
        )
        if not store.exists(identity):
            store.write_json(
                artifact_id=identity,
                artifact_type="actor-decisions-registration",
                envelope=envelope,
                files={"spec.json": spec},
            )
        elif (
            ActorDecisionSpec.model_validate_json(store.read_bytes(identity, "spec.json")) != spec
            or store.read(identity).manifest.inputs != envelope.inputs
        ):
            raise ValueError("actor probe registration changed; inspect its immutable evidence")
        registration = artifact_input(store.read(identity).manifest)
        cells, inputs = [], {registration}
        for index, planned in enumerate(_planned(spec)):
            key = f"{identity}-cell-{index}"
            if store.exists(key):
                cell = ActorDecisionCell.model_validate_json(store.read_bytes(key, "cell.json"))
            else:
                request = spec.cases[planned.case].for_arm(planned.arm)
                if store.exists(key + "-intent"):
                    cell = planned.model_copy(update={"failure": "UnfinishedDispatchIntent"})
                elif store.exists(key + "-stopped") or datetime.now(UTC) >= spec.deadline:
                    if not store.exists(key + "-stopped"):
                        store.write_json(
                            artifact_id=key + "-stopped",
                            artifact_type="actor-decision-not-dispatched",
                            envelope=envelope.model_copy(update={"inputs": (registration,)}),
                            files={"request.json": request},
                        )
                    cell = planned.model_copy(update={"failure": "DeadlineBeforeDispatch"})
                else:
                    store.write_json(
                        artifact_id=key + "-intent",
                        artifact_type="actor-decision-intent",
                        envelope=envelope.model_copy(update={"inputs": (registration,)}),
                        files={"request.json": request},
                    )
                    intent = artifact_input(store.read(key + "-intent").manifest)
                    failure_code = "ConfigurationUnavailable"
                    try:
                        definition = load_model_catalog(project.model_catalog_path)
                        if sha256_json(definition.model_dump(mode="json")) != spec.catalog_sha256:
                            failure_code = "CatalogChanged"
                            raise ValueError("actor probe catalog changed before dispatch")
                        failure_code = "TransportSetupFailed"
                        # One physical attempt per draw prevents provider retry rerolls.
                        transport = AdmittedJsonTransport(
                            HttpxJsonTransport(),
                            ledger=DispatchLedger(
                                store.project_directory / (key + ".sqlite"), 1, sha256_json(key)
                            ),
                            artifacts=store,
                            code_revision=code_revision,
                            deadline=spec.deadline,
                        )
                        failure_code = "ConfigurationUnavailable"
                        catalog = RuntimeModelCatalog(
                            definition,
                            transport_factory=lambda transport=transport: transport,
                        )
                        model = catalog.resolve(spec.actor_alias)
                        if model.snapshot != spec.actor_model:
                            failure_code = "ModelChanged"
                            raise ValueError("actor probe model changed before dispatch")
                        failure_code = "ProviderFailure"
                        with canonical_call(intent), call_consumer("actor"):
                            response = model.client.complete(request)
                        cell = planned.model_copy(
                            update={
                                "response": response,
                                "failure": _failure(response, spec, spec.cases[planned.case]),
                            }
                        )
                    except (ValueError, RuntimeError, OSError):
                        cell = planned.model_copy(update={"failure": failure_code})
                suffix = "stopped" if cell.failure == "DeadlineBeforeDispatch" else "intent"
                intent = artifact_input(store.read(key + "-" + suffix).manifest)
                store.write_json(
                    artifact_id=key,
                    artifact_type="actor-decision-cell",
                    envelope=envelope.model_copy(update={"inputs": (intent,)}),
                    files={"cell.json": cell},
                )
            cells.append(cell)
            inputs.add(artifact_input(store.read(key).manifest))
        report = ActorDecisionReport(spec=spec, cells=tuple(cells))
        store.write_json(
            artifact_id=identity + "-result",
            artifact_type="actor-decisions-report",
            envelope=envelope.model_copy(
                update={"inputs": tuple(sorted(inputs, key=lambda ref: ref.artifact_id))}
            ),
            files={"report.json": report},
        )
        verified = saved_actor_decisions(project, spec, code_revision=code_revision)
        assert verified is not None
        return verified
