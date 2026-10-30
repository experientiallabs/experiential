"""Opt-in bounded memory diagnostics through the existing trace-analysis and call owners."""

import json
from datetime import UTC, datetime

from pydantic import AwareDatetime, Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.models import ModelSnapshot, load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.analysis.memory_evidence import (
    MemoryInspection,
    inspect_memory,
    located_value,
)
from exp.optimize.evaluation.analysis.source_access import require_development_sources
from exp.runtime.models.budget import RequestBudget
from exp.runtime.models.call_evidence import call_consumer
from exp.runtime.models.providers.admitted_transport import (
    AdmittedJsonTransport,
    require_admitted_transport,
)
from exp.runtime.models.providers.transport import HttpxJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.runtime.models.structured_completion import CompletionPlan, StructuredCompletion
from exp.runtime.sessions.dispatch_ledger import DispatchLedger
from exp.simulation.evaluation.analysis import (
    ANALYSIS_PROMPT,
    MEMORY_ANALYSIS_GUIDANCE,
    TraceAnalysis,
    analysis_request,
    analyze_traces,
)


class MemoryAnalystSpec(ContractModel):
    """Frozen diagnostic request with bounded spending and no score or learning authority.

    Attributes:
        inspection: Explicit canonical-call inventory, including failed calls.
        exposure_sha256: Frozen passive report; later receipt arrivals require a new registration.
        analyst_alias: Configured model alias used only for analysis.
        analyst_model: Exact model/profile snapshot.
        catalog_sha256: Frozen credential-free catalog definition.
        output_tokens: Maximum generated tokens for the one logical diagnostic request.
        maximum_input_bytes: Whole diagnostic input allowance; no silent truncation.
        maximum_physical_attempts: Total admitted HTTP attempts, including retries.
        deadline: Latest dispatch/response boundary.
    """

    inspection: MemoryInspection
    exposure_sha256: Sha256
    analyst_alias: str = Field(min_length=1)
    analyst_model: ModelSnapshot
    catalog_sha256: Sha256
    output_tokens: int = Field(default=4096, ge=1, le=16384)
    maximum_input_bytes: int = Field(default=128000, ge=512, le=512000)
    maximum_physical_attempts: int = Field(default=2, ge=1, le=5)
    deadline: AwareDatetime

    def identity(self, revision: str) -> str:
        """Bind the complete frozen request contract and the exact analyst rubric."""
        return stable_id(
            "memory-analysis",
            {
                "spec": sha256_json(self),
                "code": revision,
                "source_protocol": "complete-input-and-output-receipts-v1",
                "rubric": sha256_json(
                    {
                        "base": ANALYSIS_PROMPT,
                        "memory": MEMORY_ANALYSIS_GUIDANCE,
                        "schema": TraceAnalysis.model_json_schema(),
                    }
                ),
            },
        )


def diagnostic_definition(project: ProjectStore, spec: MemoryAnalystSpec) -> JsonObject:
    """Resolve verified originals for analyst reading, with role and dispatch limits explicit."""
    report = inspect_memory(project.artifacts, spec.inspection)
    if sha256_json(report) != spec.exposure_sha256:
        raise ValueError("memory exposure changed since analyst registration")
    require_development_sources(
        project.artifacts, report.sources(), maximum_artifacts=spec.inspection.maximum_artifacts
    )
    inventory: JsonObject = {}
    for call in report.calls:
        root = "memory:" + call.intent.artifact_id
        inventory[root] = {
            "call": call.intent.artifact_id,
            "kind": "coverage",
            "consumer": call.consumer,
            "coverage": call.coverage,
            "attempts": [row.model_dump(mode="json") for row in call.attempts],
        }
        for index, fragment in enumerate(call.fragments):
            inventory[root + f":input:{index}"] = {
                "call": call.intent.artifact_id,
                "consumer": call.consumer,
                "kind": "input",
                "channel": fragment.channel,
                "location": fragment.location.model_dump(mode="json"),
                "value": located_value(project.artifacts, fragment.location),
            }
        for attempt_index, attempt in enumerate(call.attempts):
            for evidence_index, evidence in enumerate(attempt.reasoning):
                if evidence.availability not in {"exposed_text", "provider_summary"}:
                    continue
                for index, location in enumerate(evidence.locations):
                    inventory[root + f":output:{attempt_index}:{evidence_index}:{index}"] = {
                        "call": call.intent.artifact_id,
                        "consumer": call.consumer,
                        "kind": "exposed_text",
                        "availability": evidence.availability,
                        "location": location.model_dump(mode="json"),
                        "value": located_value(project.artifacts, location),
                    }
    definition: JsonObject = {
        "memory_diagnostics": inventory,
        "limitations": [
            report.scope,
            "uncalibrated diagnostic hypotheses; ordinary task scores unchanged",
        ],
    }
    body, _ = analysis_request(definition, ())
    if len(json.dumps(body, ensure_ascii=False).encode()) > spec.maximum_input_bytes:
        raise ValueError("complete memory diagnostic exceeds its byte allowance")
    return definition


def preflight_memory_analyst(project: ProjectStore, spec: MemoryAnalystSpec) -> JsonObject:
    """Validate sources and model identity before credentials or paid execution."""
    definition = diagnostic_definition(project, spec)
    catalog = load_model_catalog(project.model_catalog_path)
    if sha256_json(catalog.model_dump(mode="json")) != spec.catalog_sha256:
        raise ValueError("memory analyst catalog changed")
    snapshot, capabilities = RuntimeModelCatalog(catalog, environment={}).snapshot(
        spec.analyst_alias
    )
    if snapshot != spec.analyst_model or capabilities.supports_tools is not True:
        raise ValueError("memory analyst requires the frozen tool-capable model")
    body, _ = analysis_request(definition, ())
    required = len(json.dumps(body, ensure_ascii=False).encode()) + spec.output_tokens + 1024
    if capabilities.context_window_tokens is None or required > capabilities.context_window_tokens:
        raise ValueError("memory analyst complete request exceeds declared context capacity")
    if (
        capabilities.maximum_output_tokens is not None
        and spec.output_tokens > capabilities.maximum_output_tokens
    ):
        raise ValueError("memory analyst output exceeds declared capacity")
    require_admitted_transport(catalog, spec.analyst_alias, consumer="memory analyst")
    return definition


def _analysis_sources(
    project: ProjectStore, spec: MemoryAnalystSpec, identity: str
) -> tuple[ArtifactInput, ...]:
    """Bind every inspected original and the exact analyst request into the saved graph."""
    report = inspect_memory(project.artifacts, spec.inspection)
    if sha256_json(report) != spec.exposure_sha256:
        raise ValueError("memory exposure changed since analyst registration")
    canonical = tuple(
        artifact_input(project.artifacts.read(call).manifest)
        for call in project.artifacts.list_ids(artifact_type="structured-call-intent")
        if json.loads(project.artifacts.read_bytes(call, "request.json")).get("logical_id")
        == identity
    )
    if len(canonical) != 1:
        raise ValueError("memory analysis lacks its exact canonical analyst call")
    declarations = require_development_sources(
        project.artifacts, report.sources(), maximum_artifacts=spec.inspection.maximum_artifacts
    )
    return tuple(
        sorted({*report.sources(), *canonical, *declarations}, key=lambda ref: ref.artifact_id)
    )


def saved_memory_analysis(
    project: ProjectStore, spec: MemoryAnalystSpec, *, code_revision: str
) -> TraceAnalysis | None:
    """Replay a completed diagnostic without resolving credentials or calling a provider."""
    identity = spec.identity(code_revision)
    if not project.artifacts.exists(identity):
        return None
    diagnostic_definition(project, spec)
    if (
        MemoryAnalystSpec.model_validate_json(project.artifacts.read_bytes(identity, "spec.json"))
        != spec
    ):
        raise ValueError("saved memory analyst contract differs")
    manifest = project.artifacts.read(identity).manifest
    if (
        manifest.artifact_type != "memory-diagnostic-analysis"
        or manifest.code_revision != code_revision
    ):
        raise ValueError("saved memory analyst identity differs")
    if set(manifest.inputs) != set(_analysis_sources(project, spec, identity)):
        raise ValueError("memory analysis lost its original call evidence")
    for reference in manifest.inputs:
        if artifact_input(project.artifacts.read(reference.artifact_id).manifest) != reference:
            raise ValueError("memory analyst dependency differs")
    return TraceAnalysis.model_validate_json(
        project.artifacts.read_bytes(identity, "analysis.json")
    )


def run_memory_analysis(
    project: ProjectStore, spec: MemoryAnalystSpec, *, code_revision: str
) -> TraceAnalysis:
    """Execute one metered diagnostic under the same lease as source classification.

    Admission and dispatch linearize before any later protection declaration. The lease
    prevents concurrent classification changes while approved content is being disclosed.
    """
    with CoordinationLock(project.artifacts.project_directory / "source-access.lock", timeout=0):
        return _run_admitted_memory_analysis(project, spec, code_revision=code_revision)


def _run_admitted_memory_analysis(
    project: ProjectStore, spec: MemoryAnalystSpec, *, code_revision: str
) -> TraceAnalysis:
    """Run or replay a diagnostic while its source classification lease is held."""
    saved = saved_memory_analysis(project, spec, code_revision=code_revision)
    if saved is not None:
        return saved
    definition = preflight_memory_analyst(project, spec)
    identity = spec.identity(code_revision)
    directory = project.artifacts.project_directory
    with CoordinationLock(directory / (identity + ".lock"), timeout=0):
        saved = saved_memory_analysis(project, spec, code_revision=code_revision)
        if saved is not None:
            return saved
        transport = AdmittedJsonTransport(
            HttpxJsonTransport(),
            ledger=DispatchLedger(
                directory / (identity + ".sqlite"),
                spec.maximum_physical_attempts,
                sha256_json(spec),
            ),
            artifacts=project.artifacts,
            code_revision=code_revision,
            deadline=spec.deadline,
        )
        catalog = load_model_catalog(project.model_catalog_path)
        if sha256_json(catalog.model_dump(mode="json")) != spec.catalog_sha256:
            raise ValueError("memory analyst catalog changed before dispatch")
        registry = RuntimeModelCatalog(catalog, transport_factory=lambda: transport)
        model = registry.resolve(spec.analyst_alias)
        if model.snapshot != spec.analyst_model:
            raise ValueError("memory analyst profile changed before dispatch")
        completion = StructuredCompletion(
            client=model.client,
            artifacts=project.artifacts,
            plan=CompletionPlan(
                model=model.snapshot,
                capabilities=model.capabilities,
                deadline=spec.deadline,
                output_tokens=spec.output_tokens,
                maximum_cost_per_call=1,
                code_revision=code_revision,
                output_protocol="tool",
            ),
            budget=RequestBudget(project, identity=identity, maximum_cost_usd=None),
        )
        with call_consumer("analyst"):
            result = analyze_traces(definition, (), complete=completion, logical_id=identity)
        project.artifacts.write_json(
            artifact_id=identity,
            artifact_type="memory-diagnostic-analysis",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                inputs=_analysis_sources(project, spec, identity),
                code_revision=code_revision,
            ),
            files={"analysis.json": result, "spec.json": spec},
        )
        return result
