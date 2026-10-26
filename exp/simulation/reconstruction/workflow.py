"""Parent-free, single-attempt construction and current-prefix initialization."""

import copy
import json
from collections.abc import Callable
from datetime import datetime
from importlib.resources import files
from typing import Literal

from jsonschema import Draft202012Validator, SchemaError
from pydantic import Field, StrictInt, StrictStr, ValidationError

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ContractModel,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.project import ArtifactStore, artifact_input

from .models import (
    Claim,
    InitializationArtifact,
    InitializationProposal,
    Interface,
    RawInputs,
    ReconstructionBundle,
    ReconstructionProposal,
    SourceRef,
    Status,
    SuppliedBehaviorProposal,
    ToolContract,
    resolve,
    state_leaves,
    validate_state,
)

Completion = Callable[[JsonObject, str], JsonObject]


class _StateLayout(ContractModel):
    """State representation proposed separately from behavioral knowledge.

    Attributes:
        state_schema: Local JSON Schema for the instance representation.
        initialization_requirements: Public facts required to initialize it.
    """

    state_schema: JsonObject
    initialization_requirements: tuple[str, ...] = ()


class _Behavior(ContractModel):
    """Semantic knowledge generated against an already validated state layout.

    Attributes:
        claims: Source-linked propositions and explicit unknowns.
        text_rules: Behavioral instructions consistent with the claims.
        python_source: Optional representation retained without execution.
    """

    claims: tuple[Claim, ...] = Field(min_length=1)
    text_rules: str = Field(min_length=1)
    python_source: str | None = None


class _IndexedClaim(ContractModel):
    """Model-chosen support from a finite admitted span inventory.

    Attributes:
        statement: Behavioral proposition with explicit scope.
        status: Epistemic label, checked by ordinary claim admission.
        source_indices: Exact entries in the supplied span registry.
        scope: Conditions or operations covered by the proposition.
        origin: Admitted evidence or explicitly unsupported prior.
    """

    statement: str = Field(min_length=1)
    status: Status
    source_indices: tuple[StrictInt, ...] = ()
    scope: str = Field(min_length=1)
    origin: Literal["evidence", "model_prior"] = "evidence"


class _IndexedBehavior(ContractModel):
    """Behavioral output whose citation bookkeeping is owned by code.

    Attributes:
        claims: Model-authored propositions and support selections.
        text_rules: Reusable behavioral instructions consistent with the claims.
        python_source: Optional source retained without execution.
    """

    claims: tuple[_IndexedClaim, ...] = Field(min_length=1)
    text_rules: str = Field(min_length=1)
    python_source: str | None = None

    def materialize(self, spans: list[SourceRef]) -> _Behavior:
        """Resolve selected spans without inventing support or asserting entailment."""
        claims = []
        for claim in self.claims:
            if any(index < 0 or index >= len(spans) for index in claim.source_indices):
                raise ValueError("Claim selected an unavailable source span")
            claims.append(
                Claim(
                    statement=claim.statement,
                    status=claim.status,
                    scope=claim.scope,
                    origin=claim.origin,
                    source_refs=tuple(spans[index] for index in claim.source_indices),
                )
            )
        return _Behavior(
            claims=tuple(claims), text_rules=self.text_rules, python_source=self.python_source
        )


class _IndexedProposal(_IndexedBehavior):
    """Inferred interface with code-owned source-reference bookkeeping.

    Attributes:
        interface: Model-proposed tools and state schema, checked before admission.
    """

    interface: Interface


class _IndexedSuppliedProposal(_IndexedBehavior):
    """Behavioral proposal preserving an externally supplied tool inventory.

    Attributes:
        state_schema: Proposed current-instance representation.
        initialization_requirements: Public values needed to open an instance.
    """

    state_schema: JsonObject
    initialization_requirements: tuple[str, ...] = ()


def _indexed_output_schema(proposal: type[_IndexedBehavior], spans: list[SourceRef]) -> JsonObject:
    """Advertise provenance requirements already enforced by bundle admission."""
    schema = proposal.model_json_schema()
    claim = schema["$defs"]["_IndexedClaim"]
    claim["required"] = list(dict.fromkeys([*claim["required"], "origin", "source_indices"]))
    claim["properties"]["source_indices"]["items"] = {
        "type": "integer",
        "minimum": 0,
        "maximum": len(spans) - 1,
    }
    claim["allOf"] = [
        {
            "if": {"properties": {"status": {"enum": ["observed", "documented", "contradicted"]}}},
            "then": {
                "properties": {"origin": {"const": "evidence"}, "source_indices": {"minItems": 1}}
            },
        },
        {
            "if": {
                "properties": {"status": {"const": "inferred"}, "source_indices": {"maxItems": 0}}
            },
            "then": {"properties": {"origin": {"const": "model_prior"}}},
        },
    ]
    return schema


class _InitializationBinding(ContractModel):
    """One model-chosen assignment from the current public prefix.

    Attributes:
        path: Nonempty literal object-key path in the candidate state.
        pointer: RFC 6901 pointer into public_prefix.
    """

    path: tuple[StrictStr, ...] = Field(min_length=1)
    pointer: str


class _ReferenceInitialization(ContractModel):
    """Reference-only output; code copies values and derives conservative coverage.

    Attributes:
        bindings: Nonoverlapping assignments without generated instance values.
    """

    bindings: tuple[_InitializationBinding, ...] = Field(min_length=1)


def _reference_initialization(
    proposal: _ReferenceInitialization,
    prefix: list[JsonObject],
    schema: JsonObject,
) -> InitializationProposal:
    """Copy only cited public values; derive required omissions without guessing facts."""
    state: JsonObject = {}
    sources: dict[str, tuple[SourceRef, ...]] = {}
    paths: list[tuple[str, ...]] = []
    for binding in proposal.bindings:
        if any(
            binding.path == old[: len(binding.path)] or old == binding.path[: len(old)]
            for old in paths
        ):
            raise ValueError("Initializer bindings overlap")
        paths.append(binding.path)
        value = copy.deepcopy(resolve(prefix, binding.pointer))
        target = state
        for part in binding.path[:-1]:
            child = target.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError("Initializer path traverses a non-object value")
            target = child
        target[binding.path[-1]] = value
        path = "".join("/" + part.replace("~", "~0").replace("/", "~1") for part in binding.path)
        for suffix in state_leaves(value):
            sources[path + suffix] = (
                SourceRef(source_id="public_prefix", pointer=binding.pointer + suffix),
            )
    unknown: set[str] = set()
    for error in Draft202012Validator(schema).iter_errors(state):
        if error.validator != "required" or not isinstance(error.instance, dict):
            raise error
        for name in error.validator_value:
            if name not in error.instance:
                parts = (*error.absolute_path, name)
                unknown.add(
                    "".join("/" + str(part).replace("~", "~0").replace("/", "~1") for part in parts)
                )
    return InitializationProposal(
        state=state,
        unknown_paths=tuple(sorted(unknown)),
        complete_scopes=(),
        field_sources=sources,
        field_statuses={path: "observed" for path in sources},
    )


def _call(
    *,
    phase: str,
    payload: JsonObject,
    artifacts: ArtifactStore,
    run_id: str,
    complete: Completion,
    clock: Callable[[], datetime],
    deadline: datetime,
    code_revision: str,
    prompt_name: str | None = None,
    original_inputs: JsonObject | None = None,
    guidance: str = "",
) -> tuple[JsonObject, str]:
    """Persist an exclusive intent and raw response; failures never automatically retry.

    Registered constructor ``guidance`` is rendered into the request before the intent is
    saved, so the saved request is exactly the one dispatched.
    """
    if deadline.tzinfo is None or clock() >= deadline:
        raise ValueError("An unexpired aware deadline is required before dispatch")
    identity = stable_id("reconstruction", {"run_id": run_id, "phase": phase})
    prompt = (
        files("exp.simulation.reconstruction")
        .joinpath("prompts", (prompt_name or phase) + ".txt")
        .read_text(encoding="utf-8")
    )
    request: JsonObject = {
        "messages": [
            {"role": "system", "content": prompt},
            *([{"role": "developer", "content": guidance}] if guidance else []),
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)},
        ]
    }
    request["response_schema"] = payload["output_schema"]
    source_payload = original_inputs if original_inputs is not None else payload["public_prefix"]
    source_record: JsonObject = {
        "source_authority": "claimed_public_not_native",
        "origin": "caller_supplied_raw_inputs"
        if original_inputs is not None
        else "caller_supplied_current_prefix",
        "materialized_at": clock().isoformat(),
        "payload_sha256": sha256_json(source_payload),
        "payload": source_payload,
    }
    artifacts.write_json(
        artifact_id=identity + "-source",
        artifact_type="reconstruction-public-source",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=clock(), inputs=(), code_revision=code_revision
        ),
        files={"source.json": source_record},
    )
    source_ref = artifact_input(artifacts.read(identity + "-source").manifest)
    envelope = ArtifactEnvelope(
        schema_version=1, created_at=clock(), inputs=(source_ref,), code_revision=code_revision
    )
    artifacts.write_json(
        artifact_id=identity + "-intent",
        artifact_type="reconstruction-intent",
        envelope=envelope,
        files={
            "request.json": {"request": request, "deadline": deadline.isoformat(), "phase": phase}
        },
    )
    try:
        response = complete(request, identity)
        artifacts.write_json(
            artifact_id=identity + "-response",
            artifact_type="reconstruction-response",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=clock(),
                inputs=(source_ref,),
                code_revision=code_revision,
            ),
            files={"response.json": response},
        )
    except Exception as error:  # noqa: BLE001 - retain failed injected provider attempts.
        artifacts.write_json(
            artifact_id=identity + "-failure",
            artifact_type="reconstruction-failure",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=clock(),
                inputs=(source_ref,),
                code_revision=code_revision,
            ),
            files={
                "failure.json": {
                    "error_type": type(error).__name__,
                    "logical_scope_consumed": True,
                    "dispatch_status": "callback_owned",
                }
            },
        )
        raise
    if clock() >= deadline:
        raise TimeoutError("Late constructor response retained without admission")
    return response, identity


def _save(
    artifacts: ArtifactStore,
    identity: str,
    value: ReconstructionBundle | InitializationArtifact,
    clock: Callable[[], datetime],
    code_revision: str,
    prior_stages: tuple[str, ...] = (),
) -> None:
    """Publish the admitted artifact with exact intent and response references."""
    refs = tuple(
        sorted(
            (
                artifact_input(artifacts.read(stage + suffix).manifest)
                for stage in (*prior_stages, identity)
                for suffix in ("-intent", "-response")
            ),
            key=lambda ref: ref.artifact_id,
        )
    )
    artifacts.write_json(
        artifact_id=identity,
        artifact_type="reconstruction",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=clock(), inputs=refs, code_revision=code_revision
        ),
        files={"artifact.json": value.model_dump(mode="json")},
    )


def reconstruct(
    inputs: RawInputs,
    *,
    representation: Literal["text", "python", "both"] = "text",
    strategy: Literal["flat", "staged"] = "flat",
    maximum_layout_repairs: Literal[0, 1] = 0,
    citation_protocol: Literal["pointers", "indexed"] = "pointers",
    artifacts: ArtifactStore,
    run_id: str,
    complete: Completion,
    clock: Callable[[], datetime],
    deadline: datetime,
    code_revision: str,
    guidance: str = "",
) -> ReconstructionBundle:
    """Propose an initial interface and dynamics from raw public inputs, without a parent.

    The injected callback owns physical-call metering, input capacity and cost limits.
    Flat construction makes one call; supplied staged construction validates the state
    layout before a separate behavioral call. Each response is retained unchanged and
    bound to the admitted bundle. Optional layout format repair allows one separately
    recorded correction of schema errors, never a retry based on native scores. Neither
    strategy claims native correctness. Registered ``guidance`` is part of every saved
    request.
    """
    if citation_protocol not in ("pointers", "indexed"):
        raise ValueError("Unsupported citation protocol")
    if type(maximum_layout_repairs) is not int or maximum_layout_repairs not in (0, 1):
        raise ValueError("Layout format repair is limited to zero or one declared call")
    if maximum_layout_repairs and strategy != "staged":
        raise ValueError("Layout format repair requires staged construction")
    if strategy not in ("flat", "staged"):
        raise ValueError("Unsupported construction strategy")
    if strategy == "staged" and inputs.mode != "supplied":
        raise ValueError("Staged construction currently requires supplied tool contracts")
    if representation not in ("text", "python", "both"):
        raise ValueError("Unsupported reconstruction representation")
    inputs = RawInputs.model_validate_json(inputs.model_dump_json())
    supplied_tools: tuple[ToolContract, ...] | None = None
    if inputs.mode == "supplied":
        tools = (inputs.published_contracts or {}).get("tools")
        if not isinstance(tools, list):
            raise ValueError("Supplied contracts must contain a tools list")
        supplied_tools = tuple(ToolContract.model_validate(tool) for tool in tools)
        Interface(tools=supplied_tools, state_schema={})
    stages: list[str] = []
    layout: _StateLayout | None = None
    source_payload = inputs.model_dump(mode="json")
    payload: JsonObject = {
        "representation": representation,
        "input_mode": inputs.mode,
        "sources": inputs.source_map(),
        "source_ids": list(inputs.source_map()),
        "source_locators": [
            {"source_id": source_id, "pointer": pointer}
            for source_id, value in inputs.source_map().items()
            for pointer in (
                ["", *(f"/{i}" for i in range(len(value)))] if isinstance(value, list) else [""]
            )
        ],
    }
    locators = payload["source_locators"]
    assert isinstance(locators, list)
    spans = [SourceRef.model_validate(row) for row in locators]
    if citation_protocol == "indexed":
        payload["claim_source_spans"] = [
            {"span_index": index, **span.model_dump(mode="json")}
            for index, span in enumerate(spans)
        ]
    if strategy == "staged":
        response, stage = _call(
            phase="reconstruct-layout",
            prompt_name="reconstruct-layout",
            original_inputs=source_payload,
            payload={**payload, "output_schema": _StateLayout.model_json_schema()},
            artifacts=artifacts,
            run_id=run_id,
            complete=complete,
            clock=clock,
            deadline=deadline,
            code_revision=code_revision,
            guidance=guidance,
        )
        assert supplied_tools is not None
        stages.append(stage)
        try:
            layout = _StateLayout.model_validate(response)
            Interface(tools=supplied_tools, **layout.model_dump())
        except (ValidationError, SchemaError, ValueError) as error:
            if maximum_layout_repairs == 0:
                raise
            response, stage = _call(
                phase="reconstruct-layout-format-repair",
                prompt_name="reconstruct-layout-repair",
                original_inputs=source_payload,
                payload={
                    **payload,
                    "invalid_layout": response,
                    "validation_error": str(error),
                    "output_schema": _StateLayout.model_json_schema(),
                },
                artifacts=artifacts,
                run_id=run_id,
                complete=complete,
                clock=clock,
                deadline=deadline,
                code_revision=code_revision,
                guidance=guidance,
            )
            stages.append(stage)
            layout = _StateLayout.model_validate(response)
            Interface(tools=supplied_tools, **layout.model_dump())
        payload["state_layout"] = layout.model_dump(mode="json")
        schema = (
            _IndexedBehavior.model_json_schema()
            if citation_protocol == "indexed"
            else _Behavior.model_json_schema()
        )
        if citation_protocol == "indexed":
            schema = _indexed_output_schema(_IndexedBehavior, spans)
        prompt_name = (
            "reconstruct-behavior-indexed"
            if citation_protocol == "indexed"
            else "reconstruct-behavior"
        )
        phase = "reconstruct-behavior"
    else:
        schema = (
            SuppliedBehaviorProposal.model_json_schema()
            if supplied_tools is not None
            else ReconstructionProposal.model_json_schema()
        )
        prompt_name = "reconstruct-supplied" if supplied_tools is not None else "reconstruct"
        if citation_protocol == "indexed":
            schema = _indexed_output_schema(
                _IndexedSuppliedProposal if supplied_tools is not None else _IndexedProposal,
                spans,
            )
            prompt_name = "reconstruct-indexed"
        phase = "reconstruct"
    response, identity = _call(
        phase=phase,
        prompt_name=prompt_name,
        original_inputs=source_payload,
        payload={**payload, "output_schema": schema},
        artifacts=artifacts,
        run_id=run_id,
        complete=complete,
        clock=clock,
        deadline=deadline,
        code_revision=code_revision,
        guidance=guidance,
    )
    if layout is not None:
        behavior = (
            _IndexedBehavior.model_validate(response).materialize(spans)
            if citation_protocol == "indexed"
            else _Behavior.model_validate(response)
        )
        assert supplied_tools is not None
        proposal = ReconstructionProposal(
            interface=Interface(tools=supplied_tools, **layout.model_dump()),
            **behavior.model_dump(),
        )
    elif citation_protocol == "indexed":
        if supplied_tools is None:
            inferred = _IndexedProposal.model_validate(response)
            proposal = ReconstructionProposal(
                interface=inferred.interface, **inferred.materialize(spans).model_dump()
            )
        else:
            supplied = _IndexedSuppliedProposal.model_validate(response)
            proposal = ReconstructionProposal(
                interface=Interface(
                    tools=supplied_tools,
                    state_schema=supplied.state_schema,
                    initialization_requirements=supplied.initialization_requirements,
                ),
                **supplied.materialize(spans).model_dump(),
            )
    elif supplied_tools is None:
        proposal = ReconstructionProposal.model_validate(response)
    else:
        supplied_behavior = SuppliedBehaviorProposal.model_validate(response)
        proposal = ReconstructionProposal(
            interface=Interface(
                tools=supplied_tools,
                state_schema=supplied_behavior.state_schema,
                initialization_requirements=supplied_behavior.initialization_requirements,
            ),
            claims=supplied_behavior.claims,
            text_rules=supplied_behavior.text_rules,
            python_source=supplied_behavior.python_source,
        )
    if representation in ("python", "both") and not (proposal.python_source or "").strip():
        raise ValueError("Requested Python representation requires nonempty python_source")
    bundle = ReconstructionBundle(
        **proposal.model_dump(),
        artifact_id=identity,
        sources=inputs,
        inputs_sha256=sha256_json(inputs.model_dump(mode="json")),
    )
    _save(artifacts, identity, bundle, clock, code_revision, tuple(stages))
    return bundle


def initialize(
    bundle: ReconstructionBundle,
    public_prefix: tuple[JsonObject, ...],
    *,
    strategy: Literal["values", "references"] = "values",
    artifacts: ArtifactStore,
    run_id: str,
    complete: Completion,
    clock: Callable[[], datetime],
    deadline: datetime,
    code_revision: str,
) -> InitializationArtifact:
    """Construct instance state from a supplied current-world public prefix only.

    Development episodes and descriptions are intentionally absent from this request.
    The inferred schema guides representation, not instance values. Missing facts must
    remain explicitly unknown; no evaluator or historical development state is supplied.
    """
    if strategy not in ("values", "references"):
        raise ValueError("Unsupported initialization strategy")
    bundle = ReconstructionBundle.model_validate_json(bundle.model_dump_json())
    prefix = json.loads(json.dumps(list(public_prefix), allow_nan=False))
    response, identity = _call(
        phase="initialize" if strategy == "values" else "initialize-references",
        payload={
            "interface": bundle.interface.model_dump(mode="json"),
            "public_prefix": prefix,
            "output_schema": (
                InitializationProposal.model_json_schema()
                if strategy == "values"
                else _ReferenceInitialization.model_json_schema()
            ),
        },
        artifacts=artifacts,
        run_id=run_id,
        complete=complete,
        clock=clock,
        deadline=deadline,
        code_revision=code_revision,
    )
    proposal = (
        InitializationProposal.model_validate(response)
        if strategy == "values"
        else _reference_initialization(
            _ReferenceInitialization.model_validate(response), prefix, bundle.interface.state_schema
        )
    )
    result = InitializationArtifact(
        **proposal.model_dump(),
        artifact_id=identity,
        bundle_sha256=bundle.digest,
        prefix_sha256=sha256_json(prefix),
        public_prefix=tuple(prefix),
    )
    validate_state(bundle.interface.state_schema, result.state, result.unknown_paths)
    _save(artifacts, identity, result, clock, code_revision)
    return result
