"""Raw source authority, proposed interfaces and public-prefix initialization."""

from typing import Literal, Self

from jsonschema import Draft202012Validator
from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactId,
    ContractModel,
    JsonObject,
    JsonValue,
    Sha256,
    sha256_json,
)
from exp.simulation.reconstruction.context import prior_records

Status = Literal["observed", "documented", "inferred", "contradicted", "unknown"]


def resolve(value: JsonValue, pointer: str) -> JsonValue:
    """Resolve a source span without interpreting its content as instructions."""
    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise ValueError("Source pointer must be absolute or empty")
    for token in pointer[1:].split("/"):
        if "~" in token.replace("~0", "").replace("~1", ""):
            raise ValueError("Invalid source pointer escape")
        key = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict) and key in value:
            value = value[key]
        elif (
            isinstance(value, list)
            and key.isascii()
            and key.isdigit()
            and str(int(key)) == key
            and int(key) < len(value)
        ):
            value = value[int(key)]
        else:
            raise ValueError("Source pointer does not resolve")
    return value


def check_schema(schema: JsonObject) -> None:
    """Validate JSON Schema while forbidding remote or filesystem resolution."""

    def visit(value: JsonValue) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in ("$ref", "$dynamicRef") and (
                    not isinstance(child, str) or not child.startswith("#")
                ):
                    raise ValueError("Only local JSON Schema references are permitted")
                if key == "$id":
                    raise ValueError("Schema base URI changes are not permitted")
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(schema)
    Draft202012Validator.check_schema(schema)


def path_parts(path: str) -> tuple[str, ...]:
    """Validate structural paths even when the designated unknown field is absent."""
    if path == "":
        return ()
    if not path.startswith("/"):
        raise ValueError("State paths must be JSON pointers")
    tokens = path[1:].split("/")
    if any("~" in token.replace("~0", "").replace("~1", "") for token in tokens):
        raise ValueError("Invalid state pointer escape")
    return tuple(token.replace("~1", "/").replace("~0", "~") for token in tokens)


def validate_state(schema: JsonObject, state: JsonObject, unknown: tuple[str, ...]) -> None:
    """Validate known values, allowing only explicitly unknown required omissions."""
    unknown_parts = tuple(path_parts(path) for path in unknown)
    for error in Draft202012Validator(schema).iter_errors(state):
        if error.validator == "required" and isinstance(error.instance, dict):
            base = tuple(str(part) for part in error.absolute_path)
            missing = tuple(name for name in error.validator_value if name not in error.instance)
            if all(
                any(target[: len(path)] == path for path in unknown_parts)
                for name in missing
                for target in (base + (name,),)
            ):
                continue
        raise error


def overlaps(left: str, right: str) -> bool:
    """Detect either ancestor relationship on decoded pointer segments."""
    a, b = path_parts(left), path_parts(right)
    return a == b[: len(a)] or b == a[: len(b)]


def state_leaves(value: JsonValue, path: str = "") -> list[str]:
    """Include empty collections as values that require source grounding."""
    if isinstance(value, dict) and value:
        return [
            leaf
            for key, child in value.items()
            for leaf in state_leaves(child, path + "/" + key.replace("~", "~0").replace("/", "~1"))
        ]
    if isinstance(value, list) and value:
        return [
            leaf
            for index, child in enumerate(value)
            for leaf in state_leaves(child, path + "/" + str(index))
        ]
    return [path]


class RawEpisode(ContractModel):
    """Exact ordered public development records; no native qualification is implied.

    Attributes:
        id: Unique episode alias within this construction.
        records: Original records in public chronology, including call/result identities.
    """

    id: str = Field(min_length=1)
    records: tuple[JsonObject, ...] = Field(min_length=1)


class RawInputs(ContractModel):
    """Explicit JSON episode input with optional descriptive or published authority.

    Attributes:
        episodes: Ordered admitted source episodes, possibly absent for description-only inputs.
        description: Optional external environment description.
        published_contracts: Optional actual tool specification, distinct from inferred schemas.
        mode: Induced discovers interfaces; supplied preserves published tool contracts.
        prior_context: Source-linked prior observations, hypotheses and investigations.
            Generated rules and research do not acquire documented/original authority.
    """

    episodes: tuple[RawEpisode, ...] = ()
    description: str | None = None
    published_contracts: JsonObject | None = None
    mode: Literal["induced", "supplied"] = "induced"
    prior_context: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_sources(self) -> Self:
        """Require some evidence, unique aliases and contracts in supplied mode."""
        if not self.episodes and not self.description and not self.published_contracts:
            raise ValueError(
                "At least one raw episode, description or published contract is required"
            )
        if len({episode.id for episode in self.episodes}) != len(self.episodes):
            raise ValueError("Duplicate raw episode ID")
        if self.mode == "supplied" and self.published_contracts is None:
            raise ValueError("Supplied mode requires published_contracts")
        return self

    def source_map(self) -> dict[str, JsonValue]:
        """Resolve stable evidence aliases to their admitted original JSON."""
        values: dict[str, JsonValue] = {
            "episode:" + episode.id: list(episode.records) for episode in self.episodes
        }
        if self.description is not None:
            values["description"] = self.description
        if self.published_contracts is not None:
            values["contracts"] = self.published_contracts
        for kind in ("trajectory", "rule", "skill", "research"):
            for record in prior_records(self.prior_context, (kind,)):
                entry = record["entry"]
                assert isinstance(entry, dict)
                identity = entry.get("entry_id")
                if not isinstance(identity, str) or not identity:
                    raise ValueError("prior source requires its immutable library entry ID")
                source_id = "prior:" + kind + ":" + identity
                if source_id in values:
                    raise ValueError("prior library selection repeats a source identity")
                values[source_id] = record
        return values


class SourceRef(ContractModel):
    """Exact source identifier and JSON pointer, not an authority assertion.

    Attributes:
        source_id: Alias in the admitted raw-input source map.
        pointer: RFC 6901 pointer relative to that original value; empty means its root.
    """

    source_id: str = Field(min_length=1)
    pointer: str


class Claim(ContractModel):
    """Model-authored epistemic label retained for audit, not automatically certified.

    Attributes:
        statement: Scoped behavioral proposition, including any uncertainty.
        status: Observed, documented, inferred, contradicted or unknown.
        source_refs: Exact original spans; resolvability does not prove entailment.
        scope: Operations and conditions to which the proposition applies.
        origin: Evidence interpretation or an explicitly unsupported model hypothesis.
    """

    statement: str = Field(min_length=1)
    status: Status
    source_refs: tuple[SourceRef, ...] = ()
    scope: str = Field(min_length=1)
    origin: Literal["evidence", "model_prior"] = "evidence"


class ToolContract(ContractModel):
    """Proposed structural interface; schema validity is not proof of business behavior.

    Attributes:
        name: Exact external tool identifier.
        description: Generated public interface description.
        parameters: Local-only Draft 2020-12 argument schema.
        reply_schema: Optional observed reply schema; absent leaves reply shape unspecified.
    """

    name: str = Field(min_length=1)
    description: str
    parameters: JsonObject
    reply_schema: JsonObject | None = None

    @model_validator(mode="after")
    def schemas(self) -> Self:
        """Validate argument and reply schemas without permitting external references."""
        check_schema(self.parameters)
        if self.reply_schema is not None:
            check_schema(self.reply_schema)
        return self


class Interface(ContractModel):
    """Frozen generated interface and explicit initialization requirements.

    Attributes:
        tools: Nonempty inventory with unique names.
        state_schema: Proposed representation of the current instance.
        initialization_requirements: Facts or public reads needed before execution.
    """

    tools: tuple[ToolContract, ...] = Field(min_length=1)
    state_schema: JsonObject
    initialization_requirements: tuple[str, ...] = ()

    @model_validator(mode="after")
    def schemas(self) -> Self:
        """Require unique operations and a structurally valid local state schema."""
        if len({tool.name for tool in self.tools}) != len(self.tools):
            raise ValueError("Duplicate tool names")
        check_schema(self.state_schema)
        return self


class ReconstructionProposal(ContractModel):
    """Constructor-owned semantic content, excluding all source bookkeeping.

    Attributes:
        interface: Proposed tool and state schemas.
        claims: Nonempty inventory of source-qualified propositions and unknowns.
        text_rules: Reusable behavioral instructions, not current-instance values.
        python_source: Optional generated replica source; storing it does not execute it.
    """

    interface: Interface
    claims: tuple[Claim, ...] = Field(min_length=1)
    text_rules: str = Field(min_length=1)
    python_source: str | None = None


class SuppliedBehaviorProposal(ContractModel):
    """Generated semantics when public tool contracts are owned by the caller.

    Attributes:
        state_schema: Proposed current-instance representation, allowing unknown facts.
        initialization_requirements: Public information needed to initialize that representation.
        claims: Source-qualified behavioral propositions and remaining unknowns.
        text_rules: Reusable behavioral rules, excluding immutable tool metadata.
        python_source: Optional generated source; it does not authorize execution.
    """

    state_schema: JsonObject
    initialization_requirements: tuple[str, ...] = ()
    claims: tuple[Claim, ...] = Field(min_length=1)
    text_rules: str = Field(min_length=1)
    python_source: str | None = None


class ReconstructionBundle(ReconstructionProposal):
    """Source-bound cold artifact; no native or protected acceptance is asserted.

    Attributes:
        artifact_id: Immutable project artifact containing this construction.
        sources: Original caller-admitted construction inputs.
        inputs_sha256: Digest binding those exact source bytes and input mode.
    """

    artifact_id: ArtifactId
    sources: RawInputs
    inputs_sha256: Sha256

    @model_validator(mode="after")
    def source_binding(self) -> Self:
        """Reject inconsistent source, schema or instance declarations."""
        if sha256_json(self.sources.model_dump(mode="json")) != self.inputs_sha256:
            raise ValueError(
                "Reconstruction inputs differ; reconstruct a fresh bundle from the raw sources"
            )
        values = self.sources.source_map()
        for claim in self.claims:
            for ref in claim.source_refs:
                if ref.source_id not in values:
                    raise ValueError("Claim cites an unavailable source")
                resolve(values[ref.source_id], ref.pointer)
            if claim.status in ("observed", "documented", "contradicted") and not claim.source_refs:
                raise ValueError(
                    "Observed, documented and contradicted labels require source spans"
                )
            if claim.status == "contradicted" and not any(
                ref.source_id.startswith(("episode:", "prior:trajectory:"))
                or ref.source_id in ("description", "contracts")
                for ref in claim.source_refs
            ):
                raise ValueError(
                    "Contradiction requires original observations or supplied documentation"
                )
            if claim.status == "observed" and any(
                not ref.source_id.startswith(("episode:", "prior:trajectory:"))
                for ref in claim.source_refs
            ):
                raise ValueError(
                    "Observed claims require raw target or prior-original episode spans"
                )
            if claim.status == "documented" and any(
                ref.source_id not in ("description", "contracts") for ref in claim.source_refs
            ):
                raise ValueError("Documented claims require descriptive or contract sources")
            if claim.origin == "model_prior" and claim.status not in ("inferred", "unknown"):
                raise ValueError("Model priors cannot claim observed or documented authority")
            if (
                claim.status == "inferred"
                and not claim.source_refs
                and claim.origin != "model_prior"
            ):
                raise ValueError(
                    "Unsupported inference must explicitly identify model_prior origin"
                )
        if self.sources.mode == "supplied":
            contracts = self.sources.published_contracts or {}
            tools = contracts.get("tools")
            if not isinstance(tools, list):
                raise ValueError("Supplied contracts must contain a tools list")
            if tuple(ToolContract.model_validate(tool) for tool in tools) != self.interface.tools:
                raise ValueError("Supplied tool contracts cannot be rewritten by the constructor")
        return self

    @property
    def digest(self) -> str:
        """Return the content identity of this immutable artifact."""
        return sha256_json(self.model_dump(mode="json"))

    def public_context(self) -> JsonObject:
        """Return the declared public behavioral model and its evidence."""
        return self.model_dump(mode="json")


class InitializationProposal(ContractModel):
    """Instance facts derived only from the current public prefix.

    Attributes:
        state: Known instance values; unknown fields are omitted rather than guessed.
        unknown_paths: Explicit missing scopes expressed as JSON Pointers.
        complete_scopes: Claimed complete public observations, not native certification.
        field_sources: Current-prefix spans covering every initialized leaf.
        field_statuses: Observed equality or declared inference for each sourced path.
    """

    state: JsonObject
    unknown_paths: tuple[str, ...] = ()
    complete_scopes: tuple[str, ...] = ()
    field_sources: dict[str, tuple[SourceRef, ...]] = Field(default_factory=dict)
    field_statuses: dict[str, Literal["observed", "inferred"]] = Field(default_factory=dict)


class InitializationArtifact(InitializationProposal):
    """Bound public initializer; complete scopes remain declared modeling scope.

    Attributes:
        artifact_id: Immutable initializer artifact identity.
        bundle_sha256: Exact behavioral representation used for initialization.
        prefix_sha256: Digest of this instance public chronology.
        public_prefix: Original permitted observations, separate from development evidence.
    """

    artifact_id: ArtifactId
    bundle_sha256: Sha256
    prefix_sha256: Sha256
    public_prefix: tuple[JsonObject, ...]

    @model_validator(mode="after")
    def bindings(self) -> Self:
        """Reject inconsistent source, schema or instance declarations."""
        prefix: JsonValue = list(self.public_prefix)
        if sha256_json(prefix) != self.prefix_sha256:
            raise ValueError("Initializer prefix identity differs")
        if set(self.field_sources) != set(self.field_statuses):
            raise ValueError("Each initialized field requires both provenance and status")
        for path, refs in self.field_sources.items():
            current = resolve(self.state, path)
            if not refs:
                raise ValueError("Initialized fields need current public prefix sources")
            for ref in refs:
                if ref.source_id != "public_prefix":
                    raise ValueError("Initialization cannot copy development or hidden world state")
                observed = resolve(prefix, ref.pointer)
                if self.field_statuses[path] == "observed" and sha256_json(current) != sha256_json(
                    observed
                ):
                    raise ValueError("Observed initializer value differs from cited prefix span")
        if len(set(self.unknown_paths)) != len(self.unknown_paths) or len(
            set(self.complete_scopes)
        ) != len(self.complete_scopes):
            raise ValueError("Repeated unknown or complete scope")
        for path in (*self.field_sources, *self.unknown_paths, *self.complete_scopes):
            path_parts(path)
        for path in self.unknown_paths:
            if any(overlaps(path, known) for known in (*self.field_sources, *self.complete_scopes)):
                raise ValueError("Unknown scope cannot overlap an initialized or complete scope")
        for leaf in state_leaves(self.state) if self.state else []:
            if any(
                path_parts(path) == path_parts(leaf)[: len(path_parts(path))]
                for path in self.field_sources
            ):
                continue
            raise ValueError("Every initialized state leaf requires source coverage")
        for scope in self.complete_scopes:
            resolve(self.state, scope)
            if not any(overlaps(scope, path) for path in self.field_sources):
                raise ValueError("Complete scope requires public evidence")
        return self

    @property
    def digest(self) -> str:
        """Return the content identity of this immutable artifact."""
        return sha256_json(self.model_dump(mode="json"))
