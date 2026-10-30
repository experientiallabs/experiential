"""Instance facts derived from declared task input files, attached through their existing owners.

The ``derive-facts`` continuation action runs one backend-declared deriver over one declared
starting file per task and attaches the validated facts where the pipeline already reads them:

- a chat JSON trace package: each matched conversation's request context gains
  ``instance_facts``. Import moves that entry to ``Trace.instance_facts``, and
  ``TracePackageFacts`` delivers it to learning, retest and audit replays of that trace.
- ``facts.json``: per trace key, a :class:`ReplayFacts` value for the ``facts`` field of a
  ``GroundedRuntimeDefinition``, and so for any continuation job, mixed definition or executable
  task whose configuration is that definition.

Derivation is local and makes no provider call. Host file paths are read once and never
recorded: facts name the file's path inside the task's world and its SHA-256. A deriver refuses
content it cannot represent exactly, and the action stops on any refusal unless the spec
declares ``allow_refusals``, in which case each refused trace is reported and receives no facts.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ContractModel,
    JsonValue,
    canonical_json_bytes,
    stable_id,
    validate_artifact_file_path,
)
from exp.common.core.files import write_bytes_atomic, write_text_atomic
from exp.common.project import ProjectStore
from exp.common.project.paths import validate_local_id
from exp.common.tasks.facts import INSTANCE_FACTS_CONTEXT_KEY, InstanceFacts, InstanceFactsError
from exp.common.traces.ingest.chat_json import attach_request_context
from exp.optimize.evaluation.fidelity.instance_facts import ReplayFacts
from exp.runtime.environments.facts_derivers import FactsDeriverDefinition


class DeclaredFactsInput(ContractModel):
    """One task's declared starting file.

    Attributes:
        trace: The trace package's key for the task's trace. For chat JSON this is the
            conversation's declared ``trace_id``, ``conversation_id``, ``session_id``,
            ``thread_id`` or ``id`` (the first present).
        file: Host file holding the exact starting bytes; read once and never recorded.
        path: The file's canonical relative path inside the task's world, such as
            ``input.xlsx``. It names the facts and their provenance source.
    """

    trace: str = Field(min_length=1, max_length=512)
    file: Path
    path: str = Field(default="input.xlsx", min_length=1, max_length=512)

    @model_validator(mode="after")
    def _canonical_path(self) -> Self:
        """Require a canonical relative world path, never a host path."""
        if validate_artifact_file_path(self.path).as_posix() != self.path:
            raise ValueError(
                f"input path {self.path!r} must be a canonical relative path inside the task's "
                "world, such as input.xlsx"
            )
        return self


class FactsDerivation(ContractModel):
    """Derive starting-state instance facts for declared task inputs.

    Attributes:
        deriver: The backend deriver and its bounds, such as ``{"kind": "xlsx-workbook"}``.
        inputs: Declared starting files; a trace may declare several files at distinct paths.
        traces: Optional chat JSON trace package whose matched conversations receive the
            facts in their request context. Every trace with facts must match one conversation.
        output: Where the annotated trace package is written; required with ``traces`` and
            never the same file.
        allow_refusals: Report refused inputs and give their traces no facts, instead of
            stopping before anything is written.
    """

    deriver: FactsDeriverDefinition
    inputs: tuple[DeclaredFactsInput, ...] = Field(min_length=1, max_length=10_000)
    traces: Path | None = None
    output: Path | None = None
    allow_refusals: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        """Refuse repeated inputs and an output that is missing, stray or overwrites the source."""
        seen: set[tuple[str, str]] = set()
        for row in self.inputs:
            if (row.trace, row.path) in seen:
                raise ValueError(f"trace {row.trace!r} declares {row.path!r} twice")
            seen.add((row.trace, row.path))
        if (self.traces is None) != (self.output is None):
            raise ValueError("traces and output go together: name the package and its copy")
        if self.traces is not None and self.output is not None:
            if self.traces.resolve() == self.output.resolve():
                raise ValueError("output must be a new file, not the source trace package")
        return self


@dataclass(frozen=True)
class DerivedInput:
    """The outcome for one declared input.

    Attributes:
        trace: Trace key the input was declared for.
        path: World path of the file.
        sha256: Digest of the file bytes read.
        fact_bytes: Canonical JSON size of the derived facts, or None when refused.
        refusal: Why the deriver refused the file, or None when it derived facts.
    """

    trace: str
    path: str
    sha256: str
    fact_bytes: int | None
    refusal: str | None


@dataclass(frozen=True)
class FactsDerivationResult:
    """Validated facts per trace and the outcome of every declared input.

    Attributes:
        identity: Content identity of the derivation: deriver, inputs and file digests.
        inputs: One outcome per declared input, in spec order.
        facts: Starting-state facts by trace key, for traces with at least one derived file.
    """

    identity: str
    inputs: tuple[DerivedInput, ...]
    facts: Mapping[str, ReplayFacts]

    @property
    def refused(self) -> tuple[DerivedInput, ...]:
        """Inputs the deriver refused."""
        return tuple(row for row in self.inputs if row.refusal is not None)


def derive_facts(spec: FactsDerivation) -> FactsDerivationResult:
    """Run the deriver over every declared input, without writing anything.

    Args:
        spec: Deriver, inputs and refusal policy.

    Returns:
        Facts per trace and every input's outcome.

    Raises:
        ValueError: An input file cannot be read.
        InstanceFactsError: A deriver refused an input and ``allow_refusals`` is false.
    """
    outcomes: list[DerivedInput] = []
    derived: dict[str, list[InstanceFacts]] = {}
    for row in spec.inputs:
        try:
            payload = row.file.read_bytes()
        except OSError as error:
            raise ValueError(
                f"cannot read the declared input for trace {row.trace!r} ({row.path}): {error}"
            ) from error
        digest = sha256(payload, usedforsecurity=False).hexdigest()
        try:
            facts = spec.deriver.derive(row.path, payload)
        except InstanceFactsError as error:
            outcomes.append(DerivedInput(row.trace, row.path, digest, None, str(error)))
            continue
        size = len(canonical_json_bytes(facts))
        outcomes.append(DerivedInput(row.trace, row.path, digest, size, None))
        derived.setdefault(row.trace, []).append(facts)
    refused = [row for row in outcomes if row.refusal is not None]
    if refused and not spec.allow_refusals:
        listed = "; ".join(f"{row.trace} {row.path}: {row.refusal}" for row in refused[:3])
        more = f" (and {len(refused) - 3} more)" if len(refused) > 3 else ""
        raise InstanceFactsError(
            f"the {spec.deriver.kind} deriver refused {len(refused)} of {len(outcomes)} inputs: "
            f"{listed}{more}. Fix those inputs, or set allow_refusals to give their traces no "
            "facts"
        )
    facts = {
        trace: ReplayFacts(
            facts=InstanceFacts(facts=tuple(fact for item in rows for fact in item.facts))
        )
        for trace, rows in derived.items()
    }
    identity = stable_id(
        "instance-facts",
        {
            "deriver": spec.deriver.model_dump(mode="json"),
            "inputs": [[row.trace, row.path, row.sha256] for row in outcomes],
        },
    )
    return FactsDerivationResult(identity=identity, inputs=tuple(outcomes), facts=facts)


def attach_trace_package(
    spec: FactsDerivation, result: FactsDerivationResult, *, write: bool
) -> frozenset[str]:
    """Add each trace's facts to its conversation in the declared chat JSON package.

    Args:
        spec: Derivation with ``traces`` and ``output`` set.
        result: Facts from :func:`derive_facts` for the same spec.
        write: Write ``output`` atomically; False only checks that every trace matches.

    Returns:
        Trace keys that received facts.

    Raises:
        ValueError: The spec names no package, the package cannot be read or annotated, or a
            trace with facts matches no conversation.
    """
    if spec.traces is None or spec.output is None:
        raise ValueError("attaching facts to a trace package needs traces and output")
    try:
        payload = spec.traces.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read trace package {spec.traces}: {error}") from error
    entries: dict[str, JsonValue] = {
        trace: facts.facts.model_dump(mode="json") for trace, facts in result.facts.items()
    }
    annotated, matched = attach_request_context(payload, INSTANCE_FACTS_CONTEXT_KEY, entries)
    missing = sorted(set(entries) - matched)
    if missing:
        listed = ", ".join(repr(trace) for trace in missing[:5])
        raise ValueError(
            f"trace package {spec.traces} has no conversation keyed {listed}; declare each "
            "input's trace as its conversation's trace_id, conversation_id, session_id, "
            "thread_id or id"
        )
    if write:
        write_bytes_atomic(spec.output, annotated)
    return matched


def save_derived_facts(store: ProjectStore, result: FactsDerivationResult) -> Path:
    """Write ``facts.json`` (trace key to ``ReplayFacts``) under the project's runtime directory.

    Args:
        store: Project the derivation ran for.
        result: Derived facts.

    Returns:
        ``<project>/runtime/facts/<identity>/facts.json``, written atomically.
    """
    validate_local_id(result.identity, label="facts identity")
    path = store.paths.runtime_directory / "facts" / result.identity / "facts.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {trace: facts.model_dump(mode="json") for trace, facts in result.facts.items()}
    write_text_atomic(path, canonical_json_bytes(document).decode("utf-8"))
    return path
