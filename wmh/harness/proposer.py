"""Typed delta proposers for direct providers and project-backed meta agents."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Protocol

from wmh.agents.project import AgentProjectRun
from wmh.core.types import JsonObject
from wmh.harness.delta import FailureSignature, HarnessDelta, apply_delta
from wmh.harness.doc import HarnessDoc
from wmh.harness.mutate import parse_delta, propose_delta
from wmh.harness.runtime import HarnessSearchCancelled
from wmh.providers.base import Provider, ToolCallingProvider

_CONTEXT_CONTENT_CHUNK_CHARS = 12_000
_MAX_PROJECT_REPAIR_TURNS = 2
_SUPPORTED_RUNTIME_KINDS = frozenset({"kit-python", "pi-node"})


@dataclass(frozen=True)
class _ProposalValidation:
    """Host-validated proposal slots plus the raw files needed to protect valid siblings."""

    proposals: dict[int, HarnessDelta]
    raw_files: dict[int, str]
    child_hashes: dict[int, str]
    errors: dict[int, str]


class AgentProject(Protocol):
    """Project operations required by the optimizer wiring."""

    workspace: str

    def write_text(self, path: str, content: str) -> None: ...

    def read_text(self, path: str) -> str: ...

    def run(
        self,
        agent: HarnessDoc,
        provider: ToolCallingProvider,
        instruction: str,
        *,
        should_cancel: Callable[[], bool] | None = None,
        writable_files: Collection[str] | None = None,
    ) -> AgentProjectRun: ...


class DeltaProposer(Protocol):
    """Produce sibling deltas against one selected parent."""

    def propose_batch(
        self,
        parent: HarnessDoc,
        trigger: FailureSignature,
        evidence: str,
        *,
        history: list[HarnessDelta],
        count: int,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[HarnessDelta | ProposalFailure | None]: ...


@dataclass(frozen=True)
class ProposalFailure:
    """One proposal slot whose provider or agent call failed."""

    reason: str


class ProviderDeltaProposer:
    """Adapt the original single-completion proposer to the batched search contract."""

    def __init__(self, provider: Provider) -> None:
        self._provider = provider

    @property
    def provider(self) -> Provider:
        """Return the provider used for direct proposal calls."""
        return self._provider

    def propose_batch(
        self,
        parent: HarnessDoc,
        trigger: FailureSignature,
        evidence: str,
        *,
        history: list[HarnessDelta],
        count: int,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[HarnessDelta | ProposalFailure | None]:
        """Make ``count`` independent proposal calls against the same parent."""
        if count < 1:
            raise ValueError(f"proposal count must be positive, got {count}")
        proposals: list[HarnessDelta | ProposalFailure | None] = []
        for _ in range(count):
            _check_cancelled(should_cancel)
            try:
                proposal = propose_delta(
                    parent,
                    trigger,
                    evidence,
                    self._provider,
                    history=history,
                )
            except HarnessSearchCancelled:
                raise
            except Exception as error:  # noqa: BLE001 - isolate one flaky sibling call
                proposal = ProposalFailure(reason=str(error))
            proposals.append(proposal)
        return proposals


class ProjectDeltaProposer:
    """Wire a normal agent in a persistent project into harness search."""

    def __init__(
        self,
        project: AgentProject,
        agent: HarnessDoc,
        provider: ToolCallingProvider,
        *,
        preserve_runtime_kind: bool = False,
    ) -> None:
        self._project = project
        self._agent = agent
        self._provider = provider
        self._iteration = 0
        self._evaluation_dirs: dict[str, str] = {}
        self._proposal_files: dict[str, str] = {}
        self._parent_manifests: dict[str, JsonObject] = {}
        self._should_cancel: Callable[[], bool] | None = None
        self._preserve_runtime_kind = preserve_runtime_kind

    def propose_batch(
        self,
        parent: HarnessDoc,
        trigger: FailureSignature,
        evidence: str,
        *,
        history: list[HarnessDelta],
        count: int,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[HarnessDelta | ProposalFailure | None]:
        """Run one meta-agent turn that writes ``count`` proposal files."""
        if count < 1:
            raise ValueError(f"proposal count must be positive, got {count}")
        self._should_cancel = should_cancel
        _check_cancelled(should_cancel)
        self._iteration += 1
        iteration_dir = f"iteration-{self._iteration:04d}"
        context_dir = f"context/{iteration_dir}"
        proposal_dir = f"proposals/{iteration_dir}"
        parent_context = self._parent_manifests.get(parent.doc_hash)
        if parent_context is None:
            parent_context = _materialize_parent(
                self._project,
                parent,
                context_dir=f"parents/{parent.doc_hash}",
                should_cancel=should_cancel,
            )
            self._parent_manifests[parent.doc_hash] = parent_context
        _write_project_text(
            self._project,
            f"{context_dir}/parent.json",
            json.dumps(parent_context, indent=2),
            should_cancel=should_cancel,
        )
        evidence_context = _materialize_context_content(
            self._project,
            evidence,
            directory=f"{context_dir}/evidence",
            extension=".md",
            should_cancel=should_cancel,
        )
        _write_project_text(
            self._project,
            f"{context_dir}/evidence.json",
            json.dumps(
                {
                    "kind": "failure-evidence",
                    "format": "markdown",
                    **evidence_context,
                },
                indent=2,
            ),
            should_cancel=should_cancel,
        )
        history_content = json.dumps(
            [
                _project_history_entry(
                    delta,
                    proposal_file=self._proposal_files.get(delta.delta_id),
                    evaluation_dir=self._evaluation_dirs.get(delta.delta_id),
                    workspace=self._project.workspace,
                )
                for delta in history
            ],
            indent=2,
        )
        history_context = _materialize_context_content(
            self._project,
            history_content,
            directory=f"{context_dir}/history",
            extension=".json.part",
            should_cancel=should_cancel,
        )
        _write_project_text(
            self._project,
            f"{context_dir}/history.json",
            json.dumps(
                {
                    "kind": "judged-history",
                    "format": "json-array",
                    "entry_count": len(history),
                    **history_context,
                },
                indent=2,
            ),
            should_cancel=should_cancel,
        )
        request = _project_request(
            workspace=self._project.workspace,
            context_dir=context_dir,
            proposal_dir=proposal_dir,
            count=count,
            runtime_kind=parent.runtime_kind(),
            preserve_runtime_kind=self._preserve_runtime_kind,
        )
        _write_project_text(
            self._project,
            f"{context_dir}/REQUEST.md",
            request,
            should_cancel=should_cancel,
        )
        run_error: Exception | None = None
        try:
            self._project.run(
                self._agent,
                self._provider,
                request,
                should_cancel=should_cancel,
                writable_files=[
                    f"{proposal_dir}/proposal-{index:02d}.json" for index in range(1, count + 1)
                ],
            )
        except HarnessSearchCancelled:
            raise
        except Exception as error:  # noqa: BLE001 - durable project files may still be complete
            run_error = error
        _check_cancelled(should_cancel)

        slots = list(range(1, count + 1))
        validation = _validate_project_proposals(
            self._project,
            parent,
            trigger,
            proposal_dir=proposal_dir,
            slots=slots,
            history=history,
            preserve_runtime_kind=self._preserve_runtime_kind,
            should_cancel=should_cancel,
        )
        # A lost terminal frame does not invalidate durable files. Host-preflight whatever was
        # written, then repair only the bad/missing slots in an ordinary follow-up project turn.
        # The last project-channel error matters only while a slot remains unresolved.
        terminal_error = run_error
        if validation.errors:
            validation_report_ready = True
            try:
                _write_project_text(
                    self._project,
                    f"{context_dir}/proposal-validation-attempt-01.json",
                    _proposal_validation_report(
                        parent=parent,
                        proposal_dir=proposal_dir,
                        attempt=1,
                        valid_slots=validation.proposals,
                        errors=validation.errors,
                    ),
                    should_cancel=should_cancel,
                )
            except HarnessSearchCancelled:
                raise
            except Exception as error:  # noqa: BLE001 - no report means no safe repair prompt
                terminal_error = error
                validation_report_ready = False

            for repair_turn in range(1, _MAX_PROJECT_REPAIR_TURNS + 1):
                if not validation.errors or not validation_report_ready:
                    break
                validation_report_ready = False
                validation_path = (
                    f"{context_dir}/proposal-validation-attempt-{repair_turn:02d}.json"
                )
                repair_request = _project_repair_request(
                    workspace=self._project.workspace,
                    validation_path=validation_path,
                    request_path=f"{context_dir}/REQUEST.md",
                    proposal_dir=proposal_dir,
                    errors=validation.errors,
                    valid_slots=validation.proposals,
                    runtime_kind=parent.runtime_kind(),
                    preserve_runtime_kind=self._preserve_runtime_kind,
                    repair_turn=repair_turn,
                )
                invalid_slots = sorted(validation.errors)
                protected_restore_error: Exception | None = None
                turn_error: Exception | None = None
                try:
                    _write_project_text(
                        self._project,
                        f"{context_dir}/REPAIR-{repair_turn:02d}.md",
                        repair_request,
                        should_cancel=should_cancel,
                    )
                    try:
                        self._project.run(
                            self._agent,
                            self._provider,
                            repair_request,
                            should_cancel=should_cancel,
                            writable_files=[
                                f"{proposal_dir}/proposal-{index:02d}.json"
                                for index in invalid_slots
                            ],
                        )
                    except HarnessSearchCancelled:
                        raise
                    except Exception as error:  # noqa: BLE001 - salvage durable repaired files
                        turn_error = error
                    finally:
                        # The agent is asked to rewrite only invalid slots. Restore every
                        # byte-exact valid sibling after each turn as an enforcement boundary.
                        for index in sorted(validation.proposals):
                            try:
                                _write_project_text(
                                    self._project,
                                    f"{proposal_dir}/proposal-{index:02d}.json",
                                    validation.raw_files[index],
                                    should_cancel=should_cancel,
                                )
                            except HarnessSearchCancelled:
                                raise
                            except Exception as error:  # noqa: BLE001 - attempt every restore
                                if protected_restore_error is None:
                                    protected_restore_error = error
                except HarnessSearchCancelled:
                    raise
                except Exception as error:  # noqa: BLE001 - preserve good siblings, fail bad ones
                    turn_error = error
                if protected_restore_error is not None:
                    # The in-memory delta and its durable proposal_file must always name the
                    # same bytes. If protection cannot be proven, abort this batch instead of
                    # returning a valid object whose persistent provenance may have been changed.
                    raise protected_restore_error
                terminal_error = turn_error

                repaired = _validate_project_proposals(
                    self._project,
                    parent,
                    trigger,
                    proposal_dir=proposal_dir,
                    slots=invalid_slots,
                    history=history,
                    valid_proposals=validation.proposals,
                    preserve_runtime_kind=self._preserve_runtime_kind,
                    should_cancel=should_cancel,
                )
                validation = _ProposalValidation(
                    proposals=repaired.proposals,
                    raw_files={**validation.raw_files, **repaired.raw_files},
                    child_hashes={**validation.child_hashes, **repaired.child_hashes},
                    errors=repaired.errors,
                )
                # Each host result becomes the next turn's nested input and the durable final
                # audit. These turns happen wholly inside proposal generation, before search.
                try:
                    _write_project_text(
                        self._project,
                        f"{context_dir}/proposal-validation-attempt-{repair_turn + 1:02d}.json",
                        _proposal_validation_report(
                            parent=parent,
                            proposal_dir=proposal_dir,
                            attempt=repair_turn + 1,
                            valid_slots=validation.proposals,
                            errors=validation.errors,
                        ),
                        should_cancel=should_cancel,
                    )
                    validation_report_ready = True
                except HarnessSearchCancelled:
                    raise
                except Exception as error:  # noqa: BLE001 - preserve validated in-memory output
                    if terminal_error is None:
                        terminal_error = error
                if not validation.errors:
                    # A prior turn can lose its terminal control frame after durable repaired
                    # files were written. Successful preflight is authoritative salvage.
                    terminal_error = None

        proposals: list[HarnessDelta | ProposalFailure | None] = []
        for index in slots:
            stamped = validation.proposals.get(index)
            if stamped is not None:
                proposals.append(stamped)
                self._proposal_files.setdefault(
                    stamped.delta_id,
                    f"{self._project.workspace}/{proposal_dir}/proposal-{index:02d}.json",
                )
                self._evaluation_dirs.setdefault(
                    stamped.delta_id,
                    f"evaluations/{iteration_dir}/proposal-{index:02d}",
                )
            elif terminal_error is not None:
                proposals.append(ProposalFailure(reason=str(terminal_error)))
            else:
                proposals.append(
                    ProposalFailure(
                        reason=validation.errors.get(
                            index,
                            "proposal slot remained invalid after bounded repair turns",
                        )
                    )
                )
        return proposals

    def record_evaluation(self, delta: HarnessDelta, *, stage: str, content: str) -> None:
        """Persist one candidate's judged evidence for later project-agent iterations."""
        should_cancel = self._should_cancel
        _check_cancelled(should_cancel)
        root = self._evaluation_dirs.get(
            delta.delta_id,
            f"evaluations/by-delta/{delta.delta_id}",
        )
        context = _materialize_context_content(
            self._project,
            content,
            directory=f"{root}/{stage}",
            extension=".md",
            should_cancel=should_cancel,
        )
        _write_project_text(
            self._project,
            f"{root}/{stage}.json",
            json.dumps(
                {
                    "kind": "candidate-evaluation",
                    "stage": stage,
                    "delta_id": delta.delta_id,
                    "format": "markdown",
                    **context,
                },
                indent=2,
            ),
            should_cancel=should_cancel,
        )


def _check_cancelled(should_cancel: Callable[[], bool] | None) -> None:
    """Raise the shared search signal without converting it into a failed proposal slot."""
    if should_cancel is not None and should_cancel():
        raise HarnessSearchCancelled("harness search cancelled")


def _write_project_text(
    project: AgentProject,
    path: str,
    content: str,
    *,
    should_cancel: Callable[[], bool] | None,
) -> None:
    """Make each E2B filesystem RPC a cancellation boundary."""
    _check_cancelled(should_cancel)
    project.write_text(path, content)
    _check_cancelled(should_cancel)


def _read_project_text(
    project: AgentProject,
    path: str,
    *,
    should_cancel: Callable[[], bool] | None,
) -> str:
    """Read one project output without hiding cancellation behind the next iteration."""
    _check_cancelled(should_cancel)
    content = project.read_text(path)
    _check_cancelled(should_cancel)
    return content


def _materialize_parent(
    project: AgentProject,
    parent: HarnessDoc,
    *,
    context_dir: str,
    should_cancel: Callable[[], bool] | None = None,
) -> JsonObject:
    """Write a bounded manifest plus individually readable parent-surface chunks.

    A real pi document is hundreds of kilobytes, while one project ``read_file`` observation is
    intentionally capped. Putting every surface inline in one parent.json therefore hid most of
    the harness from the proposer. The manifest remains small and points to ordered chunks below
    the read cap; concatenating a surface's chunks reconstructs its exact content.
    """
    surface_index: list[JsonObject] = []
    for index, surface in enumerate(parent.surfaces, 1):
        surface_dir = f"{context_dir}/parent-surfaces/surface-{index:03d}"
        content_context = _materialize_context_content(
            project,
            surface.content,
            directory=surface_dir,
            extension=".txt",
            include_contract=False,
            should_cancel=should_cancel,
        )
        source_file: str | None = None
        if surface.path is not None:
            source_relative = f"{context_dir}/parent-source/{surface.path}"
            _write_project_text(
                project,
                source_relative,
                surface.content,
                should_cancel=should_cancel,
            )
            source_file = f"{project.workspace}/{source_relative}"
        entry: JsonObject = {
            "id": surface.id,
            "kind": surface.kind.value,
            "content_hash": surface.content_hash,
            **content_context,
        }
        if surface.budget is not None:
            entry["budget"] = surface.budget
        if surface.path is not None:
            entry["path"] = surface.path
            entry["source_file"] = source_file
        surface_manifest_relative = f"{surface_dir}/manifest.json"
        _write_project_text(
            project,
            surface_manifest_relative,
            json.dumps(entry, indent=2),
            should_cancel=should_cancel,
        )
        surface_index.append(
            {
                "id": surface.id,
                "kind": surface.kind.value,
                "content_hash": surface.content_hash,
                "manifest_file": f"{project.workspace}/{surface_manifest_relative}",
            }
        )
    index_content = json.dumps(surface_index, indent=2)
    index_context = _materialize_context_content(
        project,
        index_content,
        directory=f"{context_dir}/parent-surface-index",
        extension=".json.part",
        should_cancel=should_cancel,
    )
    index_manifest_relative = f"{context_dir}/parent-surfaces.json"
    _write_project_text(
        project,
        index_manifest_relative,
        json.dumps(
            {
                "kind": "parent-surface-index",
                "format": "json-array",
                "entry_count": len(surface_index),
                **index_context,
            },
            indent=2,
        ),
        should_cancel=should_cancel,
    )
    return {
        "name": parent.name,
        "version": parent.version,
        "doc_hash": parent.doc_hash,
        "source_root": f"{project.workspace}/{context_dir}/parent-source",
        "surface_count": len(surface_index),
        "surface_index_manifest": f"{project.workspace}/{index_manifest_relative}",
        "content_contract": (
            "Read surface_index_manifest, then concatenate its content_files and parse that JSON "
            "array. Each index entry points to one independently readable surface manifest. "
            "Within a surface manifest, concatenate content_files exactly to reconstruct the "
            "surface. Pathful code is also mirrored beneath source_root at its exact path."
        ),
    }


def _materialize_context_content(
    project: AgentProject,
    content: str,
    *,
    directory: str,
    extension: str,
    include_contract: bool = True,
    should_cancel: Callable[[], bool] | None = None,
) -> JsonObject:
    """Write exact ordered chunks that each fit in one project ``read_file`` result."""
    chunk_count = max(1, math.ceil(len(content) / _CONTEXT_CONTENT_CHUNK_CHARS))
    width = max(3, len(str(chunk_count)))
    content_files: list[str] = []
    for chunk_index in range(chunk_count):
        start = chunk_index * _CONTEXT_CONTENT_CHUNK_CHARS
        chunk = content[start : start + _CONTEXT_CONTENT_CHUNK_CHARS]
        relative = (
            f"{directory}/part-{chunk_index + 1:0{width}d}-of-{chunk_count:0{width}d}{extension}"
        )
        _write_project_text(
            project,
            relative,
            chunk,
            should_cancel=should_cancel,
        )
        content_files.append(f"{project.workspace}/{relative}")
    result: JsonObject = {
        "content_length": len(content),
        "content_files": content_files,
    }
    if include_contract:
        result["content_contract"] = (
            "Read content_files in listed order and concatenate them exactly. Each file is "
            "independently readable without truncation."
        )
    return result


def _project_history_entry(
    delta: HarnessDelta,
    *,
    proposal_file: str | None,
    evaluation_dir: str | None,
    workspace: str,
) -> JsonObject:
    """Compact judged metadata while raw proposals retain exact replacement payloads.

    Re-serializing every prior full code surface into every later iteration makes persistent history
    quadratic in run length. The project already owns each raw proposal file, so history carries
    queryable identities, rationales, sizes, and verdicts plus a direct pointer to the exact bytes.
    """
    ops: list[JsonObject] = []
    for op in delta.ops:
        item: JsonObject = {
            "op": op.op,
            "surface_id": op.surface_id,
            "rationale": op.rationale[:2_000],
            "content_length": len(op.content) if op.content is not None else 0,
        }
        if op.kind is not None:
            item["kind"] = op.kind.value
        if op.path is not None:
            item["path"] = op.path
        if op.budget is not None:
            item["budget"] = op.budget
        ops.append(item)
    return {
        "delta_id": delta.delta_id,
        "parent_doc_hash": delta.parent_doc_hash,
        "child_doc_hash": delta.child_doc_hash,
        "trigger": delta.trigger.model_dump(mode="json"),
        "preconditions": dict(delta.preconditions),
        "expected_effect": delta.expected_effect[:2_000],
        "ops": ops,
        "verdict": delta.verdict.model_dump(mode="json") if delta.verdict is not None else None,
        "proposal_file": proposal_file,
        "evaluation_dir": (f"{workspace}/{evaluation_dir}" if evaluation_dir is not None else None),
        "content_contract": (
            "Exact op content remains in proposal_file; this entry intentionally omits it to "
            "keep cumulative judged history linear and fast."
        ),
    }


def _stamp_project_preconditions(
    parent: HarnessDoc, proposal: HarnessDelta | None
) -> HarnessDelta | None:
    """Stamp missing concurrency metadata from the exact project-iteration parent.

    The ordinary agent still chooses every semantic operation. The host owns this mechanical
    identity field because it wrote the immutable parent snapshot for the same iteration.
    An explicitly supplied but incorrect hash is preserved so normal validation rejects it.
    """
    if proposal is None:
        return None
    for op in proposal.ops:
        if op.op not in ("replace", "remove") or op.surface_id in proposal.preconditions:
            continue
        surface = parent.surface(op.surface_id)
        if surface is not None:
            proposal.preconditions[op.surface_id] = surface.content_hash
    return proposal


def _validate_project_proposals(
    project: AgentProject,
    parent: HarnessDoc,
    trigger: FailureSignature,
    *,
    proposal_dir: str,
    slots: list[int],
    history: list[HarnessDelta],
    valid_proposals: dict[int, HarnessDelta] | None = None,
    preserve_runtime_kind: bool,
    should_cancel: Callable[[], bool] | None,
) -> _ProposalValidation:
    """Parse, stamp, apply, and de-duplicate selected project proposal slots.

    Applying a deep copy exercises the complete typed ``HarnessDoc`` boundary without stamping
    ``child_doc_hash`` onto the delta the search will later apply and archive. Previously the
    project proposer returned syntactically parsed deltas and left this check to the search loop,
    where a missing skill frontmatter block consumed an iteration as ``invalid before eval``.
    """
    accepted = dict(valid_proposals or {})
    raw_files: dict[int, str] = {}
    child_hashes: dict[int, str] = {}
    errors: dict[int, str] = {}
    history_ids = {delta.delta_id for delta in history}
    history_child_hashes = {
        delta.child_doc_hash for delta in history if delta.child_doc_hash is not None
    }
    sibling_ids = {delta.delta_id: index for index, delta in accepted.items()}
    for accepted_index, accepted_delta in accepted.items():
        accepted_child = apply_delta(
            parent,
            accepted_delta.model_copy(deep=True),
            f"{parent.name}-accepted-preflight-{accepted_index:02d}",
        )
        child_hashes[accepted_index] = accepted_child.doc_hash
    sibling_child_hashes = {child_hash: index for index, child_hash in child_hashes.items()}
    parent_runtime_kind = parent.runtime_kind()
    for index in slots:
        _check_cancelled(should_cancel)
        relative = f"{proposal_dir}/proposal-{index:02d}.json"
        try:
            raw = _read_project_text(project, relative, should_cancel=should_cancel)
        except HarnessSearchCancelled:
            raise
        except Exception as error:  # noqa: BLE001 - one missing file is one repairable slot
            errors[index] = f"proposal file is missing or unreadable: {error}"
            continue
        raw_files[index] = raw
        proposal = parse_delta(parent, trigger, raw)
        if proposal is None:
            errors[index] = "proposal is not a parseable typed delta JSON object"
            continue
        stamped = _stamp_project_preconditions(parent, proposal)
        assert stamped is not None
        try:
            child = apply_delta(
                parent,
                stamped.model_copy(deep=True),
                f"{parent.name}-proposal-preflight-{index:02d}",
            )
            child_runtime_kind = child.runtime_kind()
            if child_runtime_kind not in _SUPPORTED_RUNTIME_KINDS:
                supported = ", ".join(sorted(_SUPPORTED_RUNTIME_KINDS))
                raise ValueError(
                    f"project proposal resolves to unsupported runtime kind "
                    f"{child_runtime_kind!r}; choose one of: {supported}"
                )
            if preserve_runtime_kind and child_runtime_kind != parent_runtime_kind:
                raise ValueError(
                    "project proposals must preserve the parent's runtime kind "
                    f"{parent_runtime_kind!r}; this proposal resolves to {child_runtime_kind!r}"
                )
        except ValueError as error:
            errors[index] = f"delta does not apply to the supplied parent: {error}"
            continue
        child_hash = child.doc_hash
        if child_hash == parent.doc_hash:
            errors[index] = "delta is a semantic no-op: its child document equals the parent"
            continue
        if stamped.delta_id in history_ids:
            errors[index] = (
                f"delta {stamped.delta_id} duplicates a proposal already present in judged history"
            )
            continue
        if child_hash in history_child_hashes:
            errors[index] = (
                f"child document {child_hash} duplicates a proposal already present in "
                "judged history"
            )
            continue
        duplicate_slot = sibling_ids.get(stamped.delta_id)
        if duplicate_slot is not None:
            errors[index] = (
                f"delta {stamped.delta_id} duplicates valid sibling proposal-{duplicate_slot:02d}"
            )
            continue
        duplicate_child_slot = sibling_child_hashes.get(child_hash)
        if duplicate_child_slot is not None:
            errors[index] = (
                f"child document {child_hash} duplicates valid sibling "
                f"proposal-{duplicate_child_slot:02d}"
            )
            continue
        accepted[index] = stamped
        sibling_ids[stamped.delta_id] = index
        child_hashes[index] = child_hash
        sibling_child_hashes[child_hash] = index
    _check_cancelled(should_cancel)
    return _ProposalValidation(
        proposals=accepted,
        raw_files=raw_files,
        child_hashes=child_hashes,
        errors=errors,
    )


def _proposal_validation_report(
    *,
    parent: HarnessDoc,
    proposal_dir: str,
    attempt: int,
    valid_slots: dict[int, HarnessDelta],
    errors: dict[int, str],
) -> str:
    """Serialize actionable per-slot host validation for the project and run audit."""
    return json.dumps(
        {
            "kind": "proposal-validation",
            "attempt": attempt,
            "parent_doc_hash": parent.doc_hash,
            "parent_runtime_kind": parent.runtime_kind(),
            "valid_slots": sorted(valid_slots),
            "errors": [
                {
                    "slot": index,
                    "proposal_file": f"{proposal_dir}/proposal-{index:02d}.json",
                    "reason": errors[index],
                }
                for index in sorted(errors)
            ],
        },
        indent=2,
    )


def _project_repair_request(
    *,
    workspace: str,
    validation_path: str,
    request_path: str,
    proposal_dir: str,
    errors: dict[int, str],
    valid_slots: dict[int, HarnessDelta],
    runtime_kind: str,
    preserve_runtime_kind: bool,
    repair_turn: int,
) -> str:
    """Render one of the bounded repair turns for only invalid batch slots."""
    invalid_outputs = "\n".join(
        f"- {workspace}/{proposal_dir}/proposal-{index:02d}.json" for index in sorted(errors)
    )
    protected_outputs = "\n".join(
        f"- {workspace}/{proposal_dir}/proposal-{index:02d}.json" for index in sorted(valid_slots)
    )
    if not protected_outputs:
        protected_outputs = "- (none)"
    runtime_constraint = (
        f"preserve its resolved runtime kind {runtime_kind!r}"
        if preserve_runtime_kind
        else "produce a valid resolved runtime kind"
    )
    return f"""Repair exactly {len(errors)} invalid proposal slot(s) from this iteration.
This is repair turn {repair_turn} of {_MAX_PROJECT_REPAIR_TURNS}.

Read the host validation report: {workspace}/{validation_path}
It contains the exact error for each invalid slot. Re-read the original iteration request at
{workspace}/{request_path} and its supplied parent manifests as needed, then rewrite ONLY these
invalid files:
{invalid_outputs}

These siblings already passed host preflight. Do not rewrite them:
{protected_outputs}

Every repaired file must follow the original typed delta JSON schema, apply cleanly to that same
parent, {runtime_constraint}, produce a child document different from the parent, differ from
judged history, and differ from every sibling. A skill's content must include the complete
four-line frontmatter shown in the original request. Rewrite every invalid file immediately after
reading the validation report; only then spend remaining actions on optional evidence. Validate
every rewritten file before calling submit with a short summary."""


def _project_request(
    *,
    workspace: str,
    context_dir: str,
    proposal_dir: str,
    count: int,
    runtime_kind: str,
    preserve_runtime_kind: bool,
) -> str:
    """Render one filesystem-first proposal task for the ordinary meta agent."""
    absolute_context = f"{workspace}/{context_dir}"
    absolute_proposals = f"{workspace}/{proposal_dir}"
    outputs = "\n".join(
        f"- {absolute_proposals}/proposal-{index:02d}.json" for index in range(1, count + 1)
    )
    runtime_constraint = (
        f"This project must preserve the parent's resolved runtime kind {runtime_kind!r}; do not "
        "add, replace, or remove runtime-kind in a way that changes it."
        if preserve_runtime_kind
        else (
            "A runtime-kind edit is allowed only when the resulting child remains a valid harness; "
            "the search backend makes the final executability decision."
        )
    )
    heading = (
        f"Produce exactly {count} independent harness proposals for this optimization iteration."
    )
    return f"""{heading}

Read:
- parent manifest: {absolute_context}/parent.json
  - follow surface_index_manifest to find every independently readable surface manifest
  - each surface manifest lists ordered content_files; concatenate them to inspect exact content
  - pathful code is also mirrored under source_root with exact source_file paths for direct reads
- failure evidence manifest: {absolute_context}/evidence.json
  - read its content_files in listed order and concatenate them exactly
- judged history manifest: {absolute_context}/history.json
  - read its content_files in listed order, concatenate them exactly, then parse the JSON array
- earlier raw proposals, when useful: {workspace}/proposals/
- earlier candidate evaluation manifests and traces: {workspace}/evaluations/

Write exactly these files, without changing earlier iterations:
{outputs}

Each file must be one JSON object:
{{"expected_effect":"<falsifiable prediction>",
 "preconditions":{{"<surface id>":"<hash copied from parent>"}},
 "ops":[{{"op":"add|replace|remove","surface_id":"<kind:slug>",
           "kind":"<required for add>","content":"<full content>",
           "rationale":"<why this helps>"}}]}}

For a replacement, you may omit content and use compact exact edits instead:
"edits":[{{"old":"<nonempty text occurring exactly once>","new":"<replacement>"}}].
The optimizer expands those edits against the parent before validation.

Typed surface constraints (host preflight enforces all of these before evaluation):
- Every surface id is `<kind>:<kebab-slug>` and its prefix must exactly match `kind`.
- `add` needs a fresh id, `kind`, full `content`, and a nonempty `rationale`. `replace` needs an
  existing id, full `content` or exact `edits`, and a nonempty `rationale`; if it declares `kind`,
  that kind must match the parent. `remove` needs an existing id and rationale and must omit
  content. Every replace/remove target must have its exact parent hash in `preconditions`.
- A `skill:<slug>` add/replace has kind `skill`; its content is the complete markdown below,
  beginning at the first character, with kebab-case `name` exactly equal to `<slug>`:
  ---
  name: <slug>
  description: <one-line description of when the agent should use this skill>
  ---
  <nonempty reusable technique body>
- Prompt content is plain text, and the child must retain at least one prompt surface.
- `tool_policy:main` is one registered tool name per line and must retain `submit`.
- Supported scalar params are `param:max-turns` and `param:max-output-tokens` (integers >= 1),
  `param:temperature` (number in [0, 2]), and `param:runtime-kind` (`kit-python` or `pi-node`).
  {runtime_constraint}
- A path-less code surface can only be `code:runtime` and must remain valid Python defining
  `run(kit)`. Pathful code surfaces use safe relative paths without `..`; replacements inherit the
  parent's path unless explicitly supplied, and paths must stay unique.
- Respect each surface's character budget. Do not remove required singleton surfaces or create
  duplicate ids/paths. Do not emit a semantic no-op or repeat any child document from judged
  history or another sibling, even through differently ordered operations.

Every proposal must be focused, valid against the same supplied parent, and meaningfully different
from its siblings. Your project tool budget is bounded: after reading the three root manifests,
write a complete, parseable draft to every output before doing deeper optional exploration. Keep
those files valid as you refine them. The host will parse, stamp mechanical missing preconditions,
deep-copy apply, and de-duplicate every file. Invalid slots receive at most two repair turns and
are never evaluated. After all files exist, call submit with a short summary."""
