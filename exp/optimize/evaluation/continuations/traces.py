"""Turn saved continuation trajectories into canonical trace imports under world authority.

Every trajectory producer (a study, a campaign, an actor panel, an environment comparison)
persists its episodes as continuation-study branches. This module is the one path from those
saved branches back to trace evidence that ``exp build PROJECT --import-id ID`` can learn from.
It reads only saved artifacts: no runtime code, world or provider is opened.

Authority. Each executed step carries the world receipt of the backend that answered it, and
only a reply whose receipt names a ``native`` owner is source evidence. A branch whose own steps
include at least one native reply and whose conversation, inherited prefix included, holds no
simulated reply is exported as ``generated-native``: any reply without proven native authority
(an unverifiable prefix, a backend without receipts) is excluded from the trace and marked
``excluded`` in its step ledger, so the call stays visible with no recorded reply and learning
skips it. A branch with any simulated reply is exported as ``generated-simulated`` with every
reply retained for inspection, because its later actions and its verifier grade over the final
world both depend on simulated execution; its traces carry ``simulation`` provenance, and the
build refuses them as learning source. A branch that executed no step with known authority
carries no world evidence and is skipped.

Lineage. Every fork and continuation of one captured task declares the source task's lineage
group as its conversation and as its declared task identity (``task_lineage``), under one
customer boundary per source project. A build therefore keeps all of them in one leakage group,
fit or held out together, and the root exposure ledger joins them by declared task identity, so
a reservation or exposure of one fork covers the rest. Studies only run from fit tasks of the
source build (the study owner refuses reserved ones), and the export checks each study's task
again against the current build, because a rebuild can move a fit lineage to held out; so no
export carries held-out evidence.

Admission. Every export applies one registered admission policy, which every study it judges
must cite by pinning it as ``admission_policy`` before running. Each branch's per-dimension
verdicts and overall verdict are recorded in its occurrence under
``exp.trace.metadata.occurrences[].admission`` and counted per branch in the summary;
``calibrations`` names saved comparisons under the same policy that may admit simulated
observations. Admission is provenance for later consumers: refused and unassessable branches
are still exported, so the evidence denominator stays complete, and verdict wording never
changes a trace's content identity.

Identity. Identical trajectories collapse into one trace whose provenance lists every
occurrence. Trace IDs derive from content, and each authority is persisted as one immutable,
content-addressed import, so an unchanged export replays its existing import.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactInput,
    ContractModel,
    JsonObject,
    JsonValue,
    SourceIdentity,
    sha256_json,
    stable_id,
)
from exp.common.models import AssistantAction, ModelMessage, ModelRequest, ModelResponse, ToolCall
from exp.common.project import ArtifactStore, ArtifactStoreError, ProjectStore, artifact_input
from exp.common.project.paths import validate_local_id
from exp.common.traces.ingest.chat_json import CHAT_JSON_SOURCE
from exp.common.traces.ingest.otlp import TraceNormalizationIssue
from exp.common.traces.ingest.persistence import persist_normalized_import
from exp.common.traces.ingest.streaming import normalization_workspace
from exp.common.traces.ingest.trace_extensions import (
    OUTCOME_FAILURE_CODE_KEY,
    OUTCOME_FAILURE_MESSAGE_KEY,
    OUTCOME_NAME_KEY,
    OUTCOME_STATUS_KEY,
    REQUEST_CONTEXT_KEY,
)
from exp.common.traces.ingest.vendor_records import VendorTraceFormatError
from exp.optimize.evaluation.continuations.admission import require_pinned_policy
from exp.optimize.evaluation.continuations.prefix import _executed_parent
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.continuations.study import ContinuationBranch, ContinuationStudy
from exp.optimize.evaluation.equivalence.dimensions import StepOwner, step_owner, world_identity
from exp.optimize.evaluation.equivalence.policy import (
    AdmissionPolicy,
    Verdict,
    load_admission_policy,
)
from exp.optimize.evaluation.equivalence.verdicts import (
    CalibrationEvidence,
    admit_trajectory,
    policy_calibrations,
)
from exp.runtime.environments import Observation
from exp.simulation.continuations import ContinuationCheckpoint, load_continuation_checkpoint
from exp.simulation.development_scope import require_development_task
from exp.simulation.mining.exposure_ledger import TASK_LINEAGE_KEY

TRAJECTORY_TRACE_PROTOCOL = "generated-trajectory-v2"
Authority = Literal["generated-native", "generated-simulated"]
_KIND: dict[Authority, Literal["native-world", "simulation"]] = {
    "generated-native": "native-world",
    "generated-simulated": "simulation",
}
_REPLAY_FROM_KEY = "replay_from"


class TrajectoryTraceExport(ContractModel):
    """Saved trajectory generators to export as trace imports, with no provider calls.

    Attributes:
        generators: Exact saved results: a continuation study, campaign, checkpoint
            experiment, actor panel, environment comparison or any report whose cells bind
            continuation-study results.
        policy: Registered admission policy applied to every trajectory; it must be
            registered before every study the generators reach.
        calibrations: Saved ``compare-environments`` reports registered under the same policy;
            only their calibration can admit a simulated observation.
        project: Project receiving the imports, so ``exp build PROJECT --import-id`` accepts
            them. Defaults to the project holding the generators.
    """

    generators: tuple[ArtifactInput, ...] = Field(min_length=1)
    policy: ArtifactInput
    calibrations: tuple[ArtifactInput, ...] = ()
    project: str | None = None


class TrajectorySkip(ContractModel):
    """One saved branch that produced no trace, and why.

    Attributes:
        study: Study result holding the branch.
        branch: Planned branch index.
        reason: Concrete cause a reader can act on.
    """

    study: str
    branch: int
    reason: str


class TrajectoryImport(ContractModel):
    """One persisted, or in a dry run planned, import for one authority.

    Attributes:
        authority: ``generated-native`` learning evidence or labelled ``generated-simulated``.
        traces: Canonical traces in the import.
        import_id: Immutable import ID; None in a dry run.
        issues: Normalization exclusions retained with the import.
    """

    authority: Authority
    traces: int
    import_id: str | None = None
    issues: tuple[str, ...] = ()


class TrajectoryTraceSummary(ContractModel):
    """Counts by authority for one export, with every skipped branch.

    Attributes:
        project: Project the imports are associated with.
        studies: Continuation-study results the generators reached.
        branches: Saved branches considered.
        native: Native traces, including mixed ones.
        mixed: Native traces with at least one excluded reply of unproven authority.
        simulated: Traces labelled ``generated-simulated``.
        duplicates: Branches collapsed into an identical trajectory's trace.
        skipped: Branches that produced no trace.
        imports: One entry per authority with at least one trace.
        policy: Admission policy applied.
        admission: Exported branches per overall admission verdict.
    """

    project: str
    studies: int
    branches: int
    native: int
    mixed: int
    simulated: int
    duplicates: int
    skipped: tuple[TrajectorySkip, ...] = ()
    imports: tuple[TrajectoryImport, ...] = ()
    policy: ArtifactInput
    admission: dict[Verdict, int]


@dataclass(frozen=True)
class _Admission:
    """The registered policy and calibrations one export applies.

    Attributes:
        reference: Registered policy reference.
        policy: The registered policy.
        calibrations: Supplied calibrations under the policy.
    """

    reference: ArtifactInput
    policy: AdmissionPolicy
    calibrations: tuple[CalibrationEvidence, ...]


@dataclass(frozen=True)
class _Reply:
    """Authority and error status of one recorded tool reply.

    Attributes:
        owner: Backend authority that produced the reply, or unknown without proof.
        is_error: Whether the observation reported an error.
    """

    owner: StepOwner
    is_error: bool = False


@dataclass(frozen=True)
class _Entry:
    """One actor-visible message with the authority of a tool reply.

    Attributes:
        message: Exact message the actor saw.
        phase: Inherited checkpoint prefix or this branch's own execution.
        reply: Authority of a tool message; None for other roles.
    """

    message: ModelMessage
    phase: Literal["prefix", "branch"]
    reply: _Reply | None = None


@dataclass
class _Trajectory:
    """One exported trajectory and every branch that produced exactly its content.

    Attributes:
        authority: Export authority.
        conversation: Chat conversation with provenance extensions, minus occurrences.
        mixed: Whether a reply was excluded.
        occurrences: Per-branch provenance of identical trajectories, each with its verdict.
        verdicts: Overall admission verdict of each occurrence, in the same order.
    """

    authority: Authority
    conversation: JsonObject
    mixed: bool
    verdicts: list[Verdict] = field(default_factory=list)
    occurrences: list[JsonObject] = field(default_factory=list)


def export_trajectory_traces(
    project: ProjectStore, spec: TrajectoryTraceExport, *, dry_run: bool = False
) -> TrajectoryTraceSummary:
    """Convert every branch the generators reached and persist one import per authority.

    Args:
        project: Project holding the generators and their studies.
        spec: Generators and the receiving project.
        dry_run: Count and normalize in private storage without writing an import.

    Returns:
        Counts by authority, skipped branches and the import of each authority.

    Raises:
        ValueError: A generator differs from its saved artifact or reaches no study, the
            policy is not the one a study pinned, a calibration is not under the policy, or
            a study's task is no longer development data under the project's current build.
    """
    destination = validate_local_id(spec.project or project.paths.project_id, label="project ID")
    store = project.artifacts
    admission = _Admission(
        spec.policy,
        load_admission_policy(store, spec.policy),
        policy_calibrations(project, spec.calibrations, spec.policy),
    )
    studies: dict[str, tuple[ArtifactInput, list[str]]] = {}
    for generator in spec.generators:
        for reference in trajectory_studies(store, generator):
            studies.setdefault(reference.artifact_id, (reference, []))[1].append(
                generator.artifact_id
            )
    trajectories: dict[str, _Trajectory] = {}
    skipped: list[TrajectorySkip] = []
    branches = duplicates = 0
    for study_id in sorted(studies):
        reference, generators = studies[study_id]
        study = load_study(store, reference)
        require_pinned_policy(store, reference, study, spec.policy)
        require_development_task(
            store, load_continuation_checkpoint(store, study.spec.checkpoint).task
        )
        for branch in study.branches:
            branches += 1
            converted = _trajectory(project, study, branch, reference, generators, admission)
            if isinstance(converted, TrajectorySkip):
                skipped.append(converted)
                continue
            trace_id = str(converted.conversation["trace_id"])
            if trace_id in trajectories:
                duplicates += 1
                trajectories[trace_id].occurrences.extend(converted.occurrences)
                trajectories[trace_id].verdicts.extend(converted.verdicts)
            else:
                trajectories[trace_id] = converted
    imports = tuple(
        _persist(project, destination, authority, rows, spec.generators, dry_run=dry_run)
        for authority in ("generated-native", "generated-simulated")
        if (rows := [row for row in trajectories.values() if row.authority == authority])
    )
    return TrajectoryTraceSummary(
        project=destination,
        studies=len(studies),
        branches=branches,
        native=sum(row.authority == "generated-native" for row in trajectories.values()),
        mixed=sum(row.mixed for row in trajectories.values()),
        simulated=sum(row.authority == "generated-simulated" for row in trajectories.values()),
        duplicates=duplicates,
        skipped=tuple(skipped),
        imports=imports,
        policy=spec.policy,
        admission={
            verdict: sum(row.verdicts.count(verdict) for row in trajectories.values())
            for verdict in ("admitted", "refused", "unassessable")
        },
    )


def trajectory_studies(store: ArtifactStore, generator: ArtifactInput) -> tuple[ArtifactInput, ...]:
    """Find the continuation-study results one saved generator produced.

    The walk follows result and cell artifacts only, the shape every trajectory producer
    saves, so registrations, checkpoints and evidence a study merely consumed are never
    mistaken for its output.

    Args:
        store: Project artifact store.
        generator: Exact saved study, campaign, experiment, panel or comparison result.

    Returns:
        Study result references in artifact ID order.

    Raises:
        ValueError: The reference differs from the saved artifact or reaches no study.
    """
    if artifact_input(store.read(generator.artifact_id).manifest) != generator:
        raise ValueError(
            f"generator {generator.artifact_id} differs from its saved artifact; pass the "
            "reference its producing command printed"
        )
    found: dict[str, ArtifactInput] = {}
    pending, seen = [generator], set[str]()
    while pending:
        reference = pending.pop()
        if reference.artifact_id in seen:
            continue
        seen.add(reference.artifact_id)
        manifest = store.read(reference.artifact_id).manifest
        result = reference.artifact_id.endswith("-result")
        if manifest.artifact_type == "continuation-study":
            if result:
                found[reference.artifact_id] = reference
            continue
        if result or manifest.artifact_type.endswith("-cell"):
            pending.extend(manifest.inputs)
    if not found:
        raise ValueError(
            f"generator {generator.artifact_id} reaches no saved continuation study; pass a "
            "study, campaign, actor panel or environment comparison result"
        )
    return tuple(found[key] for key in sorted(found))


def _trajectory(
    project: ProjectStore,
    study: ContinuationStudy,
    branch: ContinuationBranch,
    reference: ArtifactInput,
    generators: Sequence[str],
    admission: _Admission,
) -> _Trajectory | TrajectorySkip:
    """Reconstruct one branch's actor-visible conversation and classify its authority.

    Args:
        project: Source project.
        study: Verified study holding the branch.
        branch: Saved branch, including interrupted ones.
        reference: Study result reference.
        generators: Every requested generator artifact ID that reached the study.
        admission: Policy and calibrations applied to the branch.

    Returns:
        The exportable trajectory, or why the branch produced none.
    """
    store = project.artifacts

    def skip(reason: str) -> TrajectorySkip:
        """Record why this branch produced no trace."""
        return TrajectorySkip(study=reference.artifact_id, branch=branch.index, reason=reason)

    checkpoint = load_continuation_checkpoint(store, study.spec.checkpoint)
    replies, lineage, executed = _prefix_replies(store, study.spec.checkpoint)
    entries = _prefix_entries(checkpoint, replies)
    reconstructed = _branch_entries(study, branch, entries)
    if isinstance(reconstructed, str):
        return skip(reconstructed)
    owners = {
        entry.reply.owner for entry in reconstructed if entry.phase == "branch" and entry.reply
    }
    if not owners & {"native", "simulated"}:
        return skip(
            "the branch executed no step with native or simulated authority evidence, so it "
            "carries no world evidence"
        )
    simulated = any(entry.reply and entry.reply.owner == "simulated" for entry in reconstructed)
    authority: Authority = "generated-simulated" if simulated else "generated-native"
    verdict = admit_trajectory(
        admission.reference,
        admission.policy,
        branch,
        checkpoint=checkpoint,
        verifier=study.spec.verifier,
        actor_model=study.spec.actor_model,
        calibrations=admission.calibrations,
    )
    rendered = _conversation(project, study, branch, checkpoint, reconstructed, authority)
    if isinstance(rendered, str):
        return skip(rendered)
    conversation, mixed = rendered
    manifest = store.read(reference.artifact_id).manifest
    occurrence: JsonObject = {
        "generators": sorted(generators),
        "study": reference.artifact_id,
        "branch": branch.index,
        "branch_status": branch.status,
        "checkpoint": study.spec.checkpoint.artifact_id,
        "fork_lineage": list(lineage),
        "prefix": "executed" if executed else "unverified",
        "code_revision": manifest.code_revision,
        "admission": verdict.model_dump(mode="json"),
    }
    return _Trajectory(authority, conversation, mixed, [verdict.status], [occurrence])


def _prefix_replies(
    store: ArtifactStore, reference: ArtifactInput
) -> tuple[tuple[_Reply, ...], tuple[str, ...], bool]:
    """Recover the authority of every inherited tool reply through executed ancestors.

    Args:
        store: Project artifact store.
        reference: Checkpoint the branch resumed from.

    Returns:
        One reply per prefix tool message, the fork lineage from this checkpoint back to its
        root, and whether every ancestor was verified as executed.
    """
    point = load_continuation_checkpoint(store, reference)
    tools = sum(message.role == "tool" for message in point.actor.messages)
    unknown = tuple(_Reply("unknown") for _ in range(tools))
    if point.actor.calls == 0:
        return unknown, (reference.artifact_id,), not tools
    try:
        parent = _executed_parent(store, reference, point)
    except (ValueError, ArtifactStoreError):
        return unknown, (reference.artifact_id,), False
    inherited, lineage, executed = _prefix_replies(store, parent)
    intent = next(ref for ref in point.sources if ref.artifact_id.endswith("-intent"))
    record = store.read_bytes(intent.artifact_id.removesuffix("-intent"), "record.json")
    steps = ContinuationBranch.model_validate_json(record).outcome.get("steps")
    assert isinstance(steps, list)  # _executed_parent verified the saved observations.
    added = tuple(_step_reply(step) for step in steps[: tools - len(inherited)])
    return (*inherited, *added), (reference.artifact_id, *lineage), executed


def _prefix_entries(checkpoint: ContinuationCheckpoint, replies: Sequence[_Reply]) -> list[_Entry]:
    """Pair the checkpoint's actor messages with the authority of their tool replies.

    Args:
        checkpoint: Resumed checkpoint.
        replies: One reply per tool message, in order.

    Returns:
        Prefix entries in conversation order.
    """
    entries, remaining = [], iter(replies)
    for message in checkpoint.actor.messages:
        reply = next(remaining) if message.role == "tool" else None
        entries.append(_Entry(message, "prefix", reply))
    return entries


def _branch_entries(
    study: ContinuationStudy, branch: ContinuationBranch, prefix: list[_Entry]
) -> list[_Entry] | str:
    """Append the branch's saved actor responses and executed replies to its prefix.

    Args:
        study: Study holding the branch.
        branch: Saved branch.
        prefix: Inherited entries.

    Returns:
        The complete conversation, or why the saved records do not reconstruct it.
    """
    receipts = branch.outcome.get("model_calls", [])
    steps = branch.outcome.get("steps", [])
    if not isinstance(receipts, list) or not isinstance(steps, list):
        return "the branch outcome has no saved actor calls or steps"
    entries, consumed = list(prefix), 0
    for receipt in receipts:
        if not isinstance(receipt, dict) or "response" not in receipt:
            break  # An interrupted call has no response; nothing after it was executed.
        request = ModelRequest.model_validate(receipt.get("request"))
        response = ModelResponse.model_validate(receipt["response"])
        if request.messages != tuple(entry.message for entry in entries):
            return "a saved actor request differs from the reconstructed conversation"
        if response.model != study.spec.actor_model:
            return "a saved actor response comes from another model than the study registered"
        entries.append(
            _Entry(ModelMessage(role="assistant", assistant_action=response.output), "branch")
        )
        for action in response.output.tool_calls:
            if consumed >= len(steps):
                break
            step = steps[consumed]
            consumed += 1
            if not isinstance(step, dict) or ToolCall.model_validate(step.get("action")) != action:
                return "a saved step names a different action than the actor response"
            raw_reply = step.get("reply")
            if step.get("status") != "valid" or not isinstance(raw_reply, dict):
                continue  # Dispatched but unobserved, or refused before dispatch: no reply.
            reply = Observation.model_validate(raw_reply)
            entries.append(
                _Entry(
                    ModelMessage(role="tool", content=reply.content, tool_call_id=action.call_id),
                    "branch",
                    _step_reply(step),
                )
            )
    return entries


def _step_reply(step: JsonValue) -> _Reply:
    """Read the authority of one executed step from its exact world receipt.

    Args:
        step: Saved step with action, reply and the backend's execution receipt.

    Returns:
        The receipt's owner, or unknown when no receipt binds this exact action.
    """
    assert isinstance(step, dict)
    reply = step.get("reply")
    return _Reply(step_owner(step), isinstance(reply, dict) and reply.get("is_error") is True)


def _conversation(
    project: ProjectStore,
    study: ContinuationStudy,
    branch: ContinuationBranch,
    checkpoint: ContinuationCheckpoint,
    entries: Sequence[_Entry],
    authority: Authority,
) -> tuple[JsonObject, bool] | str:
    """Render one trajectory as a chat conversation carrying its provenance extensions.

    Args:
        project: Source project, the trajectory's customer boundary.
        study: Study holding the branch.
        branch: Saved branch.
        checkpoint: Resumed checkpoint.
        entries: Complete reconstructed conversation.
        authority: Export authority deciding which replies are retained.

    Returns:
        The chat-json conversation, without occurrences, and whether a reply was excluded;
        or why a tool reply cannot be paired with the call it answers.
    """
    messages: list[JsonValue] = []
    steps: list[JsonObject] = []
    names: dict[str, list[tuple[str, str, JsonObject]]] = {}
    used: set[str] = set()
    replay_from = 0
    for entry in entries:
        message = entry.message
        if message.role == "assistant":
            action = message.assistant_action or AssistantAction(content=message.content)
            calls: list[JsonValue] = []
            for call in action.tool_calls:
                exported = _unique(call.call_id, used)
                step: JsonObject = {
                    "call_id": exported,
                    "tool": call.name,
                    "phase": entry.phase,
                    "owner": "unknown",
                    "reply": "absent",
                }
                steps.append(step)
                names.setdefault(call.call_id, []).append((exported, call.name, step))
                calls.append(
                    {
                        "id": exported,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, sort_keys=True),
                        },
                    }
                )
            rendered: JsonObject = {"role": "assistant", "content": action.content}
            if calls:
                rendered["tool_calls"] = calls
            messages.append(rendered)
        elif message.role == "tool":
            pending = names.get(message.tool_call_id or "")
            if entry.reply is None or not pending:
                return "a tool reply in the actor context answers no earlier call"
            exported, name, step = pending.pop(0)
            keep = authority == "generated-simulated" or entry.reply.owner == "native"
            step.update(owner=entry.reply.owner, reply="recorded" if keep else "excluded")
            if keep:
                replay_from += entry.phase == "prefix"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": exported,
                        "name": name,
                        "content": message.content or "",
                        "tool_is_error": entry.reply.is_error,
                    }
                )
        else:
            messages.append({"role": message.role, "content": message.content or ""})
    task = checkpoint.task
    model = study.spec.actor_model
    metadata: JsonObject = {
        "protocol": TRAJECTORY_TRACE_PROTOCOL,
        "authority": authority,
        "task": {
            "task_id": task.task_id,
            "lineage_group_id": task.lineage_group_id,
            "partition": task.partition,
            "origin": task.origin,
            "source_trace_ids": list(task.source_trace_ids),
        },
        "world": world_identity(checkpoint),
        "actor_model": model.model_dump(mode="json"),
        "verifier": None
        if branch.assessment is None
        else branch.assessment.model_dump(mode="json"),
        "steps": list[JsonValue](steps),
    }
    conversation: JsonObject = {
        "messages": messages,
        "tools": [tool.model_dump(mode="json") for tool in task.tools],
        "provider": model.provider,
        "model": model.model_id,
        "exp.conversation.id": task.lineage_group_id,
        "exp.customer.id": f"project:{project.paths.project_id}",
        REQUEST_CONTEXT_KEY: {
            TASK_LINEAGE_KEY: task.lineage_group_id,
            _REPLAY_FROM_KEY: replay_from,
        },
        **_outcome(branch),
    }
    if model.revision is not None:
        conversation["model_revision"] = model.revision
    conversation["trace_id"] = stable_id(
        "generated-trace", {"conversation": conversation, "metadata": metadata}
    )
    conversation["exp.trace.metadata"] = metadata
    return conversation, any(step["reply"] == "excluded" for step in steps)


def _unique(call_id: str, used: set[str]) -> str:
    """Return a call ID unique within one trace, suffixing an actor's repeated ID.

    Args:
        call_id: Actor-issued call ID.
        used: IDs already exported in this trace; updated in place.

    Returns:
        The original ID, or the first free ``ID~N``.
    """
    candidate, index = call_id, 1
    while candidate in used:
        candidate, index = f"{call_id}~{index}", index + 1
    used.add(candidate)
    return candidate


def _outcome(branch: ContinuationBranch) -> JsonObject:
    """Map the branch's verifier grade to canonical trace-outcome extensions.

    Args:
        branch: Saved branch.

    Returns:
        Outcome extension attributes; a missing or unassessable grade is unknown.
    """
    grade = branch.assessment
    if grade is not None and grade.status == "pass":
        return {OUTCOME_STATUS_KEY: "success", OUTCOME_NAME_KEY: "verifier-pass"}
    if grade is not None and grade.status == "fail":
        return {
            OUTCOME_STATUS_KEY: "failure",
            OUTCOME_NAME_KEY: "verifier-fail",
            OUTCOME_FAILURE_CODE_KEY: "validation",
            OUTCOME_FAILURE_MESSAGE_KEY: grade.reason,
        }
    status = "no-grade" if grade is None else grade.status
    return {OUTCOME_STATUS_KEY: "unknown", OUTCOME_NAME_KEY: f"verifier-{status}"}


def _persist(
    project: ProjectStore,
    destination: str,
    authority: Authority,
    rows: Sequence[_Trajectory],
    generators: Sequence[ArtifactInput],
    *,
    dry_run: bool,
) -> TrajectoryImport:
    """Normalize one authority's trajectories and persist them as one shared import.

    Args:
        project: Source project; its root holds the shared trace database.
        destination: Project receiving the import.
        authority: Authority of every row.
        rows: Deduplicated trajectories.
        generators: Generators named by the source identity.
        dry_run: Normalize without writing.

    Returns:
        The import's counts, retained issues and ID.
    """
    conversations = sorted(
        (
            {
                **row.conversation,
                "exp.trace.metadata": {
                    **_object(row.conversation["exp.trace.metadata"]),
                    "occurrences": sorted(row.occurrences, key=sha256_json),
                },
            }
            for row in rows
        ),
        key=lambda conversation: str(conversation["trace_id"]),
    )
    ids = sorted(generator.artifact_id for generator in generators)
    label = ids[0] if len(ids) == 1 else stable_id("trajectory-generators", list[JsonValue](ids))
    with normalization_workspace(CHAT_JSON_SOURCE.vendor) as normalized:
        normalized.source = SourceIdentity(
            kind=_KIND[authority],
            source_id=f"{authority}:{label}",
            sha256=sha256_json(list[JsonValue](conversations)),
        )
        for conversation in conversations:
            try:
                normalized.vendor(CHAT_JSON_SOURCE, conversation)
            except VendorTraceFormatError as error:
                normalized.issues.append(
                    TraceNormalizationIssue(str(conversation["trace_id"]), str(error))
                )
        normalized.finish()
        summary, receipt = persist_normalized_import(
            destination,
            root=project.paths.root,
            source_format=authority,
            normalized=normalized,
            dry_run=dry_run,
        )
    return TrajectoryImport(
        authority=authority,
        traces=summary.trace_count,
        import_id=None if receipt is None else receipt.import_id,
        issues=tuple(f"{issue.source_record}: {issue.message}" for issue in summary.issues),
    )


def _object(value: JsonValue) -> JsonObject:
    """Narrow a JSON value this module built as an object."""
    assert isinstance(value, dict)
    return value
