"""Durable Harbor tool access to the existing executable or mixed world runtime."""

import base64
import json
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
from threading import RLock

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    JsonObject,
    sha256_bytes,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.models import ToolCall
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.study import (
    CheckpointSession,
    restore_continuation_world,
)
from exp.optimize.evaluation.executable.loader import ContinuationRuntime, load_runtime_definition
from exp.optimize.evaluation.executable.provider_grant import ProviderGrant, ProviderLimits
from exp.optimize.evaluation.executable.qualification import load_executable_qualification
from exp.optimize.evaluation.executable.task import ExecutableTask, load_executable_task
from exp.runtime.environments.interface import Observation
from exp.runtime.environments.world_artifacts import load_world_image, persist_world_image
from exp.runtime.environments.world_contracts import ExecutableCheckpoint
from exp.runtime.environments.world_outcomes import (
    WorldExecutionReceipt,
    execute_with_recovery,
    execution_receipt,
    known_failed_effects,
)
from exp.simulation.continuations import WorldCheckpoint, load_continuation_checkpoint
from exp.simulation.evaluation.tasks import grade


class ExecutableHarborTrial:
    """One leased trial with private probes and immutable call, submission and grade records.

    A saved reply is never dispatched twice. A durable intent without a response permanently
    closes this trial as unknown, since the native call may have run. New trials restore the
    original checkpoint; they cannot masquerade as resumes of an unknown operation.
    """

    def __init__(
        self,
        project: ProjectStore,
        task_ref: ArtifactInput,
        task: ExecutableTask,
        session: CheckpointSession,
        runtime: ContinuationRuntime,
        *,
        identity: str,
        code_revision: str,
        qualification: ArtifactInput | None,
        registration: ArtifactInput,
    ) -> None:
        """Bind capabilities while the enclosing context holds the cross-process trial lease."""
        self.project, self.task_ref, self.task = project, task_ref, task
        self.session, self.runtime = session, runtime
        self.identity, self.code_revision = identity, code_revision
        self.qualification = qualification
        self.registration = registration
        self._lock = RLock()
        self._steps: list[JsonObject] = []
        self._calls: dict[str, tuple[ToolCall, JsonObject]] = {}
        checkpoint = load_continuation_checkpoint(project.artifacts, task.job.study.checkpoint)
        self._public = checkpoint.task
        self._initial = runtime.probe(checkpoint.world)
        self._failed = False
        self._outcome_unknown = False
        for index in range(task.maximum_tool_calls):
            key = f"{identity}-call-{index}"
            if project.artifacts.exists(key):
                row = json.loads(project.artifacts.read_bytes(key, "record.json"))
                action = ToolCall.model_validate(row["action"])
                self._steps.append(row["step"])
                self._calls[action.call_id] = (action, row["reply"])
            elif project.artifacts.exists(key + "-intent"):
                intent = json.loads(project.artifacts.read_bytes(key + "-intent", "record.json"))
                failed = self._saved(f"call-{index}-failure")
                if failed is not None:
                    step = failed.get("step")
                    if not isinstance(step, dict) or step.get("action") != intent["action"]:
                        raise ValueError("saved failure differs from its original tool intent")
                    self._steps.append(step)
                    raw = step.get("execution")
                    receipt = WorldExecutionReceipt.model_validate(raw) if raw is not None else None
                    self._outcome_unknown = not (
                        step.get("phase") == "tool_dispatch"
                        and receipt is not None
                        and receipt.action_sha256 == sha256_json(intent["action"])
                        and known_failed_effects(receipt)
                    )
                else:
                    self._steps.append({"action": intent["action"], "status": "failure"})
                    self._outcome_unknown = True
                self._failed = True
                break
            else:
                break

    def _save(self, suffix: str, value: JsonObject, inputs: tuple[ArtifactInput, ...] = ()) -> None:
        """Publish immutable evidence without exposing the project store to actors."""
        self.project.artifacts.write_json(
            artifact_id=self.identity + "-" + suffix,
            artifact_type="executable-trial",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=self.code_revision,
                inputs=tuple(
                    sorted(
                        {self.task_ref, self.registration, *inputs}, key=lambda ref: ref.artifact_id
                    )
                ),
            ),
            files={"record.json": value},
        )

    def _saved(self, suffix: str) -> JsonObject | None:
        """Read a completed operation under the live owner lock."""
        key = self.identity + "-" + suffix
        return (
            json.loads(self.project.artifacts.read_bytes(key, "record.json"))
            if self.project.artifacts.exists(key)
            else None
        )

    def tools(self) -> JsonObject:
        """Return only actor-visible tool contracts, never private probes or state."""
        return {"tools": [tool.model_dump(mode="json") for tool in self._public.tools]}

    def call(self, action: ToolCall) -> JsonObject:
        """Commit intent before execution and the world image before acknowledging a reply."""
        with self._lock:
            prior = self._calls.get(action.call_id)
            if prior is not None:
                if prior[0] != action:
                    raise ValueError("call_id was already used for different arguments")
                return deepcopy(prior[1])
            if self._failed or self._saved("answer") is not None:
                raise ValueError("trial is closed or has an unknown dispatch outcome")
            if len(self._steps) >= self.task.maximum_tool_calls:
                raise ValueError("trial tool-call allowance exhausted")
            ordinal = len(self._steps)
            self._save(f"call-{ordinal}-intent", {"action": action.model_dump(mode="json")})
            returned = False
            try:
                observation = (
                    execute_with_recovery(self.session, action)
                    if action.name in {tool.name for tool in self._public.tools}
                    else Observation(content="Tool is not exposed by this task.", is_error=True)
                )
                returned = True
                captured = self.session.checkpoint()
                final_state = self.runtime.probe(captured)
                world = persist_world_image(
                    self.project.artifacts,
                    captured,
                    created_at=datetime.now(UTC),
                    code_revision=self.code_revision,
                )
                step: JsonObject = {
                    "action": action.model_dump(mode="json"),
                    "status": "valid",
                    "reply": observation.model_dump(mode="json"),
                    "after": final_state,
                }
                receipt = execution_receipt(self.session, action)
                if receipt is not None:
                    step["execution"] = receipt.model_dump(mode="json")
                reply = observation.model_dump(mode="json")
                self._save(
                    f"call-{ordinal}",
                    {
                        "action": action.model_dump(mode="json"),
                        "step": step,
                        "reply": reply,
                        "world": world.model_dump(mode="json"),
                    },
                    (world,),
                )
                self._steps.append(step)
                self._calls[action.call_id] = (action, reply)
                return deepcopy(reply)
            except BaseException as error:
                failed: JsonObject = {
                    "action": action.model_dump(mode="json"),
                    "status": "failure",
                    "phase": "tool_observation" if returned else "tool_dispatch",
                }
                receipt = execution_receipt(self.session, action)
                if receipt is not None:
                    failed["execution"] = receipt.model_dump(mode="json")
                self._steps.append(failed)
                self._failed = True
                self._outcome_unknown = returned or not known_failed_effects(receipt)
                self._save(
                    f"call-{ordinal}-failure",
                    {
                        "step": failed,
                        "error_type": type(error).__name__,
                    },
                )
                raise

    def submit(self, answer: JsonObject) -> JsonObject:
        """Freeze a final answer and the evaluator projection; repeat submission must agree."""
        with self._lock:
            if self._failed:
                raise ValueError("failed or unknown execution cannot be submitted as complete")
            saved = self._saved("answer")
            if saved is not None:
                if saved["answer"] != answer:
                    raise ValueError("trial already has a different final answer")
                return {"submitted": True}
            checkpoint = self.session.checkpoint()
            if self.task.output_paths:
                self._output_files(checkpoint)
            state = self.runtime.probe(checkpoint)
            self._save("answer", {"answer": answer, "final_state": state})
            return {"submitted": True}

    def result(self) -> JsonObject:
        """Grade observed goals and policy separately; only a qualified contract emits reward."""
        with self._lock:
            saved = self._saved("result")
            if saved is not None:
                return saved
            answer = self._saved("answer")
            outcome: JsonObject = {
                "initial_state": self._initial,
                "final_state": None if answer is None else answer["final_state"],
                "final_answer": None if answer is None else answer["answer"],
                "steps": self._steps,
                "stop_reason": "failure" if self._failed or answer is None else "completed",
            }
            assessment = grade(self.task.job.study.verifier, outcome)
            qualified = False
            scope = "diagnostic; observable mechanical predicate only"
            if self.qualification is not None:
                report = load_executable_qualification(self.project, self.qualification)
                if report.task != self.task_ref or report.profile_sha256 != self.task.profile():
                    raise ValueError("qualification differs from this exact executable profile")
                qualified, scope = report.qualified, report.scope
            complete = (
                answer is not None and not self._failed and assessment.status != "unassessable"
            )
            result: JsonObject = {
                "status": "completed" if complete and qualified else "incomplete",
                "reward": int(assessment.status == "pass") if complete and qualified else None,
                "mechanical": assessment.model_dump(mode="json"),
                "qualification": "qualified_observable_contract"
                if qualified
                else "diagnostic_only",
                "scope": scope,
                "outcome_unknown": self._outcome_unknown,
                "execution_failed": self._failed,
                "execution_registration": self.registration.model_dump(mode="json"),
            }
            if answer is not None or self._failed:
                self._save(
                    "result", result, () if self.qualification is None else (self.qualification,)
                )
            return result

    def output_files(self) -> JsonObject:
        """Export only declared final files after submission, with exact digest verification."""
        with self._lock:
            if self._saved("answer") is None or self._failed or not self.task.output_paths:
                raise ValueError(
                    "public outputs require a complete submitted trial and declared paths"
                )
            return self._output_files(self.session.checkpoint())

    def _output_files(self, checkpoint: WorldCheckpoint) -> JsonObject:
        """Validate one captured output inventory before submission or actor delivery."""
        worlds = (
            (checkpoint,)
            if isinstance(checkpoint, ExecutableCheckpoint)
            else (checkpoint.native, checkpoint.candidate)
        )
        rows = []
        size = 0
        for path in self.task.output_paths:
            matches = [
                world.image.parts[path]
                for world in worlds
                if world is not None and path in world.image.parts
            ]
            if len(matches) != 1:
                raise ValueError("declared public output is missing or has ambiguous ownership")
            payload = matches[0]
            size += len(payload)
            if size > 32 * 1024 * 1024:
                raise ValueError("public output exceeds the 32 MiB allowance")
            rows.append(
                {
                    "path": path,
                    "content_base64": base64.b64encode(payload).decode("ascii"),
                    "sha256": sha256_bytes(payload),
                }
            )
        return {"files": rows}


@contextmanager
def open_executable_harbor_trial(
    project: ProjectStore,
    task_ref: ArtifactInput,
    *,
    trial_id: str,
    code_revision: str,
    qualification: ArtifactInput | None = None,
    provider_limits: ProviderLimits | None = None,
) -> Iterator[ExecutableHarborTrial]:
    """Restore the last committed world and exclusively serve one durable trial identity."""
    task = load_executable_task(project, task_ref)
    if project.artifacts.read(task_ref.artifact_id).manifest.code_revision != code_revision:
        raise ValueError(
            "executable task execution revision changed; register and qualify a new task "
            "under the current implementation"
        )
    checkpoint = load_continuation_checkpoint(project.artifacts, task.job.study.checkpoint)
    if qualification is not None:
        report = load_executable_qualification(project, qualification)
        if report.task != task_ref or report.profile_sha256 != task.profile():
            raise ValueError("qualification differs from the selected task before runtime startup")
    identity = stable_id(
        "executable-trial",
        {
            "task": task_ref.model_dump(mode="json"),
            "trial": trial_id,
            "code": code_revision,
            "qualification": None
            if qualification is None
            else qualification.model_dump(mode="json"),
        },
    )
    store = project.artifacts
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        start: JsonObject = {
            "task": task_ref.model_dump(mode="json"),
            "provider_limits": None
            if provider_limits is None
            else provider_limits.model_dump(mode="json"),
        }
        start_id = identity + "-start"
        if store.exists(start_id):
            if json.loads(store.read_bytes(start_id, "record.json")) != start:
                raise ValueError(
                    "trial provider authority changed; resumes must use the original limits"
                )
        else:
            if provider_limits is not None and provider_limits.deadline <= datetime.now(UTC):
                raise ValueError(
                    "new trial provider authority has expired; supply a future deadline"
                )
            store.write_json(
                artifact_id=start_id,
                artifact_type="executable-trial",
                envelope=ArtifactEnvelope(
                    schema_version=1,
                    created_at=datetime.now(UTC),
                    code_revision=code_revision,
                    inputs=(task_ref,),
                ),
                files={"record.json": start},
            )
        registration = artifact_input(store.read(start_id).manifest)
        restored = checkpoint.world
        for index in range(task.maximum_tool_calls):
            key = f"{identity}-call-{index}"
            if not store.exists(key):
                break
            record = json.loads(store.read_bytes(key, "record.json"))
            reference = ArtifactInput.model_validate(record["world"])
            restored = load_world_image(store, reference)
        with (
            load_runtime_definition(
                project,
                runtime_file=task.job.runtime_file,
                runtime_sha256=task.job.runtime_sha256,
                configuration=task.job.configuration,
                probe_sha256=task.job.study.probe_sha256,
                provider_grant=None
                if provider_limits is None
                else ProviderGrant(
                    registration=registration,
                    limits=provider_limits,
                ),
            ) as runtime,
            restore_continuation_world(runtime.world, checkpoint.world_task, restored) as session,
        ):
            yield ExecutableHarborTrial(
                project,
                task_ref,
                task,
                session,
                runtime,
                identity=identity,
                code_revision=code_revision,
                qualification=qualification,
                registration=registration,
            )
