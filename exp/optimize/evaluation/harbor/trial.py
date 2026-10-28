"""Actor-scoped Harbor access to one frozen durable simulator session."""

import base64
import hashlib
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock

from filelock import FileLock

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    JsonObject,
    sha256_json,
    stable_id,
    strict_equal,
)
from exp.common.core.locks import CoordinationLock
from exp.common.evaluations.reconstruction import DualReport, WorkflowEvidence, WorkflowStep
from exp.common.models import ToolCall
from exp.common.project import ArtifactStore, artifact_input
from exp.optimize.evaluation.native_artifacts import (
    NativeMaterializationSpec,
    materialize_native_artifacts,
)
from exp.runtime.environments.artifact_edits import ArtifactEditAdapter, NativeArtifactBinding
from exp.simulation.evaluation.calibration import task_decision
from exp.simulation.evaluation.tasks import EvaluationTask, grade, load_evaluation_task
from exp.simulation.world_model.checkpoints import (
    capture_world_checkpoint,
    load_world_checkpoint,
    state_at,
)
from exp.simulation.world_model.durable import DurableWorldModelSession
from exp.simulation.world_model.runtime import load_grounded_world_model_artifact


@dataclass
class HarborTrial:
    """One trial's public tool access and private authoritative grading.

    Attributes:
        artifacts: Trusted evidence store, never mounted into the actor environment.
        task_ref: Exact public contract and private verifier artifact.
        session: Existing session owner, opened from this task's frozen checkpoint.
        code_revision: Producing application source identity.
        evaluate: Existing dual-evaluator callback using the shared provider owner.
        minimum_scores: Explicit task criterion thresholds, never inferred from ratings.
        calibration: Optional qualified calibration report; absent keeps trials diagnostic.
        native_adapter: Explicit pure transformer matching the public frozen native binding.
    """

    artifacts: ArtifactStore
    task_ref: ArtifactInput
    session: DurableWorldModelSession
    code_revision: str
    evaluate: Callable[[WorkflowEvidence], DualReport] | None = None
    minimum_scores: dict[str, int] = field(default_factory=dict)
    calibration: ArtifactInput | None = None
    native_adapter: ArtifactEditAdapter | None = None
    _native_binding: NativeArtifactBinding | None = field(default=None, init=False)
    _task: EvaluationTask = field(init=False)
    _initial: JsonObject = field(init=False)
    _answer: JsonObject | None = field(default=None, init=False)
    _failed: str | None = field(default=None, init=False)
    _result: JsonObject | None = field(default=None, init=False)
    _lock: RLock = field(default_factory=RLock, init=False)
    _lease: FileLock = field(init=False)
    _submitted_state_sha256: str | None = field(default=None, init=False)
    _identity: str = field(init=False)
    _profile: str = field(init=False)

    def __post_init__(self) -> None:
        """Require an actual frozen fork before exposing any actor operation."""
        self._task = load_evaluation_task(self.artifacts, self.task_ref)
        declared_native = self._task.task.initial_context.get("native_artifact_binding")
        if declared_native is not None:
            self._native_binding = NativeArtifactBinding.model_validate(declared_native)
            if (
                self.native_adapter is None
                or sha256_json(self.native_adapter.definition)
                != self._native_binding.adapter_sha256
            ):
                raise ValueError("native task requires its exact registered artifact adapter")
        elif self.native_adapter is not None:
            raise ValueError("native adapter requires a binding frozen in the public task")
        checkpoint = load_world_checkpoint(self.artifacts, self._task.checkpoint)
        record = self.session.committed_record()
        snapshot = state_at(record, 0)
        if not strict_equal(
            snapshot.business_state, checkpoint.current.business_state
        ) or not strict_equal(list(snapshot.history), list(checkpoint.current.history)):
            raise ValueError("Harbor trial must start from the exact frozen task checkpoint")
        if record.binding.knowledge_sha256 != sha256_json(
            load_grounded_world_model_artifact(self.artifacts, self._task.world_model)
        ):
            raise ValueError("Harbor session belongs to another serving artifact")
        sources = self.session.evaluation_context.get("sources")
        if not isinstance(sources, dict) or not isinstance(
            sources.get("frozen_profile_sha256"), str
        ):
            raise ValueError("Harbor trial requires a source-bound frozen serving profile")
        if sources.get("online_research_policy") is not None:
            raise ValueError(
                "scored Harbor trials require frozen knowledge with no online research"
            )
        self._profile = sources["frozen_profile_sha256"]
        self._initial = deepcopy(snapshot.business_state)
        if any(
            type(score) is not int or not 1 <= score <= 5 for score in self.minimum_scores.values()
        ):
            raise ValueError("criterion thresholds must be declared integers from 1 to 5")
        self._identity = stable_id(
            "harbor-trial",
            {"task": self.task_ref.model_dump(mode="json"), "session": record.session_id},
        )
        self._lease = CoordinationLock(
            self.artifacts.project_directory / (self._identity + ".lock"), timeout=0
        )
        binding: JsonObject = {
            "format_version": 2,
            "code_revision": self.code_revision,
            "frozen_profile_sha256": self._profile,
            "task": self.task_ref.model_dump(mode="json"),
            "session_binding": record.binding.model_dump(mode="json"),
            "minimum_scores": self.minimum_scores,
            "calibration": None
            if self.calibration is None
            else self.calibration.model_dump(mode="json"),
        }
        with self._lock, self._lease:
            if self.artifacts.exists(self._identity):
                if json.loads(self.artifacts.read_bytes(self._identity, "record.json")) != binding:
                    raise ValueError("Harbor trial binding changed; open a new trial")
            else:
                self._save(self._identity, binding)
            self._refresh()

    def _refresh(self) -> None:
        """Read durable terminal records while holding the cross-process trial lease."""
        self._answer, self._result, self._failed = None, None, None
        self._submitted_state_sha256 = None
        for suffix in ("answer", "result", "failure"):
            identity = self._identity + "-" + suffix
            if self.artifacts.exists(identity):
                value = json.loads(self.artifacts.read_bytes(identity, "record.json"))
                if suffix == "answer":
                    self._answer = value["answer"]
                    self._submitted_state_sha256 = value["session_state_sha256"]
                elif suffix == "result":
                    self._result = value
                else:
                    self._failed = value["error_type"]
        if self._submitted_state_sha256 is not None and self._submitted_state_sha256 != sha256_json(
            self.session.committed_record().state
        ):
            raise ValueError("submitted trial state changed; refuse grading and open a new trial")

    @contextmanager
    def _guarded(self) -> Iterator[None]:
        """Serialize bridge owners, then refresh closure before dispatch or grading."""
        with self._lock, self._lease:
            self._refresh()
            yield

    def _save(self, identity: str, value: JsonObject) -> None:
        """Persist through the immutable project owner without storing capabilities."""
        self.artifacts.write_json(
            artifact_id=identity,
            artifact_type="harbor-trial-record",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=self.code_revision,
                inputs=(self.task_ref,),
            ),
            files={"record.json": value},
        )

    def tools(self) -> JsonObject:
        """Return only public interfaces, with no state, references or grading artifacts."""
        return {"tools": [tool.model_dump(mode="json") for tool in self._task.task.tools]}

    def call(self, action: ToolCall) -> JsonObject:
        """Use the durable session's operation identity, admission and atomic commit."""
        with self._guarded():
            if self._answer is not None or self._failed is not None:
                raise ValueError("trial is closed")
            if action.name not in {tool.name for tool in self._task.task.tools}:
                raise ValueError("tool is not part of this public task")
            try:
                observation = self.session.step((action,), operation_id=action.call_id)[0]
            except Exception as error:  # noqa: BLE001 - preserve simulator failure without retry.
                self._failed = type(error).__name__
                self._save(self._identity + "-failure", {"error_type": self._failed})
                raise
            return observation.model_dump(mode="json")

    def submit(self, answer: JsonObject) -> JsonObject:
        """Freeze actor-authored output without accepting actor-authored evidence."""
        with self._guarded():
            if self._answer is not None and not strict_equal(self._answer, answer):
                raise ValueError("trial already has a different final answer")
            if self._answer is None:
                state_sha256 = sha256_json(self.session.committed_record().state)
                self._save(
                    self._identity + "-answer",
                    {"answer": answer, "session_state_sha256": state_sha256},
                )
                self._submitted_state_sha256 = state_sha256
            self._answer = deepcopy(answer)
            return {"submitted": True}

    def output_files(self) -> JsonObject:
        """Return only matched declared output bytes, never private state or oracle evidence."""
        with self._guarded():
            if self._answer is None or self._failed is not None:
                raise ValueError("submit a completed trial before retrieving native outputs")
            if self._native_binding is None or self.native_adapter is None:
                raise ValueError("this task has no native artifact contract")
            checkpoint = capture_world_checkpoint(
                self.session,
                self._task.world_model,
                self.artifacts,
                parent_checkpoint=self._task.checkpoint,
                created_at=datetime.now(UTC),
                code_revision=self.code_revision,
            )
            reference = materialize_native_artifacts(
                self.artifacts,
                NativeMaterializationSpec(
                    checkpoint=artifact_input(self.artifacts.read(checkpoint.artifact_id).manifest),
                    binding=self._native_binding,
                ),
                self.native_adapter,
                code_revision=self.code_revision,
            )
            assert reference is not None
            receipt = json.loads(self.artifacts.read_bytes(reference.artifact_id, "receipt.json"))
            if receipt["status"] != "matched":
                raise ValueError(
                    "native output contradicts the candidate; inspect private divergence evidence"
                )
            files = []
            for path in self._native_binding.output_paths:
                payload = self.artifacts.read_bytes(reference.artifact_id, "outputs/" + path)
                files.append(
                    {
                        "path": path,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "content_base64": base64.b64encode(payload).decode("ascii"),
                    }
                )
            return {"files": files}

    def evidence(self) -> WorkflowEvidence:
        """Build evaluator-only evidence from the durable owner's committed history."""
        steps = []
        for index, transition in enumerate(self.session.transitions):
            actions, replies = transition.get("actions"), transition.get("replies")
            if (
                not isinstance(actions, list)
                or len(actions) != 1
                or not isinstance(replies, list)
                or len(replies) != 1
            ):
                raise ValueError("Harbor profile requires one tool per committed operation")
            action = actions[0]
            if not isinstance(action, dict):
                raise ValueError("committed tool identity is unavailable")
            steps.append(
                WorkflowStep.model_validate(
                    {
                        "step": index,
                        "action": {"name": action["name"], "arguments": action["arguments"]},
                        "status": "valid",
                        "reply": replies[0],
                        "before": transition["before"],
                        "after": transition["after"],
                    }
                )
            )
        return WorkflowEvidence.model_validate(
            {
                **self.session.evaluation_context,
                "workflow_id": self.session.session_id,
                "task": self._task.task.instruction,
                "initial_state": self._initial,
                "steps": [step.model_dump(mode="json") for step in steps],
                "final_answer": self._answer,
                "stop_reason": "simulator_failure"
                if self._failed
                else "completed"
                if self._answer is not None
                else "actor_failure",
            }
        )

    def result(self) -> JsonObject:
        """Compute private grading once, refusing unsupported positive rewards."""
        with self._guarded():
            if self._result is not None:
                return deepcopy(self._result)
            evidence = self.evidence()
            mechanical = grade(self._task.verifier, evidence)
            result: JsonObject = {
                "status": "incomplete",
                "reward": None,
                "mechanical": mechanical.model_dump(mode="json"),
                "qualification": "diagnostic_only",
                "judge_status": "not_run",
                "native_environment_accessed": False,
                "simulator_failure": self._failed,
            }
            if self._answer is None or self._failed is not None:
                return result
            if self.evaluate is not None:
                judged = self.evaluate(evidence)
                if judged.evidence_sha256 != sha256_json(evidence):
                    raise ValueError("judge result belongs to a different actual trial")
                result["judge_status"] = judged.judge_status
                result["judge"] = judged.model_dump(mode="json")
                if self.calibration is not None:
                    manifest = self.artifacts.read(self.calibration.artifact_id).manifest
                    if (
                        artifact_input(manifest) != self.calibration
                        or manifest.artifact_type != "task-calibration"
                    ):
                        raise ValueError("calibration must be an exact task calibration artifact")
                    calibration = json.loads(
                        self.artifacts.read_bytes(self.calibration.artifact_id, "report.json")
                    )
                    if (
                        calibration.get("frozen_profile_sha256") is None
                        or calibration.get("frozen_profile_sha256") != self._profile
                        or calibration.get("task") != self.task_ref.model_dump(mode="json")
                        or calibration.get("candidate_sha256") != sha256_json(self._task.verifier)
                        or calibration.get("judge_sha256") != judged.judge_config_sha256
                        or calibration.get("judge_completion_sha256") is None
                        or calibration.get("judge_completion_sha256")
                        != judged.judge_completion_sha256
                        or calibration.get("minimum_scores") != self.minimum_scores
                    ):
                        raise ValueError("calibration differs from this task, predicate or judge")
                    axes = {axis.dimension_id: axis for axis in judged.axes}
                    if (
                        calibration.get("qualification") == "qualified_in_scope"
                        and bool(self.minimum_scores)
                        and set(self.minimum_scores)
                        == set(calibration.get("required_dimensions", []))
                        and set(self.minimum_scores) <= set(axes)
                        and judged.judge_status == "completed"
                        and all(
                            axis.status == "assessed" and axis.score is not None
                            for axis in (axes[key] for key in self.minimum_scores)
                        )
                        and mechanical.status != "unassessable"
                    ):
                        passed = (
                            task_decision(mechanical.status, judged.axes, self.minimum_scores)
                            == "pass"
                        )
                        result.update(
                            status="completed",
                            reward=int(passed),
                            qualification="qualified_in_scope",
                        )
            if self._native_binding is not None:
                result.update(
                    status="incomplete",
                    reward=None,
                    qualification="diagnostic_only",
                    qualification_blocker="Native artifact verifier calibration is required",
                )
            self._save(self._identity + "-result", result)
            self._result = deepcopy(result)
            return result
