"""Decision points on actually reached states, labelled by the main model's verified outcome.

Every saved actor call in a complete continuation study is one decision point. The actor-visible
request is the reached state, the returned action is the main model's choice of next action (its
first tool, or ``respond``), and the branch's verifier grade is the verified outcome of the
trajectory that choice belongs to. Failed trajectories are kept, so a mistake is labelled by its
failed outcome instead of disappearing; unassessable trajectories stay counted as unknown.

Branches from one checkpoint reach some states more than once. Pooling them within that
checkpoint gives a state-level label for each decision: ``verified_correct`` when some trajectory
taking it from that state passed, ``verified_error`` when every assessed trajectory taking it
failed, and ``unverified`` otherwise. Variants of one task can share a visible state while their
hidden worlds differ, so pooling never crosses checkpoints. A verified-correct decision shows that
a successful continuation exists after it; it does not certify the decision in isolation. Only
the main model's own free choices count: calls it made under a forced tool choice and studies run
with a cascade are excluded.
"""

from collections import defaultdict
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import ArtifactInput, ContractModel, Sha256, sha256_json, stable_id
from exp.common.models import ModelRequest, ModelResponse, ModelSnapshot
from exp.common.project import ArtifactStore
from exp.optimize.evaluation.continuations.actors import ActorPanelReport
from exp.optimize.evaluation.continuations.records import load_study
from exp.runtime.agents.cascade import action_key, action_projection, decision_candidates
from exp.simulation.continuations import load_continuation_checkpoint
from exp.simulation.evaluation.tasks import Verdict

StateLabel = Literal["verified_correct", "verified_error", "unverified"]


def state_request(request: ModelRequest) -> ModelRequest:
    """Return the reached state a decider sees: visible messages and tools, no sampling."""
    return ModelRequest(
        messages=request.messages,
        tools=request.tools,
        tool_choice="auto" if request.tools else None,
    )


class DecisionPoint(ContractModel):
    """One main-model decision at a reached state, with its trajectory's verified outcome.

    Attributes:
        id: Stable point identity over study, branch and call.
        study: Complete saved continuation study.
        branch: Planned branch index within that study.
        call: Exact saved actor request.
        checkpoint: Exact paired actor/world checkpoint the branch continued from; labels and
            observed continuations are pooled only within it.
        lineage: Task lineage, the independent unit.
        task_id: Public task identity.
        state_sha256: Identity of the actor-visible reached state; branches reaching it again
            share it, as does any other world presenting the same visible state.
        candidates: Ordered next-action options at that state.
        main_choice: The main model's decision.
        main_action_sha256: The main model's complete action, ignoring generated call IDs.
        outcome: Verified trajectory outcome; ``unassessable`` when it is unknown.
        policy: Verified trajectory policy compliance; ``unassessable`` when it is unknown.
    """

    id: str
    study: ArtifactInput
    branch: int = Field(ge=0)
    call: ArtifactInput
    checkpoint: ArtifactInput
    lineage: str
    task_id: str
    state_sha256: Sha256
    candidates: tuple[str, ...]
    main_choice: str
    main_action_sha256: Sha256
    outcome: Verdict
    policy: Verdict


class Trajectory(ContractModel):
    """One planned branch of the main model, including one with no decision point.

    Attributes:
        study: Complete saved continuation study.
        branch: Planned branch index within that study.
        lineage: Task lineage, the independent unit.
        outcome: Verified outcome; ``unassessable`` when it is unknown.
        policy: Verified policy compliance; ``unassessable`` when it is unknown.
    """

    study: ArtifactInput
    branch: int = Field(ge=0)
    lineage: str
    outcome: Verdict
    policy: Verdict


class DecisionPointSet(ContractModel):
    """Every decision point of the main model across complete saved studies.

    Attributes:
        main_model: The model whose own trajectories were read.
        studies: Every study read, in source order.
        trajectories: Every planned branch read, including unknown outcomes.
        points: Every decision point, in study, branch and call order.
    """

    main_model: ModelSnapshot
    studies: tuple[ArtifactInput, ...]
    trajectories: tuple[Trajectory, ...]
    points: tuple[DecisionPoint, ...]

    @property
    def lineages(self) -> tuple[str, ...]:
        """Return the task lineages covered, sorted."""
        return tuple(sorted({row.lineage for row in self.trajectories}))

    def states(self) -> dict[Sha256, ArtifactInput]:
        """Return each distinct reached state with the first saved request that reached it."""
        found: dict[Sha256, ArtifactInput] = {}
        for point in self.points:
            found.setdefault(point.state_sha256, point.call)
        return found

    def labels(
        self, *, by: Literal["decision", "action"]
    ) -> dict[tuple[str, Sha256, str], StateLabel]:
        """Label each checkpoint's observed state and main decision (or action) by its outcomes.

        Keys are ``(checkpoint artifact, state, decision)``: outcomes from another world never
        label a decision, even at a byte-identical visible state.
        """
        outcomes: dict[tuple[str, Sha256, str], list[Verdict]] = defaultdict(list)
        for point in self.points:
            key = point.main_choice if by == "decision" else point.main_action_sha256
            outcomes[point.checkpoint.artifact_id, point.state_sha256, key].append(point.outcome)
        return {
            key: "verified_correct"
            if "pass" in values
            else "verified_error"
            if "fail" in values
            else "unverified"
            for key, values in outcomes.items()
        }


def study_sources(store: ArtifactStore, sources: tuple[ArtifactInput, ...]) -> list[ArtifactInput]:
    """Expand actor-panel reports into their saved studies and keep study results as given.

    Raises:
        ValueError: A source is neither a continuation-study result nor an actor-panel report.
    """
    studies: list[ArtifactInput] = []
    for source in sources:
        manifest = store.read(source.artifact_id).manifest
        if manifest.artifact_type == "continuation-study":
            studies.append(source)
        elif manifest.artifact_type == "actor-panel-report":
            report = ActorPanelReport.model_validate_json(
                store.read_bytes(source.artifact_id, "report.json")
            )
            studies.extend(cell.result for cell in report.cells if cell.result is not None)
        else:
            raise ValueError(
                f"decision source {source.artifact_id} is not a continuation study or actor "
                "panel; pass saved study results or panel reports"
            )
    return studies


def extract_decision_points(
    store: ArtifactStore, sources: tuple[ArtifactInput, ...], *, main_model: ModelSnapshot
) -> DecisionPointSet:
    """Read every main-model decision from complete saved studies, mistakes included.

    Panel cells of other models and cascade studies are not the main model's own trajectories
    and are skipped; a directly named study of another model or with a cascade is refused.

    Raises:
        ValueError: A source is unsupported, belongs to another model, ran a cascade, or a saved
            call lost its response binding.
    """
    named = set(sources)
    studies: list[ArtifactInput] = []
    trajectories: list[Trajectory] = []
    points: list[DecisionPoint] = []
    for reference in study_sources(store, sources):
        study = load_study(store, reference)
        if study.spec.actor_model != main_model or study.spec.cascade is not None:
            if reference in named:
                raise ValueError(
                    "decision source is not a main-model-only study; fit on the main model's "
                    "own trajectories"
                )
            continue
        if reference in studies:
            continue
        studies.append(reference)
        task = load_continuation_checkpoint(store, study.spec.checkpoint).task
        for branch in study.branches:
            grade = branch.assessment
            outcome: Verdict = "unassessable" if grade is None else grade.status
            policy: Verdict = "unassessable" if grade is None else grade.policy.status
            trajectories.append(
                Trajectory(
                    study=reference,
                    branch=branch.index,
                    lineage=task.lineage_group_id,
                    outcome=outcome,
                    policy=policy,
                )
            )
            calls = list(branch.actor_calls)
            for index, call in enumerate(calls):
                if store.read(call.artifact_id).manifest.artifact_type != (
                    "continuation-actor-request"
                ):
                    continue
                answer = calls[index + 1] if index + 1 < len(calls) else None
                if answer is None or store.read(answer.artifact_id).manifest.artifact_type != (
                    "continuation-actor-response"
                ):
                    continue
                request = ModelRequest.model_validate_json(
                    store.read_bytes(call.artifact_id, "request.json")
                )
                response = ModelResponse.model_validate_json(
                    store.read_bytes(answer.artifact_id, "response.json")
                )
                if request.tool_choice not in (None, "auto") or response.model != main_model:
                    continue
                state = state_request(request)
                points.append(
                    DecisionPoint(
                        id=stable_id(
                            "decision-point",
                            {
                                "study": reference.artifact_id,
                                "branch": branch.index,
                                "call": call.artifact_id,
                            },
                        ),
                        study=reference,
                        branch=branch.index,
                        call=call,
                        checkpoint=study.spec.checkpoint,
                        lineage=task.lineage_group_id,
                        task_id=task.task_id,
                        state_sha256=sha256_json(state),
                        candidates=decision_candidates(state),
                        main_choice=action_key(response.output),
                        main_action_sha256=action_projection(response.output),
                        outcome=outcome,
                        policy=policy,
                    )
                )
    return DecisionPointSet(
        main_model=main_model,
        studies=tuple(studies),
        trajectories=tuple(trajectories),
        points=tuple(points),
    )
