"""Session-local investigations from explicitly disclosed post-prediction feedback."""

import copy
import json
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Protocol, Self

from filelock import FileLock
from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    JsonValue,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.evaluations.reconstruction import strict_equal
from exp.common.models import ToolCall
from exp.common.project import ArtifactStore
from exp.runtime.sessions.store import SessionStore
from exp.simulation.engines.text.prompt import MEMORY_CONTEXT_GUIDANCE, MemoryContext
from exp.simulation.engines.transition import (
    Completion,
    PreparedPrediction,
    ProposedTransition,
    TransitionEngine,
)
from exp.simulation.retrieval.selection import EvidencePolicy, SelectionQuery, selected_evidence

_INVESTIGATE = """Investigate one completed simulator prediction using only this input.
The disclosed observation is feedback about this past action, not a future answer or full
native state. Compare it with the prediction and candidate state. Describe the discrepancy
and propose a scoped, tentative hypothesis for later turns. Distinguish a wrong behavioral
rule from an incorrect current-state belief and from insufficient information. Do not invent
hidden values. Do not rewrite the world, change past scores, or promote a global rule.
You may supersede IDs from previous_investigations or prior_trajectory_investigations only
when the supplied feedback contradicts them; superseded notes stay in the audit archive.
Preserve uncertainty when several explanations fit.
Fictional example: predicted {"status":"pending"}, observed {"status":"shipped"} for a
read of entity E. A supported note says the candidate's status belief about E was wrong at
that read; it does not establish that all entities are shipped or explain why E changed.
Set transferable true only for a rule about how this environment behaves that a later
prediction could apply from what it can see: the tool, its arguments, and visible state or
history. Then name that visible condition in observable_cue and phrase the hypothesis in those
terms, never in this instance's values. Set transferable false, with observable_cue "", when
the miss came from information the simulator could not see (unshown records, hidden state)
or only describes this instance; say why in uncertainty. Abstain rather than guess.
Return the required structured investigation, with supersedes:[] if nothing is disproved.
"""
INVESTIGATION_PROMPT = _INVESTIGATE


class OnlineResearchPolicy(ContractModel):
    """Prospective limits and authority for supervised adaptation within one session.

    Attributes:
        feedback_source: Declared origin of disclosed past native replies.
        maximum_updates: Maximum model-generated investigations, not prediction retries.
        retained_notes: Maximum active investigations supplied to the investigator.
        selection_policy_sha256: Exact query-dependent memory selection policy.
        generate_hypotheses: False retains raw discrepancy feedback without an investigator call.
        notes_reach_predictions: False keeps this session's feedback and investigations away
            from its own later predictions: every prediction reads only the memory pinned before
            the session, so a scored trajectory is predicted whole before any of its feedback is
            used. Investigations still run and are published with the trajectory.
    """

    selection_policy_sha256: Sha256
    feedback_source: str = Field(min_length=1)
    generate_hypotheses: bool = Field(default=True, strict=True, exclude_if=bool)
    notes_reach_predictions: bool = Field(default=True, strict=True, exclude_if=bool)
    maximum_updates: int = Field(default=24, gt=0, strict=True)
    retained_notes: int = Field(default=24, gt=0, strict=True)


class Investigation(ContractModel):
    """Provisional interpretation that cannot directly edit a session or global knowledge.

    Attributes:
        discrepancy: Specific predicted-versus-observed difference.
        hypothesis: Tentative explanation or revised state belief.
        scope: Entities, tools and conditions to which the hypothesis might apply.
        uncertainty: Remaining alternatives or missing observations.
        supersedes: Earlier active note IDs contradicted by the supplied feedback.
        transferable: Whether this is a behavioral rule a later prediction could apply from
            what it can observe. Only transferable notes become memory rules. Records written
            before this field read as not transferable.
        observable_cue: The visible condition (tool, arguments, state) that triggers the rule;
            required when transferable.
    """

    discrepancy: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    uncertainty: str = Field(min_length=1)
    supersedes: tuple[str, ...] = ()
    transferable: bool = Field(default=False, strict=True)
    observable_cue: str = ""

    @model_validator(mode="after")
    def cued(self) -> Self:
        """A transferable rule must name the observable condition that triggers it."""
        if self.transferable and not self.observable_cue.strip():
            raise ValueError("a transferable rule needs its observable cue")
        return self


def investigation_schema() -> JsonObject:
    """Wire schema: the model must decide transferability and give the cue explicitly."""
    schema = Investigation.model_json_schema()
    required = schema.get("required", [])
    assert isinstance(required, list)
    schema["required"] = [*required, "transferable", "observable_cue"]
    return schema


class SessionResearch:
    """Retain append-only feedback and revisions without mutating candidate state.

    The same session lock serializes prediction and feedback. Immutable intents prevent
    an interrupted investigation from being silently purchased again after restart.
    Prior records come from a frozen memory version and are never edited by this session.
    """

    def __init__(
        self,
        *,
        session_id: str,
        policy: OnlineResearchPolicy | None,
        store: SessionStore,
        artifacts: ArtifactStore,
        complete: Completion | None,
        lock_path: Path,
        clock: Callable[[], datetime],
        code_revision: str,
        selection: EvidencePolicy,
        prior: Mapping[str, JsonObject] | None = None,
        prior_identity: JsonObject | None = None,
        overview: Sequence[JsonObject] = (),
        prior_sources: Mapping[str, Sequence[ArtifactInput]] | None = None,
        overview_sources: Sequence[ArtifactInput] = (),
        overview_view: Callable[[frozenset[str]], list[JsonObject]] | None = None,
        overview_sources_of: Callable[[Sequence[JsonObject]], Sequence[ArtifactInput]]
        | None = None,
        retrieved_sources_of: Callable[[Sequence[str]], Sequence[ArtifactInput]] | None = None,
    ) -> None:
        """Bind feedback policy, metered investigator and an optional frozen prior memory.

        Args:
            session_id: Current trajectory's world identity.
            policy: Feedback learning policy; None only delivers the frozen prior memory.
            store: Shared durable session owner.
            artifacts: Immutable receipt owner.
            complete: Metered investigator, required exactly when a policy is present.
            lock_path: Session single-writer lease.
            clock: Aware clock.
            code_revision: Implementation identity.
            selection: Query-dependent selector over prior and session records.
            prior: Active research from earlier trajectories, keyed by record identity.
            prior_identity: Memory version bound into this session's identity.
            overview: Budgeted memory overview at session start.
            prior_sources: Source trajectories behind each prior record, for provenance.
            overview_sources: Source trajectories behind the overview rows.
            overview_view: Re-renders the overview with this session's retired notes removed.
            overview_sources_of: Resolves the sources behind any rendered overview, so each
                delivered overview's sources are recorded with the prediction that saw it.
            retrieved_sources_of: Resolves retrieved transition identities to the recorded
                traces they came from, so research written after seeing retrieved evidence
                depends on those traces' lineages.
        """
        if (policy is None) != (complete is None):
            raise ValueError("feedback learning requires both policy and investigator")
        if policy is not None and selection.identity != policy.selection_policy_sha256:
            raise ValueError("memory selector differs from registered online policy")
        if policy is None and not prior_identity:
            raise ValueError("research delivery without feedback requires a frozen memory")
        self._selection = selection
        self.session_id, self.policy = session_id, policy
        self._prior = copy.deepcopy(dict(prior or {}))
        self._prior_identity = copy.deepcopy(prior_identity)
        self.overview = copy.deepcopy(list(overview))
        self._prior_sources = {key: tuple(refs) for key, refs in (prior_sources or {}).items()}
        self._overview_sources = tuple(overview_sources)
        self._overview_view = overview_view
        self._overview_sources_of = overview_sources_of
        self._retrieved_sources_of = retrieved_sources_of
        self._store, self._artifacts, self._complete = store, artifacts, complete
        self._lock = FileLock(lock_path, timeout=0)
        self._clock, self._code_revision = clock, code_revision

    @property
    def identity(self) -> Sha256:
        """Bind adaptation policy, prior memory and actual investigation and serving text."""
        return sha256_json(
            {
                "session": self.session_id,
                "policy": None if self.policy is None else self.policy.model_dump(mode="json"),
                "selection": self._selection.identity,
                "prior": self._prior_identity,
                "overview": sha256_json(self.overview),
                "investigate": _INVESTIGATE,
                "use": MEMORY_CONTEXT_GUIDANCE,
                "schema": investigation_schema(),
            }
        )

    def _record_id(self, operation_id: str) -> str:
        """Bind each observation to its session, immutable execution binding and operation."""
        binding = self._store.read_session(self.session_id).binding
        return stable_id(
            "session-research",
            {
                "session": self.session_id,
                "binding": binding.model_dump(mode="json"),
                "operation": operation_id,
                "policy": self.identity,
            },
        )

    def _records(self, transitions: list[JsonValue]) -> list[JsonObject]:
        """Load only feedback attached to already committed operations in this session."""
        records: list[JsonObject] = []
        if self.policy is None:
            return records
        for transition in transitions:
            if not isinstance(transition, dict) or not isinstance(
                transition.get("operation_id"), str
            ):
                raise ValueError("research requires committed operation identities")
            identity = self._record_id(str(transition["operation_id"]))
            if self._artifacts.exists(identity):
                records.append(json.loads(self._artifacts.read_bytes(identity, "record.json")))
            elif self._artifacts.exists(identity + "-intent"):
                raise ValueError("unfinished research update requires explicit recovery")
        return records

    def _prior_research(self, records: list[JsonObject]) -> list[JsonObject]:
        """Prior-trajectory hypotheses not yet retired by this session's investigations."""
        retired = _superseded(records)
        return [
            row
            for key, row in self._prior.items()
            if row.get("kind") != "original-observation" and key not in retired
        ]

    def _active(self, records: list[JsonObject]) -> list[JsonObject]:
        """Render the current view while retaining superseded notes in immutable records."""
        active: dict[str, JsonObject] = {}
        for record in records:
            raw = record.get("investigation")
            if not isinstance(raw, dict):
                if (
                    self.policy is not None
                    and not self.policy.generate_hypotheses
                    and record.get("reply_matched") is False
                ):
                    active[str(record["id"])] = record
                continue
            update = Investigation.model_validate(raw)
            for identity in update.supersedes:
                active.pop(identity, None)
            active[str(record["id"])] = record
        return list(active.values())

    def observe(self, operation_id: str, observed_reply: JsonValue, source: str) -> JsonObject:
        """Learn from the latest completed single-tool reply, never from future actions.

        This API accepts explicitly disclosed reply feedback, not reference states or
        private implementations. The caller owns the source's authenticity. Replaying
        identical feedback returns the saved record without another model call.
        """
        policy, complete = self.policy, self._complete
        if policy is None or complete is None:
            raise ValueError("frozen memory sessions do not accept feedback")
        if source != policy.feedback_source:
            raise ValueError("feedback source differs from the registered disclosure policy")
        with self._lock:
            current = self._store.read_session(self.session_id)
            identity = self._record_id(operation_id)
            if self._artifacts.exists(identity):
                saved = json.loads(self._artifacts.read_bytes(identity, "record.json"))
                if not strict_equal(saved["observed_reply"], observed_reply):
                    raise ValueError("feedback replay changed the disclosed observation")
                return saved
            if self._artifacts.exists(identity + "-intent"):
                raise ValueError("unfinished research update requires explicit recovery")
            transitions = current.state.get("transitions")
            if not isinstance(transitions, list) or not transitions:
                raise ValueError("feedback requires a completed prediction")
            latest = transitions[-1]
            if not isinstance(latest, dict) or latest.get("operation_id") != operation_id:
                raise ValueError("new feedback must concern the latest completed prediction")
            replies = latest.get("replies")
            if not isinstance(replies, list) or len(replies) != 1:
                raise ValueError("reply feedback requires a single-tool prediction")
            if self._clock() >= current.binding.deadline:
                raise TimeoutError("session deadline reached before investigation")
            records = self._records(transitions)
            matched = strict_equal(replies[0], observed_reply)
            record: JsonObject = {
                "id": identity,
                "session_id": self.session_id,
                "operation_id": operation_id,
                "actions": latest["actions"],
                "committed_version": current.version,
                "source": source,
                "transition_sha256": sha256_json(latest),
                "predicted_reply": replies[0],
                "observed_reply": observed_reply,
                "reply_matched": matched,
                "authority": "disclosed-past-reply; hypotheses-unverified",
                "investigation": None,
            }
            local = {str(row["id"]): row for row in records}
            supplied = self._supplied(transitions, local)
            exhausted = sum(isinstance(row.get("investigation"), dict) for row in records) >= (
                policy.maximum_updates
            )
            if not matched and policy.generate_hypotheses and exhausted:
                # The bound limits purchases, never consumption: the miss is still recorded
                # as raw feedback, so the trajectory completes and is learned.
                record["investigation_skipped"] = "session investigation budget exhausted"
            elif not matched and policy.generate_hypotheses:
                active = self._active(records)[-policy.retained_notes :]
                prior = self._prior_research(records)[-policy.retained_notes :]
                supplied |= {str(row["id"]) for row in (*active, *prior)}
                body: JsonObject = {
                    "messages": [
                        {"role": "system", "content": _INVESTIGATE},
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "feedback": record,
                                    "candidate_transition": latest,
                                    "previous_investigations": active,
                                    "prior_trajectory_investigations": prior,
                                },
                                ensure_ascii=False,
                                allow_nan=False,
                            ),
                        },
                    ],
                    "response_schema": investigation_schema(),
                }
                self._save(identity + "-intent", {"request": body})
                proposal = complete(body, identity + "-investigate")
                self._save(identity + "-proposal", {"proposal": proposal})
                update = Investigation.model_validate(proposal)
                available = {str(row["id"]) for row in (*active, *prior)}
                if len(set(update.supersedes)) != len(update.supersedes) or not set(
                    update.supersedes
                ).issubset(available):
                    raise ValueError("investigation can supersede only supplied active notes")
                if self._clock() >= current.binding.deadline:
                    raise TimeoutError("late investigation retained but not activated")
                record["investigation"] = update.model_dump(mode="json")
            record["depends_on"] = self._dependencies(
                supplied,
                local,
                [*_recorded_overview_sources(transitions), *self._retrieved(transitions)],
            )
            self._save(identity, record)
            return copy.deepcopy(record)

    def observe_all(
        self,
        feedback: Sequence[tuple[str, JsonValue]],
        source: str,
        *,
        workers: int = 8,
    ) -> list[JsonObject | Exception]:
        """Disclose every reply after the whole trajectory is predicted; investigate concurrently.

        Only for pinned delivery, where no investigation can inform a prediction. Each
        investigation sees the disclosed reply, its own predicted transition and the pinned
        prior research, never another investigation of this session, so the outcome does not
        depend on completion order. At most ``maximum_updates`` misses across all batches
        are investigated, in supplied step order; later misses are recorded as raw feedback.
        Failed investigation intents consume budget too. One failed investigation affects
        only its own receipt. The session lease is held until every worker settles.

        Args:
            feedback: ``(operation_id, disclosed reply)`` for each predicted step, in order.
            source: Declared disclosure source.
            workers: Concurrent investigations.

        Returns:
            One saved record, or the exception that prevented it, per feedback item.
        """
        policy, complete = self.policy, self._complete
        if policy is None or complete is None:
            raise ValueError("frozen memory sessions do not accept feedback")
        if policy.notes_reach_predictions:
            raise ValueError("deferred feedback requires predictions on pinned memory")
        if source != policy.feedback_source:
            raise ValueError("feedback source differs from the registered disclosure policy")
        operations = [operation_id for operation_id, _ in feedback]
        if len(set(operations)) != len(operations):
            raise ValueError("feedback repeats an operation; supply each prediction only once")
        with self._lock:
            return self._observe_all_locked(feedback, source, policy, complete, workers)

    def _observe_all_locked(
        self,
        feedback: Sequence[tuple[str, JsonValue]],
        source: str,
        policy: OnlineResearchPolicy,
        complete: Completion,
        workers: int,
    ) -> list[JsonObject | Exception]:
        """Plan and settle a batch while the caller owns the session lease."""
        current = self._store.read_session(self.session_id)
        transitions = current.state.get("transitions")
        if not isinstance(transitions, list):
            raise ValueError("feedback requires completed predictions")
        by_operation = {
            str(row["operation_id"]): row
            for row in transitions
            if isinstance(row, dict) and isinstance(row.get("operation_id"), str)
        }
        prior = self._prior_research([])[-policy.retained_notes :]
        supplied = self._supplied(transitions, {}) | {str(row["id"]) for row in prior}
        depends_on = self._dependencies(
            supplied, {}, [*_recorded_overview_sources(transitions), *self._retrieved(transitions)]
        )
        identities = {key: self._record_id(key) for key in by_operation}
        claimed = {
            identity
            for identity in identities.values()
            if self._artifacts.exists(identity + "-intent")
        }
        budget = max(0, policy.maximum_updates - len(claimed))
        plans: list[tuple[str, JsonObject, bool] | Exception] = []
        for operation_id, observed_reply in feedback:
            identity = identities.get(operation_id)
            latest = by_operation.get(operation_id)
            replies = latest.get("replies") if latest is not None else None
            if (
                identity is None
                or latest is None
                or not isinstance(replies, list)
                or len(replies) != 1
            ):
                plans.append(ValueError("feedback requires one single-tool prediction"))
                continue
            matched = strict_equal(replies[0], observed_reply)
            investigate = (
                not matched
                and policy.generate_hypotheses
                and budget > 0
                and identity not in claimed
                and not self._artifacts.exists(identity)
            )
            if investigate:
                budget -= 1
            record: JsonObject = {
                "id": identity,
                "session_id": self.session_id,
                "operation_id": operation_id,
                "actions": latest["actions"],
                "committed_version": current.version,
                "source": source,
                "transition_sha256": sha256_json(latest),
                "predicted_reply": replies[0],
                "observed_reply": observed_reply,
                "reply_matched": matched,
                "authority": "disclosed-past-reply; hypotheses-unverified",
                "investigation": None,
                "depends_on": depends_on,
            }
            if not matched and policy.generate_hypotheses and not investigate:
                record["investigation_skipped"] = "session investigation budget exhausted"
            plans.append((identity, record, investigate))

        def settle(plan: tuple[str, JsonObject, bool]) -> JsonObject:
            """Replay a saved receipt, or investigate and save a new one."""
            identity, record, investigate = plan
            if self._artifacts.exists(identity):
                saved = json.loads(self._artifacts.read_bytes(identity, "record.json"))
                if not strict_equal(saved["observed_reply"], record["observed_reply"]):
                    raise ValueError("feedback replay changed the disclosed observation")
                return saved
            if self._artifacts.exists(identity + "-intent"):
                raise ValueError("unfinished research update requires explicit recovery")
            if self._clock() >= current.binding.deadline:
                raise TimeoutError("session deadline reached before investigation")
            if investigate:
                body: JsonObject = {
                    "messages": [
                        {"role": "system", "content": _INVESTIGATE},
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "feedback": {
                                        k: v for k, v in record.items() if k != "depends_on"
                                    },
                                    "candidate_transition": by_operation[
                                        str(record["operation_id"])
                                    ],
                                    "previous_investigations": [],
                                    "prior_trajectory_investigations": prior,
                                },
                                ensure_ascii=False,
                                allow_nan=False,
                            ),
                        },
                    ],
                    "response_schema": investigation_schema(),
                }
                self._save(identity + "-intent", {"request": body})
                proposal = complete(body, identity + "-investigate")
                self._save(identity + "-proposal", {"proposal": proposal})
                update = Investigation.model_validate(proposal)
                available = {str(row["id"]) for row in prior}
                if len(set(update.supersedes)) != len(update.supersedes) or not set(
                    update.supersedes
                ).issubset(available):
                    raise ValueError("investigation can supersede only supplied prior notes")
                if self._clock() >= current.binding.deadline:
                    raise TimeoutError("late investigation retained but not activated")
                record = {**record, "investigation": update.model_dump(mode="json")}
            self._save(identity, record)
            return copy.deepcopy(record)

        results: list[JsonObject | Exception | None] = [
            plan if isinstance(plan, Exception) else None for plan in plans
        ]
        pending = [(i, plan) for i, plan in enumerate(plans) if not isinstance(plan, Exception)]
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(pending) or 1))) as pool:
            futures = {pool.submit(settle, plan): i for i, plan in pending}
            for future, index in futures.items():
                try:
                    results[index] = future.result()
                except Exception as error:  # noqa: BLE001 - one failure stays with its receipt.
                    results[index] = error
        return [row for row in results if row is not None]

    def _retrieved(self, transitions: list[JsonValue]) -> Sequence[ArtifactInput]:
        """Recorded traces behind every retrieved transition this session's predictions saw."""
        ids: set[str] = set()
        for transition in transitions:
            evidence = transition.get("evidence") if isinstance(transition, dict) else None
            rows = evidence.get("grounded_transition_ids") if isinstance(evidence, dict) else None
            ids.update(str(row) for row in rows or ())
        if not ids:
            return ()
        if self._retrieved_sources_of is None:
            raise ValueError("retrieved evidence reached a prediction without its provenance")
        return self._retrieved_sources_of(sorted(ids))

    def _supplied(self, transitions: list[JsonValue], local: Mapping[str, JsonObject]) -> set[str]:
        """Prior and local record identities delivered to any committed prediction."""
        supplied: set[str] = set()
        for transition in transitions:
            evidence = transition.get("evidence") if isinstance(transition, dict) else None
            selection = (
                evidence.get("session_research_selection") if isinstance(evidence, dict) else None
            )
            ids = selection.get("selected_ids") if isinstance(selection, dict) else None
            supplied.update(
                str(key) for key in ids or () if str(key) in self._prior or str(key) in local
            )
        return supplied

    def _dependencies(
        self,
        supplied: set[str],
        local: Mapping[str, JsonObject],
        recorded: Sequence[ArtifactInput] = (),
    ) -> list[JsonValue]:
        """Every source behind memory this session was given, including earlier local notes.

        A local note's own dependencies are inherited, so a chain A, then this session's
        first note, then its second note keeps A as a dependency of both notes.
        """
        refs = set(self._overview_sources) | set(recorded)
        for key in supplied:
            refs.update(self._prior_sources.get(key, ()))
            inherited = local.get(key, {}).get("depends_on")
            if isinstance(inherited, list):
                refs.update(ArtifactInput.model_validate(row) for row in inherited)
        return [ref.model_dump(mode="json") for ref in sorted(refs, key=lambda r: r.artifact_id)]

    def delivered_overview(self) -> tuple[list[JsonObject], list[JsonValue]]:
        """The overview for the next prediction and the sources behind exactly those rows."""
        rows = self.serving_overview()
        refs = (
            self._overview_sources_of(rows)
            if self._overview_sources_of is not None
            else self._overview_sources
        )
        ordered = sorted(set(refs), key=lambda ref: ref.artifact_id)
        return rows, [ref.model_dump(mode="json") for ref in ordered]

    def serving_overview(self) -> list[JsonObject]:
        """The overview for the next prediction, minus notes this session has retired."""
        if self._overview_view is None or not self._notes_reach_predictions:
            return copy.deepcopy(self.overview)
        state = self._store.read_session(self.session_id).state
        transitions = state.get("transitions")
        retired = _superseded(self._records(transitions if isinstance(transitions, list) else []))
        return self._overview_view(frozenset(retired))

    def deliver(
        self, history: tuple[JsonObject, ...], action: ToolCall, candidate_state: JsonObject
    ) -> tuple[list[JsonObject], JsonObject]:
        """Select among prior-trajectory and active session notes for this exact chronology."""
        state = self._store.read_session(self.session_id).state
        if not strict_equal(state.get("history"), list(history)):
            raise ValueError("research delivery history differs from the committed session")
        transitions = state.get("transitions")
        if not isinstance(transitions, list):
            raise ValueError("research requires a committed transition inventory")
        if self.policy is not None and (
            self._selection.identity != self.policy.selection_policy_sha256
        ):
            raise ValueError("online memory selection policy changed")
        # Pinned delivery: this session's own feedback never reaches its predictions.
        records = self._records(transitions) if self._notes_reach_predictions else []
        active = self._active(records)
        retired = _superseded(records)
        catalog: dict[str, JsonValue] = {
            key: value for key, value in self._prior.items() if key not in retired
        }
        catalog.update({str(row["id"]): row for row in active})
        selected, receipt = selected_evidence(
            self._selection,
            SelectionQuery(action=action, state=candidate_state, history=history),
            catalog,
        )
        return [value for value in selected.values() if isinstance(value, dict)], receipt

    @property
    def _notes_reach_predictions(self) -> bool:
        """Whether this session's feedback may inform its own later predictions."""
        return self.policy is None or self.policy.notes_reach_predictions

    def _save(self, identity: str, payload: JsonObject) -> None:
        """Publish immutable receipts outside business-state storage."""
        self._artifacts.write_json(
            artifact_id=identity,
            artifact_type="session-research",
            envelope=ArtifactEnvelope(
                schema_version=1, created_at=self._clock(), code_revision=self._code_revision
            ),
            files={"record.json": payload},
        )


class MemoryAwareEngine(TransitionEngine, Protocol):
    """A transition engine whose renderer accepts typed cross-trajectory memory."""

    def prepare(
        self,
        state: JsonObject,
        history: tuple[JsonObject, ...],
        actions: tuple[ToolCall, ...],
        *,
        memory: MemoryContext | None = None,
    ) -> PreparedPrediction:
        """Render one prediction request including the supplied memory channels."""
        ...


class ResearchTransitionEngine:
    """Deliver session investigations through the existing grounded request and parser."""

    def __init__(self, base: MemoryAwareEngine, research: SessionResearch) -> None:
        """Keep prediction semantics and output parsing with the existing engine."""
        self._base, self._research = base, research

    @property
    def identity(self) -> Sha256:
        """Freeze the policy while its session-local feedback grows."""
        return sha256_json({"engine": self._base.identity, "research": self._research.identity})

    def prepare(
        self, state: JsonObject, history: tuple[JsonObject, ...], actions: tuple[ToolCall, ...]
    ) -> PreparedPrediction:
        """Select memory, split it by authority, and let the base engine render it once.

        Observations, this session's notes, prior hypotheses and the overview reach the
        renderer as a typed ``MemoryContext``; no prompt text is parsed or edited here.
        The request hash therefore identifies the exact body that is dispatched.
        """
        if len(actions) != 1:
            raise ValueError("online research supports one tool action per prediction")
        notes, selection = self._research.deliver(history, actions[0], state)
        observed = [n for n in notes if n.get("kind") == "original-observation"]
        research = [n for n in notes if n.get("kind") != "original-observation"]
        rows, overview_sources = self._research.delivered_overview()
        memory = MemoryContext(
            observations=tuple(observed),
            session=tuple(n for n in research if n.get("session_id") == self._research.session_id),
            prior_trajectories=tuple(
                n for n in research if n.get("session_id") != self._research.session_id
            ),
            overview=tuple(rows),
        )
        prepared = self._base.prepare(state, history, actions, memory=memory)
        return PreparedPrediction(
            body=prepared.body,
            evidence={
                **prepared.evidence,
                "session_research_sha256": sha256_json(notes),
                "session_research_ids": [row["id"] for row in notes],
                "feedback_policy_sha256": self._research.identity,
                "session_research_selection": selection,
                "memory_overview_ids": [str(row.get("id")) for row in rows],
                "memory_overview_sources": overview_sources,
            },
        )

    def parse(
        self,
        proposal: JsonObject,
        actions: tuple[ToolCall, ...],
        *,
        current_state: JsonObject | None = None,
    ) -> ProposedTransition:
        """Preserve the original proposal validation and state-effect semantics."""
        return self._base.parse(proposal, actions, current_state=current_state)


def _superseded(records: list[JsonObject]) -> set[str]:
    """Identities explicitly retired by this session's validated investigations."""
    retired: set[str] = set()
    for record in records:
        raw = record.get("investigation")
        if isinstance(raw, dict):
            retired.update(Investigation.model_validate(raw).supersedes)
    return retired


def _recorded_overview_sources(transitions: list[JsonValue]) -> list[ArtifactInput]:
    """Sources of every overview delivered to a committed prediction in this session."""
    refs: list[ArtifactInput] = []
    for transition in transitions:
        evidence = transition.get("evidence") if isinstance(transition, dict) else None
        rows = evidence.get("memory_overview_sources") if isinstance(evidence, dict) else None
        if isinstance(rows, list):
            refs.extend(ArtifactInput.model_validate(row) for row in rows)
    return refs
