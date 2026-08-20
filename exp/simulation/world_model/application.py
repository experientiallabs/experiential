"""Public project-name loader and bounded sessions for grounded world models."""

from __future__ import annotations

import json
import math
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock, RLock
from typing import cast

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionUserMessageParam,
)

from exp.common.core.artifacts import ArtifactInput, JsonObject, canonical_json_bytes, stable_id
from exp.common.models import AssistantAction, ModelMessage, load_model_catalog
from exp.common.project import ProjectStore, ProjectStoreError
from exp.common.tasks import TaskCase
from exp.runtime.models import CapabilityRequirement, RuntimeModelCatalog
from exp.runtime.models.providers.transport import RetryPolicy
from exp.simulation.retrieval import RAGEmbedderBinding
from exp.simulation.world_model.runtime import (
    GroundedWorldModel,
    load_grounded_world_model,
)


class WorldModelLoadError(ValueError):
    """A named project cannot load one exact executable serving world model."""


class WorldModelSessionError(ValueError):
    """A public world-model session request violates its bounded text contract."""


@dataclass(frozen=True)
class WorldModelSessionLimits:
    """Explicit resource limits for one in-memory public world-model runtime."""

    maximum_sessions: int = 256
    session_ttl_seconds: float = 3_600.0
    maximum_messages_per_session: int = 128
    maximum_transcript_bytes: int = 1_048_576
    maximum_observation_bytes: int = 65_536
    maximum_task_bytes: int = 65_536
    maximum_initial_context_bytes: int = 262_144

    def __post_init__(self) -> None:
        """Validate every count, time, and byte ceiling.

        Raises:
            ValueError: A limit is nonpositive or the TTL is not finite.
        """
        integer_limits = (
            self.maximum_sessions,
            self.maximum_messages_per_session,
            self.maximum_transcript_bytes,
            self.maximum_observation_bytes,
            self.maximum_task_bytes,
            self.maximum_initial_context_bytes,
        )
        if any(value <= 0 for value in integer_limits):
            raise ValueError("world-model session count and byte limits must be positive")
        if not math.isfinite(self.session_ttl_seconds) or self.session_ttl_seconds <= 0:
            raise ValueError("world-model session TTL must be finite and positive")
        if self.maximum_observation_bytes > self.maximum_transcript_bytes:
            raise ValueError("world-model observation bytes cannot exceed transcript bytes")


@dataclass(frozen=True)
class WorldModelSession:
    """One opaque public session identity and its request-visible starting state."""

    id: str
    task: str
    _initial_context_json: bytes = field(repr=False)

    @property
    def initial_context(self) -> JsonObject:
        """Return a detached copy of the session's canonical initial context.

        Returns:
            JSON object whose mutation cannot change the active session.
        """
        return cast(JsonObject, json.loads(self._initial_context_json))


@dataclass(frozen=True)
class WorldModelObservation:
    """One OpenAI-shaped user message predicted by a grounded world model."""

    message: ChatCompletionUserMessageParam
    terminal: bool


@dataclass
class _SessionState:
    """Mutable transcript and lifecycle bookkeeping for one public session."""

    session: WorldModelSession
    task_case: TaskCase
    messages: tuple[ModelMessage, ...]
    transcript_bytes: int
    expires_at: float
    lock: Lock = field(default_factory=Lock)
    closed: bool = False


def _new_session_id() -> str:
    """Return one opaque random session identity."""
    return uuid.uuid4().hex


class WorldModel:
    """Expose one verified serving-RAG world model through bounded text sessions."""

    def __init__(
        self,
        runtime: GroundedWorldModel,
        *,
        limits: WorldModelSessionLimits | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        session_id_factory: Callable[[], str] = _new_session_id,
    ) -> None:
        """Bind session state to one already verified grounded runtime.

        Args:
            runtime: Exact executable world model over the project's serving RAG.
            limits: Optional explicit resource ceilings. Stable safe defaults are used when
                omitted.
            monotonic: Monotonic clock injectable for deterministic expiry tests.
            session_id_factory: Opaque session identity supplier.
        """
        self._runtime = runtime
        self.limits = limits or WorldModelSessionLimits()
        self._monotonic = monotonic
        self._session_id_factory = session_id_factory
        self._sessions: OrderedDict[str, _SessionState] = OrderedDict()
        self._registry_lock = RLock()

    def new_session(
        self,
        *,
        task: str,
        initial_context: JsonObject | None = None,
    ) -> WorldModelSession:
        """Create one bounded in-memory session without model or embedder dispatch.

        Args:
            task: Nonempty request-visible scenario instruction.
            initial_context: Optional safe JSON context rendered into every grounded step.

        Returns:
            Opaque session handle accepted by ``step`` and ``end_session``.

        Raises:
            WorldModelSessionError: Input or active-session capacity exceeds a fixed limit.
        """
        if not task.strip():
            raise WorldModelSessionError("world-model session task must not be empty")
        if len(task.encode("utf-8")) > self.limits.maximum_task_bytes:
            raise WorldModelSessionError("world-model session task exceeds the byte limit")
        context: JsonObject = {} if initial_context is None else dict(initial_context)
        try:
            context_json = canonical_json_bytes(context)
        except (TypeError, ValueError) as exc:
            raise WorldModelSessionError("world-model initial context must be JSON") from exc
        if len(context_json) > self.limits.maximum_initial_context_bytes:
            raise WorldModelSessionError("world-model initial context exceeds the byte limit")
        now = self._monotonic()
        with self._registry_lock:
            self._expire_sessions(now)
            if len(self._sessions) >= self.limits.maximum_sessions:
                raise WorldModelSessionError(
                    "world-model session capacity reached; end a session or wait for expiry"
                )
            session_id = self._session_id_factory()
            if not session_id or session_id in self._sessions:
                raise WorldModelSessionError("world-model session identity is empty or repeated")
            session = WorldModelSession(
                id=session_id,
                task=task,
                _initial_context_json=context_json,
            )
            self._sessions[session_id] = _SessionState(
                session=session,
                task_case=_session_task(session),
                messages=(),
                transcript_bytes=0,
                expires_at=now + self.limits.session_ttl_seconds,
            )
            return session

    def step(
        self,
        session_id: str,
        action: ChatCompletionAssistantMessageParam,
    ) -> WorldModelObservation:
        """Predict and append one user turn from an official OpenAI assistant message.

        The persisted artifact is text-only. Any tool call, function call, audio payload,
        non-text content, or non-assistant role is rejected before retrieval or provider dispatch.

        Args:
            session_id: Opaque identity returned by ``new_session``.
            action: Official OpenAI assistant-message input containing visible text only.

        Returns:
            Official OpenAI user-message shape and the world-model terminal flag.

        Raises:
            WorldModelSessionError: The session is absent, expired, closed, unsupported, or would
                exceed a transcript limit.
        """
        action_text = _assistant_text(action)
        action_message = ModelMessage(role="assistant", content=action_text)
        now = self._monotonic()
        with self._registry_lock:
            self._expire_sessions(now)
            state = self._sessions.get(session_id)
            if state is None:
                raise WorldModelSessionError("world-model session is unknown or expired")
            state.lock.acquire()
            if state.closed:
                state.lock.release()
                raise WorldModelSessionError("world-model session is closed")
            self._sessions.move_to_end(session_id)
        try:
            self._require_step_capacity(state, action_message)
            prepared = self._runtime.prepare_turn(
                task=state.task_case,
                visible_messages=state.messages,
                candidate_response=AssistantAction(content=action_text),
                excluded_lineage_ids=(),
                maximum_output_tokens=1_024,
            )
            dispatched = self._runtime.complete_turn(prepared)
            if dispatched.response.model != self._runtime.artifact.model:
                raise WorldModelSessionError(
                    "world-model response identity differs from its build artifact"
                )
            transition = self._runtime.parse_turn(dispatched).transition
            user_message = ModelMessage(role="user", content=transition.message)
            user_bytes = _message_bytes(user_message)
            if user_bytes > self.limits.maximum_observation_bytes:
                raise WorldModelSessionError(
                    "world-model observation exceeds the provider-output byte limit"
                )
            final_bytes = state.transcript_bytes + _message_bytes(action_message) + user_bytes
            if final_bytes > self.limits.maximum_transcript_bytes:
                raise WorldModelSessionError("world-model transcript exceeds the byte limit")
            state.messages = (*state.messages, action_message, user_message)
            state.transcript_bytes = final_bytes
            state.expires_at = self._monotonic() + self.limits.session_ttl_seconds
            state.closed = transition.terminal
            return WorldModelObservation(
                message=ChatCompletionUserMessageParam(
                    role="user",
                    content=transition.message,
                ),
                terminal=transition.terminal,
            )
        finally:
            state.lock.release()

    def end_session(self, session_id: str) -> None:
        """Remove one session and its complete transcript immediately.

        Args:
            session_id: Opaque identity returned by ``new_session``.

        Raises:
            WorldModelSessionError: The session is absent or already expired.
        """
        with self._registry_lock:
            self._expire_sessions(self._monotonic())
            state = self._sessions.get(session_id)
            if state is None:
                raise WorldModelSessionError("world-model session is unknown or expired")
            with state.lock:
                state.closed = True
                self._sessions.pop(session_id)

    def _require_step_capacity(self, state: _SessionState, action: ModelMessage) -> None:
        """Reject a known transcript overflow before retrieval or provider dispatch.

        Args:
            state: Locked active session state.
            action: Validated text-only assistant message.

        Raises:
            WorldModelSessionError: The next assistant and reserved observation exceed a limit.
        """
        if len(state.messages) + 2 > self.limits.maximum_messages_per_session:
            raise WorldModelSessionError("world-model transcript exceeds the message-count limit")
        reserved_bytes = (
            state.transcript_bytes + _message_bytes(action) + self.limits.maximum_observation_bytes
        )
        if reserved_bytes > self.limits.maximum_transcript_bytes:
            raise WorldModelSessionError("world-model transcript exceeds the byte limit")

    def _expire_sessions(self, now: float) -> None:
        """Remove idle sessions without interrupting a step already in progress.

        Args:
            now: Current monotonic time under the registry lock.
        """
        for session_id, state in tuple(self._sessions.items()):
            if state.expires_at > now or not state.lock.acquire(blocking=False):
                continue
            try:
                if state.expires_at <= now:
                    state.closed = True
                    self._sessions.pop(session_id, None)
            finally:
                state.lock.release()


def load_world_model(
    project: str,
    root: Path = Path(".exp"),
    *,
    environment: Mapping[str, str] | None = None,
    runtime_catalog: RuntimeModelCatalog | None = None,
    limits: WorldModelSessionLimits | None = None,
) -> WorldModel:
    """Load one named project's verified serving-RAG world model.

    Loading verifies immutable project pointers, model and embedder snapshots, the serving RAG,
    and the grounded prompt without making a provider or embedder call.

    Args:
        project: Canonical local project identifier.
        root: Local ``.exp`` root. Defaults to the happy-path project location.
        environment: Optional credential mapping used only to construct runtime clients.
        runtime_catalog: Optional explicit runtime catalog for deterministic applications.
        limits: Optional explicit in-memory session ceilings.

    Returns:
        Public bounded world model over the project's exact serving RAG.

    Raises:
        WorldModelLoadError: Project, build, catalog, model, embedder, or artifact identity is
            absent, malformed, or inconsistent.
    """
    try:
        store = ProjectStore(root, project)
        config = store.load_project()
        if config.models is None:
            raise WorldModelLoadError("project has no model roles; run exp build first")
        if config.build is None:
            raise WorldModelLoadError("project has no completed world model; run exp build first")
        catalog = runtime_catalog
        if catalog is None:
            catalog = RuntimeModelCatalog(
                load_model_catalog(store.model_catalog_path),
                environment=environment,
            )
        resolved_world = catalog.resolve(config.models.world_model, role="world_model")
        resolved_embedder = catalog.preflight(
            config.models.embedder,
            CapabilityRequirement(requires_embeddings=True),
        )
        if resolved_embedder.embedding_client is None:
            raise WorldModelLoadError("project embedder alias has no embedding client")
        embedding_price = resolved_embedder.capabilities.input_cost_per_million_tokens_usd
        runtime = load_grounded_world_model(
            store.artifacts,
            config.build.world_model.artifact_id,
            client=resolved_world.client,
            embedder=RAGEmbedderBinding(
                client=resolved_embedder.embedding_client,
                snapshot=resolved_embedder.snapshot,
                maximum_attempts=RetryPolicy().maximum_attempts,
                input_usd_per_million_tokens=(0.0 if embedding_price is None else embedding_price),
            ),
        )
        _require_project_binding(runtime, config.build.world_model, config.build.serving_rag)
        if runtime.artifact.model_alias != config.models.world_model:
            raise WorldModelLoadError("project world-model alias differs from its build artifact")
        if runtime.artifact.model != resolved_world.snapshot:
            raise WorldModelLoadError("project world-model snapshot differs from its runtime")
        if runtime.retriever.rag_input != config.build.serving_rag:
            raise WorldModelLoadError("project world model is not bound to its exact serving RAG")
        return WorldModel(runtime, limits=limits)
    except WorldModelLoadError:
        raise
    except (OSError, ProjectStoreError, ValueError) as exc:
        raise WorldModelLoadError(str(exc)) from exc


def _assistant_text(action: ChatCompletionAssistantMessageParam) -> str:
    """Return one supported assistant text action before any runtime work.

    Args:
        action: Official OpenAI assistant-message parameter.

    Returns:
        Nonempty visible assistant text.

    Raises:
        WorldModelSessionError: The action is not one plain text-only assistant message.
    """
    if action.get("role") != "assistant":
        raise WorldModelSessionError("world-model actions must use role='assistant'")
    if action.get("tool_calls") is not None or action.get("function_call") is not None:
        raise WorldModelSessionError(
            "this text-only world model does not accept assistant tool_calls"
        )
    if action.get("audio") is not None:
        raise WorldModelSessionError("this text-only world model does not accept assistant audio")
    content = action.get("content")
    if not isinstance(content, str) or not content.strip():
        raise WorldModelSessionError(
            "this text-only world model requires nonempty assistant text content"
        )
    return content


def _message_bytes(message: ModelMessage) -> int:
    """Return visible UTF-8 content bytes charged to one text transcript message."""
    assert message.content is not None
    return len(message.content.encode("utf-8"))


def _session_task(session: WorldModelSession) -> TaskCase:
    """Build the canonical prompt task for one public session.

    Args:
        session: Public session with validated task and initial context.

    Returns:
        Stable synthetic task case accepted by the artifact's pinned prompt builder.
    """
    task_identity = {
        "task": session.task,
        "initial_context": session.initial_context,
    }
    task_id = stable_id("public-world-model-task", task_identity)
    return TaskCase(
        task_id=task_id,
        lineage_group_id=stable_id("public-world-model-lineage", task_identity),
        partition="held_out",
        instruction=session.task,
        initial_context=session.initial_context,
        workload_weight=1.0,
        source_trace_ids=(task_id,),
    )


def _require_project_binding(
    runtime: GroundedWorldModel,
    world_model_input: ArtifactInput,
    serving_rag_input: ArtifactInput,
) -> None:
    """Require exact project pointers without accepting a fit-RAG substitution.

    Args:
        runtime: Fully verified grounded runtime.
        world_model_input: Project's exact grounded-world-model manifest input.
        serving_rag_input: Project's exact serving-RAG manifest input.

    Raises:
        WorldModelLoadError: Runtime inputs differ from either project pointer.
    """
    if runtime.artifact_input != world_model_input:
        raise WorldModelLoadError("project world-model manifest differs from its build pointer")
    if runtime.artifact.serving_rag != serving_rag_input:
        raise WorldModelLoadError("project world model does not name its serving RAG")
