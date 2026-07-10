"""The WorldModel: a frontier LLM acting as the environment.

This is the public API agents call (in-process or via the local backend). Each `step` retrieves
similar past steps, builds the env prompt, completes it with the serving provider, and updates the
session — including the env's free-text scratchpad "database" — to stay consistent across it.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from wmh.config import ArtifactPaths, load_config
from wmh.core.parsing import parse_observation
from wmh.core.types import Action, EnvState, Observation, Session, Step
from wmh.engine.prompts import BASE_ENV_PROMPT, build_env_prompt
from wmh.optimize.reward import EpisodeRewardJudge, EpisodeScore
from wmh.providers.base import Embedder, Message, Provider
from wmh.retrieval import EmbeddingRetriever, Retriever
from wmh.retrieval.embedders import get_embedder
from wmh.telemetry import capture
from wmh.tracking import MeteredProvider, Phase, RunRecord, RunTracker


class WorldModel:
    def __init__(
        self,
        provider: Provider,
        retriever: Retriever,
        env_prompt: str = BASE_ENV_PROMPT,
        top_k: int = 5,
        telemetry_root: str | Path = ".wmh",
        reward_provider: Provider | None = None,
    ) -> None:
        self._provider = provider
        self._retriever = retriever
        self._env_prompt = env_prompt
        self._top_k = top_k
        self._telemetry_root = Path(telemetry_root)
        self._sessions: dict[str, Session] = {}
        # Per-session token/cost/time accounting (serve-time observability). One tracker per
        # session, started when the session is created; `session_usage` exposes running totals.
        self._trackers: dict[str, RunTracker] = {}
        # Reward judging (`score_session`) defaults to the serve provider; pass `reward_provider`
        # to judge with a different model than the one simulating the environment.
        self._reward_provider = reward_provider or provider
        # Online index enrichment (DreamGym-style): serving sessions feed generated steps back
        # into retrieval. Evaluation rollouts must NOT (see `frozen`), or one episode's
        # predictions become another's retrieved demos and results turn order-dependent.
        self._enrich_index = True

    @classmethod
    def load(
        cls,
        artifact_dir: str,
        provider: Provider,
        embedder: Embedder | None = None,
        telemetry_root: str | Path | None = None,
        reward_provider: Provider | None = None,
    ) -> WorldModel:
        """Construct from a built `.wmh/` artifact (optimized prompt + indexed replay buffer).

        `provider` serves the live world model (generation). `embedder` supplies phi for retrieval;
        when omitted we reconstruct the configured embedder (`embed_provider` + `embed_dim`), which
        defaults to the offline `HashingEmbedder` so loading needs no embedding credentials.
        `reward_provider` backs `score_session` (defaults to `provider`) — pass it to judge with a
        different model than the one simulating the environment.
        """
        config = load_config(artifact_dir)
        paths = ArtifactPaths(artifact_dir)
        env_prompt = (
            paths.optimized_prompt.read_text(encoding="utf-8")
            if paths.optimized_prompt.exists()
            else BASE_ENV_PROMPT
        )
        # Reconstruct the *same* embedder the build used (provider + dim persisted in config), so
        # query vectors match the stored matrix. A caller-supplied embedder overrides.
        retriever = EmbeddingRetriever(embedder or get_embedder(config))
        if paths.index.exists():
            retriever.load(paths.index)
        return cls(
            provider,
            retriever,
            env_prompt=env_prompt,
            top_k=config.top_k,
            telemetry_root=telemetry_root or _default_telemetry_root(artifact_dir),
            reward_provider=reward_provider,
        )

    def new_session(
        self,
        task: str | None = None,
        seed_state: EnvState | None = None,
        *,
        enrich: bool = True,
    ) -> Session:
        """Open a session. `enrich=False` keeps its steps out of the shared retrieval buffer —
        required for evaluation rollouts, whose PREDICTED observations must not become demos for
        later rollouts (order-dependent, self-reinforcing scores otherwise)."""
        session = Session(
            id=uuid.uuid4().hex, task=task, state=seed_state or EnvState(), enrich=enrich
        )
        self._sessions[session.id] = session
        tracker = RunTracker(run_id=session.id, kind="serve")
        tracker.start()
        self._trackers[session.id] = tracker
        capture(
            "wmh generated trace started",
            {"generated_trace_count": 1},
            root=self._telemetry_root,
        )
        return session

    def session_usage(self, session_id: str) -> RunRecord:
        """Return the running token/cost/time record for `session_id` (DreamGym serve metering).

        The tracker is started at `new_session` and intentionally left running for the life of the
        session (serve sessions have no explicit end), so `duration_seconds` is live wall-clock
        since the session was created — not just time spent inside `step`. Raises `KeyError` if
        `session_id` is unknown (callers that take session ids from clients should validate first,
        as the serving API does).
        """
        return self._trackers[session_id].record_summary()

    def get_session(self, session_id: str) -> Session:
        return self._sessions[session_id]

    def end_session(self, session_id: str) -> RunRecord:
        """End `session_id`: stop its metering, drop it from the model, return the final usage.

        Batch rollouts open thousands of sessions against one WorldModel; without an explicit end,
        every Session (full step history) and its still-ticking RunTracker stays resident forever.
        Raises `KeyError` for an unknown/already-ended session id.
        """
        self._sessions.pop(session_id)  # KeyError for unknown ids, same as get_session
        tracker = self._trackers.pop(session_id)
        tracker.stop()
        return tracker.record_summary()

    def sample_steps(self, n: int) -> list[Step]:
        """Return up to `n` steps from the replay buffer (used to seed the demo agent)."""
        return self._retriever.sample(n)

    def score_session(self, session_id: str) -> EpisodeScore:
        """Judge the session's rollout so far: episode reward, per-step rewards, and a critique.

        This is the RL reward signal. The reward judge sees the session's task and its full step
        history — never a gold trace — and returns everything the algorithms need in one call:
        scalar reward/success (GRPO, PPO, REINFORCE++), `step_rewards` (dense diagnostics), and
        `critique` (SDPO's teacher feedback). Raises `KeyError` for an unknown session id.

        Judge usage is metered onto the session's tracker under `Phase.JUDGE`, so `session_usage`
        keeps reward cost separate from the world model's SERVE cost.
        """
        session = self._sessions[session_id]
        tracker = self._trackers.get(session_id)
        provider = (
            MeteredProvider(self._reward_provider, tracker, base_phase=Phase.JUDGE)
            if tracker is not None
            else self._reward_provider
        )
        score = EpisodeRewardJudge(provider).score(session.task, session.history)
        # The scalar rides the final observation too, so replay-buffer consumers that read
        # Observation.reward (DreamGym-style terminal r) see the same number the API returned.
        if session.history:
            session.history[-1].observation.reward = score.reward
        return score

    @contextmanager
    def frozen(self) -> Iterator[WorldModel]:
        """Suspend online index enrichment for the duration of the block.

        Evaluation rollouts (scenario verification, score matrices) step the world model many
        times; if those generated steps were indexed, later episodes would retrieve earlier
        episodes' predictions instead of only the built trace corpus — results would depend on
        evaluation order. Serving resumes enrichment when the block exits.
        """
        previous = self._enrich_index
        self._enrich_index = False
        try:
            yield self
        finally:
            self._enrich_index = previous

    def render_step_prompt(self, session_id: str, action: Action) -> str:
        """Assemble the exact (system + user) env prompt `step` would send, without calling the LLM.

        Used by `wmh demo` to display what the world model sees. Read-only: no session mutation.
        """
        session = self._sessions[session_id]
        demos = self._retriever.topk(session.state, action, self._top_k)
        system, user = build_env_prompt(self._env_prompt, session, action, demos)
        return f"{system}\n\n=== USER ===\n{user}"

    def step(self, session_id: str, action: Action) -> Observation:
        """Predict the observation for `action` and advance the session. DreamGym Eq. (4)."""
        started = time.monotonic()
        session = self._sessions[session_id]

        # (1) retrieve top-k similar past steps conditioned on the latest state + action
        demos = self._retriever.topk(session.state, action, self._top_k)

        # (2) assemble the env prompt and (3) predict the observation
        system, user = build_env_prompt(self._env_prompt, session, action, demos)
        try:
            completion = self._provider.complete(system, [_user_message(user)])
        except Exception:
            capture(
                "wmh generated step failed",
                {
                    "success": False,
                    "duration_seconds": round(time.monotonic() - started, 3),
                },
                root=self._telemetry_root,
            )
            raise
        observation = parse_observation(completion.text)

        # serve-time metering: attribute this call's tokens/cost to the session
        usage_cost_usd: float | None = None
        tracker = self._trackers.get(session_id)
        if tracker is not None:
            # Prefer the model that actually served (failover chains set completion.model).
            served_model = completion.model or self._provider.config.model
            usage_event = tracker.record(Phase.SERVE, served_model, completion.usage)
            usage_cost_usd = usage_event.cost_usd

        self._advance(session, action, observation)
        capture(
            "wmh generated step completed",
            {
                "success": True,
                "generated_step_count": 1,
                "session_step_count": len(session.history),
                "duration_seconds": round(time.monotonic() - started, 3),
                "input_tokens": completion.usage.input_tokens,
                "output_tokens": completion.usage.output_tokens,
                "cost_usd": round(usage_cost_usd, 6) if usage_cost_usd is not None else None,
            },
            root=self._telemetry_root,
        )
        return observation

    def step_open_loop(self, session_id: str, action: Action, actual: Observation) -> Observation:
        """Predict like `step`, but advance the session with the RECORDED observation.

        Teacher-forced replay: the prediction is returned for display/scoring while the session
        continues from ground truth, so later predictions are conditioned on the real trajectory
        (the open-loop protocol used by `wmh demo` and the replay eval).
        """
        prediction = self.step(session_id, action)
        session = self._sessions[session_id]
        # Re-pin the just-appended step to the actual observation (history + scratchpad note).
        session.history[-1] = session.history[-1].model_copy(update={"observation": actual})
        return prediction

    def seed_session(self, session_id: str, steps: list[Step]) -> None:
        """Advance a session with already-recorded steps, no prediction (open-loop resume).

        Used when a replay continues on a fresh WorldModel (e.g. after a provider switch):
        teacher-forced history is the recorded trajectory, so seeding is just advancing.
        """
        session = self._sessions[session_id]
        for step in steps:
            self._advance(session, step.action, step.observation)

    def _advance(self, session: Session, action: Action, observation: Observation) -> None:
        """Append the step, fold its state note into the scratchpad, and enrich the buffer.

        `state_before` is a deep copy: `_update_state` mutates `session.state` in place, and an
        aliased reference would rewrite every recorded step (and the text the retriever embeds)
        to the post-mutation state.
        """
        step = Step(
            action=action,
            observation=observation,
            state_before=session.state.model_copy(deep=True),
            task=session.task,
        )
        session.history.append(step)
        self._update_state(session, step)
        # Enrich the shared retrieval buffer only when the model enriches online AND this session
        # opts in — eval/closed-loop rollouts pass enrich=False so predicted steps never leak in.
        if self._enrich_index and session.enrich:
            self._retriever.add(step)

    def _update_state(self, session: Session, step: Step) -> None:
        """Fold the step's effect into session.state (the env's free-text scratchpad "database").

        The world model emits a one-line `state_note` (carried in `observation.metadata`) describing
        what changed; we append it to the scratchpad so later steps in the session stay consistent
        (e.g. "booked r_900 for u_kath"). Structured state is left to explicit seeding/tooling.
        """
        note = step.observation.metadata.get("state_note")
        if isinstance(note, str) and note.strip():
            prefix = f"{session.state.scratchpad}\n" if session.state.scratchpad else ""
            session.state.scratchpad = f"{prefix}- {note.strip()}"


def _user_message(text: str) -> Message:
    return Message(role="user", content=text)


def _default_telemetry_root(artifact_dir: str | Path) -> Path:
    path = Path(artifact_dir)
    if path.parent.name == "models":
        return path.parent.parent
    return path
