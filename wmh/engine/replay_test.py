"""Tests for the replay/reconstruction-fidelity harness, with fakes (no network)."""

from __future__ import annotations

import pytest

from wmh.core.types import Action, ActionKind, EnvState, Observation, Step, Trace
from wmh.engine.grounding import GroundingResult, SourceResolver
from wmh.engine.replay import replay
from wmh.optimize.judge import JudgeResult
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmh.retrieval import EmbeddingRetriever, HashingEmbedder


class FakeProvider:
    def __init__(self, reply: str) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")
        self._reply = reply
        self.last_user: str | None = None

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        self.last_user = messages[0].content
        return Completion(text=self._reply)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


class FakeJudge:
    def __init__(self, score: float, dimensions: dict[str, float] | None = None) -> None:
        self._score = score
        self._dimensions = dimensions or {}
        self.calls = 0

    def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult:
        self.calls += 1
        return JudgeResult(score=self._score, critique="ok", dimensions=dict(self._dimensions))


class PerActionJudge:
    """Scores by tool-call arg `i` (lets a trace produce a spread of per-step scores)."""

    def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult:
        i = context.action.arguments.get("i", 0)
        return JudgeResult(score=1.0 if i == 0 else 0.0, critique="ok")


def _trace(tid: str, n: int = 2) -> Trace:
    return Trace(
        trace_id=tid,
        steps=[
            Step(
                action=Action(kind=ActionKind.TOOL_CALL, name="get_user", arguments={"i": i}),
                observation=Observation(content=f"real-{i}", is_error=False),
                state_before=EnvState(structured={"loc": "shop"}),
                task="look up",
            )
            for i in range(n)
        ],
    )


def test_replay_scores_and_aggregates() -> None:
    provider = FakeProvider('{"output": "real-0", "is_error": false}')
    report = replay("BASE", [_trace("h", n=2)], provider, FakeJudge(0.8))
    assert report.n_steps == 2
    assert report.mean_score == 0.8
    # Predicted is_error (false) matches actual (false) for both.
    assert report.error_flag_accuracy == 1.0
    assert report.results[0].actual == "real-0"


def test_replay_includes_all_prior_teacher_forced_history_by_default() -> None:
    provider = FakeProvider('{"output": "real-2", "is_error": false}')
    report = replay("BASE", [_trace("h", n=3)], provider, FakeJudge(1.0))

    assert report.n_steps == 3
    user_prompt = provider.last_user or ""
    assert "OBSERVATION (is_error=False): real-0" in user_prompt
    assert "OBSERVATION (is_error=False): real-1" in user_prompt
    assert "OBSERVATION (is_error=False): real-2" not in user_prompt


def test_replay_tracks_error_flag_mismatch() -> None:
    # Model predicts an error, but the actual observation is not an error -> flag mismatch.
    provider = FakeProvider('{"output": "boom", "is_error": true}')
    report = replay("BASE", [_trace("h", n=1)], provider, FakeJudge(0.0))
    assert report.error_flag_accuracy == 0.0
    assert report.results[0].is_error_predicted is True
    assert report.results[0].is_error_actual is False


def test_replay_rag_is_leakfree() -> None:
    # The held-out trace's own steps must never appear as demos in its prompt.
    train = [_trace("train-A", n=2)]
    holdout = [_trace("train-A", n=2)]  # same trace_id as a train trace -> must be excluded
    provider = FakeProvider('{"output": "x", "is_error": false}')
    retriever = EmbeddingRetriever(HashingEmbedder(dim=64))
    report = replay(
        "BASE", holdout, provider, FakeJudge(0.5), retriever=retriever, train=train, top_k=3
    )
    assert report.n_steps == 2
    # With train and holdout sharing the trace_id, every demo is excluded -> no leakage into prompt.
    assert "real-" not in (provider.last_user or "").split("SIMILAR PAST EXAMPLES")[-1]


def test_replay_threads_knowledge_and_reasoning_through_the_shared_assembly() -> None:
    provider = FakeProvider(
        '{"reasoning": "auth gate passed", "output": "real-0", "is_error": false}'
    )
    report = replay(
        "BASE",
        [_trace("h", n=1)],
        provider,
        FakeJudge(1.0),
        knowledge="- gate: modifying a booking requires auth",
        reasoning=True,
    )
    user = provider.last_user or ""
    assert "KNOWLEDGE BASE" in user and "gate: modifying a booking requires auth" in user
    assert '"reasoning"' in user  # deliberate-then-answer contract requested
    # The deliberation is stripped: the judge scores only what the agent would observe —
    # but it is preserved on the StepResult so humans can inspect what the env deliberated.
    assert report.results[0].predicted == "real-0"
    assert report.results[0].reasoning == "auth gate passed"


def test_replay_defaults_render_no_knowledge_section() -> None:
    provider = FakeProvider('{"output": "real-0", "is_error": false}')
    replay("BASE", [_trace("h", n=1)], provider, FakeJudge(1.0))
    assert "KNOWLEDGE BASE" not in (provider.last_user or "")


class _RecordingFetchGrounder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def ground(self, query: str) -> list[GroundingResult]:
        self.queries.append(query)
        return [GroundingResult(title=query, url=query, snippet='{"home_page": null}')]


def test_replay_grounder_prefetches_curl_get_urls_into_the_prompt() -> None:
    curl = Trace(
        trace_id="c",
        steps=[
            Step(
                action=Action(
                    kind=ActionKind.TOOL_CALL,
                    name="bash",
                    arguments={"command": "curl -s https://pypi.org/pypi/flask/json | jq .info"},
                ),
                observation=Observation(content="null"),
            )
        ],
    )
    provider = FakeProvider('{"output": "null", "is_error": false}')
    grounder = _RecordingFetchGrounder()
    replay("BASE", [curl], provider, FakeJudge(1.0), grounder=grounder)
    assert grounder.queries == ["https://pypi.org/pypi/flask/json"]
    user = provider.last_user or ""
    assert "live fetch: https://pypi.org/pypi/flask/json" in user
    assert '{"home_page": null}' in user  # the fetched body reached the model


class _DraftThenReviseProvider(FakeProvider):
    """First call returns a wrong draft; the verify call returns the correction."""

    def __init__(self) -> None:
        super().__init__("")
        self.users: list[str] = []

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        self.users.append(messages[0].content)
        if len(self.users) == 1:
            return Completion(text='{"output": "30 /tmp/folded.txt", "is_error": false}')
        return Completion(text='{"output": "29 /tmp/folded.txt", "is_error": false}')


def test_replay_verify_pass_rechecks_the_draft() -> None:
    provider = _DraftThenReviseProvider()
    report = replay("BASE", [_trace("h", n=1)], provider, FakeJudge(1.0), verify=True)
    assert len(provider.users) == 2  # draft + verify completions
    # The verify prompt contains the draft for re-examination...
    assert "YOUR DRAFT RESPONSE" in provider.users[1]
    assert "30 /tmp/folded.txt" in provider.users[1]
    # ...and the REVISED observation is what gets scored.
    assert report.results[0].predicted == "29 /tmp/folded.txt"


def test_replay_without_verify_is_single_completion() -> None:
    provider = _DraftThenReviseProvider()
    replay("BASE", [_trace("h", n=1)], provider, FakeJudge(1.0))
    assert len(provider.users) == 1


def test_replay_threads_confidence_through_the_shared_assembly() -> None:
    provider = FakeProvider('{"output": "real-0", "is_error": false, "confidence": 0.7}')
    report = replay("BASE", [_trace("h", n=1)], provider, FakeJudge(1.0), confidence=True)
    assert '"confidence"' in (provider.last_user or "")  # the contract asked for it
    # Stripped from the scored observation, carried on the StepResult (same rule as reasoning).
    assert report.results[0].predicted == "real-0"
    assert report.results[0].confidence == 0.7
    assert report.results[0].verified is False


def test_replay_confidence_why_lands_on_the_step_result() -> None:
    provider = FakeProvider(
        '{"output": "real-0", "is_error": false, '
        '"confidence_why": "demo shows the same lookup", "confidence": 0.9}'
    )
    report = replay(
        "BASE",
        [_trace("h", n=1)],
        provider,
        FakeJudge(1.0),
        confidence=True,
        confidence_why=True,
    )
    assert '"confidence_why"' in (provider.last_user or "")
    assert report.results[0].confidence_why == "demo shows the same lookup"


def test_replay_defaults_render_no_confidence_field() -> None:
    provider = FakeProvider('{"output": "real-0", "is_error": false}')
    report = replay("BASE", [_trace("h", n=1)], provider, FakeJudge(1.0))
    assert '"confidence"' not in (provider.last_user or "")
    assert report.results[0].confidence is None


class _ConfidentDraftProvider(FakeProvider):
    """Draft with a chosen stated confidence; the verify call returns a revision."""

    def __init__(self, draft: str) -> None:
        super().__init__("")
        self.users: list[str] = []
        self._draft = draft

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        self.users.append(messages[0].content)
        if len(self.users) == 1:
            return Completion(text=self._draft)
        return Completion(text='{"output": "revised", "is_error": false, "confidence": 0.9}')


def test_replay_gated_verify_runs_only_below_the_threshold() -> None:
    low = _ConfidentDraftProvider('{"output": "draft", "is_error": false, "confidence": 0.3}')
    report = replay(
        "BASE", [_trace("h", n=1)], low, FakeJudge(1.0), confidence=True, verify_below=0.7
    )
    assert len(low.users) == 2  # 0.3 < 0.7 -> the self-check ran
    assert report.results[0].predicted == "revised"
    assert report.results[0].verified is True
    # The recorded confidence belongs to the SCORED (revised) prediction, not the draft.
    assert report.results[0].confidence == 0.9

    high = _ConfidentDraftProvider('{"output": "draft", "is_error": false, "confidence": 0.8}')
    report = replay(
        "BASE", [_trace("h", n=1)], high, FakeJudge(1.0), confidence=True, verify_below=0.7
    )
    assert len(high.users) == 1  # confident draft -> no second completion spent
    assert report.results[0].predicted == "draft"
    assert report.results[0].verified is False
    assert report.results[0].confidence == 0.8


def test_replay_gated_escalation_repredicts_on_the_strong_model() -> None:
    cheap = FakeProvider('{"output": "cheap guess", "is_error": false, "confidence": 0.3}')
    strong = FakeProvider('{"output": "strong answer", "is_error": false, "confidence": 0.9}')
    report = replay(
        "BASE",
        [_trace("h", n=1)],
        cheap,
        FakeJudge(1.0),
        confidence=True,
        escalate_provider=strong,
        escalate_below=0.6,
    )
    # Unconfident cheap draft -> fresh re-prediction on the strong model is what gets scored.
    assert report.results[0].predicted == "strong answer"
    assert report.results[0].escalated is True
    assert report.results[0].confidence == 0.9
    assert strong.last_user is not None


def test_replay_gated_escalation_keeps_confident_cheap_drafts() -> None:
    cheap = FakeProvider('{"output": "cheap sure", "is_error": false, "confidence": 0.8}')
    strong = FakeProvider('{"output": "strong answer", "is_error": false, "confidence": 0.9}')
    report = replay(
        "BASE",
        [_trace("h", n=1)],
        cheap,
        FakeJudge(1.0),
        confidence=True,
        escalate_provider=strong,
        escalate_below=0.6,
    )
    assert report.results[0].predicted == "cheap sure"
    assert report.results[0].escalated is False
    assert strong.last_user is None  # the strong model was never spent


def test_replay_gated_verify_treats_missing_confidence_as_low() -> None:
    silent = _ConfidentDraftProvider('{"output": "draft", "is_error": false}')
    report = replay(
        "BASE", [_trace("h", n=1)], silent, FakeJudge(1.0), confidence=True, verify_below=0.7
    )
    assert len(silent.users) == 2  # no stated confidence -> assume low, verify
    assert report.results[0].verified is True


def test_replay_gates_without_confidence_flag_are_rejected() -> None:
    # Without confidence=True the contract never asks for a rating, every draft gates as low,
    # and a "gated" run silently pays the always-on bill — fail fast instead.
    provider = FakeProvider('{"output": "x", "is_error": false}')
    with pytest.raises(ValueError, match="confidence=True"):
        replay("BASE", [_trace("h", n=1)], provider, FakeJudge(1.0), verify_below=0.7)
    with pytest.raises(ValueError, match="confidence=True"):
        replay(
            "BASE",
            [_trace("h", n=1)],
            provider,
            FakeJudge(1.0),
            escalate_provider=provider,
            escalate_below=0.7,
        )
    # Escalation needs both halves; either alone would be a silent no-op.
    with pytest.raises(ValueError, match="together"):
        replay(
            "BASE",
            [_trace("h", n=1)],
            provider,
            FakeJudge(1.0),
            confidence=True,
            escalate_provider=provider,
        )
    with pytest.raises(ValueError, match="together"):
        replay(
            "BASE",
            [_trace("h", n=1)],
            provider,
            FakeJudge(1.0),
            confidence=True,
            escalate_below=0.5,
        )


def test_replay_verify_after_escalation_revises_on_the_strong_model() -> None:
    # The reviser must be the model whose draft is kept: a cheap-model verify pass on an
    # escalated prediction would undo the escalation on exactly the steps it was bought for.
    cheap = FakeProvider('{"output": "cheap guess", "is_error": false, "confidence": 0.2}')
    strong = _ConfidentDraftProvider(
        '{"output": "strong draft", "is_error": false, "confidence": 0.5}'
    )
    report = replay(
        "BASE",
        [_trace("h", n=1)],
        cheap,
        FakeJudge(1.0),
        confidence=True,
        escalate_provider=strong,
        escalate_below=0.6,
        verify_below=0.6,  # strong draft states 0.5 -> the verify gate fires too
    )
    assert report.results[0].escalated is True
    assert report.results[0].verified is True
    # Both the escalated prediction AND its verify revision came from the strong provider.
    assert len(strong.users) == 2
    assert "YOUR DRAFT RESPONSE" in strong.users[1]
    assert report.results[0].predicted == "revised"


def test_replay_source_resolver_grounds_first_touch_reads_only() -> None:
    def fake_fetch(url: str, headers: dict[str, str]) -> str:
        return "\n".join(f"src{i}" for i in range(1, 40))

    resolver = SourceResolver(
        {"inst-1": {"repo": "o/r", "base_commit": "c0ffee"}}, fetch=fake_fetch
    )
    trace = Trace(
        trace_id="s",
        metadata={"instance_id": "inst-1"},
        steps=[
            Step(  # write touches the file FIRST -> the later read must NOT be grounded
                action=Action(
                    kind=ActionKind.TOOL_CALL,
                    name="bash",
                    arguments={"command": "echo x >> /testbed/pkg/edited.py"},
                ),
                observation=Observation(content=""),
            ),
            Step(
                action=Action(
                    kind=ActionKind.TOOL_CALL,
                    name="bash",
                    arguments={"command": "sed -n '1,3p' /testbed/pkg/edited.py"},
                ),
                observation=Observation(content="whatever"),
            ),
            Step(  # first touch of fresh.py -> grounded
                action=Action(
                    kind=ActionKind.TOOL_CALL,
                    name="bash",
                    arguments={"command": "sed -n '2,4p' /testbed/pkg/fresh.py"},
                ),
                observation=Observation(content="src2\nsrc3\nsrc4"),
            ),
        ],
    )
    provider = FakeProvider('{"output": "src2\\nsrc3\\nsrc4", "is_error": false}')
    prompts: list[str] = []

    class _Capture(FakeProvider):
        def complete(self, system, messages, *, temperature=0.7, max_tokens=8192):  # noqa: ANN001, ANN202
            prompts.append(messages[0].content)
            return super().complete(
                system, messages, temperature=temperature, max_tokens=max_tokens
            )

    provider = _Capture('{"output": "x", "is_error": false}')
    replay("BASE", [trace], provider, FakeJudge(1.0), source=resolver)
    # 3 steps -> 3 prompts; only the fresh.py step carries the grounded source slice.
    assert len(prompts) == 3
    assert "source file: pkg/fresh.py" in prompts[2]
    assert "src2" in prompts[2]
    assert "source file" not in prompts[1]  # edited file: stale-gate held


def test_replay_never_leaks_trace_metadata_into_prompts() -> None:
    # swe traces carry the GOLD patch in metadata ("submission") — one careless injection away
    # from leaking the answer. Pin: nothing in the prompt path may see trace.metadata.
    trace = Trace(
        trace_id="g",
        metadata={"submission": "GOLD-PATCH-SECRET", "instance_id": "inst-9"},
        steps=[
            Step(
                action=Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": "ls"}),
                observation=Observation(content="a.py"),
            )
        ],
    )
    provider = FakeProvider('{"output": "a.py", "is_error": false}')
    replay(
        "BASE",
        [trace],
        provider,
        FakeJudge(1.0),
        reasoning=True,
        knowledge="- kb fact",
        verify=False,
    )
    assert "GOLD-PATCH-SECRET" not in (provider.last_user or "")


def test_replay_profile_digests_history_into_a_revised_belief_state() -> None:
    provider = _DigestThenPredictProvider()
    trace = _trace("p", n=3)
    replay("BASE", [trace], provider, FakeJudge(1.0), profile=True)
    # Step 3 (2 prior steps): digest completion first, then the prediction sees the profile.
    assert provider.digest_calls >= 1, "history digest completion missing"
    predict_prompts = [u for u in provider.users if "environment profile (revised" in u]
    assert any("service X is UP" in u for u in predict_prompts)  # digest output reached predict


def test_replay_profile_skips_historyless_first_steps() -> None:
    provider = _DigestThenPredictProvider()
    replay("BASE", [_trace("p", n=1)], provider, FakeJudge(1.0), profile=True)
    assert len(provider.users) == 1  # no history -> no digest completion


class _DigestThenPredictProvider(FakeProvider):
    """Answers digest completions with a canned profile; predictions with a canned observation."""

    def __init__(self) -> None:
        super().__init__("")
        self.users: list[str] = []
        self.digest_calls = 0

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        self.users.append(messages[0].content)
        if "CURRENT environment profile" in system:
            self.digest_calls += 1
            return Completion(text="- service X is UP\n- file /tmp/a.txt exists")
        return Completion(text='{"output": "real-0", "is_error": false}')


def test_replay_grounder_skips_non_curl_steps() -> None:
    provider = FakeProvider('{"output": "real-0", "is_error": false}')
    grounder = _RecordingFetchGrounder()
    replay("BASE", [_trace("h", n=1)], provider, FakeJudge(1.0), grounder=grounder)
    assert grounder.queries == []  # get_user tool call: nothing to fetch
    assert "live fetch" not in (provider.last_user or "")


def test_replay_empty_is_safe() -> None:
    report = replay("BASE", [], FakeProvider("{}"), FakeJudge(1.0))
    assert report.n_steps == 0
    assert report.mean_score == 0.0


def test_replay_reports_std_across_steps() -> None:
    # Two steps scored 1.0 and 0.0 -> mean 0.5, population std 0.5 across steps.
    report = replay("BASE", [_trace("h", n=2)], FakeProvider('{"output": "x"}'), PerActionJudge())
    assert report.n_steps == 2
    assert report.mean_score == 0.5
    assert report.score_std == 0.5


def test_replay_carries_rubric_dimensions() -> None:
    dims = {"format": 1.0, "factuality": 0.6, "consistency": 0.8, "realism": 1.0, "quality": 0.7}
    report = replay(
        "BASE", [_trace("h", n=1)], FakeProvider('{"output": "x"}'), FakeJudge(0.82, dims)
    )
    assert report.results[0].dimensions == dims


def test_replay_sampled_turns_scores_five_for_long_traces() -> None:
    judge = FakeJudge(0.5)
    report = replay(
        "BASE",
        [_trace("h", n=10)],
        FakeProvider('{"output": "x"}'),
        judge,
        sample_turns="sampled",
        seed=0,
    )
    assert report.n_steps == 5  # first, last, 3 middle
    # Deterministic under a fixed seed.
    judge2 = FakeJudge(0.5)
    report2 = replay(
        "BASE",
        [_trace("h", n=10)],
        FakeProvider('{"output": "x"}'),
        judge2,
        sample_turns="sampled",
        seed=0,
    )
    assert [r.action for r in report.results] == [r.action for r in report2.results]


def test_replay_sampled_turns_history_uses_original_trace_prefix() -> None:
    provider = FakeProvider('{"output": "x"}')
    replay("BASE", [_trace("h", n=10)], provider, FakeJudge(0.5), sample_turns="sampled", seed=0)

    user_prompt = provider.last_user or ""
    assert "OBSERVATION (is_error=False): real-8" in user_prompt
    assert "OBSERVATION (is_error=False): real-9" not in user_prompt


def test_replay_sample_turns_all_scores_every_step() -> None:
    report = replay(
        "BASE",
        [_trace("h", n=10)],
        FakeProvider('{"output": "x"}'),
        FakeJudge(0.5),
        sample_turns="all",
    )
    assert report.n_steps == 10


class InvalidOnceJudge:
    """First call is an invalid judgement (judge failure), the rest score 0.8."""

    def __init__(self) -> None:
        self.calls = 0

    def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult:
        self.calls += 1
        if self.calls == 1:
            return JudgeResult(score=0.0, critique="judge broke", valid=False)
        return JudgeResult(score=0.8, critique="ok")


def test_replay_excludes_invalid_judgements_from_fidelity() -> None:
    provider = FakeProvider('{"output": "real-0", "is_error": false}')
    report = replay("BASE", [_trace("h", n=2)], provider, InvalidOnceJudge())
    assert report.n_steps == 2
    assert report.n_invalid == 1
    # The judge failure is excluded from the mean, not recorded as a 0.0 prediction score.
    assert report.mean_score == 0.8
    assert report.results[0].valid is False
    assert "invalid" in report.summary()


def test_replay_all_invalid_judgements_report_zero_with_count() -> None:
    class AlwaysInvalid:
        def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult:
            return JudgeResult(score=0.0, critique="judge broke", valid=False)

    provider = FakeProvider('{"output": "x"}')
    report = replay("BASE", [_trace("h", n=2)], provider, AlwaysInvalid())
    assert report.n_steps == 2
    assert report.n_invalid == 2
    assert report.mean_score == 0.0


def test_replay_concurrency_preserves_results_and_order() -> None:
    # PerActionJudge scores by step index, so the per-step result sequence is order-sensitive:
    # concurrent scoring must return the identical ordered results as serial.
    traces = [_trace("a", n=4), _trace("b", n=3)]
    fp = FakeProvider('{"output": "x"}')
    serial = replay("BASE", traces, fp, PerActionJudge(), concurrency=1)
    parallel = replay("BASE", traces, fp, PerActionJudge(), concurrency=4)
    assert serial.n_steps == parallel.n_steps == 7
    assert [r.score for r in serial.results] == [r.score for r in parallel.results]
    assert serial.mean_score == parallel.mean_score
