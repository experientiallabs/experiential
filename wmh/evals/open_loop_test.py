"""Tests for the `wmh eval` orchestration layer (evaluate_files), with fakes (no network)."""

from __future__ import annotations

import json

from wmh.core.types import Observation, Step
from wmh.evals.open_loop import evaluate_files
from wmh.optimize.judge import JudgeResult
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmh.retrieval import HashingEmbedder


class FakeProvider:
    def __init__(self, reply: str) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")
        self._reply = reply

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        return Completion(text=self._reply)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


class FakeJudge:
    def __init__(self, score: float) -> None:
        self._score = score

    def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult:
        return JudgeResult(score=self._score, critique="ok")


def _write_corpus(path, n_traces: int) -> None:  # noqa: ANN001 - tmp_path fixture
    """Write a tiny OTel JSONL with `n_traces` single-step traces (one tool call + reply each)."""
    lines = []
    for t in range(n_traces):
        tid = f"{t:032x}"
        lines.append(
            json.dumps(
                {
                    "traceId": tid,
                    "spanId": f"{tid[:8]}0000",
                    "name": "chat",
                    "startTimeUnixNano": 1,
                    "attributes": [
                        {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
                        {"key": "gen_ai.tool.name", "value": {"stringValue": "get_user"}},
                        {
                            "key": "gen_ai.tool.call.arguments",
                            "value": {"stringValue": json.dumps({"id": f"u{t}"})},
                        },
                        {"key": "gen_ai.prompt", "value": {"stringValue": f"look up u{t}"}},
                    ],
                }
            )
        )
        lines.append(
            json.dumps(
                {
                    "traceId": tid,
                    "spanId": f"{tid[:8]}0001",
                    "name": "execute_tool",
                    "startTimeUnixNano": 2,
                    "attributes": [
                        {"key": "gen_ai.operation.name", "value": {"stringValue": "execute_tool"}},
                        {"key": "gen_ai.tool.message", "value": {"stringValue": f"found u{t}"}},
                    ],
                }
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def test_evaluate_files_scores_and_aggregates(tmp_path) -> None:  # noqa: ANN001 - fixture
    corpus = tmp_path / "bench.otel.jsonl"
    _write_corpus(corpus, n_traces=4)

    report = evaluate_files(
        [corpus],
        "BASE",
        FakeProvider('{"output": "found u0", "is_error": false}'),
        FakeJudge(0.75),
        embedder=HashingEmbedder(dim=32),
        train_split=0.5,
    )
    assert "bench" in report.per_file
    assert report.total_steps > 0
    assert report.overall_fidelity == 0.75  # constant judge -> step-weighted mean is the score


def test_evaluate_files_uses_example_folder_name_for_standard_trace_file(tmp_path) -> None:  # noqa: ANN001
    tau = tmp_path / "examples" / "tau-bench"
    terminal = tmp_path / "examples" / "terminal-tasks"
    tau.mkdir(parents=True)
    terminal.mkdir(parents=True)
    _write_corpus(tau / "traces.otel.jsonl", n_traces=2)
    _write_corpus(terminal / "traces.otel.jsonl", n_traces=2)

    report = evaluate_files(
        [tau / "traces.otel.jsonl", terminal / "traces.otel.jsonl"],
        "BASE",
        FakeProvider('{"output": "found u0", "is_error": false}'),
        FakeJudge(0.75),
        embedder=HashingEmbedder(dim=32),
        train_split=0.5,
    )

    assert set(report.per_file) == {"tau-bench", "terminal-tasks"}


def test_evaluate_files_zero_shot_without_embedder(tmp_path) -> None:  # noqa: ANN001 - fixture
    corpus = tmp_path / "bench.otel.jsonl"
    _write_corpus(corpus, n_traces=2)
    report = evaluate_files(
        [corpus], "BASE", FakeProvider('{"output": "x"}'), FakeJudge(0.5), embedder=None
    )
    assert report.total_steps > 0
    assert report.overall_fidelity == 0.5


def test_evaluate_files_excludes_invalid_judgements_from_overall(tmp_path) -> None:  # noqa: ANN001
    class InvalidOnceJudge:
        def __init__(self) -> None:
            self.calls = 0

        def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult:
            self.calls += 1
            if self.calls == 1:
                return JudgeResult(score=0.0, critique="judge broke", valid=False)
            return JudgeResult(score=0.6, critique="ok")

    corpus = tmp_path / "bench.otel.jsonl"
    _write_corpus(corpus, n_traces=4)
    report = evaluate_files(
        [corpus],
        "BASE",
        FakeProvider('{"output": "x"}'),
        InvalidOnceJudge(),
        embedder=None,
        train_split=0.5,
    )
    # The judge failure must not drag the overall mean down as a spurious 0.0.
    assert report.overall_fidelity == 0.6
    assert report.total_invalid == 1
    assert report.total_steps > 0
    assert report.total_valid == report.total_steps - 1
    # Every reporting surface must disclose the shrunken denominator.
    assert "1 judge-invalid excluded" in report.summary()


def test_evaluate_files_empty_when_no_traces(tmp_path) -> None:  # noqa: ANN001 - fixture
    empty = tmp_path / "empty.otel.jsonl"
    empty.write_text("", encoding="utf-8")
    report = evaluate_files([empty], "BASE", FakeProvider("{}"), FakeJudge(1.0))
    assert report.total_steps == 0
    assert report.overall_fidelity == 0.0


class _AgenticFakeProvider(FakeProvider):
    """Serves both the knowledge-seeding extraction call and the prediction calls."""

    def __init__(self, reply: str) -> None:
        super().__init__(reply)
        self.seed_calls = 0
        self.predict_users: list[str] = []

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        if "KNOWLEDGE BASE" in system:  # the seeding extraction pass
            self.seed_calls += 1
            return Completion(
                text='{"rules": "- gate: lookups need a valid id", "entities": "", "schemas": ""}'
            )
        self.predict_users.append(messages[0].content)
        return Completion(text=self._reply)


def test_evaluate_files_knowledge_and_reasoning(tmp_path) -> None:  # noqa: ANN001 - fixture
    corpus = tmp_path / "bench.otel.jsonl"
    _write_corpus(corpus, n_traces=4)
    provider = _AgenticFakeProvider('{"reasoning": "r", "output": "found u0", "is_error": false}')
    report = evaluate_files(
        [corpus],
        "BASE",
        provider,
        FakeJudge(0.75),
        embedder=HashingEmbedder(dim=32),
        train_split=0.5,
        knowledge=True,
        reasoning=True,
    )
    assert report.total_steps > 0
    assert provider.seed_calls >= 1  # KB was seeded (from the train split) before scoring
    for user in provider.predict_users:
        assert "gate: lookups need a valid id" in user  # every prediction saw the seeded KB
        assert '"reasoning"' in user  # deliberate-then-answer contract requested
