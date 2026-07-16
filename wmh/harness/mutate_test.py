"""Proposer tests: scripted meta-agent replies, prompt contents, and evidence rendering."""

from __future__ import annotations

import json

import pytest

from wmh.evals.closed_loop import ClosedLoopReport, RolloutEvidence, TaskOutcome
from wmh.evals.gold import AssertionResult, GoldVerdict
from wmh.evals.tasks import TaskSpec
from wmh.harness.delta import FailureSignature
from wmh.harness.doc import HarnessDoc
from wmh.harness.mutate import parse_delta, propose_delta, render_evidence
from wmh.harness.runtime import StopReason
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind


class ScriptedProvider:
    """Returns one canned completion; records every call for assertions."""

    def __init__(self, text: str) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")
        self._text = text
        self.systems: list[str] = []
        self.users: list[str] = []
        self.temperatures: list[float] = []
        self.max_tokens_seen: list[int] = []

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        self.systems.append(system)
        self.users.append(messages[-1].content)
        self.temperatures.append(temperature)
        self.max_tokens_seen.append(max_tokens)
        return Completion(text=self._text)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201 - test fake never calls it
        raise NotImplementedError


def _trigger() -> FailureSignature:
    return FailureSignature(
        mechanism="the file was created",
        task_ids=["t1"],
        unmet_assertions=["the file was created"],
    )


def _reply(parent: HarnessDoc) -> str:
    core = parent.surface("prompt:core")
    assert core is not None
    body = {
        "expected_effect": "t1 flips to pass",
        "preconditions": {"prompt:core": core.content_hash},
        "ops": [
            {
                "op": "replace",
                "surface_id": "prompt:core",
                "content": "You are a careful agent.",
                "rationale": "verify before submitting",
            }
        ],
    }
    return f"Here is my change:\n{json.dumps(body)}"


def test_propose_delta_parses_scripted_reply_and_fills_ground_truth() -> None:
    parent = HarnessDoc.baseline("parent")
    provider = ScriptedProvider(_reply(parent))
    delta = propose_delta(parent, _trigger(), "the agent never verified its work", provider)
    assert delta is not None
    assert delta.parent_doc_hash == parent.doc_hash
    assert delta.trigger == _trigger()  # from the caller, never trusted from the model
    assert delta.expected_effect == "t1 flips to pass"
    assert [op.surface_id for op in delta.ops] == ["prompt:core"]
    assert delta.verdict is None and delta.child_doc_hash is None
    assert provider.temperatures == [0.9]


def test_prompt_shows_surfaces_with_hashes_and_evidence() -> None:
    parent = HarnessDoc.baseline("parent")
    core = parent.surface("prompt:core")
    assert core is not None
    provider = ScriptedProvider(_reply(parent))
    propose_delta(parent, _trigger(), "the agent never verified its work", provider)
    user = provider.users[0]
    # Preconditions are copy-not-guess: every surface's id and hash is printed.
    for surface in parent.surfaces:
        assert surface.id in user
        assert surface.content_hash in user
    assert parent.doc_hash in user
    assert "the agent never verified its work" in user
    assert "meta-agent improving an agent harness" in provider.systems[0]


def test_propose_delta_returns_none_on_garbage() -> None:
    parent = HarnessDoc.baseline("parent")
    trigger = _trigger()
    assert propose_delta(parent, trigger, "e", ScriptedProvider("no json at all")) is None
    # JSON that doesn't fit the delta shape is also rejected, not coerced.
    assert propose_delta(parent, trigger, "e", ScriptedProvider('{"foo": 1}')) is None
    # A shape-valid delta with a malformed op (add without kind) is rejected at parse time.
    no_kind = json.dumps(
        {
            "expected_effect": "x",
            "preconditions": {},
            "ops": [{"op": "add", "surface_id": "skill:s", "content": "c", "rationale": "r"}],
        }
    )
    assert propose_delta(parent, trigger, "e", ScriptedProvider(no_kind)) is None


@pytest.mark.parametrize(
    ("field", "invalid_text"),
    [
        ("content", "before\x00after"),
        ("rationale", "before\ud800after"),
        ("expected_effect", "before\udcffafter"),
    ],
)
def test_parse_delta_rejects_nonpersistent_model_text(field: str, invalid_text: str) -> None:
    parent = HarnessDoc.baseline("parent")
    core = parent.surface("prompt:core")
    assert core is not None
    payload = {
        "expected_effect": "t1 passes",
        "preconditions": {"prompt:core": core.content_hash},
        "ops": [
            {
                "op": "replace",
                "surface_id": "prompt:core",
                "content": "safe content",
                "rationale": "safe rationale",
            }
        ],
    }
    if field == "expected_effect":
        payload[field] = invalid_text
    else:
        payload["ops"][0][field] = invalid_text

    assert parse_delta(parent, _trigger(), json.dumps(payload)) is None


def test_parse_delta_expands_compact_exact_replacement_edits() -> None:
    parent = HarnessDoc.baseline("parent")
    core = parent.surface("prompt:core")
    assert core is not None
    old = "You are a capable command-line agent"
    assert core.content.count(old) == 1
    raw = json.dumps(
        {
            "expected_effect": "verification improves",
            "preconditions": {"prompt:core": core.content_hash},
            "ops": [
                {
                    "op": "replace",
                    "surface_id": "prompt:core",
                    "edits": [{"old": old, "new": "You are a careful agent"}],
                    "rationale": "add care",
                }
            ],
        }
    )

    delta = parse_delta(parent, _trigger(), raw)

    assert delta is not None
    assert delta.ops[0].content == core.content.replace(old, "You are a careful agent")


def test_render_evidence_shows_cluster_tasks_and_unmet_assertions() -> None:
    report = ClosedLoopReport(
        label="parent",
        per_task={
            "t1": TaskOutcome(task_id="t1", success_rate=0.0, mean_fraction=0.0, passes=2),
            "t2": TaskOutcome(task_id="t2", success_rate=1.0, mean_fraction=1.0, passes=2),
        },
    )
    tasks = [
        TaskSpec(task_id="t1", instruction="create the file", gold=["the file was created"]),
        TaskSpec(task_id="t2", instruction="easy one", gold=["done"]),
    ]
    evidence = render_evidence(_trigger(), report, tasks)
    assert "the file was created" in evidence
    assert "create the file" in evidence
    assert "0.00 over 2 passes" in evidence
    assert "[TARGET] t1" in evidence
    assert "[other] t2" in evidence
    assert "easy one" in evidence  # passing behavior remains visible in the compact scorecard


def test_render_evidence_includes_execution_trace_answer_and_judge_reason() -> None:
    report = ClosedLoopReport(
        label="parent",
        per_task={
            "t1": TaskOutcome(
                task_id="t1",
                success_rate=0.0,
                mean_fraction=0.5,
                passes=1,
                attempts=[
                    RolloutEvidence(
                        answer="I could not verify it",
                        transcript="[1] tool_call: curl endpoint\n    -> connection reset",
                        stop_reason=StopReason.SUBMITTED,
                        turns=2,
                    )
                ],
                verdicts=[
                    GoldVerdict(
                        passed=False,
                        fraction=0.5,
                        assertions=[
                            AssertionResult(
                                assertion="the value was printed",
                                passed=False,
                                why="the final answer contained no value",
                            )
                        ],
                    )
                ],
            )
        },
    )
    trigger = FailureSignature(
        mechanism="the value was printed",
        task_ids=["t1"],
        unmet_assertions=["the value was printed"],
    )

    evidence = render_evidence(
        trigger,
        report,
        [TaskSpec(task_id="t1", instruction="fetch the value", gold=["the value was printed"])],
    )

    assert "assertion_fraction=0.50" in evidence
    assert "Stop: submitted; turns=2" in evidence
    assert "I could not verify it" in evidence
    assert "connection reset" in evidence
    assert "the final answer contained no value" in evidence


def test_all_pass_evidence_asks_for_generalization() -> None:
    trigger = FailureSignature(mechanism="none: all tasks pass")
    report = ClosedLoopReport(label="x", success_rate=1.0)
    evidence = render_evidence(trigger, report, [TaskSpec(task_id="t", instruction="do it")])
    assert "passed every task" in evidence
    assert "GENERALIZE" in evidence


def test_prompt_exposes_pi_code_files_as_editable_levers() -> None:
    """A pi-node harness shows the meta agent its real source files, and the code lever names them.

    "Real harness getting optimized": the vendored pi `code:` surfaces are in the prompt with
    their content and hashes, and the system instructions call pathful `code:` files an editable
    lever — so the proposer can diff `src/...`, not just the prompt/skills.
    """
    from wmh.harness.doc import RUNTIME_KIND_ID, TOOL_POLICY_ID, Surface, SurfaceKind
    from wmh.harness.pi_vendor import pi_agent_code_surfaces

    code_surfaces = pi_agent_code_surfaces()
    parent = HarnessDoc(
        name="pi",
        surfaces=[
            Surface(id="prompt:core", kind=SurfaceKind.PROMPT, content="p"),
            Surface(id=TOOL_POLICY_ID, kind=SurfaceKind.TOOL_POLICY, content="bash\nsubmit"),
            Surface(id=RUNTIME_KIND_ID, kind=SurfaceKind.PARAM, content="pi-node"),
            *code_surfaces,
        ],
    )
    provider = ScriptedProvider(_reply(parent))
    propose_delta(parent, _trigger(), "the agent never compacts context", provider)
    user, system = provider.users[0], provider.systems[0]

    # Every pi source file is in the prompt with its id, hash, and full content.
    for surface in code_surfaces:
        assert surface.id in user
        assert surface.content_hash in user
        assert surface.content in user
    # The system instructions frame pathful code surfaces as an editable lever, not just runtime.
    assert "pathful" in system and "code:" in system


def test_proposer_reply_budget_fits_a_full_pi_file_rewrite() -> None:
    """Ops carry complete replacement content: the reply cap must fit the largest pi source.

    Regression: max_tokens=4096 truncated every real code-surface proposal (largest vendored
    file ~36 KB ≈ ~10k tokens), so multi-iteration searches silently skipped every iteration.
    """
    from wmh.harness.pi_vendor import pi_agent_code_surfaces

    parent = HarnessDoc.baseline("parent")
    provider = ScriptedProvider(_reply(parent))
    propose_delta(parent, _trigger(), "evidence", provider)
    [max_tokens] = provider.max_tokens_seen
    largest = max(len(s.content) for s in pi_agent_code_surfaces())
    # ~4 bytes/token; JSON escaping and rationale need headroom beyond the raw file.
    assert max_tokens * 4 > largest * 1.3
