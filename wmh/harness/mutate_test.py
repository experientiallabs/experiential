"""Proposer tests: scripted meta-agent replies, prompt contents, and evidence rendering."""

from __future__ import annotations

import json

from wmh.evals.closed_loop import ClosedLoopReport, TaskOutcome
from wmh.evals.tasks import TaskSpec
from wmh.harness.delta import FailureSignature
from wmh.harness.doc import HarnessDoc
from wmh.harness.mutate import propose_delta, render_evidence
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind


class ScriptedProvider:
    """Returns one canned completion; records every call for assertions."""

    def __init__(self, text: str) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")
        self._text = text
        self.systems: list[str] = []
        self.users: list[str] = []
        self.temperatures: list[float] = []

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
    assert "easy one" not in evidence  # the passing task is not reflection fuel


def test_all_pass_evidence_asks_for_generalization() -> None:
    trigger = FailureSignature(mechanism="none: all tasks pass")
    report = ClosedLoopReport(label="x", success_rate=1.0)
    evidence = render_evidence(trigger, report, [TaskSpec(task_id="t", instruction="do it")])
    assert "passed every task" in evidence
    assert "GENERALIZE" in evidence
