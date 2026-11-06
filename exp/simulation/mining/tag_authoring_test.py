"""Agentic tag authoring over scripted authors: variants, reads, repairs, failures and replay."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import (
    AssistantAction,
    BillingSource,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    OperationEconomics,
    ToolCall,
)
from exp.common.project import ProjectStore
from exp.simulation.mining.environment_review import ReviewerError
from exp.simulation.mining.questions import QuestionBankError
from exp.simulation.mining.tag_authoring import (
    AUTHOR_VARIANTS,
    AUTHORING_TYPE,
    READ_EPISODE,
    SUBMIT_TOOL,
    TagAuthor,
    TagAuthoringSpec,
    author_questions,
    authored_bank_for_view,
    authored_questions,
    sample_traces,
    validate_proposals,
)
from exp.simulation.mining.views import Unit

SNAPSHOT = ModelSnapshot(
    provider="fixture",
    model_id="author-fixture",
    billing_source=BillingSource.CUSTOMER_MANAGED,
    capabilities_sha256="a" * 64,
    connection_sha256="b" * 64,
)

PROPERTIES = {
    "A": lambda index: index % 2 == 0,
    "B": lambda index: index % 3 == 0,
    "C": lambda index: True,
}
"""Planted properties of each trace: A splits in half, B in thirds, C holds everywhere."""


def corpus(count: int = 24) -> tuple[Unit, ...]:
    """Task and observation units whose text names the planted properties each trace has."""
    units: list[Unit] = []
    for index in range(count):
        trace = f"t{index:02d}"
        held = [name for name, rule in PROPERTIES.items() if rule(index)]
        text = f"episode {trace} properties: {' '.join(f'PROP_{name}' for name in held)}"
        units.append(Unit(trace, "task", f"TASK {text}"))
        units.append(Unit(trace, "observations", f"OBS {text}"))
    return tuple(units)


def question(slug: str, text: str, sample: Sequence[str]) -> JsonObject:
    """One proposal citing the first sampled episode as yes and the second as no."""
    return {
        "slug": slug,
        "text": text,
        "rationale": "splits the sample",
        "holds_in": [sample[0]],
        "fails_in": [sample[1]],
    }


def default_proposals(variant: str, sample: Sequence[str], repairing: bool) -> list[JsonObject]:
    """Planted proposals: [A], a constant [C] and a duplicate [E] of A; [B] and ambiguous [D]."""
    if variant == "task_shape":
        return [
            question("makes_file", "[A] Does the request ask for a file?", sample),
            question("always", "[C] Is there a request?", sample),
            question("makes_output", "[E] Does the request ask for an output file?", sample),
        ]
    if variant == "environment_state":
        return [
            question("tabular", "[B] Do the tools return tables?", sample),
            question("vague", "[D] Is the state complicated?", sample),
        ]
    first = question("checks_first", "[A] Does the agent check before acting?", sample)
    if not repairing:
        first["fails_in"] = []
    return [first]


Propose = Callable[[str, Sequence[str], bool], list[JsonObject]]


@dataclass
class ScriptedAuthor:
    """Reads one sampled episode, then submits; saves responses by identity like a draw cache.

    Attributes:
        propose: Proposals per variant, sample and whether this is a repair.
        refuse: Raise for every new draw.
        requests: Every request answered, in order.
        saved: Responses by identity.
    """

    propose: Propose = default_proposals
    refuse: bool = False
    requests: list[ModelRequest] = field(default_factory=list)
    saved: dict[str, ModelResponse] = field(default_factory=dict)

    def __call__(self, request: ModelRequest, *, identity: str) -> ModelResponse:
        """Answer one draw."""
        if identity in self.saved:
            return self.saved[identity]
        if self.refuse:
            raise ReviewerError("scripted outage")
        self.requests.append(request)
        system = request.messages[0].content or ""
        variant = next(name for name, v in AUTHOR_VARIANTS.items() if v.stance in system)
        packet = json.loads(request.messages[1].content or "{}")
        sample = [row["trace_id"] for row in packet["episodes"]]
        tools = {tool.name for tool in request.tools}
        read = any(message.role == "tool" for message in request.messages)
        repairing = any("problems:" in (m.content or "") for m in request.messages)
        if READ_EPISODE in tools and not read:
            call = ToolCall(call_id="r", name=READ_EPISODE, arguments={"trace_id": sample[0]})
        else:
            call = ToolCall(
                call_id="s",
                name=SUBMIT_TOOL,
                arguments={"questions": self.propose(variant, sample, repairing)},
            )
        response = ModelResponse(
            output=AssistantAction(tool_calls=(call,)),
            model=SNAPSHOT,
            economics=OperationEconomics(),
        )
        self.saved[identity] = response
        return response


def test_every_variant_reads_then_proposes_and_a_failed_draft_is_repaired() -> None:
    """Each variant reads an episode, later variants see earlier questions, a repair is kept."""
    script = ScriptedAuthor()
    record = author_questions(
        TagAuthor(draw=script, model=SNAPSHOT), scope="family-x", units=corpus()
    )

    assert [d.variant for d in record.drafts] == list(AUTHOR_VARIANTS)
    assert all(d.failure is None and d.reads == 1 for d in record.drafts)
    assert [d.view for d in record.drafts] == ["task", "observations", "observations"]
    assert record.drafts[2].repairs == 1
    assert len(record.drafts[0].sample) == TagAuthoringSpec().sample_size
    later = json.loads(script.requests[2].messages[1].content or "{}")
    assert {row["question_id"] for row in later["existing_questions"]} == {
        "task_shape.makes_file",
        "task_shape.always",
        "task_shape.makes_output",
    }
    bank = record.candidate_bank("authored-family-x")
    assert bank.question_ids() == (
        "task_shape.makes_file",
        "task_shape.always",
        "task_shape.makes_output",
        "environment_state.tabular",
        "environment_state.vague",
        "behavior_error.checks_first",
    )
    task_part = authored_bank_for_view(bank, "task")
    assert task_part is not None and {q.family for q in task_part.questions} == {"task_shape"}
    observed = authored_bank_for_view(bank, "observations")
    assert observed is not None and len(observed.questions) == 3


def test_samples_are_deterministic_and_rounds_draw_fresh_samples() -> None:
    """One salt gives one sample; two rounds of one variant see different samples."""
    traces = [f"t{i:02d}" for i in range(30)]
    assert sample_traces(traces, 5, "s") == sample_traces(list(reversed(traces)), 5, "s")
    record = author_questions(
        TagAuthor(draw=ScriptedAuthor(), model=SNAPSHOT),
        scope="family-x",
        units=corpus(30),
        spec=TagAuthoringSpec(variants=("environment_state",), rounds=2, sample_size=6),
    )
    assert record.drafts[0].sample != record.drafts[1].sample
    assert record.drafts[1].failure is not None  # second round repeats slugs and is refused


def test_validation_names_every_structural_problem() -> None:
    """Outside citations, missing splits, repeats and non-questions are all reported at once."""
    sample = ("t00", "t01")
    with pytest.raises(ValueError) as raised:
        validate_proposals(
            {
                "questions": [
                    {"slug": "Bad Slug", "text": "Not a question at all", "holds_in": ["t00"]},
                    {
                        "slug": "dup",
                        "text": "Is it known?",
                        "holds_in": ["t00"],
                        "fails_in": ["zz"],
                    },
                ]
            },
            sample=sample,
            taken={"task_shape.dup": "Is it already known?"},
            variant="task_shape",
            limit=1,
        )
    message = str(raised.value)
    for fragment in (
        "at most 1",
        "lowercase letters",
        "question mark",
        "holds_in and fails_in",
        "already used",
        "outside the sample",
    ):
        assert fragment in message


def test_a_refused_author_is_recorded_and_an_empty_record_has_no_bank() -> None:
    """A refused draw keeps its reason; a record with no questions refuses to form a bank."""
    record = author_questions(
        TagAuthor(draw=ScriptedAuthor(refuse=True), model=SNAPSHOT),
        scope="family-x",
        units=corpus(),
    )
    assert all("scripted outage" in (d.failure or "") for d in record.drafts)
    with pytest.raises(QuestionBankError, match="produced a question"):
        record.candidate_bank("authored-family-x")


def test_a_variant_without_its_view_is_refused() -> None:
    """Authoring a task-shape variant needs task-view units."""
    observations = tuple(u for u in corpus() if u.view == "observations")
    with pytest.raises(ValueError, match="task view"):
        author_questions(
            TagAuthor(draw=ScriptedAuthor(), model=SNAPSHOT), scope="x", units=observations
        )


def test_authored_questions_replay_from_their_artifact(tmp_path: Path) -> None:
    """The saved record replays with no draw, even from an author that would refuse."""
    store = ProjectStore(tmp_path, "case").artifacts
    script = ScriptedAuthor()
    first, reference = authored_questions(
        store,
        TagAuthor(draw=script, model=SNAPSHOT),
        scope="family-x",
        units=corpus(),
        code_revision="test",
    )
    assert store.read(reference.artifact_id).manifest.artifact_type == AUTHORING_TYPE
    again, same = authored_questions(
        store,
        TagAuthor(draw=ScriptedAuthor(refuse=True), model=SNAPSHOT),
        scope="family-x",
        units=corpus(),
        code_revision="test",
    )
    assert again == first and same == reference
