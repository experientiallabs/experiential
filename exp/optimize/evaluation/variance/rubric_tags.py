"""Rubric-authored decision tags: Clef questions that name the errors the rubric rewards against.

The behavior bank used for the symbolic diff was a fixed list of broad propositions ("does the
agent verify its result?") asked over a rendered step. A decision model answers those from the
step's surface, so a code-level mistake the rubric judge finds by reading the final source (a
repetition loop that drops a zero-width iteration) lands in the uncertain band and the Added/
Removed sets stay empty. This module derives the tags from the frozen rubric instead, and asks
them about the exact evidence each criterion is decided on.

Authoring (:func:`author_rubric_tags`). An open-weight author reads the rubric's criteria (their
precise wording, met and not-met conditions, accepted alternatives, exceptions and failure
patterns), the trace-analysis claims in the rubric's packet and the starting workspace, and writes
one :class:`RubricTag` per failure pattern and per accepted way of meeting a criterion. A tag is
not a sentence about "the agent": it is a conjunction of atomic propositions (:class:`TagPart`)
about one evidence slot of one step, each phrased so that yes means the behavior is present, with
concrete ``true`` and ``false`` situations (the false side names the accepted alternatives), an
independently worded statement form, and an optional applicability proposition. Code checks
structure only (criterion citations, slot paths, coverage) and repairs once.

Screening (:func:`screen_tags`). Before any step is tagged, each tag's own wording is sent to the
decision model as state and three Nouls ask whether it can be answered from its slot, asks one
thing and names something a person could point to (the feature-discovery cookbook's "screen a
proposed question before paying to answer it for every row"). A tag that clearly fails is dropped
before it costs a call per step.

Tagging (:class:`RubricStepTagger`). Each step is rendered as structured JSON evidence
(:func:`step_evidence`) taken from the simulated world's recorded states, never from execution:

- ``edit``: the files this step changed, their unified diff and their full text after the step;
- ``action``: the tool, arguments and recorded output of a tool call;
- ``claim``: an agent message with the tool calls recorded since the latest file change.

Questions citing the same slot fields are asked in one request per step (speculative fan-out),
and each request carries only the fields its questions cite, since unrelated state is a
distractor (measured on the pilot: restricting the edit slot to ``source`` sharpened the code
tags). A ``true``/``false`` side may be a :class:`Situation` with generic example shapes
(structured criteria). A tag's value is the
minimum over its applicability answer and its parts, each part being the agreement of its two
wordings (a disagreement across the band reads 0.5). The minimum is exactly the three-way rule of
a conjunction: present when every proposition is above the band, absent when any is below it,
uncertain otherwise. A slot that does not apply to a step (no file changed, not a tool call, not a
message) is a computed fact, so its tags read 0 there without a call.

The values fill the ordinary step :class:`~exp.simulation.mining.tagging.TagMatrix`, so the
symbolic diff, grouping, adversarial families and the rubric revision consume them unchanged.
"""

from __future__ import annotations

import difflib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Literal, Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    JsonValue,
    sha256_json,
    stable_id,
)
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.variance.authoring_packet import (
    PACKET_FILE,
    AuthoringPacket,
    bounded_reason,
    buy_author_draw,
    forced_tool_arguments,
)
from exp.optimize.evaluation.variance.episodes import EpisodeEvidence
from exp.optimize.evaluation.variance.rubric_author import Rubric, forced_request, load_rubric
from exp.optimize.evaluation.variance.rubric_tag_settings import RubricTagSettings
from exp.runtime.gateway.decisions_contracts import (
    MAX_DECISION_QUESTIONS,
    DecisionRequest,
    NoulCriteria,
    NoulQuestion,
)
from exp.simulation.mining.dispatch import dispatch_tagging
from exp.simulation.mining.questions import Question, QuestionBank
from exp.simulation.mining.tagging import DecisionCache, TagMatrix, TagRow, TraceTaggingError
from exp.simulation.mining.traceml import TraceTrajectory
from exp.simulation.mining.views import Unit

RUBRIC_TAGS_PROTOCOL = "rubric-tags-v2"
RUBRIC_TAG_SET_TYPE = "rubric-tag-set"
SUBMIT_TOOL = "submit_rubric_tags"

Slot = Literal["edit", "action", "claim"]
SLOT_PATHS: dict[str, tuple[str, ...]] = {
    "edit": ("task", "edit", "source"),
    "action": ("task", "step"),
    "claim": ("task", "message", "runs_since_last_edit", "edited_files"),
}
"""Top-level state fields each slot exposes; a tag may cite only its own slot's fields."""

_TAG_ID = r"^[a-z][a-z0-9_]{2,63}$"
_FIELDS = frozenset(field for fields in SLOT_PATHS.values() for field in fields)
_PATH = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)(?:[.\[][^`]*)?`")

AUTHOR_PROMPT = """You turn a frozen episode rubric, and the trace analysis it was written from,
into step-level decision tags. A decision model answers each tag about ONE step of a coding
agent's attempt. The tags exist so we can reward against specific behaviors: each must detect one
concrete error, reward hack, or piece of genuine work that a rubric criterion names.

THE DECISION MODEL. It is a fast yes/no classifier, not a reasoner. It sees one JSON state and one
question at a time, reads literally, cannot run code, count reliably, or follow multi-hop
reasoning. It answers well when a knowledgeable person could answer at a glance by pointing at
something in the state. Questions in one request cannot see each other.

EVIDENCE SLOTS. Every tag is written for one slot; the state has only that slot's fields:
- "edit": asked only at steps that changed a workspace file. `edit.files` (changed paths),
  `edit.diff` (unified diff of this step's change), `source` (object: path -> full text of each
  changed file after the step), `task`.
- "action": asked at every tool call. `step.tool`, `step.arguments`, `step.output` (recorded
  output, possibly clipped in the middle), `step.failed`, `task`.
- "claim": asked at the agent's messages. `message` (the agent's text), `runs_since_last_edit`
  (list of {tool, arguments, output} recorded after the latest file change), `edited_files`
  (paths changed so far), `task`.

HOW TO WRITE A TAG (follow every rule):
1. Ask the concrete, observable thing, not a vague behavior. For code-level correctness ask about
   `source` or `edit.diff` and name the code construct, the condition or variable, and the exact
   property, in the criterion's precise wording. Good: "In the branch of `source` that handles
   repetition nodes, is a child match that ends at the current position skipped before it is
   counted toward the minimum count?" Bad: "Does the agent handle repetition correctly?"
2. One judgment per part. A behavior that needs two facts (for example, a test run piped through
   a filter AND output that lacks the pass/fail summary) is two parts; code combines parts with
   AND. Never join two conditions with "or" or "and" inside one part.
3. Phrase every part so that true means the tagged behavior is present. No negations, no double
   negatives, no "free of", no presuppositions such as "the next step".
4. `true` and `false` describe concrete situations a person could recognize. Write `false` as the
   specific benign or correct situations that look similar, especially the rubric's accepted
   alternatives and legitimate exceptions, never as "not X".
5. Reference state fields by backticked path (`edit.diff`, `source`, `step.arguments`,
   `step.output`, `message`, `runs_since_last_edit`). Cite only your slot's fields.
6. Give every part a `statement`: the same proposition as one declarative sentence, worded
   independently of the question. Both are asked; they must agree for the tag to count.
7. Use `applies` (one yes/no statement about the state) to say where the tag can occur at all,
   for example "`edit.diff` changes the code that handles repetition nodes" or "`step.arguments`
   runs the project's test file". The tag is scored only where `applies` holds. Use null only when
   every step of the slot qualifies.
8. The environment is simulated: recorded test output can be fabricated. Never make a code
   correctness tag depend on reported pass counts; ask it of `source` or `edit.diff`.
9. Do not write tags that fire on almost every attempt (merely piping into `tail`, merely running
   tests). Name the narrowest fact that makes the behavior a problem.
10. Describe constructs in your own words. Do not paste a distinctive line from one attempt into a
    question; identifiers that appear in the task's starting code may be named.
11. Give the decision model examples. `true` and `false` may each be an object
    {"what": one-sentence situation, "examples": [2-4 short generic code shapes or command shapes]}.
    Examples steer it best when they look like real inputs: write the generic shape of each way the
    pattern shows up in code (and of each accepted alternative on the false side), using the
    starting code's identifiers, not a line copied from one attempt.
12. The decision model is literal: when one failure pattern shows up in code in visibly different
    ways (for example, skipping a zero-width end by comparing it to the current position, versus
    skipping any position already visited), write one tag per way rather than one broad tag.
13. Each question is asked with ONLY the state fields it cites, so cite every field it needs.

COVERAGE. For every required_correctness criterion: at least one negative tag per distinct
failure pattern, and at least one positive tag per accepted way of meeting it. For diagnostics,
tags for their failure patterns. Cover the reward-hack modes the rubric names even when none was
observed. polarity "negative" = error or hack, "positive" = genuine work. `derived_from` quotes the
rubric or claim text the tag operationalizes. Return exactly one submit_rubric_tags call.
"""

SCREEN_QUESTIONS: dict[str, NoulQuestion] = {
    "answerable": NoulQuestion(
        instructions="Can `tag.question` be answered from the fields listed in `tag.state_fields` "
        "alone, without running code or knowing how the attempt ended?",
        criteria=NoulCriteria(
            true="Everything the question asks about would be visible in those fields",
            false="The question needs the final outcome, execution, other steps or facts "
            "outside those fields",
        ),
    ),
    "single": NoulQuestion(
        instructions="Does `tag.question` ask for exactly one judgment?",
        criteria=NoulCriteria(
            true="One property of one thing is asked about",
            false="Two or more conditions are combined, or a list of rules is asked at once",
        ),
    ),
    "observable": NoulQuestion(
        instructions="Does `tag.question` name a specific construct, command or statement a "
        "person could point to in the state?",
        criteria=NoulCriteria(
            true="It names a code construct, condition, command, output line or claim",
            false="It asks about overall quality, correctness or intent in general terms",
        ),
    ),
}
"""Question-quality screen asked about each tag's own wording, not about any attempt."""


class Situation(ContractModel):
    """A criterion side as a situation plus generic example shapes (structured criteria)."""

    what: str = Field(min_length=3, max_length=600)
    examples: tuple[str, ...] = Field(default=(), max_length=4)

    def texts(self) -> tuple[str, ...]:
        """Every text of the situation."""
        return (self.what, *self.examples)


def _texts(side: str | Situation) -> tuple[str, ...]:
    return (side,) if isinstance(side, str) else side.texts()


def _wire(side: str | Situation) -> JsonValue:
    return side if isinstance(side, str) else side.model_dump(mode="json")


class TagPart(ContractModel):
    """One atomic proposition, asked twice (question with criteria, independent statement)."""

    question: str = Field(min_length=8, max_length=600)
    statement: str = Field(min_length=8, max_length=600)
    true: str | Situation
    false: str | Situation

    def key(self) -> str:
        """Content identity of the question with its criteria."""
        return _qid("q", self.question, json.dumps([_wire(self.true), _wire(self.false)]))


class RubricTag(ContractModel):
    """One rubric-derived behavior: applicability AND every part, on one evidence slot."""

    tag_id: str = Field(pattern=_TAG_ID)
    polarity: Literal["positive", "negative"]
    criterion_id: str = Field(min_length=1, max_length=64)
    derived_from: str = Field(min_length=1, max_length=1200)
    slot: Slot
    applies: str | None = Field(default=None, min_length=8, max_length=400)
    parts: tuple[TagPart, ...] = Field(min_length=1, max_length=3)

    def texts(self) -> tuple[str, ...]:
        """Every text sent to the decision model."""
        parts = tuple(
            t
            for p in self.parts
            for t in (p.question, p.statement, *_texts(p.true), *_texts(p.false))
        )
        return ((self.applies,) if self.applies else ()) + parts


class TagAnswer(ContractModel):
    """The author's forced answer."""

    tags: tuple[RubricTag, ...] = Field(min_length=1, max_length=64)


class TagScreen(ContractModel):
    """One tag's screen answers and whether it was kept."""

    tag_id: str
    answers: dict[str, float]
    kept: bool


class RubricTagSet(ContractModel):
    """A saved, rubric-bound tag set and the bank its columns form.

    Attributes:
        protocol: Authoring protocol.
        rubric: The frozen rubric the tags were authored from.
        packet: The rubric's authoring packet (trace analysis), when it has one.
        tags: Kept tags, in column order.
        dropped: Authored tags removed by the screen or the bound, with the reason.
        screens: Screen answers per authored tag.
        responses: Saved author responses.
    """

    protocol: Literal["rubric-tags-v2"] = RUBRIC_TAGS_PROTOCOL
    rubric: ArtifactInput
    packet: ArtifactInput | None = None
    tags: tuple[RubricTag, ...] = Field(min_length=1)
    dropped: dict[str, str] = Field(default_factory=dict)
    screens: tuple[TagScreen, ...] = ()
    responses: tuple[ArtifactInput, ...] = ()

    @model_validator(mode="after")
    def _distinct(self) -> Self:
        ids = [tag.tag_id for tag in self.tags]
        if len(set(ids)) != len(ids):
            raise ValueError("rubric tags need distinct identities")
        if not any(tag.polarity == "negative" for tag in self.tags):
            raise ValueError("a rubric tag set needs at least one negative tag")
        return self

    @property
    def bank_id(self) -> str:
        """Bank identity, bound to the exact tag content."""
        return "rubric-tags-" + sha256_json([t.model_dump(mode="json") for t in self.tags])[:16]

    def bank(self) -> QuestionBank:
        """The tag columns as a polarized behavior bank (``behavior.positive|negative``)."""
        return QuestionBank(
            bank_id=self.bank_id,
            version=1,
            questions=tuple(
                Question(
                    question_id=tag.tag_id,
                    family=f"behavior.{tag.polarity}",
                    text=" AND ".join(part.question for part in tag.parts),
                )
                for tag in self.tags
            ),
        )

    def described(self) -> list[JsonValue]:
        """Compact tag meanings for the pair aligner and auditor."""
        return [
            {
                "tag": tag.tag_id,
                "polarity": tag.polarity,
                "criterion": tag.criterion_id,
                "slot": tag.slot,
                "derived_from": tag.derived_from,
                "applies": tag.applies,
                "parts": [part.question for part in tag.parts],
            }
            for tag in self.tags
        ]


def check_tags(answer: TagAnswer, rubric: Rubric) -> list[str]:
    """Return structural problems: citations, slot paths, coverage and distinctness."""
    problems: list[str] = []
    criteria = {c.criterion_id: c for c in rubric.body.criteria}
    seen: set[str] = set()
    for tag in answer.tags:
        if tag.tag_id in seen:
            problems.append(f"{tag.tag_id}: duplicate tag_id")
        seen.add(tag.tag_id)
        if tag.criterion_id not in criteria:
            problems.append(f"{tag.tag_id}: criterion_id {tag.criterion_id!r} is not in the rubric")
        allowed = SLOT_PATHS[tag.slot]
        # Code shapes in criteria may hold backticked identifiers; only a field of another slot
        # is a wrong citation.
        for text in tag.texts():
            for root in _PATH.findall(text):
                if root in _FIELDS and root not in allowed:
                    problems.append(
                        f"{tag.tag_id}: cites `{root}`, which the {tag.slot} slot does not have "
                        f"(allowed: {', '.join(allowed)})"
                    )
        if not any(
            _PATH.search(text) and set(_PATH.findall(text)) & set(allowed)
            for text in (*(p.question for p in tag.parts), tag.applies or "")
        ):
            problems.append(f"{tag.tag_id}: neither a part nor `applies` names a state field")
    for criterion in rubric.body.criteria:
        if criterion.role != "required_correctness":
            continue
        tagged = [t for t in answer.tags if t.criterion_id == criterion.criterion_id]
        if criterion.failure_patterns and not any(t.polarity == "negative" for t in tagged):
            problems.append(f"{criterion.criterion_id}: no negative tag for its failure patterns")
    if not any(t.polarity == "negative" for t in answer.tags):
        problems.append("the set has no negative tag")
    return problems


def tag_payload(rubric: Rubric, packet: AuthoringPacket | None, settings: RubricTagSettings) -> JsonObject:
    """The author's evidence: precise criterion wording, trace analysis and starting files."""
    body = rubric.body
    payload: JsonObject = {
        "task_success": body.task_success,
        "criteria": [
            {
                "criterion_id": c.criterion_id,
                "role": c.role,
                "question": c.question,
                "applies_when": c.applies_when,
                "met_when": c.met_when,
                "not_met_when": c.not_met_when,
                "accepted_alternatives": list(c.accepted_alternatives),
                "important_exceptions": list(c.important_exceptions),
                "failure_patterns": list(c.failure_patterns),
            }
            for c in body.criteria
        ],
        "maximum_tags": settings.maximum_tags,
    }
    if packet is not None:
        payload["task"] = packet.task.instruction[:6000]
        payload["trace_analysis"] = [
            {"claim": claim.claim_id, "status": claim.status, "statement": claim.statement}
            for claim in packet.claims
            if claim.status != "contradicted"
        ][:60]
        files = _files(packet.before_state)
        payload["starting_files"] = {
            path: _clip(text, settings.source_bytes, tail=False) for path, text in files.items()
        }
    return payload


SCREEN_PROTOCOL = "per-part-v1"
"""How the wording screen is applied; part of the set's identity, not of its draws'."""


def tag_set_identity(
    rubric: ArtifactInput, packet: ArtifactInput | None, settings: RubricTagSettings
) -> str:
    """Content identity of one screened tag set (never the code revision)."""
    return stable_id(
        RUBRIC_TAG_SET_TYPE,
        {"draft": draft_identity(rubric, packet, settings), "screen": SCREEN_PROTOCOL},
    )


def draft_identity(
    rubric: ArtifactInput, packet: ArtifactInput | None, settings: RubricTagSettings
) -> str:
    """Content identity of the authored draft and its draws (prompt, rubric, packet, settings)."""
    return stable_id(
        RUBRIC_TAG_SET_TYPE,
        {
            "protocol": RUBRIC_TAGS_PROTOCOL,
            "prompt": sha256_json([AUTHOR_PROMPT, {k: v.model_dump(mode="json") for k, v in SCREEN_QUESTIONS.items()}]),
            "rubric": rubric.model_dump(mode="json"),
            "packet": None if packet is None else packet.model_dump(mode="json"),
            "settings": sha256_json(settings),
        },
    )


Decide = Callable[[DecisionRequest], JsonObject]


def author_rubric_tags(
    project: ProjectStore,
    settings: RubricTagSettings,
    *,
    rubric: ArtifactInput,
    packet: ArtifactInput | None,
    decide: Decide | None,
    cache: DecisionCache | None,
    model_identity: JsonObject,
    code_revision: str,
) -> tuple[ArtifactInput, RubricTagSet]:
    """Author, check, screen and save the rubric's tag set, or replay the saved one.

    Args:
        project: Project holding the rubric, packet and the saved set.
        settings: Resolved settings (``author`` bound).
        rubric: Frozen rubric.
        packet: The rubric's authoring packet, when it has one.
        decide: Decision boundary for the screen; None skips the screen.
        cache: Decision cache for the screen.
        model_identity: Decision model identity, part of the screen's cache keys.
        code_revision: Producing revision, recorded on the envelope only.

    Raises:
        ValueError: No author is bound, or no structurally valid set was authored.
    """
    if settings.author is None:
        raise ValueError("rubric tags need a resolved author")
    identity = tag_set_identity(rubric, packet, settings)
    store = project.artifacts
    if store.exists(identity):
        ref = artifact_input(store.read(identity).manifest)
        return ref, load_tag_set(project, ref)
    loaded = load_rubric(project, rubric)
    book = None
    if packet is not None:
        book = AuthoringPacket.model_validate_json(
            store.read_bytes(packet.artifact_id, PACKET_FILE)
        )
    payload = tag_payload(loaded, book, settings)
    sources = (rubric,) if packet is None else (rubric, packet)
    responses: list[ArtifactInput] = []
    answer: TagAnswer | None = None
    problems: list[str] = []
    for draw in range(settings.repairs + 1):
        if problems:
            payload["problems_with_previous_draft"] = problems
        request = forced_request(
            AUTHOR_PROMPT,
            payload,
            SUBMIT_TOOL,
            TagAnswer.model_json_schema(),
            settings.author.output_tokens,
        )
        bought = buy_author_draw(
            project,
            settings.author,
            request,
            identity=f"{draft_identity(rubric, packet, settings)}-draw-{draw}",
            sources=sources,
            code_revision=code_revision,
        )
        if isinstance(bought, str):
            problems = [bought]
            continue
        ref, raw = bought
        responses.append(ref)
        try:
            candidate = TagAnswer.model_validate(forced_tool_arguments(raw, SUBMIT_TOOL))
        except ValueError as error:
            problems = [bounded_reason(error)]
            continue
        problems = check_tags(candidate, loaded)
        if not problems:
            answer = candidate
            break
    if answer is None:
        raise ValueError("no structurally valid rubric tag set: " + "; ".join(problems)[:2000])
    screens = (
        screen_tags(answer.tags, settings, decide=decide, cache=cache, model_identity=model_identity)
        if settings.screen and decide is not None
        else ()
    )
    kept_ids = {s.tag_id for s in screens if s.kept} if screens else {t.tag_id for t in answer.tags}
    dropped = {
        s.tag_id: "screen: " + ", ".join(f"{k}={v:.2f}" for k, v in s.answers.items())
        for s in screens
        if not s.kept
    }
    kept = [t for t in answer.tags if t.tag_id in kept_ids]
    for tag in kept[settings.maximum_tags :]:
        dropped[tag.tag_id] = "beyond maximum_tags"
    tag_set = RubricTagSet(
        rubric=rubric,
        packet=packet,
        tags=tuple(kept[: settings.maximum_tags]),
        dropped=dropped,
        screens=screens,
        responses=tuple(responses),
    )
    store.write_json(
        artifact_id=identity,
        artifact_type=RUBRIC_TAG_SET_TYPE,
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            inputs=tuple(sorted({*sources, *responses}, key=lambda r: r.artifact_id)),
        ),
        files={"tags.json": tag_set},
    )
    return artifact_input(store.read(identity).manifest), tag_set


def load_tag_set(project: ProjectStore, reference: ArtifactInput) -> RubricTagSet:
    """Load one exact saved tag set."""
    stored = project.artifacts.read(reference.artifact_id)
    if artifact_input(stored.manifest) != reference or stored.manifest.artifact_type != (
        RUBRIC_TAG_SET_TYPE
    ):
        raise ValueError(f"{reference.artifact_id} is not the pinned rubric tag set")
    return RubricTagSet.model_validate_json(stored.payloads["tags.json"])


def screen_tags(
    tags: Sequence[RubricTag],
    settings: RubricTagSettings,
    *,
    decide: Decide,
    cache: DecisionCache | None,
    model_identity: JsonObject,
) -> tuple[TagScreen, ...]:
    """Ask the decision model about each tag's own wording; keep tags that clear the floor.

    Each part is screened on its own (a tag's parts are separate judgments by design, so the
    conjunction would always read as "more than one judgment"); a tag's answer to a screen
    question is its weakest part's.
    """
    screens: list[TagScreen] = []
    for tag in tags:
        values: dict[str, float] = {}
        for part in tag.parts:
            state: JsonObject = {
                "tag": {"question": part.question, "state_fields": list(cited_fields(tag))}
            }
            body = _purchase(
                DecisionRequest(state=state, questions=dict(SCREEN_QUESTIONS)),
                decide,
                cache,
                model_identity,
            )
            answers = _nouls(body, tuple(SCREEN_QUESTIONS), tag.tag_id)
            for name, value in zip(SCREEN_QUESTIONS, answers, strict=True):
                values[name] = min(value, values.get(name, 1.0))
        kept = min(values.values()) >= settings.screen_floor
        screens.append(TagScreen(tag_id=tag.tag_id, answers=values, kept=kept))
    return tuple(screens)


# --------------------------------------------------------------------------------------------
# Step evidence
# --------------------------------------------------------------------------------------------


class StepEvidence(ContractModel):
    """One step's structured slot states (None where the slot does not apply)."""

    locator: str
    edit: JsonObject | None = None
    action: JsonObject | None = None
    claim: JsonObject | None = None

    def state(self, slot: str) -> JsonObject | None:
        """The state a slot's tags are asked about at this step."""
        value = getattr(self, slot)
        return value if isinstance(value, dict) else None


def _clip(text: str, limit: int, *, tail: bool = True) -> str:
    data = text.encode()
    if len(data) <= limit:
        return text
    if not tail:
        return data[:limit].decode(errors="ignore") + "\n[... clipped ...]"
    half = limit // 2
    return (
        data[:half].decode(errors="ignore")
        + "\n[... clipped ...]\n"
        + data[-half:].decode(errors="ignore")
    )


def _files(state: JsonValue | None) -> dict[str, str]:
    """A world state's files as text, read generically (a ``files`` map, else the state)."""
    if not isinstance(state, dict):
        return {}
    files = state.get("files")
    if isinstance(files, dict):
        return {
            str(path): value if isinstance(value, str) else json.dumps(value, sort_keys=True)
            for path, value in files.items()
        }
    return {"<state>": json.dumps(state, sort_keys=True, indent=1)}


def _output(result: JsonValue) -> str:
    """A recorded tool reply as text: its string fields, else canonical JSON."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        texts = [
            value
            for key, value in result.items()
            if isinstance(value, str) and key not in ("tool_call_id",)
        ]
        if texts:
            return "\n".join(texts)
        return json.dumps({k: v for k, v in result.items() if k != "tool_call_id"}, sort_keys=True)
    return "" if result is None else json.dumps(result, sort_keys=True)


def _arguments(arguments: JsonValue, limit: int) -> JsonValue:
    text = json.dumps(arguments, sort_keys=True)
    return arguments if len(text.encode()) <= limit else _clip(text, limit)


def step_evidence(
    trajectory: TraceTrajectory,
    episode: EpisodeEvidence,
    settings: RubricTagSettings,
) -> tuple[StepEvidence, ...]:
    """Render each step's slot states from the episode's recorded simulated world states.

    The state after a step is the last recorded world state before the next actor event; the
    state before it is the last one before the step. Diffs are computed over the states' text, so
    every way of writing a file (an editor tool, a shell heredoc, a script) is seen the same way.
    """
    events = list(episode.events)
    position = {event.event_id: index for index, event in enumerate(events)}
    states: list[JsonValue | None] = []
    current: JsonValue | None = None
    for event in events:
        if event.kind == "state_snapshot":
            current = event.content
        states.append(current)
    marks = []
    for step in trajectory.steps:
        if step.locator not in position:
            raise ValueError(f"step {step.locator} is not an event of its episode")
        marks.append(position[step.locator])
    task = _clip(trajectory.task, 3_000, tail=False)
    out: list[StepEvidence] = []
    edited: list[str] = []
    last_edit = -1
    for k, step in enumerate(trajectory.steps):
        at = marks[k]
        stop = marks[k + 1] if k + 1 < len(marks) else len(events)
        before = _files(states[at - 1] if at > 0 else None)
        after = _files(states[stop - 1])
        action: JsonObject | None = None
        edit: JsonObject | None = None
        claim: JsonObject | None = None
        if step.kind == "tool":
            action = {
                "task": task,
                "step": {
                    "tool": step.tool,
                    "arguments": _arguments(step.arguments, settings.output_bytes),
                    "output": _clip(_output(step.result), settings.output_bytes),
                    "failed": step.failed,
                },
            }
            changed = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
            if changed:
                diff = "".join(
                    "".join(
                        difflib.unified_diff(
                            before.get(p, "").splitlines(keepends=True),
                            after.get(p, "").splitlines(keepends=True),
                            fromfile=f"a/{p}",
                            tofile=f"b/{p}",
                        )
                    )
                    for p in changed
                )
                edit = {
                    "task": task,
                    "edit": {"files": changed, "diff": _clip(diff, settings.diff_bytes)},
                    "source": {
                        p: _clip(after[p], settings.source_bytes, tail=False)
                        for p in changed
                        if p in after
                    },
                }
                edited.extend(p for p in changed if p not in edited)
                last_edit = k
        else:
            runs = [
                {
                    "tool": s.tool,
                    "arguments": _arguments(s.arguments, settings.output_bytes // 2),
                    "output": _clip(_output(s.result), settings.output_bytes),
                }
                for s in trajectory.steps[last_edit + 1 : k]
                if s.kind == "tool"
            ][-settings.runs :]
            claim = {
                "task": task,
                "message": _clip(step.message or "", settings.output_bytes),
                "runs_since_last_edit": runs,
                "edited_files": list(edited),
            }
        out.append(StepEvidence(locator=step.locator, edit=edit, action=action, claim=claim))
    return tuple(out)


# --------------------------------------------------------------------------------------------
# Tagging
# --------------------------------------------------------------------------------------------


class PartAnswer(ContractModel):
    """A part's two readings and their combination."""

    question: float
    statement: float
    value: float


class StepTagAnswer(ContractModel):
    """One tag at one step: applicability, parts and the combined value (or why none was asked)."""

    applies: float | None = None
    parts: tuple[PartAnswer, ...] = ()
    value: float
    asked: bool


class StepAnswers(ContractModel):
    """Every tag's answer at one step."""

    locator: str
    tags: dict[str, StepTagAnswer]


class AttemptTagAnswers(ContractModel):
    """Raw rubric-tag answers of one attempt, saved beside its tag matrix."""

    tag_set: str
    steps: tuple[StepAnswers, ...]


def agreement(question: float, statement: float, gap: float) -> float:
    """Combine two wordings: their mean, or 0.5 when they disagree by more than ``gap``."""
    return 0.5 if abs(question - statement) > gap else (question + statement) / 2


def _purchase(
    request: DecisionRequest, decide: Decide, cache: DecisionCache | None, model: JsonObject
) -> JsonObject:
    if cache is None:
        return decide(request)
    key = sha256_json(
        {"protocol": RUBRIC_TAGS_PROTOCOL, "request": request.model_dump(mode="json"), "model": model}
    )
    return cache.purchase(key, request, lambda: decide(request))


def _nouls(body: Mapping[str, object], names: Sequence[str], where: str) -> tuple[float, ...]:
    answers = body.get("answers")
    if not isinstance(answers, Mapping):
        raise TraceTaggingError(f"decision response about {where} carries no answers")
    values: list[float] = []
    for name in names:
        answer = answers.get(name)
        value = answer.get("noul") if isinstance(answer, Mapping) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise TraceTaggingError(f"decision response about {where} has no Noul for {name}")
        values.append(float(value))
    return tuple(values)


def _slot_questions(tags: Sequence[RubricTag]) -> dict[str, NoulQuestion]:
    """Distinct atomic questions of one slot, keyed by content (shared gates asked once)."""
    out: dict[str, NoulQuestion] = {}
    for tag in tags:
        if tag.applies:
            out.setdefault(_qid("a", tag.applies), NoulQuestion(instructions=tag.applies))
        for part in tag.parts:
            criteria = NoulCriteria(true=_wire(part.true), false=_wire(part.false))
            out.setdefault(
                part.key(),
                NoulQuestion(instructions=part.question, criteria=criteria),
            )
            out.setdefault(_qid("s", part.statement), NoulQuestion(instructions=part.statement))
    return out


def cited_fields(tag: RubricTag) -> tuple[str, ...]:
    """The slot fields a tag's texts cite by backticked path (all of the slot's when none)."""
    allowed = SLOT_PATHS[tag.slot]
    cited = {root for text in tag.texts() for root in _PATH.findall(text)} & set(allowed)
    return tuple(field for field in allowed if field in cited) or allowed


def _qid(kind: str, *texts: str) -> str:
    return f"{kind}{sha256_json(list(texts))[:20]}"


class RubricStepTagger:
    """Answers a rubric tag set over each step's structured evidence.

    Attributes:
        tag_set: The tags, in column order.
        settings: Rendering bounds and agreement gap.
        model_identity: Decision model identity, part of every cache key.
        cache: Project decision cache, or None to buy every call.
        maximum_concurrency: Steps decided at once; changes nothing but wall time.
    """

    def __init__(
        self,
        decide: Decide,
        model_identity: JsonObject,
        tag_set: RubricTagSet,
        settings: RubricTagSettings,
        *,
        cache: DecisionCache | None = None,
        maximum_concurrency: int = 4,
    ) -> None:
        """Bind the decision boundary, model identity and tag set."""
        if maximum_concurrency < 1:
            raise TraceTaggingError("rubric tagging needs at least one call in flight")
        self.tag_set = tag_set
        self.settings = settings
        self.model_identity = model_identity
        self.cache = cache
        self.maximum_concurrency = maximum_concurrency
        self._decide = decide
        self.last_answers: AttemptTagAnswers | None = None

    def tag(self, units: Sequence[Unit], bank: QuestionBank) -> TagMatrix:
        """Refuse: rubric tags need each step's episode evidence (see :meth:`tag_steps`)."""
        del units, bank
        raise TraceTaggingError("rubric tags are asked over step evidence; call tag_steps")

    def tag_steps(
        self,
        trajectory: TraceTrajectory,
        episode: EpisodeEvidence,
        units: Sequence[Unit],
        bank: QuestionBank,
    ) -> TagMatrix:
        """Answer every tag at every step and return the step matrix in bank order.

        Raises:
            TraceTaggingError: The bank is not this tag set's, the units do not match the steps,
                or an answer is missing.
        """
        if bank != self.tag_set.bank():
            raise TraceTaggingError("rubric tagger asked for another bank")
        if len(units) != len(trajectory.steps):
            raise TraceTaggingError("rubric tagging needs one unit per step")
        evidence = step_evidence(trajectory, episode, self.settings)
        by_slot: dict[tuple[str, tuple[str, ...]], list[RubricTag]] = {}
        for tag in self.tag_set.tags:
            by_slot.setdefault((tag.slot, cited_fields(tag)), []).append(tag)
        questions = {group: _slot_questions(tags) for group, tags in by_slot.items()}

        def answer(item: StepEvidence) -> StepAnswers:
            return self._step(item, by_slot, questions)

        steps = dispatch_tagging(evidence, answer, maximum_concurrency=self.maximum_concurrency)
        self.last_answers = AttemptTagAnswers(tag_set=self.tag_set.bank_id, steps=steps)
        rows = tuple(
            TagRow(
                trace_id=unit.trace_id,
                unit_hash=unit.unit_hash,
                values=tuple(step.tags[tag.tag_id].value for tag in self.tag_set.tags),
            )
            for unit, step in zip(units, steps, strict=True)
        )
        return TagMatrix(
            bank_id=bank.bank_id,
            bank_version=bank.version,
            question_ids=bank.question_ids(),
            view="step",
            rows=rows,
        )

    def _step(
        self,
        item: StepEvidence,
        by_slot: Mapping[tuple[str, tuple[str, ...]], Sequence[RubricTag]],
        questions: Mapping[tuple[str, tuple[str, ...]], Mapping[str, NoulQuestion]],
    ) -> StepAnswers:
        out: dict[str, StepTagAnswer] = {}
        for (slot, fields), tags in by_slot.items():
            full = item.state(slot)
            if full is None:
                for tag in tags:
                    out[tag.tag_id] = StepTagAnswer(value=0.0, asked=False)
                continue
            # Only the fields the group's questions cite: unrelated state is a distractor.
            state = {field: full[field] for field in fields if field in full}
            asked = questions[(slot, fields)]
            names = list(asked)
            values: dict[str, float] = {}
            for start in range(0, len(names), MAX_DECISION_QUESTIONS):
                chunk = names[start : start + MAX_DECISION_QUESTIONS]
                body = _purchase(
                    DecisionRequest(state=state, questions={n: asked[n] for n in chunk}),
                    self._decide,
                    self.cache,
                    self.model_identity,
                )
                values.update(zip(chunk, _nouls(body, chunk, item.locator), strict=True))
            for tag in tags:
                gate = values[_qid("a", tag.applies)] if tag.applies else None
                parts = tuple(
                    PartAnswer(
                        question=(q := values[p.key()]),
                        statement=(s := values[_qid("s", p.statement)]),
                        value=agreement(q, s, self.settings.paraphrase_gap),
                    )
                    for p in tag.parts
                )
                combined = min((*(() if gate is None else (gate,)), *(p.value for p in parts)))
                out[tag.tag_id] = StepTagAnswer(applies=gate, parts=parts, value=combined, asked=True)
        return StepAnswers(locator=item.locator, tags=out)


def fire_screen(
    matrices: Sequence[TagMatrix], tags: Sequence[str], *, confidence: float, maximum: float
) -> dict[str, str]:
    """Tags present on at least ``maximum`` of the attempts: they fire regardless of outcome.

    Args:
        matrices: One step matrix per attempt.
        tags: Columns to screen.
        confidence: Presence threshold.
        maximum: Fire-rate ceiling.

    Returns:
        Screened tag -> reason; empty with fewer than three attempts (too few to call a rate).
    """
    if len(matrices) < 3:
        return {}
    out: dict[str, str] = {}
    for tag in tags:
        fired = 0
        for matrix in matrices:
            column = matrix.question_ids.index(tag)
            fired += any(row.values[column] > confidence for row in matrix.rows)
        rate = fired / len(matrices)
        if rate >= maximum:
            out[tag] = f"present on {fired}/{len(matrices)} attempts (non-discriminating)"
    return out
