"""Versioned banks of true/false tag questions asked about captured model traffic.

A tag is one proposition a decision model answers with a probability in ``[0, 1]``. Tags come
in two scopes: conversation tags are asked once about a conversation's opening, and turn tags are
asked once per response about what is new in that request. A bank fixes the exact tag ids,
wording and order; score arrays everywhere follow ``TagBank.for_scope`` order, so editing a bank
requires a new ``version``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import Field, field_validator, model_validator

from exp.common.core.artifacts import ArtifactId, ContractModel, Sha256, sha256_json

# Most tags one scope may hold, matching the per-request decision question limit.
MAX_TAGS_PER_SCOPE = 32
# Longest question, true_when or false_when text, in characters.
MAX_TAG_TEXT_CHARS = 300
# Words the tag rule language uses as keywords, so they cannot name a tag.
RESERVED_TAG_IDS = frozenset({"all", "and", "not", "or"})

TagId = Annotated[
    str, Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
]
TagText = Annotated[str, Field(min_length=1, max_length=MAX_TAG_TEXT_CHARS)]


class TagScope(StrEnum):
    """Which slice of a conversation a tag question is asked about.

    Attributes:
        CONVERSATION: Asked once per conversation about its opening: system and developer
            prompts, the tool list and the first user message.
        TURN: Asked once per response about the new slice: messages after the last assistant
            message in the request, plus the response.
    """

    CONVERSATION = "conversation"
    TURN = "turn"


class TagDefinition(ContractModel):
    """One true/false question with explicit criteria for both outcomes.

    Attributes:
        id: Stable snake_case identifier, at most 64 characters, never a rule keyword.
        scope: Whether the question is asked about the opening or about each turn.
        question: The proposition, phrased as a yes/no question about the rendered window.
        true_when: What in the window makes the answer true.
        false_when: What in the window makes the answer false, including near misses.
    """

    id: TagId
    scope: TagScope
    question: TagText
    true_when: TagText
    false_when: TagText

    @field_validator("id")
    @classmethod
    def _reject_keywords(cls, value: str) -> str:
        """Keep tag ids distinct from the rule language keywords.

        Args:
            value: Candidate tag id.

        Returns:
            The unchanged id.

        Raises:
            ValueError: The id is one of ``RESERVED_TAG_IDS``.
        """
        if value in RESERVED_TAG_IDS:
            raise ValueError(f"tag id {value!r} is a rule keyword; choose another id")
        return value

    @field_validator("question", "true_when", "false_when")
    @classmethod
    def _require_visible_text(cls, value: str) -> str:
        """Reject blank wording, which would leave the decision model guessing.

        Args:
            value: Question or criterion text.

        Returns:
            The unchanged text.

        Raises:
            ValueError: The text is only whitespace.
        """
        if not value.strip():
            raise ValueError("tag questions and criteria must contain visible text")
        return value


class TagBank(ContractModel):
    """An ordered, versioned set of tag definitions.

    Attributes:
        bank_id: Lowercase identifier shared by every version of this bank.
        version: Positive version; any change to ids, wording or order needs a new one.
        tags: Definitions in score order. Ids are unique across scopes and each scope holds at
            most ``MAX_TAGS_PER_SCOPE`` tags.
    """

    bank_id: ArtifactId
    version: int = Field(ge=1)
    tags: tuple[TagDefinition, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_tags(self) -> TagBank:
        """Require unique ids and bounded scope sizes.

        Returns:
            The validated bank.

        Raises:
            ValueError: An id repeats or one scope exceeds ``MAX_TAGS_PER_SCOPE`` tags.
        """
        seen: set[str] = set()
        for tag in self.tags:
            if tag.id in seen:
                raise ValueError(f"tag id {tag.id!r} appears more than once in the bank")
            seen.add(tag.id)
        for scope in TagScope:
            count = sum(1 for tag in self.tags if tag.scope is scope)
            if count > MAX_TAGS_PER_SCOPE:
                raise ValueError(
                    f"the {scope.value} scope has {count} tags; at most {MAX_TAGS_PER_SCOPE} "
                    "fit in one decision request, so split the bank or drop tags"
                )
        return self

    @property
    def ids(self) -> tuple[str, ...]:
        """Return every tag id in bank order."""
        return tuple(tag.id for tag in self.tags)

    @property
    def digest(self) -> Sha256:
        """Return the SHA-256 of the canonical bank JSON, covering ids, wording and order."""
        return sha256_json(self)

    def for_scope(self, scope: TagScope) -> tuple[TagDefinition, ...]:
        """Return one scope's tags in score order.

        Args:
            scope: The scope to select.

        Returns:
            The scope's definitions in bank order; position ``i`` owns score ``i``.
        """
        return tuple(tag for tag in self.tags if tag.scope is scope)

    def get(self, tag_id: str) -> TagDefinition:
        """Return one tag definition by id.

        Args:
            tag_id: The id to look up.

        Returns:
            The matching definition.

        Raises:
            KeyError: No tag has this id; the message lists every valid id.
        """
        for tag in self.tags:
            if tag.id == tag_id:
                return tag
        raise KeyError(
            f"unknown tag {tag_id!r} in bank {self.bank_id} v{self.version}; "
            f"valid tags: {', '.join(self.ids)}"
        )

    def index(self, tag_id: str) -> int:
        """Return a tag's position in its scope's score array.

        Args:
            tag_id: The id to look up.

        Returns:
            The zero-based index into ``for_scope(tag.scope)`` and into that scope's scores.

        Raises:
            KeyError: No tag has this id; the message lists every valid id.
        """
        tag = self.get(tag_id)
        return self.for_scope(tag.scope).index(tag)


def _tag(
    tag_id: str, scope: TagScope, question: str, true_when: str, false_when: str
) -> TagDefinition:
    """Build one default definition with positional wording.

    Args:
        tag_id: Stable id.
        scope: Conversation or turn scope.
        question: The yes/no question.
        true_when: Criteria for a true answer.
        false_when: Criteria for a false answer.

    Returns:
        The validated definition.
    """
    return TagDefinition(
        id=tag_id, scope=scope, question=question, true_when=true_when, false_when=false_when
    )


_C = TagScope.CONVERSATION
_T = TagScope.TURN

# The built-in bank: 15 conversation tags followed by 10 turn tags, in score order.
DEFAULT_TAG_BANK = TagBank(
    bank_id="experiential-default",
    version=1,
    tags=(
        _tag(
            "coding",
            _C,
            "Is this conversation about writing, changing, debugging, reviewing or explaining "
            "software code?",
            "The request is for code, scripts, SQL, configuration or shell commands, or for help "
            "with a program, codebase, build, test, error message, API or developer tool.",
            "Code is not the subject: prose, analysis, advice or chat with no program to write, "
            "fix or explain. Naming a software product or calling tools through JSON is not "
            "enough.",
        ),
        _tag(
            "cybersecurity",
            _C,
            "Is this conversation about computer or network security?",
            "It involves security testing, exploits, malware, vulnerabilities, penetration "
            "testing, threat detection, incident response, hardening or attacking or defending "
            "systems, accounts or networks.",
            "Security is absent or incidental, such as ordinary login code, a privacy notice, or "
            "physical safety with no computer or network security work.",
        ),
        _tag(
            "customer_support",
            _C,
            "Is the assistant acting as a business's support agent, answering its customers or "
            "users about its product or service?",
            "The system prompt sets the assistant up to help a company's customers with its "
            "products, orders, accounts, billing, policies or troubleshooting, or the user writes "
            "to such a business as its customer.",
            "No business is serving its customer: a developer building software, a person doing "
            "their own tasks, or general questions not addressed to a company's support.",
        ),
        _tag(
            "personal_assistant",
            _C,
            "Is the assistant handling personal tasks for one person, such as email, calendar, "
            "scheduling or errands?",
            "The assistant reads or writes the person's email or messages, manages a calendar or "
            "reminders, books travel, meals or appointments, shops, or runs other everyday "
            "errands for them.",
            "There is no personal errand: coding, research, a business serving its customers, "
            "creative writing, or general questions with no task carried out for the person.",
        ),
        _tag(
            "research",
            _C,
            "Is the assistant asked to gather information and combine it into an answer?",
            "The task is to look up, read, compare or synthesize facts from several documents, "
            "search results, papers or datasets, such as a literature review, market scan, due "
            "diligence or fact check.",
            "No information gathering is needed: a direct reply from memory, rewriting text the "
            "user supplied, coding, casual chat, or one simple fact.",
        ),
        _tag(
            "web_lookup",
            _C,
            "Does the request call for searching or browsing the web?",
            "The user asks to search online, open or read a web page or URL, or find current "
            "information from the internet, or the assistant's main job is web search or browsing.",
            "Nothing needs to be looked up online. Web tools that are merely available among "
            "others, or links that are only mentioned, do not count.",
        ),
        _tag(
            "data_analysis",
            _C,
            "Is this conversation about analyzing data, such as tables, SQL, spreadsheets or "
            "statistics?",
            "The user wants data queried, cleaned, aggregated, charted, modeled or interpreted, "
            "such as SQL queries, CSV or spreadsheet work, dataframes, dashboards, metrics or "
            "statistical tests.",
            "No dataset is analyzed: general coding, prose, or questions that mention numbers "
            "without analyzing them.",
        ),
        _tag(
            "writing_editing",
            _C,
            "Is this conversation about drafting or revising prose?",
            "The user asks for natural-language text to be written, rewritten, edited, "
            "proofread, shortened or restyled, such as emails, essays, articles, posts, "
            "marketing copy, documentation or stories.",
            "The output is not prose to write or polish: code, data, short factual answers, "
            "actions taken with tools, or casual chat.",
        ),
        _tag(
            "tutoring",
            _C,
            "Is the assistant teaching or explaining a subject to someone who wants to learn it?",
            "The user wants to understand or practice something, such as homework help, "
            "step-by-step explanations, practice problems or quizzes, or the system prompt sets "
            "the assistant up as a tutor.",
            "The user wants a task done rather than to learn, such as code written, a document "
            "drafted or an action taken, with no teaching intent.",
        ),
        _tag(
            "roleplay_companion",
            _C,
            "Is this conversation roleplay, interactive fiction or companionship chat?",
            "The assistant plays a character or persona, co-writes an interactive story or game, "
            "or acts as a friend, partner or companion for social or emotional conversation.",
            "The assistant acts as itself to complete practical tasks. A brand voice, a support "
            "persona or a one-off story request with no ongoing character does not count.",
        ),
        _tag(
            "math_science",
            _C,
            "Is this conversation about mathematics or natural science?",
            "It involves math problems, proofs, calculations or quantitative reasoning, or "
            "physics, chemistry, biology, astronomy, earth science or similar scientific content.",
            "Math or science is absent or trivial, such as simple arithmetic inside an unrelated "
            "task, or the subject is software, business, the arts or everyday life.",
        ),
        _tag(
            "finance_legal_compliance",
            _C,
            "Is this conversation about finance, law or regulatory compliance?",
            "It involves investing, accounting, banking, taxes, payments, KYC, fraud, "
            "underwriting, insurance, contracts, legal questions or regulatory rules.",
            "Finance and law are absent or incidental, such as a price mentioned in passing or a "
            "software license header, with no financial, legal or compliance task.",
        ),
        _tag(
            "health_medical",
            _C,
            "Is this conversation about health, medicine or clinical care?",
            "It involves symptoms, diagnoses, treatments, medications, mental health, nutrition "
            "or fitness for health, medical records, clinical notes or healthcare operations.",
            "Health is absent or incidental, such as a passing mention of feeling tired, or "
            "software and business work unrelated to healthcare.",
        ),
        _tag(
            "autonomous_agent",
            _C,
            "Is the assistant an agent working through many steps on its own, with no person "
            "replying between steps?",
            "The setup has the assistant plan and call tools in a loop toward a goal, such as a "
            "coding, browsing or workflow agent told to work until done, or the first user "
            "message is an automated task.",
            "A person chats turn by turn and is expected to reply, or the assistant answers in "
            "one step without a tool loop.",
        ),
        _tag(
            "harmful_request",
            _C,
            "Does the user ask for illegal, abusive or dangerous content?",
            "The request seeks real help with serious harm: weapons, violent attacks, sexual "
            "content involving minors, harassment or hate, self-harm methods, illegal drugs, "
            "fraud, or breaking into systems without permission.",
            "The request is benign, or treats a sensitive topic for safety, education, fiction, "
            "news or authorized defensive work without seeking harmful help.",
        ),
        _tag(
            "tool_error",
            _T,
            "Does a tool result in this turn report a failure?",
            "A tool result shows an error message, exception, stack trace, non-zero exit code, "
            "timeout, permission denied, missing file or page, or a failed status.",
            "The turn has no tool results, or every tool result succeeded. Text that only "
            "discusses errors, or a search that returns zero matches, is not a tool failure.",
        ),
        _tag(
            "user_frustrated",
            _T,
            "Does the user show annoyance, impatience or anger at the assistant in this turn?",
            "The new user message complains about the assistant, repeats a request with "
            "irritation, uses angry words, all caps or profanity aimed at it, or says it keeps "
            "failing or wasting time.",
            "The user is neutral, polite or pleased, or is upset about something other than the "
            "assistant, or the turn has no new user message.",
        ),
        _tag(
            "positive_feedback",
            _T,
            "Does the user thank, praise or approve of the assistant's work in this turn?",
            "The new user message says thanks, great, perfect, that works, looks good, or "
            "otherwise approves of what the assistant did or said.",
            "The user only gives instructions, asks questions, corrects or criticizes the "
            "assistant, or there is no new user message. A polite please in a request does not "
            "count.",
        ),
        _tag(
            "user_corrected_model",
            _T,
            "Does the user say the assistant's previous answer or action was wrong?",
            "The new user message points out a mistake, says the result is incorrect, broken or "
            "not what was asked, or tells the assistant to undo or redo what it did.",
            "The user accepts the previous work, asks a new or follow-up question, adds "
            "requirements without saying anything was wrong, or there is no new user message.",
        ),
        _tag(
            "model_mistake",
            _T,
            "Is the assistant's reply in this turn wrong, or does it contradict the evidence "
            "shown?",
            "The reply states something false, misreads a tool result or the user's message, "
            "makes a calculation or code error, ignores an explicit instruction, or contradicts "
            "what the turn shows.",
            "The reply is consistent with the request and the evidence in the turn, or its "
            "correctness cannot be judged from what is shown.",
        ),
        _tag(
            "model_refused",
            _T,
            "Does the assistant decline to do what was asked in this turn?",
            "The reply refuses, or says it cannot or will not help, citing policy, safety, "
            "ability or missing access, and does not do the requested task.",
            "The assistant does the task, even partly, only asks a clarifying question, or "
            "reports a tool failure while still trying.",
        ),
        _tag(
            "model_asked_clarification",
            _T,
            "Does the assistant ask the user a clarifying question in this turn?",
            "The reply asks the user to choose, confirm or supply missing details, such as which "
            "option, file, account or format they mean, before or instead of finishing the task.",
            "The reply asks nothing, or only offers optional further help at the end, such as "
            "asking whether anything else is needed.",
        ),
        _tag(
            "task_finished",
            _T,
            "Does the assistant's reply in this turn deliver the final answer or say the task is "
            "done?",
            "The reply gives the complete answer the user asked for, or reports that the "
            "requested work is finished, with no tool calls or further steps pending.",
            "The reply only calls tools, gives a progress update, asks a question, refuses, or "
            "says more work remains.",
        ),
        _tag(
            "claimed_success_without_evidence",
            _T,
            "Does the assistant claim the task is done or fixed without anything in this turn "
            "confirming it?",
            "The reply says something is fixed, working, passing, sent or complete, but no tool "
            "result in the turn shows it, or the results shown contradict the claim.",
            "The reply makes no success claim, or a tool result in the turn, such as passing "
            "tests, a success status or the expected output, backs the claim up.",
        ),
        _tag(
            "took_external_action",
            _T,
            "Does the assistant call a tool in this turn that changes something outside the "
            "conversation?",
            "A tool call in the reply sends a message, pays or buys, deletes, writes or moves "
            "files or records, deploys, merges, posts, books, or otherwise changes an outside "
            "system.",
            "The reply makes no tool calls, or only reads: search, fetch, list, view, or "
            "commands that inspect without changing anything.",
        ),
    ),
)

__all__ = [
    "DEFAULT_TAG_BANK",
    "MAX_TAGS_PER_SCOPE",
    "MAX_TAG_TEXT_CHARS",
    "RESERVED_TAG_IDS",
    "TagBank",
    "TagDefinition",
    "TagId",
    "TagScope",
]
