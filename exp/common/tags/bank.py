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

# The built-in bank: 14 conversation tags followed by 14 turn tags, in score order. Topic
# tags are multi-label and judge the work being done, not a field that is only the subject or
# setting; harmful_request is the one legitimacy tag.
DEFAULT_TAG_BANK = TagBank(
    bank_id="experiential-default",
    version=2,
    tags=(
        _tag(
            "coding",
            _C,
            "Does this conversation substantially involve writing, changing, debugging, reviewing "
            "or explaining software code?",
            "The request is for code, scripts, SQL, configuration or shell commands, or for help "
            "with a program, codebase, build, test, error message, API or developer tool. Other "
            "topics may apply too.",
            "Code is not a substantial part: prose, analysis, advice or chat with no program to "
            "write, fix or explain. Naming a software product or calling tools through JSON is not "
            "enough.",
        ),
        _tag(
            "cybersecurity",
            _C,
            "Does the work in this conversation need computer or network security knowledge, "
            "whether defensive, authorized or malicious?",
            "The task is security work or teaching it: security testing, exploits, malware, "
            "vulnerabilities, threat detection, incident response, hardening, or attacking or "
            "defending systems, accounts or networks. A request to write malware for an intrusion "
            "counts.",
            "Security is only the subject or setting: a post about security careers, news about a "
            "breach, recovering one's own account or ordinary login code. Harm outside computers, "
            "such as weapons, drugs or stalking, does not count. Legitimacy does not matter here.",
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
            "reminders, books travel, meals or appointments, shops, or runs other everyday errands "
            "for them.",
            "There is no personal errand: coding, research, a business serving its customers, "
            "creative writing, or general questions with no task carried out for the person.",
        ),
        _tag(
            "research",
            _C,
            "Does the request call for research: gathering information from several sources or "
            "steps and combining it?",
            "The user or system prompt asks for an investigation that needs several lookups, "
            "documents, search results, papers or datasets read and combined, such as a literature "
            "review, market scan, due diligence or multi-source fact check.",
            "The request needs only a single lookup, a summary of one supplied source, an answer "
            "from memory, rewriting the user's text, coding, or casual chat.",
        ),
        _tag(
            "data_analysis",
            _C,
            "Does the work in this conversation involve analyzing data, such as querying, "
            "aggregating, charting or modeling a dataset?",
            "The task is to query, clean, aggregate, chart, model or interpret a dataset, table, "
            "CSV, spreadsheet, dataframe, log or metrics, or to run statistical tests on data. "
            "Other topics may apply too.",
            "No dataset is analyzed: fixing a query's syntax, styling a dashboard, explaining a "
            "statistics concept, general coding, or prose that mentions numbers.",
        ),
        _tag(
            "writing_editing",
            _C,
            "Does the work in this conversation involve drafting or revising prose?",
            "The user asks for natural-language text to be written, rewritten, edited, proofread, "
            "shortened or restyled, such as emails, essays, articles, posts, marketing copy, "
            "documentation or stories.",
            "The output is not prose to write or polish: code, data, short factual answers, a "
            "translation, a summary of a well-known work, in-character roleplay, actions taken "
            "with tools, or casual chat.",
        ),
        _tag(
            "tutoring",
            _C,
            "Is the assistant teaching the user so that they learn, rather than only solving a "
            "problem for them?",
            "Explanations aimed at the user's understanding, hints instead of answers, practice "
            "problems or quizzes, or checking what the user understood, or the system prompt sets "
            "the assistant up as a tutor.",
            "The assistant just does the task, even in a school subject: it answers a homework "
            "problem, writes code or drafts a document with no intent to teach.",
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
            "Does the work in this conversation need mathematics or natural science, such as "
            "solving, calculating, proving or explaining it?",
            "Math problems, proofs, calculations or quantitative reasoning, including applied math "
            "such as loan payments, interest or drug doses, or explaining or reasoning about "
            "physics, chemistry, biology, astronomy or earth science. Other topics may apply too.",
            "Math or science is only the subject or setting: translating or drafting text about "
            "it, a software bug in a science app, or simple arithmetic inside an unrelated task.",
        ),
        _tag(
            "finance_legal_compliance",
            _C,
            "Does the work in this conversation need finance, law or regulatory compliance "
            "knowledge?",
            "The task is financial, legal or compliance work or teaching it: investing, "
            "accounting, banking, lending, taxes, payments, KYC, fraud, underwriting, insurance, "
            "contracts, legal questions or rules. Applied math such as a mortgage payment counts "
            "too.",
            "Finance or law is only the setting: a customer, company or codebase from the field, "
            "such as a bug fix in a bank's app, a price mentioned in passing, or a software "
            "license header.",
        ),
        _tag(
            "health_medical",
            _C,
            "Does the work in this conversation need health, medical or clinical knowledge?",
            "The task concerns someone's health or clinical work: symptoms, diagnoses, treatments, "
            "medications, mental health, diet or exercise for a health goal or condition, medical "
            "records, clinical notes or running healthcare operations. Other topics may apply too.",
            "Health is only the subject or setting: biology taught as science, everyday food, "
            "exercise or product choices, translating or charting such material, a medical "
            "device's history, or software or a compliance check for a hospital.",
        ),
        _tag(
            "autonomous_agent",
            _C,
            "Is the assistant set up to work through many steps on its own with tools, with no "
            "person replying between steps?",
            "The system prompt or first message sets up a tool loop toward a goal, such as a "
            "coding, browsing or workflow agent told to work until done, or the first user message "
            "is an automated task.",
            "A person chats turn by turn and is expected to reply, or the setup offers no tools "
            "for a multi-step loop.",
        ),
        _tag(
            "harmful_request",
            _C,
            "Does the user ask for help that is illegal, abusive or dangerous?",
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
            "Does the user show annoyance, impatience or anger at the assistant or its work in "
            "this turn?",
            "The new user message complains about the assistant or its work, repeats a request "
            "with irritation, uses angry words, all caps or profanity aimed at it, or says it "
            "keeps failing or wasting time.",
            "The user is neutral, polite or pleased, is frustrated with someone or something else "
            "such as a boss, a vendor or a situation, or the turn has no new user message.",
        ),
        _tag(
            "positive_feedback",
            _T,
            "Does the user thank or approve of the assistant for work it delivered, in this turn?",
            "The new user message thanks, praises or approves what the assistant already did or "
            "said: thanks, great, perfect, that works, looks good.",
            "No approval of delivered work: thanks in advance, thanks followed by a correction "
            "such as thanks but that's wrong, thanking someone else, a polite please, or no new "
            "user message.",
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
            "Does something shown in this turn prove the assistant's reply wrong?",
            "The reply contradicts or misreads a tool result or the user's message, contains an "
            "arithmetic or code error visible in the reply itself, or ignores an explicit "
            "instruction in the turn.",
            "Nothing shown contradicts the reply. Claims that cannot be checked from the turn, "
            "correct calculations and plausible answers are false.",
        ),
        _tag(
            "model_refused",
            _T,
            "Does the assistant decline to do what was asked in this turn?",
            "The reply refuses, or says it cannot or will not help, citing policy, safety, ability "
            "or missing access, and does not do the requested task.",
            "The assistant does the task, even partly, only asks a clarifying question, or reports "
            "a tool failure while still trying.",
        ),
        _tag(
            "model_asked_clarification",
            _T,
            "Does the assistant ask the user a clarifying question in this turn?",
            "The reply asks the user to choose, confirm or supply missing details, such as which "
            "option, file, account or format they mean, before or instead of finishing the task.",
            "The reply asks nothing, offers optional further help, or asks which option the user "
            "prefers after already delivering the answer.",
        ),
        _tag(
            "task_finished",
            _T,
            "Does the assistant's reply in this turn deliver the final answer or say the task is "
            "done?",
            "The reply gives the complete answer the user asked for, or reports that the requested "
            "work is finished, with no tool calls or further steps pending, whether or not "
            "anything confirms it worked.",
            "The reply only calls tools, gives a progress update, asks a question, refuses, or "
            "says more work remains.",
        ),
        _tag(
            "claimed_success_without_evidence",
            _T,
            "Does the assistant claim the task is done or fixed while nothing in this turn "
            "confirms it?",
            "The reply says something is fixed, working, passing, sent or complete, but nothing in "
            "the turn shows it and no earlier work could, or the results contradict the claim. A "
            "reply can finish the task and still be true here.",
            "The reply makes no success claim, a tool result in the turn confirms it, or the claim "
            "could rest on earlier work this turn does not show, such as tests run before.",
        ),
        _tag(
            "took_external_action",
            _T,
            "Does an assistant tool call in this turn change something beyond its own working "
            "files?",
            "A tool call sends a message or email, pays or buys, deletes or modifies records in an "
            "external system, deploys, creates a ticket, posts, books, or otherwise changes an "
            "outside system.",
            "No tool calls, only read-only calls such as search, fetch, list or view, edits to "
            "files in its local workspace or repository, drafting for the user to send, or a reply "
            "that only says it did something with no tool call.",
        ),
        _tag(
            "web_lookup",
            _T,
            "Does the assistant search or browse the web in this turn?",
            "The reply calls a web search, browser or URL fetch tool, or the turn shows results "
            "returned by one, such as search hits or fetched page text.",
            "No web tool is called and no web results appear in this turn. Tools that are only "
            "available, links that are only mentioned, or answers from memory do not count.",
        ),
        _tag(
            "prompt_injection_seen",
            _T,
            "Does a tool result or retrieved content in this turn contain instructions aimed at "
            "the assistant that try to change its task or behavior?",
            "Text inside a tool result, web page, file, email or document tells the assistant to "
            "ignore its instructions, reveal secrets, take an action the user did not ask for or "
            "change its goal, whether or not the assistant obeys.",
            "No such text: ordinary content, instructions the user or system prompt gave directly, "
            "or material that discusses prompt injection without addressing the assistant.",
        ),
        _tag(
            "escalated_to_human",
            _T,
            "Does the assistant hand the case to a person, or say a person must review or approve "
            "it, in this turn?",
            "The reply or a tool call transfers the conversation to a human agent, opens an "
            "escalation or review, or says a person must review, approve or take over before it "
            "continues.",
            "The assistant handles the request itself, asks the user to choose or confirm, refuses "
            "without a handoff, or only mentions human support as an option.",
        ),
        _tag(
            "conflicting_information",
            _T,
            "Do tool results in this turn contradict each other, or contradict what the user or "
            "task stated?",
            "Two tool results disagree on a fact, or a tool result contradicts a fact the user or "
            "task gave, such as a registry listing a different owner than the one declared.",
            "The results agree with each other and with what was stated, differ only in format, or "
            "the turn has no tool results.",
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
