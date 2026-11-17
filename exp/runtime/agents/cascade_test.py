"""The cascade delegates only accepted decisions and records who decided each step."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from exp.common.models import (
    AssistantAction,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    ToolCall,
    ToolChoice,
)
from exp.common.models.unknown_outcome import DispatchOutcomeUnknown
from exp.common.tasks import ToolSchema
from exp.runtime.agents.cascade import (
    OTHER,
    RESPOND,
    CascadeDecisionRecord,
    CascadeModelClient,
    CascadePolicy,
    DeciderDefinition,
    NextActionDecision,
    action_key,
    action_projection,
    decision_candidates,
    delegates,
)
from exp.runtime.agents.chat_test import _Model, _response
from exp.runtime.models.providers.transport import ProviderTransportError
from exp.runtime.sessions.dispatch_ledger import DispatchCapExhaustedError


def snapshot(model_id: str) -> ModelSnapshot:
    """Return a fixture model identity distinct from the main candidate."""
    return _response(AssistantAction(content="x")).model.model_copy(update={"model_id": model_id})


def definition(kind: str = "systemone") -> DeciderDefinition:
    """Return a cheap decider definition of the given kind."""
    if kind == "systemone":
        return DeciderDefinition(kind="systemone", alias="jev", model=snapshot("jev-v1"))
    return DeciderDefinition(
        kind="sampled_drafts", alias="small", model=snapshot("small-v1"), samples=3
    )


def request(*names: str) -> ModelRequest:
    """Return an actor request offering the named tools."""
    return ModelRequest(
        messages=(ModelMessage(role="user", content="Close ticket 1."),),
        tools=tuple(
            ToolSchema(
                name=name,
                description=f"{name} tool",
                input_schema={
                    "type": "object",
                    "properties": {"ticket": {"type": "integer"}},
                    "required": ["ticket"],
                },
            )
            for name in names
        ),
        tool_choice="auto",
    )


class Decider:
    """Return scripted decisions and record the requests it saw."""

    def __init__(self, kind: str, decisions: Sequence[NextActionDecision | Exception]) -> None:
        """Bind one definition and the answers served in order."""
        self._definition = definition(kind)
        self._decisions = list(decisions)
        self.seen: list[tuple[str, ...]] = []

    @property
    def definition(self) -> DeciderDefinition:
        """Return the scripted decider's frozen definition."""
        return self._definition

    def decide(self, request: ModelRequest, candidates: tuple[str, ...]) -> NextActionDecision:
        """Return or raise the next scripted answer."""
        del request
        self.seen.append(candidates)
        answer = self._decisions.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def ranked(selected: str, confidence: float, candidates: tuple[str, ...]) -> NextActionDecision:
    """Return a System One style decision putting the remaining mass on OTHER."""
    rest = {key: 0.0 for key in (*candidates, OTHER)}
    rest[selected] = confidence
    if selected != OTHER:
        rest[OTHER] = round(1 - confidence, 10)
    return NextActionDecision(selected=selected, probabilities=rest)


CANDIDATES = ("close_ticket", "open_ticket", RESPOND)
CLOSE = ToolCall(call_id="c1", name="close_ticket", arguments={"ticket": 1})


def test_decision_projection_and_reserved_options() -> None:
    """The decision is the first tool or respond; generated IDs never change an action."""
    assert decision_candidates(request("close_ticket", "open_ticket")) == CANDIDATES
    assert action_key(AssistantAction(content="done")) == RESPOND
    assert action_key(AssistantAction(tool_calls=(CLOSE,))) == "close_ticket"
    renamed = CLOSE.model_copy(update={"call_id": "other"})
    assert action_projection(AssistantAction(tool_calls=(CLOSE,))) == action_projection(
        AssistantAction(tool_calls=(renamed,))
    )
    with pytest.raises(ValueError, match="reserved cascade option"):
        decision_candidates(request("respond"))


def test_acceptance_rule_applies_restrictions_before_confidence() -> None:
    """OTHER abstains and a restricted option never delegates, whatever its probability."""
    assert delegates("close_ticket", 0.9, threshold=0.9, allowed=CANDIDATES)
    assert not delegates("close_ticket", 0.89, threshold=0.9, allowed=CANDIDATES)
    assert not delegates(OTHER, 1.0, threshold=0.0, allowed=CANDIDATES)
    assert not delegates("close_ticket", 1.0, threshold=0.0, allowed=(RESPOND,))


def test_policy_and_decider_contracts_refuse_ignored_or_ambiguous_settings() -> None:
    """Settings a kind would ignore, unsorted restrictions and bad distributions are refused."""
    with pytest.raises(ValidationError, match="takes no draft samples"):
        DeciderDefinition(kind="systemone", alias="jev", model=snapshot("jev"), samples=2)
    with pytest.raises(ValidationError, match="at least two samples"):
        DeciderDefinition(kind="sampled_drafts", alias="small", model=snapshot("small"))
    with pytest.raises(ValidationError, match="sorted"):
        CascadePolicy(decider=definition(), threshold=0.5, never_delegate=("b", "a"))
    with pytest.raises(ValidationError, match="sum to one"):
        NextActionDecision(selected="close_ticket", probabilities={"close_ticket": 0.5, OTHER: 0})
    with pytest.raises(ValidationError, match="differs from the selected decision"):
        NextActionDecision(
            selected=RESPOND,
            probabilities={RESPOND: 1.0, OTHER: 0.0},
            action=AssistantAction(tool_calls=(CLOSE,)),
            response=_response(AssistantAction(tool_calls=(CLOSE,))),
        )


def test_ranked_decision_forces_the_main_model_and_records_the_confidence() -> None:
    """An accepted System One choice delegates the decision but not its arguments."""
    main = _Model((_response(AssistantAction(tool_calls=(CLOSE,))),))
    decider = Decider("systemone", [ranked("close_ticket", 0.95, CANDIDATES)])
    records: list[CascadeDecisionRecord] = []
    cascade = CascadeModelClient(
        main=main,
        decider=decider,
        policy=CascadePolicy(decider=decider.definition, threshold=0.9),
        record=records.append,
    )
    response = cascade.complete(request("close_ticket", "open_ticket"))
    assert response.output.tool_calls == (CLOSE,)
    assert main.requests[0].tool_choice == ToolChoice(name="close_ticket")
    assert records == cascade.records
    row = records[0]
    assert (row.decided_by, row.main_called, row.followed) == ("decider", True, True)
    assert row.confidence == 0.95 and row.decision_model == decider.definition.model


def test_below_threshold_restricted_and_failed_decisions_leave_the_main_model_responsible() -> None:
    """The main model receives the original request whenever the decider is not accepted."""
    responses = tuple(_response(AssistantAction(content="done")) for _ in range(4))
    main = _Model(responses)
    decider = Decider(
        "systemone",
        [
            ranked("close_ticket", 0.5, CANDIDATES),
            ranked("open_ticket", 1.0, CANDIDATES),
            ranked(OTHER, 1.0, CANDIDATES),
            ValueError("System One returned no next-action choice"),
        ],
    )
    cascade = CascadeModelClient(
        main=main,
        decider=decider,
        policy=CascadePolicy(
            decider=decider.definition, threshold=0.9, never_delegate=("open_ticket",)
        ),
    )
    original = request("close_ticket", "open_ticket")
    for _ in range(4):
        cascade.complete(original)
    assert all(sent == original for sent in main.requests)
    assert [row.note for row in cascade.records] == [
        "below_threshold",
        "restricted",
        "abstained",
        "decider_failure:ValueError",
    ]
    assert {row.decided_by for row in cascade.records} == {"main"}
    assert cascade.records[0].decision_model == responses[0].model


@pytest.mark.parametrize(
    "failure",
    [
        DispatchOutcomeUnknown("decision dispatch saved no response"),
        ProviderTransportError("decision provider unavailable"),
        TimeoutError("decision deadline passed"),
        DispatchCapExhaustedError("decision dispatch cap reached"),
    ],
)
def test_decider_dispatch_failures_stop_the_step_without_a_main_draw(failure: Exception) -> None:
    """An unknown or failed decider dispatch is never resolved by a fresh main-model draw."""
    main = _Model((_response(AssistantAction(content="done")),))
    decider = Decider("systemone", [failure])
    cascade = CascadeModelClient(
        main=main, decider=decider, policy=CascadePolicy(decider=decider.definition, threshold=0.9)
    )
    with pytest.raises(type(failure)):
        cascade.complete(request("close_ticket", "open_ticket"))
    assert not main.requests and not cascade.records


def test_accepted_draft_avoids_the_main_call_but_invalid_drafts_do_not() -> None:
    """A schema-valid draft is executed as is; an invalid one falls back to the main model."""
    small = definition("sampled_drafts")
    draft = AssistantAction(tool_calls=(CLOSE,))
    bad = AssistantAction(tool_calls=(CLOSE.model_copy(update={"arguments": {"ticket": "x"}}),))

    def drafted(action: AssistantAction) -> NextActionDecision:
        """Return a unanimous drafted decision."""
        response = _response(action).model_copy(update={"model": small.model})
        return NextActionDecision(
            selected="close_ticket",
            probabilities={"close_ticket": 1.0, "open_ticket": 0.0, RESPOND: 0.0, OTHER: 0.0},
            action=action,
            response=response,
        )

    main = _Model((_response(AssistantAction(content="main")),))
    decider = Decider("sampled_drafts", [drafted(draft), drafted(bad)])
    cascade = CascadeModelClient(
        main=main, decider=decider, policy=CascadePolicy(decider=small, threshold=1.0)
    )
    first = cascade.complete(request("close_ticket", "open_ticket"))
    assert first.output == draft and first.model == small.model and not main.requests
    second = cascade.complete(request("close_ticket", "open_ticket"))
    assert second.output.content == "main"
    assert [(row.decided_by, row.main_called, row.note) for row in cascade.records] == [
        ("decider", False, None),
        ("main", True, "invalid_draft"),
    ]


def test_draft_with_a_restricted_later_call_falls_back_to_the_main_model() -> None:
    """Every drafted call is checked against never_delegate, not only the first."""
    small = definition("sampled_drafts")
    restricted = ToolCall(call_id="c2", name="open_ticket", arguments={"ticket": 1})
    draft = AssistantAction(tool_calls=(CLOSE, restricted))
    response = _response(draft).model_copy(update={"model": small.model})
    decision = NextActionDecision(
        selected="close_ticket",
        probabilities={"close_ticket": 1.0, "open_ticket": 0.0, RESPOND: 0.0, OTHER: 0.0},
        action=draft,
        response=response,
    )
    main = _Model((_response(AssistantAction(content="main")),))
    cascade = CascadeModelClient(
        main=main,
        decider=Decider("sampled_drafts", [decision]),
        policy=CascadePolicy(decider=small, threshold=1.0, never_delegate=("open_ticket",)),
    )
    result = cascade.complete(request("close_ticket", "open_ticket"))
    assert result.output.content == "main"
    assert [(row.decided_by, row.note) for row in cascade.records] == [("main", "invalid_draft")]


def test_caller_forced_requests_and_foreign_deciders_are_not_reinterpreted() -> None:
    """A caller's own tool choice is never overridden, and only the bound decider is accepted."""
    main = _Model((_response(AssistantAction(content="done")),))
    decider = Decider("systemone", [])
    cascade = CascadeModelClient(
        main=main, decider=decider, policy=CascadePolicy(decider=decider.definition, threshold=0)
    )
    forced = request("close_ticket").model_copy(update={"tool_choice": "none"})
    cascade.complete(forced)
    assert not decider.seen and cascade.records[0].note == "caller_tool_choice"
    with pytest.raises(ValueError, match="frozen decider"):
        CascadeModelClient(
            main=main,
            decider=decider,
            policy=CascadePolicy(decider=definition("sampled_drafts"), threshold=0.5),
        )


def test_response_fixture_identity_is_the_main_model() -> None:
    """The fixture main model differs from both decider identities."""
    main: ModelResponse = _response(AssistantAction(content="x"))
    assert main.model not in (definition().model, definition("sampled_drafts").model)
