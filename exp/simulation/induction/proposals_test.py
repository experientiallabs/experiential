"""The provider-free proposal loop: requests, retained proposals, feedback and re-asks."""

import json

from exp.common.models import AssistantAction, ModelFinishReason, ModelResponse, OperationEconomics
from exp.simulation.induction.contracts import ProgramProposal, ValidationReport
from exp.simulation.induction.proposals import (
    InductionSettings,
    induce_programs,
    inducible_tools,
    parse_proposal,
)
from exp.simulation.induction.sandbox import ProgramSandbox
from exp.simulation.induction.tests.fixtures_test import (
    RECORD_FIT_CALLS,
    RECORD_PROGRAMS,
    RECORD_TOOLS,
    SNAPSHOT,
    ScriptedInducer,
    record_episode,
    record_manifest,
    task,
)


def test_a_rejected_program_is_reasked_with_its_failing_steps_until_accepted() -> None:
    """The second count_open attempt sees the quoted failure and is accepted."""
    wrong = "def transition(state, args, facts):\n    return {'open': 0}\n"
    model = ScriptedInducer(
        {
            **{tool: [source] for tool, source in RECORD_PROGRAMS.items()},
            "count_open": [wrong, RECORD_PROGRAMS["count_open"]],
            "add_note": [None, None],
        }
    )
    records: list[ProgramProposal | ValidationReport] = []
    with ProgramSandbox() as sandbox:
        report = induce_programs(
            manifest=record_manifest(),
            tools=task(RECORD_TOOLS).tools,
            episodes=(record_episode("fit-1", RECORD_FIT_CALLS),),
            draw=model,
            record=records.append,
            sandbox=sandbox,
            settings=InductionSettings(maximum_attempts=2, minimum_support=1),
        )
    assert report.coverage.induced_tools == tuple(sorted(RECORD_PROGRAMS))
    assert [row.tool for row in records if isinstance(row, ProgramProposal)].count(
        "count_open"
    ) == 2
    assert sum(isinstance(row, ValidationReport) for row in records) == 2
    reask = next(
        request for operation, request in model.requests if operation == "tool-2-attempt-1"
    )
    assert "not accepted" in str(reask.messages[-1].content)
    assert '"expected": {"open": 1}' in str(reask.messages[-1].content)
    refused = [
        row for row in records if isinstance(row, ProgramProposal) and row.tool == "add_note"
    ]
    assert len(refused) == 2 and all(row.program is None for row in refused)
    assert "refused" in str(
        next(request for operation, request in model.requests if operation == "tool-0-attempt-1")
        .messages[-1]
        .content
    )


def test_requests_carry_fit_evidence_and_effects_but_no_other_state() -> None:
    """The request frames one tool with its declared effects and the recorded episodes."""
    model = ScriptedInducer({tool: [RECORD_PROGRAMS.get(tool)] for tool in RECORD_TOOLS})
    with ProgramSandbox() as sandbox:
        induce_programs(
            manifest=record_manifest(),
            tools=task(RECORD_TOOLS).tools,
            episodes=(record_episode("fit-1", RECORD_FIT_CALLS),),
            draw=model,
            record=lambda row: None,
            sandbox=sandbox,
            settings=InductionSettings(maximum_attempts=1, minimum_support=1),
        )
    operation, request = model.requests[2]
    context = json.loads(str(request.messages[1].content))
    assert operation == "tool-2-attempt-0" and context["tool"] == "count_open"
    assert context["reads"] == ["tickets"] and context["writes"] == []
    assert context["episodes"][0]["calls"][0]["reply"] == {"open": 1}
    assert request.tool_choice is not None and len(model.requests) == len(RECORD_TOOLS)


def test_inducible_tools_need_fit_steps_and_simulated_domains() -> None:
    """Tools never called in fit evidence are not proposed."""
    episode = record_episode("fit-1", RECORD_FIT_CALLS[:2])
    assert inducible_tools(record_manifest(), (episode,)) == ("count_open", "open_ticket")
    assert InductionSettings(maximum_attempts=3).draws(4) == 12


def test_incomplete_or_untyped_responses_are_recorded_refusals() -> None:
    """A truncated response or one without the typed submission produces no program."""
    truncated = ModelResponse(
        output=AssistantAction(content="partial"),
        model=SNAPSHOT,
        economics=OperationEconomics(),
        finish_reason=ModelFinishReason.LENGTH,
    )
    assert "stopped" in str(parse_proposal(truncated, "t", 0, "op").refusal)
    untyped = truncated.model_copy(update={"finish_reason": ModelFinishReason.COMPLETED})
    assert parse_proposal(untyped, "t", 0, "op").program is None
