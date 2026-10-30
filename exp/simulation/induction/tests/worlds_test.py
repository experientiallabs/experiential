"""End to end on two world kinds: induce from fit evidence, then run held-out episodes.

A scripted model proposes programs; validation accepts them only on exact fit replay; the
induced backend then serves held-out calls in the real world kernel, deferring to the grounded
predictor outside the validated domain. Held-out replies are compared with the native record
world and the reference workbook.
"""

from exp.common.core.artifacts import JsonObject
from exp.common.models import ToolCall
from exp.runtime.environments.worlds import ExecutableWorldRuntime
from exp.simulation.induction.backend import InducedWorldFactory
from exp.simulation.induction.contracts import InductionEpisode
from exp.simulation.induction.proposals import InductionSettings, induce_programs
from exp.simulation.induction.sandbox import ProgramSandbox
from exp.simulation.induction.tests.fixtures_test import (
    RECORD_FIT_CALLS,
    RECORD_PROGRAMS,
    RECORD_TOOLS,
    WORKBOOK_FIT_CALLS,
    WORKBOOK_PROGRAMS,
    WORKBOOK_TOOLS,
    ScriptedInducer,
    decoded,
    grounded_fallback,
    record_episode,
    record_manifest,
    task,
    workbook_episode,
    workbook_manifest,
    workbook_state,
)


def _replay(
    runtime: ExecutableWorldRuntime, tools: tuple[str, ...], episode: InductionEpisode
) -> list[bool]:
    """Replay one held-out episode and return per-step exactness against its recorded replies."""
    exact = []
    with runtime.open(task(tools)) as world:
        for index, step in enumerate(episode.steps):
            observation = world.execute(
                ToolCall(name=step.tool, call_id=f"s{index}", arguments=step.arguments)
            )
            exact.append(
                (decoded(observation.content), observation.is_error) == (step.reply, step.is_error)
            )
    return exact


def test_record_world_tools_are_induced_and_held_out_episodes_replay_exactly() -> None:
    """The ticket service is owned by induced code; the note tool stays predicted."""
    fit = (
        record_episode("fit-1", RECORD_FIT_CALLS),
        record_episode("fit-2", RECORD_FIT_CALLS[3:] + (("close_ticket", {"id": 1}),)),
    )
    model = ScriptedInducer({**{t: [s] for t, s in RECORD_PROGRAMS.items()}, "add_note": [None]})
    with ProgramSandbox() as sandbox:
        report = induce_programs(
            manifest=record_manifest(),
            tools=task(RECORD_TOOLS).tools,
            episodes=fit,
            draw=model,
            record=lambda row: None,
            sandbox=sandbox,
            settings=InductionSettings(maximum_attempts=1),
        )
        assert report.coverage.induced_domains == ("tickets",)
        held_out = record_episode(
            "held-out",
            (
                ("open_ticket", {"title": "laptop"}),
                ("close_ticket", {"id": 2}),
                ("close_ticket", {"id": 2}),
                ("get_ticket", {"id": 2}),
                ("count_open", {}),
            ),
        )
        initial: JsonObject = dict(fit[0].initial_state)
        fallback, _, requests = grounded_fallback(
            record_manifest(), task(RECORD_TOOLS), initial, []
        )
        runtime = ExecutableWorldRuntime(
            record_manifest(),
            InducedWorldFactory(report.coverage, fallback, sandbox),
            owner="simulated",
        )
        assert _replay(runtime, RECORD_TOOLS, held_out) == [True] * 5
    assert requests == []


def test_workbook_tools_are_induced_and_unknown_sheets_defer_to_the_predictor() -> None:
    """A workbook write path seen in fit is exact; an unknown sheet is predicted."""
    fit = (workbook_episode("book-1", WORKBOOK_FIT_CALLS),)
    model = ScriptedInducer({t: [s] for t, s in WORKBOOK_PROGRAMS.items()})
    with ProgramSandbox() as sandbox:
        report = induce_programs(
            manifest=workbook_manifest(),
            tools=task(WORKBOOK_TOOLS).tools,
            episodes=fit,
            draw=model,
            record=lambda row: None,
            sandbox=sandbox,
            settings=InductionSettings(maximum_attempts=1, minimum_support=1),
        )
        assert report.coverage.induced_tools == tuple(sorted(WORKBOOK_TOOLS))
        held_out = workbook_episode(
            "book-held-out",
            (
                ("write_cell", {"sheet": "Sheet1", "cell": "B5", "value": 10}),
                ("sum_column", {"sheet": "Sheet1", "column": "B"}),
                ("read_cell", {"sheet": "Sheet1", "cell": "B5"}),
                ("read_cell", {"sheet": "Other", "cell": "A1"}),
            ),
        )
        fallback, _, requests = grounded_fallback(
            workbook_manifest(),
            task(WORKBOOK_TOOLS),
            workbook_state(),
            [("s3", {"error": "no sheet Other"}, [])],
        )
        runtime = ExecutableWorldRuntime(
            workbook_manifest(),
            InducedWorldFactory(report.coverage, fallback, sandbox),
            owner="simulated",
        )
        # The predictor answers the unknown sheet with a non-error reply: only that step differs.
        assert _replay(runtime, WORKBOOK_TOOLS, held_out) == [True, True, True, False]
    assert len(requests) == 1
