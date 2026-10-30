"""Joint replay validation of induced programs and the induced ownership partition."""

import pytest

from exp.runtime.environments.world_contracts import StateDomain, ToolEffectPolicy, WorldManifest
from exp.simulation.induction.contracts import InductionEpisode, InductionStep, TransitionProgram
from exp.simulation.induction.sandbox import ProgramSandbox
from exp.simulation.induction.tests.fixtures_test import (
    RECORD_FIT_CALLS,
    RECORD_PROGRAMS,
    WORKBOOK_FIT_CALLS,
    WORKBOOK_PROGRAMS,
    programs,
    record_episode,
    record_manifest,
    workbook_episode,
    workbook_manifest,
)
from exp.simulation.induction.validation import induced_partition, validate_programs


@pytest.fixture(scope="module")
def sandbox():  # noqa: ANN201 - pytest fixture generator
    """Share one worker across this module's replays."""
    with ProgramSandbox() as shared:
        yield shared


def test_correct_programs_reproduce_native_record_world_replies_and_state(
    sandbox: ProgramSandbox,
) -> None:
    """Every ticket tool is accepted; the unproposed note tool leaves its domain predicted."""
    episode = record_episode("fit-1", RECORD_FIT_CALLS)
    report = validate_programs(
        programs(RECORD_PROGRAMS), (episode,), record_manifest(), sandbox, minimum_support=1
    )
    verdicts = {row.tool: row for row in report.programs}
    assert {tool: row.status for tool, row in verdicts.items()} == dict.fromkeys(
        RECORD_PROGRAMS, "accepted"
    )
    assert verdicts["get_ticket"].exact == 3 and not verdicts["get_ticket"].failures
    assert report.coverage.induced_tools == tuple(sorted(RECORD_PROGRAMS))
    assert report.coverage.induced_domains == ("tickets",)
    assert report.rounds == 1 and report.steps == len(RECORD_FIT_CALLS)
    assert ("id",) in report.coverage.domains["get_ticket"].argument_keys


def test_a_wrong_reply_rejects_only_that_program_with_its_failing_step(
    sandbox: ProgramSandbox,
) -> None:
    """The failing check quotes the expected and actual replies."""
    sources = dict(RECORD_PROGRAMS)
    sources["count_open"] = (
        "def transition(state, args, facts):\n    return {'open': len(state['tickets']['rows'])}\n"
    )
    report = validate_programs(
        programs(sources), (record_episode("fit-1", RECORD_FIT_CALLS),), record_manifest(), sandbox
    )
    verdicts = {row.tool: row for row in report.programs}
    assert verdicts["count_open"].status == "rejected"
    failure = verdicts["count_open"].failures[0]
    assert failure.status == "reply_mismatch"
    assert failure.expected == {"open": 1} and failure.actual == {"open": 2}
    assert "count_open" not in report.coverage.induced_tools
    assert report.coverage.induced_tools == ()


def test_recorded_state_catches_a_reply_exact_program_with_wrong_state(
    sandbox: ProgramSandbox,
) -> None:
    """A program that replies correctly but writes the wrong row is rejected on state."""
    sources = dict(RECORD_PROGRAMS)
    sources["close_ticket"] = sources["close_ticket"].replace(
        'row[2] = "closed"', 'row[2] = "done"'
    )
    report = validate_programs(
        programs(sources), (record_episode("fit-1", RECORD_FIT_CALLS),), record_manifest(), sandbox
    )
    verdict = next(row for row in report.programs if row.tool == "close_ticket")
    assert verdict.status == "rejected" and verdict.failures[0].status == "state_mismatch"


def test_without_state_knowledge_reads_are_unverifiable_until_recorded_state_grounds_them(
    sandbox: ProgramSandbox,
) -> None:
    """Without a starting state nothing reads known state, so nothing can be accepted."""
    episode = record_episode("fit-1", RECORD_FIT_CALLS, with_state=False)
    report = validate_programs(programs(RECORD_PROGRAMS), (episode,), record_manifest(), sandbox)
    verdicts = {row.tool: row for row in report.programs}
    assert verdicts["get_ticket"].unverifiable == 3 and verdicts["get_ticket"].evaluable == 0
    assert verdicts["get_ticket"].status == "rejected"
    grounded = episode.model_copy(
        update={
            "steps": (
                episode.steps[0].model_copy(
                    update={"state_after": record_episode("x", RECORD_FIT_CALLS[:1]).initial_state}
                ),
                *episode.steps[1:],
            )
        }
    )
    regrounded = validate_programs(
        programs(RECORD_PROGRAMS), (grounded,), record_manifest(), sandbox
    )
    assert {row.status for row in regrounded.programs} == {"accepted"}


def test_a_writer_rejection_makes_its_readers_unverifiable_in_the_next_round(
    sandbox: ProgramSandbox,
) -> None:
    """Joint validation reruns without the rejected writer instead of trusting its writes."""
    sources = dict(RECORD_PROGRAMS)
    sources["open_ticket"] = sources["open_ticket"].replace('"status": "open"}', '"state": "open"}')
    episode = record_episode("fit-1", RECORD_FIT_CALLS)
    report = validate_programs(programs(sources), (episode,), record_manifest(), sandbox)
    verdicts = {row.tool: row for row in report.programs}
    assert verdicts["open_ticket"].status == "rejected"
    assert verdicts["get_ticket"].status == "accepted"
    assert verdicts["get_ticket"].unverifiable == 0
    # open_ticket is rejected, so tickets stays predicted and nothing on it is owned.
    assert report.coverage.induced_tools == ()


def test_programs_writing_undeclared_domains_are_rejected(sandbox: ProgramSandbox) -> None:
    """A read-only tool that edits its domain is refused even when its reply is exact."""
    sources = dict(RECORD_PROGRAMS)
    sources["count_open"] = (
        "def transition(state, args, facts):\n"
        "    state['tickets']['rows'] = []\n"
        "    return {'open': 1}\n"
    )
    episode = record_episode("fit-1", RECORD_FIT_CALLS[:1])
    report = validate_programs(programs(sources), (episode,), record_manifest(), sandbox)
    verdict = next(row for row in report.programs if row.tool == "count_open")
    assert verdict.failures[0].status == "undeclared_write"


def test_workbook_programs_are_validated_against_the_workbook_state_schema(
    sandbox: ProgramSandbox,
) -> None:
    """The workbook-like tool is fully induced; a schema-invalid write is refused."""
    episode = workbook_episode("book-1", WORKBOOK_FIT_CALLS)
    report = validate_programs(
        programs(WORKBOOK_PROGRAMS), (episode,), workbook_manifest(), sandbox
    )
    assert {row.status for row in report.programs} == {"accepted"}
    assert report.coverage.induced_domains == ("book.xlsx",)
    sources = dict(WORKBOOK_PROGRAMS)
    sources["write_cell"] = sources["write_cell"].replace('"data_type": kind', '"data_type": 1')
    broken = validate_programs(programs(sources), (episode,), workbook_manifest(), sandbox)
    verdict = next(row for row in broken.programs if row.tool == "write_cell")
    assert verdict.failures[0].status == "invalid_state"


def test_minimum_support_and_unsupported_tools_are_explicit(sandbox: ProgramSandbox) -> None:
    """Too little evidence rejects; a tool on native state is never induced."""
    manifest = WorldManifest(
        domains=(
            StateDomain(name="files", owner="native"),
            StateDomain(name="memo", owner="simulated"),
        ),
        tools=(
            ToolEffectPolicy(name="native_read", reads=("files",)),
            ToolEffectPolicy(name="memo", reads=("memo",), writes=("memo",)),
        ),
    )
    source = "def transition(state, args, facts):\n    return 'ok'\n"
    episode = InductionEpisode(
        episode_id="e",
        initial_state={"memo": {}},
        steps=(
            InductionStep(tool="memo", arguments={}, reply="ok"),
            InductionStep(tool="native_read", arguments={}, reply="ok"),
        ),
    )
    report = validate_programs(
        (
            TransitionProgram(tool="memo", source=source),
            TransitionProgram(tool="native_read", source=source),
        ),
        (episode,),
        manifest,
        sandbox,
        minimum_support=2,
    )
    verdicts = {row.tool: row for row in report.programs}
    assert verdicts["native_read"].status == "unsupported"
    assert verdicts["memo"].status == "rejected" and "at least 2" in str(verdicts["memo"].reason)
    with pytest.raises(ValueError, match="one program per tool"):
        validate_programs(
            (TransitionProgram(tool="memo", source=source),) * 2, (episode,), manifest, sandbox
        )


def test_the_partition_never_lets_a_tool_span_induced_and_predicted_domains() -> None:
    """An accepted tool sharing a domain with a predicted tool is left to the predictor."""
    manifest = WorldManifest(
        domains=tuple(StateDomain(name=name, owner="simulated") for name in ("a", "b", "c")),
        tools=(
            ToolEffectPolicy(name="read_a", reads=("a",)),
            ToolEffectPolicy(name="move", reads=("a",), writes=("b",)),
            ToolEffectPolicy(name="write_b", reads=("b",), writes=("b",)),
            ToolEffectPolicy(name="write_c", writes=("c",)),
        ),
    )
    tools, domains = induced_partition(manifest, ("read_a", "move", "write_c"))
    assert tools == {"write_c"} and domains == {"c"}
    tools, domains = induced_partition(manifest, ("read_a", "move", "write_b", "write_c"))
    assert tools == {"read_a", "move", "write_b", "write_c"} and domains == {"a", "b", "c"}
