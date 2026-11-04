"""Offline tests for the world author over the grounded ticket fixture and scripted replies."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.optimize.evaluation.variance.authoring_packet import AuthoringPacket
from exp.optimize.evaluation.variance.authoring_packet_test import ticket_packet, tool_reply, wire
from exp.optimize.evaluation.variance.tests.grounded_world_test import INITIAL, grounded_fixture
from exp.optimize.evaluation.variance.world_author import (
    WORLD_AUTHOR_PROMPT,
    WORLD_TOOL,
    author_world,
    world_protocol_sha256,
    world_request,
    world_runtime_definition,
)
from exp.simulation.induction.sandbox import ProgramSandbox

CLOSE = """def transition(state, args, facts):
    ticket = state["tickets"]
    if ticket.get("lock") != "none":
        raise ToolError({"error": "ticket is held by another agent"})
    ticket["status"] = "closed"
    ticket["outcome"] = "right" if ticket["severity"] == "low" else "wrong"
    return {"status": "closed"}
"""
"""A close program that refuses a held or unknown lock."""

CARELESS_CLOSE = """def transition(state, args, facts):
    ticket = state["tickets"]
    if ticket.get("lock") == "held":
        raise ToolError({"error": "ticket is held by another agent"})
    ticket["status"] = "closed"
    ticket["outcome"] = "right" if ticket["severity"] == "low" else "wrong"
    return {"status": "closed"}
"""
"""A close program that silently closes a ticket whose lock state is missing."""

READ = """def transition(state, args, facts):
    return {"ticket": "T-1", "status": state["tickets"]["status"]}
"""
"""A read program showing only the handling status."""

CHECK = """def transition(state, args, facts):
    ticket = state["tickets"]
    if ticket["lock"] == "held":
        raise ToolError({"error": "ticket is held by another agent"})
    return {"severity": ticket["severity"], "lock": ticket["lock"]}
"""
"""An SLA program showing the tier and lock, failing on a held ticket."""

ESCALATE = """def transition(state, args, facts):
    ticket = state["tickets"]
    ticket["status"] = "escalated"
    ticket["outcome"] = "right" if ticket["severity"] == "high" else "wrong"
    return {"status": "escalated"}
"""
"""An escalation program resolving a high tier ticket."""


def ticket(**update: JsonValue) -> JsonObject:
    """The fixture ticket domain with fields replaced."""
    base = INITIAL["tickets"]
    assert isinstance(base, dict)
    return {**base, **update}


def world_answer(
    packet: AuthoringPacket,
    *,
    close: str = CLOSE,
    bindings: tuple[str, ...] = ("tickets.tier", "tickets.lock"),
    **update: JsonValue,
) -> JsonObject:
    """A ready world answer for the ticket packet, with world fields optionally replaced.

    Args:
        packet: The packet the answer is for.
        close: Source of the close program.
        bindings: Rules the world binds.
        **update: World fields to replace.

    Returns:
        The forced tool arguments.
    """
    tools: list[JsonValue] = [
        {
            "name": name,
            "input_schema": {"type": "object"},
            "reads": ["tickets"],
            "writes": ["tickets"] if name in ("escalate", "close") else [],
            "preconditions": "close needs the lock to be none" if name == "close" else "none",
            "effects": "changes handling status" if name in ("escalate", "close") else "none",
            "errors": "a held ticket is refused" if name in ("close", "check_sla") else "none",
            "observations": "the handling status or tier",
            "supported_by": [f"tool:{name}", "rule:tickets.lock"],
        }
        for name in ("read_ticket", "check_sla", "escalate", "close")
    ]
    world: JsonObject = {
        "domains": [{"name": "tickets", "owner": "simulated"}],
        "tools": tools,
        "rule_bindings": [
            {
                "rule_id": rule_id,
                "use": "implemented" if rule_id == "tickets.tier" else "enforced",
                "note": "bound",
            }
            for rule_id in bindings
        ],
        "initializations": [
            {"variant_id": variant.variant_id, "mode": "resume", "state": variant.state}
            for variant in packet.variants
        ],
        "probe_paths": {"tickets": "/tickets"},
        "evidence_hooks": [
            {"kind": "assistant_message", "owner": "harness", "detail": "every message"},
            {"kind": "tool_call", "owner": "harness", "detail": "name and arguments"},
            {"kind": "tool_result", "owner": "world", "detail": "reply"},
            {"kind": "tool_error", "owner": "world", "detail": "error reply"},
            {
                "kind": "state_snapshot",
                "owner": "world",
                "detail": "handling status",
                "state_path": "/tickets/status",
            },
        ],
        "programs": [
            {"tool": "read_ticket", "source": READ},
            {"tool": "check_sla", "source": CHECK},
            {"tool": "escalate", "source": ESCALATE},
            {"tool": "close", "source": close},
        ],
        "fixtures": [
            {
                "episode_id": "fixture-held",
                "initial_state": {"tickets": ticket(severity="low", lock="held")},
                "steps": [
                    {
                        "tool": "check_sla",
                        "arguments": {},
                        "reply": {"error": "ticket is held by another agent"},
                        "is_error": True,
                    },
                    {
                        "tool": "close",
                        "arguments": {},
                        "reply": {"error": "ticket is held by another agent"},
                        "is_error": True,
                    },
                ],
            },
            {
                "episode_id": "fixture-escalate",
                "initial_state": {"tickets": ticket(severity="high")},
                "steps": [
                    {
                        "tool": "read_ticket",
                        "arguments": {},
                        "reply": {"ticket": "T-1", "status": "open"},
                    },
                    {
                        "tool": "check_sla",
                        "arguments": {},
                        "reply": {"severity": "high", "lock": "none"},
                    },
                    {
                        "tool": "escalate",
                        "arguments": {},
                        "reply": {"status": "escalated"},
                        "state_after": {
                            "tickets": ticket(severity="high", status="escalated", outcome="right")
                        },
                    },
                ],
            },
        ],
    }
    world.update(update)
    return {"status": "ready", "world": world, "limitations": ["Escalation of held tickets."]}


def research_answer(*cited: str) -> JsonObject:
    """A needs_research world answer citing the given locators."""
    return {
        "status": "needs_research",
        "research": [
            {
                "question": "Does escalating a held ticket succeed?",
                "why_it_matters": "The urgent variant depends on it.",
                "blocks": ["urgent"],
                "cited": list(cited),
            }
        ],
        "limitations": ["Escalation of a held ticket is unobserved."],
    }


def test_the_request_is_the_packet_alone_with_the_runtime_abi() -> None:
    """The prompt pins the transition ABI and the request carries only the packet."""
    for phrase in (
        "transition(state, args, facts)",
        "ToolError(reply)",
        "OutOfDomain(reason)",
        "Never\n  return a ProgramOutcome",
        "needs_research",
        "Do not\nimplement task success",
    ):
        assert phrase in WORLD_AUTHOR_PROMPT
    assert len(world_protocol_sha256()) == 64


def test_a_ready_world_is_validated_saved_replayed_and_bound_to_the_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Accepted programs and coherent initializations make a ready world bound per variant."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    packet, reference = ticket_packet(fixture)
    transport = wire(monkeypatch, tool_reply(WORLD_TOOL, world_answer(packet)))

    with ProgramSandbox() as sandbox:
        draft = author_world(
            fixture.project,
            fixture.settings,
            reference,
            sandbox=sandbox,
            identity="world-ok",
            code_revision="test",
        )
        replay = author_world(
            fixture.project,
            fixture.settings,
            reference,
            sandbox=sandbox,
            identity="world-ok",
            code_revision="test",
        )

    assert draft.status == "ready", draft.failure
    assert replay == draft and len(transport.requests) == 1
    assert draft.validation is not None
    assert {row.status for row in draft.validation.programs} == {"accepted"}
    assert draft.world is not None
    definition = world_runtime_definition(packet, draft.world, "routine", base=fixture.definition)
    assert definition.initial_state == packet.variant("routine").state
    assert definition.rule_book == packet.rule_book
    assert definition.task == packet.task


def test_needs_research_carries_no_world_and_invented_citations_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A blocked answer has research and no world; one citing an invented source is refused."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    _, reference = ticket_packet(fixture)
    wire(
        monkeypatch,
        tool_reply(WORLD_TOOL, research_answer("question:0", "rule:tickets.lock")),
        tool_reply(WORLD_TOOL, research_answer("trace:invented-9")),
    )
    with ProgramSandbox() as sandbox:
        blocked = author_world(
            fixture.project,
            fixture.settings,
            reference,
            sandbox=sandbox,
            identity="world-research",
            code_revision="test",
        )
        invented = author_world(
            fixture.project,
            fixture.settings,
            reference,
            sandbox=sandbox,
            identity="world-invented",
            code_revision="test",
        )

    assert blocked.status == "needs_research" and blocked.world is None
    assert blocked.response is not None and blocked.research[0].blocks == ("urgent",)
    assert invented.status == "refused" and invented.world is None
    assert invented.failure is not None and "trace:invented-9" in invented.failure


@pytest.mark.parametrize(
    ("close", "message"),
    [
        ("import os\n" + CLOSE, "refused"),
        (CLOSE.replace('"error": "ticket is held', '"error": "locked'), "rejected"),
    ],
    ids=["outside-subset", "wrong-reply"],
)
def test_an_invalid_program_is_refused_by_induction_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, close: str, message: str
) -> None:
    """A program outside the language subset, or one that misreplays a fixture, is refused."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    packet, reference = ticket_packet(fixture)
    wire(monkeypatch, tool_reply(WORLD_TOOL, world_answer(packet, close=close)))
    with ProgramSandbox() as sandbox:
        draft = author_world(
            fixture.project,
            fixture.settings,
            reference,
            sandbox=sandbox,
            identity="world-bad-program",
            code_revision="test",
        )

    assert draft.status == "refused" and draft.world is None
    assert draft.failure is not None and "induction validation" in draft.failure
    assert message in draft.failure


def test_incoherent_worlds_are_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Invented tools, edited resume states, absent probes and misowned hooks are refused."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    packet, reference = ticket_packet(fixture)
    good = world_answer(packet)
    world = good["world"]
    assert isinstance(world, dict)
    tools = world["tools"]
    assert isinstance(tools, list)
    cases: tuple[tuple[JsonObject, str], ...] = (
        (world_answer(packet, tools=[*tools, {**tools[0], "name": "refund"}]), "task tools"),
        (
            world_answer(
                packet,
                initializations=[
                    {"variant_id": v.variant_id, "mode": "resume", "state": INITIAL}
                    for v in packet.variants
                ],
            ),
            "differs from its variant state",
        ),
        (world_answer(packet, probe_paths={"x": "/tickets/absent"}), "absent"),
        (
            world_answer(
                packet,
                evidence_hooks=[{"kind": "assistant_message", "owner": "world", "detail": "x"}],
            ),
            "recorded by the harness",
        ),
    )
    wire(monkeypatch, *(tool_reply(WORLD_TOOL, answer) for answer, _ in cases))
    with ProgramSandbox() as sandbox:
        for index, (_, message) in enumerate(cases):
            draft = author_world(
                fixture.project,
                fixture.settings,
                reference,
                sandbox=sandbox,
                identity=f"world-incoherent-{index}",
                code_revision="test",
            )
            assert draft.status == "refused", message
            assert draft.failure is not None and message in draft.failure
    request = world_request(packet, output_tokens=fixture.settings.output_tokens)
    assert '"rule:tickets.lock"' in str(request.messages[1].content)
