"""The Astra refinement with ``[models.astra]`` on a ``codex-cli`` connection, scripted Codex.

The catalog points the ``astra`` alias at ``provider = "codex-cli"``; nothing at a call site
names Codex. ``codex exec`` is a scripted runner that reads the prompt from stdin, answers as the
fixture's Astra, writes the answer file and an event stream. The runtime model and the reviewer
keep their scripted HTTP wire. Everything else is the real path: the registry's codex-cli client,
the admitted transport's attempt receipts, the draw machine, the loop and replay.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

import pytest

from exp.common.models import (
    ConnectionConfig,
    ModelCapabilities,
    ModelRecord,
    load_model_catalog,
    write_model_catalog,
)
from exp.common.models.model import BillingSource
from exp.common.project import ProjectStore
from exp.optimize.evaluation.variance.astra_prompt_attempts import TransientDraw
from exp.optimize.evaluation.variance.astra_prompt_contracts import RefinementSettings
from exp.optimize.evaluation.variance.astra_prompt_loop import refine_prompt
from exp.optimize.evaluation.variance.frontdoor_ledger import PhaseLedger
from exp.optimize.evaluation.variance.prompt_targets import target
from exp.optimize.evaluation.variance.refined_prompts import journal_of
from exp.optimize.evaluation.variance.rubric_evolution_test import settings
from exp.optimize.evaluation.variance.tests.astra_prompt_test import (
    FIX,
    STANCE,
    Wire,
    calibration,
    script,
    write_astra_catalog,
)
from exp.runtime.models.providers import codex_cli
from exp.runtime.models.providers.codex_cli import CodexRun

CLEAN = [
    {"type": "thread.started"},
    {"type": "turn.started"},
    {"type": "item.completed", "item": {"type": "agent_message", "text": "{}"}},
    {"type": "turn.completed", "usage": {"input_tokens": 14000, "output_tokens": 40}},
]


class Codex:
    """The scripted ``codex exec``: Astra's fixture answers, or scripted faults first.

    Attributes:
        calls: Every argv.
        faults: Runs (or exceptions) returned before answering, in order.
    """

    def __init__(self, faults: Sequence[object] = ()) -> None:
        """Start with the given faults."""
        self.calls: list[list[str]] = []
        self.faults: list[object] = list(faults)
        self.answer = script(lambda content: f"{content['current_text']}\n{FIX}")

    def __call__(self, argv: Sequence[str], stdin: str, cwd: Path, timeout: float) -> CodexRun:
        """Answer one call from the prompt on stdin."""
        self.calls.append(list(argv))
        assert list(cwd.iterdir()) == [] and argv[argv.index("-m") + 1] == "gpt-6-astra"
        if self.faults:
            fault = self.faults.pop(0)
            if isinstance(fault, BaseException):
                raise fault
            assert isinstance(fault, CodexRun)
            return fault
        system = re.search(r"<system>\n(.*?)\n</system>", stdin, re.S)
        user = re.search(r"<user>\n(.*?)\n</user>", stdin, re.S)
        tool = re.search(r"the arguments of `([a-z_]+)`", stdin)
        assert system and user and tool
        content = json.loads(user.group(1))
        answer = self.answer("gpt-6-astra", tool.group(1), system.group(1), content)
        Path(argv[argv.index("-o") + 1]).write_text(json.dumps(answer))
        return CodexRun(0, "\n".join(json.dumps(e) for e in CLEAN), "", None)


def codex_world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, codex: Codex
) -> tuple[ProjectStore, RefinementSettings, Wire]:
    """The Astra catalog with ``astra`` moved onto a codex-cli connection."""
    project = ProjectStore(tmp_path, "case")
    write_astra_catalog(tmp_path)
    catalog = load_model_catalog(tmp_path / "models.toml")
    connections = {**catalog.connections, "codex": ConnectionConfig(provider="codex-cli")}
    models = dict(catalog.models)
    models["astra"] = ModelRecord(
        connection="codex",
        model="gpt-6-astra",
        revision="0.166.0",
        billing_source=BillingSource.CUSTOMER_MANAGED,
        capabilities=ModelCapabilities(
            supports_tools=True,
            supports_structured_output=True,
            supports_reasoning=True,
            reasoning_effort="high",
            supports_temperature=False,
            context_window_tokens=1_050_000,
            maximum_output_tokens=128_000,
        ),  # fmt: skip
    )
    write_model_catalog(
        tmp_path / "models.toml",
        catalog.model_copy(update={"connections": connections, "models": models}),
    )
    monkeypatch.setenv("HELD_OUT_TEST_KEY", "fixture-only")
    wire = Wire(script(lambda content: ""))
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: wire
    )
    monkeypatch.setattr(codex_cli, "run_codex", codex)
    monkeypatch.setattr(codex_cli, "codex_version", lambda binary: "0.166.0")
    refinement = RefinementSettings(
        runtime=settings(project, "glm-5.3-flash"),
        astra=settings(project, "astra"),
        reviewer=settings(project, "kimi-k3"),
        astra_alias="astra",
    )
    return project, refinement, wire


def draw_moves(project: ProjectStore) -> list[str]:
    """Accepted draw-machine moves, in order."""
    return [
        str(row["to"])
        for row in journal_of(project).rows()
        if row["machine"] == "astra_prompt_draw" and not row["refused"]
    ]


def test_astra_through_codex_reaches_parity_and_replays_with_no_codex_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every reference, judgment and rewrite goes through Codex; the HTTP wire never sees Astra."""
    codex = Codex()
    project, refinement, wire = codex_world(tmp_path, monkeypatch, codex)
    assert refinement.astra.model.provider == "codex-cli"
    assert refinement.astra.model.revision == "0.166.0"
    found = target(STANCE)
    ledger = PhaseLedger(tmp_path / "astra-ledger.json", unpriced_attempts=20_000)
    record = refine_prompt(
        project, found, calibration(found), refinement, code_revision="t", ledger=ledger
    )
    assert record.state == "target_met" and FIX in record.prompt
    assert record.final is not None and record.final.heldout_parity == 1.0
    assert "gpt-6-astra" not in wire.models()
    calls = len(codex.calls)
    assert calls == 9 + 9 + 1 + 9  # references, baseline judgments, one rewrite, its judgments
    (phase,) = ledger.state.phases
    assert phase.reserved_usd == 0 and phase.measured_attempts is not None
    assert calls + len(wire.requests) <= phase.measured_attempts <= phase.reserved_attempts
    again = refine_prompt(
        project, found, calibration(found), refinement, code_revision="t", ledger=ledger
    )
    assert again == record and len(codex.calls) == calls


@pytest.mark.parametrize(
    ("fault", "kind"),
    [
        (TimeoutError("codex exec exceeded 900s"), "transient_failed"),
        (
            CodexRun(
                0,
                "\n".join(
                    json.dumps(e)
                    for e in [
                        CLEAN[0],
                        {"type": "item.started", "item": {"type": "command_execution"}},
                        *CLEAN[1:],
                    ]
                ),
                "",
                json.dumps({"label": "billing", "escalate": True, "reasons": ["x"]}),
            ),
            "refused",
        ),
    ],
)
def test_a_timeout_or_a_tool_event_leaves_the_draw_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: object, kind: str
) -> None:
    """The failed attempt is a receipt, the run saves nothing, and the rerun completes."""
    codex = Codex([fault])
    project, refinement, _ = codex_world(tmp_path, monkeypatch, codex)
    found = target(STANCE)
    with pytest.raises(TransientDraw) as pending:
        refine_prompt(project, found, calibration(found), refinement, code_revision="t")
    assert pending.value.kind == kind
    assert draw_moves(project) == ["pending", kind]
    receipt = f"{pending.value.identity}-failed-0"
    assert project.artifacts.exists(receipt)
    assert not project.artifacts.exists(f"{pending.value.identity}-response")
    record = refine_prompt(project, found, calibration(found), refinement, code_revision="t")
    assert record.state == "target_met"
    assert draw_moves(project)[:4] == ["pending", kind, "pending", "answered"]
    assert project.artifacts.exists(f"{pending.value.identity}-retry-1-response")
    assert len(codex.calls) == 1 + 28
