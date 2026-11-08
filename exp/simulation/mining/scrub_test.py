"""Intent test: a credential-bearing trace is redacted and kept, not refused at the boundary."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from exp.common.core.artifacts import SourceIdentity
from exp.common.project.request_budget import validate_provider_response
from exp.common.traces import Trace, TraceSource, TraceSpan
from exp.runtime.gateway.decisions_contracts import DecisionRequest, NoulQuestion
from exp.simulation.mining.scrub import scrub_unit
from exp.simulation.mining.views import Unit, observations_view, task_view

_STARTED = datetime(2026, 10, 1, tzinfo=UTC)
_SOURCE = TraceSource(
    identity=SourceIdentity(kind="file", source_id="fixture"),
    semantic_convention_version="1.37.0",
)
# Assembled at runtime so no complete credential-shaped literal sits in the source.
_KEY_BODY = "Fx7QfixtureNotARealKey9a2"
_API_KEY = "sk-" + _KEY_BODY
_PEM_BODY = "MIIEfixtureFAKEpemBODYonly0123456789abcdefWXYZ"
_PEM_KIND = "RSA PRIVATE " + "KEY"
_PRIVATE_KEY = f"-----BEGIN {_PEM_KIND}-----\n{_PEM_BODY}\n-----END {_PEM_KIND}-----"


def _trace(trace_id: str, task: str, tool_output: str) -> Trace:
    spans = (
        TraceSpan(
            span_id="span-0",
            name="agent.step",
            started_at=_STARTED,
            ended_at=_STARTED + timedelta(seconds=1),
            attributes={
                "gen_ai.tool.name": "read_file",
                "gen_ai.tool.call.arguments": json.dumps({"path": "deploy/config"}),
            },
        ),
        TraceSpan(
            span_id="span-1",
            name="agent.step",
            started_at=_STARTED + timedelta(seconds=1),
            ended_at=_STARTED + timedelta(seconds=2),
            attributes={
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": "read_file",
                "gen_ai.tool.message": json.dumps({"content": tool_output}),
            },
        ),
    )
    return Trace(trace_id=trace_id, task=task, spans=spans, source=_SOURCE)


def _passes_boundary(unit: Unit) -> bool:
    """Run the exact credential boundary a decision request is checked against."""
    request = DecisionRequest(
        state=unit.text, questions={"q": NoulQuestion(instructions="Did it deploy?")}
    )
    try:
        validate_provider_response(json.dumps(request.provider_body("fixture"), allow_nan=False))
    except Exception:  # noqa: BLE001 - any boundary failure means refused
        return False
    return True


def _windows(secret: str, width: int = 6) -> set[str]:
    return {secret[start : start + width] for start in range(len(secret) - width + 1)}


def test_secret_bearing_trace_is_redacted_and_kept_and_clean_trace_is_untouched() -> None:
    """Placeholders replace every secret span in every view; a clean view stays byte-identical."""
    leaky = _trace(
        "leaky",
        f"Deploy the service with key {_API_KEY} and check the config.",
        f"region=eu-west-1\n{_PRIVATE_KEY}\nreplicas=3",
    )
    clean = _trace("clean", "Deploy the service and check the config.", "region=eu-west-1")

    for render in (task_view, observations_view):
        raw = render(leaky)
        assert not _passes_boundary(raw), "fixture must exercise the refusal it replaces"
        scrubbed, counts = scrub_unit(raw)
        assert _passes_boundary(scrubbed)
        assert scrubbed == scrub_unit(render(leaky))[0], "the scrub must be deterministic"
        assert "[REDACTED:openai_key:32]" in scrubbed.text
        assert not any(part in scrubbed.text for part in _windows(_KEY_BODY))
        assert scrubbed.meta["redactions"] == dict(counts)
        if render is observations_view:
            assert counts == {"openai_key": 1, "private_key": 1}
            assert "[REDACTED:private_key:" in scrubbed.text
            assert not any(part in scrubbed.text for part in _windows(_PEM_BODY))
            assert "replicas=3" in scrubbed.text

        untouched = render(clean)
        kept, none = scrub_unit(untouched)
        assert not none
        assert kept is untouched
        assert kept.text.encode() == render(clean).text.encode()
