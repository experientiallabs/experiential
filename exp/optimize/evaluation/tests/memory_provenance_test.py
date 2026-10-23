"""Retrieved evidence keeps its provenance when an investigation writes it into memory."""

import json
import re
from collections.abc import Mapping
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.optimize.evaluation.held_out_fidelity_test import (
    SimulatedProvider,
    _arr,
    _at,
    _obj,
    _tool_reply,
    _user_content,
    build_project,
    episode,
)
from exp.optimize.evaluation.trace_learning import learning_plan
from exp.optimize.evaluation.trace_learning_test import memory_delivered
from exp.runtime.models.providers.transport import JsonHttpResponse, RecordedRequest
from exp.simulation.world_model.experience_memory import memory_head

CANARY = re.compile(r"CANARY_user_\d+")
HASH = re.compile(r"[0-9a-f]{20,}")


class CopyingProvider(SimulatedProvider):
    """The first prediction that retrieves a canary copies it, misses, and investigates it.

    The investigator writes the canary into a transferable rule, the way a real model may
    restate a retrieved fact, so memory would carry another task's recorded value.
    """

    def __init__(self, episodes: list[JsonObject]) -> None:
        """Track the copied canary."""
        super().__init__(episodes)
        self.canary: str | None = None

    def _transition(self, payload: JsonObject) -> JsonObject:
        """Copy the first retrieved canary and report a miss so it is investigated."""
        result = super()._transition(payload)
        if self.canary is None:
            body = _user_content(payload)
            found = CANARY.search(json.dumps(body.get("grounded_examples")))
            if found:
                self.canary = found.group()
                _obj(_arr(result["observations"])[0])["reply"] = {"error": "copied " + self.canary}
        return result

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Answer investigations with a rule that restates the canary when it was seen."""
        if url.endswith("/chat/completions"):
            name = str(_at(payload, "tool_choice", "function", "name"))
            schema = _obj(_at(payload, "tools", 0, "function", "parameters", "properties"))
            if name != "submit_transition" and "axes" not in schema and "summary" not in schema:
                self.requests.append(RecordedRequest(url, dict(headers), payload))
                seen = self.canary if self.canary and self.canary in json.dumps(payload) else "-"
                return _tool_reply(
                    "world-v1",
                    name,
                    {
                        "discrepancy": "wrong reply",
                        "hypothesis": f"order reads return {seen}",
                        "scope": "order reads",
                        "uncertainty": "restated from a retrieved example",
                        "supersedes": [],
                        "transferable": True,
                        "observable_cue": "get_order_details",
                    },
                )
        return super().post(url, headers=headers, payload=payload, timeout_seconds=timeout_seconds)


def test_a_retrieved_fact_written_into_memory_never_reaches_its_own_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Memory written after seeing task A's retrieved trace is hidden from A's predictions."""
    episodes = [episode(i) for i in range(4)]
    for item in episodes:
        exchanges = item["exchanges"]
        assert isinstance(exchanges, list)
        exchanges[1][2]["canary"] = f"CANARY_{item['user']}"
    provider = CopyingProvider(episodes)
    build_project(tmp_path, monkeypatch, provider, episodes=episodes, learn=True)
    assert provider.canary is not None, "a prediction retrieved another task's canary"
    owner = provider.canary.removeprefix("CANARY_")
    own = [
        prediction
        for prediction in provider.predictions()
        if _at(prediction, "task", "initial_context", "task_lineage") == owner
    ]
    others = [
        prediction
        for prediction in provider.predictions()
        if _at(prediction, "task", "initial_context", "task_lineage") != owner
        and provider.canary in memory_delivered(prediction)
    ]
    assert own, "the canary's own task was predicted after the rule was learned"
    assert others, "the rule does reach other tasks through memory"
    for prediction in own:
        assert provider.canary not in memory_delivered(prediction)
        assert provider.canary not in json.dumps(prediction["grounded_examples"])


class BitProvider(SimulatedProvider):
    """The first prediction that retrieves the audit bit misses only when the bit is set."""

    def __init__(self, episodes: list[JsonObject]) -> None:
        """Track whether the bit has been seen."""
        super().__init__(episodes)
        self.seen = False

    def _transition(self, payload: JsonObject) -> JsonObject:
        """Let the retrieved bit decide whether this prediction matches."""
        result = super()._transition(payload)
        examples = _user_content(payload).get("grounded_examples")
        if not self.seen and "audit_bit" in json.dumps(examples):
            self.seen = True
            bits = [
                json.loads(str(_at(row, "observation", "content"))).get("audit_bit")
                for row in _arr(examples)
            ]
            if 1 in bits:
                _obj(_arr(result["observations"])[0])["reply"] = {"error": "wrong"}
        return result


def test_a_trace_memory_never_reflects_its_own_recorded_answer_through_another_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flipping one bit in B's answer leaves the memory delivered to B's predictions unchanged.

    Another trace retrieves B's bit and its prediction succeeds or fails on it; nothing that
    outcome writes to memory may reach B.
    """
    delivered = []
    for bit in (0, 1):
        episodes = [episode(i) for i in range(4)]
        exchanges = episodes[2]["exchanges"]
        assert isinstance(exchanges, list)
        _, _, reply = exchanges[1]
        reply["audit_bit"] = bit
        root = tmp_path / f"bit-{bit}"
        root.mkdir()
        provider = BitProvider(episodes)
        _, store, _ = build_project(root, monkeypatch, provider, episodes=episodes, learn=True)
        head = memory_head(store.artifacts, learning_plan(store).lineage)
        assert head is not None and head.version == len(episodes), "every trace was learned"
        assert provider.seen, "another trace retrieved B's recorded answer"
        own = [
            prediction
            for prediction in provider.predictions()
            if _at(prediction, "task", "initial_context", "task_lineage") == "user_02"
        ]
        assert len(own) == 3, "the source task completed all predictions without re-asks"
        assert all(
            _obj(prediction.get("research_history") or {}).get("memory_overview")
            for prediction in own
        ), "the comparison must include memory learned from another trace"
        # Content hashes (lineage and trajectory IDs) differ because B's trace differs.
        delivered.append([HASH.sub("#", memory_delivered(prediction)) for prediction in own])
    assert delivered[0] == delivered[1]
