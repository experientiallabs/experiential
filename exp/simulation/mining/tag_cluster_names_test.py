"""Agentic cluster naming over scripted models, its failures and its replay."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from exp.common.models import AssistantAction, BillingSource, ModelSnapshot
from exp.common.models.model import ModelRequest, ModelResponse
from exp.common.project import ProjectStore
from exp.runtime.models.providers.transport import ProviderTransportError
from exp.simulation.mining.tag_cluster_names import (
    NAMES_TYPE,
    ClusterNamer,
    ClusterNamingSpec,
    NamingPrompt,
    name_clusters,
    named_clusters,
)
from exp.simulation.mining.tag_clusters import ClusteringSpec, TagClustering, cluster_tags
from exp.simulation.mining.tag_clusters_test import planted

SNAPSHOT = ModelSnapshot(
    provider="fixture",
    model_id="namer-fixture",
    billing_source=BillingSource.CUSTOMER_MANAGED,
    capabilities_sha256="a" * 64,
    connection_sha256="b" * 64,
)

Reply = Callable[[ModelRequest], str]


class ScriptedNamer:
    """A completion client answering each naming request from a function, recording requests.

    Attributes:
        requests: Every request, in order.
    """

    def __init__(self, reply: Reply) -> None:
        """Store the reply function."""
        self._reply = reply
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Answer one request."""
        self.requests.append(request)
        return ModelResponse.completed(
            output=AssistantAction(content=self._reply(request)),
            configured_model=SNAPSHOT,
            served_model_id=None,
            usage=None,
            latency_seconds=0.01,
        )


def _prompt(request: ModelRequest) -> str:
    """Return the user prompt of a naming request."""
    content = request.messages[-1].content
    assert isinstance(content, str)
    return content


def _cluster_of(prompt: str) -> str:
    """Read the cluster identity a prompt is about."""
    return prompt.split("CLUSTER ", 1)[1].split(":", 1)[0]


def name_reply(request: ModelRequest) -> str:
    """Draft a name from the stance and cluster; the reconcile picks a final name."""
    prompt = _prompt(request)
    cluster = _cluster_of(prompt)
    if "DRAFTS" in prompt:
        return (
            "```json\n"
            + json.dumps({"name": f"Final {cluster}", "description": f"Episodes of {cluster}."})
            + "\n```"
        )
    stance = prompt.split("STANCE\n", 1)[1].split()[1]
    return json.dumps({"name": f"{stance} {cluster}", "description": "drafted"})


def _clustering() -> TagClustering:
    """The planted three-cluster clustering."""
    matrix, _ = planted()
    return cluster_tags([matrix], ClusteringSpec(max_clusters=6))


QUESTIONS = {f"task:q{i}": f"Does proposition {i} hold?" for i in range(12)}


def test_each_cluster_is_drafted_from_every_variant_then_reconciled() -> None:
    """Three drafts and one reconcile per cluster; the reconcile sees names already given."""
    clustering = _clustering()
    exemplars = {trace: f"episode text of {trace}" for trace in clustering.trace_ids}
    client = ScriptedNamer(name_reply)

    names = name_clusters(
        clustering,
        ClusterNamer(client, SNAPSHOT),
        questions=QUESTIONS,
        exemplar_texts=exemplars,
    )

    assert len(client.requests) == ClusterNamingSpec().calls(3) == 12
    assert [row.name for row in names.names] == ["Final c00", "Final c01", "Final c02"]
    assert all(row.status == "named" for row in names.names)
    first = names.names[0]
    assert [d.variant_id for d in first.drafts] == ["behavior", "task", "contrast", "reconcile"]
    draft_prompt = _prompt(client.requests[0])
    assert "Does proposition 0 hold?" in draft_prompt
    assert "episode text of" in draft_prompt and "NEAREST OTHER CLUSTERS" in draft_prompt
    last_reconcile = _prompt(client.requests[-1])
    assert "- Final c00" in last_reconcile and "- Final c01" in last_reconcile
    assert names.mapping() == {"c00": "Final c00", "c01": "Final c01", "c02": "Final c02"}
    assert names.clustering == clustering.identity()


def test_refused_or_unreadable_calls_are_recorded_and_never_redrawn() -> None:
    """A cluster with no readable draft fails; one bad draft out of several does not."""
    clustering = _clustering()

    def reply(request: ModelRequest) -> str:
        """Refuse everything about c01; answer prose for one c00 variant."""
        prompt = _prompt(request)
        cluster = _cluster_of(prompt)
        if cluster == "c01":
            raise ProviderTransportError("refused", status_code=503)
        if cluster == "c00" and "STANCE\nName the kind" in prompt:
            return "a cluster of episodes"
        return name_reply(request)

    client = ScriptedNamer(reply)
    names = name_clusters(
        clustering, ClusterNamer(client, SNAPSHOT), questions=QUESTIONS, exemplar_texts={}
    )

    c00, c01, c02 = names.names
    assert c00.status == "named" and c00.drafts[1].error is not None
    assert c01.status == "failed" and c01.name is None
    assert all("ProviderTransportError" in (d.error or "") for d in c01.drafts)
    assert len(c01.drafts) == 3
    assert c02.status == "named"
    assert len(client.requests) == 3 + 1 + 3 + 3 + 1
    assert names.mapping() == {"c00": "Final c00", "c02": "Final c02"}


def test_names_are_saved_and_replayed_without_any_call(tmp_path: Path) -> None:
    """A second run over the same clustering, model and evidence reads the saved names."""
    clustering = _clustering()
    store = ProjectStore(tmp_path, "case").artifacts
    spec = ClusterNamingSpec(prompts=(NamingPrompt(variant_id="behavior", stance="Behavior."),))
    names, reference = named_clusters(
        store,
        clustering,
        ClusterNamer(ScriptedNamer(name_reply), SNAPSHOT),
        questions=QUESTIONS,
        exemplar_texts={},
        spec=spec,
        code_revision="test",
    )
    assert store.read(reference.artifact_id).manifest.artifact_type == NAMES_TYPE

    def refuse(request: ModelRequest) -> str:
        """Fail the test: replay must not call the model."""
        raise AssertionError("replay made a naming call")

    replayed, again = named_clusters(
        store,
        clustering,
        ClusterNamer(ScriptedNamer(refuse), SNAPSHOT),
        questions=QUESTIONS,
        exemplar_texts={},
        spec=spec,
        code_revision="test",
    )
    assert replayed == names and again == reference
    with pytest.raises(AssertionError, match="naming call"):
        named_clusters(
            store,
            clustering,
            ClusterNamer(ScriptedNamer(refuse), SNAPSHOT),
            questions={**QUESTIONS, "task:q0": "Reworded?"},
            exemplar_texts={},
            spec=spec,
            code_revision="test",
        )
