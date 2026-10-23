"""Construction protocol preserves supplied interfaces without model copying."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.project import ProjectConfig, ProjectStore
from exp.simulation.reconstruction.models import RawEpisode, RawInputs
from exp.simulation.reconstruction.workflow import reconstruct


def test_supplied_constructor_generates_behavior_not_interface(tmp_path: Path) -> None:
    """The actual public call accepts flat semantics and retains immutable tool metadata."""
    tools: list[JsonObject] = [
        {
            "name": "read",
            "description": "Actual published description.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        }
    ]
    raw = RawInputs(
        mode="supplied",
        episodes=(RawEpisode(id="observed", records=({"action": "read", "reply": None},)),),
        published_contracts={"tools": tools},
    )
    project = ProjectStore(tmp_path, "construct")
    project.initialize(ProjectConfig(project_id="construct"))
    requests: list[JsonObject] = []

    def complete(body: JsonObject, identity: str) -> JsonObject:
        """Return only semantic fields, never a copied tools inventory."""
        requests.append(body)
        return {
            "state_schema": {"type": "object"},
            "initialization_requirements": [],
            "claims": [
                {
                    "statement": "The recorded read returned null.",
                    "status": "observed",
                    "source_refs": [{"source_id": "episode:observed", "pointer": "/0"}],
                    "scope": "This recorded call.",
                    "origin": "evidence",
                }
            ],
            "text_rules": "Preserve uncertainty where supplied observations are insufficient.",
            "python_source": None,
        }

    now = datetime(2030, 1, 1, tzinfo=UTC)
    bundle = reconstruct(
        raw,
        artifacts=project.artifacts,
        run_id="supplied",
        complete=complete,
        clock=lambda: now,
        deadline=now + timedelta(minutes=1),
        code_revision="test",
    )
    assert len(requests) == 1
    messages = requests[0]["messages"]
    assert isinstance(messages, list) and isinstance(messages[1], dict)
    content = messages[1]["content"]
    assert isinstance(content, str)
    payload = json.loads(content)
    assert "inputs" not in payload
    assert payload["sources"]["episode:observed"] == [{"action": "read", "reply": None}]
    assert {"source_id": "episode:observed", "pointer": "/0"} in payload["source_locators"]
    tool = bundle.interface.tools[0]
    assert tool.name == "read"
    assert tool.description == tools[0]["description"]
    assert tool.parameters == tools[0]["parameters"]
    assert tool.reply_schema is None
    assert bundle.sources == raw


def test_supplied_constructor_rejects_extra_generated_tools(tmp_path: Path) -> None:
    """The new protocol rejects an invented interface rather than silently repairing it."""
    raw = RawInputs(
        mode="supplied",
        published_contracts={
            "tools": [{"name": "read", "description": "Read.", "parameters": {"type": "object"}}]
        },
    )
    project = ProjectStore(tmp_path, "construct")
    project.initialize(ProjectConfig(project_id="construct"))
    now = datetime(2030, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError):
        reconstruct(
            raw,
            artifacts=project.artifacts,
            run_id="invalid",
            complete=lambda *_: {"interface": {"tools": []}, "text_rules": "Do nothing."},
            clock=lambda: now,
            deadline=now + timedelta(minutes=1),
            code_revision="test",
        )


@pytest.mark.parametrize("run_id", ["two-stage", "cold-actor-staged"])
def test_staged_supplied_construction_binds_both_responses(tmp_path: Path, run_id: str) -> None:
    """Separate schema generation from claims without repairing malformed responses."""
    raw = RawInputs(
        mode="supplied",
        description="Read a stored value.",
        published_contracts={
            "tools": [{"name": "read", "description": "Read.", "parameters": {"type": "object"}}]
        },
    )
    project = ProjectStore(tmp_path, "staged")
    project.initialize(ProjectConfig(project_id="staged"))
    calls: list[str] = []

    def complete(body: JsonObject, identity: str) -> JsonObject:
        """Produce two deliberately disjoint stage envelopes."""
        calls.append(identity)
        if len(calls) == 1:
            return {"state_schema": {"type": "object"}, "initialization_requirements": []}
        return {
            "claims": [
                {
                    "statement": "Unobserved behavior is unknown.",
                    "status": "unknown",
                    "scope": "read",
                    "origin": "model_prior",
                }
            ],
            "text_rules": "Do not invent values.",
            "python_source": None,
        }

    now = datetime(2030, 1, 1, tzinfo=UTC)
    bundle = reconstruct(
        raw,
        strategy="staged",
        artifacts=project.artifacts,
        run_id=run_id,
        complete=complete,
        clock=lambda: now,
        deadline=now + timedelta(minutes=1),
        code_revision="test",
    )
    assert len(calls) == 2 and calls[0] != calls[1]
    assert bundle.interface.tools[0].name == "read"
    manifest = project.artifacts.read(bundle.artifact_id).manifest
    assert {ref.artifact_id for ref in manifest.inputs} == {
        identity + suffix for identity in calls for suffix in ("-intent", "-response")
    }


def test_staged_invalid_schema_stops_before_behavior_call(tmp_path: Path) -> None:
    """Invalid schema is preserved as failure evidence, never reparented into an artifact."""
    raw = RawInputs(
        mode="supplied",
        published_contracts={
            "tools": [{"name": "read", "description": "Read.", "parameters": {"type": "object"}}]
        },
    )
    project = ProjectStore(tmp_path, "invalid-layout")
    project.initialize(ProjectConfig(project_id="invalid-layout"))
    calls: list[str] = []

    def complete(body: JsonObject, identity: str) -> JsonObject:
        """Return the saved failure class: an unsupported JSON Schema type."""
        calls.append(identity)
        return {"state_schema": {"type": "any"}, "initialization_requirements": []}

    now = datetime(2030, 1, 1, tzinfo=UTC)
    from jsonschema import SchemaError

    with pytest.raises(SchemaError):
        reconstruct(
            raw,
            strategy="staged",
            artifacts=project.artifacts,
            run_id="bad-layout",
            complete=complete,
            clock=lambda: now,
            deadline=now + timedelta(minutes=1),
            code_revision="test",
        )
    assert len(calls) == 1
    assert project.artifacts.exists(calls[0] + "-response")
    assert not project.artifacts.exists(calls[0])


def test_reference_initializer_copies_values_and_derives_unknowns(tmp_path: Path) -> None:
    """The model chooses bindings; generic code preserves bytes without claiming completeness."""
    from exp.common.core.artifacts import sha256_json
    from exp.simulation.reconstruction.models import ReconstructionBundle
    from exp.simulation.reconstruction.workflow import initialize

    raw = RawInputs(description="Read public entities.")
    bundle = ReconstructionBundle.model_validate(
        {
            "artifact_id": "reference-bundle",
            "sources": raw.model_dump(mode="json"),
            "inputs_sha256": sha256_json(raw.model_dump(mode="json")),
            "interface": {
                "tools": [
                    {"name": "read", "description": "Read.", "parameters": {"type": "object"}}
                ],
                "state_schema": {
                    "type": "object",
                    "properties": {"users": {"type": "object"}, "orders": {"type": "object"}},
                    "required": ["users", "orders"],
                },
            },
            "claims": [
                {
                    "statement": "Missing entities are unknown.",
                    "status": "unknown",
                    "scope": "read",
                    "origin": "model_prior",
                }
            ],
            "text_rules": "Preserve uncertainty.",
        }
    )
    prefix: tuple[JsonObject, ...] = ({"reply": {"id": "p7", "exact_id": "5745575001"}},)
    project = ProjectStore(tmp_path, "reference-init")
    project.initialize(ProjectConfig(project_id="reference-init"))
    now = datetime(2030, 1, 1, tzinfo=UTC)
    result = initialize(
        bundle,
        prefix,
        strategy="references",
        artifacts=project.artifacts,
        run_id="reference-init",
        complete=lambda *_: {"bindings": [{"path": ["users", "p7"], "pointer": "/0/reply"}]},
        clock=lambda: now,
        deadline=now + timedelta(minutes=1),
        code_revision="test",
    )
    assert result.state == {"users": {"p7": prefix[0]["reply"]}}
    assert result.unknown_paths == ("/orders",)
    assert result.complete_scopes == ()
    assert result.field_sources["/users/p7/exact_id"][0].pointer == "/0/reply/exact_id"


def test_layout_format_repair_is_bounded_and_retains_failed_stage(tmp_path: Path) -> None:
    """One declared structural correction is distinct from retrying a wrong native prediction."""
    raw = RawInputs(
        mode="supplied",
        published_contracts={
            "tools": [{"name": "read", "description": "Read.", "parameters": {"type": "object"}}]
        },
    )
    project = ProjectStore(tmp_path, "format-repair")
    project.initialize(ProjectConfig(project_id="format-repair"))
    calls: list[str] = []

    def complete(body: JsonObject, identity: str) -> JsonObject:
        """A failed layout is followed by exactly one corrected layout and behavior."""
        calls.append(identity)
        if len(calls) == 1:
            return {"state_schema": {"type": "any"}}
        if len(calls) == 2:
            return {"state_schema": {"type": "object"}}
        return {
            "claims": [
                {
                    "statement": "Unobserved behavior remains unknown.",
                    "status": "unknown",
                    "scope": "read",
                    "origin": "model_prior",
                }
            ],
            "text_rules": "Preserve uncertainty.",
        }

    now = datetime(2030, 1, 1, tzinfo=UTC)
    bundle = reconstruct(
        raw,
        strategy="staged",
        maximum_layout_repairs=1,
        artifacts=project.artifacts,
        run_id="format-repair",
        complete=complete,
        clock=lambda: now,
        deadline=now + timedelta(minutes=1),
        code_revision="test",
    )
    assert len(calls) == 3
    assert json.loads(project.artifacts.read_bytes(calls[0] + "-response", "response.json"))[
        "state_schema"
    ] == {"type": "any"}
    assert len(project.artifacts.read(bundle.artifact_id).manifest.inputs) == 6


@pytest.mark.parametrize(
    "invalid_claim",
    [
        {"status": "observed", "source_indices": [], "origin": "evidence"},
        {"status": "inferred", "source_indices": [], "origin": "evidence"},
        {"status": "observed", "source_indices": [0], "origin": "model_prior"},
        {"status": "inferred", "source_indices": []},
    ],
)
def test_indexed_wire_schema_matches_claim_authority_admission(
    tmp_path: Path, invalid_claim: JsonObject
) -> None:
    """The advertised schema rejects the same missing provenance as bundle admission."""
    from jsonschema import Draft202012Validator, ValidationError

    raw = RawInputs(episodes=(RawEpisode(id="original", records=({"reply": None},)),))
    project = ProjectStore(tmp_path, "authority")
    project.initialize(ProjectConfig(project_id="authority"))

    def complete(body: JsonObject, identity: str) -> JsonObject:
        """Check the actual framed schema before returning an admitted proposal."""
        messages = body["messages"]
        assert isinstance(messages, list) and isinstance(messages[-1], dict)
        payload = json.loads(str(messages[-1]["content"]))
        proposal = {
            "interface": {
                "tools": [{"name": "read", "description": "Read", "parameters": {}}],
                "state_schema": {"type": "object"},
            },
            "claims": [
                {
                    "statement": "Unknown behavior",
                    "scope": "Unobserved calls",
                    "status": "unknown",
                    "source_indices": [],
                    "origin": "model_prior",
                }
            ],
            "text_rules": "Unobserved behavior is unknown.",
            "python_source": None,
        }
        invalid = {
            **proposal,
            "claims": [{"statement": "A claim", "scope": "A scope", **invalid_claim}],
        }
        validator = Draft202012Validator(payload["output_schema"])
        with pytest.raises(ValidationError):
            validator.validate(invalid)
        validator.validate(proposal)
        return proposal

    now = datetime(2030, 1, 1, tzinfo=UTC)
    reconstruct(
        raw,
        citation_protocol="indexed",
        artifacts=project.artifacts,
        run_id="authority",
        complete=complete,
        clock=lambda: now,
        deadline=now + timedelta(minutes=1),
        code_revision="test",
    )


def test_indexed_claims_preserve_exact_selected_source_spans(tmp_path: Path) -> None:
    """Only the model's support choice is generated; source locators cannot be hallucinated."""
    raw = RawInputs(
        mode="supplied",
        episodes=(RawEpisode(id="observed", records=({"action": "read", "reply": None},)),),
        published_contracts={
            "tools": [{"name": "read", "description": "Read.", "parameters": {"type": "object"}}]
        },
    )
    project = ProjectStore(tmp_path, "indexed")
    project.initialize(ProjectConfig(project_id="indexed"))
    requests: list[JsonObject] = []

    def complete(body: JsonObject, identity: str) -> JsonObject:
        """Choose the actual record's registry entry rather than copy its alias."""
        requests.append(body)
        if len(requests) == 1:
            return {"state_schema": {"type": "object"}}
        messages = body["messages"]
        assert isinstance(messages, list) and isinstance(messages[1], dict)
        payload = json.loads(str(messages[1]["content"]))
        span = next(
            s
            for s in payload["claim_source_spans"]
            if s["source_id"] == "episode:observed" and s["pointer"] == "/0"
        )
        return {
            "claims": [
                {
                    "statement": "The recorded read returned null.",
                    "status": "observed",
                    "scope": "That recorded call.",
                    "origin": "evidence",
                    "source_indices": [span["span_index"]],
                }
            ],
            "text_rules": "Unobserved calls remain unknown.",
        }

    now = datetime(2030, 1, 1, tzinfo=UTC)
    bundle = reconstruct(
        raw,
        strategy="staged",
        citation_protocol="indexed",
        artifacts=project.artifacts,
        run_id="indexed",
        complete=complete,
        clock=lambda: now,
        deadline=now + timedelta(minutes=1),
        code_revision="test",
    )
    assert bundle.claims[0].source_refs[0].source_id == "episode:observed"
    assert bundle.claims[0].source_refs[0].pointer == "/0"


@pytest.mark.parametrize("supplied", [False, True])
def test_flat_indexed_construction_copies_selected_source_references(
    tmp_path: Path, supplied: bool
) -> None:
    """A cold constructor selects finite witnesses without reproducing aliases or pointers."""
    contracts: JsonObject | None = (
        {
            "tools": [
                {"name": "read", "description": "Published read", "parameters": {"type": "object"}}
            ]
        }
        if supplied
        else None
    )
    raw = RawInputs(
        episodes=(RawEpisode(id="native-record", records=({"action": "read", "reply": None},)),),
        mode="supplied" if supplied else "induced",
        published_contracts=contracts,
    )
    project = ProjectStore(tmp_path, "flat-indexed")
    project.initialize(ProjectConfig(project_id="flat-indexed"))

    def complete(body: JsonObject, operation: str) -> JsonObject:
        """Choose the supplied record's index from the actual request inventory."""
        messages = body["messages"]
        assert isinstance(messages, list) and isinstance(messages[1], dict)
        payload = json.loads(str(messages[1]["content"]))
        index = next(
            s["span_index"]
            for s in payload["claim_source_spans"]
            if s["source_id"] == "episode:native-record" and s["pointer"] == "/0"
        )
        result: JsonObject = {
            "claims": [
                {
                    "statement": "Recorded read returned null",
                    "status": "observed",
                    "source_indices": [index],
                    "scope": "That record",
                    "origin": "evidence",
                }
            ],
            "text_rules": "Other calls are unknown",
        }
        if supplied:
            result.update(state_schema={"type": "object"})
        else:
            result["interface"] = {
                "tools": [
                    {
                        "name": "read",
                        "description": "Inferred read",
                        "parameters": {"type": "object"},
                    }
                ],
                "state_schema": {"type": "object"},
            }
        return result

    now = datetime(2030, 1, 1, tzinfo=UTC)
    bundle = reconstruct(
        raw,
        citation_protocol="indexed",
        artifacts=project.artifacts,
        run_id="flat-indexed",
        complete=complete,
        clock=lambda: now,
        deadline=now + timedelta(minutes=1),
        code_revision="test",
    )
    assert bundle.claims[0].source_refs[0].source_id == "episode:native-record"
    assert bundle.claims[0].source_refs[0].pointer == "/0"
    assert bundle.sources.published_contracts == contracts
