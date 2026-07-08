"""Delta tests: op-shape validation and the atomic application guarantees."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from wmh.harness.delta import (
    FailureSignature,
    HarnessDelta,
    SurfaceOp,
    apply_delta,
    compute_delta_id,
)
from wmh.harness.doc import TOOL_POLICY_ID, HarnessDoc, Surface, SurfaceKind


def _parent() -> HarnessDoc:
    doc = HarnessDoc.baseline("parent")
    skill = Surface(
        id="skill:old-skill",
        kind=SurfaceKind.SKILL,
        content="---\nname: old-skill\ndescription: stale advice\n---\ndo the old thing",
    )
    return HarnessDoc(name="parent", surfaces=[*doc.surfaces, skill])


def _delta(
    ops: list[SurfaceOp],
    *,
    parent: HarnessDoc,
    preconditions: dict[str, str] | None = None,
) -> HarnessDelta:
    return HarnessDelta(
        delta_id=compute_delta_id(parent.doc_hash, ops),
        parent_doc_hash=parent.doc_hash,
        trigger=FailureSignature(mechanism="m", task_ids=["t1"], unmet_assertions=["a"]),
        preconditions=preconditions if preconditions is not None else {},
        ops=ops,
        expected_effect="t1 flips to pass",
    )


def _replace_prompt(parent: HarnessDoc, content: str) -> HarnessDelta:
    core = parent.surface("prompt:core")
    assert core is not None
    ops = [SurfaceOp(op="replace", surface_id="prompt:core", content=content, rationale="r")]
    return _delta(ops, parent=parent, preconditions={"prompt:core": core.content_hash})


# -- SurfaceOp shape -------------------------------------------------------------------------


def test_op_shapes_are_validated_at_construction() -> None:
    with pytest.raises(ValidationError, match="declare a kind"):
        SurfaceOp(op="add", surface_id="skill:x", content="c", rationale="r")
    with pytest.raises(ValidationError, match="must carry content"):
        SurfaceOp(op="replace", surface_id="prompt:core", rationale="r")
    with pytest.raises(ValidationError, match="never writes"):
        SurfaceOp(op="remove", surface_id="skill:x", content="c", rationale="r")
    with pytest.raises(ValidationError, match="no rationale"):
        SurfaceOp(op="replace", surface_id="prompt:core", content="c", rationale="  ")


def test_delta_requires_at_least_one_op() -> None:
    parent = _parent()
    with pytest.raises(ValidationError):
        _delta([], parent=parent)


# -- apply_delta: the atomic gate --------------------------------------------------------------


def test_apply_replaces_content_and_records_child_hash() -> None:
    parent = _parent()
    delta = _replace_prompt(parent, "You are a careful agent.")
    child = apply_delta(parent, delta, "child")
    assert child.name == "child"
    assert child.system_prompt() == "You are a careful agent."
    assert delta.child_doc_hash == child.doc_hash
    assert child.doc_hash != parent.doc_hash
    # Untouched surfaces carry over identically.
    assert child.surface(TOOL_POLICY_ID) == parent.surface(TOOL_POLICY_ID)


def test_apply_add_and_remove() -> None:
    parent = _parent()
    old = parent.surface("skill:old-skill")
    assert old is not None
    new_skill = "---\nname: verify-work\ndescription: check results\n---\nRe-run checks."
    ops = [
        SurfaceOp(
            op="add",
            surface_id="skill:verify-work",
            kind=SurfaceKind.SKILL,
            content=new_skill,
            rationale="r",
        ),
        SurfaceOp(op="remove", surface_id="skill:old-skill", rationale="r"),
    ]
    delta = _delta(ops, parent=parent, preconditions={"skill:old-skill": old.content_hash})
    child = apply_delta(parent, delta, "child")
    assert [s.name for s in child.skills()] == ["verify-work"]


def test_apply_rejects_wrong_parent_hash() -> None:
    parent = _parent()
    delta = _replace_prompt(parent, "new prompt")
    other = HarnessDoc.baseline("other")
    with pytest.raises(ValueError, match="proposed against doc"):
        apply_delta(other, delta, "child")


def test_apply_rejects_precondition_mismatch() -> None:
    parent = _parent()
    ops = [SurfaceOp(op="replace", surface_id="prompt:core", content="c", rationale="r")]
    stale = _delta(ops, parent=parent, preconditions={"prompt:core": "0" * 32})
    with pytest.raises(ValueError, match="precondition mismatch"):
        apply_delta(parent, stale, "child")
    unknown = _delta(
        [
            SurfaceOp(
                op="add", surface_id="skill:s", kind=SurfaceKind.SKILL, content="x", rationale="r"
            )
        ],
        parent=parent,
        preconditions={"skill:ghost": "0" * 32},
    )
    with pytest.raises(ValueError, match="unknown surface"):
        apply_delta(parent, unknown, "child")


def test_replace_and_remove_require_a_precondition() -> None:
    parent = _parent()
    naked = _delta(
        [SurfaceOp(op="replace", surface_id="prompt:core", content="c", rationale="r")],
        parent=parent,
    )
    with pytest.raises(ValueError, match="no precondition"):
        apply_delta(parent, naked, "child")


def test_apply_rejects_unknown_target_and_add_collision() -> None:
    parent = _parent()
    ghost = _delta(
        [SurfaceOp(op="remove", surface_id="skill:ghost", rationale="r")],
        parent=parent,
        preconditions={"skill:ghost": "0" * 32},
    )
    with pytest.raises(ValueError, match="unknown surface"):
        apply_delta(parent, ghost, "child")
    collide = _delta(
        [
            SurfaceOp(
                op="add",
                surface_id="prompt:core",
                kind=SurfaceKind.PROMPT,
                content="c",
                rationale="r",
            )
        ],
        parent=parent,
    )
    with pytest.raises(ValueError, match="already exists"):
        apply_delta(parent, collide, "child")


def test_apply_rejects_kind_mismatch_and_duplicate_targets() -> None:
    parent = _parent()
    core = parent.surface("prompt:core")
    assert core is not None
    wrong_kind = _delta(
        [
            SurfaceOp(
                op="replace",
                surface_id="prompt:core",
                kind=SurfaceKind.SKILL,
                content="c",
                rationale="r",
            )
        ],
        parent=parent,
        preconditions={"prompt:core": core.content_hash},
    )
    with pytest.raises(ValueError, match="declares kind"):
        apply_delta(parent, wrong_kind, "child")
    doubled = _delta(
        [
            SurfaceOp(op="replace", surface_id="prompt:core", content="a", rationale="r"),
            SurfaceOp(op="replace", surface_id="prompt:core", content="b", rationale="r"),
        ],
        parent=parent,
        preconditions={"prompt:core": core.content_hash},
    )
    with pytest.raises(ValueError, match="multiple ops"):
        apply_delta(parent, doubled, "child")


def test_invalid_child_fails_whole_document_validation() -> None:
    # Dropping `submit` from the tool policy must fail the child's construction gate.
    parent = _parent()
    policy = parent.surface(TOOL_POLICY_ID)
    assert policy is not None
    delta = _delta(
        [SurfaceOp(op="replace", surface_id=TOOL_POLICY_ID, content="bash", rationale="r")],
        parent=parent,
        preconditions={TOOL_POLICY_ID: policy.content_hash},
    )
    with pytest.raises(ValueError, match="submit"):
        apply_delta(parent, delta, "child")


def test_replace_inherits_budget_unless_overridden() -> None:
    core = Surface(id="prompt:core", kind=SurfaceKind.PROMPT, content="p", budget=100)
    parent = HarnessDoc(name="parent", surfaces=[core])
    delta = _delta(
        [SurfaceOp(op="replace", surface_id="prompt:core", content="q", rationale="r")],
        parent=parent,
        preconditions={"prompt:core": core.content_hash},
    )
    child = apply_delta(parent, delta, "child")
    replaced = child.surface("prompt:core")
    assert replaced is not None and replaced.budget == 100
    # A budget is enforced through replacement: over-budget content rejects the delta.
    over = _delta(
        [SurfaceOp(op="replace", surface_id="prompt:core", content="x" * 101, rationale="r")],
        parent=parent,
        preconditions={"prompt:core": core.content_hash},
    )
    with pytest.raises(ValueError, match="over its budget"):
        apply_delta(parent, over, "child")


def test_delta_id_is_deterministic_and_content_addressed() -> None:
    parent = _parent()
    ops_a = [SurfaceOp(op="replace", surface_id="prompt:core", content="a", rationale="r")]
    ops_b = [SurfaceOp(op="replace", surface_id="prompt:core", content="b", rationale="r")]
    assert compute_delta_id(parent.doc_hash, ops_a) == compute_delta_id(parent.doc_hash, ops_a)
    assert compute_delta_id(parent.doc_hash, ops_a) != compute_delta_id(parent.doc_hash, ops_b)
    assert compute_delta_id(parent.doc_hash, ops_a) != compute_delta_id("other", ops_a)


def _pi_parent() -> HarnessDoc:
    """A pi-node doc with a pathful code surface, like the vendored pi harness the search edits."""
    from wmh.harness.doc import RUNTIME_KIND_ID

    return HarnessDoc(
        name="pi",
        surfaces=[
            Surface(id="prompt:core", kind=SurfaceKind.PROMPT, content="p"),
            Surface(id=TOOL_POLICY_ID, kind=SurfaceKind.TOOL_POLICY, content="bash\nsubmit"),
            Surface(id=RUNTIME_KIND_ID, kind=SurfaceKind.PARAM, content="pi-node"),
            Surface(
                id="code:src-agent-ts", kind=SurfaceKind.CODE, path="src/agent.ts", content="// v1"
            ),
        ],
    )


def test_apply_preserves_code_surface_path_on_replace() -> None:
    # Editing a vendored pi code surface must keep its path — else the child is a path-less CODE
    # surface and construction rejects it (the search could never mutate pi source before this fix).
    parent = _pi_parent()
    src = parent.surface("code:src-agent-ts")
    assert src is not None
    ops = [SurfaceOp(op="replace", surface_id="code:src-agent-ts", content="// v2", rationale="r")]
    delta = _delta(ops, parent=parent, preconditions={"code:src-agent-ts": src.content_hash})
    child = apply_delta(parent, delta, "child")
    edited = child.surface("code:src-agent-ts")
    assert edited is not None
    assert edited.content == "// v2" and edited.path == "src/agent.ts"


def test_apply_add_pathful_code_surface() -> None:
    parent = _pi_parent()
    ops = [
        SurfaceOp(
            op="add",
            surface_id="code:src-b-ts",
            kind=SurfaceKind.CODE,
            path="src/b.ts",
            content="// b",
            rationale="r",
        )
    ]
    child = apply_delta(parent, _delta(ops, parent=parent), "child")
    added = child.surface("code:src-b-ts")
    assert added is not None and added.path == "src/b.ts"
