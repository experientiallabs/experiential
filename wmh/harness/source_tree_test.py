"""Tests for the portable, editable harness source-tree representation."""

from __future__ import annotations

import pytest

from wmh.harness.doc import (
    MAX_OUTPUT_TOKENS_ID,
    RUNTIME_KIND_ID,
    TOOL_POLICY_ID,
    HarnessDoc,
    Surface,
    SurfaceKind,
)
from wmh.harness.pi_vendor import pi_agent_code_surfaces
from wmh.harness.skills import Skill
from wmh.harness.source_tree import HarnessSourceFile, HarnessSourceTree


def _pi_document() -> HarnessDoc:
    skill = Skill(name="inspect-code", description="Inspect code", body="Use rg before editing.")
    return HarnessDoc(
        name="pi",
        surfaces=[
            Surface(id="prompt:core", kind=SurfaceKind.PROMPT, content="Work carefully."),
            Surface(id=TOOL_POLICY_ID, kind=SurfaceKind.TOOL_POLICY, content="bash\nsubmit"),
            Surface(id=RUNTIME_KIND_ID, kind=SurfaceKind.PARAM, content="pi-node"),
            Surface(id=MAX_OUTPUT_TOKENS_ID, kind=SurfaceKind.PARAM, content="8192"),
            Surface(
                id="skill:inspect-code",
                kind=SurfaceKind.SKILL,
                content=skill.to_markdown(),
            ),
            *pi_agent_code_surfaces(),
        ],
    )


def test_source_tree_round_trips_a_complete_pathful_harness() -> None:
    """The public call site carries every executable file without store metadata."""
    original = _pi_document()

    tree = HarnessSourceTree.from_doc(original)
    restored = tree.to_doc("restored")

    paths = set(tree.file_map())
    assert "SYSTEM.md" in paths
    assert "config.toml" in paths
    assert "skills/inspect-code.md" in paths
    assert "doc.json" not in paths
    assert restored.system_prompt() == original.system_prompt()
    assert restored.tools() == original.tools()
    assert restored.runtime_kind() == original.runtime_kind()
    assert restored.max_output_tokens() == original.max_output_tokens()
    assert [skill.name for skill in restored.skills()] == ["inspect-code"]
    assert {(item.path, item.content) for item in restored.code_files()} == {
        (item.path, item.content) for item in original.code_files()
    }


def test_source_tree_reparse_preserves_deletions_and_rewrites() -> None:
    original = HarnessSourceTree.from_doc(_pi_document())
    deleted_path = next(item.path for item in original.files if item.path.startswith("src/"))
    rewritten = HarnessSourceTree(
        files=tuple(
            item.model_copy(update={"content": "Changed prompt."})
            if item.path == "SYSTEM.md"
            else item
            for item in original.files
            if item.path != deleted_path
        )
    )

    candidate = rewritten.to_doc("candidate")

    assert candidate.system_prompt() == "Changed prompt."
    assert deleted_path not in {item.path for item in candidate.code_files()}
    assert len(candidate.code_files()) == len(_pi_document().code_files()) - 1
    assert rewritten.tree_hash != original.tree_hash


@pytest.mark.parametrize("path", ["doc.json", "aliases.toml"])
def test_source_tree_rejects_store_authority_files(path: str) -> None:
    with pytest.raises(ValueError, match="store metadata"):
        HarnessSourceTree(files=(HarnessSourceFile(path=path, content="{}"),))


@pytest.mark.parametrize("path", ["/absolute.ts", "../escape.ts", "src/../escape.ts", "src\\x.ts"])
def test_source_tree_rejects_noncanonical_paths(path: str) -> None:
    with pytest.raises(ValueError, match="canonical relative POSIX path"):
        HarnessSourceTree(files=(HarnessSourceFile(path=path, content="x"),))


def test_source_tree_rejects_duplicate_paths() -> None:
    with pytest.raises(ValueError, match="duplicate source path"):
        HarnessSourceTree(
            files=(
                HarnessSourceFile(path="SYSTEM.md", content="one"),
                HarnessSourceFile(path="SYSTEM.md", content="two"),
            )
        )


def test_source_tree_rejects_case_insensitive_path_collisions() -> None:
    """{SYSTEM.md, system.md} silently corrupts renders on case-insensitive filesystems."""
    with pytest.raises(
        ValueError, match=r"'SYSTEM\.md' and 'system\.md' differ only by letter case"
    ):
        HarnessSourceTree(
            files=(
                HarnessSourceFile(path="SYSTEM.md", content="one"),
                HarnessSourceFile(path="system.md", content="two"),
            )
        )


def test_source_tree_rejects_file_and_directory_prefix_conflicts() -> None:
    """{src, src/a.ts} would crash a store write halfway instead of failing validation."""
    with pytest.raises(
        ValueError, match=r"'src' is a file but is also the directory holding 'src/a\.ts'"
    ):
        HarnessSourceTree(
            files=(
                HarnessSourceFile(path="SYSTEM.md", content="p"),
                HarnessSourceFile(path="src", content="conflict"),
                HarnessSourceFile(path="src/a.ts", content="x"),
            )
        )


def test_source_tree_rejects_a_file_that_aliases_the_runtime_surface() -> None:
    """A file named exactly `runtime` must not hijack the in-process code:runtime surface."""
    tree = HarnessSourceTree(
        files=(
            HarnessSourceFile(path="SYSTEM.md", content="p"),
            HarnessSourceFile(path="runtime", content="x"),
        )
    )

    with pytest.raises(ValueError, match="would alias the reserved in-process runtime"):
        tree.to_doc("invalid")


def test_source_tree_rejects_unparsed_skill_namespace_files() -> None:
    tree = HarnessSourceTree(
        files=(
            HarnessSourceFile(path="SYSTEM.md", content="p"),
            HarnessSourceFile(path="skills/nested/ignored.md", content="ignored"),
        )
    )

    with pytest.raises(ValueError, match="skill source path"):
        tree.to_doc("invalid")


@pytest.mark.parametrize("path", ["src/agent_utils.ts", "Upper.ts", "src/a..b.ts"])
def test_source_tree_names_the_file_behind_an_invalid_code_surface_id(path: str) -> None:
    """A path outside the surface-id grammar fails naming the file, not just the bad slug."""
    tree = HarnessSourceTree(
        files=(
            HarnessSourceFile(path="SYSTEM.md", content="p"),
            HarnessSourceFile(path=path, content="x"),
        )
    )

    with pytest.raises(ValueError, match=rf"code file path '{path}'.*kebab-slug"):
        tree.to_doc("invalid")


def test_source_tree_names_both_files_behind_a_surface_id_collision() -> None:
    """`/` and `.` both map to `-`, so distinct paths can collide on one surface id."""
    tree = HarnessSourceTree(
        files=(
            HarnessSourceFile(path="SYSTEM.md", content="p"),
            HarnessSourceFile(path="src/a-b.ts", content="one"),
            HarnessSourceFile(path="src/a/b.ts", content="two"),
        )
    )

    with pytest.raises(ValueError, match=r"'src/a-b\.ts' and 'src/a/b\.ts' both map"):
        tree.to_doc("invalid")


@pytest.mark.parametrize(
    "config, error",
    [
        ("[provider]\nmodel = 'mutable'\n", "unknown top-level"),
        ("[harness]\ntools = ['submit']\nmodel = 'mutable'\n", "unknown field"),
        ("[harness]\nruntime_kind = false\n", "runtime_kind must be a string"),
    ],
)
def test_source_tree_rejects_config_that_would_not_affect_the_parsed_harness(
    config: str,
    error: str,
) -> None:
    tree = HarnessSourceTree(
        files=(
            HarnessSourceFile(path="SYSTEM.md", content="p"),
            HarnessSourceFile(path="config.toml", content=config),
        )
    )

    with pytest.raises(ValueError, match=error):
        tree.to_doc("invalid")


def test_bounds_validation_reports_files_and_bytes() -> None:
    tree = HarnessSourceTree(
        files=(
            HarnessSourceFile(path="SYSTEM.md", content="p"),
            HarnessSourceFile(path="src/a.ts", content="abcdef"),
        )
    )

    tree.validate_bounds(max_files=2, max_bytes=7)
    assert tree.total_bytes == 7
    with pytest.raises(ValueError, match="more than 1 files"):
        tree.validate_bounds(max_files=1, max_bytes=7)
    with pytest.raises(ValueError, match="more than 6 bytes"):
        tree.validate_bounds(max_files=2, max_bytes=6)
