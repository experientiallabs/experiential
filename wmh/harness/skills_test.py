"""Tests for SKILL.md skills: round-trip, directory IO, progressive-disclosure index."""

from __future__ import annotations

from pathlib import Path

import pytest

from wmh.harness.skills import Skill, SkillLibrary


def test_skill_markdown_roundtrip() -> None:
    skill = Skill(name="grep-logs", description="find errors in logs", body="grep -i error *.log")
    assert Skill.from_markdown(skill.to_markdown()) == skill


def test_malformed_skill_raises() -> None:
    with pytest.raises(ValueError, match="frontmatter"):
        Skill.from_markdown("just a body, no frontmatter")
    with pytest.raises(ValueError, match="kebab-case"):
        Skill.from_markdown("---\nname: Not Kebab\ndescription: d\n---\nbody")


def test_library_dir_roundtrip(tmp_path: Path) -> None:
    library = SkillLibrary(
        [Skill(name="find-big-files", description="locate large files", body="du -ah | sort -h")]
    )
    library.write_dir(tmp_path)
    reloaded = SkillLibrary.from_dir(tmp_path)
    assert reloaded.names() == ["find-big-files"]
    got = reloaded.get("find-big-files")
    assert got is not None and "du -ah" in got.body


def test_from_missing_dir_is_empty(tmp_path: Path) -> None:
    assert len(SkillLibrary.from_dir(tmp_path / "nope")) == 0


def test_render_index_is_names_and_descriptions_only() -> None:
    library = SkillLibrary([Skill(name="a-skill", description="does A", body="secret body A")])
    index = library.render_index()
    assert "a-skill: does A" in index
    assert "secret body A" not in index  # bodies are disclosed only via read_skill
    assert SkillLibrary().render_index() == ""
