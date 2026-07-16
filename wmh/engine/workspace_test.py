"""Tests for repo-tree grounding (fakes, no network)."""

from __future__ import annotations

import json

from wmh.core.types import Action, ActionKind
from wmh.engine.grounding import SourceResolver
from wmh.engine.workspace import (
    RepoTreeResolver,
    SessionFiles,
    TreeQuery,
    extract_tree_query,
    textop_answer,
    textop_grounded_knowledge,
    tree_grounded_knowledge,
)

_PINS = {"i-1": {"repo": "o/r", "base_commit": "c0ffee"}}
_TREE = ["pkg", "pkg/a.py", "pkg/b.py", "pkg/sub", "pkg/sub/c.py", "README.md"]


def _tree_fetch(url: str, headers: dict[str, str]) -> str:
    if url.startswith("https://api.github.com/"):
        return json.dumps({"tree": [{"path": p} for p in _TREE], "truncated": False})
    # raw file content
    name = url.rsplit("/", 1)[-1]
    return f"import os\nVALUE = '{name}'\ndef needle():\n    pass\n"


def _bash(command: str) -> Action:
    return Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": command})


def test_extract_tree_query_parses_ls_find_grep() -> None:
    q = extract_tree_query(_bash("ls /testbed/pkg"))
    assert q == TreeQuery(kind="list", target="pkg", pattern=None, recursive=False)
    q2 = extract_tree_query(_bash("find /testbed/pkg -name '*.py'"))
    assert q2 == TreeQuery(kind="list", target="pkg", pattern="*.py", recursive=True)
    q3 = extract_tree_query(_bash("cd /testbed && grep -rn 'needle' pkg/"))
    assert q3 is not None and q3.kind == "grep" and q3.pattern == "needle"
    assert q3.target == "pkg" and q3.recursive and q3.line_numbers


def test_extract_tree_query_rejects_pipes_and_writes() -> None:
    assert extract_tree_query(_bash("ls /testbed/pkg | wc -l")) is None
    assert extract_tree_query(_bash("grep -rn x pkg/ > hits.txt")) is None
    assert extract_tree_query(_bash("echo hi")) is None


def test_listing_served_exactly_from_the_tree() -> None:
    tree = RepoTreeResolver(_PINS, fetch=_tree_fetch)
    out = tree.answer("i-1", TreeQuery(kind="list", target="pkg", pattern=None, recursive=False))
    assert out is not None
    assert "a.py" in out and "sub" in out
    assert "c.py" not in out  # non-recursive ls does not descend


def test_find_with_name_pattern_is_recursive() -> None:
    tree = RepoTreeResolver(_PINS, fetch=_tree_fetch)
    out = tree.answer("i-1", TreeQuery(kind="list", target="pkg", pattern="*.py", recursive=True))
    assert out is not None
    assert "pkg/a.py" in out and "pkg/sub/c.py" in out and "README.md" not in out


def test_grep_runs_locally_over_fetched_files() -> None:
    tree = RepoTreeResolver(_PINS, fetch=_tree_fetch)
    source = SourceResolver(_PINS, fetch=_tree_fetch)
    out = tree.answer(
        "i-1",
        TreeQuery(kind="grep", target="pkg", pattern="needle", recursive=True, line_numbers=True),
        source=source,
    )
    assert out is not None
    assert "pkg/a.py:3:" in out and "def needle" in out  # path:lineno:content, grep -rn shape


def test_grep_degrades_to_filename_list_over_the_fetch_cap() -> None:
    big_tree = [f"pkg/f{i}.py" for i in range(200)]

    def fetch(url: str, headers: dict[str, str]) -> str:
        if url.startswith("https://api.github.com/"):
            return json.dumps({"tree": [{"path": p} for p in big_tree], "truncated": False})
        raise AssertionError("must not fetch file content over the cap")

    tree = RepoTreeResolver(_PINS, fetch=fetch, grep_fetch_cap=30)
    out = tree.answer(
        "i-1",
        TreeQuery(kind="grep", target="pkg", pattern="x", recursive=True),
        source=SourceResolver(_PINS, fetch=fetch),
    )
    assert out is not None
    assert "too many files to search" in out  # honest partial: listing only, clearly labeled
    assert "pkg/f0.py" in out


def test_tree_grounded_knowledge_gates_on_session_writes() -> None:
    tree = RepoTreeResolver(_PINS, fetch=_tree_fetch)
    source = SourceResolver(_PINS, fetch=_tree_fetch)
    # grep whose target dir contains a session-edited file -> annotated, not asserted as truth
    prior = [_bash("sed -i 's/a/b/' /testbed/pkg/a.py")]
    text = tree_grounded_knowledge(
        None, _bash("grep -rn 'needle' /testbed/pkg"), "i-1", prior, tree, source
    )
    assert text is not None
    assert "session has since modified" in text  # annotation present
    unpinned = tree_grounded_knowledge(
        None, _bash("grep -rn 'needle' /testbed/pkg"), "unknown", prior, tree, source
    )
    assert unpinned is None  # unpinned instance: silent no-op


def test_session_files_reconstructs_heredocs_and_appends() -> None:
    history = [
        _bash("cat > /tmp/data.txt << 'EOF'\nbeta\nalpha\nalpha\nEOF"),
        _bash("echo 'gamma' >> /tmp/data.txt"),
    ]
    files = SessionFiles.from_actions(history)
    # EXACT file bytes: heredoc + echo each end with the newline the real shell writes.
    assert files.content("/tmp/data.txt") == "beta\nalpha\nalpha\ngamma\n"


def test_textop_computes_over_known_content_only() -> None:
    history = [_bash("cat > /tmp/data.txt << 'EOF'\nbeta\nalpha\nalpha\nEOF")]
    files = SessionFiles.from_actions(history)
    assert textop_answer(_bash("wc -l /tmp/data.txt"), files) == "3 /tmp/data.txt"
    assert textop_answer(_bash("sort /tmp/data.txt"), files) == "alpha\nalpha\nbeta"
    assert textop_answer(_bash("sort -u /tmp/data.txt"), files) == "alpha\nbeta"
    # unknown file or unmodeled op: refuse (None), never guess
    assert textop_answer(_bash("wc -l /tmp/other.txt"), files) is None
    assert textop_answer(_bash("awk '{print}' /tmp/data.txt"), files) is None


def test_textop_grounded_knowledge_injects_labeled_answer() -> None:
    history = [_bash("cat > /tmp/n.txt << 'EOF'\n1\n2\nEOF")]
    text = textop_grounded_knowledge(None, _bash("wc -l /tmp/n.txt"), [h for h in history])
    assert text is not None and "computed deterministically" in text and "2 /tmp/n.txt" in text


def test_textop_answers_the_final_segment_of_a_compound_command() -> None:
    # the measured corpus shape: write the file and query it in ONE command
    cmd = (
        "cat > /tmp/scores.txt << 'EOF'\nAlice 95\nBob 87\nCharlie 92\nEOF\n"
        "sort -k2 -n -r /tmp/scores.txt"
    )
    text = textop_grounded_knowledge(None, _bash(cmd), [])
    assert text is not None
    assert text.index("Alice 95") < text.index("Charlie 92") < text.index("Bob 87")


def test_textop_printf_write_then_sort_unique() -> None:
    cmd = 'printf "apple\\nbanana\\napple\\n" > /tmp/d.txt && sort -u /tmp/d.txt'
    text = textop_grounded_knowledge(None, _bash(cmd), [])
    assert text is not None and "apple\nbanana" in text


def test_textop_cat_cut_head_over_known_content() -> None:
    history = [_bash("cat > /tmp/p.csv << 'EOF'\na,1\nb,2\nc,3\nEOF")]
    files = SessionFiles.from_actions(history)
    assert textop_answer(_bash("cat /tmp/p.csv"), files) == "a,1\nb,2\nc,3"
    assert textop_answer(_bash("cut -d, -f1 /tmp/p.csv"), files) == "a\nb\nc"
    assert textop_answer(_bash("head -n 2 /tmp/p.csv"), files) == "a,1\nb,2"
    assert textop_answer(_bash("wc -l < /tmp/p.csv"), files) == "3"


def test_textop_refuses_redirected_finals_and_same_command_taint() -> None:
    # final segment redirects: the observation is not the op's output
    cmd = 'printf "x\\n" > /tmp/a.txt && sort /tmp/a.txt > /tmp/b.txt'
    assert textop_grounded_knowledge(None, _bash(cmd), []) is None
    # an unknowable write BETWEEN the known write and the query taints the path
    cmd2 = "cat > /tmp/f.txt << 'EOF'\nx\nEOF\nsed -i 's/x/y/' /tmp/f.txt\nsort /tmp/f.txt"
    assert textop_grounded_knowledge(None, _bash(cmd2), []) is None


def test_repo_relative_dot_means_root_and_flags_never_parse_as_paths() -> None:
    # `find . -name '*.py'` is the most common exploration shape: '.' must mean the repo
    # root, and a bare `ls -la` (no path) must refuse, not treat '-la' as a directory.
    q = extract_tree_query(_bash("find . -name '*.py'"))
    assert q is not None and q.target == ""
    q2 = extract_tree_query(_bash("ls ."))
    assert q2 is not None and q2.target == ""
    assert extract_tree_query(_bash("ls -la")) is None


def test_grep_ignore_case_flag_is_honored() -> None:
    tree = RepoTreeResolver(_PINS, fetch=_tree_fetch)
    source = SourceResolver(_PINS, fetch=_tree_fetch)
    out = tree.answer(
        "i-1",
        TreeQuery(kind="grep", target="pkg", pattern="NEEDLE", recursive=True, ignore_case=True),
        source=source,
    )
    assert out is not None and "needle" in out  # matched case-insensitively


def test_wc_is_exact_for_printf_without_trailing_newline() -> None:
    cmd = "printf 'abc' > /tmp/x.txt"
    files = SessionFiles()
    files.apply_command(cmd)
    assert textop_answer(_bash("wc -c /tmp/x.txt"), files) == "3 /tmp/x.txt"
    assert textop_answer(_bash("wc -l /tmp/x.txt"), files) == "0 /tmp/x.txt"


def test_taint_covers_extensionless_files() -> None:
    files = SessionFiles()
    files.apply_command("echo 'one' > results")
    files.apply_command("sed -i 's/one/two/' results")
    assert files.content("results") is None  # unknowable after in-place edit


def test_ls_of_an_existing_file_answers_the_file() -> None:
    tree = RepoTreeResolver(_PINS, fetch=_tree_fetch)
    out = tree.answer("i-1", TreeQuery(kind="list", target="pkg/a.py", pattern=None))
    assert out == "pkg/a.py"  # real `ls <file>` prints the path, not "(no matches)"


def test_grep_labels_skipped_extensions_honestly() -> None:
    paths = ["pkg/a.py", "pkg/config.yaml", "pkg/Makefile"]

    def fetch(url: str, headers: dict[str, str]) -> str:
        if url.startswith("https://api.github.com/"):
            return json.dumps({"tree": [{"path": p} for p in paths], "truncated": False})
        return "no hits here\n"

    tree = RepoTreeResolver(_PINS, fetch=fetch)
    source = SourceResolver(_PINS, fetch=fetch)
    out = tree.answer(
        "i-1",
        TreeQuery(kind="grep", target="pkg", pattern="needle", recursive=True),
        source=source,
    )
    assert out is not None
    assert "2 other files under this path were not searched" in out  # yaml + Makefile
