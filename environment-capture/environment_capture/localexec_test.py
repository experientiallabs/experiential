"""Tests for the local bash workspace environment."""

from __future__ import annotations

from pathlib import Path

from environment_capture.localexec import LocalBashEnv


def test_executes_in_workspace_and_captures_output(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.txt").write_text("hello capex")
    env = LocalBashEnv(workspace=tmp_path)
    try:
        result = env.execute("ls docs && grep -c capex docs/a.txt")
        assert result.returncode == 0
        assert "a.txt" in result.output
        assert "1" in result.output
    finally:
        env.close()


def test_nonzero_returncode_and_stderr_are_captured(tmp_path: Path) -> None:
    env = LocalBashEnv(workspace=tmp_path)
    try:
        result = env.execute("cat missing.txt")
        assert result.returncode != 0
        assert "missing.txt" in result.output  # stderr folded into the observation
    finally:
        env.close()


def test_timeout_returns_error_result(tmp_path: Path) -> None:
    env = LocalBashEnv(workspace=tmp_path, timeout_s=1)
    try:
        result = env.execute("sleep 5")
        assert result.returncode != 0
        assert "timed out" in result.output
    finally:
        env.close()


def test_state_does_not_leak_between_commands(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    env = LocalBashEnv(workspace=tmp_path)
    try:
        env.execute("export SECRET=42; cd sub >/dev/null")
        result = env.execute("pwd && echo ${SECRET:-unset}")
        assert str(tmp_path) in result.output  # fresh subshell, cwd reset
        assert "unset" in result.output
    finally:
        env.close()


def test_containment_guard_blocks_host_targeting_commands(tmp_path: Path) -> None:
    """The env is DEFINED as workspace-scoped: a host-targeting command is refused without
    executing, and the refusal is the real observation the agent (and the corpus) sees."""
    env = LocalBashEnv(workspace=tmp_path)
    try:
        for command in ("ls ~", "find / -name x", "cat /Users/someone/.ssh/config", "cd .."):
            result = env.execute(command)
            assert result.returncode != 0, command
            assert "workspace" in result.output, command
        # And nothing was actually executed against the host.
        ok = env.execute("echo safe && ls")
        assert ok.returncode == 0
    finally:
        env.close()


def test_containment_guard_can_be_disabled(tmp_path: Path) -> None:
    env = LocalBashEnv(workspace=tmp_path, contain=False)
    try:
        result = env.execute("ls /tmp >/dev/null; echo ran")
        assert "ran" in result.output
    finally:
        env.close()
