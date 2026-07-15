"""Tests for corpus hygiene: detecting trajectories that escaped the task workspace."""

from __future__ import annotations

import json
from pathlib import Path

from environment_capture.hygiene import (
    command_targets_host,
    host_escape_findings,
    partition_contained,
    scan_spans_jsonl,
)
from environment_capture.trajectory import StepRecord, Task, ToolCall, Trajectory


def _trajectory(command: str, output: str) -> Trajectory:
    return Trajectory(
        task=Task(task_id="t0", prompt="q", data={}),
        steps=[
            StepRecord(
                action=ToolCall(name="bash", arguments={"command": command}),
                output=output,
                is_error=False,
            )
        ],
    )


def test_workspace_contained_trajectory_is_clean() -> None:
    clean = _trajectory(
        "ls docs && grep -RinE 'Net sales|/shares/' docs/*.txt | head -5",
        "docs/a.txt:12: Net sales were $1,577",
    )
    assert host_escape_findings(clean) == []


def test_host_targeting_commands_are_flagged() -> None:
    for command in (
        "ls -R /home 2>/dev/null | head -50",
        "find / -name database.db 2>/dev/null",
        "ls ~",
        "ls -la $HOME",
        "cd .. && ls",
        "cat /Users/someone/.ssh/config",
        "cd /root",
    ):
        findings = host_escape_findings(_trajectory(command, "whatever"))
        assert findings, f"expected flag for command: {command}"
        assert findings[0].field == "command"


def test_host_content_in_observations_is_flagged() -> None:
    for output in (
        "drwx------@ 3 someuser staff 96 Jul 1 22:06 Desktop\n.ssh\nid_ecdsa.pub",
        'File "/Users/someone/anaconda3/lib/python3.11/re/__init__.py", line 176',
        "bash: line 0: cd: /root: No such file or directory",
        "/home/user/project/node_modules",
    ):
        findings = host_escape_findings(_trajectory("echo hi", output))
        assert findings, f"expected flag for output: {output[:40]}"
        assert findings[0].field == "output"


def test_machine_username_in_observation_is_flagged() -> None:
    """`ls -l` ownership columns leak the machine username even without leaving the workspace;
    the detector learns the CURRENT user/home at runtime so no personal string lives in code."""
    import getpass

    user = getpass.getuser()
    listing = f"total 0\ndrwx------@ 3 {user}  staff  96 Jul  1 22:06 ."
    findings = host_escape_findings(_trajectory("ls -la", listing))
    assert findings and findings[0].field == "output"
    assert user in findings[0].marker  # matched in ownership context, not as a bare word


def test_partition_contained_splits_and_preserves_order() -> None:
    clean_1 = _trajectory("ls docs", "a.txt")
    dirty = _trajectory("ls ~", "Desktop")
    clean_2 = _trajectory("cat docs/a.txt", "text")
    clean, flagged = partition_contained([clean_1, dirty, clean_2])
    assert clean == [clean_1, clean_2]
    assert flagged == [dirty]


def test_generic_path_markers_false_allows_simulated_paths_in_observations() -> None:
    """A benchmark whose OWN environment uses ~/ or /home paths as content (e.g. AppWorld's
    simulated file system) can opt out of the generic path markers for observations."""
    sim = _trajectory(
        "print(apis.file_system.show_file(path='~/documents/report.txt'))",
        "wrote 3 rows to ~/documents/report.txt (see /home/appuser/documents)",
    )
    # By default the simulated ~/ path is treated as a host marker and flagged...
    assert host_escape_findings(sim)
    # ...but with generic_path_markers=False the observation path markers are skipped.
    assert host_escape_findings(sim, generic_path_markers=False) == []


def test_generic_path_markers_false_still_flags_real_identity_leak() -> None:
    """Opting out of GENERIC path markers must NOT disable the runtime identity markers: a real
    username / home leak (e.g. os.path.expanduser echoing the account) is still caught."""
    import getpass

    user = getpass.getuser()
    leak = _trajectory("print(os.path.expanduser('~'))", f"home is {str(Path.home())} for {user}")
    findings = host_escape_findings(leak, generic_path_markers=False)
    assert findings and findings[0].field == "output"
    assert findings[0].marker in (user, str(Path.home()))


def test_generic_path_markers_false_still_flags_commands() -> None:
    """Command-level host targeting is checked unconditionally, regardless of the flag."""
    dirty = _trajectory("cat /Users/someone/.ssh/config", "ok")
    findings = host_escape_findings(dirty, generic_path_markers=False)
    assert findings and findings[0].field == "command"


def test_partition_contained_respects_generic_path_markers() -> None:
    sim = _trajectory("print('save')", "saved to ~/documents/out.csv")
    clean, flagged = partition_contained([sim])
    assert flagged == [sim]  # default: flagged
    clean, flagged = partition_contained([sim], generic_path_markers=False)
    assert clean == [sim] and flagged == []


def test_scan_spans_jsonl_maps_trace_ids_to_findings(tmp_path: Path) -> None:
    def span(trace_id: str, key: str, value: str) -> dict[str, object]:
        return {
            "traceId": trace_id,
            "spanId": f"{trace_id}-s",
            "attributes": [{"key": key, "value": {"stringValue": value}}],
        }

    path = tmp_path / "traces.otel.jsonl"
    lines = [
        span("aaa", "gen_ai.tool.call.arguments", json.dumps({"command": "ls docs"})),
        span("aaa", "gen_ai.tool.message", "a.txt"),
        span("bbb", "gen_ai.tool.call.arguments", json.dumps({"command": "ls -R /home"})),
        span("ccc", "gen_ai.tool.message", 'File "/Users/x/anaconda3/lib/re.py" line 1'),
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")

    flagged = scan_spans_jsonl(path)
    assert set(flagged) == {"bbb", "ccc"}
    assert flagged["bbb"][0].field == "command"
    assert flagged["ccc"][0].field == "output"


def test_import_survives_missing_username(monkeypatch) -> None:  # noqa: ANN001
    """A container run as a bare uid (no passwd entry, no USER env) must not crash the harness:
    runtime identity markers degrade to whatever is resolvable instead of raising at import."""
    import environment_capture.hygiene as hygiene

    def _boom() -> str:
        raise KeyError("getpwuid(): uid not found")

    monkeypatch.setattr(hygiene.getpass, "getuser", _boom)
    hygiene._runtime_markers_cache = None
    hygiene._identity_regexes_cache = None
    try:
        markers = hygiene._runtime_markers()
        assert str(Path.home()) in " ".join(markers)  # home still contributes
        clean = _trajectory("ls docs", "a.txt")
        assert host_escape_findings(clean) == []  # detection still works
        assert hygiene._identity_regexes() == ()
    finally:
        hygiene._runtime_markers_cache = None
        hygiene._identity_regexes_cache = None
    # The failure must not be memoized: once getuser resolves again (monkeypatch undone),
    # identity detection comes back in the same process — the cache is success-only.
    monkeypatch.undo()
    assert hygiene._identity_regexes() != ()


def test_bare_username_word_does_not_flag() -> None:
    """A common username appearing as an ordinary WORD is not an identity leak — only
    ownership-column (ls -l) and home-path contexts are. Guards CI boxes with usernames like
    'runner'/'ubuntu' from mass false drops."""
    import getpass

    user = getpass.getuser()
    prose = _trajectory("echo status", f"test {user} passed all checks for {user} mode")
    assert host_escape_findings(prose) == []
    listing = _trajectory("ls -la", f"drwxr-xr-x@ 3 {user}  staff  96 Jul  1 22:06 .")
    assert host_escape_findings(listing), "ls -l ownership column must still flag"
    home_leak = _trajectory("python3 x.py", f"home is {Path.home()}/secrets")
    assert host_escape_findings(home_leak), "home-path leak must still flag"


def test_own_workspace_tempdir_path_does_not_flag() -> None:
    """macOS workspaces live under /var/folders — an observation echoing the workspace's own
    absolute path (pwd, tracebacks, sqlite errors) is not a host escape and must not be dropped."""
    own_path = _trajectory("pwd", "/var/folders/wy/l2n7bpj15sgb25k6ylm9txkh0000gn/T/envcap-abc123")
    assert host_escape_findings(own_path) == []


def test_generic_flag_keeps_credential_markers() -> None:
    """generic_path_markers=False relaxes only PATH-shaped markers; credential markers
    (.ssh, id_rsa, site-packages, ...) always run — a simulated-filesystem benchmark must not
    silence real key leaks."""
    sim_path = _trajectory("apis.fs.ls()", "saved to ~/documents/out.csv")
    assert host_escape_findings(sim_path, generic_path_markers=False) == []
    key_leak = _trajectory("apis.fs.ls()", "found ~/keys/id_rsa and .ssh/config")
    findings = host_escape_findings(key_leak, generic_path_markers=False)
    assert findings and findings[0].field == "output"


def test_scan_spans_jsonl_honors_marker_policy(tmp_path: Path) -> None:
    """The corpus auditor accepts the same policy flag as capture-time filtering, so a
    benchmark's declared relaxation is auditable rather than silently unenforceable."""
    span = {
        "traceId": "sim1",
        "spanId": "sim1-s",
        "attributes": [
            {"key": "gen_ai.tool.message", "value": {"stringValue": "wrote ~/notes/a.txt"}}
        ],
    }
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps(span) + "\n")
    assert "sim1" in scan_spans_jsonl(path)
    assert scan_spans_jsonl(path, generic_path_markers=False) == {}


def test_bare_root_sweep_is_flagged() -> None:
    """`ls /` with the slash at end-of-command (the common form) is a filesystem-root sweep —
    it must be refused like `ls / <anything>` is, or a lost agent lists the real host root."""
    from environment_capture.hygiene import command_targets_host

    assert command_targets_host("ls /")
    assert command_targets_host("ls -la /")
    assert command_targets_host("du -sh /")
    assert command_targets_host("find / -name products.db")
    assert not command_targets_host("ls ./")
    assert not command_targets_host("ls data/")
    assert not command_targets_host("grep -r pattern src/")


def test_scan_tolerates_non_object_tool_arguments(tmp_path: Path) -> None:
    """Some tool schemas emit scalar/array-shaped arguments; the audit scans them as raw text
    instead of crashing — and still catches a leak inside one."""
    path = tmp_path / "traces.otel.jsonl"
    spans = [
        {
            "traceId": "s1",
            "attributes": [
                {"key": "gen_ai.tool.call.arguments", "value": {"stringValue": "[1, 2]"}}
            ],
        },
        {
            "traceId": "s2",
            "attributes": [
                {
                    "key": "gen_ai.tool.call.arguments",
                    "value": {"stringValue": json.dumps("cat /Users/someone/.zshrc")},
                }
            ],
        },
    ]
    path.write_text("\n".join(json.dumps(s) for s in spans) + "\n")
    flagged = scan_spans_jsonl(path)
    assert set(flagged) == {"s2"}


def test_quoted_redirected_and_indirect_host_paths_are_flagged() -> None:
    """Quoting, input redirection, `${var}` indirection, and relative traversal must not evade
    the command guard — these all reach real host content in an unguarded shell."""
    for command in (
        'cat "/etc/passwd"',
        "cat '/etc/passwd'",
        "cat </etc/hostname",
        'cat "$HOME/.ssh/id_rsa"',
        "cat ${HOME}/.ssh/id_rsa",
        "H=/etc; cat ${H}/passwd",
        "cat ../../../root/.ssh/id_rsa",
    ):
        assert command_targets_host(command), command
        findings = host_escape_findings(_trajectory(command, "whatever"))
        assert findings and findings[0].field == "command", command


def test_env_dumps_and_secret_var_reads_are_flagged() -> None:
    """`env`/`printenv` dumps and reads of credential-shaped variables are refused; setting env
    for one command (`env VAR=value cmd`) stays legitimate."""
    for command in (
        "env | grep -i aws",
        "env",
        "printenv ANTHROPIC_API_KEY",
        "echo $AWS_SECRET_ACCESS_KEY",
        'printf "%s" "$HF_TOKEN"',
        "echo ${ANTHROPIC_API_KEY}",
    ):
        assert command_targets_host(command), command
    for command in (
        "env FOO=bar python3 analyze.py",
        "env -u AWS_SECRET_ACCESS_KEY python3 analyze.py",
        "env -i bash run.sh",
        "ls data/ && cat data/manual.md",
        "grep -r pattern src/",
        "cp results.csv results{,.bak}",
    ):
        assert not command_targets_host(command), command


def test_secret_values_in_observation_are_flagged() -> None:
    """Opaque credential VALUES with no path/filename marker still drop the trajectory, and the
    finding never re-emits the secret itself."""
    for output in (
        "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXk...\n-----END OPENSSH PRIVATE KEY-",
        "aws_access_key_id = AKIAIOSFODNN7EXAMPLE",
        "your key is sk-ant-api03-abcDEF123456_gh-ijkl",
        "token=hf_abcdefghijklmnopqrstuvwxyz012345",
        "GH_TOKEN=github_pat_11ABCDEFG0abcdefghij_KLMNOPqrstuvwxyz0123456789",
        "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "export ANTHROPIC_API_KEY=sk-ant-secretbody",
    ):
        findings = host_escape_findings(_trajectory("cat data/out.txt", output))
        assert findings, f"expected flag for output: {output[:40]}"
        assert findings[0].field == "output"
        assert output not in findings[0].excerpt  # value is redacted, not re-leaked

    # Secret values are flagged even for a benchmark that relaxed the generic path markers.
    key_leak = _trajectory("apis.fs.read()", "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMIexamplekeybody")
    assert host_escape_findings(key_leak, generic_path_markers=False)


def test_ordinary_key_shaped_assignments_are_not_flagged() -> None:
    """Neither lowercase data nor an UPPERCASE config/db assignment with a short, non-secret
    value is a credential dump — only a secret-shaped value (>=16 chars) drops the trajectory."""
    for output in (
        "row primary_key = 5\nother_value = 12",
        "PRIMARY_KEY=1001",
        "BUILD_TOKEN=github_runner",
        "API_KEY=none",
    ):
        assert host_escape_findings(_trajectory("cat data/rows.csv", output)) == [], output
    # ...but a genuine long secret value on such a line is still caught.
    leak = _trajectory("cat data/rows.csv", "GENERIC_API_KEY=A1b2C3d4E5f6G7h8J9k0")
    findings = host_escape_findings(leak)
    assert findings and findings[0].marker == "env-dump-secret"


def test_single_relative_reference_after_cd_is_not_flagged() -> None:
    """A single `../` can legitimately reach a workspace-internal sibling after `cd`; only
    multi-level traversal (which must leave a freshly-rooted workspace) is flagged."""
    assert not command_targets_host("cd data && cat ../manual.md")
    assert not command_targets_host("diff ./actual/out.txt ../expected/out.txt")
    assert command_targets_host("cat ../../../root/.ssh/id_rsa")


def test_scan_names_the_corrupt_line(tmp_path: Path) -> None:
    """A truncated/corrupt line fails the audit LOUDLY with its line number — never a bare
    traceback, and never a silent pass over content that couldn't be screened."""
    import pytest

    path = tmp_path / "traces.otel.jsonl"
    good = json.dumps({"traceId": "ok", "attributes": []})
    path.write_text(good + "\n{truncated\n")
    with pytest.raises(ValueError, match="line 2"):
        scan_spans_jsonl(path)
