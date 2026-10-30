"""Linux namespace containment tests: detection, enforced limits, kill-on-timeout and cleanup.

Integration tests drive the public ``LocalProcessEnvironmentRuntime`` on this host and skip with
the detected refusal or unavailable reason when the host cannot enforce what they exercise.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.common.models import ToolCall
from exp.common.rollouts.containment import ContainmentGuarantee
from exp.common.tasks import TaskCase
from exp.runtime.environments.containment.contract import (
    LocalProcessCleanupError,
    LocalProcessCrashError,
    ProcessLaunch,
    ResourceLimits,
)
from exp.runtime.environments.containment.linux import (
    LINUX_CONTAINMENT_PROTOCOL,
    detect_linux_containment,
    linux_containment_from_probe,
)
from exp.runtime.environments.interface import EnvironmentSession
from exp.runtime.environments.local import LocalProcessEnvironmentRuntime, LocalProcessLimits

_ON_LINUX = sys.platform.startswith("linux")
_REPORT = detect_linux_containment().report if _ON_LINUX else None
_REFUSAL = "not a Linux host" if _REPORT is None else _REPORT.refusal
_CONTAINED = pytest.mark.skipif(
    _REFUSAL is not None,
    reason=f"this host refuses Linux local process containment: {_REFUSAL}",
)


def _requires(guarantee: ContainmentGuarantee) -> pytest.MarkDecorator:
    """Skip with the detected reason unless this host enforces one guarantee."""
    if _REPORT is None or _REPORT.refusal is not None:
        return pytest.mark.skipif(True, reason=f"Linux containment refused: {_REFUSAL}")
    reasons = {item.guarantee: item.reason for item in _REPORT.unavailable}
    return pytest.mark.skipif(
        guarantee in reasons,
        reason=f"{guarantee.value} is unavailable on this host: {reasons.get(guarantee)}",
    )


def _probe_facts(
    *,
    command: dict[str, JsonValue] | None = None,
    **facts: JsonValue,
) -> JsonObject:
    """Return probe facts in which every mechanism worked, with explicit overrides."""
    observed: dict[str, JsonValue] = {
        "contained": True,
        "no_new_privs": True,
        "pid": 2,
        "parent": 1,
        "network_reachable": False,
        "writes_confined": True,
        "host_environment_readable": False,
        "procfs_private": True,
        "host_signalable": False,
        "process_limit_enforced": True,
    }
    observed.update(command or {})
    complete: dict[str, JsonValue] = {
        "landlock_abi": 6,
        "network_namespace": True,
        "pidfd": True,
        "init_verdict": "ready",
        "command": observed,
    }
    complete.update(facts)
    return complete


def test_full_probe_enforces_every_guarantee_with_its_mechanisms() -> None:
    """A probe that observed every mechanism working yields the complete Linux level."""
    backend = linux_containment_from_probe(_probe_facts())

    assert backend.report.refusal is None
    assert backend.report.level.protocol == LINUX_CONTAINMENT_PROTOCOL
    assert set(backend.report.level.guarantees) == set(ContainmentGuarantee)
    assert backend.report.unavailable == ()
    assert {"pid_namespace", "network_namespace", "landlock_abi_6"} <= set(
        backend.report.mechanisms
    )
    assert backend.deny_network and backend.landlock_abi == 6


def test_unenforced_process_limit_withholds_the_process_count_guarantee() -> None:
    """A probe that forked past RLIMIT_NPROC, as real root does, never claims the limit."""
    backend = linux_containment_from_probe(_probe_facts(command={"process_limit_enforced": False}))

    reasons = {item.guarantee: item.reason for item in backend.report.unavailable}
    assert backend.report.refusal is None
    assert ContainmentGuarantee.PROCESS_COUNT_LIMIT not in backend.report.level.guarantees
    assert "RLIMIT_NPROC" in reasons[ContainmentGuarantee.PROCESS_COUNT_LIMIT]


def test_namespace_refusal_reports_no_guarantees_and_how_to_fix_it() -> None:
    """Hosts without unprivileged namespaces refuse local execution with actionable advice."""
    backend = linux_containment_from_probe(
        {
            "landlock_abi": 6,
            "network_namespace": False,
            "namespace_error": "Operation not permitted",
        }
    )

    refusal = backend.report.refusal
    assert refusal is not None
    assert "Operation not permitted" in refusal and "unprivileged_userns_clone" in refusal
    assert backend.report.level.guarantees == ()
    assert {item.guarantee for item in backend.report.unavailable} == set(ContainmentGuarantee)
    with pytest.raises(LocalProcessCleanupError, match="unprivileged user and PID namespaces"):
        backend.launch(_launch(Path("/nonexistent-exp-workspace")))


def test_missing_network_namespace_never_claims_network_denial() -> None:
    """Without a network namespace the report names the gap instead of claiming denial."""
    facts = _probe_facts(
        command={"network_reachable": True},
        network_namespace=False,
        network_namespace_error="Operation not permitted",
    )

    backend = linux_containment_from_probe(facts)

    assert not backend.deny_network
    assert ContainmentGuarantee.NETWORK_DENIED not in backend.report.level.guarantees
    reasons = {item.guarantee: item.reason for item in backend.report.unavailable}
    assert "network namespaces are unavailable" in reasons[ContainmentGuarantee.NETWORK_DENIED]
    assert "network_namespace" not in backend.report.mechanisms


def test_old_landlock_abi_does_not_claim_write_confinement_or_host_isolation() -> None:
    """Landlock before ABI 3 cannot restrict truncation, so neither guarantee is claimed."""
    facts = _probe_facts(landlock_abi=2)

    backend = linux_containment_from_probe(facts)

    assert backend.landlock_abi == 0
    reasons = {item.guarantee: item.reason for item in backend.report.unavailable}
    assert "Landlock ABI 3" in reasons[ContainmentGuarantee.FILESYSTEM_WRITE_CONFINEMENT]
    assert "Landlock ABI 3" in reasons[ContainmentGuarantee.HOST_PROCESS_ISOLATION]


def test_probe_that_observes_a_leak_withholds_the_guarantee() -> None:
    """An enforced mechanism that the probe still sees through is never reported as enforced."""
    facts = _probe_facts(command={"host_environment_readable": True})

    backend = linux_containment_from_probe(facts)

    assert ContainmentGuarantee.HOST_PROCESS_ISOLATION not in backend.report.level.guarantees
    assert ContainmentGuarantee.FILESYSTEM_WRITE_CONFINEMENT in backend.report.level.guarantees


def test_host_procfs_visibility_withholds_host_process_isolation() -> None:
    """A host that refuses private procfs preserves only independently proven guarantees."""
    backend = linux_containment_from_probe(
        _probe_facts(command={"procfs_private": False, "procfs_error": "Permission denied"})
    )

    assert backend.report.refusal is None
    assert ContainmentGuarantee.HOST_PROCESS_ISOLATION not in backend.report.level.guarantees
    assert ContainmentGuarantee.VERIFIED_CLEANUP in backend.report.level.guarantees
    reason = next(
        item.reason
        for item in backend.report.unavailable
        if item.guarantee == ContainmentGuarantee.HOST_PROCESS_ISOLATION
    )
    assert "private procfs" in reason and "Permission denied" in reason
    assert not backend.private_procfs


def test_failed_probe_refuses_local_execution() -> None:
    """A probe that cannot place its child in a PID namespace refuses instead of degrading."""
    facts = _probe_facts(command={"pid": 4242})

    assert linux_containment_from_probe(facts).report.refusal is not None
    assert linux_containment_from_probe({"probe_error": "TimeoutExpired"}).report.refusal


@pytest.mark.skipif(not _ON_LINUX, reason="Linux capability detection runs only on Linux")
def test_host_detection_accounts_for_every_guarantee() -> None:
    """The startup report on this host explains every guarantee it does not enforce."""
    assert _REPORT is not None
    accounted = set(_REPORT.level.guarantees) | {item.guarantee for item in _REPORT.unavailable}
    assert accounted == set(ContainmentGuarantee)
    assert detect_linux_containment() is detect_linux_containment()


@_CONTAINED
def test_runaway_cpu_is_killed_at_its_cpu_limit(tmp_path: Path) -> None:
    """A busy loop dies at its CPU-second limit long before the session timeout."""
    runtime = _runtime(tmp_path, resources=ResourceLimits(cpu_time_seconds=1))
    started = time.monotonic()

    with runtime.open(_task()) as session:
        with pytest.raises(LocalProcessCrashError, match=r"return code (137|152)"):
            session.execute(_call("spin"))

    assert time.monotonic() - started < 8.0
    assert list(tmp_path.glob("exp-sandbox-*")) == []


@_CONTAINED
def test_runaway_memory_is_refused_at_its_address_space_limit(tmp_path: Path) -> None:
    """An allocation beyond the memory limit fails inside the command instead of the host."""
    runtime = _runtime(tmp_path, resources=ResourceLimits(memory_bytes=512 * 1024**2))

    with runtime.open(_task()) as session:
        observation = session.execute(_call("allocate", megabytes=2048))

    assert observation.content == "MemoryError"


@_CONTAINED
def test_file_size_limit_stops_an_oversized_write(tmp_path: Path) -> None:
    """A write beyond the file-size limit fails with EFBIG and leaves the limit intact."""
    runtime = _runtime(tmp_path, resources=ResourceLimits(file_size_bytes=1024**2))

    with runtime.open(_task()) as session:
        observation = session.execute(_call("write-large", megabytes=4))

    assert observation.content == "EFBIG"
    assert observation.metadata["size"] == 1024**2


@_requires(ContainmentGuarantee.PROCESS_COUNT_LIMIT)
def test_process_count_limit_bounds_the_whole_session(tmp_path: Path) -> None:
    """The command plus its descendants never exceed the configured process count."""
    runtime = _runtime(tmp_path, resources=ResourceLimits(process_count=8))

    with runtime.open(_task()) as session:
        observation = session.execute(_call("fork-count", marker=_marker()))

    assert observation.metadata["children"] == 7


@_requires(ContainmentGuarantee.PROCESS_COUNT_LIMIT)
def test_fork_bomb_is_bounded_and_fully_cleaned(tmp_path: Path) -> None:
    """An exponential fork bomb is held to the process limit and leaves nothing behind."""
    marker = _marker()
    limits = LocalProcessLimits(
        request_timeout_seconds=5.0,
        cleanup_timeout_seconds=2.0,
        resources=ResourceLimits(process_count=32),
    )
    runtime = _runtime(tmp_path, limits=limits)
    started = time.monotonic()

    with runtime.open(_task()) as session:
        observation = session.execute(_call("fork-bomb", marker=marker))
        assert observation.content == "detonated"
        _wait_until(lambda: len(_marked_pids(marker)) >= 16)
        assert len(_marked_pids(marker)) <= 32
        namespaces = _session_namespaces(session)

    assert time.monotonic() - started < 10.0
    assert _marked_pids(marker) == ()
    assert _namespace_members(namespaces) == ()


@_CONTAINED
def test_kill_on_timeout_leaves_no_process_from_any_escape(tmp_path: Path) -> None:
    """Children, setsid daemons and double-forked orphans all die with a timed-out session."""
    marker = _marker()
    limits = LocalProcessLimits(
        request_timeout_seconds=0.5,
        session_timeout_seconds=5.0,
        cleanup_timeout_seconds=2.0,
    )
    runtime = _runtime(tmp_path, limits=limits)

    with runtime.open(_task()) as session:
        session.execute(_call("spawn-tree", marker=marker))
        _wait_until(lambda: len(_marked_pids(marker)) == 3)
        namespaces = _session_namespaces(session)
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="response exceeded"):
            session.execute(_call("sleep"))
        assert time.monotonic() - started < 0.5 + 2.0 + 0.5
        assert _marked_pids(marker) == ()

    assert _namespace_members(namespaces) == ()
    assert list(tmp_path.glob("exp-sandbox-*")) == []


@_CONTAINED
def test_unresponsive_supervisor_is_killed_within_the_bound_and_fails_closed(
    tmp_path: Path,
) -> None:
    """A supervisor that never reports teardown is killed on time and cleanup is unproven."""
    runtime = _runtime(tmp_path, limits=LocalProcessLimits(cleanup_timeout_seconds=0.5))

    with pytest.raises(LocalProcessCleanupError, match="containment is unproven"):
        with runtime.open(_task()) as session:
            os.kill(_supervisor(session).pid, signal.SIGSTOP)
            started = time.monotonic()

    assert time.monotonic() - started < 0.5 + 1.5
    assert list(tmp_path.glob("exp-sandbox-*")) == []


@_CONTAINED
def test_supervisor_death_tears_down_the_namespace_and_fails_cleanup_closed(
    tmp_path: Path,
) -> None:
    """If the supervisor dies, as when EXP is killed, the kernel still kills every process."""
    marker = _marker()
    runtime = _runtime(tmp_path)

    with pytest.raises(LocalProcessCleanupError, match="without namespace teardown evidence"):
        with runtime.open(_task()) as session:
            session.execute(_call("spawn-tree", marker=marker))
            _wait_until(lambda: len(_marked_pids(marker)) == 3)
            namespaces = _session_namespaces(session)
            supervisor = _supervisor(session)
            os.kill(supervisor.pid, signal.SIGKILL)
            supervisor.wait(timeout=2.0)
            started = time.monotonic()
            while (_marked_pids(marker) or _namespace_members(namespaces)) and (
                time.monotonic() - started < 2.0
            ):
                time.sleep(0.01)
            leftovers = _marked_pids(marker) + _namespace_members(namespaces)

    assert leftovers == ()
    assert list(tmp_path.glob("exp-sandbox-*")) == []


@_CONTAINED
def test_abrupt_exp_exit_still_kills_every_session_process(tmp_path: Path) -> None:
    """When the owning EXP process dies without cleanup, the lifeline ends the session."""
    marker = _marker()
    fixture = tmp_path / "containment_fixture.py"
    fixture.write_text(_FIXTURE_SOURCE, encoding="utf-8")
    owner = tmp_path / "owner.py"
    owner.write_text(_ABRUPT_OWNER_SOURCE, encoding="utf-8")

    completed = subprocess.run(
        (sys.executable, str(owner), str(fixture), str(tmp_path), marker),
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    assert completed.stdout.strip() == b"3"
    _wait_until(lambda: not _marked_pids(marker))


@_CONTAINED
def test_environment_descriptors_and_credentials_do_not_cross(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only allowlisted variables and the three protocol streams reach the command."""
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-cross")
    inherited = os.open(os.devnull, os.O_RDONLY)
    os.set_inheritable(inherited, True)
    try:
        runtime = _runtime(tmp_path)
        with runtime.open(_task()) as session:
            environment = session.execute(_call("environment"))
            descriptors = session.execute(_call("descriptors"))
    finally:
        os.close(inherited)

    assert environment.metadata["keys"] == [
        "EXP_SANDBOX_WORKSPACE",
        "HOME",
        "LANG",
        "LC_ALL",
        "PATH",
        "TMPDIR",
        "TZ",
    ]
    assert descriptors.metadata["descriptors"] == [0, 1, 2]


@_CONTAINED
def test_procfs_visibility_matches_the_reported_guarantee(tmp_path: Path) -> None:
    """Host PID visibility is permitted only when the report explicitly withholds isolation."""
    runtime = _runtime(tmp_path)

    with runtime.open(_task()) as session:
        observation = session.execute(_call("proc-view"))

    assert observation.metadata["self_pid"] == 2
    if ContainmentGuarantee.HOST_PROCESS_ISOLATION in runtime.containment.level.guarantees:
        assert observation.metadata["pids"] == [1, 2]
        assert observation.metadata["proc_self"] == "2"
    else:
        assert any(
            item.guarantee == ContainmentGuarantee.HOST_PROCESS_ISOLATION
            for item in runtime.containment.unavailable
        )


@_requires(ContainmentGuarantee.HOST_PROCESS_ISOLATION)
def test_host_process_environment_is_unreadable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The command cannot read EXP's environment, where provider credentials live."""
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-cross")
    runtime = _runtime(tmp_path)

    with runtime.open(_task()) as session:
        observation = session.execute(_call("read", path=f"/proc/{os.getpid()}/environ"))

    assert observation.content == "FileNotFoundError"


@_requires(ContainmentGuarantee.FILESYSTEM_WRITE_CONFINEMENT)
def test_writes_outside_the_private_workspace_are_denied(tmp_path: Path) -> None:
    """Files can be written in the workspace but never beside it."""
    runtime = _runtime(tmp_path)
    outside = tmp_path / "escaped.txt"

    with runtime.open(_task()) as session:
        inside = session.execute(_call("write", path="inside.txt"))
        escaped = session.execute(_call("write", path=str(outside)))

    assert inside.content == "written"
    assert escaped.content == "PermissionError"
    assert not outside.exists()


@_requires(ContainmentGuarantee.FILESYSTEM_WRITE_CONFINEMENT)
def test_write_confinement_claims_no_metadata_confinement_landlock_cannot_enforce(
    tmp_path: Path,
) -> None:
    """Landlock has no right for chmod, chown, utimes or xattrs, so the claim must exclude them.

    A mode change beside the workspace still succeeds under the guarantee, and the guarantee's
    documented scope names file metadata as not confined.
    """
    runtime = _runtime(tmp_path)
    outside = tmp_path / "host-file.txt"
    outside.write_text("host", encoding="utf-8")
    outside.chmod(0o644)

    with runtime.open(_task()) as session:
        changed = session.execute(_call("chmod", path=str(outside)))

    assert changed.content == "changed"
    assert outside.stat().st_mode & 0o777 == 0o600
    scope = " ".join(str(ContainmentGuarantee.__doc__).split())
    assert "Metadata of existing files (mode, owner, timestamps, extended attributes) is not" in (
        scope
    )


@_requires(ContainmentGuarantee.NETWORK_DENIED)
def test_network_is_denied_including_host_loopback(tmp_path: Path) -> None:
    """Neither a host loopback listener nor an unrouted address is reachable."""
    runtime = _runtime(tmp_path)
    with socket.create_server(("127.0.0.1", 0)) as listener:
        port = listener.getsockname()[1]
        with runtime.open(_task()) as session:
            loopback = session.execute(_call("connect", host="127.0.0.1", port=port))
            external = session.execute(_call("connect", host="192.0.2.1", port=80))

    assert loopback.content == "unreachable"
    assert external.content == "unreachable"


@_CONTAINED
def test_cleanup_proves_no_process_namespace_or_listening_port_remains(tmp_path: Path) -> None:
    """After close no process holds the session namespaces, so its sockets are gone too."""
    runtime = _runtime(tmp_path)

    with runtime.open(_task()) as session:
        listening = session.execute(_call("listen"))
        assert listening.content == "listening"
        namespaces = _session_namespaces(session)
        assert _namespace_members(namespaces)
        workspace = Path(str(session.execute(_call("workspace")).metadata["workspace"]))

    assert _namespace_members(namespaces) == ()
    assert not workspace.exists()
    port = listening.metadata["port"]
    assert isinstance(port, int)
    with socket.create_server(("127.0.0.1", port)):
        pass


@_CONTAINED
def test_forked_descendants_are_contained_rather_than_refused(tmp_path: Path) -> None:
    """Unlike Darwin, an orphaned grandchild is killed and cleanup still succeeds."""
    marker = _marker()
    runtime = _runtime(tmp_path)

    with runtime.open(_task()) as session:
        session.execute(_call("spawn-tree", marker=marker))
        _wait_until(lambda: len(_marked_pids(marker)) == 3)

    assert _marked_pids(marker) == ()


@_CONTAINED
def test_missing_command_is_a_start_crash_without_leftovers(tmp_path: Path) -> None:
    """A command that cannot execute fails as a typed crash and removes its workspace."""
    runtime = LocalProcessEnvironmentRuntime(
        (str(tmp_path / "missing-executable"),),
        workspace_parent=tmp_path,
    )

    with pytest.raises(LocalProcessCrashError, match="could not start: FileNotFoundError"):
        with runtime.open(_task()):
            pass

    assert list(tmp_path.glob("exp-sandbox-*")) == []


def _launch(workspace: Path) -> ProcessLaunch:
    """Return one launch request for backend-level refusal checks."""
    return ProcessLaunch(
        command=("true",),
        workspace=workspace,
        environment={},
        cleanup_timeout_seconds=1.0,
        startup_timeout_seconds=1.0,
        resources=ResourceLimits(),
    )


def _runtime(
    workspace_parent: Path,
    *,
    limits: LocalProcessLimits | None = None,
    resources: ResourceLimits | None = None,
) -> LocalProcessEnvironmentRuntime:
    """Create the JSONL fixture runtime with optional limits or resource overrides."""
    fixture = workspace_parent / "containment_fixture.py"
    fixture.write_text(_FIXTURE_SOURCE, encoding="utf-8")
    if resources is not None:
        limits = LocalProcessLimits(resources=resources)
    return LocalProcessEnvironmentRuntime(
        (sys.executable, str(fixture)),
        limits=limits,
        workspace_parent=workspace_parent,
    )


def _call(name: str, **arguments: JsonValue) -> ToolCall:
    """Return one fixture tool call."""
    return ToolCall(call_id=f"call-{name}", name=name, arguments=arguments)


def _task() -> TaskCase:
    """Return one task whose canonical payload the fixture ignores."""
    return TaskCase(
        task_id="task-1",
        lineage_group_id="lineage-1",
        partition="fit",
        instruction="Exercise containment.",
        workload_weight=1.0,
        source_trace_ids=("trace-1",),
    )


def _marker() -> str:
    """Return a unique argv token that identifies processes spawned by one test."""
    return f"exp-containment-marker-{uuid.uuid4().hex}"


def _marked_pids(marker: str) -> tuple[int, ...]:
    """Return live, non-zombie host PIDs whose argv carries the marker."""
    return tuple(
        pid for pid in _live_pids() if marker.encode() in _read_or_empty(f"/proc/{pid}/cmdline")
    )


def _live_pids() -> tuple[int, ...]:
    """Return every host PID that is not a zombie awaiting its parent's reap."""
    pids: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        stat = _read_or_empty(f"/proc/{entry.name}/stat").decode(errors="replace")
        if stat and stat[stat.rfind(")") + 2 :].split()[0] != "Z":
            pids.append(int(entry.name))
    return tuple(pids)


def _read_or_empty(path: str) -> bytes:
    """Read a procfs file, treating a process that already exited as empty."""
    try:
        return Path(path).read_bytes()
    except OSError:
        return b""


def _supervisor(session: EnvironmentSession) -> subprocess.Popen[bytes]:
    """Return the supervisor process behind one open local session."""
    process = session.__dict__["_contained"].process
    assert isinstance(process, subprocess.Popen)
    return process


def _session_namespaces(session: EnvironmentSession) -> tuple[str, ...]:
    """Return the user, PID, and network namespace identities of a session's init process."""
    contained = session.__dict__["_contained"]
    init_pid = contained.__dict__["_init_pid"]
    return tuple(os.readlink(f"/proc/{init_pid}/ns/{kind}") for kind in ("user", "pid", "net"))


def _namespace_members(namespaces: tuple[str, ...]) -> tuple[int, ...]:
    """Return every live host PID still inside any of the given namespaces."""
    members: list[int] = []
    for pid in _live_pids():
        for kind, identity in zip(("user", "pid", "net"), namespaces, strict=True):
            try:
                if os.readlink(f"/proc/{pid}/ns/{kind}") == identity:
                    members.append(pid)
                    break
            except OSError:
                continue
    return tuple(members)


def _wait_until(condition: Callable[[], bool], timeout_seconds: float = 5.0) -> None:
    """Poll a zero-argument predicate until it holds or fail the test at the deadline."""
    deadline = time.monotonic() + timeout_seconds
    while not condition():
        if time.monotonic() >= deadline:
            pytest.fail("condition did not hold before the deadline")
        time.sleep(0.02)


_ABRUPT_OWNER_SOURCE = """\
import os
import sys
import time
from pathlib import Path

from exp.common.models import ToolCall
from exp.common.tasks import TaskCase
from exp.runtime.environments.local import LocalProcessEnvironmentRuntime

fixture, parent, marker = sys.argv[1:]
runtime = LocalProcessEnvironmentRuntime((sys.executable, fixture), workspace_parent=Path(parent))
task = TaskCase(
    task_id="task-1",
    lineage_group_id="lineage-1",
    partition="fit",
    instruction="Exercise containment.",
    workload_weight=1.0,
    source_trace_ids=("trace-1",),
)
session = runtime.open(task).__enter__()
session.execute(ToolCall(call_id="spawn", name="spawn-tree", arguments={"marker": marker}))
deadline = time.monotonic() + 5.0
count = 0
while time.monotonic() < deadline:
    count = sum(
        marker.encode() in (entry / "cmdline").read_bytes()
        for entry in Path("/proc").iterdir()
        if entry.name.isdigit()
        and entry.name != str(os.getpid())
        and (entry / "cmdline").exists()
    )
    if count == 3:
        break
    time.sleep(0.02)
sys.stdout.write(str(count))
sys.stdout.flush()
os._exit(0)
"""


_FIXTURE_SOURCE = """\
import errno
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def reply(content, **metadata):
    print(json.dumps({"content": content, "metadata": metadata}), flush=True)


def sleeper(marker, **options):
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)", marker],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **options,
    )


held = []
for line in sys.stdin:
    request = json.loads(line)
    if request["kind"] == "open":
        print(json.dumps({"ready": True}), flush=True)
        continue
    if request["kind"] == "close":
        break
    action = request["action"]
    name = action["name"]
    arguments = action["arguments"]
    if name == "spin":
        while True:
            pass
    if name == "sleep":
        time.sleep(60)
    if name == "allocate":
        try:
            held.append(bytearray(arguments["megabytes"] * 1024 * 1024))
            reply("allocated")
        except MemoryError:
            reply("MemoryError")
        continue
    if name == "write-large":
        path = Path(os.environ["EXP_SANDBOX_WORKSPACE"]) / "large.bin"
        try:
            with path.open("wb") as handle:
                for _ in range(arguments["megabytes"]):
                    handle.write(b"x" * 1024 * 1024)
            reply("written", size=path.stat().st_size)
        except OSError as error:
            reply(errno.errorcode.get(error.errno, "OSError"), size=path.stat().st_size)
        continue
    if name == "fork-count":
        children = []
        while True:
            try:
                children.append(sleeper(arguments["marker"]))
            except OSError:
                break
        reply("counted", children=len(children))
        continue
    if name == "fork-bomb":
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import os, time\\nwhile True:\\n    try:\\n        os.fork()\\n"
                "    except OSError:\\n        time.sleep(0.01)",
                arguments["marker"],
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        reply("detonated")
        continue
    if name == "spawn-tree":
        held.append(sleeper(arguments["marker"]))
        held.append(sleeper(arguments["marker"], start_new_session=True))
        middle = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import subprocess, sys; subprocess.Popen([sys.executable, '-c', "
                "'import time; time.sleep(120)', sys.argv[1]], start_new_session=True)",
                arguments["marker"],
            ]
        )
        middle.wait()
        reply("spawned")
        continue
    if name == "environment":
        reply("ok", keys=sorted(os.environ))
        continue
    if name == "descriptors":
        listed = [int(fd) for fd in os.listdir("/proc/self/fd")]
        open_now = []
        for fd in listed:
            try:
                os.fstat(fd)
            except OSError:
                continue
            open_now.append(fd)
        reply("ok", descriptors=sorted(open_now))
        continue
    if name == "proc-view":
        reply(
            "ok",
            self_pid=os.getpid(),
            pids=sorted(int(pid) for pid in os.listdir("/proc") if pid.isdigit()),
            proc_self=os.readlink("/proc/self"),
        )
        continue
    if name == "read":
        try:
            Path(arguments["path"]).read_bytes()
            reply("read")
        except OSError as error:
            reply(type(error).__name__)
        continue
    if name == "write":
        try:
            (Path(os.environ["EXP_SANDBOX_WORKSPACE"]) / arguments["path"]).write_text("x")
            reply("written")
        except OSError as error:
            reply(type(error).__name__)
        continue
    if name == "chmod":
        try:
            os.chmod(arguments["path"], 0o600)
            reply("changed")
        except OSError as error:
            reply(type(error).__name__)
        continue
    if name == "connect":
        try:
            socket.create_connection((arguments["host"], arguments["port"]), timeout=1.0).close()
            reply("connected")
        except OSError:
            reply("unreachable")
        continue
    if name == "listen":
        server = socket.socket()
        server.bind(("0.0.0.0", 0))
        server.listen(1)
        held.append(server)
        reply("listening", port=server.getsockname()[1])
        continue
    if name == "workspace":
        reply("ok", workspace=os.environ["EXP_SANDBOX_WORKSPACE"])
        continue
    reply("ok")
"""
