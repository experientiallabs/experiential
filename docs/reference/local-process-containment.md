# Local process containment

`LocalProcessEnvironmentRuntime` runs a customer's JSONL environment program on the machine that
runs EXP. Every session gets a fresh private workspace and a bounded process lifetime. What else
is contained depends on the host, so the runtime detects the host's capabilities once at startup,
reports them, and enforces exactly that report for every session or refuses the session. It
never runs a session with less containment than it reports.

```python
from exp.runtime.environments import (
    LocalProcessEnvironmentRuntime,
    LocalProcessLimits,
    ResourceLimits,
)

runtime = LocalProcessEnvironmentRuntime(
    ("python3", "environment.py"),
    limits=LocalProcessLimits(
        request_timeout_seconds=30,
        cleanup_timeout_seconds=5,
        resources=ResourceLimits(
            cpu_time_seconds=60,
            memory_bytes=1024**3,
            file_size_bytes=64 * 1024**2,
            process_count=32,
        ),
    ),
)
report = runtime.containment          # ContainmentReport, detected once per EXP process
report.refusal                        # None, or why local execution is refused on this host
report.level.guarantees               # enforced ContainmentGuarantee values
report.unavailable                    # every other guarantee, each with its reason
```

## Guarantees

Each `ContainmentGuarantee` is an independent property. A report lists a guarantee only when
the host enforces it for every session; every other guarantee appears in `unavailable` with the
reason.

| Guarantee | Meaning | Linux | Darwin |
| --- | --- | --- | --- |
| `session_isolation` | Own session and process group. | yes | yes |
| `descendant_containment` | Forked, `setsid`, and double-forked descendants stay in a kernel-bounded tree and die with the session. | yes | no: any fork fails cleanup closed |
| `verified_cleanup` | Close succeeds only with kernel evidence that no process remains and the workspace is deleted. | yes | yes |
| `private_workspace` | Fresh owner-only directory that is also the working directory, `HOME`, and `TMPDIR`. | yes | yes |
| `environment_allowlist` | Only `LANG`, `LC_ALL`, `LC_CTYPE`, `PATH`, `TZ` (plus the workspace variables) and the three protocol streams reach the command. | yes | yes |
| `cpu_time_limit`, `memory_limit`, `file_size_limit` | Per-process `RLIMIT_CPU`, `RLIMIT_AS`, `RLIMIT_FSIZE`. | yes | no |
| `process_count_limit` | At most `process_count` processes in the whole session, the command included. | yes | no |
| `no_new_privileges` | Set-user-ID files and file capabilities cannot raise privileges. | yes | no |
| `host_process_isolation` | Processes outside the session cannot be addressed by PID, signaled, traced, or have their environment or memory read. | Landlock ABI 3+ and verified private procfs | no |
| `filesystem_write_confinement` | Writes, creations, renames, and deletions only inside the workspace (and `/dev/null`, `/dev/zero`, `/dev/full`). | Landlock ABI 3+ | no |
| `network_denied` | No IP network, including host loopback, and no abstract Unix sockets outside the session. | network namespace | no |

Explicit non-guarantees on every host: filesystem reads are not confined (files the user can
read stay readable), Unix sockets bound to a filesystem path that the user can reach (for example
a D-Bus session bus or `docker.sock`) remain connectable, `memory_limit` is per process (the
session total is bounded by `process_count × memory_bytes`), and host-process metadata stays
visible through `/proc` when `host_process_isolation` is unavailable. Use a remote
`EnvironmentRuntime` when those matter.

## Linux

Linux requires unprivileged user and PID namespaces and `pidfd_open` (Linux 5.3+). A host without
them refuses local execution with the kernel error and the settings to change
(`kernel.unprivileged_userns_clone`, `user.max_user_namespaces`, or an AppArmor policy that
permits unprivileged user namespaces). Network denial additionally needs network namespaces;
without them `network_denied` is reported unavailable and a caller that requires it is refused.
Write confinement needs Landlock ABI 3 or newer. Host-process isolation additionally requires
a private mount namespace and a procfs mount for the session's PID namespace. The probe checks
that `/proc/self` matches the command's PID and that procfs lists only session processes.
A denied or ineffective mount leaves `host_process_isolation` unavailable; the other proven
guarantees remain available. A run requiring host-process isolation is refused before dispatch.

On the current qualification host, private procfs mounts are denied by host policy. Its
`linux-namespaces-v2` report therefore enforces 12 guarantees and explicitly withholds
`host_process_isolation`; it must not be presented as full host-process isolation.

Each session runs three processes:

- **supervisor**: a stdlib-only program (`containment/linux_gate.py`) started by EXP that unshares
  the user, PID, and network namespaces and then waits for a termination request;
- **init**: PID 1 of the new PID namespace. It is killed if the supervisor dies, reaps orphans,
  and forwards termination to every process inside;
- **command**: starts its own session, sets the resource limits, `no_new_privs`, and the Landlock
  ruleset, then executes the customer argv. When private procfs was verified, it first mounts
  procfs in a private mount namespace; a later mount failure refuses startup.

Termination closes the supervisor's lifeline descriptor. The supervisor sends `SIGTERM` to the
namespace, then `SIGKILL` after a quarter of `cleanup_timeout_seconds`. The kernel kills every
process in a PID namespace when its init exits, and the supervisor can reap init only once the
namespace is empty; it reports that reap as the teardown proof. Sockets live in the session's
network namespace and disappear with its last process. If the proof is missing, or the
supervisor does not exit within the cleanup bound (it is then killed, which still tears the
namespace down), close raises `LocalProcessCleanupError`. If EXP itself dies, the lifeline
closes and the supervisor tears the namespace down; the session's `exp-sandbox-*` workspace
directory is then left under the workspace parent because no EXP process remains to delete it.

Inside the namespaces the command's user ID is unmapped and appears as the overflow ID
(normally 65534) on every host; files it writes are still owned by the invoking user.

Detection runs the same supervisor in probe mode once per process. The probe applies every
mechanism and checks the effect from inside: that its child is PID 2 under init, that a host
loopback listener is unreachable, that a write beside its workspace is denied, and that it can
neither see host PIDs through procfs nor read or signal a host process. Only what the probe saw
enforced is reported. The result is cached for the life of the EXP process.

## Darwin

Darwin has no unprivileged process namespace, so containment is proven rather than enforced. An
exec gate keeps the command blocked until a kqueue fork monitor is attached, cleanup signals only
exact PID-plus-start-time identities, and any observed fork fails cleanup closed. Resource limits,
write confinement, and network denial are not applied and are reported unavailable.

## Requiring guarantees in an evaluation

A sandbox simulation spec can require guarantees before any cell dispatches:

```python
SandboxSettings(
    environment_id="customer-environment",
    environment_sha256=digest,
    required_containment=(ContainmentGuarantee.NETWORK_DENIED, ContainmentGuarantee.VERIFIED_CLEANUP),
)
```

`SandboxSimulator` reads the level from the environment runtime and raises
`SandboxSimulationError` naming the missing guarantees before opening any session. The runtime's
`ContainmentLevel` (protocol `linux-namespaces-v2` or `darwin-kqueue-v1` plus the sorted
guarantees) is recorded in every `SandboxSimulationCellBinding` as `environment_containment`.
It is part of the rollout identity, so a resumed or replayed run never mixes containment levels.
A runtime that reports no containment records none and cannot satisfy any requirement.

## Verifying on a host

```console
uv run python -c "from exp.runtime.environments.containment.host import host_containment; print(host_containment().report.model_dump_json(indent=2))"
uv run pytest -q exp/runtime/environments/containment exp/runtime/environments/local_test.py
```

The Linux tests exercise runaway CPU, memory, and file size, an exponential fork bomb, kill on
timeout across child, `setsid`, and double-fork escapes, supervisor stall and death, environment
and descriptor stripping, host environment reads, writes beside the workspace, network denial,
and namespace teardown. Tests for a guarantee the host does not enforce are skipped with the
reported reason.
