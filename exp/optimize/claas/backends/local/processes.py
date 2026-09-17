"""Find and terminate live descendants across all groups in one owned POSIX session."""

import os
import signal
import subprocess
import time

import psutil


class OwnedSession:
    """Retain leader identity while finding reparented descendants by their session ID."""

    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        """Capture the fresh child's identity before its parent can reap it."""
        self.identity = psutil.Process(process.pid)
        self.created_at = self.identity.create_time()
        self.session_id = process.pid
        if os.getsid(process.pid) != self.session_id:
            raise RuntimeError("learner process did not enter its owned session")

    def live(self) -> list[psutil.Process]:
        """Reject leader PID reuse and return live processes still belonging to this session."""
        self._leader()
        members: list[psutil.Process] = []
        for candidate in psutil.process_iter():
            try:
                # Reconstruct instead of trusting process_iter's cached PID identity.
                member = psutil.Process(candidate.pid)
                if os.getsid(member.pid) != self.session_id:
                    continue
                if member.status() != psutil.STATUS_ZOMBIE and member.is_running():
                    members.append(member)
            except (ProcessLookupError, psutil.NoSuchProcess):
                continue
            except PermissionError:
                # Foreign sessions may be inaccessible; our own UID's session must be visible.
                if candidate.uids().real == os.getuid():
                    raise
        self._leader()
        return members

    def _leader(self) -> psutil.Process:
        """Require the unreaped leader to reserve the owned session ID throughout cleanup."""
        try:
            leader = psutil.Process(self.session_id)
            if leader.create_time() != self.created_at:
                raise RuntimeError("learner session leader PID was reused; cleanup is uncertain")
            # A session leader cannot call setsid again. Its retained identity
            # reserves this SID even during exit, when macOS getsid returns ESRCH.
            if not leader.is_running():
                raise RuntimeError("learner session leader disappeared; cleanup is uncertain")
            return leader
        except (ProcessLookupError, psutil.NoSuchProcess) as error:
            raise RuntimeError(
                "learner session leader disappeared; cleanup is uncertain"
            ) from error

    def run(self, process: subprocess.Popen[bytes], payload: bytes, timeout_seconds: float) -> None:
        """Deliver bounded stdin and observe exit without reaping the session identity anchor."""
        assert process.stdin is not None
        deadline = time.monotonic() + timeout_seconds
        os.set_blocking(process.stdin.fileno(), False)
        pending = memoryview(payload)
        try:
            while pending:
                self._pause(deadline, process, timeout_seconds)
                try:
                    pending = pending[os.write(process.stdin.fileno(), pending) :]
                except BlockingIOError:
                    continue
                except BrokenPipeError:
                    break
        finally:
            process.stdin.close()
        while self._leader().status() != psutil.STATUS_ZOMBIE:
            self._pause(deadline, process, timeout_seconds)

    @staticmethod
    def _pause(deadline: float, process: subprocess.Popen[bytes], timeout: float) -> None:
        """Bound polling and stdin backpressure by the same absolute run deadline."""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout)
        time.sleep(min(0.01, remaining))

    def signal(self, number: int) -> None:
        """Signal current members through psutil's process-identity/reuse checks."""
        for member in self.live():
            try:
                self._leader()
                if os.getsid(member.pid) == self.session_id:
                    member.send_signal(number)
            except (ProcessLookupError, psutil.NoSuchProcess):
                continue

    def kill_leader(self) -> None:
        """Kill only the retained direct child when broader descendant cleanup is uncertain."""
        leader = self._leader()
        if leader.status() != psutil.STATUS_ZOMBIE:
            try:
                leader.send_signal(signal.SIGKILL)
            except psutil.NoSuchProcess:
                self._leader()  # Disappearance of the ownership anchor still fails closed.

    def finish(self, process: subprocess.Popen[bytes], grace_seconds: float) -> None:
        """Allow graceful exit, then repeatedly kill all live owned-session descendants."""
        self.signal(signal.SIGTERM)
        deadline = time.monotonic() + grace_seconds
        while self.live() and time.monotonic() < deadline:
            time.sleep(min(0.025, max(0, deadline - time.monotonic())))
        deadline = time.monotonic() + 30
        while self.live():
            self.signal(signal.SIGKILL)
            if time.monotonic() >= deadline:
                raise TimeoutError("live learner descendants remain; cleanup is uncertain")
            time.sleep(0.01)
        process.wait(timeout=max(0.01, deadline - time.monotonic()))
