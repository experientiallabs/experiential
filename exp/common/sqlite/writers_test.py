"""Fair local writer ordering and finite SQLite lock deadline coverage."""

import sqlite3
import threading
from pathlib import Path

import pytest

from exp.common.sqlite import writers
from exp.common.sqlite.writers import writer_turn


def test_waiting_writer_precedes_a_reentering_worker(tmp_path: Path) -> None:
    """A busy worker cannot reacquire ahead of an older progress/checkpoint writer."""
    path = tmp_path / "content.db"
    order: list[int] = []
    errors: list[BaseException] = []

    def write(index: int) -> None:
        """Record admission order without relying on SQLite's unfair busy polling."""
        try:
            with writer_turn(path, timeout_s=2):
                order.append(index)
        except BaseException as error:  # noqa: BLE001 - report terminal worker errors
            errors.append(error)

    with writer_turn(path, timeout_s=2):
        first = threading.Thread(target=write, args=(1,))
        first.start()
        # Observe membership while the current holder keeps the queue alive.
        queue = writers._queues[path]
        for _ in range(200):
            with queue.lock:
                if len(queue.waiters) == 2:
                    break
            threading.Event().wait(0.005)
        else:
            pytest.fail("older writer did not queue")
    with writer_turn(path, timeout_s=2):
        order.append(2)
    first.join(timeout=2)
    assert not first.is_alive()
    assert errors == []
    assert order == [1, 2]


def test_timed_out_waiter_does_not_block_later_writers(tmp_path: Path) -> None:
    """Timed-out metadata writers leave the queue while the holder remains authoritative."""
    path = tmp_path / "content.db"
    errors: list[sqlite3.OperationalError] = []

    def wait_and_expire() -> None:
        """Report the same SQLITE_BUSY family expected by lease admission."""
        try:
            with writer_turn(path, timeout_s=0.02):
                pytest.fail("writer unexpectedly entered")
        except sqlite3.OperationalError as error:
            errors.append(error)

    with writer_turn(path, timeout_s=2):
        worker = threading.Thread(target=wait_and_expire)
        worker.start()
        worker.join(timeout=1)
        assert not worker.is_alive()
    assert len(errors) == 1
    assert errors[0].sqlite_errorcode == sqlite3.SQLITE_BUSY
    with writer_turn(path, timeout_s=0):
        pass
