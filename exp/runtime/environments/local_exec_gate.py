"""Tiny exec gate that lets the parent install kernel containment before customer code runs."""

from __future__ import annotations

import os
import sys


def main() -> None:
    """Wait for one parent byte, then replace this process with the requested executable.

    Raises:
        SystemExit: The parent did not provide a command or declined to open the execution gate.
        OSError: Reading the gate or replacing the process fails.
    """
    if len(sys.argv) < 3:
        raise SystemExit(64)
    gate_fd = int(sys.argv[1])
    command = sys.argv[2:]
    if os.read(gate_fd, 1) != b"1":
        raise SystemExit(70)
    os.close(gate_fd)
    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    main()
