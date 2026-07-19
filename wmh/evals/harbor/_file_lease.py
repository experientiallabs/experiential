"""Compatibility import for the shared durable-store file lease."""

from wmh.core.file_lease import exclusive_posix_file_lease

__all__ = ["exclusive_posix_file_lease"]
