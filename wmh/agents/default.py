"""The default pi agent definition shipped by wmh."""

from wmh.harness.doc import HarnessDoc
from wmh.harness.pi_runner import pi_node_baseline


def default_agent(name: str = "default") -> HarnessDoc:
    """Return an independent default-agent document backed by vendored pi."""
    return pi_node_baseline(name)
