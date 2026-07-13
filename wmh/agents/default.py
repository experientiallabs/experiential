"""The default pi agent definition shipped by wmh."""

from wmh.harness.doc import RUNTIME_KIND_ID, HarnessDoc, Surface, SurfaceKind
from wmh.harness.pi_vendor import pi_agent_code_surfaces


def default_agent(name: str = "default") -> HarnessDoc:
    """Return an independent default-agent document backed by vendored pi."""
    base = HarnessDoc.baseline(name)
    return HarnessDoc(
        name=name,
        surfaces=[
            *base.surfaces,
            Surface(id=RUNTIME_KIND_ID, kind=SurfaceKind.PARAM, content="pi-node"),
            *pi_agent_code_surfaces(),
        ],
    )
