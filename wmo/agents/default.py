"""The default pi agent definition shipped by wmo."""

from wmo.harness.doc import (
    MAX_OUTPUT_TOKENS_ID,
    RUNTIME_KIND_ID,
    HarnessDoc,
    Surface,
    SurfaceKind,
)
from wmo.harness.pi_vendor import pi_agent_code_surfaces
from wmo.harness.runtime import DEFAULT_MAX_OUTPUT_TOKENS


def default_agent(name: str = "default") -> HarnessDoc:
    """Return an independent default-agent document backed by vendored pi."""
    base = HarnessDoc.baseline(name)
    return HarnessDoc(
        name=name,
        surfaces=[
            *base.surfaces,
            Surface(
                id=MAX_OUTPUT_TOKENS_ID,
                kind=SurfaceKind.PARAM,
                content=str(DEFAULT_MAX_OUTPUT_TOKENS),
            ),
            Surface(id=RUNTIME_KIND_ID, kind=SurfaceKind.PARAM, content="pi-node"),
            *pi_agent_code_surfaces(),
        ],
    )
