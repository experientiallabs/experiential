"""Gateway-specific adapters producing canonical traces for shared ingestion."""

from exp.runtime.gateway.ingest.conversion import load_gateway_capture
from exp.runtime.gateway.ingest.streaming import ingest_gateway_capture

__all__ = ["ingest_gateway_capture", "load_gateway_capture"]
