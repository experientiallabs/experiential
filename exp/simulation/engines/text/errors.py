"""Public failure types for immutable text-world-model simulation."""


class SimulationConfigurationError(ValueError):
    """A sparse simulation recipe cannot be executed against supplied local bindings."""


class SimulationResumeError(RuntimeError):
    """An immutable simulation artifact cannot safely be resumed or reused."""


class SimulationContentionError(SimulationResumeError):
    """Another live runner owns paid work, so this run may be retried without artifacts."""
