"""Canonical representative-task mining from normalized production traces."""

from wmo.simulation.mining.cleanup import (
    InstructionCleanupModel,
    InstructionCleanupResult,
    clean_instruction,
)
from wmo.simulation.mining.coverage import CoverageReport
from wmo.simulation.mining.descriptors import (
    DescriptorEmbedder,
    HashingDescriptorEmbedder,
    RoutingDescriptor,
    coverage_descriptor,
    routing_descriptor,
)
from wmo.simulation.mining.lineage import LineageAssignment, assign_source_lineages
from wmo.simulation.mining.service import MiningSpec, TaskMiningResult, mine_tasks, persist_task_set

__all__ = [
    "CoverageReport",
    "DescriptorEmbedder",
    "HashingDescriptorEmbedder",
    "InstructionCleanupModel",
    "InstructionCleanupResult",
    "LineageAssignment",
    "MiningSpec",
    "RoutingDescriptor",
    "TaskMiningResult",
    "assign_source_lineages",
    "clean_instruction",
    "coverage_descriptor",
    "mine_tasks",
    "persist_task_set",
    "routing_descriptor",
]
