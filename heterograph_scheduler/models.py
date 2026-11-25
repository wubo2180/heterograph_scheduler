"""Core workload and resource contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TaskKind(str, Enum):
    NEIGHBOR_AGGREGATION = "neighbor_aggregation"
    SPARSE_MM = "sparse_mm"
    SUBGRAPH_SAMPLING = "subgraph_sampling"


class DeviceKind(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    FPGA = "fpga"


@dataclass(frozen=True)
class GraphTask:
    task_id: str
    kind: TaskKind
    num_nodes: int
    num_edges: int
    feature_dim: int
    edges: tuple[tuple[int, int], ...] = ()
    node_features: tuple[tuple[float, ...], ...] = ()
    priority: float = 1.0
    degree_skew: float = 1.0
    communication_bytes: int | None = None

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id must not be empty")
        if self.num_nodes < 0 or self.num_edges < 0 or self.feature_dim < 0:
            raise ValueError("graph dimensions must be non-negative")
        if self.priority <= 0:
            raise ValueError("priority must be greater than zero")
        if self.degree_skew < 1:
            raise ValueError("degree_skew must be at least 1")
        if self.edges and len(self.edges) != self.num_edges:
            raise ValueError("num_edges must match the supplied edges")
        for source, target in self.edges:
            if not (0 <= source < self.num_nodes and 0 <= target < self.num_nodes):
                raise ValueError(f"edge ({source}, {target}) is outside the graph")
        if self.node_features:
            if len(self.node_features) != self.num_nodes:
                raise ValueError("node_features must contain one row per node")
            if any(len(row) != self.feature_dim for row in self.node_features):
                raise ValueError("node feature rows must match feature_dim")
        if self.communication_bytes is not None and self.communication_bytes < 0:
            raise ValueError("communication_bytes must be non-negative")

    @property
    def estimated_flops(self) -> float:
        if self.kind is TaskKind.SUBGRAPH_SAMPLING:
            return float(self.num_edges * 4)
        return float(self.num_edges * max(self.feature_dim, 1) * 2)

    @property
    def estimated_memory_bytes(self) -> int:
        feature_bytes = self.num_nodes * self.feature_dim * 4
        edge_bytes = self.num_edges * 2 * 8
        return feature_bytes + edge_bytes

    @property
    def estimated_communication_bytes(self) -> int:
        if self.communication_bytes is not None:
            return self.communication_bytes
        return self.num_edges * max(self.feature_dim, 1) * 4


@dataclass
class ResourceProfile:
    device: DeviceKind
    compute_gflops: float
    bandwidth_gbps: float
    memory_bytes: int
    available: bool = True
    queue_depth: int = 0
    launch_overhead_ms: float = 0.0
    name: str = ""

    def __post_init__(self) -> None:
        if self.compute_gflops <= 0 or self.bandwidth_gbps <= 0:
            raise ValueError("compute and bandwidth rates must be positive")
        if self.memory_bytes < 0 or self.queue_depth < 0:
            raise ValueError("memory and queue depth must be non-negative")
        if self.launch_overhead_ms < 0:
            raise ValueError("launch_overhead_ms must be non-negative")
        if not self.name:
            self.name = self.device.value

    def estimated_latency_ms(self, task: GraphTask) -> float:
        if not self.available or task.estimated_memory_bytes > self.memory_bytes:
            return float("inf")
        compute_ms = task.estimated_flops / (self.compute_gflops * 1_000_000)
        transfer_ms = (
            task.estimated_communication_bytes / (self.bandwidth_gbps * 1_000_000)
        )
        queue_penalty = 1.0 + self.queue_depth * 0.2
        return self.launch_overhead_ms + (compute_ms + transfer_ms) * queue_penalty


@dataclass(frozen=True)
class ScheduleDecision:
    task_id: str
    device: DeviceKind
    resource_name: str
    estimated_latency_ms: float
    scores: dict[str, float] = field(default_factory=dict)
