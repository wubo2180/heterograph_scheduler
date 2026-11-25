"""Feature extraction for graph workloads."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .models import GraphTask


@dataclass(frozen=True)
class TaskFeatures:
    log_nodes: float
    log_edges: float
    average_degree: float
    degree_skew: float
    feature_dim: float
    compute_per_byte: float
    communication_ratio: float

    def as_vector(self) -> tuple[float, ...]:
        return (
            self.log_nodes,
            self.log_edges,
            self.average_degree,
            self.degree_skew,
            self.feature_dim,
            self.compute_per_byte,
            self.communication_ratio,
        )


def extract_task_features(task: GraphTask) -> TaskFeatures:
    degree_skew = task.degree_skew
    if task.edges and task.num_nodes:
        degrees = [0] * task.num_nodes
        for _, target in task.edges:
            degrees[target] += 1
        mean = sum(degrees) / task.num_nodes
        if mean > 0:
            variance = sum((degree - mean) ** 2 for degree in degrees) / task.num_nodes
            degree_skew = max(degree_skew, math.sqrt(variance) / mean + 1.0)

    memory = max(task.estimated_memory_bytes, 1)
    communication = task.estimated_communication_bytes
    return TaskFeatures(
        log_nodes=math.log1p(task.num_nodes),
        log_edges=math.log1p(task.num_edges),
        average_degree=task.num_edges / max(task.num_nodes, 1),
        degree_skew=degree_skew,
        feature_dim=float(task.feature_dim),
        compute_per_byte=task.estimated_flops / memory,
        communication_ratio=communication / memory,
    )
