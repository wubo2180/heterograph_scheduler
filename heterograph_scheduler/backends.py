"""Execution backend interfaces and reference implementations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from .models import DeviceKind, GraphTask, TaskKind


@dataclass(frozen=True)
class ExecutionResult:
    task_id: str
    device: DeviceKind
    elapsed_ms: float
    output: object


class ExecutionBackend(Protocol):
    device: DeviceKind

    def execute(self, task: GraphTask) -> ExecutionResult: ...


def _aggregate_python(task: GraphTask) -> list[list[float]]:
    if not task.node_features:
        raise ValueError("neighbor aggregation requires node_features")
    output = [[0.0] * task.feature_dim for _ in range(task.num_nodes)]
    for source, target in task.edges:
        source_row = task.node_features[source]
        target_row = output[target]
        for column, value in enumerate(source_row):
            target_row[column] += value
    return output


class CPUBackend:
    device = DeviceKind.CPU

    def execute(self, task: GraphTask) -> ExecutionResult:
        started = perf_counter()
        if task.kind is TaskKind.SUBGRAPH_SAMPLING:
            output: object = task.edges[::2]
        else:
            output = _aggregate_python(task)
        elapsed_ms = (perf_counter() - started) * 1000
        return ExecutionResult(task.task_id, self.device, elapsed_ms, output)


class CudaBackend:
    """CUDA neighbor aggregation backend using PyTorch tensor operations."""

    device = DeviceKind.GPU

    def __init__(self) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("CudaBackend requires PyTorch") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CudaBackend requires an available CUDA device")
        self._torch = torch

    def execute(self, task: GraphTask) -> ExecutionResult:
        if task.kind is TaskKind.SUBGRAPH_SAMPLING:
            raise NotImplementedError("CUDA subgraph sampling is not implemented")
        if not task.node_features:
            raise ValueError("CUDA aggregation requires node_features")
        torch = self._torch
        started = perf_counter()
        device = torch.device("cuda")
        features = torch.tensor(task.node_features, dtype=torch.float32, device=device)
        edges = torch.tensor(task.edges, dtype=torch.long, device=device)
        output = torch.zeros_like(features)
        if edges.numel():
            output.index_add_(0, edges[:, 1], features[edges[:, 0]])
        torch.cuda.synchronize(device)
        elapsed_ms = (perf_counter() - started) * 1000
        return ExecutionResult(
            task.task_id, self.device, elapsed_ms, output.cpu().tolist()
        )


class FPGAAdapter:
    """Adapter for a caller-supplied FPGA runtime function."""

    device = DeviceKind.FPGA

    def __init__(self, executor: Callable[[GraphTask], object]) -> None:
        self._executor = executor

    def execute(self, task: GraphTask) -> ExecutionResult:
        started = perf_counter()
        output = self._executor(task)
        elapsed_ms = (perf_counter() - started) * 1000
        return ExecutionResult(task.task_id, self.device, elapsed_ms, output)


def to_pyg_data(task: GraphTask) -> object:
    """Convert a task to PyG Data; importing PyG is deferred until requested."""
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as exc:
        raise RuntimeError(
            "to_pyg_data requires torch and torch-geometric"
        ) from exc
    edge_index = torch.tensor(task.edges, dtype=torch.long).t().contiguous()
    if not task.edges:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    x = (
        torch.tensor(task.node_features, dtype=torch.float32)
        if task.node_features
        else None
    )
    return Data(x=x, edge_index=edge_index, num_nodes=task.num_nodes)
