"""Command-line demo and CPU execution entry point."""

from __future__ import annotations

import argparse
import json
import random

from .backends import CPUBackend
from .engine import ExecutionEngine
from .models import DeviceKind, GraphTask, ResourceProfile, TaskKind
from .simulation import cpu_only_estimate, simulate, synthetic_resources


def make_tasks(count: int = 12, seed: int = 7) -> tuple[GraphTask, ...]:
    rng = random.Random(seed)
    tasks = []
    for index in range(count):
        nodes = rng.randint(16, 96)
        edges = tuple(
            (rng.randrange(nodes), rng.randrange(nodes))
            for _ in range(rng.randint(nodes, nodes * 4))
        )
        dimension = rng.choice((8, 16, 32))
        features = tuple(
            tuple(rng.random() for _ in range(dimension)) for _ in range(nodes)
        )
        tasks.append(
            GraphTask(
                task_id=f"graph-task-{index:03d}",
                kind=TaskKind.NEIGHBOR_AGGREGATION,
                num_nodes=nodes,
                num_edges=len(edges),
                feature_dim=dimension,
                edges=edges,
                node_features=features,
                priority=rng.choice((1.0, 1.0, 1.5)),
            )
        )
    return tuple(tasks)


def make_simulation_tasks(count: int = 12, seed: int = 19) -> tuple[GraphTask, ...]:
    """Create metadata-only workloads large enough to expose device trade-offs."""
    rng = random.Random(seed)
    tasks = []
    for index in range(count):
        nodes = rng.randint(20_000, 100_000)
        tasks.append(
            GraphTask(
                task_id=f"sim-task-{index:03d}",
                kind=TaskKind.NEIGHBOR_AGGREGATION,
                num_nodes=nodes,
                num_edges=nodes * rng.randint(4, 12),
                feature_dim=rng.choice((64, 128)),
                priority=rng.choice((1.0, 1.0, 1.5)),
            )
        )
    return tuple(tasks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("heuristic", "dqn"), default="heuristic")
    parser.add_argument("--execute-cpu", action="store_true")
    args = parser.parse_args()
    if args.execute_cpu:
        tasks = make_tasks()
        cpu_profile = ResourceProfile(
            DeviceKind.CPU,
            compute_gflops=80,
            bandwidth_gbps=40,
            memory_bytes=8_000_000_000,
            name="cpu",
        )
        results = ExecutionEngine(
            [cpu_profile], {"cpu": CPUBackend()}
        ).run(tasks[:3])
        output = {
            "mode": "cpu_execution",
            "results": [
                {
                    "task_id": decision.task_id,
                    "device": result.device.value,
                    "estimated_latency_ms": round(decision.estimated_latency_ms, 6),
                    "measured_elapsed_ms": round(result.elapsed_ms, 6),
                }
                for decision, result in results
            ],
        }
    else:
        tasks = make_simulation_tasks()
        resources = synthetic_resources()
        output = simulate(tasks, resources, args.policy)
        output["cpu_only_estimated_latency_ms"] = round(
            cpu_only_estimate(tasks, resources), 6
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
