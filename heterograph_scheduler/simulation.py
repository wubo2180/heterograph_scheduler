"""Synthetic simulation for policy iteration; not a hardware benchmark."""

from __future__ import annotations

from collections.abc import Sequence

from .models import DeviceKind, GraphTask, ResourceProfile
from .scheduler import DQNScheduler, HeuristicScheduler


def synthetic_resources() -> tuple[ResourceProfile, ...]:
    return (
        ResourceProfile(
            DeviceKind.CPU,
            compute_gflops=80,
            bandwidth_gbps=40,
            memory_bytes=64_000_000_000,
            launch_overhead_ms=0.01,
            name="cpu",
        ),
        ResourceProfile(
            DeviceKind.GPU,
            compute_gflops=8000,
            bandwidth_gbps=1200,
            memory_bytes=40_000_000_000,
            launch_overhead_ms=0.2,
            name="gpu",
        ),
        ResourceProfile(
            DeviceKind.FPGA,
            compute_gflops=1200,
            bandwidth_gbps=200,
            memory_bytes=16_000_000_000,
            launch_overhead_ms=0.1,
            name="fpga",
        ),
    )


def simulate(
    tasks: Sequence[GraphTask],
    resources: Sequence[ResourceProfile],
    policy: str = "heuristic",
) -> dict[str, object]:
    if policy not in {"heuristic", "dqn"}:
        raise ValueError("policy must be 'heuristic' or 'dqn'")
    heuristic = HeuristicScheduler()
    dqn = DQNScheduler(len(resources)) if policy == "dqn" else None
    records: list[dict[str, object]] = []
    total_latency = 0.0
    for task in tasks:
        if dqn is None:
            decision = heuristic.schedule(task, resources)
            selected_index = next(
                index
                for index, resource in enumerate(resources)
                if resource.device == decision.device
            )
        else:
            valid = [
                index
                for index, resource in enumerate(resources)
                if resource.available
                and task.estimated_memory_bytes <= resource.memory_bytes
            ]
            if not valid:
                raise RuntimeError(f"no available resource can run task {task.task_id}")
            state = dqn.encode_state(task, resources)
            selected_index = dqn.select_action(state, valid)
            resource = resources[selected_index]
            latency = resource.estimated_latency_ms(task)
            decision = heuristic.schedule(task, resources)
            reward = (
                decision.estimated_latency_ms - latency
            ) / max(decision.estimated_latency_ms, 1e-9)
            dqn.observe(state, selected_index, reward, state, done=True)
            decision = type(decision)(
                task_id=task.task_id,
                device=resource.device,
                resource_name=resource.name,
                estimated_latency_ms=latency,
                scores={
                    candidate.name: candidate.estimated_latency_ms(task)
                    for candidate in resources
                    if candidate.available
                },
            )
        total_latency += decision.estimated_latency_ms
        records.append(
            {
                "task_id": task.task_id,
                "device": decision.device.value,
                "estimated_latency_ms": round(decision.estimated_latency_ms, 6),
            }
        )
    return {
        "mode": "synthetic_simulation",
        "policy": policy,
        "task_count": len(tasks),
        "estimated_total_latency_ms": round(total_latency, 6),
        "assignments": records,
    }


def cpu_only_estimate(
    tasks: Sequence[GraphTask], resources: Sequence[ResourceProfile]
) -> float:
    cpu = next(resource for resource in resources if resource.device is DeviceKind.CPU)
    return sum(cpu.estimated_latency_ms(task) for task in tasks)
