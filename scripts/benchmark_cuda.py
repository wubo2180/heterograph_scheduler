"""Compare PyTorch neighbor aggregation on CPU and CUDA with real timings."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from typing import Any


def _synchronize(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_ms(
    torch: Any,
    device: Any,
    operation: Callable[[], Any],
    warmup: int,
    repeats: int,
) -> float:
    for _ in range(warmup):
        operation()
    _synchronize(torch, device)

    durations = []
    for _ in range(repeats):
        _synchronize(torch, device)
        started = time.perf_counter()
        operation()
        _synchronize(torch, device)
        durations.append((time.perf_counter() - started) * 1000)
    return statistics.median(durations)


def _aggregate(torch: Any, features: Any, edge_index: Any) -> Any:
    result = torch.zeros_like(features)
    if edge_index.numel():
        result.index_add_(0, edge_index[1], features[edge_index[0]])
    return result


def _benchmark_case(
    torch: Any,
    nodes: int,
    edges: int,
    feature_dim: int,
    warmup: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    features_cpu = torch.randn(
        (nodes, feature_dim), generator=generator, dtype=torch.float32
    )
    edge_index_cpu = torch.randint(
        nodes, (2, edges), generator=generator, dtype=torch.int64
    )
    cpu_operation = lambda: _aggregate(torch, features_cpu, edge_index_cpu)
    cpu_result = cpu_operation()
    cpu_ms = _measure_ms(
        torch, torch.device("cpu"), cpu_operation, warmup, repeats
    )

    cuda_device = torch.device("cuda")
    features_cuda = features_cpu.to(cuda_device)
    edge_index_cuda = edge_index_cpu.to(cuda_device)
    cuda_result = _aggregate(torch, features_cuda, edge_index_cuda).cpu()
    torch.testing.assert_close(cuda_result, cpu_result, rtol=1e-4, atol=1e-4)

    def resident_cuda_operation() -> Any:
        return _aggregate(torch, features_cuda, edge_index_cuda)

    resident_cuda_ms = _measure_ms(
        torch, cuda_device, resident_cuda_operation, warmup, repeats
    )

    def end_to_end_cuda_operation() -> Any:
        features = features_cpu.to(cuda_device)
        edge_index = edge_index_cpu.to(cuda_device)
        result = _aggregate(torch, features, edge_index)
        return result.cpu()

    end_to_end_cuda_ms = _measure_ms(
        torch, cuda_device, end_to_end_cuda_operation, warmup, repeats
    )
    return {
        "nodes": nodes,
        "edges": edges,
        "feature_dim": feature_dim,
        "input_bytes_approx": nodes * feature_dim * 4 + edges * 2 * 8,
        "cpu_torch_median_ms": round(cpu_ms, 4),
        "cuda_resident_median_ms": round(resident_cuda_ms, 4),
        "cuda_end_to_end_median_ms": round(end_to_end_cuda_ms, 4),
        "resident_cuda_speedup_vs_cpu": round(cpu_ms / resident_cuda_ms, 3),
        "end_to_end_cuda_speedup_vs_cpu": round(cpu_ms / end_to_end_cuda_ms, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", nargs="+", type=int, default=[10_000, 50_000, 100_000])
    parser.add_argument("--avg-degree", type=int, default=10)
    parser.add_argument("--features", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if any(value <= 0 for value in args.nodes + args.features):
        parser.error("--nodes and --features values must be positive")
    if args.avg_degree <= 0 or args.warmup < 0 or args.repeats <= 0:
        parser.error("--avg-degree and --repeats must be positive; --warmup cannot be negative")

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required. Install a CUDA-enabled PyTorch build.") from exc
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available in this Python environment. Check `nvidia-smi` "
            "and install a PyTorch build compatible with the installed NVIDIA driver."
        )
    if args.threads is not None:
        torch.set_num_threads(args.threads)

    properties = torch.cuda.get_device_properties(0)
    output: dict[str, Any] = {
        "benchmark": "torch_index_add_neighbor_aggregation",
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "python_cuda_available": torch.cuda.is_available(),
        "gpu_name": properties.name,
        "gpu_memory_bytes": properties.total_memory,
        "cpu_threads": torch.get_num_threads(),
        "warmup_iterations": args.warmup,
        "timed_iterations": args.repeats,
        "timing_note": (
            "Median wall-clock per iteration, with CUDA synchronization. "
            "Resident timing excludes host-device transfer; end-to-end timing "
            "includes feature/edge copies to GPU and result copy to CPU."
        ),
        "cases": [],
    }
    for nodes in args.nodes:
        for feature_dim in args.features:
            edges = nodes * args.avg_degree
            torch.cuda.reset_peak_memory_stats()
            case = _benchmark_case(
                torch,
                nodes,
                edges,
                feature_dim,
                args.warmup,
                args.repeats,
                args.seed,
            )
            case["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            output["cases"].append(case)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
