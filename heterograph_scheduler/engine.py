"""Execution orchestration for registered resources and backends."""

from __future__ import annotations

from collections.abc import Sequence

from .backends import ExecutionBackend, ExecutionResult
from .models import GraphTask, ResourceProfile, ScheduleDecision
from .scheduler import HeuristicScheduler


class ExecutionEngine:
    def __init__(
        self,
        resources: Sequence[ResourceProfile],
        backends: dict[str, ExecutionBackend],
        scheduler: HeuristicScheduler | None = None,
    ) -> None:
        self.resources = tuple(resources)
        self.backends = backends
        self.scheduler = scheduler or HeuristicScheduler()
        available_names = {resource.name for resource in self.resources}
        missing = available_names.difference(backends)
        if missing:
            raise ValueError(f"no execution backend registered for: {sorted(missing)}")

    def run(self, tasks: Sequence[GraphTask]) -> list[tuple[ScheduleDecision, ExecutionResult]]:
        results = []
        for task in sorted(tasks, key=lambda item: item.priority, reverse=True):
            decision = self.scheduler.schedule(task, self.resources)
            results.append(
                (decision, self.backends[decision.resource_name].execute(task))
            )
        return results
(myenv) wubo@wubo-MS-Terminator-B850M-PRO:~/heterograph_scheduler$ for n in 1 2 4 6 8 12; do
  echo "CPU threads=$n"
  python scripts/benchmark_cuda.py --nodes 10000 50000 100000 \
    --avg-degree 10 --features 64 128 --threads "$n" --warmup 10 --repeats 50
done
CPU threads=1
{
  "benchmark": "torch_index_add_neighbor_aggregation",
  "torch_version": "2.14.0+cu130",
  "torch_cuda_version": "13.0",
  "python_cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 5070 Ti",
  "gpu_memory_bytes": 16608788480,
  "cpu_threads": 1,
  "warmup_iterations": 10,
  "timed_iterations": 50,
  "timing_note": "Median wall-clock per iteration, with CUDA synchronization. Resident timing excludes host-device transfer; end-to-end timing includes feature/edge copies to GPU and result copy to CPU.",
  "cases": [
    {
      "nodes": 10000,
      "edges": 100000,
      "feature_dim": 64,
      "input_bytes_approx": 4160000,
      "cpu_torch_median_ms": 6.2053,
      "cuda_resident_median_ms": 0.1373,
      "cuda_end_to_end_median_ms": 0.7173,
      "resident_cuda_speedup_vs_cpu": 45.179,
      "end_to_end_cuda_speedup_vs_cpu": 8.651,
      "cuda_peak_allocated_bytes": 36480000
    },
    {
      "nodes": 10000,
      "edges": 100000,
      "feature_dim": 128,
      "input_bytes_approx": 6720000,
      "cpu_torch_median_ms": 31.3887,
      "cuda_resident_median_ms": 0.1827,
      "cuda_end_to_end_median_ms": 1.2,
      "resident_cuda_speedup_vs_cpu": 171.76,
      "end_to_end_cuda_speedup_vs_cpu": 26.158,
      "cuda_peak_allocated_bytes": 69760000
    },
    {
      "nodes": 50000,
      "edges": 500000,
      "feature_dim": 64,
      "input_bytes_approx": 20800000,
      "cpu_torch_median_ms": 91.0657,
      "cuda_resident_median_ms": 0.6492,
      "cuda_end_to_end_median_ms": 3.4102,
      "resident_cuda_speedup_vs_cpu": 140.278,
      "end_to_end_cuda_speedup_vs_cpu": 26.704,
      "cuda_peak_allocated_bytes": 182571520
    },
    {
      "nodes": 50000,
      "edges": 500000,
      "feature_dim": 128,
      "input_bytes_approx": 33600000,
      "cpu_torch_median_ms": 180.5194,
      "cuda_resident_median_ms": 1.1194,
      "cuda_end_to_end_median_ms": 6.4944,
      "resident_cuda_speedup_vs_cpu": 161.265,
      "end_to_end_cuda_speedup_vs_cpu": 27.796,
      "cuda_peak_allocated_bytes": 348800000
    },
    {
      "nodes": 100000,
      "edges": 1000000,
      "feature_dim": 64,
      "input_bytes_approx": 41600000,
      "cpu_torch_median_ms": 212.6171,
      "cuda_resident_median_ms": 1.3453,
      "cuda_end_to_end_median_ms": 7.5239,
      "resident_cuda_speedup_vs_cpu": 158.039,
      "end_to_end_cuda_speedup_vs_cpu": 28.259,
      "cuda_peak_allocated_bytes": 364800000
    },
    {
      "nodes": 100000,
      "edges": 1000000,
      "feature_dim": 128,
      "input_bytes_approx": 67200000,
      "cpu_torch_median_ms": 412.7146,
      "cuda_resident_median_ms": 2.7261,
      "cuda_end_to_end_median_ms": 31.898,
      "resident_cuda_speedup_vs_cpu": 151.395,
      "end_to_end_cuda_speedup_vs_cpu": 12.939,
      "cuda_peak_allocated_bytes": 697600000
    }
  ]
}
CPU threads=2
{
  "benchmark": "torch_index_add_neighbor_aggregation",
  "torch_version": "2.14.0+cu130",
  "torch_cuda_version": "13.0",
  "python_cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 5070 Ti",
  "gpu_memory_bytes": 16608788480,
  "cpu_threads": 2,
  "warmup_iterations": 10,
  "timed_iterations": 50,
  "timing_note": "Median wall-clock per iteration, with CUDA synchronization. Resident timing excludes host-device transfer; end-to-end timing includes feature/edge copies to GPU and result copy to CPU.",
  "cases": [
    {
      "nodes": 10000,
      "edges": 100000,
      "feature_dim": 64,
      "input_bytes_approx": 4160000,
      "cpu_torch_median_ms": 3.2929,
      "cuda_resident_median_ms": 0.1382,
      "cuda_end_to_end_median_ms": 0.7206,
      "resident_cuda_speedup_vs_cpu": 23.833,
      "end_to_end_cuda_speedup_vs_cpu": 4.569,
      "cuda_peak_allocated_bytes": 36480000
    },
    {
      "nodes": 10000,
      "edges": 100000,
      "feature_dim": 128,
      "input_bytes_approx": 6720000,
      "cpu_torch_median_ms": 19.0518,
      "cuda_resident_median_ms": 0.1825,
      "cuda_end_to_end_median_ms": 1.1958,
      "resident_cuda_speedup_vs_cpu": 104.401,
      "end_to_end_cuda_speedup_vs_cpu": 15.932,
      "cuda_peak_allocated_bytes": 69760000
    },
    {
      "nodes": 50000,
      "edges": 500000,
      "feature_dim": 64,
      "input_bytes_approx": 20800000,
      "cpu_torch_median_ms": 54.9136,
      "cuda_resident_median_ms": 0.6495,
      "cuda_end_to_end_median_ms": 3.4343,
      "resident_cuda_speedup_vs_cpu": 84.549,
      "end_to_end_cuda_speedup_vs_cpu": 15.99,
      "cuda_peak_allocated_bytes": 182571520
    },
    {
      "nodes": 50000,
      "edges": 500000,
      "feature_dim": 128,
      "input_bytes_approx": 33600000,
      "cpu_torch_median_ms": 107.9599,
      "cuda_resident_median_ms": 1.1191,
      "cuda_end_to_end_median_ms": 6.4727,
      "resident_cuda_speedup_vs_cpu": 96.472,
      "end_to_end_cuda_speedup_vs_cpu": 16.679,
      "cuda_peak_allocated_bytes": 348800000
    },
    {
      "nodes": 100000,
      "edges": 1000000,
      "feature_dim": 64,
      "input_bytes_approx": 41600000,
      "cpu_torch_median_ms": 124.4006,
      "cuda_resident_median_ms": 1.3448,
      "cuda_end_to_end_median_ms": 7.5291,
      "resident_cuda_speedup_vs_cpu": 92.505,
      "end_to_end_cuda_speedup_vs_cpu": 16.523,
      "cuda_peak_allocated_bytes": 364800000
    },
    {
      "nodes": 100000,
      "edges": 1000000,
      "feature_dim": 128,
      "input_bytes_approx": 67200000,
      "cpu_torch_median_ms": 245.9822,
      "cuda_resident_median_ms": 2.7263,
      "cuda_end_to_end_median_ms": 32.2212,
      "resident_cuda_speedup_vs_cpu": 90.225,
      "end_to_end_cuda_speedup_vs_cpu": 7.634,
      "cuda_peak_allocated_bytes": 697600000
    }
  ]
}
CPU threads=4
{
  "benchmark": "torch_index_add_neighbor_aggregation",
  "torch_version": "2.14.0+cu130",
  "torch_cuda_version": "13.0",
  "python_cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 5070 Ti",
  "gpu_memory_bytes": 16608788480,
  "cpu_threads": 4,
  "warmup_iterations": 10,
  "timed_iterations": 50,
  "timing_note": "Median wall-clock per iteration, with CUDA synchronization. Resident timing excludes host-device transfer; end-to-end timing includes feature/edge copies to GPU and result copy to CPU.",
  "cases": [
    {
      "nodes": 10000,
      "edges": 100000,
      "feature_dim": 64,
      "input_bytes_approx": 4160000,
      "cpu_torch_median_ms": 1.9704,
      "cuda_resident_median_ms": 0.1384,
      "cuda_end_to_end_median_ms": 0.7235,
      "resident_cuda_speedup_vs_cpu": 14.241,
      "end_to_end_cuda_speedup_vs_cpu": 2.723,
      "cuda_peak_allocated_bytes": 36480000
    },
    {
      "nodes": 10000,
      "edges": 100000,
      "feature_dim": 128,
      "input_bytes_approx": 6720000,
      "cpu_torch_median_ms": 12.92,
      "cuda_resident_median_ms": 0.1818,
      "cuda_end_to_end_median_ms": 1.1888,
      "resident_cuda_speedup_vs_cpu": 71.051,
      "end_to_end_cuda_speedup_vs_cpu": 10.868,
      "cuda_peak_allocated_bytes": 69760000
    },
    {
      "nodes": 50000,
      "edges": 500000,
      "feature_dim": 64,
      "input_bytes_approx": 20800000,
      "cpu_torch_median_ms": 35.0571,
      "cuda_resident_median_ms": 0.6486,
      "cuda_end_to_end_median_ms": 3.391,
      "resident_cuda_speedup_vs_cpu": 54.051,
      "end_to_end_cuda_speedup_vs_cpu": 10.338,
      "cuda_peak_allocated_bytes": 182571520
    },
    {
      "nodes": 50000,
      "edges": 500000,
      "feature_dim": 128,
      "input_bytes_approx": 33600000,
      "cpu_torch_median_ms": 72.1934,
      "cuda_resident_median_ms": 1.1192,
      "cuda_end_to_end_median_ms": 6.47,
      "resident_cuda_speedup_vs_cpu": 64.502,
      "end_to_end_cuda_speedup_vs_cpu": 11.158,
      "cuda_peak_allocated_bytes": 348800000
    },
    {
      "nodes": 100000,
      "edges": 1000000,
      "feature_dim": 64,
      "input_bytes_approx": 41600000,
      "cpu_torch_median_ms": 82.427,
      "cuda_resident_median_ms": 1.3446,
      "cuda_end_to_end_median_ms": 7.5687,
      "resident_cuda_speedup_vs_cpu": 61.3,
      "end_to_end_cuda_speedup_vs_cpu": 10.89,
      "cuda_peak_allocated_bytes": 364800000
    },
    {
      "nodes": 100000,
      "edges": 1000000,
      "feature_dim": 128,
      "input_bytes_approx": 67200000,
      "cpu_torch_median_ms": 171.7538,
      "cuda_resident_median_ms": 2.7258,
      "cuda_end_to_end_median_ms": 31.9267,
      "resident_cuda_speedup_vs_cpu": 63.01,
      "end_to_end_cuda_speedup_vs_cpu": 5.38,
      "cuda_peak_allocated_bytes": 697600000
    }
  ]
}
CPU threads=6
{
  "benchmark": "torch_index_add_neighbor_aggregation",
  "torch_version": "2.14.0+cu130",
  "torch_cuda_version": "13.0",
  "python_cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 5070 Ti",
  "gpu_memory_bytes": 16608788480,
  "cpu_threads": 6,
  "warmup_iterations": 10,
  "timed_iterations": 50,
  "timing_note": "Median wall-clock per iteration, with CUDA synchronization. Resident timing excludes host-device transfer; end-to-end timing includes feature/edge copies to GPU and result copy to CPU.",
  "cases": [
    {
      "nodes": 10000,
      "edges": 100000,
      "feature_dim": 64,
      "input_bytes_approx": 4160000,
      "cpu_torch_median_ms": 1.436,
      "cuda_resident_median_ms": 0.1388,
      "cuda_end_to_end_median_ms": 0.7274,
      "resident_cuda_speedup_vs_cpu": 10.347,
      "end_to_end_cuda_speedup_vs_cpu": 1.974,
      "cuda_peak_allocated_bytes": 36480000
    },
    {
      "nodes": 10000,
      "edges": 100000,
      "feature_dim": 128,
      "input_bytes_approx": 6720000,
      "cpu_torch_median_ms": 11.9153,
      "cuda_resident_median_ms": 0.1827,
      "cuda_end_to_end_median_ms": 1.1969,
      "resident_cuda_speedup_vs_cpu": 65.224,
      "end_to_end_cuda_speedup_vs_cpu": 9.956,
      "cuda_peak_allocated_bytes": 69760000
    },
    {
      "nodes": 50000,
      "edges": 500000,
      "feature_dim": 64,
      "input_bytes_approx": 20800000,
      "cpu_torch_median_ms": 31.4636,
      "cuda_resident_median_ms": 0.652,
      "cuda_end_to_end_median_ms": 3.4141,
      "resident_cuda_speedup_vs_cpu": 48.26,
      "end_to_end_cuda_speedup_vs_cpu": 9.216,
      "cuda_peak_allocated_bytes": 182571520
    },
    {
      "nodes": 50000,
      "edges": 500000,
      "feature_dim": 128,
      "input_bytes_approx": 33600000,
      "cpu_torch_median_ms": 63.734,
      "cuda_resident_median_ms": 1.1205,
      "cuda_end_to_end_median_ms": 6.5208,
      "resident_cuda_speedup_vs_cpu": 56.879,
      "end_to_end_cuda_speedup_vs_cpu": 9.774,
      "cuda_peak_allocated_bytes": 348800000
    },
    {
      "nodes": 100000,
      "edges": 1000000,
      "feature_dim": 64,
      "input_bytes_approx": 41600000,
      "cpu_torch_median_ms": 83.3252,
      "cuda_resident_median_ms": 1.3457,
      "cuda_end_to_end_median_ms": 7.5042,
      "resident_cuda_speedup_vs_cpu": 61.922,
      "end_to_end_cuda_speedup_vs_cpu": 11.104,
      "cuda_peak_allocated_bytes": 364800000
    },
    {
      "nodes": 100000,
      "edges": 1000000,
      "feature_dim": 128,
      "input_bytes_approx": 67200000,
      "cpu_torch_median_ms": 151.2883,
      "cuda_resident_median_ms": 2.7253,
      "cuda_end_to_end_median_ms": 32.0674,
      "resident_cuda_speedup_vs_cpu": 55.512,
      "end_to_end_cuda_speedup_vs_cpu": 4.718,
      "cuda_peak_allocated_bytes": 697600000
    }
  ]
}
CPU threads=8
{
  "benchmark": "torch_index_add_neighbor_aggregation",
  "torch_version": "2.14.0+cu130",
  "torch_cuda_version": "13.0",
  "python_cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 5070 Ti",
  "gpu_memory_bytes": 16608788480,
  "cpu_threads": 8,
  "warmup_iterations": 10,
  "timed_iterations": 50,
  "timing_note": "Median wall-clock per iteration, with CUDA synchronization. Resident timing excludes host-device transfer; end-to-end timing includes feature/edge copies to GPU and result copy to CPU.",
  "cases": [
    {
      "nodes": 10000,
      "edges": 100000,
      "feature_dim": 64,
      "input_bytes_approx": 4160000,
      "cpu_torch_median_ms": 1.6273,
      "cuda_resident_median_ms": 0.1381,
      "cuda_end_to_end_median_ms": 0.7202,
      "resident_cuda_speedup_vs_cpu": 11.781,
      "end_to_end_cuda_speedup_vs_cpu": 2.259,
      "cuda_peak_allocated_bytes": 36480000
    },
    {
      "nodes": 10000,
      "edges": 100000,
      "feature_dim": 128,
      "input_bytes_approx": 6720000,
      "cpu_torch_median_ms": 12.1312,
      "cuda_resident_median_ms": 0.182,
      "cuda_end_to_end_median_ms": 1.1947,
      "resident_cuda_speedup_vs_cpu": 66.638,
      "end_to_end_cuda_speedup_vs_cpu": 10.154,
      "cuda_peak_allocated_bytes": 69760000
    },
    {
      "nodes": 50000,
      "edges": 500000,
      "feature_dim": 64,
      "input_bytes_approx": 20800000,
      "cpu_torch_median_ms": 32.8148,
      "cuda_resident_median_ms": 0.6493,
      "cuda_end_to_end_median_ms": 3.401,
      "resident_cuda_speedup_vs_cpu": 50.539,
      "end_to_end_cuda_speedup_vs_cpu": 9.648,
      "cuda_peak_allocated_bytes": 182571520
    },
    {
      "nodes": 50000,
      "edges": 500000,
      "feature_dim": 128,
      "input_bytes_approx": 33600000,
      "cpu_torch_median_ms": 64.9958,
      "cuda_resident_median_ms": 1.1192,
      "cuda_end_to_end_median_ms": 6.515,
      "resident_cuda_speedup_vs_cpu": 58.071,
      "end_to_end_cuda_speedup_vs_cpu": 9.976,
      "cuda_peak_allocated_bytes": 348800000
    },
    {
      "nodes": 100000,
      "edges": 1000000,
      "feature_dim": 64,
      "input_bytes_approx": 41600000,
      "cpu_torch_median_ms": 73.4941,
      "cuda_resident_median_ms": 1.3449,
      "cuda_end_to_end_median_ms": 7.5721,
      "resident_cuda_speedup_vs_cpu": 54.646,
      "end_to_end_cuda_speedup_vs_cpu": 9.706,
      "cuda_peak_allocated_bytes": 364800000
    },
    {
      "nodes": 100000,
      "edges": 1000000,
      "feature_dim": 128,
      "input_bytes_approx": 67200000,
      "cpu_torch_median_ms": 147.6896,
      "cuda_resident_median_ms": 2.7259,
      "cuda_end_to_end_median_ms": 31.9732,
      "resident_cuda_speedup_vs_cpu": 54.18,
      "end_to_end_cuda_speedup_vs_cpu": 4.619,
      "cuda_peak_allocated_bytes": 697600000
    }
  ]
}
CPU threads=12
{
  "benchmark": "torch_index_add_neighbor_aggregation",
  "torch_version": "2.14.0+cu130",
  "torch_cuda_version": "13.0",
  "python_cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 5070 Ti",
  "gpu_memory_bytes": 16608788480,
  "cpu_threads": 12,
  "warmup_iterations": 10,
  "timed_iterations": 50,
  "timing_note": "Median wall-clock per iteration, with CUDA synchronization. Resident timing excludes host-device transfer; end-to-end timing includes feature/edge copies to GPU and result copy to CPU.",
  "cases": [
    {
      "nodes": 10000,
      "edges": 100000,
      "feature_dim": 64,
      "input_bytes_approx": 4160000,
      "cpu_torch_median_ms": 1.3695,
      "cuda_resident_median_ms": 0.1381,
      "cuda_end_to_end_median_ms": 0.7258,
      "resident_cuda_speedup_vs_cpu": 9.919,
      "end_to_end_cuda_speedup_vs_cpu": 1.887,
      "cuda_peak_allocated_bytes": 36480000
    },
    {
      "nodes": 10000,
      "edges": 100000,
      "feature_dim": 128,
      "input_bytes_approx": 6720000,
      "cpu_torch_median_ms": 11.3362,
      "cuda_resident_median_ms": 0.1831,
      "cuda_end_to_end_median_ms": 1.1945,
      "resident_cuda_speedup_vs_cpu": 61.927,
      "end_to_end_cuda_speedup_vs_cpu": 9.491,
      "cuda_peak_allocated_bytes": 69760000
    },
    {
      "nodes": 50000,
      "edges": 500000,
      "feature_dim": 64,
      "input_bytes_approx": 20800000,
      "cpu_torch_median_ms": 34.5546,
      "cuda_resident_median_ms": 0.649,
      "cuda_end_to_end_median_ms": 3.4166,
      "resident_cuda_speedup_vs_cpu": 53.244,
      "end_to_end_cuda_speedup_vs_cpu": 10.114,
      "cuda_peak_allocated_bytes": 182571520
    },
    {
      "nodes": 50000,
      "edges": 500000,
      "feature_dim": 128,
      "input_bytes_approx": 33600000,
      "cpu_torch_median_ms": 64.3113,
      "cuda_resident_median_ms": 1.1193,
      "cuda_end_to_end_median_ms": 6.4838,
      "resident_cuda_speedup_vs_cpu": 57.459,
      "end_to_end_cuda_speedup_vs_cpu": 9.919,
      "cuda_peak_allocated_bytes": 348800000
    },
    {
      "nodes": 100000,
      "edges": 1000000,
      "feature_dim": 64,
      "input_bytes_approx": 41600000,
      "cpu_torch_median_ms": 71.9448,
      "cuda_resident_median_ms": 1.3446,
      "cuda_end_to_end_median_ms": 7.5763,
      "resident_cuda_speedup_vs_cpu": 53.508,
      "end_to_end_cuda_speedup_vs_cpu": 9.496,
      "cuda_peak_allocated_bytes": 364800000
    },
    {
      "nodes": 100000,
      "edges": 1000000,
      "feature_dim": 128,
      "input_bytes_approx": 67200000,
      "cpu_torch_median_ms": 137.1477,
      "cuda_resident_median_ms": 2.726,
      "cuda_end_to_end_median_ms": 32.1749,
      "resident_cuda_speedup_vs_cpu": 50.311,
      "end_to_end_cuda_speedup_vs_cpu": 4.263,
      "cuda_peak_allocated_bytes": 697600000
    }
  ]
}