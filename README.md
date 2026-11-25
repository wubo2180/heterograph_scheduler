# Heterogeneous Graph Scheduler

A runnable prototype for task-aware graph workload modeling, resource matching,
adaptive scheduling, communication compression, and backend execution.

## What is implemented

- Multi-dimensional task features derived from graph shape and edge topology.
- A transparent latency/memory based matcher for CPU, GPU, and FPGA profiles.
- Optional online DQN scheduler (`torch` is only required when that policy is used).
- CPU neighbor aggregation/subgraph sampling, an optional CUDA aggregation backend,
  and a callback-based FPGA adapter.
- A zlib edge-payload codec and a simulator for comparing scheduler decisions.
- Unit tests covering task validation, scheduling, DQN state shape, compression, and
  CPU graph execution.

The simulator uses synthetic device profiles and **does not measure real hardware
speedups**. The FPGA backend is an integration interface, not a vendor-specific
driver. GPU execution requires a CUDA-enabled PyTorch installation. The project
does not claim the proposal's 10% target has been achieved; that requires controlled
benchmarks on the target hardware and datasets.

## Quick start

From this directory:

```powershell
python -m unittest discover -s tests -v
python -m heterograph_scheduler
python -m heterograph_scheduler --policy dqn
```

The DQN option requires PyTorch. The default heuristic and simulator use only the
Python standard library.

## Run a real CPU example

```powershell
python -m heterograph_scheduler --execute-cpu
```

The command runs neighbor aggregation over a generated graph and prints execution
timing and routing information as JSON.

## Measure CPU versus CUDA on real hardware

Run this on the Linux machine from the project directory (not from its parent):

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
python scripts/benchmark_cuda.py --nodes 10000 50000 100000 --avg-degree 10 --features 64 128 --warmup 5 --repeats 20
```

The benchmark runs the same PyTorch `index_add_` aggregation for both devices,
checks CPU/CUDA output agreement, synchronizes CUDA for timing, and reports
resident-kernel and end-to-end medians separately. Start with the defaults; only
increase graph sizes after confirming the workload fits in VRAM. `input_bytes_approx`
is just the feature/edge input size, not the full peak memory requirement.
Record the JSON along with the GPU driver, PyTorch/CUDA versions, CPU thread count,
and system memory use. Repeat the command at least three times and compare medians.

## Integrating hardware

Register a `ResourceProfile` and a matching backend with `ExecutionEngine`.
`CudaBackend` checks for an available CUDA device and fails explicitly if it is
unavailable. `FPGAAdapter` accepts a callable that invokes the chosen FPGA runtime;
the adapter does not pretend to discover or configure a board automatically.
`to_pyg_data()` provides a conversion helper when `torch-geometric` is installed.

## Benchmarking

The default command simulates task placement against synthetic profiles and prints
the estimated latency for a CPU-only baseline and the selected policy. Its
metadata-only workloads are intentionally larger than the CPU execution demo so
that device launch/transfer trade-offs become visible. For valid
research results, replace synthetic tasks and profiles with representative
Open Graph Benchmark/application data, use measured device/communication costs,
run repeated trials, and report hardware, software versions, and variance.
The DQN option starts with randomly initialized weights and performs online
updates; it is an algorithm prototype, not a pre-trained or performance-guaranteed
policy.
