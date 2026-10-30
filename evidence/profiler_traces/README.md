# Locally generated GPU profiler traces

The public [P1 benchmark harness](../../benchmarks/reference/benchmark_forward.py) can export separate Chrome/Perfetto-compatible trace JSON files for the Dense SDPA, masked-dense GQA and gathered sparse GQA callables.

These traces are generated per run, may be large, and include environment-specific timings. They are **not included** in the public repository. The checked-in [Profiler operator tables](../profiler_tables/minimal_self_time_tables.txt) are historical recorded evidence, not a substitute for full timelines.

Use the [benchmark reproduction guide](../../benchmarks/README.md) on a compatible NVIDIA CUDA machine. Open generated traces with Perfetto to inspect CPU launch gaps, device-to-host synchronization and kernel sequences. CUDA Event timing is the separate latency measurement; Profiler spans must not be presented as uninstrumented latency.
