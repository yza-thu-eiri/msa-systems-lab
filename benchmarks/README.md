# P1-C05 benchmark reproduction

This directory contains the public GPU timing and profiling harness. See the case study and evidence links on the repository README.

## Historical environment

- NVIDIA RTX A4000, SM86
- PyTorch 2.11.0+cu128, CUDA runtime 12.8
- FP32, B=1, N=128, Hq=8, Hkv=2, D=64
- Block size 32; fixed prefix selections Ksel=1, 2, 3, 4

The harness compares Dense SDPA, masked-dense GQA, and gathered sparse GQA. Only masked-dense and gathered sparse share the selected causal support and form a correctness pair.


## Run

Run from the repository root on Linux with a compatible NVIDIA driver and CUDA-enabled PyTorch:

- Historical teaching setup: python benchmarks/reference/benchmark_forward.py --output-dir artifacts/p1-c05
- Larger fresh timing sweep: python benchmarks/reference/benchmark_forward.py --warmup 10 --repetitions 50 --skip-profile --output-dir artifacts/p1-c05-rerun

The default two warmups and two timing samples preserve the historical setup; they do not establish stable latency estimates. Use more repetitions for a fresh measurement. Profiler instrumentation perturbs execution and should be treated as a separate diagnostic.


## Outputs and measurement boundaries

The output directory contains raw CUDA Event samples, software/hardware metadata, summary statistics, aggregate CPU/CUDA/memory/FLOPs/call-stack tables, and Chrome trace JSON files. Generated outputs are machine-specific and ignored by Git; retain the complete output directory when reporting a new run.

- Timing covers the full callable, including Python control flow, support validation/expansion, indexing, allocations, kernel launches, and device idle gaps between submitted operations.
- Input construction and the output/LSE parity gate occur outside the timing interval.
- The dense baseline uses full causal support. Only masked-dense and gathered sparse are a same-support correctness pair.
- Use Profiler to explain operator/kernel structure, not to replace CUDA Event latency.

## Published evidence

- [Historical GPU facts and limitations](../evidence/p1_c05_gpu_facts.md)
- [Machine-readable A4000 Ksel sweep](../evidence/p1_c05_a4000.csv)
- [Raw aggregate Profiler tables](../evidence/profiler_tables/minimal_self_time_tables.txt)
- [Case Study 01](../docs/case-studies/01-fewer-flops-slower-execution.md)

The historical raw Profiler table was collected on 2026-08-29. The CUDA Event sweep was recorded on 2026-08-30. They are separate runs and should not be interpreted as synchronized observations.
