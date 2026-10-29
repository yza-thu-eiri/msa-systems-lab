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
