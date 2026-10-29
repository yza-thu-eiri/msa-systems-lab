# P1-C05 benchmark reproduction

This directory contains the public GPU timing and profiling harness. See the case study and evidence links on the repository README.

## Historical environment

- NVIDIA RTX A4000, SM86
- PyTorch 2.11.0+cu128, CUDA runtime 12.8
- FP32, B=1, N=128, Hq=8, Hkv=2, D=64
- Block size 32; fixed prefix selections Ksel=1, 2, 3, 4

The harness compares Dense SDPA, masked-dense GQA, and gathered sparse GQA. Only masked-dense and gathered sparse share the selected causal support and form a correctness pair.
