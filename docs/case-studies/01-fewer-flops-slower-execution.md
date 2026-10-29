# Case Study 01 — Fewer FLOPs, Slower Sparse Attention

> **Question:** Why did a gathered sparse-attention implementation run slower than a masked-dense implementation, despite calculating fewer query–key interactions?
>
> **Status:** Analysis of recorded internal measurements. The public source, benchmark harness, and raw profiler artifacts are being curated for release; this case is **not yet independently reproducible from this repository**.

## 1. Three implementations, three distinct jobs

| Implementation | Computation | Role |
| --- | --- | --- |
| Dense SDPA baseline | Full causal history; current baseline materializes expanded GQA heads | Engineering comparison only; **not** a same-support correctness oracle |
| Masked-dense GQA | Full score matrix, then masks everything outside selected causal support | Independent semantic oracle |
| Gathered sparse GQA | Gathers only the supported K/V tokens for each query/KV-head group | Transparent sparse reference, not an optimized GPU kernel |

For the two latter paths, the contract is the same. Raw selected block IDs are expanded into **effective causal token support**; only this final support defines which keys may contribute. Output and log-sum-exp (LSE) must agree within the tested numerical tolerance.

The transparent gathered reference deliberately retains Python loops, variable-length token indices, and small operations, because its job is to expose and verify semantics rather than maximize throughput.

## 2. Scope and measurement boundary

Recorded experiment (2026-08-30):

| Parameter | Value |
| --- | --- |
| Hardware | NVIDIA RTX A4000, SM86 |
| Software | PyTorch 2.11.0+cu128; CUDA runtime 12.8 |
| Precision | FP32 |
| Tensor dimensions | B=1, N=128, Hq=8, Hkv=2, D=64 |
| Block size | 32 tokens |
| Selection | Ksel=1..4, fixed prefix blocks |
| CUDA Event timing | 2 warmups + **2 samples** per callable; sample median |
| Profile | One active recorded step for Ksel=2 |

The timed callable covers input validation, support construction, Python row control, indexing, memory operations, and device execution. Input construction and correctness assertions are excluded.

**Limitation:** two samples identify an execution-cost order of magnitude but do not establish stable latency, uncertainty intervals, or any production performance ranking.

## 3. Mathematical savings are real

At Ksel=2 the mean effective support is 48.25 key tokens per query, compared with N=128 for masked-dense QK/PV arithmetic:

```text
Masked-dense QK/PV  = 4 × B × Hq × N × N × D
                    = 33,554,432 FLOPs

Gathered QK/PV      = 4 × B × Hq × N × mean(M) × D
                    = 12,648,448 FLOPs

Arithmetic reduction = 2.65×
```

This ledger counts attention QK/PV multiplications/additions. It does **not** account for metadata, dynamic indexing, memory transfers, kernel launch, temporary buffers, or synchronization. It is not a prediction of end-to-end speedup.

## 4. Runtime went in the opposite direction

| Selected blocks | Masked-dense median (ms) | Gathered median (ms) |
| ---: | ---: | ---: |
| 1 | 94.062 | 192.512 |
| **2** | **126.732** | **229.488** |
| 3 | 155.844 | 261.577 |
| 4 | 180.553 | 294.738 |

For Ksel=2, the gathered callable was approximately **1.81× slower** than the masked-dense oracle in this diagnostic run.

Profiler event counts for the representative Ksel=2 case:

| Event | Masked-dense | Gathered |
| --- | ---: | ---: |
| GPU kernels | 2,598 | **7,189** |
| GPU memcpy | 1,028 | 1,795 |
| Device-to-host operations | 1,027 | 1,283 |
| `cudaStreamSynchronize` | 1,027 | 1,283 |
| `aten::mm` | 0 | 512 |

The gathered execution traverses 256 (batch, query, KV-head) rows, each issuing two tiny matrix multiplications: 512 small MM calls. Dynamic `where`, indexing, scalar decisions, and validation introduce host synchronization and many fragmented kernels. The mathematical sparsity does not become an efficient device work schedule.

Profiler event counts are a **structural diagnosis**, not timing percentages. CUDA Event latency, operator self-time, and timeline spans answer different questions and should not be treated as interchangeable measurements.

## 5. Engineering decision

**Preserve the readable reference as an oracle. Build a separate execution path.** An optimized implementation would need to:

1. Represent sparse work in GPU-friendly groups/tiles instead of one host-driven row at a time.
2. Remove avoidable dynamic CPU decisions and DtoH synchronization from the measured critical path.
3. Reduce tiny kernel launches and account for irregular support and masking.
4. Exploit actual GQA shared K/V reuse rather than assuming mathematical sharing yields physical reuse.
5. Establish O/LSE and gradient parity against the frozen semantic reference *before* speed comparisons.

An optimized sparse kernel must report a named baseline, matching correctness semantics, hardware, dtype, shapes, timing boundaries, and sample statistics.

## 6. What this case establishes

**Established within the recorded tiny FP32 experiment:** same-support oracle/reference forward parity; reduced main attention arithmetic; a slower host-driven gathered callable; profiler evidence of synchronization and fragmented execution.

**Not established:** long-context or low-precision behavior, stable benchmark distributions, optimal GQA memory traffic, fused sparse-kernel performance, any general claim that sparse attention is slower than dense attention.

**Release status:** The reviewed analysis is published first. Source, focused tests, harness, and selected raw evidence will be added as separate, verifiable commits.
