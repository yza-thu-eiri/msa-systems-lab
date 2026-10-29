# P1-C05 GPU Facts / Benchmark / Profiling 证据

> **Release status:** The benchmark harness and selected historical evidence are now public. The A4000 measurements below were recorded internally on 2026-08-30 and have **not** been independently re-run from the public release. They are retained as historical observations, not fresh CI results or statistically stable performance estimates.
>
> Public artifacts: [benchmark harness](../benchmarks/reference/benchmark_forward.py) · [run guide](../benchmarks/README.md) · [machine-readable timing ledger](p1_c05_a4000.csv) · [raw Profiler tables](profiler_tables/minimal_self_time_tables.txt).

## 证据范围

本记录面向冻结版 Phase 1 前向可调用对象：

- `DenseSDPABackend`：工程稠密基线；
- `masked_dense_gqa`：掩码稠密语义 Oracle；
- `gathered_sparse_gqa`：逐行聚集的透明稀疏 Reference。

计时覆盖每个完整 callable，包括其内部校验、支持集展开、Python 控制流、
索引、内存操作与 CUDA 工作。输入构造和正确性断言位于计时区间之外。
稠密基线看完整 causal history，因此只对 masked/gathered 检查数值等价性。

## 环境与固定配置

- PyTorch：2.11.0+cu128
- CUDA runtime：12.8
- GPU：NVIDIA RTX A4000，SM86
- `B=1, N=128, Hq=8, Hkv=2, D=64`
- `block_size=32`
- `dtype=torch.float32`
- `device=cuda`
- `Ksel sweep=(1,2,3,4)`，每行选择前缀块 `[0, ..., Ksel-1]`
- CUDA Event：2 次预热、2 次采样；表中使用中位数
- Profiler：`WAIT=1, WARMUP=1, ACTIVE=1, repeat=1`，代表点 `Ksel=2`

## 2026-08-30 教学扫描

完整 harness 成功通过冒烟输入合同、三个 forward 调用，以及
masked/gathered 的 output、LSE 等价性检查。

| Ksel | min M | mean M | max M | Dense 中位数 ms | Masked 中位数 ms | Gathered 中位数 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 28.125 | 32 | 0.080 | 94.062 | 192.512 |
| 2 | 1 | 48.250 | 64 | 0.060 | 126.732 | 229.488 |
| 3 | 1 | 60.375 | 96 | 0.071 | 155.844 | 261.577 |
| 4 | 1 | 64.500 | 128 | 0.060 | 180.553 | 294.738 |

这些结果只有两次样本，用于验证扫描接线和暴露数量级差异，不构成稳定延迟、
分位数或生产性能证据。Dense 与 sparse 的数学 support 不同，不能把它们当作
correctness pair。

## Ksel=2 主 FLOPs 台账

masked-dense 的 QK/PV 都保留完整 `N` 轴：

```text
4 * B * Hq * N^2 * D = 33,554,432 FLOPs
```

gathered 使用逐行有效支持集，`mean M=48.25`：

```text
4 * B * Hq * N * mean_M * D = 12,648,448 FLOPs
```

因此当前主分支 ledger 的理论 reduction 约为 `2.65x`。这只统计 QK/PV
主数学量，不包含校验、索引、归约、内存分配、复制、同步和 kernel launch。

## Profiler 结构证据

完整 Profiler 会显著扰动绝对时间；下表只用调用计数描述执行结构。

| 事件 | Dense | Masked | Gathered |
|---|---:|---:|---:|
| GPU kernel | 3 | 2598 | 7189 |
| GPU memcpy | 0 | 1028 | 1795 |
| `cudaStreamSynchronize` | 0 | 1027 | 1283 |
| DtoH | 0 | 1027 | 1283 |
| DtoD | 0 | 1 | 512 |
| `aten::where` | 0 | 0 | 256 |
| `aten::nonzero` | 0 | 256 | 512 |
| `aten::index` | 0 | 256 | 768 |
| `aten::bmm` | 0 | 2 | 0 |
| `aten::mm` | 0 | 0 | 512 |

结构归因：

- Dense 通过 `repeat_interleave` 物化扩展 K/V，再进入 fused SDPA；它不是
  理想的 production GQA KV reuse 实现。
- Masked 的逐行 metadata validator 产生大量 `item`、`nonzero`、`unique`、
  DtoH 和同步；随后仍执行完整 `N x N` QK/PV，并分别执行 mask、softmax、
  LSE 和输出聚合。
- Gathered 共享相同 validator，又对 `B*N*Hkv=256` 个 row 分别执行动态
  `where`、K/V indexing、QK、softmax/LSE、PV 和结果写回。每行两个 MM，
  因而产生 512 次 tiny MM；动态输出长度额外增加 256 次 DtoH/同步。
- Gathered trace 中 GPU kernel 累计约 18.6 ms，但 active annotation 跨度约
  332.4 ms；该差异主要用于识别 host-driven 空洞，不能当作稳定利用率指标。

## 结论边界

当前证据说明：在上述 RTX A4000、FP32 tiny case 中，冻结版 gathered
Reference 虽然减少了主 attention FLOPs，但 validation、动态 shape、DtoH
同步、逐行 indexing、数千个 tiny kernel 和小块写回使其端到端延迟高于
masked-dense Oracle。

这些结果只描述当前 frozen transparent Reference callable 的实际执行成本，
不能说明 production sparse attention kernel 必然比 dense 慢，也不能直接
外推到更大 `N`、BF16/FP16、批量/融合调度、理想 GQA KV reuse 或
Triton/CUDA production kernel。

## Public reproduction

From the repository root on Linux with a compatible NVIDIA driver and CUDA-enabled PyTorch:

```bash
python benchmarks/reference/benchmark_forward.py --output-dir artifacts/p1-c05
```

The default two warmups and two timing samples preserve the historical teaching configuration.
For a new measurement, increase the sample count, for example:

```bash
python benchmarks/reference/benchmark_forward.py --warmup 10 --repetitions 50 --output-dir artifacts/p1-c05-rerun
```

The harness writes machine-readable CUDA Event samples and metadata to `benchmark_results.json`,
five aggregate Profiler tables per callable, a profiler manifest, and Chrome traces. Generated
outputs live under `artifacts/` and are intentionally not committed: timing and trace files are
machine-specific, and the Profiler materially perturbs execution.

The raw aggregate table linked above is a separate historical Profiler run recorded on 2026-08-29.
It is not the same collection as the CUDA Event sweep recorded on 2026-08-30. The two artifacts
answer different questions and must not be combined as if they were one synchronized run.

## Provenance

The historical facts in this note are sourced from the internal Phase 1 evidence record
(blob SHA `52e1774233d651f3cc9927836656c7df67e519e8`). The raw aggregate Profiler table was copied
verbatim from blob SHA `254b1a1a24d9e2358a26d580ead6f5adf0f0d1c7`. The public harness is a reviewed
release version based on the internal benchmark source (blob SHA
`a149eacf24004242397cbc2c84fec0b4748d6cfe`); it adds output persistence and CLI controls.
It is not byte-identical to the internal learner-oriented script.
