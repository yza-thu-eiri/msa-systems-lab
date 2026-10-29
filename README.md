# MSA Systems Lab

**From sparse-attention semantics to GPU execution.**

A correctness-first engineering investigation of **block-sparse attention**, **learned routing**, and the gap between **reduced mathematical work** and **faster hardware execution**.

> **Release status:** The first two technical case studies and the P1 causal-support source/tests are public. Other source, configs, and raw experimental evidence are being selected for incremental release. The recorded internal results below are **not yet independently reproducible from this public repository**.
>
> **Scope:** Independent reproduction and systems analysis—not an official MiniMax implementation, a production attention library, or a claim of a new attention algorithm.

## Three questions

1. **Semantics** — Are masked-dense and gathered sparse attention computing the same function over the same causal support?
2. **Training** — What does imperfect learned block routing actually change in downstream model behavior?
3. **Execution** — When does mathematical sparsity translate into faster GPU work?

```mermaid
flowchart LR
    A[Block selection] --> B[Effective causal support]
    B --> C[Masked-dense oracle]
    B --> D[Gathered sparse reference]
    C <-->|Output / LSE parity| D
    D --> E[Fixed-selection backward]
    B --> F[Learned routing and FULL/SPARSE training]
    E --> G[GPU kernel parity and optimization]
    F --> G
```

The central design decision is to keep **readable correctness oracles** separate from future high-performance execution paths.

## Selected engineering findings

### 01 · Fewer FLOPs, slower execution

A transparent gathered reference reduced main QK/PV arithmetic by **2.65×** versus a masked-dense oracle in a small RTX A4000 / FP32 test, but its complete forward callable was **1.81× slower** in a two-sample diagnostic run. Host-driven irregular work, synchronization, and thousands of tiny kernels explain the mismatch.

| Ksel=2 (B=1, N=128, Hq=8, Hkv=2, D=64) | Masked-dense | Gathered sparse |
| --- | ---: | ---: |
| QK/PV FLOPs | 33,554,432 | 12,648,448 |
| CUDA Event median (ms) | 126.732 | 229.488 |
| Profiler GPU kernel events | 2,598 | 7,189 |

**[Read Case Study 01 — Fewer FLOPs, Slower Sparse Attention](docs/case-studies/01-fewer-flops-slower-execution.md)**

*Measurement boundary: complete PyTorch/Python forward callables including validation and support construction, after two warmups and with only two timing samples. These are not statistically stable performance estimates, and do not measure an optimized sparse GPU kernel.*

### 02 · Routing error is not task failure

A small-model block router reached a persistent Routing-KL plateau near **0.205** in a related training investigation. A separate paired FULL/SPARSE replay showed that routing retained most teacher attention mass, with a modest average task accuracy gap—but **thousands of real prediction changes**.

| Held-out sparse-active positions | Recorded result |
| --- | ---: |
| Prediction positions audited | 83,616 |
| Teacher attention mass retained | 95.7631% |
| FULL / SPARSE top-1 accuracy | 54.988% / 54.409% |
| Top-1 prediction identity changed | 4.73% |

**[Read Case Study 02 — Routing Error Is Not the Same as Task Failure](docs/case-studies/02-routing-error-and-task-impact.md)**

*The KL plateau and paired functional audit are related but distinct experiment contexts. Conclusions are limited to controlled small-model observations, not deep-model or long-context guarantees.*

## What is established—and what is not

| System component | Evidence maturity | Public artifact |
| --- | --- | --- |
| Causal support contract | Phase 1 internally verified; public tests provided, CI result pending | [Source](src/msa_lab/contracts/support.py) · [Tests](tests/test_support.py) · [Contract](docs/reference/causal-support.md) |\n| Sparse forward and manual fixed-selection backward | Phase 1 internally verified | Case Study 01; source and focused tests pending |
| Learned routing, mode switching, exact resume, one-/two-layer diagnosis | Phase 2 internally verified; 77 regression tests recorded | Case Study 02; source/configs pending |
| GPU-friendly Triton/kernel execution and speedup | **Not established** | Future work |
| Irregular metadata, backward scheduling, inference runtime | **Not established** | Future work |

## How to navigate

- **Fast technical review:** Start with [Case Study 01](docs/case-studies/01-fewer-flops-slower-execution.md) and [Case Study 02](docs/case-studies/02-routing-error-and-task-impact.md).
- **Evidence discipline and release queue:** [Publication and evidence standards](docs/PUBLICATION.md).
- **First released code:** [Causal support semantics](docs/reference/causal-support.md) includes [source](src/msa_lab/contracts/support.py), [acceptance tests](tests/test_support.py), and a [CPU CI workflow](.github/workflows/reference-tests.yml). Run `python -m pip install -e ".[test]"` followed by `python -m pytest -q tests/test_support.py`. CI result pending; do not assume a pass.\n- **Further reproducibility:** The other references, tests, benchmark harnesses, and selected raw artifacts will appear as independently reviewed commits. Until then, their findings remain labeled as internal observations.

## Next frontier: turning sparsity into GPU work

The next work asks how to transform a semantically transparent reference into an efficient GPU execution path without silently changing its meaning: SM/warp execution, registers and memory movement, launch overhead, tiling, irregular support, physical GQA K/V reuse, precision, and forward/backward parity.

Any future speedup claim will require a specified baseline, hardware, shapes, dtype, timing methodology, correctness verification, and sampling statistics.

---

*MSA stands for MiniMax Sparse Attention. This is an unaffiliated, independent educational and engineering investigation.*

*中文简介：围绕稀疏注意力正确性、Router 的实际任务影响以及 GPU 执行效率，将经过验证的技术成果逐步整理为可审阅的工程案例。*
