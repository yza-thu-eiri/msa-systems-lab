# MSA Systems Lab

**Sparse-attention systems engineering: correctness → training → GPU performance diagnosis.**

I use block-sparse attention to work through a real systems question: **how do we turn an algorithm that does less mathematical work into a correct, trainable, and eventually faster implementation?** This repository contains the PyTorch reference implementation, training system, controlled investigations, and public tests.

[![CPU acceptance](https://github.com/yza-thu-eiri/msa-systems-lab/actions/workflows/reference-tests.yml/badge.svg)](https://github.com/yza-thu-eiri/msa-systems-lab/actions/workflows/reference-tests.yml)
[Latest verified CI: 187 passed, 1 CUDA-only skipped](https://github.com/yza-thu-eiri/msa-systems-lab/actions/runs/41000797337)

## What this project demonstrates

| Engineering capability | Concrete evidence |
| --- | --- |
| **Define and verify operator semantics** | Causal support contract; masked-dense vs gathered-sparse **output/LSE parity**; manually derived **dQ/dK/dV**, checked against autograd and finite differences. [Read the correctness derivation](docs/reference/manual-backward.md) |
| **Build a testable training system** | Learned block Router, gradient ownership, per-layer FULL/SPARSE execution, shared Dense initialization, optimizer state and **exact checkpoint resume** in TinyLM. [Inspect the training implementation](docs/training.md) |
| **Diagnose hardware costs instead of equating FLOPs with speed** | At one historical A4000 test shape, the gathered PyTorch reference did **2.65× less QK/PV arithmetic** yet ran **1.81× slower** than masked-dense. Profiler analysis traced the cost to fragmented launches and host-driven execution. [Read the profiling case](docs/case-studies/01-fewer-flops-slower-execution.md) |

**Scope:** these are correctness-first PyTorch references and small-model experiments, **not** a production sparse CUDA/Triton kernel or a multi-GPU scaling claim. The published A4000 timing used only two samples; it is a diagnostic, not a stable performance estimate.

## Where to start

| Reader | Start here | Why |
| --- | --- | --- |
| **AI Infra / GPU systems engineer** | [01 — Fewer FLOPs, slower execution](docs/case-studies/01-fewer-flops-slower-execution.md) | Concrete execution bottleneck, benchmark boundary, kernel-design implications |
| **Training systems engineer** | [TinyLM training + exact resume](docs/training.md) | Runnable code, mode switching, optimizer/RNG recovery and tests |
| **ML / algorithm researcher** | [02 — Routing errors vs task impact](docs/case-studies/02-routing-error-and-task-impact.md) | Hypothesis-driven Router diagnosis, paired token outcomes and four original charts |

Two follow-on investigations explore [training adaptation after switching to SPARSE](docs/case-studies/03-adapting-to-sparse-execution.md) and [FF/SF/FS/SS interactions across two layers](docs/case-studies/04-do-sparse-errors-cascade-across-layers.md).

## Run the public implementation

On **Python 3.12 / CPU**:

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test]"
python -m pytest tests/ -ra

python examples/fixed_support_walkthrough.py
python examples/routing_walkthrough.py
python examples/c07_four_state_walkthrough.py
```

The [CI workflow](.github/workflows/reference-tests.yml) also runs synthetic TinyLM **train → eval → generate**. [Training instructions](docs/training.md) · [P1 GPU benchmark harness](benchmarks/README.md). The GPU-only exact-resume regression is published but skipped in CPU CI; that skip is **not** GPU validation.

## Evidence boundaries

- **Publicly runnable:** reference forward/backward and Router tests, TinyLM training/resume, synthetic four-state replay, and C07 logical-work estimates.
- **Historical, not independently replayed from this checkout:** A4000 profiler/timing numbers and C06/C07 held-out experiments. Four [original Router diagnostic figures](docs/case-studies/02-routing-error-and-task-impact.md) are byte-identical copies from the private experiment review; source hashes and missing inputs are documented in the [evidence guide](docs/p2-experiment-evidence.md).
- **Not yet established:** an optimized sparse GPU kernel, production-speed benchmarks, long-context/deep-model reliability, or distributed scaling.

**Next engineering step:** implement a GPU-friendly sparse execution path and verify output/LSE/backward parity against these references before claiming acceleration.

[Operator support contract](docs/reference/causal-support.md) · [Routing semantics](docs/routing.md) · [Evidence standards](docs/PUBLICATION.md)

---

*MSA = MiniMax Sparse Attention. This is an independent, unaffiliated engineering investigation; not an official MiniMax implementation. MIT licensed.*
