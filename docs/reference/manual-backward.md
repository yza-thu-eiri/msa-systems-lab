# Fixed-selection backward: manual gradients for gathered sparse GQA

**Question:** Can we explicitly derive and implement `dQ`, `dK`, and `dV` for sparse attention, including GQA sharing and optional LSE gradients, without relying on autograd inside the implementation?

**Answer within the Phase 1 boundary:** yes, for *fixed causal support*. This page presents the mathematical/dataflow contract and links to the published source and verification.

> This is a deliberately transparent PyTorch reference, **not** a CUDA/Triton kernel, custom `autograd.Function`, or differentiable routing/selection operator.

## One row, with GQA sharing

For a fixed `[batch, query, hkv]` row:

| Symbol | Shape | Interpretation |
| --- | --- | --- |
| `Q` | `[G, D]` | `G = Hq/Hkv` query heads sharing one KV head |
| `K, V` | `[M, D]` | Only the `M` effective causal-visible selected tokens |
| `S = Q Kᵀ / √D` | `[G, M]` | Scaled scores on selected keys |
| `P = softmax(S, dim=-1)` | `[G, M]` | Separate normalization for each query head |
| `O = P V` | `[G, D]` | Attention output |
| `LSE = logsumexp(S, dim=-1)` | `[G]` | Forward normalization log-sum-exp |

The implementation recomputes the small selected forward state when calculating backward. It does not save a global dense attention probability matrix.

## Derivation and gradient ownership

For an upstream gradient `dO [G,D]`:

```text
dV = Pᵀ dO                                  [M,D]
dP = dO Vᵀ                                  [G,M]
u  = sum_M(P ⊙ dP)                           [G,1]
dS = P ⊙ (dP - u)                            [G,M]
```

If the loss *also* depends on `LSE` as a differentiable output, its upstream `dLSE [G]` contributes:

```text
dS += dLSE[:, None] ⊙ P                     [G,M]
```

Finally, propagate through the scaled dot product:

```text
dQ = dS K / √D                               [G,D]
dK = dSᵀ Q / √D                              [M,D]
```

An important distinction: **saving LSE as numerical state does not itself create a gradient path**. The extra term applies only if a caller actually supplies `grad_lse`.

### From selected rows back to global tensors

```text
Q gradient:
  each [b, query, hkv, group] has one owner
  → write dQ directly to its unique query position

K/V gradients:
  different query rows may select the same original key token
  and multiple query heads in a GQA group share K/V
  → sum within the G group, then accumulate at original token indices

Selection metadata:
  fixed, discrete, and not differentiated
```

The difference between *direct write* and *scatter accumulation* is part of the algorithm's gradient semantics, not merely an implementation detail.

## Public artifacts

| Artifact | Role |
| --- | --- |
| [`gathered_sparse_backward.py`](../../src/msa_lab/reference/gathered_sparse_backward.py) | Frozen Phase 1 manual reference; `AttentionGradients(dq,dk,dv)` |
| [`test_gathered_sparse_backward.py`](../../tests/test_gathered_sparse_backward.py) | Seven original acceptance tests: autograd comparison, output-only vs output+LSE, masked-dense cross-check, unsupported-token zero gradients, one-key boundary, and invalid-input rejection |
| [`test_backward_finite_difference.py`](../../tests/test_backward_finite_difference.py) | Additional independent central-difference check against the public forward objective, without using autograd as the oracle |
| [Forward parity](forward-parity.md) | Why the same selected support is essential before checking gradient correctness |

**Provenance:** the manual source is reproduced verbatim from the Phase 1 frozen Git blob `c482d4e01736ed2a5af0e9750e66184e70de2347`. The original seven-test acceptance file is unchanged from Git blob `88cc63aea063b3bb69cffd80cb43e9ede03d0498`. The finite-difference test is new and intentionally isolated from the frozen implementation.

## Verification

Use Python 3.12 and the public repository dependencies:

```bash
python -m pip install -e ".[test]"
python -m pytest tests/test_gathered_sparse_backward.py tests/test_backward_finite_difference.py
```

For the whole public correctness suite, run `python -m pytest tests/`. The [GitHub Actions workflow](../../.github/workflows/reference-tests.yml) performs the CPU test suite on every source/test change.

## What this demonstrates—and what it does not

The tests examine **fixed-selection** gradient agreement at tiny FP64/FP32 shapes, including optional LSE loss and explicit gradient accumulation. The independent finite-difference test probes the derivative of the *published forward objective*, so agreement does not depend exclusively on PyTorch autograd as a comparison mechanism.

They do **not** prove gradients through discrete top-k selection, learned router objectives, efficient fused backward, numerical stability under BF16/FP16, large-context scaling, or performance on any GPU. Those are separate later-stage engineering questions.

A future optimized backward should preserve the same semantic invariants while changing data layout, scheduling and memory traffic.
