# P2 — Routing semantics and Main integration

This batch releases a transparent, CPU-tested reference for causal Index scoring, fixed-budget block selection, FULL/SPARSE attention execution, and Routing-KL supervision. It is a correctness baseline, **not** an optimized GPU kernel or a training-quality claim.

## Dataflow and tensor contracts

| Stage | Input → output | Core invariant |
| --- | --- | --- |
| [Index BlockMax](../src/msa_lab/reference/index_blockmax.py) | Index Q [B,N,Hkv,didx], shared Index K [B,N,1,didx] → scores [B,N,Hkv,Nblocks] | Only causal-visible keys contribute. A future-only block scores -inf |
| [Canonical Selection](../src/msa_lab/reference/selection.py) | Scores → block IDs [B,N,Hkv,k] and valid_count [B,N,Hkv] | The current local block has priority; ties resolve by block index, output IDs are sorted with trailing -1 |
| [Causal support](../src/msa_lab/contracts/support.py) | Selected blocks → boolean visible-token support [B,N,Hkv,N] | Block IDs and actual visible token support are not interchangeable |
| [Main integration](../src/msa_lab/model/msa_attention.py) | Input → Main Q/K/V, Index Q/K and output | FULL uses dense causal SDPA; SPARSE uses gathered attention. Both share Main projections |
| [Routing-KL](../src/msa_lab/reference/routing_kl.py) | Teacher/Main scores and student/Index scores over the same support → scalar KL | KL(teacher || student); average GQA teacher probabilities, not logits |
| [Diagnostics](../src/msa_lab/reference/routing_diagnostics.py) | Support and distributions → coverage/distribution statistics | Coverage is not a task-quality metric |

The input tensor to the layer is [B,N,d_model]. The Index branch uses a detached copy of this input, and its Q/K projections share K-index across the KV groups. The selected blocks control **SPARSE Main execution**; FULL Main does not take sparse block IDs.

## Gradient ownership

The Routing-KL objective supplies gradients to Index projection weights, while Main teacher Q/K are detached. The task loss backpropagates through Main attention on the chosen execution path; discrete top-k block IDs are **not differentiated**.

## Public verification

The published CPU tests cover independently hand-computed BlockMax scores, future-only and partial blocks, canonical top-k ordering and support compatibility, KL direction, teacher detachment, Index-only gradients, and same-weight FULL/SPARSE equivalence when the selected support equals the full causal history.

[Run example](../examples/routing_walkthrough.py) · [BlockMax tests](../tests/test_index_blockmax.py) · [Selection tests](../tests/test_selection.py) · [Routing integration tests](../tests/test_msa_routing_integration.py) · [FULL/SPARSE tests](../tests/test_msa_full_sparse.py)

Run the complete public tests with `python -m pytest tests/` after installing the package. The [GitHub Actions run](https://github.com/yza-thu-eiri/msa-systems-lab/actions/runs/40902300189) completed **98 passing CPU tests**, plus both P1 and P2 executable walkthroughs.

## Provenance and limitations

The Index, Selection, Routing-KL, diagnostics and integrated MSA Attention sources are copied from the author's Phase 2 learning laboratory, with their original implementation semantics intact. The published `ModelConfig` is an extracted minimal subset of the private config. The public Routing-KL unit tests are an independently curated subset; the private P2 regression suite included additional training-dependent cases.

The experiments reported in [Case Study 02](case-studies/02-routing-error-and-task-impact.md) remain **recorded internal observations**. This release does not provide real-text training, optimizer/checkpoint/exact-resume workflows, raw routing experiments, multi-layer switch studies, GPU optimization, or evidence of long-context model behavior. Those require later releases.
