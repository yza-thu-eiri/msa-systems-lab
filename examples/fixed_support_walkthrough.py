"""60-second CPU walkthrough of sparse GQA forward and fixed-support backward.

Run after installation:
    python examples/fixed_support_walkthrough.py

This is a numerical correctness demo, not a throughput benchmark.
"""

import torch

from msa_lab.contracts.support import block_ids_to_group_mask
from msa_lab.reference.gathered_sparse import gathered_sparse_gqa
from msa_lab.reference.gathered_sparse_backward import gathered_sparse_gqa_backward
from msa_lab.reference.masked_dense import masked_dense_gqa


def main() -> None:
    torch.manual_seed(19)
    dtype = torch.float64
    q = torch.randn(1, 4, 4, 3, dtype=dtype)
    k = torch.randn(1, 4, 2, 3, dtype=dtype)
    v = torch.randn(1, 4, 2, 3, dtype=dtype)
    block_size = 2

    # Two KV groups. After q=2, group 1 chooses the newer block;
    # at q=3, group 0 selects both blocks.
    block_ids = torch.full((1, 4, 2, 2), -1, dtype=torch.int64)
    block_ids[..., 0] = 0
    block_ids[0, 2:, 1, 0] = 1
    block_ids[0, 3, 0, 1] = 1

    support = block_ids_to_group_mask(
        block_ids, seq_len=4, block_size=block_size
    )
    dense = masked_dense_gqa(q, k, v, block_ids, block_size=block_size)
    sparse = gathered_sparse_gqa(q, k, v, block_ids, block_size=block_size)
    assert dense.lse is not None and sparse.lse is not None

    forward_error = (dense.output - sparse.output).abs().max().item()
    lse_error = (dense.lse - sparse.lse).abs().max().item()

    grad_output = torch.randn_like(q)
    grad_lse = torch.randn_like(sparse.lse)
    manual = gathered_sparse_gqa_backward(
        q, k, v, block_ids, grad_output, grad_lse, block_size=block_size
    )

    # Independent autograd oracle: differentiate the masked-dense path.
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    expected = masked_dense_gqa(
        q_ref, k_ref, v_ref, block_ids, block_size=block_size
    )
    assert expected.lse is not None
    objective = (expected.output * grad_output).sum()
    objective = objective + (expected.lse * grad_lse).sum()
    expected_gradients = torch.autograd.grad(
        objective, (q_ref, k_ref, v_ref)
    )
    grad_errors = [
        (actual - reference).abs().max().item()
        for actual, reference in zip(
            (manual.dq, manual.dk, manual.dv), expected_gradients
        )
    ]

    print("Selected causal keys at query=3:")
    for group in range(k.shape[2]):
        key_ids = torch.where(support[0, 3, group])[0].tolist()
        print(f"  KV group {group}: {key_ids}")
    print(f"max |O_masked - O_gathered|: {forward_error:.3e}")
    print(f"max |LSE_masked - LSE_gathered|: {lse_error:.3e}")
    print(
        "max gradient errors [dQ, dK, dV]: "
        + ", ".join(f"{error:.3e}" for error in grad_errors)
    )

    assert forward_error < 1e-10
    assert lse_error < 1e-10
    assert max(grad_errors) < 1e-10
    print("PASS: fixed-support forward and backward numerical parity")


if __name__ == "__main__":
    main()
