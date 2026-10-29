"""An independent numerical-gradient check for the frozen manual backward.

This deliberately uses central finite differences on the public forward reference,
not autograd as the comparison oracle. The selected block IDs remain fixed.
"""

import torch

from msa_lab.reference.gathered_sparse import gathered_sparse_gqa
from msa_lab.reference.gathered_sparse_backward import gathered_sparse_gqa_backward


def test_manual_gradients_match_central_finite_differences() -> None:
    torch.manual_seed(404)
    dtype = torch.float64
    q = torch.randn(1, 3, 2, 2, dtype=dtype)
    k = torch.randn(1, 3, 1, 2, dtype=dtype)
    v = torch.randn(1, 3, 1, 2, dtype=dtype)
    # Different causal, fixed-support rows. Keys 0 and 1 are used repeatedly.
    block_ids = torch.tensor(
        [[[[0, -1]], [[0, 1]], [[1, 2]]]], dtype=torch.int64
    )
    grad_output = torch.randn_like(q)
    grad_lse = torch.randn(1, 3, 2, dtype=dtype)

    manual = gathered_sparse_gqa_backward(
        q, k, v, block_ids, grad_output, grad_lse, block_size=1
    )

    def objective(q_: torch.Tensor, k_: torch.Tensor, v_: torch.Tensor) -> float:
        result = gathered_sparse_gqa(q_, k_, v_, block_ids, block_size=1)
        assert result.lse is not None
        value = (result.output * grad_output).sum() + (
            result.lse * grad_lse
        ).sum()
        return float(value.item())

    eps = 1e-6
    originals = (q, k, v)
    for tensor_idx, (original, gradient) in enumerate(
        zip(originals, (manual.dq, manual.dk, manual.dv))
    ):
        numerical = torch.empty_like(original)
        for coordinate in range(original.numel()):
            plus = original.clone()
            minus = original.clone()
            plus.view(-1)[coordinate] += eps
            minus.view(-1)[coordinate] -= eps
            args_plus = list(originals)
            args_minus = list(originals)
            args_plus[tensor_idx] = plus
            args_minus[tensor_idx] = minus
            numerical.view(-1)[coordinate] = (
                objective(*args_plus) - objective(*args_minus)
            ) / (2 * eps)

        torch.testing.assert_close(gradient, numerical, rtol=1e-6, atol=1e-7)
