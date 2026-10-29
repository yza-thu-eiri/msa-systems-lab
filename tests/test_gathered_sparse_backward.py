from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from msa_lab.reference.gathered_sparse import gathered_sparse_gqa
from msa_lab.reference.gathered_sparse_backward import (
    AttentionGradients,
    gathered_sparse_gqa_backward,
)
from msa_lab.reference.masked_dense import masked_dense_gqa


ForwardFn = Callable[..., object]


def _inputs(*, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(401)
    q = torch.randn(1, 4, 4, 3, dtype=dtype)
    k = torch.randn(1, 4, 2, 3, dtype=dtype)
    v = torch.randn(1, 4, 2, 3, dtype=dtype)

    # block_size=1: token 0 is always visible; later rows also select themselves.
    block_ids = torch.full((1, 4, 2, 2), -1, dtype=torch.int64)
    block_ids[..., 0] = 0
    for query in range(1, 4):
        block_ids[0, query, :, 1] = query

    grad_output = torch.randn_like(q)
    grad_lse = torch.randn(q.shape[:3], dtype=dtype)
    return q, k, v, block_ids, grad_output, grad_lse


def _autograd_oracle(
    forward: ForwardFn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_ids: torch.Tensor,
    grad_output: torch.Tensor,
    grad_lse: torch.Tensor | None,
    *,
    block_size: int,
) -> AttentionGradients:
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    result = forward(q_ref, k_ref, v_ref, block_ids, block_size=block_size)
    loss = (result.output * grad_output).sum()
    if grad_lse is not None:
        assert result.lse is not None
        loss = loss + (result.lse * grad_lse).sum()

    dq, dk, dv = torch.autograd.grad(loss, (q_ref, k_ref, v_ref))
    return AttentionGradients(dq=dq, dk=dk, dv=dv)


def _assert_gradients_close(
    actual: AttentionGradients,
    expected: AttentionGradients,
    *,
    rtol: float,
    atol: float,
) -> None:
    torch.testing.assert_close(actual.dq, expected.dq, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual.dk, expected.dk, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual.dv, expected.dv, rtol=rtol, atol=atol)


def test_float64_output_only_matches_gathered_autograd() -> None:
    q, k, v, block_ids, grad_output, _ = _inputs(dtype=torch.float64)

    actual = gathered_sparse_gqa_backward(
        q,
        k,
        v,
        block_ids,
        grad_output,
        None,
        block_size=1,
    )
    expected = _autograd_oracle(
        gathered_sparse_gqa,
        q,
        k,
        v,
        block_ids,
        grad_output,
        None,
        block_size=1,
    )

    _assert_gradients_close(actual, expected, rtol=1e-9, atol=1e-10)


def test_float64_output_and_lse_match_gathered_autograd() -> None:
    q, k, v, block_ids, grad_output, grad_lse = _inputs(dtype=torch.float64)

    actual = gathered_sparse_gqa_backward(
        q,
        k,
        v,
        block_ids,
        grad_output,
        grad_lse,
        block_size=1,
    )
    expected = _autograd_oracle(
        gathered_sparse_gqa,
        q,
        k,
        v,
        block_ids,
        grad_output,
        grad_lse,
        block_size=1,
    )

    _assert_gradients_close(actual, expected, rtol=1e-9, atol=1e-10)


def test_float32_matches_masked_dense_semantic_oracle() -> None:
    q, k, v, block_ids, grad_output, grad_lse = _inputs(dtype=torch.float32)

    actual = gathered_sparse_gqa_backward(
        q,
        k,
        v,
        block_ids,
        grad_output,
        grad_lse,
        block_size=1,
    )
    expected = _autograd_oracle(
        masked_dense_gqa,
        q,
        k,
        v,
        block_ids,
        grad_output,
        grad_lse,
        block_size=1,
    )

    _assert_gradients_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_support_external_kv_gradients_are_exact_zero() -> None:
    torch.manual_seed(402)
    q = torch.randn(1, 4, 4, 3, dtype=torch.float64)
    k = torch.randn(1, 4, 2, 3, dtype=torch.float64)
    v = torch.randn(1, 4, 2, 3, dtype=torch.float64)
    block_ids = torch.zeros((1, 4, 2, 1), dtype=torch.int64)
    grad_output = torch.randn_like(q)
    grad_lse = torch.randn(q.shape[:3], dtype=q.dtype)

    gradients = gathered_sparse_gqa_backward(
        q,
        k,
        v,
        block_ids,
        grad_output,
        grad_lse,
        block_size=2,
    )

    # Only block 0 is selected, so original token positions 2 and 3 are absent
    # from every effective support row.
    torch.testing.assert_close(gradients.dk[:, 2:], torch.zeros_like(gradients.dk[:, 2:]))
    torch.testing.assert_close(gradients.dv[:, 2:], torch.zeros_like(gradients.dv[:, 2:]))


def test_one_token_support_output_only_has_zero_qk_gradients() -> None:
    torch.manual_seed(403)
    q = torch.randn(1, 4, 4, 3, dtype=torch.float64)
    k = torch.randn(1, 4, 2, 3, dtype=torch.float64)
    v = torch.randn(1, 4, 2, 3, dtype=torch.float64)
    block_ids = torch.zeros((1, 4, 2, 1), dtype=torch.int64)
    grad_output = torch.randn_like(q)

    gradients = gathered_sparse_gqa_backward(
        q,
        k,
        v,
        block_ids,
        grad_output,
        None,
        block_size=1,
    )

    torch.testing.assert_close(gradients.dq, torch.zeros_like(q), rtol=0, atol=0)
    torch.testing.assert_close(gradients.dk, torch.zeros_like(k), rtol=0, atol=0)
    assert torch.count_nonzero(gradients.dv).item() > 0


def test_reject_grad_output_shape_mismatch() -> None:
    q, k, v, block_ids, grad_output, _ = _inputs(dtype=torch.float32)

    with pytest.raises(ValueError, match="grad_output|shape"):
        gathered_sparse_gqa_backward(
            q,
            k,
            v,
            block_ids,
            grad_output[..., :-1],
            None,
            block_size=1,
        )


def test_reject_grad_lse_contract_mismatch() -> None:
    q, k, v, block_ids, grad_output, grad_lse = _inputs(dtype=torch.float32)

    with pytest.raises(ValueError, match="grad_lse|shape|dtype|device"):
        gathered_sparse_gqa_backward(
            q,
            k,
            v,
            block_ids,
            grad_output,
            grad_lse[..., :-1],
            block_size=1,
        )
