from __future__ import annotations

import math

import pytest
import torch

from msa_lab.reference.gathered_sparse import gathered_sparse_gqa
from msa_lab.reference.masked_dense import masked_dense_gqa


def _all_blocks_selected(
    *,
    batch: int,
    seq_len: int,
    hkv: int,
    block_size: int,
) -> torch.Tensor:
    n_blocks = (seq_len + block_size - 1) // block_size
    ids = torch.arange(n_blocks, dtype=torch.int64)
    return ids.view(1, 1, 1, n_blocks).expand(batch, seq_len, hkv, n_blocks).clone()


def _valid_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(1, 5, 4, 3)
    k = torch.randn(1, 5, 2, 3)
    v = torch.randn(1, 5, 2, 3)
    block_ids = _all_blocks_selected(batch=1, seq_len=5, hkv=2, block_size=2)
    return q, k, v, block_ids


def _assert_same(actual, expected, *, rtol: float = 1e-5, atol: float = 1e-6) -> None:
    torch.testing.assert_close(actual.output, expected.output, rtol=rtol, atol=atol)
    assert actual.lse is not None and expected.lse is not None
    torch.testing.assert_close(actual.lse, expected.lse, rtol=rtol, atol=atol)


def test_all_visible_matches_masked_dense() -> None:
    torch.manual_seed(31)
    q, k, v, block_ids = _valid_inputs()

    actual = gathered_sparse_gqa(q, k, v, block_ids, block_size=2)
    expected = masked_dense_gqa(q, k, v, block_ids, block_size=2)

    _assert_same(actual, expected)


def test_asymmetric_sparse_support_matches_masked_dense() -> None:
    torch.manual_seed(32)
    q = torch.randn(1, 6, 4, 3, dtype=torch.float64)
    k = torch.randn(1, 6, 2, 3, dtype=torch.float64)
    v = torch.randn(1, 6, 2, 3, dtype=torch.float64)

    block_ids = torch.full((1, 6, 2, 2), -1, dtype=torch.int64)
    block_ids[..., 0] = 0
    block_ids[0, 4:, 1, 0] = 2

    actual = gathered_sparse_gqa(q, k, v, block_ids, block_size=2)
    expected = masked_dense_gqa(q, k, v, block_ids, block_size=2)

    _assert_same(actual, expected, rtol=1e-10, atol=1e-10)


def test_block_size_one_matches_masked_dense() -> None:
    torch.manual_seed(33)
    q = torch.randn(1, 4, 4, 2, dtype=torch.float64)
    k = torch.randn(1, 4, 2, 2, dtype=torch.float64)
    v = torch.randn(1, 4, 2, 2, dtype=torch.float64)

    block_ids = torch.full((1, 4, 2, 2), -1, dtype=torch.int64)
    block_ids[..., 0] = 0
    block_ids[0, 1, 0] = torch.tensor([0, 1])
    block_ids[0, 2, 0] = torch.tensor([0, 2])
    block_ids[0, 3, 0] = torch.tensor([0, 3])
    block_ids[0, 1, 1, 0] = 1
    block_ids[0, 2, 1, 0] = 1
    block_ids[0, 3, 1, 0] = 2

    actual = gathered_sparse_gqa(q, k, v, block_ids, block_size=1)
    expected = masked_dense_gqa(q, k, v, block_ids, block_size=1)

    _assert_same(actual, expected, rtol=1e-10, atol=1e-10)


def test_unselected_kv_perturbation_is_invariant() -> None:
    torch.manual_seed(34)
    q = torch.randn(1, 5, 4, 3)
    k = torch.randn(1, 5, 2, 3)
    v = torch.randn(1, 5, 2, 3)
    block_ids = torch.zeros((1, 5, 2, 1), dtype=torch.int64)

    base = gathered_sparse_gqa(q, k, v, block_ids, block_size=2)

    k_changed = k.clone()
    v_changed = v.clone()
    k_changed[:, 2:] += 1000
    v_changed[:, 2:] -= 1000
    changed = gathered_sparse_gqa(q, k_changed, v_changed, block_ids, block_size=2)

    _assert_same(changed, base, rtol=0, atol=0)


def test_selection_order_and_future_only_block_are_noops() -> None:
    torch.manual_seed(35)
    q = torch.randn(1, 6, 4, 3)
    k = torch.randn(1, 6, 2, 3)
    v = torch.randn(1, 6, 2, 3)

    a = torch.zeros((1, 6, 2, 3), dtype=torch.int64)
    a[..., 1:] = -1
    b = a.clone()

    a[0, 3, :, :] = torch.tensor([0, 1, -1])
    b[0, 3, :, :] = torch.tensor([1, 0, 2])

    out_a = gathered_sparse_gqa(q, k, v, a, block_size=2)
    out_b = gathered_sparse_gqa(q, k, v, b, block_size=2)

    torch.testing.assert_close(out_a.output[:, 3], out_b.output[:, 3], rtol=1e-6, atol=1e-6)
    assert out_a.lse is not None and out_b.lse is not None
    torch.testing.assert_close(out_a.lse[:, 3], out_b.lse[:, 3], rtol=1e-6, atol=1e-6)


def test_one_token_support_returns_selected_value_and_exact_lse() -> None:
    torch.manual_seed(36)
    q = torch.randn(1, 4, 4, 3, dtype=torch.float64)
    k = torch.randn(1, 4, 2, 3, dtype=torch.float64)
    v = torch.randn(1, 4, 2, 3, dtype=torch.float64)
    block_ids = torch.zeros((1, 4, 2, 1), dtype=torch.int64)

    result = gathered_sparse_gqa(q, k, v, block_ids, block_size=1)
    assert result.lse is not None

    group_size = q.shape[2] // k.shape[2]
    for head in range(q.shape[2]):
        group = head // group_size
        expected_v = v[0, 0, group].expand(q.shape[1], -1)
        torch.testing.assert_close(result.output[0, :, head], expected_v, rtol=0, atol=0)

        expected_lse = (q[0, :, head] * k[0, 0, group]).sum(dim=-1) / math.sqrt(q.shape[-1])
        torch.testing.assert_close(result.lse[0, :, head], expected_lse, rtol=1e-10, atol=1e-10)


def test_reject_rank_mismatch() -> None:
    q, k, v, block_ids = _valid_inputs()
    with pytest.raises(ValueError, match="rank|shape"):
        gathered_sparse_gqa(q.squeeze(0), k, v, block_ids, block_size=2)


def test_reject_nondivisible_gqa_heads() -> None:
    q = torch.randn(1, 4, 3, 2)
    k = torch.randn(1, 4, 2, 2)
    v = torch.randn(1, 4, 2, 2)
    block_ids = _all_blocks_selected(batch=1, seq_len=4, hkv=2, block_size=2)

    with pytest.raises(ValueError, match="divis|head|GQA"):
        gathered_sparse_gqa(q, k, v, block_ids, block_size=2)


def test_reject_qkv_dtype_mismatch() -> None:
    q, k, v, block_ids = _valid_inputs()
    with pytest.raises(ValueError, match="dtype|type"):
        gathered_sparse_gqa(q, k.double(), v, block_ids, block_size=2)


def test_reject_invalid_raw_metadata() -> None:
    q, k, v, block_ids = _valid_inputs()
    duplicate = block_ids.clone()
    duplicate[..., 1] = duplicate[..., 0]

    with pytest.raises(ValueError, match="duplicate"):
        gathered_sparse_gqa(q, k, v, duplicate, block_size=2)


def test_reject_empty_effective_support() -> None:
    q = torch.randn(1, 4, 4, 2)
    k = torch.randn(1, 4, 2, 2)
    v = torch.randn(1, 4, 2, 2)
    block_ids = torch.zeros((1, 4, 2, 1), dtype=torch.int64)
    block_ids[0, 0, 0, 0] = 1

    with pytest.raises(ValueError, match="empty|support"):
        gathered_sparse_gqa(q, k, v, block_ids, block_size=2)
