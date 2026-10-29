from __future__ import annotations

import math

import pytest
import torch

from msa_lab.backends.dense_sdpa import DenseSDPABackend
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


def _manual_repeat_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    support_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent tiny Oracle using expanded K/V and a hand-written mask."""
    _, _, hq, d = q.shape
    hkv = k.shape[2]
    group_size = hq // hkv
    k_expanded = k.repeat_interleave(group_size, dim=2)
    v_expanded = v.repeat_interleave(group_size, dim=2)
    head_mask = support_mask.repeat_interleave(group_size, dim=2)

    scores = torch.einsum("bqhd,bkhd->bhqk", q, k_expanded) / math.sqrt(d)
    scores = scores.masked_fill(~head_mask.permute(0, 2, 1, 3), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    output = torch.einsum("bhqk,bkhd->bqhd", probs, v_expanded)
    lse = torch.logsumexp(scores, dim=-1).permute(0, 2, 1)
    return output, lse


def _valid_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(1, 4, 4, 3)
    k = torch.randn(1, 4, 2, 3)
    v = torch.randn(1, 4, 2, 3)
    block_ids = _all_blocks_selected(batch=1, seq_len=4, hkv=2, block_size=2)
    return q, k, v, block_ids


def test_all_blocks_selected_matches_dense_causal_gqa() -> None:
    torch.manual_seed(7)
    q, k, v, block_ids = _valid_inputs()

    actual = masked_dense_gqa(q, k, v, block_ids, block_size=2)
    dense = DenseSDPABackend()(q, k, v, causal=True, return_lse=True)

    torch.testing.assert_close(actual.output, dense.output, rtol=1e-5, atol=1e-6)
    assert actual.lse is not None and dense.lse is not None
    torch.testing.assert_close(actual.lse, dense.lse, rtol=1e-5, atol=1e-6)


def test_one_block_covering_sequence_matches_dense_causal_gqa() -> None:
    torch.manual_seed(8)
    q = torch.randn(1, 5, 4, 3)
    k = torch.randn(1, 5, 2, 3)
    v = torch.randn(1, 5, 2, 3)
    block_ids = torch.zeros((1, 5, 2, 1), dtype=torch.int64)

    actual = masked_dense_gqa(q, k, v, block_ids, block_size=8)
    dense = DenseSDPABackend()(q, k, v, causal=True, return_lse=True)

    torch.testing.assert_close(actual.output, dense.output, rtol=1e-5, atol=1e-6)
    assert actual.lse is not None and dense.lse is not None
    torch.testing.assert_close(actual.lse, dense.lse, rtol=1e-5, atol=1e-6)


def test_block_size_one_matches_handwritten_token_selection() -> None:
    torch.manual_seed(9)
    q = torch.randn(1, 4, 4, 2, dtype=torch.float64)
    k = torch.randn(1, 4, 2, 2, dtype=torch.float64)
    v = torch.randn(1, 4, 2, 2, dtype=torch.float64)

    block_ids = torch.full((1, 4, 2, 2), -1, dtype=torch.int64)
    block_ids[:, :, :, 0] = 0
    block_ids[0, 1, 0] = torch.tensor([0, 1])
    block_ids[0, 2, 0] = torch.tensor([0, 2])
    block_ids[0, 3, 0] = torch.tensor([0, 3])
    block_ids[0, 1, 1, 0] = 1
    block_ids[0, 2, 1, 0] = 1
    block_ids[0, 3, 1, 0] = 2

    support = torch.zeros((1, 4, 2, 4), dtype=torch.bool)
    support[0, 0, :, 0] = True
    support[0, 1, 0, [0, 1]] = True
    support[0, 2, 0, [0, 2]] = True
    support[0, 3, 0, [0, 3]] = True
    support[0, 1, 1, 1] = True
    support[0, 2, 1, 1] = True
    support[0, 3, 1, 2] = True

    expected_o, expected_lse = _manual_repeat_gqa(q, k, v, support)
    actual = masked_dense_gqa(q, k, v, block_ids, block_size=1)

    torch.testing.assert_close(actual.output, expected_o, rtol=1e-10, atol=1e-10)
    assert actual.lse is not None
    torch.testing.assert_close(actual.lse, expected_lse, rtol=1e-10, atol=1e-10)


def test_output_and_lse_shapes() -> None:
    q = torch.randn(2, 5, 6, 4)
    k = torch.randn(2, 5, 2, 4)
    v = torch.randn(2, 5, 2, 4)
    block_ids = _all_blocks_selected(batch=2, seq_len=5, hkv=2, block_size=2)

    result = masked_dense_gqa(q, k, v, block_ids, block_size=2)

    assert result.output.shape == (2, 5, 6, 4)
    assert result.lse is not None
    assert result.lse.shape == (2, 5, 6)


def test_handwritten_group_mask_matches_independent_oracle() -> None:
    torch.manual_seed(11)
    q = torch.randn(1, 4, 4, 2, dtype=torch.float64)
    k = torch.randn(1, 4, 2, 2, dtype=torch.float64)
    v = torch.randn(1, 4, 2, 2, dtype=torch.float64)

    # q0/q1: both groups use block 0. q2/q3: group 0 uses block 0,
    # group 1 uses block 1. Padding keeps Ksel fixed at 2.
    block_ids = torch.full((1, 4, 2, 2), -1, dtype=torch.int64)
    block_ids[:, :, :, 0] = 0
    block_ids[0, 2:, 1, 0] = 1

    # Independent expected support [B,Nq,Hkv,Nk], written by hand.
    support = torch.zeros((1, 4, 2, 4), dtype=torch.bool)
    support[0, 0, :, 0] = True
    support[0, 1, :, 0:2] = True
    support[0, 2, 0, 0:2] = True
    support[0, 3, 0, 0:2] = True
    support[0, 2, 1, 2] = True
    support[0, 3, 1, 2:4] = True

    expected_o, expected_lse = _manual_repeat_gqa(q, k, v, support)
    actual = masked_dense_gqa(q, k, v, block_ids, block_size=2)

    torch.testing.assert_close(actual.output, expected_o, rtol=1e-10, atol=1e-10)
    assert actual.lse is not None
    torch.testing.assert_close(actual.lse, expected_lse, rtol=1e-10, atol=1e-10)


def test_same_group_heads_share_support_but_can_produce_different_results() -> None:
    # Hkv=1 and Hq=2: both Q heads share the same K/V and support.
    q = torch.tensor(
        [[
            [[1.0, 0.0], [0.0, 2.0]],
            [[1.0, 0.0], [0.0, 2.0]],
            [[1.0, 0.0], [0.0, 2.0]],
        ]]
    )
    k = torch.tensor([[[[2.0, 0.0]], [[0.0, 2.0]], [[1.0, 1.0]]]])
    v = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]], [[2.0, 2.0]]]])
    block_ids = torch.zeros((1, 3, 1, 1), dtype=torch.int64)

    result = masked_dense_gqa(q, k, v, block_ids, block_size=3)

    assert result.lse is not None
    assert not torch.allclose(result.output[0, 2, 0], result.output[0, 2, 1])
    assert not torch.allclose(result.lse[0, 2, 0], result.lse[0, 2, 1])


def test_perturbing_unselected_kv_does_not_change_result() -> None:
    torch.manual_seed(13)
    q = torch.randn(1, 4, 4, 3)
    k = torch.randn(1, 4, 2, 3)
    v = torch.randn(1, 4, 2, 3)

    # Every query/group selects only block 0 -> tokens 0,1.
    block_ids = torch.zeros((1, 4, 2, 1), dtype=torch.int64)
    base = masked_dense_gqa(q, k, v, block_ids, block_size=2)

    k_changed = k.clone()
    v_changed = v.clone()
    k_changed[:, 2:] += 1000
    v_changed[:, 2:] -= 1000
    changed = masked_dense_gqa(q, k_changed, v_changed, block_ids, block_size=2)

    torch.testing.assert_close(base.output, changed.output, rtol=0, atol=0)
    assert base.lse is not None and changed.lse is not None
    torch.testing.assert_close(base.lse, changed.lse, rtol=0, atol=0)


def test_block_order_and_future_only_block_are_semantic_noops() -> None:
    torch.manual_seed(17)
    q = torch.randn(1, 6, 4, 3)
    k = torch.randn(1, 6, 2, 3)
    v = torch.randn(1, 6, 2, 3)

    a = torch.zeros((1, 6, 2, 3), dtype=torch.int64)
    a[..., 1:] = -1
    b = a.clone()

    # At q=3, [0,1] and [1,0] have the same support; block 2 is future-only.
    a[0, 3, :, :] = torch.tensor([0, 1, -1])
    b[0, 3, :, :] = torch.tensor([1, 0, 2])

    out_a = masked_dense_gqa(q, k, v, a, block_size=2)
    out_b = masked_dense_gqa(q, k, v, b, block_size=2)

    torch.testing.assert_close(out_a.output[:, 3], out_b.output[:, 3], rtol=1e-6, atol=1e-6)
    assert out_a.lse is not None and out_b.lse is not None
    torch.testing.assert_close(out_a.lse[:, 3], out_b.lse[:, 3], rtol=1e-6, atol=1e-6)


def test_reject_rank_mismatch() -> None:
    q, k, v, block_ids = _valid_inputs()
    with pytest.raises(ValueError, match="rank|shape"):
        masked_dense_gqa(q.squeeze(0), k, v, block_ids, block_size=2)


def test_reject_nondivisible_gqa_heads() -> None:
    q = torch.randn(1, 4, 3, 2)
    k = torch.randn(1, 4, 2, 2)
    v = torch.randn(1, 4, 2, 2)
    block_ids = _all_blocks_selected(batch=1, seq_len=4, hkv=2, block_size=2)

    with pytest.raises(ValueError, match="divis|head|GQA"):
        masked_dense_gqa(q, k, v, block_ids, block_size=2)


def test_reject_qkv_dtype_mismatch() -> None:
    q, k, v, block_ids = _valid_inputs()
    with pytest.raises(ValueError, match="dtype|type"):
        masked_dense_gqa(q, k.double(), v, block_ids, block_size=2)


def test_reject_block_metadata_alignment_mismatch() -> None:
    q, k, v, _ = _valid_inputs()
    wrong_groups = torch.zeros((1, 4, 1, 1), dtype=torch.int64)

    with pytest.raises(ValueError, match="block|group|Hkv|shape"):
        masked_dense_gqa(q, k, v, wrong_groups, block_size=2)
