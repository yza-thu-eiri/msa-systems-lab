from __future__ import annotations

import math

import pytest
import torch

from msa_lab.contracts.support import (
    block_ids_to_group_mask,
    validate_fixed_block_ids,
)
from msa_lab.reference.selection import select_topk_blocks


def _selection_oracle(
    midx: torch.Tensor,
    *,
    block_size: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """用逐 row Python 逻辑表达 C02 合同，不复用被测 Tensor 排序实现。"""
    batch_size, sequence_length, num_kv_groups, _ = midx.shape
    expected_ids = torch.full(
        (batch_size, sequence_length, num_kv_groups, k),
        -1,
        dtype=torch.int64,
        device=midx.device,
    )
    expected_count = torch.zeros(
        (batch_size, sequence_length, num_kv_groups),
        dtype=torch.int64,
        device=midx.device,
    )

    for batch_idx in range(batch_size):
        for query_idx in range(sequence_length):
            local_block_id = query_idx // block_size
            for group_idx in range(num_kv_groups):
                score_row = midx[batch_idx, query_idx, group_idx]
                visible_ids = [
                    block_id
                    for block_id, score in enumerate(score_row.tolist())
                    if score != float("-inf")
                ]
                if local_block_id not in visible_ids:
                    visible_ids.append(local_block_id)

                ranked_ids = sorted(
                    visible_ids,
                    key=lambda block_id: (
                        block_id != local_block_id,
                        -float(score_row[block_id].item()),
                        block_id,
                    ),
                )
                selected_ids = sorted(ranked_ids[:k])
                valid_count = len(selected_ids)

                expected_ids[
                    batch_idx,
                    query_idx,
                    group_idx,
                    :valid_count,
                ] = torch.tensor(
                    selected_ids,
                    dtype=torch.int64,
                    device=midx.device,
                )
                expected_count[batch_idx, query_idx, group_idx] = valid_count

    return expected_ids, expected_count


def _assert_canonical_contract(
    block_ids: torch.Tensor,
    valid_count: torch.Tensor,
    *,
    sequence_length: int,
    block_size: int,
) -> None:
    """独立检查 canonical prefix、padding、distinct 与 current-local 不变量。"""
    num_blocks = math.ceil(sequence_length / block_size)
    rows = block_ids.reshape(-1, block_ids.shape[-1])
    counts = valid_count.reshape(-1)

    for row_index, row in enumerate(rows):
        count = int(counts[row_index].item())
        valid_ids = row[:count].tolist()
        padding = row[count:].tolist()

        assert 1 <= count <= block_ids.shape[-1]
        assert valid_ids == sorted(valid_ids)
        assert len(valid_ids) == len(set(valid_ids))
        assert all(0 <= block_id < num_blocks for block_id in valid_ids)
        assert padding == [-1] * (block_ids.shape[-1] - count)

    for query_idx in range(sequence_length):
        local_block_id = query_idx // block_size
        assert torch.all(
            (block_ids[:, query_idx] == local_block_id).sum(dim=-1) == 1
        )


def test_hand_computable_selection() -> None:
    neg_inf = float("-inf")
    midx = torch.tensor(
        [
            [
                [[0.7, neg_inf, neg_inf]],
                [[0.4, neg_inf, neg_inf]],
                [[0.9, 0.2, neg_inf]],
                [[0.3, 0.8, neg_inf]],
                [[0.6, 0.6, 0.5]],
            ]
        ],
        dtype=torch.float64,
    )

    block_ids, valid_count = select_topk_blocks(midx, block_size=2, k=2)

    expected_ids = torch.tensor(
        [[[[0, -1]], [[0, -1]], [[0, 1]], [[0, 1]], [[0, 2]]]],
        dtype=torch.int64,
    )
    expected_count = torch.tensor([[[1], [1], [2], [2], [2]]])
    torch.testing.assert_close(block_ids, expected_ids)
    torch.testing.assert_close(valid_count, expected_count)


def test_k1_budget_is_exactly_current_local() -> None:
    midx = torch.tensor(
        [
            [
                [[4.0, float("-inf"), float("-inf")]],
                [[3.0, float("-inf"), float("-inf")]],
                [[100.0, -5.0, float("-inf")]],
                [[100.0, -6.0, float("-inf")]],
                [[100.0, 90.0, -7.0]],
            ]
        ]
    )

    block_ids, valid_count = select_topk_blocks(midx, block_size=2, k=1)

    assert block_ids[0, :, 0, 0].tolist() == [0, 0, 1, 1, 2]
    assert valid_count[0, :, 0].tolist() == [1, 1, 1, 1, 1]


def test_local_replaces_one_ranked_slot_without_duplication() -> None:
    midx = torch.tensor(
        [
            [
                [[1.0, float("-inf"), float("-inf")]],
                [[1.0, float("-inf"), float("-inf")]],
                [[5.0, 4.0, float("-inf")]],
                [[5.0, 4.0, float("-inf")]],
                [[100.0, 90.0, -10.0]],
            ]
        ]
    )

    block_ids, valid_count = select_topk_blocks(midx, block_size=2, k=2)

    assert block_ids[0, 4, 0].tolist() == [0, 2]
    assert valid_count[0, 4, 0].item() == 2


def test_equal_relevance_prefers_smaller_block_id_at_budget_boundary() -> None:
    midx = torch.tensor(
        [
            [
                [[1.0, float("-inf"), float("-inf")]],
                [[1.0, float("-inf"), float("-inf")]],
                [[1.0, 1.0, float("-inf")]],
                [[1.0, 1.0, float("-inf")]],
                [[7.0, 7.0, -3.0]],
            ]
        ]
    )

    block_ids, _ = select_topk_blocks(midx, block_size=2, k=2)

    # local block 2 占一个位置；同分的 0/1 中必须选择较小 ID 0。
    assert block_ids[0, 4, 0].tolist() == [0, 2]


def test_multiple_batches_and_groups_match_independent_oracle() -> None:
    batch_size, sequence_length, num_kv_groups = 2, 7, 3
    block_size, k = 3, 3
    num_blocks = math.ceil(sequence_length / block_size)
    generator = torch.Generator().manual_seed(20260907)
    # 使用小整数分数主动制造 tie，而不是依赖随机浮点几乎不相等的情况。
    midx = torch.randint(
        -2,
        4,
        (batch_size, sequence_length, num_kv_groups, num_blocks),
        generator=generator,
    ).to(torch.float64)

    for query_idx in range(sequence_length):
        first_future_block = query_idx // block_size + 1
        midx[:, query_idx, :, first_future_block:] = float("-inf")

    got_ids, got_count = select_topk_blocks(
        midx,
        block_size=block_size,
        k=k,
    )
    expected_ids, expected_count = _selection_oracle(
        midx,
        block_size=block_size,
        k=k,
    )

    torch.testing.assert_close(got_ids, expected_ids)
    torch.testing.assert_close(got_count, expected_count)
    _assert_canonical_contract(
        got_ids,
        got_count,
        sequence_length=sequence_length,
        block_size=block_size,
    )


@pytest.mark.parametrize(
    ("sequence_length", "block_size", "k"),
    [
        pytest.param(1, 1, 1, id="single-token"),
        pytest.param(2, 1, 2, id="two-singleton-blocks"),
        pytest.param(3, 2, 2, id="partial-last-block"),
        pytest.param(5, 2, 3, id="three-block-budget"),
        pytest.param(7, 3, 1, id="local-only-budget"),
        pytest.param(7, 3, 3, id="all-block-budget"),
        pytest.param(8, 4, 2, id="exact-block-geometry"),
        pytest.param(9, 4, 3, id="partial-three-block-geometry"),
    ],
)
def test_geometry_matrix_matches_independent_oracle(
    sequence_length: int,
    block_size: int,
    k: int,
) -> None:
    num_blocks = math.ceil(sequence_length / block_size)
    generator = torch.Generator().manual_seed(
        sequence_length * 100 + block_size * 10 + k
    )
    midx = torch.randint(
        -1,
        3,
        (2, sequence_length, 2, num_blocks),
        generator=generator,
    ).to(torch.float32)
    for query_idx in range(sequence_length):
        first_future_block = query_idx // block_size + 1
        midx[:, query_idx, :, first_future_block:] = float("-inf")

    got_ids, got_count = select_topk_blocks(
        midx,
        block_size=block_size,
        k=k,
    )
    expected_ids, expected_count = _selection_oracle(
        midx,
        block_size=block_size,
        k=k,
    )

    torch.testing.assert_close(got_ids, expected_ids)
    torch.testing.assert_close(got_count, expected_count)
    _assert_canonical_contract(
        got_ids,
        got_count,
        sequence_length=sequence_length,
        block_size=block_size,
    )


def test_output_is_compatible_with_frozen_phase1_support() -> None:
    sequence_length, block_size, k = 7, 3, 3
    num_blocks = math.ceil(sequence_length / block_size)
    midx = torch.full(
        (2, sequence_length, 2, num_blocks),
        float("-inf"),
    )
    for query_idx in range(sequence_length):
        visible_blocks = query_idx // block_size + 1
        midx[:, query_idx, :, :visible_blocks] = torch.arange(
            visible_blocks,
            dtype=midx.dtype,
        )

    block_ids, valid_count = select_topk_blocks(
        midx,
        block_size=block_size,
        k=k,
    )

    validate_fixed_block_ids(
        block_ids,
        seq_len=sequence_length,
        block_size=block_size,
    )
    support = block_ids_to_group_mask(
        block_ids,
        seq_len=sequence_length,
        block_size=block_size,
    )

    assert support.shape == (2, sequence_length, 2, sequence_length)
    for query_idx in range(sequence_length):
        assert torch.all(support[:, query_idx, :, query_idx])
    assert torch.equal(valid_count, (block_ids >= 0).sum(dim=-1))


def test_output_dtype_device_and_shapes() -> None:
    midx = torch.full((2, 5, 3, 3), float("-inf"), dtype=torch.float32)
    for query_idx in range(5):
        visible_blocks = query_idx // 2 + 1
        midx[:, query_idx, :, :visible_blocks] = 0.0
    block_ids, valid_count = select_topk_blocks(midx, block_size=2, k=2)

    assert block_ids.shape == (2, 5, 3, 2)
    assert valid_count.shape == (2, 5, 3)
    assert block_ids.dtype == torch.int64
    assert valid_count.dtype == torch.int64
    assert block_ids.device == midx.device
    assert valid_count.device == midx.device


@pytest.mark.parametrize(
    ("midx", "block_size", "k", "match"),
    [
        pytest.param(
            torch.zeros((1, 4, 1, 2)),
            0,
            1,
            "block_size",
            id="nonpositive-block-size",
        ),
        pytest.param(
            torch.zeros((1, 4, 1, 2)),
            2,
            0,
            "k",
            id="k-below-range",
        ),
        pytest.param(
            torch.zeros((1, 4, 1, 2)),
            2,
            3,
            "k",
            id="k-above-range",
        ),
        pytest.param(
            torch.zeros((1, 5, 1, 4)),
            2,
            1,
            "Nb",
            id="geometry-mismatch",
        ),
    ],
)
def test_minimal_validation_boundary(
    midx: torch.Tensor,
    block_size: int,
    k: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        select_topk_blocks(midx, block_size=block_size, k=k)
