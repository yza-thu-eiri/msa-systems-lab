from __future__ import annotations

import math

import pytest
import torch

from msa_lab.reference.index_blockmax import causal_index_blockmax


# P2-C01 的只读验收合同。
#
# 本文件有意在 learner-owned 源码出现前准备。第一次实现尝试前，不得把详细
# expected values 当作答案表。根目录 pytest 配置只发现 tests/；P2-C01 激活且
# scaffold 存在后，才显式运行本文件。


def test_exact_didx1_scores_and_causal_blockmax() -> None:
    # B=1, N=4, Hkv=1, didx=1, Bk=2。
    # q 始终为 1，K token 的值依次为 1、2、3、4。
    # 对每个 query，block 只能使用满足 k <= q 的 key。
    q_idx = torch.ones((1, 4, 1, 1), dtype=torch.float64)
    k_idx = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64).view(
        1, 4, 1, 1
    )

    got = causal_index_blockmax(q_idx, k_idx, block_size=2)

    neg_inf = float("-inf")
    expected = torch.tensor(
        [
            [
                [[1.0, neg_inf]],
                [[2.0, neg_inf]],
                [[2.0, 3.0]],
                [[2.0, 4.0]],
            ]
        ],
        dtype=torch.float64,
    )

    assert got.shape == (1, 4, 1, 2)
    torch.testing.assert_close(got, expected)


def test_scaled_dot_product_divides_by_sqrt_didx() -> None:
    q_idx = torch.tensor([[[[2.0, 0.0]]]], dtype=torch.float64)
    k_idx = torch.tensor([[[[2.0, 0.0]]]], dtype=torch.float64)

    got = causal_index_blockmax(q_idx, k_idx, block_size=1)

    expected = 4.0 / math.sqrt(2.0)
    torch.testing.assert_close(got[0, 0, 0, 0], torch.tensor(expected, dtype=torch.float64))


def test_current_partial_block_ignores_future_token_until_visible() -> None:
    # block 1 包含 token 2、3。token 3 的 score 最大，但对 q=2 仍属于未来，
    # 因此 q=2 只能使用 token 2；到 q=3 时 token 3 才可见。
    q_idx = torch.ones((1, 4, 1, 1), dtype=torch.float64)
    k_idx = torch.tensor([1.0, 2.0, 3.0, 100.0], dtype=torch.float64).view(
        1, 4, 1, 1
    )

    got = causal_index_blockmax(q_idx, k_idx, block_size=2)

    assert got[0, 2, 0, 1].item() == pytest.approx(3.0)
    assert got[0, 3, 0, 1].item() == pytest.approx(100.0)


def test_future_only_block_is_negative_infinity() -> None:
    q_idx = torch.ones((1, 4, 1, 1), dtype=torch.float32)
    k_idx = torch.ones((1, 4, 1, 1), dtype=torch.float32)

    got = causal_index_blockmax(q_idx, k_idx, block_size=2)

    # 在 q=0 时，block 1 只包含 key 2、3，它们都位于未来。
    assert torch.isneginf(got[0, 0, 0, 1])


def test_two_query_groups_score_the_same_shared_kidx_differently() -> None:
    q_idx = torch.zeros((1, 2, 2, 2), dtype=torch.float64)
    q_idx[0, 1, 0] = torch.tensor([1.0, 0.0], dtype=torch.float64)
    q_idx[0, 1, 1] = torch.tensor([0.0, 1.0], dtype=torch.float64)

    # Kidx 恰好有一个共享的 group 轴。
    k_idx = torch.zeros((1, 2, 1, 2), dtype=torch.float64)
    k_idx[0, 1, 0] = torch.tensor([2.0, 5.0], dtype=torch.float64)

    got = causal_index_blockmax(q_idx, k_idx, block_size=1)

    scale = math.sqrt(2.0)
    assert got[0, 1, 0, 1].item() == pytest.approx(2.0 / scale)
    assert got[0, 1, 1, 1].item() == pytest.approx(5.0 / scale)


def test_final_partial_block_is_clipped_to_sequence_length() -> None:
    # N=5, Bk=2 -> block 分别为 [0,1]、[2,3]、[4]。
    q_idx = torch.ones((1, 5, 1, 1), dtype=torch.float64)
    k_idx = torch.tensor([1.0, 2.0, 3.0, 4.0, 9.0], dtype=torch.float64).view(
        1, 5, 1, 1
    )

    got = causal_index_blockmax(q_idx, k_idx, block_size=2)

    assert got.shape == (1, 5, 1, 3)
    assert got[0, 4, 0, 2].item() == pytest.approx(9.0)


def test_output_preserves_input_dtype_and_device() -> None:
    q_idx = torch.ones((2, 3, 2, 4), dtype=torch.float64)
    k_idx = torch.ones((2, 3, 1, 4), dtype=torch.float64)

    got = causal_index_blockmax(q_idx, k_idx, block_size=2)

    assert got.dtype == q_idx.dtype
    assert got.device == q_idx.device
    assert got.shape == (2, 3, 2, 2)


@pytest.mark.parametrize(
    ("q_idx", "k_idx", "block_size", "expected_exception", "match"),
    [
        pytest.param(
            torch.ones((1, 2, 3), dtype=torch.float32),
            torch.ones((1, 2, 1, 3), dtype=torch.float32),
            1,
            ValueError,
            "q_idx|shape",
            id="q-rank",
        ),
        pytest.param(
            torch.ones((1, 2, 1, 3), dtype=torch.float32),
            torch.ones((1, 3, 1, 3), dtype=torch.float32),
            1,
            ValueError,
            "N|shape",
            id="sequence-mismatch",
        ),
        pytest.param(
            torch.ones((1, 2, 1, 3), dtype=torch.float32),
            torch.ones((2, 2, 1, 3), dtype=torch.float32),
            1,
            ValueError,
            "B|shape",
            id="batch-mismatch",
        ),
        pytest.param(
            torch.ones((1, 2, 1, 3), dtype=torch.float32),
            torch.ones((1, 2, 2, 3), dtype=torch.float32),
            1,
            ValueError,
            "k_idx|singleton|轴",
            id="k-group-axis-not-singleton",
        ),
        pytest.param(
            torch.ones((1, 2, 1, 3), dtype=torch.float32),
            torch.ones((1, 2, 1, 4), dtype=torch.float32),
            1,
            ValueError,
            "didx|shape",
            id="didx-mismatch",
        ),
        pytest.param(
            torch.ones((1, 2, 1, 3), dtype=torch.float32),
            torch.ones((1, 2, 1, 3), dtype=torch.float64),
            1,
            ValueError,
            "dtype",
            id="dtype-mismatch",
        ),
        pytest.param(
            torch.ones((1, 2, 1, 3), dtype=torch.int64),
            torch.ones((1, 2, 1, 3), dtype=torch.int64),
            1,
            TypeError,
            "浮点|dtype",
            id="non-floating",
        ),
        pytest.param(
            torch.ones((1, 2, 1, 3), dtype=torch.float32),
            torch.ones((1, 2, 1, 3), dtype=torch.float32),
            0,
            ValueError,
            "block_size|正",
            id="nonpositive-block-size",
        ),
        pytest.param(
            torch.ones((1, 2, 1, 3), dtype=torch.float32),
            torch.ones((1, 2, 1, 3), dtype=torch.float32),
            True,
            TypeError,
            "block_size|整数|bool",
            id="bool-block-size",
        ),
    ],
)
def test_reject_invalid_public_contract(
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    block_size: int,
    expected_exception: type[Exception],
    match: str,
) -> None:
    with pytest.raises(expected_exception, match=match):
        causal_index_blockmax(q_idx, k_idx, block_size=block_size)
