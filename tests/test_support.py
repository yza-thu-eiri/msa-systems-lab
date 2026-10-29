from __future__ import annotations

import pytest
import torch

from msa_lab.contracts.support import (
    block_ids_to_group_mask,
    validate_fixed_block_ids,
)


def test_current_block_keeps_only_causal_prefix() -> None:
    # N=10, Bk=4: block 1 -> tokens 4,5,6,7; block 2 -> 8,9.
    block_ids = torch.full((1, 10, 1, 2), -1, dtype=torch.int64)
    block_ids[:, :, :, 0] = 0  # keep every early query valid
    block_ids[0, 5, 0] = torch.tensor([1, 2])

    mask = block_ids_to_group_mask(block_ids, seq_len=10, block_size=4)

    assert mask.shape == (1, 10, 1, 10)
    assert mask.dtype == torch.bool
    assert torch.equal(
        torch.where(mask[0, 5, 0])[0],
        torch.tensor([4, 5]),
    )


def test_partial_last_block_clips_to_sequence_length() -> None:
    # N=10, Bk=4: final block 2 contains only tokens 8,9.
    block_ids = torch.zeros((1, 10, 1, 1), dtype=torch.int64)
    block_ids[0, 9, 0, 0] = 2

    mask = block_ids_to_group_mask(block_ids, seq_len=10, block_size=4)

    assert torch.equal(
        torch.where(mask[0, 9, 0])[0],
        torch.tensor([8, 9]),
    )


def test_two_gqa_groups_can_have_different_support() -> None:
    block_ids = torch.zeros((1, 6, 2, 2), dtype=torch.int64)
    block_ids[..., 1] = -1
    block_ids[0, 4, 0] = torch.tensor([0, 2])
    block_ids[0, 4, 1] = torch.tensor([1, 2])

    mask = block_ids_to_group_mask(block_ids, seq_len=6, block_size=2)

    assert torch.equal(torch.where(mask[0, 4, 0])[0], torch.tensor([0, 1, 4]))
    assert torch.equal(torch.where(mask[0, 4, 1])[0], torch.tensor([2, 3, 4]))


def test_block_order_does_not_change_mathematical_support() -> None:
    a = torch.tensor([[[[0, 1]]]], dtype=torch.int64).expand(1, 4, 1, 2).clone()
    b = torch.tensor([[[[1, 0]]]], dtype=torch.int64).expand(1, 4, 1, 2).clone()

    mask_a = block_ids_to_group_mask(a, seq_len=4, block_size=2)
    mask_b = block_ids_to_group_mask(b, seq_len=4, block_size=2)

    assert torch.equal(mask_a, mask_b)


def test_future_only_block_is_a_semantic_noop_when_other_support_exists() -> None:
    base = torch.zeros((1, 6, 1, 2), dtype=torch.int64)
    base[..., 1] = -1
    with_future = base.clone()
    with_future[0, 3, 0] = torch.tensor([0, 2])  # block 2 -> tokens 4,5, future for q=3

    base_mask = block_ids_to_group_mask(base, seq_len=6, block_size=2)
    future_mask = block_ids_to_group_mask(with_future, seq_len=6, block_size=2)

    assert torch.equal(base_mask[0, 3, 0], future_mask[0, 3, 0])


@pytest.mark.parametrize(
    ("block_ids", "seq_len", "block_size", "match"),
    [
        (torch.tensor([[[[0, 0]]]], dtype=torch.int64), 1, 1, "duplicate"),
        (torch.tensor([[[[0, 2]]]], dtype=torch.int64), 1, 1, "range"),
        (torch.tensor([[[[-1, 0]]]], dtype=torch.int64), 1, 1, "padding"),
        (torch.tensor([[[[-2, -1]]]], dtype=torch.int64), 1, 1, "padding|negative"),
        (torch.full((1, 1, 1, 2), -1, dtype=torch.int64), 1, 1, "empty|selection|support"),
        (torch.zeros((1, 2, 1), dtype=torch.int64), 2, 1, "rank|shape"),
        (torch.zeros((1, 1, 1, 1), dtype=torch.float32), 1, 1, "dtype|integer"),
        (torch.zeros((1, 0, 1, 1), dtype=torch.int64), 0, 1, "seq_len|positive"),
        (torch.zeros((1, 1, 1, 1), dtype=torch.int64), 1, 0, "block_size|positive"),
    ],
)
def test_reject_invalid_raw_metadata(
    block_ids: torch.Tensor,
    seq_len: int,
    block_size: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        validate_fixed_block_ids(
            block_ids,
            seq_len=seq_len,
            block_size=block_size,
        )


@pytest.mark.parametrize(
    ("block_ids", "seq_len", "block_size", "match"),
    [
        (torch.tensor([[[[0, 0]]]], dtype=torch.int64), 1, 1, "duplicate"),
        (torch.tensor([[[[-1, 0]]]], dtype=torch.int64), 1, 1, "padding"),
        (torch.tensor([[[[0, 2]]]], dtype=torch.int64), 1, 1, "range"),
        (torch.zeros((1, 1, 1, 1), dtype=torch.float32), 1, 1, "dtype|integer"),
    ],
)
def test_group_mask_entrypoint_rejects_invalid_raw_metadata(
    block_ids: torch.Tensor,
    seq_len: int,
    block_size: int,
    match: str,
) -> None:
    # The public operation must enforce its own input contract. Callers should not
    # be able to bypass validation by forgetting a separate validator call.
    with pytest.raises(ValueError, match=match):
        block_ids_to_group_mask(
            block_ids,
            seq_len=seq_len,
            block_size=block_size,
        )


def test_reject_shape_mismatch() -> None:
    block_ids = torch.zeros((1, 2, 1, 1), dtype=torch.int64)
    with pytest.raises(ValueError, match="query|shape|sequence"):
        validate_fixed_block_ids(block_ids, seq_len=3, block_size=1)


def test_reject_all_future_only_effective_support() -> None:
    # q=0 selects only block 1 (tokens 2,3), so effective support is empty.
    block_ids = torch.zeros((1, 4, 1, 1), dtype=torch.int64)
    block_ids[0, 0, 0, 0] = 1

    with pytest.raises(ValueError, match="empty|support"):
        block_ids_to_group_mask(block_ids, seq_len=4, block_size=2)
