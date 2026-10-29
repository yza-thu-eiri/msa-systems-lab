from __future__ import annotations

import math

import torch

from msa_lab.reference.routing_diagnostics import (
    routing_attention_mass_rows_gqa,
    routing_distribution_stats_gqa,
)


def test_uniform_attention_has_hand_computable_router_mass_loss() -> None:
    # N=6, block_size=2, top_k=1，只选 current-local block。
    # FULL attention 为 causal uniform，真正发生稀疏的 q=2..5 coverage 分别为：
    # 1/3, 2/4, 1/5, 2/6；均值 = 41/120，mass loss = 79/120。
    q = torch.zeros(1, 6, 2, 2)
    k = torch.zeros(1, 6, 1, 2)
    local_blocks = (torch.arange(6) // 2).view(1, 6, 1, 1)

    coverage, sparse_active = routing_attention_mass_rows_gqa(
        q,
        k,
        local_blocks,
        block_size=2,
    )

    expected_rows = torch.tensor([1.0, 1.0, 1 / 3, 1 / 2, 1 / 5, 1 / 3])
    torch.testing.assert_close(coverage[0, :, 0], expected_rows)
    torch.testing.assert_close(
        coverage[sparse_active].mean(),
        torch.tensor(41 / 120),
    )
    torch.testing.assert_close(
        (1.0 - coverage[sparse_active]).mean(),
        torch.tensor(79 / 120),
    )
    assert sparse_active[0, :, 0].tolist() == [False, False, True, True, True, True]


def test_uniform_full_causal_distributions_have_hand_computable_stats() -> None:
    q = torch.zeros(1, 3, 2, 2)
    k = torch.zeros(1, 3, 1, 2)
    q_idx = torch.zeros(1, 3, 1, 2)
    k_idx = torch.zeros(1, 3, 1, 2)
    positions = torch.arange(3)
    support = (positions[None, :] <= positions[:, None])[None, :, None, :]

    actual = routing_distribution_stats_gqa(q, k, q_idx, k_idx, support)

    expected_entropy = (0.0 + math.log(2.0) + math.log(3.0)) / 3.0
    expected_top1 = (1.0 + 0.5 + 1.0 / 3.0) / 3.0
    torch.testing.assert_close(
        actual["main_teacher_entropy"],
        torch.tensor(expected_entropy),
    )
    torch.testing.assert_close(
        actual["index_student_entropy"],
        torch.tensor(expected_entropy),
    )
    torch.testing.assert_close(
        actual["main_teacher_top1_mass"],
        torch.tensor(expected_top1),
    )
    torch.testing.assert_close(
        actual["index_student_top1_mass"],
        torch.tensor(expected_top1),
    )
