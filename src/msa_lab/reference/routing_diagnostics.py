"""C06 训练监控：直接解释 Main teacher 与 Index student 的分布形态。"""

from __future__ import annotations

import torch

from msa_lab.contracts.support import block_ids_to_group_mask


def routing_attention_mass_rows_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    block_ids: torch.Tensor,
    *,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 Router support 覆盖的 Full Main attention mass 与 sparse-active mask。

    输入：
        q: ``[B, N, Hq, D]``
        k: ``[B, N, Hkv, D]``
        block_ids: ``[B, N, Hkv, top_k]``

    输出：
        coverage: ``[B, N, Hkv]``，每个 Router row 在 Full Main 中保留的概率质量。
        sparse_active: 同 shape bool；只标记 ``visible_block_count > top_k`` 的 row。

    GQA 语义与 Routing-KL teacher 一致：每个 Q head 先独立 softmax，再在同一
    KV group 内平均概率。统计使用 FP32，避免 bf16 reduction 掩盖小差异。
    """
    batch_size, sequence_length, hq, head_dim = q.shape
    _, _, hkv, _ = k.shape
    group_size = hq // hkv

    q_grouped = q.float().reshape(
        batch_size,
        sequence_length,
        hkv,
        group_size,
        head_dim,
    )
    main_logits = torch.einsum(
        "bqhgd,bkhd->bqhgk",
        q_grouped,
        k.float(),
    ) / head_dim**0.5

    positions = torch.arange(sequence_length, device=q.device)
    causal_support = positions[None, :] <= positions[:, None]
    main_logits = main_logits.masked_fill(
        ~causal_support[None, :, None, None, :],
        float("-inf"),
    )
    teacher_probability = torch.softmax(main_logits, dim=-1).mean(dim=3)

    selected_support = block_ids_to_group_mask(
        block_ids,
        seq_len=sequence_length,
        block_size=block_size,
    )
    coverage = (
        teacher_probability * selected_support.to(teacher_probability.dtype)
    ).sum(dim=-1)

    top_k = block_ids.shape[-1]
    visible_block_count = positions // block_size + 1
    sparse_active = (visible_block_count > top_k)[None, :, None].expand(
        batch_size,
        -1,
        hkv,
    )
    return coverage, sparse_active


def routing_distribution_stats_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    token_support: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """在同一 token support 上返回 teacher/student 熵与 top-1 mass。

    这是 C06 的低频可解释性诊断，不替代冻结的 C04 Routing-KL Reference。
    所有 scalar 先在 token 轴计算，再对有效 ``[B, query, hkv]`` row 平均。
    """
    batch_size, sequence_length, hq, d_head = q.shape
    _, _, hkv, _ = k.shape
    group_size = hq // hkv
    q_grouped = q.reshape(
        batch_size,
        sequence_length,
        hkv,
        group_size,
        d_head,
    )

    main_logits = torch.einsum(
        "bqhgd,bkhd->bqhgk",
        q_grouped,
        k,
    ) / d_head**0.5
    index_dim = q_idx.shape[-1]
    index_logits = torch.einsum(
        "bqhd,bkd->bqhk",
        q_idx,
        k_idx.squeeze(2),
    ) / index_dim**0.5

    row_valid = token_support.any(dim=-1)
    main_support = token_support.unsqueeze(3)
    masked_main_logits = main_logits.masked_fill(~main_support, float("-inf"))
    masked_index_logits = index_logits.masked_fill(~token_support, float("-inf"))
    masked_main_logits = torch.where(
        row_valid.unsqueeze(-1).unsqueeze(-1),
        masked_main_logits,
        torch.zeros_like(masked_main_logits),
    )
    masked_index_logits = torch.where(
        row_valid.unsqueeze(-1),
        masked_index_logits,
        torch.zeros_like(masked_index_logits),
    )

    teacher_prob = torch.softmax(masked_main_logits, dim=-1).mean(dim=3).detach()
    student_prob = torch.softmax(masked_index_logits, dim=-1)
    # mean-of-softmax teacher 的 log 不能由 mean(log-softmax) 代替。
    safe_teacher_log = torch.where(
        teacher_prob > 0,
        teacher_prob.log(),
        torch.zeros_like(teacher_prob),
    )
    student_log = torch.log_softmax(masked_index_logits, dim=-1)

    safe_teacher_log = safe_teacher_log.masked_fill(~token_support, 0.0)
    student_log = student_log.masked_fill(~token_support, 0.0)
    teacher_entropy = -(teacher_prob * safe_teacher_log).sum(dim=-1)
    student_entropy = -(student_prob * student_log).sum(dim=-1)
    teacher_top1 = teacher_prob.masked_fill(~token_support, 0.0).amax(dim=-1)
    student_top1 = student_prob.masked_fill(~token_support, 0.0).amax(dim=-1)

    row_weight = row_valid.to(dtype=teacher_entropy.dtype)
    denominator = row_weight.sum().clamp_min(1.0)

    def reduce_rows(value: torch.Tensor) -> torch.Tensor:
        return (value * row_weight).sum() / denominator

    return {
        "main_teacher_entropy": reduce_rows(teacher_entropy),
        "index_student_entropy": reduce_rows(student_entropy),
        "main_teacher_top1_mass": reduce_rows(teacher_top1),
        "index_student_top1_mass": reduce_rows(student_top1),
    }
