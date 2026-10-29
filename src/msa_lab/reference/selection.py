"""Phase 2 P2-C02 canonical block selection learner Reference。

本文件实现 learner 与 coach 共同确认的接口、数据流和 semantic regions。
"""

from __future__ import annotations

import torch


def select_topk_blocks(
    midx: torch.Tensor,
    *,
    block_size: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把 causal BlockMax 分数转换为 canonical selection metadata。

    输入：
        midx: [B,N,Hkv,Nb]

    输出：
        block_ids:   [B,N,Hkv,k]
        valid_count: [B,N,Hkv]

    已共同确认的数据流：

    ```text
    Midx
    -> current-local ranking score = +inf
    -> stable descending relevance sort
    -> take first k values / block indices
    -> selected value == -inf means invalid slot
    -> valid_count
    -> invalid block ID -> Nb sentinel
    -> ascending block-ID canonical sort
    -> Nb sentinel -> -1 padding
    ```

    本章只负责 selection metadata；C03 integration、KL、training 与 kernel
    均不在当前范围内。
    """

    # ============================================================
    # Minimal contract boundary
    # ============================================================

    # 这是课程内部 transparent Reference。midx 来自已完成的 P2-C01，因此默认
    # 信任其基础 tensor contract，不重复写 dtype/device/rank/每维正数等 defensive
    # validation。这里只检查会直接改变 selection 数学语义的参数/geometry：
    #
    batch_size, sequence_length, num_kv_groups, num_blocks = midx.shape

    if block_size <= 0:
        raise ValueError("block_size 必须大于 0")
    if not 1 <= k <= num_blocks:
        raise ValueError("k 必须满足 1 <= k <= Nb")

    expected_num_blocks = (
        sequence_length + block_size - 1
    ) // block_size
    if num_blocks != expected_num_blocks:
        raise ValueError("Nb 必须等于 ceil(N / block_size)")

    # ============================================================
    # C02-A — local-aware relevance ranking
    # ============================================================

    # 目标：
    #
    # midx [B,N,Hkv,Nb]
    # -> 每个 query 找到 current local block
    # -> 仅在 ranking 用临时 score 中把 local 改成 +inf
    # -> stable descending relevance sort
    # -> top_values / top_ids [B,N,Hkv,k]
    #
    # Tie contract：原始 block 轴按 ID 0..Nb-1 排列；stable descending
    # sort 在 relevance 相同时保留原始顺序，因此较小 block ID 优先。

    ranking_score = midx.clone()

    # 每个 query 的 local block ID，shape [N]，值域为 0..Nb-1。
    query_positions = torch.arange(sequence_length, device=midx.device)
    local_block_ids = query_positions // block_size

    # 为每个 [b,q,hkv] row 提供一个要写入的 block 轴位置。
    local_index = local_block_ids[None, :, None, None].expand(
        batch_size, sequence_length, num_kv_groups, 1,
    )

    # 只改变 ranking 用临时分数，原始 midx 保持不变。
    ranking_score = ranking_score.scatter(
        dim=-1,
        index=local_index,
        value=float("inf"),
    )

    # 原 block 轴按 ID 升序排列；stable sort 使 relevance tie 时小 ID 优先。
    ranked_values, ranked_ids = torch.sort(
        input=ranking_score,
        dim=-1,
        descending=True,
        stable=True,
    )

    top_values = ranked_values[..., :k]
    top_ids = ranked_ids[..., :k]

    # ============================================================
    # C02-B — real validity + valid_count
    # ============================================================

    # C01 已保证 future-only block score == -inf。
    # 因此即使所有 row 固定取 k 个位置，early query 中超出真实 visible-count
    # 的位置仍会保留 -inf，可据此识别 invalid slot。
    #
    # 注意：local 的 ranking value 被人为设为 +inf，但 local 本身是合法的；
    # 不要简单使用“是否 finite”来定义 validity。

    # local 的 +inf 是合法值；只有 -inf 才表示 invalid selected slot。
    invalid_mask = torch.isneginf(top_values)

    valid_count = (~invalid_mask).sum(
        dim=-1,
        keepdim=False,
    )

    # ============================================================
    # C02-C — canonical block metadata
    # ============================================================

    # top_ids 此时仍然是 relevance ranking order。
    # Phase 1 consumer 要求：
    #
    # valid block IDs 升序
    # +
    # invalid tail = -1
    #
    # 例如：
    #
    # ranking result: [4, 1, 3, invalid]
    # canonical:      [1, 3, 4, -1]

    # 合法 ID 最大为 num_blocks - 1，因此 sentinel 会在升序排序后落到尾部。
    sentinel_id = num_blocks

    ids_with_sentinel = top_ids.masked_fill(
        mask=invalid_mask,
        value=sentinel_id,
    )

    canonical_ids = ids_with_sentinel.sort(
        dim=-1,
        descending=False,
    ).values

    block_ids = canonical_ids.masked_fill(
        mask=canonical_ids == sentinel_id,
        value=-1,
    )

    # ============================================================
    # Final contract
    # ============================================================

    return block_ids, valid_count
