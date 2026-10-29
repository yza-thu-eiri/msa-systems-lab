"""Phase 2 的透明 Index score / BlockMax Reference。

文件保留完整 C01 数据流与 checkpoint 边界，供后续章节复用和恢复理解。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def causal_index_blockmax(
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    *,
    block_size: int,
) -> torch.Tensor:
    """校验公开的 Index 输入并返回 causal BlockMax。

    完整 C01 scaffold：

    ```text
    q_idx [B,N,Hkv,didx]
    k_idx [B,N,1,didx]
    -> C01-A token_score [B,N,Hkv,N]
    -> C01-B1 causal_token_score [B,N,Hkv,N]
    -> C01-B2 block_score [B,N,Hkv,num_blocks]
    -> C01-C learning annotation
    ```

    C01-A、B1 与 B2 已完成；C01-C 保留语义注释供后续恢复理解。
    """

    if not isinstance(q_idx, torch.Tensor):
        raise TypeError("q_idx 必须是 torch.Tensor")
    if not isinstance(k_idx, torch.Tensor):
        raise TypeError("k_idx 必须是 torch.Tensor")

    if q_idx.ndim != 4:
        raise ValueError("q_idx 的 shape 必须是 [B,N,Hkv,didx]")
    if k_idx.ndim != 4:
        raise ValueError("k_idx 的 shape 必须是 [B,N,1,didx]")

    if not torch.is_floating_point(q_idx):
        raise TypeError("q_idx 必须使用浮点 dtype")
    if not torch.is_floating_point(k_idx):
        raise TypeError("k_idx 必须使用浮点 dtype")
    if q_idx.dtype != k_idx.dtype:
        raise ValueError("q_idx 与 k_idx 必须使用相同 dtype")
    if q_idx.device != k_idx.device:
        raise ValueError("q_idx 与 k_idx 必须位于同一 device")

    batch_size, sequence_length, num_kv_groups, index_dim = q_idx.shape
    if min(batch_size, sequence_length, num_kv_groups, index_dim) <= 0:
        raise ValueError("B、N、Hkv 与 didx 都必须为正数")

    expected_k_idx_shape = (batch_size, sequence_length, 1, index_dim)
    if tuple(k_idx.shape) != expected_k_idx_shape:
        raise ValueError(
            "k_idx 的 B、N 与 didx 必须和 q_idx 一致，并保留 singleton "
            "Index head 轴：预期 "
            f"{expected_k_idx_shape}，实际为 {tuple(k_idx.shape)}"
        )

    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise TypeError("block_size 必须是真正的整数，不能是 bool")
    if block_size <= 0:
        raise ValueError("block_size 必须为正数")

    # region P2-C01-A — 精确的缩放 token score
    # 构造 shape 为 [B,N,Hkv,N] 的 token_score。
    # 分别保留 query token、KV group 与 key token 轴；Kidx 的 singleton
    # head 轴表示所有 group 共享同一个 Index key。
    # 每个 didx 维点积都需要除以 sqrt(didx)。
    # C01-A 的输出会继续流入 C01-B，不再作为最终结果直接返回。

    # q_idx [B,N,Hkv,didx] / k_idx [B,N,1,didx]
    scalar = index_dim ** 0.5
    # q: [B,Nq,Hkv,didx] -> [B,Hkv,Nq,didx]
    # k: [B,Nk,1,didx]   -> [B,1,didx,Nk]
    # result: [B,Hkv,Nq,Nk]
    token_score = q_idx.permute(0, 2, 1, 3) @ k_idx.permute(0, 2, 3, 1) / scalar
    # [B,Hkv,Nq,Nk] -> [B,Nq,Hkv,Nk]
    token_score = token_score.permute(0, 2, 1, 3)

    # endregion

    # region P2-C01-B1 — causal token 可见性
    # 从 token_score 构造同 shape 的 causal_token_score。
    # 对 query token q，只允许 key token k <= q；未来 token 的 score 必须
    # 替换为当前 dtype/device 上的 -inf。

    # [N]
    position = torch.arange(sequence_length, device=token_score.device)

    # [N, N]，下三角矩阵
    causal_token_mask = position[None, :] <= position[:, None]

    # [B,Nq,Hkv,Nk] -> [B,Hkv,Nq,Nk] -> [B,Nq,Hkv,Nk]
    causal_token_score = token_score.permute(0, 2, 1, 3).masked_fill(
        ~causal_token_mask,
        float("-inf"),
    ).permute(0, 2, 1, 3)
    # endregion

    # region P2-C01-B2 — visible-token BlockMax
    # 计算 num_blocks = ceil(N / block_size)。
    num_blocks = (sequence_length + block_size - 1) // block_size
    # 右侧补齐后，将 key-token 轴拆成 [num_blocks, block_size]；再对每个
    # block 内的 visible token score 取最大值。

    padding_length = num_blocks * block_size - sequence_length

    # 右侧 padding 保证 key 轴长度可以拆成完整 block。
    # [B,Nq,Hkv,N] -> [B,Nq,Hkv,num_blocks * block_size]
    padded_token_score = F.pad(
        input=causal_token_score,
        pad=(0, padding_length),
        value=float("-inf"),
    )

    # [B,Nq,Hkv,num_blocks * block_size]
    # -> [B,Nq,Hkv,num_blocks,block_size]
    reshaped_padded_token_score = padded_token_score.reshape(
        batch_size,
        sequence_length,
        num_kv_groups,
        num_blocks,
        block_size,
    )

    # 在最后一维的 block 内 token 上取最大值。
    # [B,Nq,Hkv,num_blocks,block_size] -> [B,Nq,Hkv,num_blocks]
    block_score = reshaped_padded_token_score.max(
        dim=-1,
    ).values

    # endregion

    # region P2-C01-C — learning annotation
    # token_score 的四个轴是 [B,Nq,Hkv,Nk]，最后一轴才是 key-token 轴。
    # causal 必须发生在 BlockMax 之前，否则未来 token 的高分可能污染当前 block。
    # final partial block 右侧补 -inf，不会改变真实 token 的最大值；future-only
    # block 内所有位置都是 -inf，因此取 max 后仍保持 -inf。
    # endregion

    # 返回 causal-visible BlockMax：[B,N,Hkv,num_blocks]。
    return block_score
