from __future__ import annotations

import torch


def validate_fixed_block_ids(
    block_ids: torch.Tensor,
    *,
    seq_len: int,
    block_size: int,
) -> None:
    """Validate Phase-1 fixed-selection metadata.

    Args:
        block_ids: Integer tensor shaped ``[B, N, Hkv, Ksel]``. Valid block
            IDs may appear in any order. ``-1`` is reserved for tail padding.
        seq_len: Causal self-attention sequence length. Phase 1 requires the
            query dimension of ``block_ids`` to equal this value.
        block_size: Number of key tokens in each zero-based key block.

    This function must reject malformed metadata instead of silently sorting,
    deduplicating, clipping, or inserting a local block.

    Review focus:
        - raw metadata validation happens before any token expansion;
        - one metadata row is ``[Ksel]`` after flattening ``B/N/Hkv``;
        - boolean indexing extracts only valid non-padding block IDs;
        - duplicate detection compares the element count before/after ``unique``.
    """

    # region C01-A：原始元数据约束

    if block_ids.ndim != 4:
        raise ValueError(
            f"block_ids must be rank-4 [B, N, Hkv, Ksel], "
            f"but got shape {tuple(block_ids.shape)}"
        )

    integer_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if block_ids.dtype not in integer_dtypes:
        raise ValueError(
            f"block_ids must have an integer dtype, "
            f"but got {block_ids.dtype}"
    )

    if seq_len <= 0:
        raise ValueError(
            f"seq_len must be positive, but got {seq_len}"
        )

    if block_size <= 0:
        raise ValueError(
            f"block_size must be positive, but got {block_size}"
        )

    if block_ids.shape[1] != seq_len:
        raise ValueError(
            f"block_ids query dimension must equal seq_len, "
            f"but got query dimension {block_ids.shape[1]} "
            f"and seq_len {seq_len}"
        )

    n_blocks = (seq_len + block_size - 1) // block_size

    if torch.any(block_ids < -1):
        raise ValueError("negative block id or invalid padding")

    if torch.any(block_ids >= n_blocks):
        raise ValueError("block id out of range")

    # Shape:
    # [B, N, Hkv, Ksel] -> [B*N*Hkv, Ksel]。
    #
    # PyTorch note:
    # reshape 只要求结果拥有相同元素总数；这里把前三个“row identity”轴合并，
    # 最后一轴 Ksel 保持不变。reshape 会在可能时返回 view，但不要把它理解成
    # “永远零拷贝”；这里真正依赖的是元素逻辑顺序不变。
    metadata_rows = block_ids.reshape(-1, block_ids.shape[-1])
    for metadata_row in metadata_rows:
        seen_padding = False
        for block_id in metadata_row:
            if block_id == -1:
                seen_padding = True
            elif seen_padding:
                raise ValueError("padding must be at tail")

        # Shape:
        # metadata_row [Ksel] --boolean indexing--> block_ids_valid [Kvalid]。
        #
        # PyTorch note:
        # metadata_row >= 0 先生成同 shape 的 bool mask；tensor[bool_mask]
        # 会按 True 位置抽取元素，返回一维 packed 结果，因此 padding -1 被移除。
        block_ids_valid = metadata_row[metadata_row >= 0]
        if block_ids_valid.numel() < 1:
            raise ValueError("empty selection")

        # PyTorch note:
        # unique() 返回去重后的元素；若 unique 后 numel 变小，则原 row 存在重复 ID。
        if block_ids_valid.numel() != block_ids_valid.unique().numel():
            raise ValueError("duplicate block ids") 

    # endregion


def block_ids_to_group_mask(
    block_ids: torch.Tensor,
    *,
    seq_len: int,
    block_size: int,
) -> torch.Tensor:
    """计算每个 query/KV-group row 的精确 causal token support。

    输入：
        block_ids: ``[B, N, Hkv, Ksel]``，每个 row 选中的 block ID。
        seq_len: 序列长度 ``N``。
        block_size: 每个 block 包含的 token 数。

    输出：
        mask: ``[B, N, Hkv, N]`` 的 bool 张量。``mask[b,q,hkv,n]``
        为 True 表示 key token ``n`` 属于已选 block，并且对 query ``q``
        满足 causal 可见性。

    整体计算过程：
        1. 把每个选中的 block 展开成其中包含的全部 token 位置，使 block-level
           selection 变成 token-level 候选集合。
        2. 根据序列边界、padding 和 causal 关系过滤候选集合，只保留当前
           query 实际能够看到的 token。
        3. 把仍然可见的候选位置映射回原始长度为 N 的 key-token 轴。无效候选
           不产生贡献；多个候选指向同一 token 时，它们落到同一个位置。
        4. 将原始 key-token 轴上至少收到一次有效映射的位置标记为 True，得到
           每个 query/KV-group row 的 effective support。
        5. 检查每个 row 是否至少保留一个可见 token，避免后续 attention 在
           空 support 上归一化。

    Review focus:
        - ``None`` / broadcasting 如何把 block 起点展开成 block 内 token；
        - ``flatten`` 如何形成候选 token 轴 ``T = Ksel * block_size``；
        - ``clamp`` 只保证 scatter index 合法，不会把非法候选修复成有效 support；
        - ``scatter_add_`` 把候选槽位重新累计到原始 key-token 轴；
        - ``any(dim=-1)`` 如何从 token support 判断 row 是否为空。

    这个函数只表达数学 support，不规定后续 gathered Reference 的存储顺序。
    """

    # region C01-B：Block ID 展开为候选 token

    validate_fixed_block_ids(
        block_ids,
        seq_len=seq_len,
        block_size=block_size,
    )
    B, N, Hkv, _ = block_ids.shape

    # Math:
    # block_start = block_id * block_size。
    #
    # Shape:
    # [B, N, Hkv, Ksel] * scalar -> [B, N, Hkv, Ksel]。
    #
    # Why:
    # 先把 block ID 转成该 block 的起始 token ID，下一步再加 block 内偏移。
    token_ids = block_ids * block_size 

    # Shape:
    # token_ids[..., None]
    #     [B, N, Hkv, Ksel] -> [B, N, Hkv, Ksel, 1]
    # offset
    #     [block_size]
    # broadcast result
    #     [B, N, Hkv, Ksel, block_size]。
    #
    # PyTorch note:
    # ``None`` 等价于在该位置插入一个 size=1 的轴；这里是 unsqueeze(-1)。
    # 加法时 PyTorch 从末尾向前匹配 shape，最后的 1 与 block_size 可广播，
    # 因而每个 block 起点都会加上 [0, ..., block_size-1] 的全部偏移。
    # torch.arange(..., device=block_ids.device) 同时保证临时 tensor 在输入设备上。
    #
    # Why:
    # 把 block-level selection 展开为 block 内所有候选 token positions。
    token_ids = token_ids[..., None] + torch.arange(block_size, device=block_ids.device)

    # Shape:
    # [B, N, Hkv, Ksel, block_size]
    #     --flatten(-2)--> [B, N, Hkv, T]，T = Ksel * block_size。
    #
    # PyTorch note:
    # flatten(-2) 从倒数第 2 轴开始合并到最后一轴，保持元素逻辑顺序；
    # 这里把“选择槽位 Ksel”和“block 内位置”合成一个候选 token 轴 T。
    token_ids = token_ids.flatten(-2)

    # endregion

    # region C01-C：因果 support 与空 row 检查

    # Shape:
    # query_idx [N] = [0, 1, ..., N-1]。
    #
    # PyTorch note:
    # arange 生成连续整数位置；显式指定 device，避免 CPU/GPU 临时 tensor 混用。
    query_idx = torch.arange(seq_len, device=block_ids.device)

    # Shape:
    # [N] -> [1, N, 1, 1]。
    #
    # PyTorch note:
    # 每个 ``None`` 都插入一个 size=1 轴，不复制元素。得到的 singleton axes
    # 会在后面的比较中广播到 B、Hkv、T，使每个 query q 都与本 row 的全部候选
    # token 比较 causal 条件 token_id <= q。
    query_idx = query_idx[None, :, None, None]

    # Math:
    # visible = in_range & causal。
    #
    # Shape:
    # token_ids [B, N, Hkv, T]
    # query_idx [1, N, 1, 1]
    #     -> broadcast -> visible [B, N, Hkv, T]。
    #
    # PyTorch note:
    # 比较运算逐元素产生 bool tensor；``&`` 对 bool tensor 做逐元素逻辑与。
    # query_idx 的 size=1 轴自动广播到 B、Hkv、T。
    #
    # Why:
    # 过滤 padding/越界 token，同时执行 causal 条件，不允许 query 看到未来 token。
    visible = (token_ids >= 0) & (token_ids < seq_len) & (token_ids <= query_idx)

    # Shape:
    # [B, N, Hkv, T] -> [B, N, Hkv, T]，只改变 index 数值范围。
    #
    # PyTorch note:
    # scatter_add_ 要求 index 中每个值都落在目标轴合法范围，即使对应 src 为 0。
    # clamp 在这里仅把 padding/越界候选临时夹到 [0, N-1]，以满足 API index contract。
    # 它并没有“修复”这些非法候选：这些位置的 visible=False，后面转换成 src=0，
    # 因此不会对数学 support 产生贡献。
    safe_token_ids = token_ids.clamp(0, seq_len - 1)

    # Shape:
    # counts [B, N, Hkv, N]，最后一轴重新代表原始 key-token position。
    #
    # PyTorch note:
    # torch.zeros 显式指定 shape/dtype/device；这里用 int32 是因为后面需要做计数累加。
    #
    # Why:
    # 为“候选槽位 T -> 原始 token 轴 N”的 scatter accumulation 准备目标存储。
    counts = torch.zeros(
        B, N, Hkv, seq_len,
        dtype=torch.int32,
        device=block_ids.device,
    )

    # Shape:
    # target counts        [B, N, Hkv, N]
    # index safe_token_ids [B, N, Hkv, T]
    # src visible          [B, N, Hkv, T]
    #     --scatter_add over dim=-1--> counts [B, N, Hkv, N]。
    #
    # PyTorch note:
    # scatter_add_(dim=-1, index, src) 可以理解为对 src 的每个候选槽位 t：
    #     counts[..., index[..., t]] += src[..., t]
    # ``_`` 表示原地修改 counts；如果多个候选槽位映射到同一个 token index，
    # 它们会真正累加。visible.to(torch.int32) 把 False/True 转为 0/1 计数。
    #
    # Why:
    # 把展开后的 candidate slots 映射回原始 key-token 轴，形成 token-level union。
    counts.scatter_add_(
        dim=-1,
        index=safe_token_ids,
        src=visible.to(torch.int32),
    )

    # Math:
    # support = counts > 0。
    #
    # Shape:
    # [B, N, Hkv, N] -> bool [B, N, Hkv, N]。
    #
    # Why:
    # 这里只关心某个原始 token 是否至少收到一次有效映射，而不关心映射次数。
    support = counts > 0

    # Shape:
    # support [B, N, Hkv, N] --any over key axis--> has_support [B, N, Hkv]。
    #
    # PyTorch note:
    # any(dim=-1) 会归约最后一轴；该轴只要有一个 True，结果就是 True。
    #
    # Why:
    # 检查每个 batch/query/KV-group row 是否至少保留一个 effective token。
    has_support = support.any(dim=-1)

    # Shape:
    # [B, N, Hkv] -> [B, N, Hkv]。
    #
    # PyTorch note:
    # ``~`` 对 bool tensor 做逐元素逻辑非，因此 True/False 互换。
    empty_rows = ~has_support

    if empty_rows.any():
        raise ValueError(
            "every query/KV-group row must have non-empty causal support"
        )

    # endregion

    return support
