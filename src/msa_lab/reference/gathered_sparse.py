from __future__ import annotations

import torch

from msa_lab.backends.base import AttentionOutput
from msa_lab.contracts.support import block_ids_to_group_mask


def gathered_sparse_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_ids: torch.Tensor,
    *,
    block_size: int,
) -> AttentionOutput:
    """Phase-1 causal self-attention 的 gathered sparse GQA Reference。

    输入布局：
        q: ``[B, N, Hq, D]``
        k: ``[B, N, Hkv, D]``
        v: ``[B, N, Hkv, D]``
        block_ids: ``[B, N, Hkv, Ksel]``

    输出布局：
        output: ``[B, N, Hq, D]``
        lse: ``[B, N, Hq]``，使用自然对数

    整体计算过程：
        1. 使用 C01 得到每个 query/KV-group row 的 effective token support，
           明确当前 row 实际能够看到哪些原始 K/V token。
        2. 根据 GQA 的共享关系，把属于同一个 KV head 的 G 个 query heads
           组织在一起，使它们共同使用当前 row 选中的 K/V。
        3. 逐个处理 ``[batch, query, KV head]`` row。每个 row 只收集 support
           中的 M 个 K/V token，不保留完整的 Nk key-token 计算轴。
        4. 让当前组的 G 个 query heads 分别与这 M 个 K token 计算 scaled
           dot-product score，并且只在这 M 个 selected token 上归一化。
        5. 同时保存每个 query head 的自然对数归一化常数 LSE，并使用概率对
           selected V 加权求和，得到当前 row 的输出。
        6. 将所有 row 的结果写回内部 grouped storage，最后把 KV-group 轴和
           组内 head 轴重新合并为模型侧 Hq 布局。

    Review one-row map:
        fixed ``[b, query, hkv]``

        support_row      ``[Nk]``
          -> where       ``selected_indices [M]``
          -> gather K/V  ``[M,D]``

        q_row            ``[G,D]``
        k_selected       ``[M,D]``
          -> score       ``[G,M]``
          -> softmax     ``[G,M]``
          -> logsumexp   ``[G]``

        prob ``[G,M] @ V [M,D]``
          -> output_row  ``[G,D]``

    Review focus:
        - ``M`` 是当前 row 的 effective support 长度，不是全局固定 Nk；
        - ``torch.where(condition)[0]`` 把 bool support row 转成 token indices；
        - tensor advanced indexing 在 Nk 轴选择 M 个位置，结果得到 ``[M,D]``；
        - ``new_empty`` 继承 dtype/device，但初始值未定义，必须保证每个位置都写回；
        - 当前 row 的 matmul 只收缩 D 或 M，不再构造完整 Nk score row；
        - 最终 ``flatten(2,3)`` 只负责把 Hkv、G 合并回模型侧 Hq。

    这个函数与 C02 masked-dense Oracle 表达相同的 attention 语义，但它只为
    effective support 中的 token 计算 score 和概率。Python 循环和小型中间
    张量在当前透明 Reference 中是允许的；本函数不能调用 C02 作为实现路径。
    """

    # region C03-A：输入约束、support 与 grouped Q

    if not (
        isinstance(q, torch.Tensor)
        and isinstance(k, torch.Tensor)
        and isinstance(v, torch.Tensor)
        and isinstance(block_ids, torch.Tensor)
    ):
        raise ValueError("Q/K/V/block_ids must be torch.Tensor objects")

    if not (
        q.ndim == 4
        and k.ndim == 4
        and v.ndim == 4
        and block_ids.ndim == 4
    ):
        raise ValueError("Q/K/V/block_ids must be rank-4 tensors")

    Bq, Nq, Hq, Dq = q.shape
    Bk, Nk, Hkv_k, Dk = k.shape
    Bv, Nv, Hkv_v, Dv = v.shape
    Bb, Nb, Hkv_meta, _ = block_ids.shape

    if (Bk, Nk, Hkv_k, Dk) != (Bv, Nv, Hkv_v, Dv):
        raise ValueError("K and V must have the same shape")

    if (Bq, Nq, Dq) != (Bk, Nk, Dk):
        raise ValueError("Q/K/V must agree in batch, sequence, and head dimension")

    if Dq <= 0:
        raise ValueError("Q/K/V head dimension must be positive")

    if (Bk, Nk, Hkv_k) != (Bb, Nb, Hkv_meta):
        raise ValueError("block_ids must align with Q/K/V in B, N, and Hkv")

    if Hkv_k <= 0 or Hq <= 0 or Hq % Hkv_k != 0:
        raise ValueError("Hq must be positive and divisible by positive Hkv")
    G = Hq // Hkv_k

    if not all(tensor.is_floating_point() for tensor in (q, k, v)):
        raise ValueError("Q/K/V must have floating-point dtypes")

    if not q.dtype == k.dtype == v.dtype:
        raise ValueError("Q/K/V must have the same dtype")

    if not q.device == k.device == v.device:
        raise ValueError("Q/K/V must be on the same device")

    if block_ids.device != q.device:
        raise ValueError("block_ids must be on the same device as Q/K/V")

    if block_size <= 0:
        raise ValueError("block_size must be positive")

    # Shape:
    # C01(block_ids [B,Nq,Hkv,Ksel]) -> support [B,Nq,Hkv,Nk]。
    #
    # Why:
    # C03 不重新解释 raw selection；直接消费 C01 已验证并 canonicalize 后的
    # effective token support，当前 row 的 M 就来自该 bool row 中 True 的数量。
    support = block_ids_to_group_mask(
        block_ids=block_ids,
        seq_len=Nq,
        block_size=block_size,
    )

    # Shape:
    # q [B,Nq,Hq,D] -> q_grouped [B,Nq,Hkv,G,D]，Hq = Hkv * G。
    #
    # PyTorch note:
    # reshape 把连续 Hq 拆成相邻的 Hkv、G 两轴；这里仍依赖原始 head 顺序
    # h = hkv * G + g。与 C02 相同，reshape 只重解释 shape，不会自动理解 GQA。
    q_grouped = q.reshape(Bq, Nq, Hkv_k, G, Dq)

    # Shape:
    # output_grouped [B,Nq,Hkv,G,D]。
    #
    # PyTorch note:
    # new_empty 从 q_grouped 继承 dtype/device，但分配出来的内存初始值未定义。
    #
    # Why safe here:
    # 三重循环会完整覆盖每个合法 [b,query,hkv] row 的 [G,D] 输出，因此无需先填 0。
    output_grouped = q_grouped.new_empty((Bq, Nq, Hkv_k, G, Dq))

    # Shape:
    # lse_grouped [B,Nq,Hkv,G]。
    #
    # PyTorch note:
    # q.new_empty(...) 同样继承 q 的 dtype/device，内容未初始化；循环必须覆盖全部 row。
    lse_grouped = q.new_empty((Bq, Nq, Hkv_k, G))

    # endregion

    # 逐个计算逻辑上长度可变的 [b, query, hkv] row。
    for b in range(Bq):
        for query in range(Nq):
            for hkv in range(Hkv_k):
                # region C03-B：规范有效索引与 K/V gather

                # Shape:
                # support [B,Nq,Hkv,Nk]
                #     [b,query,hkv,:] -> support_row [Nk]。
                #
                # PyTorch note:
                # integer index b/query/hkv 会删除对应轴；最后的 ``:`` 保留完整 Nk 轴。
                support_row = support[b, query, hkv, :]

                # Shape:
                # support_row [Nk] -> selected_indices [M]。
                #
                # PyTorch note:
                # 单参数 torch.where(condition) 返回一个 tuple，每个 tuple 元素对应
                # condition 的一个坐标轴。一维 condition 只有一个坐标 tensor，
                # 因此 [0] 取出所有 True token positions。M 是 True 的数量。
                #
                # Why:
                # 把 dense bool support 压缩成当前 row 真正需要 gather 的 token indices。
                selected_indices = torch.where(support_row)[0]

                # Shape:
                # k [B,Nk,Hkv,D]
                #     [b, selected_indices[M], hkv, :] -> k_selected [M,D]。
                #
                # PyTorch note:
                # b/hkv 是 integer indices，会删除 B/Hkv 轴；selected_indices 是
                # 一维 tensor advanced index，在 Nk 轴选择 M 个位置并产生 M 轴；
                # ``:`` 保留 D。因此结果是 [M,D]，而不是 [D] 或 [Nk,D]。
                # advanced indexing 得到 gathered result，不应把它理解成普通连续 slice view。
                #
                # Why:
                # 当前 row 只读取 effective support 中的 K token。
                k_selected = k[b, selected_indices, hkv, :]

                # Shape:
                # v [B,Nk,Hkv,D]
                #     [b, selected_indices[M], hkv, :] -> v_selected [M,D]。
                #
                # PyTorch note:
                # 与 K 完全相同的 advanced indexing 规则；使用同一 selected_indices
                # 保证 K/V 的第 m 行始终对应同一个原始 token。
                v_selected = v[b, selected_indices, hkv, :]

                # endregion

                # region C03-C：选中 token attention

                # Shape:
                # q_grouped [B,Nq,Hkv,G,D]
                #     [b,query,hkv,:,:] -> q_row [G,D]。
                #
                # PyTorch note:
                # 三个 integer indices 删除 B/Nq/Hkv 轴；两个 ``:`` 保留 G、D。
                #
                # Why:
                # 这 G 个 query heads 共享当前 hkv 对应的 selected K/V。
                q_row = q_grouped[b, query, hkv, :, :]

                # Math:
                # S = Q @ K^T。
                #
                # Shape:
                # q_row [G,D] @ k_selected.T [D,M] -> score_selected [G,M]，收缩 D。
                #
                # PyTorch note:
                # transpose(-2,-1) 交换最后两个轴，把 [M,D] 变成 [D,M]；
                # 对二维 tensor，``@`` 就是普通矩阵乘法。
                #
                # Why:
                # 每个 G query head 只对当前 selected M 个 keys 产生 score。
                score_selected = q_row @ k_selected.transpose(-2, -1)

                # Math:
                # S = S / sqrt(D)。
                #
                # Shape:
                # [G,M] / scalar -> [G,M]。
                #
                # Why:
                # scaled dot-product attention 的标准缩放，不改变 row 的 G/M 语义。
                score_selected = score_selected / (Dq ** 0.5)

                # Math:
                # P = softmax(S, selected-key axis)。
                #
                # Shape:
                # [G,M] -> [G,M]。
                #
                # PyTorch note:
                # softmax(dim=-1) 沿 M 归一化；对每个固定 g，M 个 selected-key
                # 概率之和为 1。G 个 query heads 彼此拥有独立 softmax rows。
                prob_selected = torch.softmax(score_selected, dim=-1)

                # Math:
                # LSE_g = log(sum_m exp(S[g,m]))。
                #
                # Shape:
                # [G,M] --reduce M--> [G]。
                #
                # PyTorch note:
                # logsumexp(dim=-1) 稳定计算自然对数归一化常数，并删除 M 轴。
                lse_row = torch.logsumexp(score_selected, dim=-1)

                # Math:
                # O = P @ V。
                #
                # Shape:
                # [G,M] @ [M,D] -> [G,D]，公共轴 M 被收缩。
                #
                # Why:
                # 对当前 selected V 做概率加权求和，输出仍保留每个 query head 的 D 维向量。
                output_row = prob_selected @ v_selected

                # endregion

                # region C03-D1：当前 row 写回

                # Shape:
                # output_grouped[b,query,hkv,:,:] [G,D] <- output_row [G,D]。
                #
                # PyTorch note:
                # 左侧 slice 选中预分配 tensor 的唯一 [G,D] row；赋值会把当前结果
                # 写入该位置。这里是唯一 ownership，所以覆盖而不是累加。
                output_grouped[b, query, hkv, :, :] = output_row

                # Shape:
                # lse_grouped[b,query,hkv,:] [G] <- lse_row [G]。
                #
                # Why:
                # 与 output 使用相同 [b,query,hkv] ownership 保存当前 G 个 heads 的 LSE。
                lse_grouped[b, query, hkv, :] = lse_row

                # endregion

    # region C03-D2：恢复模型侧布局

    # Shape:
    # output_grouped [B,Nq,Hkv,G,D]
    #     --flatten(2,3)--> output [B,Nq,Hq,D]。
    #
    # PyTorch note:
    # flatten(2,3) 合并相邻 Hkv、G 两轴，保持线性 head 顺序 h = hkv * G + g。
    # 它只合并轴，不重新排列 head。
    output = output_grouped.flatten(2, 3)

    # Shape:
    # lse_grouped [B,Nq,Hkv,G]
    #     --flatten(2,3)--> lse [B,Nq,Hq]。
    #
    # Why:
    # LSE 按与 output 完全相同的 query-head 顺序恢复 public layout。
    lse = lse_grouped.flatten(2, 3)

    # endregion

    # region C03-E：等价性与性质（由外部 acceptance 验证）

    return AttentionOutput(output=output, lse=lse)

    # endregion
