from __future__ import annotations

import math

import torch

from msa_lab.backends.base import AttentionOutput
from msa_lab.contracts.support import block_ids_to_group_mask


def masked_dense_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_ids: torch.Tensor,
    *,
    block_size: int,
) -> AttentionOutput:
    """Phase-1 causal self-attention 的 masked-dense GQA 语义 Oracle。

    输入布局：
        q: ``[B, N, Hq, D]``
        k: ``[B, N, Hkv, D]``
        v: ``[B, N, Hkv, D]``
        block_ids: ``[B, N, Hkv, Ksel]``

    输出布局：
        output: ``[B, N, Hq, D]``
        lse: ``[B, N, Hq]``，使用自然对数

    整体计算过程：
        1. 根据 GQA 的共享关系，把 Hq 个 query heads 组织成 Hkv 个组；每组
           包含 G 个 query heads，并共享同一个 K/V head。
        2. 保留完整的 query-token 轴和 key-token 轴，为每个 query head 计算
           对全部 key token 的 scaled dot-product score。这一步仍是 dense 计算。
        3. 使用 C01 生成的 effective support 屏蔽不可见位置，使 support 之外的
           token 不参与后续概率归一化。
        4. 每个 query head 只沿 key-token 轴归一化分数，同时计算对应的自然
           对数归一化常数 LSE。
        5. 使用归一化概率对共享的 V token 加权求和，得到每个 query head 的
           attention 输出。
        6. 将内部显式的 KV-group 轴和组内 head 轴重新合并为模型侧 Hq 布局，
           返回 output 和 LSE。

    Review shape map:
        Q ``[B,Nq,Hq,D]``
          -> reshape ``[B,Nq,Hkv,G,D]``
          -> permute ``[B,Hkv,G,Nq,D]``

        K ``[B,Nk,Hkv,D]``
          -> permute ``[B,Hkv,D,Nk]``
          -> unsqueeze ``[B,Hkv,1,D,Nk]``

        batched matmul
          ``[B,Hkv,G,Nq,D] @ [B,Hkv,1,D,Nk]``
          -> score ``[B,Hkv,G,Nq,Nk]``

        support ``[B,Nq,Hkv,Nk]``
          -> permute ``[B,Hkv,Nq,Nk]``
          -> unsqueeze ``[B,Hkv,1,Nq,Nk]``
          -> broadcast across G

    Review focus:
        - ``reshape`` 让 ``Hq = Hkv * G`` 的共享关系显式化；
        - ``permute`` 只重排轴语义，矩阵乘法要求真正参与收缩的轴放在末尾；
        - ``unsqueeze`` 产生 size=1 的共享轴，随后由 broadcasting 扩到 G；
        - rank>2 的 ``@`` 只对最后两个轴做矩阵乘法，前导轴按 broadcasting 对齐；
        - ``masked_fill`` 在 ``mask=True`` 的位置填值，因此 support 需要先取反；
        - ``softmax`` 和 ``logsumexp`` 都沿最后的 key-token 轴 Nk 处理。

    这个函数故意保留完整的 Nq × Nk 分数矩阵，因此容易检查语义，但不用于
    表示真正 sparse 的执行成本。它是 C03 gathered sparse Reference 的独立
    对照标准。
    """

    # region C02-A：输入约束与显式 GQA 视图

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
        raise ValueError("Q/K/V/block_ids rank error")
    
    Bq, Nq, Hq, Dq = q.shape
    Bk, Nk, Hkv_k, Dk = k.shape
    Bv, Nv, Hkv_v, Dv = v.shape
    Bb, Nb, Hkv_meta, Ksel = block_ids.shape

    if not [Bk, Nk, Hkv_k, Dk] == [Bv, Nv, Hkv_v, Dv]:
        raise ValueError("KV shape unequal")

    if not [Bq, Nq, Dq] == [Bk, Nk, Dk]:
        raise ValueError("Q/K shape unequal in B, N, D ")

    if Dq <= 0:
        raise ValueError("Q/K/V head dimension must be positive")

    if not [Bk, Nk, Hkv_k] == [Bb, Nb, Hkv_meta]:
        raise ValueError("block_ids shape unequal in B, N, H ")

    if (Hkv_k > 0 and Hq > 0 and Hq % Hkv_k == 0):
        G = Hq // Hkv_k
    else:
        raise ValueError("GQA dimension invalid")

    if not (
        q.is_floating_point()
        and k.is_floating_point()
        and v.is_floating_point()
        and q.dtype == k.dtype == v.dtype
    ):
        raise ValueError("Q/K/V datatype unenqual")

    if not q.device == k.device == v.device:
        raise ValueError("Q/K/V device unenqual")

    if block_ids.device != q.device:
        raise ValueError("block_ids must be on the same device as Q/K/V")

    if not block_size > 0:
        raise ValueError("block size !> 0")

    # Shape:
    # q [B, Nq, Hq, D] -> q_grouped [B, Nq, Hkv, G, D]，Hq = Hkv * G。
    #
    # PyTorch note:
    # reshape 重新解释 head 轴，把连续 Hq 拆成相邻的 Hkv、G 两轴；这里依赖
    # 原模型 head 顺序满足 h = hkv * G + g。reshape 不会替你推断这种语义，
    # 所以必须先由 contract 保证 Hq 能被 Hkv 整除。
    #
    # Why:
    # 显式表示“每个 KV head 由 G 个 query heads 共享”的 GQA 结构。
    q_grouped = q.reshape(Bq, Nq, Hkv_k, G, Dq)

    # endregion

    # region C02-B：完整分数张量

    # Shape:
    # q_grouped [B, Nq, Hkv, G, D]
    #     -> q_for_scores_grouped [B, Hkv, G, Nq, D]。
    #
    # PyTorch note:
    # permute(0, 2, 3, 1, 4) 中的数字表示“旧轴按什么顺序出现在新 tensor 中”：
    #     old axes: 0=B, 1=Nq, 2=Hkv, 3=G, 4=D
    #     new order: 0, 2, 3, 1, 4
    # permute 主要改变 axis order/stride 解释，不改变每个元素的数学值。
    #
    # Why:
    # 把最后两个轴整理为 Q matrix [Nq, D]，供后面的 batched matmul 使用。
    q_for_scores_grouped = q_grouped.permute(0, 2, 3, 1, 4)

    # Shape:
    # k [B, Nk, Hkv, D]
    #     --permute--> [B, Hkv, D, Nk]
    #     --unsqueeze(2)--> [B, Hkv, 1, D, Nk]。
    #
    # PyTorch note:
    # unsqueeze(2) 插入 size=1 的新轴，不复制 K。后续 batched matmul 时，
    # 这个 1 会 broadcast 成 G，从而让同一个 KV head 服务组内全部 G 个 query heads。
    #
    # Why:
    # 把最后两个轴组成 K^T matrix [D, Nk]，同时显式保留 GQA 共享行为。
    k_for_scores_grouped = k.permute(0, 2, 3, 1).unsqueeze(2)

    # Math:
    # score = Q @ K^T。
    #
    # Shape:
    # [B, Hkv, G, Nq, D]
    # @ [B, Hkv, 1, D, Nk]
    # -> [B, Hkv, G, Nq, Nk]，矩阵公共轴 D 被收缩。
    #
    # PyTorch note:
    # 对 rank>2 tensor，``@`` 把最后两个轴作为 matrix axes，前面的轴视为
    # batch axes 并按 broadcasting rules 对齐。这里第二个 tensor 的 G 位置是 1，
    # 因而广播到 G；真正的矩阵乘法是 [Nq,D] @ [D,Nk]。
    #
    # Why:
    # 每个 query head 得到对完整 Nk key-token 轴的一行 dense scores。
    score_grouped = q_for_scores_grouped @ k_for_scores_grouped

    # Math:
    # score = (Q @ K^T) / sqrt(D)。
    #
    # Shape:
    # [B, Hkv, G, Nq, Nk] / scalar -> shape 不变。
    #
    # Why:
    # 使用 scaled dot-product attention 的标准 1/sqrt(D) 缩放。
    score_grouped = score_grouped / math.sqrt(Dq)

    # endregion

    # region C02-C：Support mask 轴对齐

    # Shape:
    # C01(block_ids [B, Nq, Hkv, Ksel])
    #     -> support [B, Nq, Hkv, Nk]。
    #
    # Why:
    # C02 不重建 selection 语义，直接消费 C01 已验证的 effective causal support。
    support_for_scores_grouped = block_ids_to_group_mask(
        block_ids=block_ids,
        seq_len=Nq,
        block_size=block_size,
    )

    # Shape:
    # [B, Nq, Hkv, Nk] -> [B, Hkv, Nq, Nk]。
    #
    # PyTorch note:
    # permute(0, 2, 1, 3) 交换 Nq 与 Hkv 的位置，使 mask 的 batch/KV/query/key
    # 轴顺序与 score_grouped 对齐；这里只重排轴，不改变 support 的 True/False 值。
    support_for_scores_grouped = support_for_scores_grouped.permute(0, 2, 1, 3)

    # Shape:
    # [B, Hkv, Nq, Nk] -> [B, Hkv, 1, Nq, Nk]。
    #
    # PyTorch note:
    # unsqueeze(2) 插入 size=1 的组内 query-head 轴；后续与 score [B,Hkv,G,Nq,Nk]
    # 对齐时，这个 1 自动 broadcast 到 G。也就是说同一 KV group 的 G 个 query heads
    # 共享同一份 support mask。
    support_for_scores_grouped = support_for_scores_grouped.unsqueeze(2)

    # Math:
    # unsupported score := -inf。
    #
    # Shape:
    # score [B,Hkv,G,Nq,Nk]
    # mask  [B,Hkv,1,Nq,Nk] -> broadcast across G -> same score shape。
    #
    # PyTorch note:
    # masked_fill(mask=True, value=...) 修改的是 mask 为 True 的位置。
    # support=True 表示“有效”，所以这里先用 ``~`` 对 bool mask 逐元素取反，
    # 让无效位置变成 True 再填充。使用 -inf 后 exp(-inf)=0，因此这些位置
    # 在 softmax 中得到严格的 0 概率，而不是一个极小但非零的近似概率。
    score_masked_grouped = score_grouped.masked_fill(
        mask=~support_for_scores_grouped,
        value=-torch.inf,
    )

    # endregion

    # region C02-D：概率、LSE 与 V 聚合

    # Math:
    # P = softmax(S, key axis)。
    #
    # Shape:
    # [B,Hkv,G,Nq,Nk] -> [B,Hkv,G,Nq,Nk]。
    #
    # PyTorch note:
    # softmax(dim=-1) 只沿最后的 Nk 轴做指数归一化；其他 B/Hkv/G/Nq 轴都保留。
    # 对每个固定 [b,hkv,g,q]，这一整行 Nk 概率之和为 1。
    prob_grouped = torch.softmax(score_masked_grouped, dim=-1)

    # Math:
    # LSE = log(sum_k exp(score_k))，使用自然对数。
    #
    # Shape:
    # [B,Hkv,G,Nq,Nk] --reduce Nk--> [B,Hkv,G,Nq]。
    #
    # PyTorch note:
    # logsumexp(dim=-1) 在数值上稳定地计算 log(sum(exp(x)))，并归约指定轴；
    # 与 softmax 一样，这里被处理/消失的是 key-token 轴 Nk。
    lse_grouped = torch.logsumexp(score_masked_grouped, dim=-1)

    # Shape:
    # v [B,Nk,Hkv,D]
    #     --permute--> [B,Hkv,Nk,D]
    #     --unsqueeze(2)--> [B,Hkv,1,Nk,D]。
    #
    # PyTorch note:
    # size=1 的 G 轴允许同一组 V 通过 broadcasting 被 G 个 query heads 共享。
    #
    # Why:
    # 把最后两个轴整理为 V matrix [Nk,D]，与 probability 的 [Nq,Nk] 相乘。
    v_for_output_grouped = v.permute(0, 2, 1, 3).unsqueeze(2)

    # Math:
    # O = P @ V。
    #
    # Shape:
    # [B,Hkv,G,Nq,Nk] @ [B,Hkv,1,Nk,D]
    # -> [B,Hkv,G,Nq,D]，公共轴 Nk 被收缩。
    #
    # PyTorch note:
    # 仍然是 batched matmul：最后两轴 [Nq,Nk] @ [Nk,D]；
    # V 的 G 位置为 1，因此广播到组内全部 G 个 query heads。
    #
    # Why:
    # 对 key-token 轴做概率加权求和，保留下来的 D 是输出 feature 维。
    output_grouped = prob_grouped @ v_for_output_grouped

    # endregion

    # region C02-E：恢复模型侧布局

    # Shape:
    # output_grouped [B,Hkv,G,Nq,D]
    #     --permute--> [B,Nq,Hkv,G,D]
    #     --flatten(2,3)--> [B,Nq,Hq,D]。
    #
    # PyTorch note:
    # 先把 Hkv、G 放回相邻位置，再 flatten 合并为 Hq；flatten 不会替你判断
    # head 语义，正确性来自前面一直保持 h = hkv * G + g 的线性顺序。
    output = output_grouped.permute(0, 3, 1, 2, 4).flatten(2, 3)

    # Shape:
    # lse_grouped [B,Hkv,G,Nq]
    #     --permute--> [B,Nq,Hkv,G]
    #     --flatten(2,3)--> [B,Nq,Hq]。
    #
    # Why:
    # LSE 与 output 使用相同的 query-head 顺序恢复模型侧布局。
    lse = lse_grouped.permute(0, 3, 1, 2).flatten(2, 3)

    # endregion

    return AttentionOutput(output=output, lse=lse)
