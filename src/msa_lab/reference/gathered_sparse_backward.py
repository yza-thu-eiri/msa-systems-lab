"""P1-C04 学习检查点：手写 gathered sparse GQA 的反向传播。

本文件在 Phase 1 中的位置
-------------------------
前面的三个章节已经完成了本文件需要复用的三层能力：

* P1-C01，``contracts/support.py``：校验固定的 selected-block metadata，
  并为每个 ``[batch, query, KV head]`` row 生成精确的 causal token support。
* P1-C02，``reference/masked_dense.py``：使用 masked-dense GQA 计算相同语义，
  作为判断 sparse 实现是否正确的语义 Oracle，而不是本函数的实现路径。
* P1-C03，``reference/gathered_sparse.py``：只 gather 被选中的 K/V token，
  计算 scaled score、softmax probability、output 和自然对数 LSE。

P1-C04 固定 ``block_ids``，手动求 C03 前向过程的梯度。对于一个
``[b, query, hkv]`` row，共有 ``G = Hq / Hkv`` 个 query heads 共享同一组
selected K/V。本反向过程需要重新计算这些 selected forward state，然后依次
完成 output、softmax/LSE、scaled QK score 的梯度，累加 G 个 heads 对 K/V 的
共同贡献，最后 scatter-add 回原始 token 位置。selection metadata 本身没有梯度。

添加本说明时的进度
------------------
当前已经完成 C04-A 至 C04-E：校验 public contract，重建 selected forward
state，手写 output、softmax、可选 LSE 和 scaled score 的逐-row 反向，
scatter-add selected K/V 梯度，恢复模型原始 Q 布局并返回完整梯度。

当前实现是用于验证数学语义的透明 Python-loop Reference，不是向量化或
生产级 kernel。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from msa_lab.contracts.support import block_ids_to_group_mask

@dataclass(frozen=True)
class AttentionGradients:
    """固定 selection 的 gathered sparse GQA Reference 梯度。"""

    dq: torch.Tensor
    dk: torch.Tensor
    dv: torch.Tensor


def gathered_sparse_gqa_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_ids: torch.Tensor,
    grad_output: torch.Tensor,
    grad_lse: torch.Tensor | None = None,
    *,
    block_size: int,
) -> AttentionGradients:
    """手动求 Phase-1 gathered sparse GQA Reference 的梯度。

    输入布局：
        q: ``[B, N, Hq, D]``
        k: ``[B, N, Hkv, D]``
        v: ``[B, N, Hkv, D]``
        block_ids: ``[B, N, Hkv, Ksel]``
        grad_output: ``[B, N, Hq, D]``
        grad_lse: 可选的 ``[B, N, Hq]``

    输出布局：
        dq: shape、dtype、device 与 ``q`` 相同
        dk: shape、dtype、device 与 ``k`` 相同
        dv: shape、dtype、device 与 ``v`` 相同

    C04 局部变量命名约定：
        - 无后缀：public contract 中模型原始布局的 tensor，例如
          ``q [B,N,Hq,D]``；
        - ``*_grouped``：把全局 Hq 轴拆成 Hkv、G 两轴，例如
          ``q_grouped [B,N,Hkv,G,D]``；
        - ``*_row``：固定 ``[b, query, hkv]`` 后的当前逻辑 row，例如
          ``q_row [G,D]``；
        - ``*_selected``：当前 row 中只对应 M 个 selected token 的 tensor，
          例如 ``k_selected [M,D]``；
        - ``grad_*``：对应前向变量的梯度，后缀继续说明其布局，例如
          ``grad_q_row [G,D]``、``grad_k_selected [M,D]``。

        最终返回的 ``dq/dk/dv`` 沿用 public contract 名称，不增加布局后缀。

    例如，``grad_output_grouped`` 表示全局 ``[B,N,Hkv,G,D]`` view，
    ``grad_output_row`` 表示其中一个 ``[G,D]`` row；``v_selected`` 和
    ``grad_v_selected`` 分别表示当前 row 的 selected V 及其梯度。

    本章约束：
        - selection metadata 固定且不可微；
        - 使用 C01 定义的精确 effective support；
        - 保持 GQA 分组关系 ``Hq = Hkv * G``；
        - 只重算反向传播需要的 selected-token forward state；
        - 在同一个 KV group 内累加多个 query heads 的共享 K/V 梯度；
        - 把 selected-token 梯度 scatter-add 回原始 token 位置；
        - 本函数不能调用 ``torch.autograd.grad`` 或 ``Tensor.backward``；
        - 本阶段不实现自定义 ``torch.autograd.Function`` 或 GPU kernel。

    学习检查点：
        C04-A：校验 public contract，并重建 selected forward state。
        C04-B：反向传播 output aggregation。
        C04-C：反向传播 softmax 和可选的 LSE 路径。
        C04-D：反向传播 scaled Q/K score。
        C04-E：scatter-add K/V 梯度，并恢复模型侧布局。

    当前 learner implementation 已完成 C04-A 至 C04-E，包括可选 grad_lse
    路径、Q 梯度布局恢复、selected K/V scatter-add 和完整梯度返回。
    """

    # ============================================================================
    # C04 one-row backward map（固定 [b, query, hkv]）
    #
    # Forward replay:
    #   Q [G,D], K/V [M,D]
    #   S = Q @ K.T / sqrt(D)            -> [G,M]
    #   P = softmax(S, dim=M)            -> [G,M]
    #   O = P @ V                        -> [G,D]
    #
    # Backward:
    #   dP = dO @ V.T                    -> [G,M]
    #   dV = P.T @ dO                    -> [M,D]   (收缩 G)
    #   c  = sum_M(dP * P)               -> [G,1]
    #   dS = P * (dP - c)                -> [G,M]
    #   optional LSE: dS += dLSE[:,None] * P
    #   dQ = dS @ K / sqrt(D)            -> [G,D]   (收缩 M)
    #   dK = dS.T @ Q / sqrt(D)          -> [M,D]   (收缩 G)
    #
    # Writeback:
    #   dQ: 当前 query row 唯一拥有，直接写回。
    #   dK/dV: 不同 query rows 可能命中同一原始 token，因此必须累加。
    # ============================================================================

    # region C04-A1：校验 public contract。
    required_tensors = (q, k, v, block_ids, grad_output)
    if not all(isinstance(tensor, torch.Tensor) for tensor in required_tensors):
        raise ValueError(
            "Q/K/V/block_ids/grad_output must be torch.Tensor objects"
        )

    if grad_lse is not None and not isinstance(grad_lse, torch.Tensor):
        raise ValueError("grad_lse must be a torch.Tensor or None")

    if not (
        q.ndim == 4
        and k.ndim == 4
        and v.ndim == 4
        and block_ids.ndim == 4
        and grad_output.ndim == 4
    ):
        raise ValueError(
            "Q/K/V/block_ids/grad_output must be rank-4 tensors"
        )

    if grad_lse is not None and grad_lse.ndim != 3:
        raise ValueError("grad_lse must be rank-3 [B, N, Hq]")

    Bq, Nq, Hq, Dq = q.shape
    Bk, Nk, Hkv_k, Dk = k.shape
    Bv, Nv, Hkv_v, Dv = v.shape
    Bb, Nb, Hkv_meta, Ksel = block_ids.shape

    if (Bk, Nk, Hkv_k, Dk) != (Bv, Nv, Hkv_v, Dv):
        raise ValueError("K and V must have the same shape")

    if (Bq, Nq, Dq) != (Bk, Nk, Dk):
        raise ValueError(
            "Q/K/V must agree in batch, sequence, and head dimension"
        )

    if Dq <= 0:
        raise ValueError("Q/K/V head dimension must be positive")

    if (Bk, Nk, Hkv_k) != (Bb, Nb, Hkv_meta):
        raise ValueError("block_ids must align with Q/K/V in B, N, and Hkv")

    if Hkv_k <= 0 or Hq <= 0 or Hq % Hkv_k != 0:
        raise ValueError("Hq must be positive and divisible by positive Hkv")

    if grad_output.shape != q.shape:
        raise ValueError("grad_output must have the same shape as Q")

    if grad_lse is not None and grad_lse.shape != (Bq, Nq, Hq):
        raise ValueError("grad_lse must have shape [B, N, Hq]")

    floating_tensors = (q, k, v, grad_output)
    if not all(tensor.is_floating_point() for tensor in floating_tensors):
        raise ValueError("Q/K/V/grad_output must have floating-point dtypes")

    if grad_lse is not None and not grad_lse.is_floating_point():
        raise ValueError("grad_lse must have a floating-point dtype")

    if not q.dtype == k.dtype == v.dtype == grad_output.dtype:
        raise ValueError("Q/K/V/grad_output must have the same dtype")

    if grad_lse is not None and grad_lse.dtype != q.dtype:
        raise ValueError("grad_lse must have the same dtype as Q/K/V")

    if not q.device == k.device == v.device == grad_output.device:
        raise ValueError("Q/K/V/grad_output must be on the same device")

    if block_ids.device != q.device:
        raise ValueError(
            "block_ids must be on the same device as Q/K/V/grad_output"
        )

    if grad_lse is not None and grad_lse.device != q.device:
        raise ValueError("grad_lse must be on the same device as Q/K/V")

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    # endregion

    # region C04-A2：重建 selected forward state。

    # support [B, Nq, Hkv, Nk]
    #     = C01(block_ids [B, Nq, Hkv, Ksel])。
    # 函数作用：block_ids_to_group_mask 校验并展开 selected block，返回 bool support。
    # 本步目的：得到每个 query/KV-group 对全部 key token 的可见性。
    support = block_ids_to_group_mask(
        block_ids=block_ids,
        seq_len=Nq,
        block_size=block_size,
    )

    # G（标量）= Hq // Hkv。
    # 运算作用：整数除法得到每个 KV head 共享的 query-head 数；A1 已保证整除。
    # 本步目的：为模型侧 Hq 轴拆分出显式的 Hkv、G 两个轴。
    G = Hq // Hkv_k

    # q_grouped [B, Nq, Hkv, G, D]
    #     = reshape(q [B, Nq, Hq, D])，其中 Hq = Hkv * G。
    # 函数作用：reshape 保持元素总数和线性顺序不变，只重新解释张量的轴。
    # 本步目的：把模型侧 query-head 轴拆成 KV-head 轴和组内 query-head 轴。
    q_grouped = q.reshape(Bq, Nq, Hkv_k, G, Dq)

    # grad_output_grouped [B, Nq, Hkv, G, D]
    #     = reshape(grad_output [B, Nq, Hq, D])，其中 Hq = Hkv * G。
    # 函数作用：reshape 保持元素总数和 head 线性顺序不变，只重新解释轴。
    # 本步目的：建立覆盖所有 row 的全局 grouped upstream-gradient view。
    grad_output_grouped = grad_output.reshape(Bq, Nq, Hkv_k, G, Dq)

    # endregion

    # region C04-A2：初始化全局梯度存储

    # grad_q_grouped [B, Nq, Hkv, G, D]
    #     = zeros_like(q_grouped [B, Nq, Hkv, G, D])。
    # 函数作用：zeros_like 创建 shape、dtype、device 均相同的全零张量。
    # 本步目的：保存所有 row 的 Q 梯度；每个 row 在循环中写入一次。
    grad_q_grouped = torch.zeros_like(q_grouped)

    # grad_k [B, Nk, Hkv, D]
    #     = zeros_like(k [B, Nk, Hkv, D])。
    # 函数作用：zeros_like 创建与原始 K 布局完全相同的全零梯度存储。
    # 本步目的：累加不同 query row 对同一个原始 K token 的梯度贡献。
    grad_k = torch.zeros_like(k)

    # grad_v [B, Nk, Hkv, D]
    #     = zeros_like(v [B, Nk, Hkv, D])。
    # 函数作用：zeros_like 创建与原始 V 布局完全相同的全零梯度存储。
    # 本步目的：累加不同 query row 对同一个原始 V token 的梯度贡献。
    grad_v = torch.zeros_like(v)

    # endregion

    # region C04-C：准备可选 LSE 上游梯度

    # grad_lse_grouped [B, Nq, Hkv, G]
    #     = reshape(grad_lse [B, Nq, Hq])，其中 Hq = Hkv * G。
    # 函数作用：reshape 只拆分 head 轴，不改变元素值和 head 线性顺序。
    # 本步目的：让循环能够按 [b, query, hkv] 取得当前 row 的 G 个 LSE 梯度。
    if grad_lse is not None:
        grad_lse_grouped = grad_lse.reshape(Bq, Nq, Hkv_k, G)

    # endregion

    # 逐个计算逻辑上长度可变的 [b, query, hkv] row。
    for b in range(Bq):
        for query in range(Nq):
            for hkv in range(Hkv_k):
                # region C04-A2：当前 row 前向状态重算

                # support_row [Nk]
                #     = support [B, Nq, Hkv, Nk] 在 b、query、hkv 轴取一个位置。
                # 索引作用：三个整数索引各移除一个轴，末尾的 : 保留完整 Nk 轴。
                # 本步目的：得到当前 row 对全部 key token 的布尔 support。
                support_row = support[b, query, hkv, :]

                # token_indices_selected [M] = torch.where(support_row [Nk])[0]。
                # 函数作用：单参数 torch.where 返回每个维度上 True 元素的坐标张量；
                # 一维输入只返回单元素 tuple ([M],)，[0] 取出其中的索引张量。
                # 本步目的：把 dense bool support 压缩为实际需要 gather 的 token 下标；
                # M 是当前 row 中 True 的数量，不是完整 key 长度 Nk。
                token_indices_selected = torch.where(support_row)[0]

                # k_selected [M, D]
                #     = k [B, Nk, Hkv, D] 在 token 轴使用
                #       token_indices_selected [M] gather。
                # 索引作用：整数 b、hkv 移除对应轴；张量索引 token_indices_selected
                # 在 Nk 轴执行高级索引，并保留其长度 M；末尾 : 保留 D。
                # 本步目的：只读取当前 row 实际可见的 M 个 K token。
                k_selected = k[b, token_indices_selected, hkv, :]

                # v_selected [M, D]
                #     = v [B, Nk, Hkv, D] 在 token 轴使用
                #       token_indices_selected [M] gather。
                # 索引作用：与上一行完全相同，但被索引的源张量是 V。
                # 本步目的：用相同 token 顺序读取与 K 对齐的 M 个 V token。
                v_selected = v[b, token_indices_selected, hkv, :]

                # q_row [G, D]
                #     = q_grouped [B, Nq, Hkv, G, D] 在 b、query、hkv 轴取一个位置。
                # 索引作用：三个整数索引移除 B、Nq、Hkv 轴，两个 : 保留 G、D 轴。
                # 本步目的：取出共享当前 KV head 的 G 个 query heads。
                q_row = q_grouped[b, query, hkv, :, :]

                # score_selected [G, M]
                #     = q_row [G, D] @ k_selected.T [D, M]。
                # 函数作用：transpose(-2, -1) 交换 K 的最后两个轴；@ 对 [G, D]
                # 与 [D, M] 做矩阵乘法，并把公共轴 D 求和收缩。
                # 本步目的：为 G 个 query heads 分别计算 M 个 selected-key 分数。
                score_selected = q_row @ k_selected.transpose(-2, -1)

                # score_selected [G, M]
                #     = score_selected [G, M] / sqrt(D)；缩放不改变维度。
                # 运算作用：** 0.5 计算标量平方根，除法把该标量广播到全部分数。
                # 本步目的：使用 scaled dot-product attention 的 1/sqrt(D) 缩放。
                score_selected = score_selected / (Dq ** 0.5)

                # prob_selected [G, M]
                #     = softmax(score_selected [G, M], dim=M)。
                # 函数作用：softmax(dim=-1) 在 M 轴做指数归一化，使每行概率和为 1；
                # 输入和输出维度保持相同。
                # 本步目的：每个 query head 只在当前选中的 M 个 key token 上归一化。
                prob_selected = torch.softmax(score_selected, dim=-1)

                # endregion

                # region C04-B：Output 聚合反向传播

                # grad_output_row [G, D]
                #     = grad_output_grouped [B, Nq, Hkv, G, D]
                #       在 b、query、hkv 轴取一个位置。
                # 索引作用：三个整数索引移除 B、Nq、Hkv 轴，两个 : 保留 G、D 轴。
                # 本步目的：取得当前 row 的 output 上游梯度。
                grad_output_row = grad_output_grouped[b, query, hkv, :, :]

                # grad_prob_selected [G, M]
                #     = grad_output_row [G, D] @ v_selected.T [D, M]。
                # 函数作用：矩阵乘法收缩 D，得到每个 probability 元素的梯度。
                # 本步目的：反向传播 output = probability @ selected V 的 probability 路径。
                grad_prob_selected = (
                    grad_output_row @ v_selected.transpose(-2, -1)
                )

                # grad_v_selected [M, D]
                #     = prob_selected.T [M, G] @ grad_output_row [G, D]。
                # 函数作用：矩阵乘法收缩共享同一个 KV head 的 G 个 query heads。
                # 本步目的：得到当前 row 中 M 个 selected V token 的合计梯度。
                grad_v_selected = (
                    prob_selected.transpose(-2, -1) @ grad_output_row
                )

                # endregion

                # region C04-C：Softmax 反向传播

                # Math:
                # C_g = sum_m(dP[g,m] * P[g,m])。
                #
                # Shape:
                # [G,M] * [G,M] -> sum over M -> [G,1]。
                #
                # PyTorch note:
                # sum(dim=-1) 会归约 M 轴；keepdim=True 不删除该轴，而是保留为 size=1，
                # 所以结果是 [G,1]，下一步可以自然 broadcast 回 [G,M]。
                #
                # Why:
                # 每个 g 都是一条独立 softmax row，因此拥有自己的公共修正量 C_g。
                softmax_dot_row = (
                    grad_prob_selected * prob_selected
                ).sum(dim=-1, keepdim=True)

                # grad_score_selected [G, M]
                #     = prob_selected [G, M]
                #       * (grad_prob_selected [G, M] - softmax_dot_row [G, 1])。
                # 函数作用：减法把 [G, 1] 广播为 [G, M]，随后逐元素相乘。
                # 本步目的：得到当前 row 中 M 个 selected score 的梯度。
                grad_score_selected = prob_selected * (
                    grad_prob_selected - softmax_dot_row
                )

                # Review note：LSE 有两种不同角色。
                # 1. saved forward state：例如 FlashAttention 用 LSE 在 backward 重建 P；
                #    仅仅保存 LSE 本身不会额外产生一条到 loss 的梯度路径。
                # 2. differentiable output：如果调用方真的传入 grad_lse = dL/dLSE，
                #    才存在 S -> LSE -> loss 的额外路径，此时 dS_lse = dLSE[:,None] * P。
                # 当前 Reference 已直接 replay prob_selected，所以不需要重算 LSE 值本身。
                if grad_lse is not None:
                    # grad_lse_row [G]
                    #     = grad_lse_grouped [B, Nq, Hkv, G]
                    #       在 b、query、hkv 轴取一个位置。
                    # 本步目的：取得当前 row 中 G 个 query heads 的 LSE 上游梯度。
                    grad_lse_row = grad_lse_grouped[b, query, hkv, :]

                    # Math:
                    # dS_lse[g,m] = dL/dLSE_g * dLSE_g/dS[g,m]
                    #               = grad_lse_row[g] * P[g,m]。
                    #
                    # Shape:
                    # grad_lse_row [G] -> [:,None] [G,1]
                    # [G,1] * prob_selected [G,M] -> [G,M]。
                    #
                    # PyTorch note:
                    # ``None`` 插入 size=1 的 M 广播轴；乘法时 [G,1] 自动 broadcast 到 [G,M]。
                    grad_score_from_lse_selected = (
                        grad_lse_row[:, None] * prob_selected
                    )

                    # grad_score_selected [G, M]
                    #     += grad_score_from_lse_selected [G, M]。
                    # 本步目的：合并 output 和 LSE 两条通往 selected score 的梯度。
                    grad_score_selected += grad_score_from_lse_selected

                # endregion

                # region C04-D：Scaled score 反向传播

                # grad_q_row [G, D]
                #     = grad_score_selected [G, M] @ k_selected [M, D]
                #       / sqrt(D)。
                # 函数作用：矩阵乘法收缩 M 轴，再把前向的缩放系数传回 Q。
                # 本步目的：得到当前 row 中 G 个 query heads 的 Q 梯度。
                grad_q_row = (
                    grad_score_selected @ k_selected / math.sqrt(Dq)
                )

                # grad_k_selected [M, D]
                #     = grad_score_selected.T [M, G] @ q_row [G, D]
                #       / sqrt(D)。
                # 函数作用：transpose 交换 G、M 两轴，矩阵乘法随后收缩 G 轴；
                # 标量除法把前向的缩放系数传回 K。
                # 本步目的：合计 G 个 query heads 对 M 个 selected K 的梯度。
                grad_k_selected = (
                    grad_score_selected.transpose(-2, -1)
                    @ q_row
                    / math.sqrt(Dq)
                )

                # endregion

                # region C04-E：梯度写回原始布局

                # grad_q_grouped [B, Nq, Hkv, G, D] 的当前 row [G, D]
                #     = grad_q_row [G, D]。
                # 索引作用：整数 b、query、hkv 固定唯一的 Q row，保留 G、D 两轴。
                # 本步目的：把当前 Q 梯度写入全局 grouped Q 梯度存储。
                grad_q_grouped[b, query, hkv, :, :] = grad_q_row

                # Shape:
                # grad_k [B,Nk,Hkv,D]
                #   [b, token_indices_selected[M], hkv, :] -> [M,D]
                #   += grad_k_selected [M,D]。
                #
                # PyTorch note:
                # integer index b/hkv 删除对应轴；tensor advanced index [M] 在 Nk 轴选择
                # M 个原始 token 并产生 M 轴；``:`` 保留 D。当前 C01 effective support
                # 保证同一 row 内 token index 唯一，因此这里不会出现一次 advanced-index
                # update 内的 duplicate-index ambiguity。不同 query rows 再次命中同一 token 时，
                # 循环会在后续 ``+=`` 中继续累加。
                #
                # Why:
                # K 被不同 query rows 复用，所以不能覆盖已有梯度贡献。
                grad_k[b, token_indices_selected, hkv, :] += grad_k_selected

                # Shape:
                # grad_v [B,Nk,Hkv,D]
                #   [b, token_indices_selected[M], hkv, :] -> [M,D]
                #   += grad_v_selected [M,D]。
                #
                # PyTorch note:
                # advanced indexing 规则与 K 完全相同；使用同一 token_indices_selected
                # 保证 K/V writeback 对齐。这里同样使用累加而不是覆盖。
                grad_v[b, token_indices_selected, hkv, :] += grad_v_selected

                # endregion

    # region C04-E：恢复模型侧 Q 布局并返回

    # grad_q [B, Nq, Hq, D]
    #     = flatten(grad_q_grouped [B, Nq, Hkv, G, D], Hkv 与 G)。
    # 函数作用：flatten(2, 3) 合并相邻的 Hkv、G 两轴，保持 head 线性顺序。
    # 本步目的：把内部 grouped Q 梯度恢复成 public contract 的原始 Q 布局。
    grad_q = grad_q_grouped.flatten(2, 3)

    # 返回字段 dq/dk/dv 分别对应原始 q/k/v，shape、dtype、device 均已保持一致。
    return AttentionGradients(
        dq=grad_q,
        dk=grad_k,
        dv=grad_v,
    )

    # endregion
