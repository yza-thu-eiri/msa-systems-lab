"""P2-C04 Routing KL 的透明 Reference 与分步学习骨架。

本文件提供独立的 KL 计算；attention forward 只负责准备 projected Tensor 与 support，
再调用这个 Reference。学习顺序是：

```
Main Q 的 GQA 分组
-> 把每行允许的 token 集合整理成 padded token IDs
-> 批量取得 Main / Index selected K
-> 构造 Main teacher 与 Index student
-> 每行 KL 与全局 reduction
-> 独立数值证据
```

当前版本优先暴露 Tensor 轴和数学语义，不追求 production kernel 的执行结构。
"""

from __future__ import annotations

import torch


def routing_kl_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    token_support: torch.Tensor,
    *,
    token_capacity: int,
) -> torch.Tensor:
    """在同一组允许 token 上计算 Main teacher 与 Index student 的 routing KL。

    输入：
        q: ``[B,N,Hq,D]`` 的 Main query。
        k: ``[B,N,Hkv,D]`` 的 Main key。
        q_idx: ``[B,N,Hkv,didx]`` 的 Index query。
        k_idx: ``[B,N,1,didx]`` 的共享 Index key。
        token_support: ``[B,N,Hkv,N]`` 的 bool Tensor；True 表示该 key token
            允许参与当前 query/KV-group 的 teacher/student 比较。

    输出：
        标量 routing KL。KL 先在每个 ``[b,query,hkv]`` row 的 token 轴求和，
        再对有效 row 取平均。

    梯度合同：
        - Main teacher probability 在本函数内 detach；
        - q_idx/k_idx 保持可导，使 KL 能更新 Index Q/K projections；
        - shared input 的 detach 由产生 q_idx/k_idx 的 attention 层负责。
    """

    # =====================================================================
    # P2-C04-A — Main Q 的 GQA 分组
    # =====================================================================
    #
    # 目标：
    #   q [B,N,Hq,D] -> q_grouped [B,N,Hkv,G,D]
    #   其中 Hq = Hkv * G，原 head h 对应 h = hkv * G + g。
    #
    # 本阶段不改变 K 的数学含义：Main K 仍是一组 KV head，输入布局是
    # [B,N,Hkv,D]。如果临时增加 singleton 轴，需要在后续 gather/matmul 前
    # 明确它代表什么，不能让一个 size=1 的轴悄悄改变接口语义。
    batch_size, sequence_length, hq, d_head = q.shape
    _, _, hkv, _ = k.shape
    group_size = hq // hkv

    q_grouped = q.reshape(batch_size, sequence_length, hkv, group_size, d_head)


    # =====================================================================
    # P2-C04-B — 允许-token集合 -> padded token IDs
    # =====================================================================
    #
    # 输入：
    #   token_support [B,N,Hkv,N] bool
    #   True 表示该 key position 允许参与当前 query/KV-group 的 KL 比较。
    #
    # 先得到：
    #   row_token_count [B,N,Hkv]
    #   M = 上游提供的固定 token_capacity；row_token_count 用于容量检查和有效行判断
    #
    # 再构造：
    #   selected_token_ids [B,N,Hkv,M]，真实 token ID 升序，尾部 padding=-1
    #   selected_valid     [B,N,Hkv,M]，True 表示对应槽位是真实 token
    #   safe_token_ids     [B,N,Hkv,M]，把 -1 临时替换为合法 gather index
    #
    # 本阶段必须保持：
    #   1. token_support 中的每个 True position 恰好出现一次；
    #   2. 每行真实 token ID 保持升序；
    #   3. -1 只表达 metadata padding，不直接进入 gather。
    #
    num_key_tokens = token_support.shape[-1]
    if token_capacity <= 0:
        raise ValueError("token_capacity 必须大于 0")
    if token_capacity > num_key_tokens:
        raise ValueError("token_capacity 不能超过 key-token 轴长度")

    row_token_count = token_support.sum(dim=-1)  # [B,N,Hkv]
    if torch.any(row_token_count > token_capacity):
        raise ValueError("token_capacity 不能截断 token_support 中的合法 token")

    # [B,N,Hkv,N]
    positions = torch.arange(
        num_key_tokens,
        device=token_support.device,
    ).expand_as(token_support)

    # [B,N,Hkv,N]
    sentinel = num_key_tokens
    selected_token_ids, _ = positions.masked_fill(
        mask= ~token_support,
        value=sentinel,
    ).sort(dim=-1, descending=False)

    # [B,N,Hkv,token_capacity]
    selected_token_ids = selected_token_ids.masked_fill(
        mask=(selected_token_ids == sentinel),
        value=-1
    )[..., :token_capacity]

    selected_valid = (selected_token_ids >= 0)
    safe_token_ids = selected_token_ids.masked_fill(
        mask= ~selected_valid,
        value=0
    )


    # =====================================================================
    # P2-C04-C — 批量取得 Main / Index selected K
    # =====================================================================
    #
    # 输入：
    #   k              [B,Nkey,Hkv,D]
    #   k_idx          [B,Nkey,1,didx]
    #   safe_token_ids [B,Nquery,Hkv,M]
    #
    # 目标：
    #   k_selected     [B,Nquery,Hkv,M,D]
    #   k_idx_selected [B,Nquery,Hkv,M,didx]
    #
    # k_selected[b,q,hkv,m,d]
    # = k[b, safe_token_ids[b,q,hkv,m], hkv, d]
    # 两个输出的同一 [b,query,hkv,m] 必须来自同一个原始 key token。

    # [B,Nquery,Hkv,M] -> [B,Nquery,Hkv,M,D]
    main_gather_index = safe_token_ids.unsqueeze(4).expand(
        -1, -1, -1, -1, d_head
    )

    # [B,Nkey,Hkv,D] -> [B,Hkv,Nkey,D] -> [B,Nquery,Hkv,Nkey,D]
    k_for_query = k.permute(0, 2, 1, 3).unsqueeze(1).expand(
        -1, sequence_length, -1, -1, -1
    )

    # [B,Nquery,Hkv,M,D]
    k_selected = torch.gather(
        input=k_for_query,
        dim=3,
        index=main_gather_index
    )

    # Index Q/K 的特征宽度；Index logits 使用 sqrt(didx) 缩放。
    d_idx = q_idx.shape[-1]

    # [B,Nk,1,didx] -> [B,1,Nk,didx] -> [B,1,1,Nk,didx]
    # -> [B,Nq,Hkv,Nk,didx]
    k_idx_for_query = k_idx.permute(0, 2, 1, 3).unsqueeze(1).expand(
        -1, sequence_length, hkv, -1, -1
    )

    # [B,Nq,Hkv,M] -> [B,Nq,Hkv,M,didx]
    idx_gather_index = safe_token_ids.unsqueeze(4).expand(
        -1, -1, -1, -1, d_idx
    )

    # [B,Nq,Hkv,M,didx]
    # k_idx_selected[b,q,hkv,m,i]
    # = k_idx[b, safe_token_ids[b,q,hkv,m], 0, i]
    k_idx_selected = torch.gather(
        input=k_idx_for_query,
        dim=3,
        index=idx_gather_index
    )

    # =====================================================================
    # P2-C04-D — 批量构造 Main / Index logits
    # =====================================================================
    #
    # 输入：
    #   q_grouped      [B,N,Hkv,G,D]
    #   k_selected     [B,N,Hkv,M,D]
    #   q_idx          [B,N,Hkv,didx]
    #   k_idx_selected [B,N,Hkv,M,didx]
    #
    # 目标：
    #   main_logits  [B,N,Hkv,G,M]，缩放因子 sqrt(D)
    #   index_logits [B,N,Hkv,M]，缩放因子 sqrt(didx)

    # [B,N,Hkv,G,D] @ [B,N,Hkv,D,M] -> [B,N,Hkv,G,M]
    # 每个 Main query head 分别给 M 个 selected token 打分。
    main_logits = (
        q_grouped @ k_selected.transpose(-2, -1)
    ) / d_head ** 0.5

    # [B,N,Hkv,didx] -> [B,N,Hkv,1,didx]
    # [B,N,Hkv,M,didx] -> [B,N,Hkv,didx,M]
    # matmul 得到 [B,N,Hkv,1,M]；squeeze 删除人为加入的 size=1 轴。
    index_logits = (
        q_idx.unsqueeze(-2) @ k_idx_selected.transpose(-2, -1)
    ).squeeze(-2) / d_idx ** 0.5


    # =====================================================================
    # P2-C04-E — 同一 token 轴上的 teacher / student
    # =====================================================================
    #
    # softmax 前先用 selected_valid 把 padding logits 设为 -inf。
    #
    # Main：
    #   main_logits --softmax over M--> per_head_prob [B,N,Hkv,G,M]
    #   --mean over G--> teacher_prob [B,N,Hkv,M]
    #   --detach--> routing KL 不更新 Main Q/K
    #
    # Index：
    #   index_logits --softmax over M--> student_prob [B,N,Hkv,M]
    #   student_prob 保持可导，使 KL 更新 Index Q/K projections。

    row_valid = row_token_count > 0                  # [B,N,Hkv]
    main_valid = selected_valid.unsqueeze(-2)        # [B,N,Hkv,1,M]

    # Main 的 size=1 轴广播到 G；Index 与 selected_valid 已经同为四维。
    masked_main_logits = main_logits.masked_fill(~main_valid, float("-inf"))
    masked_index_logits = index_logits.masked_fill(~selected_valid, float("-inf"))

    # 空 row 会在 F 阶段排除。这里临时使用全零 logits，避免对全 -inf
    # 做 softmax 产生 NaN。
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

    per_head_prob = torch.softmax(masked_main_logits, dim=-1)  # [B,N,Hkv,G,M]
    teacher_prob = per_head_prob.mean(dim=3).detach()          # [B,N,Hkv,M]
    student_prob = torch.softmax(masked_index_logits, dim=-1)  # [B,N,Hkv,M]

    # =====================================================================
    # P2-C04-F — row KL 与 valid-row reduction
    # =====================================================================
    #
    # 目标：
    #   row_kl [B,N,Hkv]：KL 内部沿 token 轴 M 求和，不额外除以 token 数。
    #   routing_kl []：只对 row_token_count > 0 的 row 取平均。
    #  KL = sum(PlogP - PlogQ)
    # 输出必须保持为 PyTorch scalar Tensor，不能用 .item() 离开 autograd graph。

    # F1 — 稳定计算 student 的 log Q。
    # 直接使用 log_softmax，避免先 softmax 再 log 时，小概率下溢成 0。
    # 对合法 row 的 padding，log Q 是 -inf；对 E 中临时置零的空 row，
    # log Q 是有限的均匀分布。两者都依据 selected_valid 统一置为 0，
    # 从而避免后续出现 0 * -inf，同时不改变 padding 的零 KL 贡献。
    student_log_prob = torch.log_softmax(
        masked_index_logits,
        dim=-1,
    )
    safe_student_log_prob = student_log_prob.masked_fill(
        ~selected_valid,
        0.0,
    )

    # F2 — 逐 token 计算 P log P - P log Q。
    # torch.xlogy(P, P) 显式实现 0 * log(0) = 0，避免 teacher 在
    # padding 或浮点下溢位置产生 NaN。teacher_prob 已在 E 中 detach，
    # 因此这一项不会把 routing KL 梯度传回 Main Q/K。
    teacher_self_term = torch.xlogy(teacher_prob, teacher_prob)
    token_kl = (
        teacher_self_term
        - teacher_prob * safe_student_log_prob
    )

    # 固定容量 M 中的 padding 不属于真实概率空间，KL 贡献必须严格为 0。
    # 这一步也把 E 中仅为避免 NaN 而构造的空-row占位分布全部清零。
    token_kl = token_kl.masked_fill(~selected_valid, 0.0)

    # F3 — KL 的定义要求沿 token/support 轴求和，不再除以 token 数。
    # [B,N,Hkv,M] -> [B,N,Hkv]
    row_kl = token_kl.sum(dim=-1)

    # F4 — 只在至少含一个真实 token 的 row 上取平均。
    # clamp_min(1) 只处理“整个 batch 没有有效 row”的边界：此时分子为
    # 可导的 0、分母为 1，返回 scalar 0，而不是空集合 mean 产生的 NaN。
    row_weight = row_valid.to(dtype=row_kl.dtype)
    valid_row_count = row_weight.sum()
    routing_kl = (
        row_kl * row_weight
    ).sum() / valid_row_count.clamp_min(1.0)

    return routing_kl
