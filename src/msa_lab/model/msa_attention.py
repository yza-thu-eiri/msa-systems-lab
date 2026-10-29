from __future__ import annotations

import torch

from torch import nn

from msa_lab.backends.dense_sdpa import DenseSDPABackend
from msa_lab.config import ModelConfig
from msa_lab.contracts.support import block_ids_to_group_mask
from msa_lab.reference.gathered_sparse import gathered_sparse_gqa
from msa_lab.reference.index_blockmax import causal_index_blockmax
from msa_lab.reference.routing_kl import routing_kl_gqa
from msa_lab.reference.routing_diagnostics import routing_distribution_stats_gqa
from msa_lab.reference.selection import select_topk_blocks


class CausalMSAAttention(nn.Module):
    """P2-C03 learner-owned one-layer MSA attention。"""

    def __init__(
            self, 
            cfg: ModelConfig, 
            layer_id: int, 
            *,
            index_dim: int,
            block_size: int,
            top_k: int
    ) -> None:
        super().__init__()

        self.layer_id = layer_id
        self.n_q_heads = cfg.n_q_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_q_heads

        self.index_dim = index_dim
        self.block_size = block_size
        self.top_k = top_k

        self.q_proj = nn.Linear(
            in_features=cfg.d_model,
            out_features=self.head_dim * self.n_q_heads,
            bias=False
        )

        self.k_proj = nn.Linear(
            in_features=cfg.d_model,
            out_features=self.head_dim * self.n_kv_heads,
            bias=False
        )

        self.v_proj = nn.Linear(
            in_features=cfg.d_model,
            out_features=self.head_dim * self.n_kv_heads,
            bias=False
        )

        self.o_proj = nn.Linear(
            in_features=self.head_dim * self.n_q_heads,
            out_features=cfg.d_model,
            bias=False
        )

        self.q_idx_proj = nn.Linear(
            in_features=cfg.d_model,
            out_features=self.index_dim * self.n_kv_heads,
            bias=False
        )

        self.k_idx_proj = nn.Linear(
            in_features=cfg.d_model,
            out_features=self.index_dim,
            bias=False
        )

        self.full_backend = DenseSDPABackend()


    def forward(
        self,
        x_attn: torch.Tensor,
        *,
        fixed_block_ids: torch.Tensor | None = None,
        compute_routing_kl: bool = False,
        compute_routing_diagnostics: bool = False,
        return_aux: bool = False,
        main_mode: str = "sparse",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

        if main_mode not in {"full", "sparse"}:
            raise ValueError("main_mode 必须是 'full' 或 'sparse'")
        if main_mode == "full" and fixed_block_ids is not None:
            raise ValueError("fixed_block_ids 只适用于 sparse Main")

        batch_size, sequence_length, _ = x_attn.shape

        # B,T,Hq*HEAD
        q = self.q_proj(x_attn).view(
            batch_size,
            sequence_length,
            self.n_q_heads,
            self.head_dim,
        )

        # B,T,Hkv*HEAD
        k = self.k_proj(x_attn).view(
            batch_size,
            sequence_length,
            self.n_kv_heads,
            self.head_dim,
        )

        # B,T,Hkv*HEAD
        v = self.v_proj(x_attn).view(
            batch_size,
            sequence_length,
            self.n_kv_heads,
            self.head_dim,
        )

        # SPARSE dynamic path 用 selection 控制 Main；FULL 只在 diagnostics 中
        # 额外观察 selection，它不参与 FULL Main 的 support。
        need_dynamic_selection = (
            (main_mode == "sparse" and fixed_block_ids is None)
            or (main_mode == "full" and return_aux)
        )

        # 动态 selection 与 routing supervision 都需要 Index Q/K。fixed support
        # 只替换 support producer；需要 KL 时仍要计算 student scorer。
        need_index_qk = (
            need_dynamic_selection
            or compute_routing_kl
            or compute_routing_diagnostics
        )
        q_idx: torch.Tensor | None = None
        k_idx: torch.Tensor | None = None
        if need_index_qk:
            x_idx = x_attn.detach()

            # B,T,Hkv*IDX -> B,T,Hkv,IDX
            q_idx = self.q_idx_proj(x_idx).view(
                batch_size,
                sequence_length,
                self.n_kv_heads,
                self.index_dim,
            )

            # B,T,IDX -> B,T,1,IDX
            k_idx = self.k_idx_proj(x_idx).unsqueeze(2)

        # Selection metadata 只在 SPARSE Main 或 FULL diagnostics 中存在。
        midx: torch.Tensor | None = None
        block_ids: torch.Tensor | None = None
        valid_count: torch.Tensor | None = None
        if need_dynamic_selection:
            assert q_idx is not None and k_idx is not None

            # [B, Nq, Hkv, num_blocks]
            midx = causal_index_blockmax(
                q_idx=q_idx,
                k_idx=k_idx,
                block_size=self.block_size,
            )

            # [B,N,Hkv,k], [B,N,Hkv]
            block_ids, valid_count = select_topk_blocks(
                midx=midx,
                block_size=self.block_size,
                k=self.top_k,
            )
        elif main_mode == "sparse":
            assert fixed_block_ids is not None
            block_ids = fixed_block_ids
            valid_count = (fixed_block_ids >= 0).sum(dim=-1)

        # Main Q/K/V projections are shared; only the execution operator changes.
        if main_mode == "full":
            main_result = self.full_backend(
                q=q,
                k=k,
                v=v,
                causal=True,
                return_lse=return_aux,
            )
        else:
            assert block_ids is not None
            main_result = gathered_sparse_gqa(
                q=q,
                k=k,
                v=v,
                block_ids=block_ids,
                block_size=self.block_size,
            )

        routing_kl: torch.Tensor | None = None
        routing_diagnostics: dict[str, torch.Tensor] = {}
        if compute_routing_kl or compute_routing_diagnostics:
            assert q_idx is not None and k_idx is not None

            if main_mode == "full":
                # [Nq,Nk]：query q 只允许 key k <= q。
                positions = torch.arange(sequence_length, device=x_attn.device)
                causal_support = positions[None, :] <= positions[:, None]

                # [Nq,Nk] -> [B,Nq,Hkv,Nk]
                token_support = causal_support[None, :, None, :].expand(
                    batch_size,
                    -1,
                    self.n_kv_heads,
                    -1,
                )
                token_capacity = sequence_length
            else:
                assert block_ids is not None
                token_support = block_ids_to_group_mask(
                    block_ids,
                    seq_len=sequence_length,
                    block_size=self.block_size,
                )
                token_capacity = min(
                    sequence_length,
                    block_ids.shape[-1] * self.block_size,
                )

            if compute_routing_kl:
                # C04 Reference 在最终 support 上局部重算 Main/Index logits；Main backend
                # 不需要扩大接口暴露内部 logits。
                routing_kl = routing_kl_gqa(
                    q=q,
                    k=k,
                    q_idx=q_idx,
                    k_idx=k_idx,
                    token_support=token_support,
                    token_capacity=token_capacity,
                )
            if compute_routing_diagnostics:
                routing_diagnostics = routing_distribution_stats_gqa(
                    q=q,
                    k=k,
                    q_idx=q_idx,
                    k_idx=k_idx,
                    token_support=token_support,
                )

        # Both Main operators return [B,N,Hq,D] in the same AttentionOutput contract.
        main_output = main_result.output.flatten(
            start_dim=2,
            end_dim=3,
        )

        # [B, N, Hq*D] -> [B, N, d_model]
        attn_out = self.o_proj(main_output)

        aux: dict[str, torch.Tensor] = {}
        # routing_kl 是训练所需量，不由诊断开关 return_aux 决定是否返回。
        if routing_kl is not None:
            aux[f"layer_{self.layer_id}.routing_kl"] = routing_kl
        for name, value in routing_diagnostics.items():
            aux[f"layer_{self.layer_id}.{name}"] = value

        if return_aux:
            if main_result.lse is not None:
                aux[f"layer_{self.layer_id}.main_lse"] = main_result.lse

            if block_ids is not None:
                assert valid_count is not None
                aux[f"layer_{self.layer_id}.block_ids"] = block_ids
                aux[f"layer_{self.layer_id}.valid_count"] = valid_count

            if midx is not None:
                aux[f"layer_{self.layer_id}.midx"] = midx

        return attn_out, aux
