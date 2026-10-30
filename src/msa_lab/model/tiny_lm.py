from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
import torch.nn.functional as F

from msa_lab.backends.registry import build_attention_backend
from msa_lab.config import ModelConfig
from msa_lab.model.msa_attention import CausalMSAAttention


@dataclass
class LMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None
    aux: dict[str, torch.Tensor] = field(default_factory=dict)


class CausalGQAAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.n_q_heads = cfg.n_q_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_q_heads
        self.backend = build_attention_backend(cfg.attention_backend)

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_q_heads * self.head_dim, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_q_heads, self.head_dim)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim)
        result = self.backend(
            q, k, v, causal=True, return_lse=return_aux, return_aux=return_aux
        )
        aux: dict[str, torch.Tensor] = {}
        if return_aux:
            if result.lse is not None:
                aux[f"layer_{self.layer_id}.main_lse"] = result.lse
            for name, value in result.aux.items():
                aux[f"layer_{self.layer_id}.{name}"] = value
        return self.o_proj(result.output.reshape(b, t, -1)), aux


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_id: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        if layer_id in cfg.resolved_msa_layer_ids():
            self.attn: CausalGQAAttention | CausalMSAAttention = CausalMSAAttention(
                cfg,
                layer_id,
                index_dim=cfg.msa_index_dim,
                block_size=cfg.msa_block_size,
                top_k=cfg.msa_top_k,
            )
        else:
            self.attn = CausalGQAAttention(cfg, layer_id)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(
        self,
        x: torch.Tensor,
        *,
        main_mode: str = "sparse",
        compute_routing_kl: bool = False,
        compute_routing_diagnostics: bool = False,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x_attn = self.ln1(x)
        if isinstance(self.attn, CausalMSAAttention):
            attn_out, aux = self.attn(
                x_attn,
                main_mode=main_mode,
                compute_routing_kl=compute_routing_kl,
                compute_routing_diagnostics=compute_routing_diagnostics,
                return_aux=return_aux,
            )
        else:
            attn_out, aux = self.attn(x_attn, return_aux=return_aux)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, aux


class TinyCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, layer_id=i) for i in range(cfg.n_layers)]
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        main_mode: str | None = None,
        main_modes: tuple[str, ...] | list[str] | None = None,
        compute_routing_kl: bool = False,
        compute_routing_diagnostics: bool = False,
        return_aux: bool = False,
    ) -> LMOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B,T]")
        _, t = input_ids.shape
        if t > self.cfg.max_seq_len:
            raise ValueError("sequence exceeds model.max_seq_len")

        positions = torch.arange(t, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None]
        aux: dict[str, torch.Tensor] = {}
        if main_mode is not None and main_modes is not None:
            raise ValueError("main_mode 与 main_modes 不能同时提供")
        if main_modes is None:
            shared_mode = "sparse" if main_mode is None else main_mode
            resolved_main_modes = (shared_mode,) * len(self.blocks)
        else:
            resolved_main_modes = tuple(main_modes)
            if len(resolved_main_modes) != len(self.blocks):
                raise ValueError("main_modes 必须为每个 Transformer layer 提供一个 mode")
        if any(mode not in {"full", "sparse"} for mode in resolved_main_modes):
            raise ValueError("main_modes 只能包含 'full' 或 'sparse'")

        for layer_id, block in enumerate(self.blocks):
            x, block_aux = block(
                x,
                main_mode=resolved_main_modes[layer_id],
                compute_routing_kl=compute_routing_kl,
                compute_routing_diagnostics=compute_routing_diagnostics,
                return_aux=return_aux,
            )
            aux.update(block_aux)
        logits = self.lm_head(self.final_norm(x))

        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must match input_ids shape")
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        return LMOutput(logits=logits, loss=loss, aux=aux)
