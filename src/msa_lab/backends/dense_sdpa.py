from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from .base import AttentionOutput


class DenseSDPABackend:
    """Dense causal GQA backend using [B, T, H, D] tensors."""

    name = "dense_sdpa"

    @staticmethod
    def _expand_kv(x: torch.Tensor, n_q_heads: int) -> torch.Tensor:
        n_kv_heads = x.shape[2]
        if n_q_heads % n_kv_heads != 0:
            raise ValueError("query heads must be divisible by KV heads")
        if n_q_heads == n_kv_heads:
            return x
        return x.repeat_interleave(n_q_heads // n_kv_heads, dim=2)

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
        return_lse: bool = False,
        return_aux: bool = False,
    ) -> AttentionOutput:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("q, k and v must be rank-4 [B,T,H,D]")
        if q.shape[0] != k.shape[0] or k.shape[:3] != v.shape[:3]:
            raise ValueError("incompatible q/k/v batch, sequence or KV-head shapes")
        if q.shape[-1] != k.shape[-1]:
            raise ValueError("Q and K head dimensions must match")

        k_expanded = self._expand_kv(k, q.shape[2])
        v_expanded = self._expand_kv(v, q.shape[2])

        q_t = q.transpose(1, 2)
        k_t = k_expanded.transpose(1, 2)
        v_t = v_expanded.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            dropout_p=0.0,
            is_causal=causal,
        ).transpose(1, 2)

        lse = None
        if return_lse:
            scores = torch.matmul(q_t.float(), k_t.float().transpose(-1, -2))
            scores = scores / math.sqrt(q.shape[-1])
            if causal:
                tq, tk = q.shape[1], k.shape[1]
                if tq != tk:
                    raise NotImplementedError(
                        "seed LSE path only supports causal self-attention Tq == Tk"
                    )
                mask = torch.ones(tq, tk, device=q.device, dtype=torch.bool).tril()
                scores = scores.masked_fill(~mask, float("-inf"))
            lse = torch.logsumexp(scores, dim=-1).transpose(1, 2)

        aux = {}
        if return_aux:
            aux["backend_code"] = torch.tensor(0, device=q.device, dtype=torch.int32)
        return AttentionOutput(output=out, lse=lse, aux=aux)
