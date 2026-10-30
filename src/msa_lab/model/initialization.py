from __future__ import annotations

from dataclasses import replace
import math

import torch
from torch import nn

from msa_lab.config import ModelConfig
from msa_lab.model.msa_attention import CausalMSAAttention
from msa_lab.model.tiny_lm import TinyCausalLM


def build_model_from_dense_anchor(cfg: ModelConfig, *, seed: int) -> TinyCausalLM:
    """构造可与同 seed Dense 母本逐参数比较的 TinyLM。"""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        msa_layer_ids = cfg.resolved_msa_layer_ids()
        if not msa_layer_ids:
            return TinyCausalLM(cfg)

        dense_cfg = replace(cfg, msa_layer_id=None, msa_layer_ids=None)
        dense_anchor = TinyCausalLM(dense_cfg)
        model = TinyCausalLM(cfg)

        incompatible = model.load_state_dict(dense_anchor.state_dict(), strict=False)
        expected_missing = {
            f"blocks.{layer_id}.attn.{name}_idx_proj.weight"
            for layer_id in msa_layer_ids
            for name in ("q", "k")
        }
        if set(incompatible.missing_keys) != expected_missing:
            raise RuntimeError(
                "Dense→MSA 复制遗漏项不符合预期: "
                f"{incompatible.missing_keys}"
            )
        if incompatible.unexpected_keys:
            raise RuntimeError(
                "Dense→MSA 复制出现非预期参数: "
                f"{incompatible.unexpected_keys}"
            )

        for layer_id in msa_layer_ids:
            attention = model.blocks[layer_id].attn
            if not isinstance(attention, CausalMSAAttention):
                raise RuntimeError("MSA layer id 没有构造出 CausalMSAAttention")

            # 每层重建相同 seed 的 generator，保留 C06“候选替换层具有相同
            # Index 初始化”的公平合同；多层模型随后可由 checkpoint 覆盖既有 Index。
            index_generator = torch.Generator(device="cpu")
            index_generator.manual_seed(seed + 2)
            nn.init.kaiming_uniform_(
                attention.q_idx_proj.weight,
                a=math.sqrt(5),
                generator=index_generator,
            )
            nn.init.kaiming_uniform_(
                attention.k_idx_proj.weight,
                a=math.sqrt(5),
                generator=index_generator,
            )
        return model
