from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 128
    max_seq_len: int = 64
    d_model: int = 128
    n_layers: int = 2
    n_q_heads: int = 4
    n_kv_heads: int = 2
    d_ff: int = 256
    dropout: float = 0.0
    attention_backend: str = "dense_sdpa"
    # C06-B: ``None`` 保持原有纯 Dense 模型；非空值指定唯一的 MSA block。
    msa_layer_id: int | None = None
    # C07: 新配置可显式指定多个 MSA blocks。旧 ``msa_layer_id`` 继续只读兼容，
    # 两个字段不能同时成为结构 authority。
    msa_layer_ids: tuple[int, ...] | None = None
    msa_index_dim: int = 16
    msa_block_size: int = 16
    msa_top_k: int = 2

    def resolved_msa_layer_ids(self) -> tuple[int, ...]:
        if self.msa_layer_ids is not None:
            return tuple(self.msa_layer_ids)
        if self.msa_layer_id is not None:
            return (self.msa_layer_id,)
        return ()

    def validate(self) -> None:
        if self.vocab_size < 8:
            raise ValueError("vocab_size must be >= 8")
        if self.max_seq_len < 2:
            raise ValueError("max_seq_len must be >= 2")
        if self.d_model % self.n_q_heads != 0:
            raise ValueError("d_model must be divisible by n_q_heads")
        if self.n_q_heads % self.n_kv_heads != 0:
            raise ValueError("n_q_heads must be divisible by n_kv_heads")
        if self.d_ff < self.d_model:
            raise ValueError("d_ff should be at least d_model")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")
        if not self.attention_backend:
            raise ValueError("attention_backend must be non-empty")
        if self.msa_layer_id is not None and self.msa_layer_ids is not None:
            raise ValueError("msa_layer_id and msa_layer_ids cannot both be set")
        msa_layer_ids = self.resolved_msa_layer_ids()
        if len(set(msa_layer_ids)) != len(msa_layer_ids):
            raise ValueError("msa_layer_ids must not contain duplicates")
        for layer_id in msa_layer_ids:
            if not 0 <= layer_id < self.n_layers:
                raise ValueError("msa_layer_id/msa_layer_ids must identify existing layers")
        if msa_layer_ids:
            if self.msa_index_dim < 1:
                raise ValueError("msa_index_dim must be positive")
            if self.msa_block_size < 1:
                raise ValueError("msa_block_size must be positive")
            if self.msa_top_k < 1:
                raise ValueError("msa_top_k must be positive")


