from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Any


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


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 1337
    device: str = "cpu"
    dtype: str = "float32"
    batch_size: int = 4
    seq_len: int = 32
    steps: int | None = 20
    # scheduled/full_only 保留单一 mode 合同；fixed_per_layer 用于 C07 分段实验。
    main_mode_policy: str = "scheduled"
    main_modes: tuple[str, ...] | None = None
    warmup_steps: int = 0
    learning_rate: float = 3e-4
    # Index 的有效学习率 = learning_rate * index_lr_multiplier。
    index_lr_multiplier: float = 1.0
    # joint 更新完整 LM + Index；index_only 冻结 Main/LM，仅拟合固定 teacher。
    parameter_update_policy: str = "joint"
    # 诊断模式：只采样一次训练 batch，之后每一步重复同一批 token。
    fixed_train_batch: bool = False
    # 可选固定数据池；训练按 batch_size 连续切片并循环，每个切片各更新一次。
    fixed_train_pool_size: int | None = None
    # Routing KL 是训练目标系数，不改变模型参数 shape。
    routing_kl_weight: float = 0.0
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    index_grad_clip: float | None = None
    log_every: int = 1
    eval_every: int = 0
    eval_batches: int = 4
    tensorboard: bool = False
    checkpoint_every: int = 10
    run_dir: str = "runs/dense_smoke"
    # 只加载模型权重并重置 optimizer/trainer/data state；用于受控新实验。
    initialize_from: str | None = None
    resume_from: str | None = None
    corpus_path: str | None = None
    train_fraction: float = 0.9

    def validate(self, model: ModelConfig) -> None:
        if self.seq_len > model.max_seq_len:
            raise ValueError("seq_len exceeds model.max_seq_len")
        if self.batch_size < 1 or (self.steps is not None and self.steps < 1):
            raise ValueError("batch_size must be positive and steps must be positive or null")
        has_msa = bool(model.resolved_msa_layer_ids())
        if self.main_mode_policy not in {"scheduled", "full_only", "fixed_per_layer"}:
            raise ValueError(
                "main_mode_policy must be scheduled, full_only, or fixed_per_layer"
            )
        if self.main_mode_policy in {"full_only", "fixed_per_layer"} and not has_msa:
            raise ValueError("full_only main_mode_policy requires an MSA layer")
        if self.main_mode_policy in {"full_only", "fixed_per_layer"} and self.warmup_steps != 0:
            raise ValueError(f"{self.main_mode_policy} main_mode_policy requires warmup_steps=0")
        if self.main_mode_policy == "fixed_per_layer":
            if self.main_modes is None or len(self.main_modes) != model.n_layers:
                raise ValueError("fixed_per_layer requires one main_modes entry per layer")
            if any(mode not in {"full", "sparse"} for mode in self.main_modes):
                raise ValueError("main_modes may only contain full or sparse")
        elif self.main_modes is not None:
            raise ValueError("main_modes requires fixed_per_layer main_mode_policy")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError(f"unsupported dtype: {self.dtype}")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.index_lr_multiplier <= 0:
            raise ValueError("index_lr_multiplier must be positive")
        if self.parameter_update_policy not in {"joint", "index_only"}:
            raise ValueError("parameter_update_policy must be joint or index_only")
        if self.parameter_update_policy == "index_only":
            if not has_msa:
                raise ValueError("index_only parameter_update_policy requires an MSA layer")
            if self.main_mode_policy != "full_only":
                raise ValueError("index_only parameter_update_policy requires full_only Main")
            if self.routing_kl_weight <= 0:
                raise ValueError("index_only parameter_update_policy requires positive routing KL")
            if self.initialize_from is None and self.resume_from is None:
                raise ValueError("index_only parameter_update_policy requires initialize_from")
        if self.fixed_train_batch and self.parameter_update_policy != "index_only":
            raise ValueError("fixed_train_batch requires index_only parameter_update_policy")
        if self.fixed_train_pool_size is not None:
            if not self.fixed_train_batch:
                raise ValueError("fixed_train_pool_size requires fixed_train_batch")
            if self.fixed_train_pool_size < self.batch_size:
                raise ValueError("fixed_train_pool_size must be >= batch_size")
            if self.fixed_train_pool_size % self.batch_size != 0:
                raise ValueError("fixed_train_pool_size must be divisible by batch_size")
        if self.initialize_from is not None and self.resume_from is not None:
            raise ValueError("initialize_from and resume_from are mutually exclusive")
        if self.routing_kl_weight < 0:
            raise ValueError("routing_kl_weight must be non-negative")
        if not has_msa and self.routing_kl_weight != 0:
            raise ValueError("routing_kl_weight requires an MSA layer")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive")
        if self.index_grad_clip is not None and self.index_grad_clip <= 0:
            raise ValueError("index_grad_clip must be positive")
        if self.log_every < 1 or self.checkpoint_every < 1:
            raise ValueError("log/checkpoint cadence must be positive")
        if self.eval_every < 0 or self.eval_batches < 1:
            raise ValueError("invalid evaluation cadence")
        if not (0.0 < self.train_fraction < 1.0):
            raise ValueError("train_fraction must be in (0, 1)")


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()

    def validate(self) -> None:
        self.model.validate()
        self.train.validate(self.model)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        model_data = dict(data.get("model", {}))
        if model_data.get("msa_layer_ids") is not None:
            model_data["msa_layer_ids"] = tuple(model_data["msa_layer_ids"])
        train_data = dict(data.get("train", {}))
        if train_data.get("main_modes") is not None:
            train_data["main_modes"] = tuple(train_data["main_modes"])
        # 旧 checkpoint/config 使用绝对 index_learning_rate；读取时迁移为乘数，
        # 新写出的配置只保留相对 Main LR 的直接语义。
        legacy_index_lr = train_data.pop("index_learning_rate", ...)
        if legacy_index_lr is not ...:
            if "index_lr_multiplier" in train_data:
                raise ValueError(
                    "cannot specify both index_learning_rate and index_lr_multiplier"
                )
            base_lr = train_data.get("learning_rate", TrainConfig.learning_rate)
            train_data["index_lr_multiplier"] = (
                1.0 if legacy_index_lr is None else legacy_index_lr / base_lr
            )
        cfg = cls(
            model=ModelConfig(**model_data),
            train=TrainConfig(**train_data),
        )
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


def assert_resume_compatible(current: ExperimentConfig, saved: dict[str, Any]) -> None:
    saved_cfg = ExperimentConfig.from_dict(saved)
    if current.model != saved_cfg.model:
        raise ValueError("resume model configuration differs from checkpoint")

    # These fields determine the actual data stream or optimizer trajectory and
    # therefore must remain identical across an exact resume. Operational fields
    # such as steps, run_dir, logging cadence and checkpoint cadence may change.
    fields = (
        "seed",
        "device",
        "dtype",
        "batch_size",
        "seq_len",
        "main_mode_policy",
        "main_modes",
        "warmup_steps",
        "learning_rate",
        "index_lr_multiplier",
        "parameter_update_policy",
        "fixed_train_batch",
        "fixed_train_pool_size",
        "routing_kl_weight",
        "weight_decay",
        "grad_clip",
        "index_grad_clip",
        "corpus_path",
        "train_fraction",
        "initialize_from",
    )
    mismatches = [
        name
        for name in fields
        if getattr(current.train, name) != getattr(saved_cfg.train, name)
    ]
    if mismatches:
        raise ValueError(f"resume training contract differs: {mismatches}")
