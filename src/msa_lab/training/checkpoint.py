from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from msa_lab.config import ExperimentConfig, assert_resume_compatible
from msa_lab.training.state import TrainerState
from msa_lab.utils.rng import capture_rng_state, restore_rng_state


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    config: ExperimentConfig,
    trainer_state: TrainerState,
    data_generator_state: torch.Tensor,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 2,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": config.to_dict(),
        "trainer_state": trainer_state.to_dict(),
        "rng_state": capture_rng_state(),
        "data_generator_state": data_generator_state,
    }
    torch.save(payload, p)


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    current_config: ExperimentConfig,
    map_location: str | torch.device,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("schema_version") != 2:
        raise ValueError("unsupported checkpoint schema")
    assert_resume_compatible(current_config, payload["config"])
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload["scheduler"] is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    restore_rng_state(payload["rng_state"])
    return payload


def initialize_model_from_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    current_config: ExperimentConfig,
    map_location: str | torch.device,
) -> dict[str, Any]:
    """只复制模型权重；optimizer、step、RNG 与数据流均从新实验重新开始。"""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("schema_version") != 2:
        raise ValueError("unsupported checkpoint schema")
    saved_config = ExperimentConfig.from_dict(payload["config"])
    if current_config.model != saved_config.model:
        raise ValueError("initialization checkpoint model configuration differs")
    model.load_state_dict(payload["model"])
    return payload
