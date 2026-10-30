import copy
import random
from pathlib import Path
from typing import Any

import pytest
import torch

from msa_lab.config import ExperimentConfig, ModelConfig, TrainConfig
from msa_lab.model.msa_attention import CausalMSAAttention
from msa_lab.training.checkpoint import load_checkpoint, save_checkpoint
from msa_lab.training.mode import main_mode_for_step
from msa_lab.training.state import TrainerState
from msa_lab.utils.rng import capture_rng_state, seed_everything


def _config(run_dir: Path) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=16,
            max_seq_len=4,
            d_model=8,
            n_layers=1,
            n_q_heads=4,
            n_kv_heads=2,
            d_ff=16,
        ),
        train=TrainConfig(
            seed=61,
            device="cpu",
            dtype="float32",
            batch_size=1,
            seq_len=4,
            steps=4,
            warmup_steps=2,
            learning_rate=1e-2,
            checkpoint_every=1,
            log_every=1,
            eval_every=0,
            run_dir=str(run_dir),
        ),
    )


def _build_stack(config: ExperimentConfig):
    attention = CausalMSAAttention(
        config.model,
        layer_id=0,
        index_dim=3,
        block_size=2,
        top_k=2,
    )
    optimizer = torch.optim.AdamW(
        attention.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: 0.95**step,
    )
    state = TrainerState()
    data_generator = torch.Generator(device="cpu")
    data_generator.manual_seed(config.train.seed + 1)
    return attention, optimizer, scheduler, state, data_generator


def _run_steps(
    *,
    attention: CausalMSAAttention,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    state: TrainerState,
    data_generator: torch.Generator,
    config: ExperimentConfig,
    steps: int,
) -> list[str]:
    observed_modes: list[str] = []

    def record_main_mode(_module, _args, kwargs) -> None:
        observed_modes.append(kwargs["main_mode"])

    hook = attention.register_forward_pre_hook(record_main_mode, with_kwargs=True)
    try:
        for _ in range(steps):
            x_attn = torch.randn(
                config.train.batch_size,
                config.train.seq_len,
                config.model.d_model,
                generator=data_generator,
            )
            # 独立 data generator 与 global Torch RNG 都进入真实数值轨迹。
            x_attn = x_attn + 0.01 * torch.rand_like(x_attn)
            main_mode = main_mode_for_step(
                state.global_step,
                config.train.warmup_steps,
            )

            optimizer.zero_grad(set_to_none=True)
            attn_out, aux = attention(
                x_attn,
                main_mode=main_mode,
                compute_routing_kl=True,
            )
            # Python RNG 也实际影响梯度，避免只比较一个从未消费的 RNG state。
            loss_scale = 0.9 + 0.2 * random.random()
            loss = loss_scale * (
                attn_out.square().mean() + aux["layer_0.routing_kl"]
            )
            loss.backward()
            optimizer.step()
            scheduler.step()

            state.global_step += 1
            state.tokens_seen += config.train.batch_size * config.train.seq_len
    finally:
        hook.remove()

    return observed_modes


def _snapshot(
    *,
    attention: CausalMSAAttention,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    state: TrainerState,
    data_generator: torch.Generator,
) -> dict[str, Any]:
    return copy.deepcopy(
        {
            "model": attention.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "trainer_state": state.to_dict(),
            "rng_state": capture_rng_state(),
            "data_generator_state": data_generator.get_state(),
        }
    )


def _assert_nested_exact(actual: Any, expected: Any, path: str) -> None:
    if isinstance(actual, torch.Tensor):
        assert isinstance(expected, torch.Tensor), path
        assert actual.shape == expected.shape, path
        assert actual.dtype == expected.dtype, path
        assert torch.equal(actual, expected), path
        return

    if isinstance(actual, dict):
        assert isinstance(expected, dict), path
        assert actual.keys() == expected.keys(), path
        for key in actual:
            _assert_nested_exact(actual[key], expected[key], f"{path}.{key}")
        return

    if isinstance(actual, (list, tuple)):
        assert type(actual) is type(expected), path
        assert len(actual) == len(expected), path
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_nested_exact(actual_item, expected_item, f"{path}[{index}]")
        return

    assert actual == expected, path


@pytest.mark.parametrize("split_step", [1, 2, 3])
def test_msa_exact_resume_across_warmup_boundary(
    tmp_path: Path,
    split_step: int,
) -> None:
    config = _config(tmp_path)
    total_steps = config.train.steps

    seed_everything(config.train.seed)
    (
        uninterrupted_attention,
        uninterrupted_optimizer,
        uninterrupted_scheduler,
        uninterrupted_state,
        uninterrupted_data_generator,
    ) = _build_stack(config)
    uninterrupted_modes = _run_steps(
        attention=uninterrupted_attention,
        optimizer=uninterrupted_optimizer,
        scheduler=uninterrupted_scheduler,
        state=uninterrupted_state,
        data_generator=uninterrupted_data_generator,
        config=config,
        steps=total_steps,
    )
    uninterrupted_snapshot = _snapshot(
        attention=uninterrupted_attention,
        optimizer=uninterrupted_optimizer,
        scheduler=uninterrupted_scheduler,
        state=uninterrupted_state,
        data_generator=uninterrupted_data_generator,
    )

    seed_everything(config.train.seed)
    attention, optimizer, scheduler, state, data_generator = _build_stack(config)
    resumed_modes = _run_steps(
        attention=attention,
        optimizer=optimizer,
        scheduler=scheduler,
        state=state,
        data_generator=data_generator,
        config=config,
        steps=split_step,
    )

    checkpoint_path = tmp_path / f"checkpoint_step_{split_step}.pt"
    save_checkpoint(
        checkpoint_path,
        model=attention,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        config=config,
        trainer_state=state,
        data_generator_state=data_generator.get_state(),
    )

    # 模拟新进程先构造运行时对象；load 必须覆盖初始化期间消耗的 RNG。
    attention, optimizer, scheduler, _, data_generator = _build_stack(config)
    payload = load_checkpoint(
        checkpoint_path,
        model=attention,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        current_config=config,
        map_location="cpu",
    )
    state = TrainerState.from_dict(payload["trainer_state"])
    data_generator.set_state(payload["data_generator_state"])

    resumed_modes.extend(
        _run_steps(
            attention=attention,
            optimizer=optimizer,
            scheduler=scheduler,
            state=state,
            data_generator=data_generator,
            config=config,
            steps=total_steps - split_step,
        )
    )
    resumed_snapshot = _snapshot(
        attention=attention,
        optimizer=optimizer,
        scheduler=scheduler,
        state=state,
        data_generator=data_generator,
    )

    assert uninterrupted_modes == resumed_modes == [
        "full",
        "full",
        "sparse",
        "sparse",
    ]
    _assert_nested_exact(resumed_snapshot, uninterrupted_snapshot, "training")
