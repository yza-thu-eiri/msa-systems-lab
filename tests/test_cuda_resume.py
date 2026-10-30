from pathlib import Path
from typing import Any

import pytest
import torch

from msa_lab.config import ExperimentConfig, ModelConfig, TrainConfig
from msa_lab.training.engine import train


def cuda_config(
    run_dir: Path,
    *,
    steps: int,
    resume_from: str | None = None,
) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=16,
            max_seq_len=8,
            d_model=16,
            n_layers=1,
            n_q_heads=2,
            n_kv_heads=1,
            d_ff=32,
            dropout=0.0,
            attention_backend="dense_sdpa",
        ),
        train=TrainConfig(
            seed=11,
            device="cuda",
            dtype="float32",
            batch_size=1,
            seq_len=4,
            steps=steps,
            learning_rate=1e-3,
            checkpoint_every=1,
            log_every=1,
            eval_every=0,
            run_dir=str(run_dir),
            resume_from=resume_from,
        ),
    )


def assert_nested_exact(actual: Any, expected: Any, path: str) -> None:
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
            assert_nested_exact(actual[key], expected[key], f"{path}.{key}")
        return

    if isinstance(actual, (list, tuple)):
        assert type(actual) is type(expected), path
        assert len(actual) == len(expected), path
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            assert_nested_exact(actual_item, expected_item, f"{path}[{index}]")
        return

    assert actual == expected, path


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for the GPU checkpoint-resume regression test",
)
def test_cuda_exact_restart_matches_uninterrupted_training(
    tmp_path: Path,
) -> None:
    uninterrupted = train(cuda_config(tmp_path / "uninterrupted", steps=2))

    first = train(cuda_config(tmp_path / "first", steps=1))
    resumed = train(
        cuda_config(
            tmp_path / "resumed",
            steps=1,
            resume_from=str(first.checkpoint_path),
        )
    )

    assert uninterrupted.final_step == resumed.final_step == 2
    assert uninterrupted.tokens_seen == resumed.tokens_seen == 8

    uninterrupted_checkpoint = torch.load(
        uninterrupted.checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    resumed_checkpoint = torch.load(
        resumed.checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    for field in (
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "trainer_state",
        "rng_state",
        "data_generator_state",
    ):
        assert_nested_exact(
            resumed_checkpoint[field],
            uninterrupted_checkpoint[field],
            field,
        )
