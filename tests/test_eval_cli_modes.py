from __future__ import annotations

from dataclasses import replace

import torch

from msa_lab.config import ExperimentConfig
from msa_lab.training.eval_cli import checkpoint_main_modes
from msa_lab.training.eval_cli import evaluation_autocast


def test_checkpoint_main_modes_preserves_fixed_per_layer_contract() -> None:
    base = ExperimentConfig()
    config = replace(
        base,
        model=replace(base.model, n_layers=2, msa_layer_ids=(0, 1)),
        train=replace(
            base.train,
            main_mode_policy="fixed_per_layer",
            main_modes=("full", "sparse"),
            warmup_steps=0,
        ),
    )

    actual = checkpoint_main_modes(config, {"trainer_state": {"global_step": 20_100}})

    assert actual == ("full", "sparse")


def test_float32_evaluation_uses_no_autocast() -> None:
    context = evaluation_autocast(torch.device("cpu"), "float32")
    with context:
        pass
