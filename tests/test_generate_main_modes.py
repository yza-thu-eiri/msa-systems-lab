"""Regression: generation preserves the same per-layer policy as evaluation."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from msa_lab.config import ExperimentConfig
from msa_lab.training.eval_cli import checkpoint_main_modes
from msa_lab.training.generate import greedy_generate


class SpyLM:
    def __init__(self):
        self.cfg = SimpleNamespace(max_seq_len=8)
        self.calls = []

    def eval(self):
        return self

    def __call__(self, tokens, *, main_mode, main_modes):
        self.calls.append((main_mode, main_modes, tokens.shape[1]))
        logits = torch.zeros(tokens.shape[0], tokens.shape[1], 8)
        logits[..., 3] = 2.0
        return SimpleNamespace(logits=logits)


@pytest.mark.parametrize(
    ("policy", "modes", "warmup", "step", "expected"),
    [
        ("scheduled", None, 2, 1, ("full", "full")),
        ("scheduled", None, 2, 2, ("sparse", "sparse")),
        ("full_only", None, 0, 100, ("full", "full")),
        ("fixed_per_layer", ("full", "sparse"), 0, 100, ("full", "sparse")),
        ("fixed_per_layer", ("sparse", "full"), 0, 100, ("sparse", "full")),
    ],
)
def test_generate_matches_checkpoint_execution_policy(policy, modes, warmup, step, expected):
    base = ExperimentConfig()
    config = replace(
        base,
        model=replace(base.model, n_layers=2, msa_layer_ids=(0, 1)),
        train=replace(
            base.train, main_mode_policy=policy, main_modes=modes,
            warmup_steps=warmup,
        ),
    )
    resolved = checkpoint_main_modes(config, {"trainer_state": {"global_step": step}})
    assert resolved == expected
    model = SpyLM()
    prompt = torch.tensor([[1, 2, 4, 5]])
    result = greedy_generate(
        model, prompt, max_new_tokens=2,
        main_mode=None, main_modes=resolved,
    )
    assert result.tolist() == [[1, 2, 4, 5, 3, 3]]
    assert model.calls == [(None, expected, 4), (None, expected, 5)]


def test_generate_legacy_shared_mode_remains_supported():
    model = SpyLM()
    greedy_generate(model, torch.tensor([[1, 2]]), max_new_tokens=1, main_mode="full")
    assert model.calls == [("full", None, 2)]


def test_generate_rejects_conflicting_modes():
    with pytest.raises(ValueError, match="cannot both"):
        greedy_generate(
            SpyLM(), torch.tensor([[1, 2]]), max_new_tokens=1,
            main_mode="full", main_modes=("sparse", "full"),
        )
