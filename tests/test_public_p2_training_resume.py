"""Public CPU integration evidence for TinyLM FULL->SPARSE training/resume."""

from dataclasses import replace
import json

import torch

from msa_lab.config import ExperimentConfig, ModelConfig, TrainConfig
from msa_lab.training.engine import train


def _config(run_dir, *, steps, resume_from=None):
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=16,
            max_seq_len=8,
            d_model=16,
            n_layers=2,
            n_q_heads=4,
            n_kv_heads=2,
            d_ff=32,
            msa_layer_id=1,
            msa_index_dim=4,
            msa_block_size=2,
            msa_top_k=2,
        ),
        train=TrainConfig(
            seed=17, device="cpu", dtype="float32",
            batch_size=1, seq_len=4, steps=steps,
            main_mode_policy="scheduled", warmup_steps=1,
            learning_rate=1e-3, routing_kl_weight=0.1,
            log_every=1, checkpoint_every=1, eval_every=0,
            tensorboard=False,
            run_dir=str(run_dir), resume_from=resume_from,
        ),
    )


def _snapshot(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def test_cpu_msa_switch_and_exact_resume_match_uninterrupted_training(tmp_path):
    continuous = train(_config(tmp_path / "continuous", steps=3))
    first = train(_config(tmp_path / "split_first", steps=1))
    resumed = train(
        _config(
            tmp_path / "split_rest",
            steps=2,
            resume_from=str(first.checkpoint_path),
        )
    )

    assert continuous.final_step == resumed.final_step == 3
    assert continuous.tokens_seen == resumed.tokens_seen == 12
    original = _snapshot(continuous.checkpoint_path)
    restored = _snapshot(resumed.checkpoint_path)

    for name, value in original["model"].items():
        assert torch.equal(value, restored["model"][name]), name
    assert original["trainer_state"] == restored["trainer_state"]

    rows = [
        json.loads(line)
        for line in (tmp_path / "continuous" / "metrics.jsonl").read_text().splitlines()
    ]
    assert [row["global_step"] for row in rows] == [1, 2, 3]
    assert [row["extras"]["layer_1_main_mode_is_sparse"] for row in rows] == [
        0.0, 1.0, 1.0
    ]
    assert all("routing_kl" in row["extras"] for row in rows)
    assert all("index_grad_norm" in row["extras"] for row in rows)
