from dataclasses import replace
from pathlib import Path

import pytest
import torch

from msa_lab.config import ExperimentConfig, ModelConfig, TrainConfig
from msa_lab.model.tiny_lm import TinyCausalLM
from msa_lab.training.engine import train
from msa_lab.training.generate import greedy_generate


def tiny_config(run_dir: Path, *, steps: int, resume_from: str | None = None) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=32, max_seq_len=16, d_model=32, n_layers=1,
            n_q_heads=4, n_kv_heads=2, d_ff=64, dropout=0.0,
            attention_backend="dense_sdpa",
        ),
        train=TrainConfig(
            seed=7, device="cpu", batch_size=2, seq_len=8, steps=steps,
            learning_rate=2e-3, checkpoint_every=max(steps, 1), log_every=1,
            eval_every=1, eval_batches=1, run_dir=str(run_dir), resume_from=resume_from,
        ),
    )


def test_model_forward_backward_and_aux() -> None:
    cfg = tiny_config(Path("unused"), steps=1)
    model = TinyCausalLM(cfg.model)
    x = torch.randint(0, cfg.model.vocab_size, (2, 8))
    labels = torch.randint(0, cfg.model.vocab_size, (2, 8))
    output = model(x, labels, return_aux=True)
    assert output.logits.shape == (2, 8, cfg.model.vocab_size)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert "layer_0.main_lse" in output.aux
    output.loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_train_writes_evidence_and_checkpoint(tmp_path: Path) -> None:
    cfg = tiny_config(tmp_path / "fresh", steps=2)
    result = train(cfg)
    assert result.final_step == 2
    assert result.checkpoint_path.exists()
    assert (tmp_path / "fresh" / "metrics.jsonl").exists()
    assert (tmp_path / "fresh" / "run_manifest.json").exists()
    assert "eval_accuracy" in (tmp_path / "fresh" / "metrics.jsonl").read_text()
    assert "grad_norm" in (tmp_path / "fresh" / "metrics.jsonl").read_text()
    assert "grad_clip" in (tmp_path / "fresh" / "metrics.jsonl").read_text()


def test_resume_continues_step_and_tokens(tmp_path: Path) -> None:
    first_cfg = tiny_config(tmp_path / "first", steps=2)
    first = train(first_cfg)
    resumed_cfg = tiny_config(tmp_path / "resumed", steps=2, resume_from=str(first.checkpoint_path))
    resumed = train(resumed_cfg)
    assert resumed.final_step == 4
    assert resumed.tokens_seen == 4 * resumed_cfg.train.batch_size * resumed_cfg.train.seq_len


def test_unbounded_training_stops_at_caller_requested_safe_point(tmp_path: Path) -> None:
    cfg = tiny_config(tmp_path / "unbounded", steps=1)
    cfg = ExperimentConfig(model=cfg.model, train=replace(cfg.train, steps=None))
    completed_steps = 0

    def should_stop() -> bool:
        nonlocal completed_steps
        completed_steps += 1
        return completed_steps == 2

    result = train(cfg, should_stop=should_stop)

    assert result.final_step == 2
    assert result.checkpoint_path.exists()


def test_resume_rejects_model_change(tmp_path: Path) -> None:
    first_cfg = tiny_config(tmp_path / "first", steps=1)
    first = train(first_cfg)
    changed = ExperimentConfig(
        model=ModelConfig(
            vocab_size=32, max_seq_len=16, d_model=64, n_layers=1,
            n_q_heads=4, n_kv_heads=2, d_ff=128,
        ),
        train=TrainConfig(
            seed=7, device="cpu", batch_size=2, seq_len=8, steps=1,
            run_dir=str(tmp_path / "changed"), resume_from=str(first.checkpoint_path),
        ),
    )
    with pytest.raises(ValueError, match="model configuration"):
        train(changed)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("learning_rate", 1e-3),
        ("weight_decay", 0.1),
        ("grad_clip", 0.5),
    ],
)
def test_resume_rejects_optimizer_contract_change(
    tmp_path: Path,
    field: str,
    value: float,
) -> None:
    first_cfg = tiny_config(tmp_path / "first", steps=1)
    first = train(first_cfg)
    changed_train = replace(
        first_cfg.train,
        **{
            field: value,
            "steps": 1,
            "run_dir": str(tmp_path / f"changed-{field}"),
            "resume_from": str(first.checkpoint_path),
        },
    )
    changed = ExperimentConfig(model=first_cfg.model, train=changed_train)

    with pytest.raises(ValueError, match=field):
        train(changed)


def test_greedy_generate_respects_max_length() -> None:
    cfg = tiny_config(Path("unused"), steps=1)
    model = TinyCausalLM(cfg.model)
    prompt = torch.tensor([[0, 1, 2]])
    generated = greedy_generate(model, prompt, max_new_tokens=4)
    assert generated.shape == (1, 7)
