from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import torch

from msa_lab.config import ExperimentConfig
from msa_lab.data.tiny_shakespeare import CharTokenizer, load_tiny_shakespeare_text, split_train_valid
from msa_lab.model.tiny_lm import TinyCausalLM
from msa_lab.training.eval import evaluate_increment_task, evaluate_token_stream
from msa_lab.training.mode import main_modes_for_policy


def checkpoint_main_modes(
    cfg: ExperimentConfig,
    payload: dict[str, object],
) -> tuple[str, ...]:
    """按 checkpoint step 与训练合同恢复逐层 Main execution mode。"""
    trainer_state = payload["trainer_state"]
    if not isinstance(trainer_state, dict):
        raise TypeError("checkpoint trainer_state must be a mapping")
    return main_modes_for_policy(
        int(trainer_state["global_step"]),
        cfg.train.warmup_steps,
        cfg.train.main_mode_policy,
        n_layers=cfg.model.n_layers,
        fixed_modes=cfg.train.main_modes,
    )


def evaluation_autocast(device: torch.device, dtype_name: str):
    """与训练引擎保持相同的 evaluation precision contract。"""
    if dtype_name == "float32":
        return nullcontext()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    if device.type == "cpu" and dtype is torch.float16:
        raise RuntimeError("CPU float16 evaluation is not supported")
    return torch.autocast(device_type=device.type, dtype=dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    cfg = ExperimentConfig.load(args.config)
    device = torch.device(cfg.train.device)
    model = TinyCausalLM(cfg.model).to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    main_modes = checkpoint_main_modes(cfg, payload)
    with evaluation_autocast(device, cfg.train.dtype):
        if cfg.train.corpus_path is None:
            generator = torch.Generator(device="cpu").manual_seed(cfg.train.seed + 10_001)
            result = evaluate_increment_task(
                model,
                cfg,
                batches=cfg.train.eval_batches,
                generator=generator,
                device=device,
                main_mode=None,
                main_modes=main_modes,
            )
        else:
            text = load_tiny_shakespeare_text(Path(cfg.train.corpus_path))
            tokenizer = CharTokenizer.from_text(text)
            if len(tokenizer.id_to_char) != cfg.model.vocab_size:
                raise ValueError("model.vocab_size must match the Tiny Shakespeare character vocabulary")
            _, valid_token_ids = split_train_valid(
                tokenizer.encode(text),
                train_fraction=cfg.train.train_fraction,
            )
            result = evaluate_token_stream(
                model,
                valid_token_ids,
                batch_size=cfg.train.batch_size,
                seq_len=cfg.train.seq_len,
                device=device,
                main_mode=None,
                main_modes=main_modes,
            )
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
