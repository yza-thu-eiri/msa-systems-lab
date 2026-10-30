from __future__ import annotations

import argparse
import torch

from msa_lab.config import ExperimentConfig
from msa_lab.model.tiny_lm import TinyCausalLM
from msa_lab.training.generate import greedy_generate
from msa_lab.training.eval_cli import checkpoint_main_modes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="0")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()
    cfg = ExperimentConfig.load(args.config)
    device = torch.device(cfg.train.device)
    model = TinyCausalLM(cfg.model).to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    main_modes = checkpoint_main_modes(cfg, payload)
    prompt = torch.tensor([[int(x) for x in args.prompt.split(",")]], device=device)
    print(
        greedy_generate(
            model,
            prompt,
            max_new_tokens=args.max_new_tokens,
            main_mode=None,
            main_modes=main_modes,
        ).cpu().tolist()
    )


if __name__ == "__main__":
    main()
