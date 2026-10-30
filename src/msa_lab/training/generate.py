from __future__ import annotations

import torch

from msa_lab.model.tiny_lm import TinyCausalLM


@torch.no_grad()
def greedy_generate(
    model: TinyCausalLM,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    main_mode: str | None = "sparse",
    main_modes: tuple[str, ...] | None = None,
) -> torch.Tensor:
    if main_modes is not None and main_mode is not None:
        raise ValueError("main_mode and main_modes cannot both be specified")
    model.eval()
    generated = input_ids
    for _ in range(max_new_tokens):
        if generated.shape[1] >= model.cfg.max_seq_len:
            break
        logits = model(generated, main_mode=main_mode, main_modes=main_modes).logits
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
    return generated
