from __future__ import annotations

import torch


def make_increment_batch(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a deterministic, learnable next-token task."""
    starts = torch.randint(
        0,
        vocab_size,
        (batch_size, 1),
        generator=generator,
        device="cpu",
    )
    offsets = torch.arange(seq_len + 1, device="cpu").view(1, -1)
    sequence = (starts + offsets) % vocab_size
    input_ids = sequence[:, :-1].to(device)
    labels = sequence[:, 1:].to(device)
    return input_ids, labels
