from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class AttentionOutput:
    """Common output envelope for public attention reference implementations."""

    output: torch.Tensor
    lse: torch.Tensor | None = None
    aux: dict[str, torch.Tensor] = field(default_factory=dict)
