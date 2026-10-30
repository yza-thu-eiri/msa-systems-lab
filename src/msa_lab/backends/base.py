from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch


@dataclass
class AttentionOutput:
    """Common output envelope for public attention reference implementations."""

    output: torch.Tensor
    lse: torch.Tensor | None = None
    aux: dict[str, torch.Tensor] = field(default_factory=dict)


class AttentionBackend(Protocol):
    """Call contract for the dense baseline backend."""

    name: str

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
        return_lse: bool = False,
        return_aux: bool = False,
    ) -> AttentionOutput: ...
