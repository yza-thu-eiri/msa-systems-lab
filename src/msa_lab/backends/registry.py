from __future__ import annotations

from .base import AttentionBackend
from .dense_sdpa import DenseSDPABackend


def build_attention_backend(name: str) -> AttentionBackend:
    if name == "dense_sdpa":
        return DenseSDPABackend()
    raise KeyError(
        f"unknown attention backend {name!r}; MSA backends are intentionally "
        "not implemented in seed v0.1"
    )
