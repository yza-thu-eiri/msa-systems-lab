from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class TrainerState:
    global_step: int = 0
    tokens_seen: int = 0
    phase: str = "dense_seed"
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainerState":
        return cls(**data)
