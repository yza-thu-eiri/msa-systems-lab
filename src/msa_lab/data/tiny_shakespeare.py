from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterator
from urllib.request import urlopen

import torch


TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
TINY_SHAKESPEARE_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


def load_tiny_shakespeare_text(cache_path: Path) -> str:
    """从本地缓存读取语料；缓存不存在时下载固定 Tiny Shakespeare 来源。"""
    if cache_path.is_file():
        raw_bytes = cache_path.read_bytes()
    else:
        with urlopen(TINY_SHAKESPEARE_URL, timeout=30) as response:
            raw_bytes = response.read()

        actual_sha256 = sha256(raw_bytes).hexdigest()
        if actual_sha256 != TINY_SHAKESPEARE_SHA256:
            raise ValueError("下载的 Tiny Shakespeare 内容与固定 SHA-256 不一致")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(raw_bytes)

    actual_sha256 = sha256(raw_bytes).hexdigest()
    if actual_sha256 != TINY_SHAKESPEARE_SHA256:
        raise ValueError("本地 Tiny Shakespeare 缓存与固定 SHA-256 不一致")
    return raw_bytes.decode("utf-8")


@dataclass(frozen=True)
class CharTokenizer:
    """固定字符 vocab 与字符 ID 双向映射。"""

    char_to_id: dict[str, int]
    id_to_char: tuple[str, ...]

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        """由完整固定语料建立 canonical 字符映射。"""

        id_to_char = tuple(sorted(set(text)))

        char_to_id = dict()
        for id, char in enumerate(id_to_char):
            char_to_id[char] = id

        return cls(char_to_id, id_to_char)

    def encode(self, text: str) -> torch.Tensor:
        """将整段文本编码为 CPU 上的一维 long token IDs。"""
        result = list()
        for char in text:
            result.append(self.char_to_id[char])

        return torch.tensor(result, dtype=torch.long)

    def decode(self, token_ids: torch.Tensor) -> str:
        """将一维 token IDs 还原为原始字符文本。"""
        result = list()
        for id in token_ids:
            result.append(self.id_to_char[id])

        return "".join(result)

def sample_next_token_batch(
    token_ids: torch.Tensor,   # CPU [L]
    *,
    batch_size: int,
    seq_len: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:

    L = len(token_ids)

    # [B] 
    starts = torch.randint(
        low=0,
        high=L-seq_len,
        generator=generator,
        size=(batch_size,),
        device=token_ids.device
    )    

    # [T+1]
    offsets = torch.arange(
        start=0,
        end=seq_len+1,
        step=1,
        device=token_ids.device,
    )

    # [B, T+1]
    positions = starts[:, None] + offsets[None, :]

    # B, T+1
    windows = token_ids[positions]

    input_ids = windows[:, :-1]
    labels = windows[:, 1:]

    return input_ids, labels


def split_train_valid(
    token_ids: torch.Tensor,  # CPU [L], long
    *,
    train_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:

    split_point = int(len(token_ids) * train_fraction)

    train_ids = token_ids[:split_point]
    valid_ids = token_ids[split_point:]

    return train_ids, valid_ids


def iter_validation_batches(
    token_ids: torch.Tensor,  # CPU [L_valid], long
    *,
    batch_size: int,
    seq_len: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """顺序产生不重叠的 held-out next-token 验证 batch。"""
    window_count = (len(token_ids) - 1) // seq_len
    offsets = torch.arange(seq_len + 1, device=token_ids.device)

    for first_window in range(0, window_count, batch_size):
        current_batch_size = min(batch_size, window_count - first_window)
        starts = (
            torch.arange(current_batch_size, device=token_ids.device) + first_window
        ) * seq_len
        positions = starts[:, None] + offsets[None, :]
        windows = token_ids[positions]
        yield windows[:, :-1], windows[:, 1:]

