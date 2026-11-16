"""Character-level data: TinyShakespeare download, tokenizer, and seeded batching."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import torch

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def download_tinyshakespeare(path: Path) -> Path:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading TinyShakespeare -> {path}")
        with urllib.request.urlopen(TINY_SHAKESPEARE_URL, timeout=60) as r:
            path.write_bytes(r.read())
    return path


class CharTokenizer:
    def __init__(self, chars: list[str]):
        self.chars = chars
        self.stoi = {c: i for i, c in enumerate(chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.chars[int(i)] for i in ids)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.chars), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


def load_splits(path: Path, val_fraction: float = 0.1):
    """Return (tokenizer, train_ids, val_ids). The split is contiguous (the
    last 10% of the text), so validation text is never seen in training."""
    text = Path(path).read_text(encoding="utf-8")
    tok = CharTokenizer.from_text(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(len(data) * (1 - val_fraction))
    return tok, data[:n], data[n:]


def get_batch(data: torch.Tensor, block_size: int, batch: int, generator: torch.Generator):
    """Random windows; targets are the inputs shifted left by one token."""
    if len(data) <= block_size + 1:
        raise ValueError("data is shorter than one block")
    ix = torch.randint(len(data) - block_size - 1, (batch,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix.tolist()])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix.tolist()])
    return x, y
