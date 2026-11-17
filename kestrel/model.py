"""GPT-style decoder in PyTorch with a KV cache and optional gradient checkpointing."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

KV = tuple[torch.Tensor, torch.Tensor]


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.d_model % cfg.n_head:
            raise ValueError("d_model must be divisible by n_head")
        self.n_head = cfg.n_head
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x: torch.Tensor, past: KV | None = None) -> tuple[torch.Tensor, KV]:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        present = (k, v)

        p = self.dropout if self.training else 0.0
        Tk = k.size(2)
        if Tk == T:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=True)
        elif T == 1:
            # One new token may attend to every cached key: no mask needed.
            # Building an all-True mask here cost two extra kernel launches per
            # layer per decoded token, which dominates on a GPU at batch size 1.
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=p)
        else:
            # The T queries are the last T of Tk positions, so query i may
            # attend to keys 0 .. (Tk - T + i). True = allowed.
            mask = torch.ones(T, Tk, dtype=torch.bool, device=x.device).tril(diagonal=Tk - T)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), present


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.fc2 = nn.Linear(4 * cfg.d_model, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc(x), approximate="tanh")))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, past: KV | None = None) -> tuple[torch.Tensor, KV]:
        a, present = self.attn(self.ln1(x), past)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, present


def _block_output(block: Block, x: torch.Tensor) -> torch.Tensor:
    return block(x)[0]


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.grad_checkpointing = False

        self.apply(self._init_weights)
        # GPT-2: scale residual-path projections by 1/sqrt(2 * n_layer)
        for name, p in self.named_parameters():
            if name.endswith(("attn.proj.weight", "mlp.fc2.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        cache: list[KV] | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[KV | None]]:
        B, T = idx.shape
        if start_pos + T > self.cfg.block_size:
            raise ValueError(f"positions {start_pos}..{start_pos + T - 1} exceed block_size {self.cfg.block_size}")
        pos = torch.arange(start_pos, start_pos + T, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos))

        use_ckpt = self.grad_checkpointing and self.training and cache is None and torch.is_grad_enabled()
        new_cache: list[KV | None] = []
        for i, block in enumerate(self.blocks):
            if use_ckpt:
                # Drop this block's activations after forward; recompute them in backward.
                x = checkpoint(_block_output, block, x, use_reentrant=False)
                new_cache.append(None)
            else:
                x, present = block(x, cache[i] if cache is not None else None)
                new_cache.append(present)

        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            # float32 loss even under bf16 autocast: log-softmax over the vocab is precision-sensitive
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, new_cache

    @staticmethod
    def _pick(logits: torch.Tensor, temperature: float, generator: torch.Generator | None) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        probs = F.softmax(logits.float() / temperature, dim=-1)
        return torch.multinomial(probs, 1, generator=generator)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        use_cache: bool = True,
        temperature: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return only the newly generated tokens, shape (B, max_new_tokens).

        With use_cache=True, keys and values are computed once per token and
        reused, so each step processes 1 token instead of the whole prefix.
        Cached generation must fit in block_size; uncached generation slides
        a block_size window and can run indefinitely.
        """
        if max_new_tokens <= 0:
            return idx[:, :0]
        if not use_cache:
            out = idx
            for _ in range(max_new_tokens):
                logits, _, _ = self(out[:, -self.cfg.block_size :])
                out = torch.cat([out, self._pick(logits[:, -1], temperature, generator)], dim=1)
            return out[:, idx.size(1) :]

        if idx.size(1) + max_new_tokens > self.cfg.block_size:
            raise ValueError("cached generation must fit in block_size; use use_cache=False to slide")
        logits, _, cache = self(idx)  # prefill: one pass over the prompt
        pos = idx.size(1)
        new = []
        for step in range(max_new_tokens):
            nxt = self._pick(logits[:, -1], temperature, generator)
            new.append(nxt)
            if step < max_new_tokens - 1:
                logits, _, cache = self(nxt, cache=cache, start_pos=pos)
                pos += 1
        return torch.cat(new, dim=1)
