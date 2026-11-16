"""Int8 weight-only quantization, implemented directly rather than through a
framework API so every step is visible and testable.

Each nn.Linear weight W (out x in) is stored as int8 with one float scale per
output row (per-channel symmetric quantization):

    scale_r = max(|W_r|) / 127,   q_r = round(W_r / scale_r)  in [-127, 127]

so the worst-case error on any weight is scale_r / 2. Activations stay in
floating point; weights are dequantized on the fly in forward.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class Int8Linear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        w = linear.weight.detach().float()
        scale = w.abs().amax(dim=1).clamp(min=1e-12) / 127.0
        q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
        self.register_buffer("qweight", q)
        self.register_buffer("scale", scale)
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().clone())
        else:
            self.bias = None
        self.in_features, self.out_features = linear.in_features, linear.out_features

    def dequantized_weight(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return self.qweight.to(dtype) * self.scale.to(dtype)[:, None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, self.dequantized_weight(x.dtype), bias)


def quantize_int8(model: nn.Module) -> nn.Module:
    """Return a copy of model with every nn.Linear replaced by Int8Linear."""
    qmodel = copy.deepcopy(model)

    def swap(module: nn.Module) -> None:
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                setattr(module, name, Int8Linear(child))
            else:
                swap(child)

    swap(qmodel)
    return qmodel.eval()


def model_bytes(model: nn.Module) -> int:
    """Bytes of all parameters and buffers: what must be stored and loaded."""
    tensors = list(model.parameters()) + list(model.buffers())
    return sum(t.numel() * t.element_size() for t in tensors)
