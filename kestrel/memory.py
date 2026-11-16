"""Measure activation memory exactly, independent of hardware.

Autograd saves tensors during forward for use in backward; that saved set is
the activation memory a training step must hold. A saved_tensors_hooks pack
hook sees every one of them, so we can sum their storage sizes directly
instead of sampling noisy process RSS.

Notes on what is counted:
* Storages are deduplicated, so a tensor saved by several ops counts once.
* Parameters are excluded (they exist regardless of activations), but casted
  copies of parameters made by autocast ARE counted, because they are real
  extra memory held until backward.
* Under non-reentrant checkpointing, PyTorch installs its own inner hooks, and
  only the innermost hooks apply. Tensors inside a checkpointed block are
  therefore never seen here, which is correct: they are freed and recomputed.
  The checkpoint keeps each block's input alive instead; ``boundary_bytes``
  lets the caller add that back so the comparison stays fair.
"""

from __future__ import annotations

from typing import Callable

import torch


def saved_tensor_bytes(fn: Callable[[], torch.Tensor], exclude: list[torch.Tensor] | None = None,
                       boundary_bytes: int = 0) -> tuple[torch.Tensor, int]:
    excluded = {p.untyped_storage().data_ptr() for p in (exclude or [])}
    storages: dict[int, int] = {}

    def pack(t: torch.Tensor) -> torch.Tensor:
        s = t.untyped_storage()
        ptr = s.data_ptr()
        if ptr not in excluded:
            storages[ptr] = s.nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn()
    return out, sum(storages.values()) + boundary_bytes
