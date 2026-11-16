"""Every efficiency technique must leave the math unchanged (or change it by a
bounded, measured amount). These tests prove that before any speed or memory
number is trusted."""

import torch

from conftest import make_model
from kestrel.memory import saved_tensor_bytes
from kestrel.train import autocast_ctx

BIG = dict(vocab_size=40, block_size=32, n_layer=4, n_head=4, d_model=64)


def _batch(seed=0, B=8, T=32, V=40):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(V, (B, T), generator=g), torch.randint(V, (B, T), generator=g)


def test_gradient_checkpointing_gives_the_same_gradients():
    a, b = make_model(**BIG), make_model(**BIG)
    b.load_state_dict(a.state_dict())
    b.grad_checkpointing = True
    a.train()
    b.train()
    x, y = _batch()
    la, lb = a(x, y)[1], b(x, y)[1]
    la.backward()
    lb.backward()
    torch.testing.assert_close(la, lb, atol=1e-7, rtol=0)
    for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        torch.testing.assert_close(pa.grad, pb.grad, atol=1e-6, rtol=1e-5, msg=name)


def test_gradient_checkpointing_cuts_activation_memory():
    x, y = _batch()
    sizes = {}
    for ckpt in (False, True):
        m = make_model(**BIG).train()
        m.grad_checkpointing = ckpt
        boundary = BIG["n_layer"] * x.numel() * BIG["d_model"] * 4 if ckpt else 0
        loss, sizes[ckpt] = saved_tensor_bytes(lambda m=m: m(x, y)[1], exclude=list(m.parameters()),
                                               boundary_bytes=boundary)
        loss.backward()  # graph must still be valid after measuring
    assert sizes[True] < 0.5 * sizes[False], sizes


def test_saved_tensor_accounting_is_exact_for_a_linear_layer():
    lin = torch.nn.Linear(16, 8)
    x = torch.randn(4, 16)
    # Linear saves its input (for the weight gradient); the weight itself is excluded.
    _, nbytes = saved_tensor_bytes(lambda: lin(x).sum(), exclude=list(lin.parameters()))
    assert nbytes == x.numel() * x.element_size()


def test_bf16_autocast_loss_is_close_to_fp32():
    m = make_model(**BIG).train()
    x, y = _batch(1)
    cpu = torch.device("cpu")
    with autocast_ctx(cpu, "fp32"):
        l32 = m(x, y)[1]
    with autocast_ctx(cpu, "bf16"):
        l16 = m(x, y)[1]
    assert l16.dtype == torch.float32  # loss is computed in float32 on purpose
    assert abs(l16.item() - l32.item()) / l32.item() < 2e-2
    l16.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())
