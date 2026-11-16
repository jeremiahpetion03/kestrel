import torch

from conftest import make_model
from kestrel.quant import Int8Linear, model_bytes, quantize_int8


def test_per_channel_error_is_bounded_by_half_a_step():
    torch.manual_seed(0)
    lin = torch.nn.Linear(64, 16)
    q = Int8Linear(lin)
    err = (lin.weight - q.dequantized_weight()).abs()
    bound = q.scale[:, None] / 2
    assert (err <= bound + 1e-7).all()
    assert q.qweight.dtype == torch.int8
    assert q.qweight.abs().max().item() == 127  # each row's largest weight uses the full range


def test_quantized_linear_matches_float_linear():
    torch.manual_seed(1)
    lin = torch.nn.Linear(64, 16)
    x = torch.randn(5, 64)
    q = Int8Linear(lin)
    torch.testing.assert_close(q(x), torch.nn.functional.linear(x, q.dequantized_weight(), lin.bias))
    assert (q(x) - lin(x)).abs().max() < 0.05 * lin(x).abs().max()


def test_quantize_model_replaces_every_linear_and_leaves_original_alone():
    model = make_model(d_model=64, n_head=4).eval()
    q = quantize_int8(model)
    assert not any(isinstance(m, torch.nn.Linear) for m in q.modules())
    assert any(isinstance(m, torch.nn.Linear) for m in model.modules())  # deep copy
    assert model_bytes(q) < 0.5 * model_bytes(model)


def test_quantized_model_loss_is_close():
    model = make_model(d_model=64, n_head=4).eval()
    q = quantize_int8(model)
    g = torch.Generator().manual_seed(0)
    x, y = torch.randint(32, (4, 32), generator=g), torch.randint(32, (4, 32), generator=g)
    with torch.no_grad():
        assert abs(model(x, y)[1].item() - q(x, y)[1].item()) < 1e-2
