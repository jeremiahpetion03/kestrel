import pytest
import torch

from conftest import make_model


def test_kv_cache_logits_match_full_forward():
    """Prefill 5 tokens, then feed one token at a time through the cache:
    every position's logits must match a single full forward pass."""
    model = make_model().eval()
    idx = torch.randint(32, (2, 12), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        full = model(idx)[0]
        logits, _, cache = model(idx[:, :5])
        torch.testing.assert_close(logits, full[:, :5], atol=1e-5, rtol=1e-5)
        for t in range(5, 12):
            step, _, cache = model(idx[:, t : t + 1], cache=cache, start_pos=t)
            torch.testing.assert_close(step[:, 0], full[:, t], atol=1e-5, rtol=1e-5)


def test_kv_cache_multi_token_chunk_uses_correct_mask():
    """Feeding several new tokens at once on top of a cache needs an offset
    causal mask (not is_causal). Check it against the full forward."""
    model = make_model().eval()
    idx = torch.randint(32, (1, 10), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        full = model(idx)[0]
        _, _, cache = model(idx[:, :4])
        chunk, _, _ = model(idx[:, 4:10], cache=cache, start_pos=4)
    torch.testing.assert_close(chunk, full[:, 4:10], atol=1e-5, rtol=1e-5)


def test_cached_and_uncached_generation_are_identical():
    model = make_model().eval()
    prompt = torch.randint(32, (3, 6), generator=torch.Generator().manual_seed(2))
    a = model.generate(prompt, 20, use_cache=True)
    b = model.generate(prompt, 20, use_cache=False)
    assert a.shape == (3, 20)
    assert torch.equal(a, b)


def test_causal_mask_blocks_the_future():
    model = make_model().eval()
    idx = torch.randint(32, (1, 16), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        base = model(idx)[0]
        for t in range(1, 16):
            changed = idx.clone()
            changed[0, t] = (changed[0, t] + 1) % 32
            out = model(changed)[0]
            torch.testing.assert_close(out[0, :t], base[0, :t], atol=1e-6, rtol=0)
            assert not torch.allclose(out[0, t], base[0, t])


def test_rejects_positions_past_block_size():
    model = make_model().eval()
    with pytest.raises(ValueError):
        model(torch.zeros(1, 33, dtype=torch.long))
    with pytest.raises(ValueError):
        model.generate(torch.zeros(1, 30, dtype=torch.long), 5, use_cache=True)
    # uncached generation slides the window, so it can go past block_size
    assert model.generate(torch.zeros(1, 30, dtype=torch.long), 5, use_cache=False).shape == (1, 5)


def test_sampling_is_reproducible_with_a_seeded_generator():
    model = make_model().eval()
    prompt = torch.zeros(2, 1, dtype=torch.long)
    a = model.generate(prompt, 15, temperature=1.0, generator=torch.Generator().manual_seed(7))
    b = model.generate(prompt, 15, temperature=1.0, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)
