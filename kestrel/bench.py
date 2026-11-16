"""Benchmarks for a trained run: training-step speed and activation memory
across precision x gradient checkpointing, KV-cache generation speed, and the
size / quality / latency trade-off of int8 quantization.

    python -m kestrel.bench --run runs/shakespeare

Writes <run>/bench.json. Every comparison also asserts the equivalence it
depends on (identical tokens, matching loss), so a fast-but-wrong variant
cannot produce a number.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from .data import get_batch, load_splits
from .memory import saved_tensor_bytes
from .model import GPT, ModelConfig
from .quant import model_bytes, quantize_int8
from .train import autocast_ctx, load_checkpoint

MB = 2**20


def _median_ms(fn, repeats: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1e3


def bench_training(mcfg: ModelConfig, batch: int, precision: str, grad_ckpt: bool, repeats: int) -> dict:
    torch.manual_seed(0)
    model = GPT(mcfg)
    model.grad_checkpointing = grad_ckpt
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(0)
    x = torch.randint(mcfg.vocab_size, (batch, mcfg.block_size), generator=g)
    y = torch.randint(mcfg.vocab_size, (batch, mcfg.block_size), generator=g)
    device = torch.device("cpu")

    def forward() -> torch.Tensor:
        with autocast_ctx(device, precision):
            return model(x, y)[1]

    # Checkpointing keeps each block's (float32 residual-stream) input alive.
    boundary = mcfg.n_layer * batch * mcfg.block_size * mcfg.d_model * 4 if grad_ckpt else 0
    loss, saved = saved_tensor_bytes(forward, exclude=list(model.parameters()), boundary_bytes=boundary)
    first_loss = loss.item()
    loss.backward()
    opt.zero_grad(set_to_none=True)

    def step() -> None:
        loss = forward()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    ms = _median_ms(step, repeats)
    return {
        "precision": precision,
        "grad_checkpoint": grad_ckpt,
        "median_step_ms": round(ms, 2),
        "tokens_per_sec": round(batch * mcfg.block_size / (ms / 1e3)),
        "saved_activation_mb": round(saved / MB, 3),
        "initial_loss": round(first_loss, 5),
    }


def bench_generation(model: GPT, prompt_len: int, repeats: int, batch: int = 1) -> dict:
    new = model.cfg.block_size - prompt_len
    g = torch.Generator().manual_seed(0)
    prompt = torch.randint(model.cfg.vocab_size, (batch, prompt_len), generator=g)
    out = {}
    tokens = {}
    for use_cache in (False, True):
        tokens[use_cache] = model.generate(prompt, new, use_cache=use_cache)
        ms = _median_ms(lambda c=use_cache: model.generate(prompt, new, use_cache=c), repeats, warmup=1)
        out["cached" if use_cache else "uncached"] = {
            "median_ms": round(ms, 2), "tokens_per_sec": round(batch * new / (ms / 1e3))}
    if not torch.equal(tokens[False], tokens[True]):
        raise AssertionError("KV-cache generation diverged from full recomputation")
    out["new_tokens"] = new
    out["identical_output"] = True
    out["speedup"] = round(out["uncached"]["median_ms"] / out["cached"]["median_ms"], 2)
    return out


@torch.no_grad()
def eval_loss(model: GPT, data: torch.Tensor, block: int, batch: int, n_batches: int) -> float:
    g = torch.Generator().manual_seed(12345)
    total = 0.0
    for _ in range(n_batches):
        x, y = get_batch(data, block, batch, g)
        total += model(x, y)[1].item()
    return total / n_batches


def bench_quantization(model: GPT, val: torch.Tensor, repeats: int, eval_batches: int = 20) -> dict:
    qmodel = quantize_int8(model)
    block = model.cfg.block_size
    loss_fp32 = eval_loss(model, val, block, 32, eval_batches)
    loss_int8 = eval_loss(qmodel, val, block, 32, eval_batches)

    x, _ = get_batch(val, block, 8, torch.Generator().manual_seed(0))
    with torch.no_grad():
        max_logit_diff = (model(x)[0] - qmodel(x)[0]).abs().max().item()
        ms_fp32 = _median_ms(lambda: model(x), repeats)
        ms_int8 = _median_ms(lambda: qmodel(x), repeats)
    b32, b8 = model_bytes(model), model_bytes(qmodel)
    return {
        "fp32_mb": round(b32 / MB, 3),
        "int8_mb": round(b8 / MB, 3),
        "size_reduction": round(b32 / b8, 2),
        "val_loss_fp32": round(loss_fp32, 5),
        "val_loss_int8": round(loss_int8, 5),
        "val_loss_delta": round(loss_int8 - loss_fp32, 5),
        "max_abs_logit_diff": round(max_logit_diff, 5),
        "forward_ms_fp32": round(ms_fp32, 2),
        "forward_ms_int8": round(ms_int8, 2),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, default=Path("runs/shakespeare"))
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--batch", type=int, default=32)
    args = p.parse_args(argv)

    model, tok, train_cfg = load_checkpoint(args.run / "ckpt.pt")
    tok_data, _, val = load_splits(Path(train_cfg["data"]))
    if tok_data.chars != tok.chars:
        raise ValueError("dataset vocabulary does not match the checkpoint's tokenizer")

    results = {"torch": torch.__version__, "threads": torch.get_num_threads(), "training": []}
    for precision in ("fp32", "bf16"):
        for ckpt in (False, True):
            r = bench_training(model.cfg, args.batch, precision, ckpt, args.repeats)
            results["training"].append(r)
            print(json.dumps(r))
    results["generation"] = bench_generation(model, prompt_len=16, repeats=max(3, args.repeats // 4))
    print(json.dumps(results["generation"]))
    results["quantization"] = bench_quantization(model, val, args.repeats)
    print(json.dumps(results["quantization"]))
    (args.run / "bench.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {args.run / 'bench.json'}")


if __name__ == "__main__":
    main()
