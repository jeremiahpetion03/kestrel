"""Benchmarks for a trained run: training-step speed and activation memory
across precision x gradient checkpointing, KV-cache generation speed, and the
size / quality / latency trade-off of int8 quantization.

    python -m kestrel.bench --run runs/shakespeare                 # CPU
    python -m kestrel.bench --run runs/gpu --device cuda           # NVIDIA GPU

Writes <run>/bench.json. Every comparison also asserts the equivalence it
depends on (identical tokens, matching loss), so a fast-but-wrong variant
cannot produce a number.

On CUDA, kernels run asynchronously: a timer read without synchronizing
measures only how long it took to *queue* the work. Every timing here
synchronizes before reading the clock.
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
from .train import autocast_ctx, hardware_info, load_checkpoint, resolve_device

MB = 2**20


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _median_ms(fn, repeats: int, device: torch.device, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    _sync(device)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1e3


def precisions_for(device: torch.device) -> list[str]:
    """fp32 and bf16 everywhere; fp16 too on CUDA, where tensor cores make it the classic fast path."""
    out = ["fp32", "bf16"]
    if device.type == "cuda":
        out.append("fp16")
    return out


def bench_training(mcfg: ModelConfig, batch: int, precision: str, grad_ckpt: bool, repeats: int,
                   device: torch.device) -> dict:
    torch.manual_seed(0)
    model = GPT(mcfg).to(device)
    model.grad_checkpointing = grad_ckpt
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # fp16 needs loss scaling or small gradients underflow to zero; bf16 and fp32 do not.
    scaler = torch.amp.GradScaler(device.type, enabled=precision == "fp16")
    g = torch.Generator().manual_seed(0)
    x = torch.randint(mcfg.vocab_size, (batch, mcfg.block_size), generator=g).to(device)
    y = torch.randint(mcfg.vocab_size, (batch, mcfg.block_size), generator=g).to(device)

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
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    peak_mb = None
    if device.type == "cuda":
        step()
        _sync(device)
        torch.cuda.reset_peak_memory_stats(device)
        step()
        _sync(device)
        peak_mb = round(torch.cuda.max_memory_allocated(device) / MB, 1)

    ms = _median_ms(step, repeats, device)
    return {
        "precision": precision,
        "grad_checkpoint": grad_ckpt,
        "median_step_ms": round(ms, 2),
        "tokens_per_sec": round(batch * mcfg.block_size / (ms / 1e3)),
        "saved_activation_mb": round(saved / MB, 3),
        "peak_gpu_mb": peak_mb,
        "initial_loss": round(first_loss, 5),
    }


def bench_generation(model: GPT, prompt_len: int, repeats: int, device: torch.device, batch: int = 1) -> dict:
    new = model.cfg.block_size - prompt_len
    g = torch.Generator().manual_seed(0)
    prompt = torch.randint(model.cfg.vocab_size, (batch, prompt_len), generator=g).to(device)
    out = {}
    tokens = {}
    for use_cache in (False, True):
        tokens[use_cache] = model.generate(prompt, new, use_cache=use_cache)
        ms = _median_ms(lambda c=use_cache: model.generate(prompt, new, use_cache=c), repeats, device, warmup=1)
        out["cached" if use_cache else "uncached"] = {
            "median_ms": round(ms, 2), "tokens_per_sec": round(batch * new / (ms / 1e3))}
    if not torch.equal(tokens[False], tokens[True]):
        raise AssertionError("KV-cache generation diverged from full recomputation")
    out["new_tokens"] = new
    out["identical_output"] = True
    out["speedup"] = round(out["uncached"]["median_ms"] / out["cached"]["median_ms"], 2)
    return out


def generation_sweep(model: GPT, batches: list[int], repeats: int, device: torch.device) -> list[dict]:
    """KV-cache speedup vs batch size. At batch 1 on a GPU every decode step is
    a handful of tiny kernels, so launch overhead hides the FLOPs the cache
    saves; larger batches make each step compute-bound and expose them."""
    rows = []
    for b in batches:
        r = bench_generation(model, prompt_len=16, repeats=max(3, repeats // 4), device=device, batch=b)
        row = {"batch": b, "uncached_tokens_per_sec": r["uncached"]["tokens_per_sec"],
               "cached_tokens_per_sec": r["cached"]["tokens_per_sec"], "speedup": r["speedup"],
               "identical_output": r["identical_output"]}
        rows.append(row)
        print(json.dumps(row))
    return rows


@torch.no_grad()
def eval_loss(model: GPT, data: torch.Tensor, block: int, batch: int, n_batches: int,
              device: torch.device) -> float:
    g = torch.Generator().manual_seed(12345)
    total = 0.0
    for _ in range(n_batches):
        x, y = get_batch(data, block, batch, g)
        total += model(x.to(device), y.to(device))[1].item()
    return total / n_batches


def bench_quantization(model: GPT, val: torch.Tensor, repeats: int, device: torch.device,
                       eval_batches: int = 20) -> dict:
    qmodel = quantize_int8(model)
    block = model.cfg.block_size
    loss_fp32 = eval_loss(model, val, block, 32, eval_batches, device)
    loss_int8 = eval_loss(qmodel, val, block, 32, eval_batches, device)

    x, _ = get_batch(val, block, 8, torch.Generator().manual_seed(0))
    x = x.to(device)
    with torch.no_grad():
        max_logit_diff = (model(x)[0] - qmodel(x)[0]).abs().max().item()
        ms_fp32 = _median_ms(lambda: model(x), repeats, device)
        ms_int8 = _median_ms(lambda: qmodel(x), repeats, device)
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
    p.add_argument("--device", default="cpu", help="cpu | cuda | auto")
    p.add_argument("--only", choices=["all", "generation"], default="all",
                   help="'generation' re-runs only the KV-cache sweep and merges it into bench.json")
    p.add_argument("--gen-batch", type=int, nargs="+", default=[1],
                   help="batch sizes for the KV-cache generation sweep, e.g. --gen-batch 1 8 32")
    args = p.parse_args(argv)
    device = resolve_device(args.device)

    model, tok, train_cfg = load_checkpoint(args.run / "ckpt.pt", device=str(device))
    tok_data, _, val = load_splits(Path(train_cfg["data"]))
    if tok_data.chars != tok.chars:
        raise ValueError("dataset vocabulary does not match the checkpoint's tokenizer")

    if args.only == "generation":
        path = args.run / "bench.json"
        results = json.loads(path.read_text()) if path.exists() else {}
        results["generation_sweep"] = generation_sweep(model, args.gen_batch, args.repeats, device)
        path.write_text(json.dumps(results, indent=2))
        print(f"updated {path}")
        return

    results = {"torch": torch.__version__, "threads": torch.get_num_threads(),
               **hardware_info(device), "training": []}
    for precision in precisions_for(device):
        for ckpt in (False, True):
            r = bench_training(model.cfg, args.batch, precision, ckpt, args.repeats, device)
            results["training"].append(r)
            print(json.dumps(r))
    results["generation"] = bench_generation(model, prompt_len=16, repeats=max(3, args.repeats // 4),
                                             device=device)
    print(json.dumps(results["generation"]))
    if args.gen_batch != [1]:
        results["generation_sweep"] = generation_sweep(model, args.gen_batch, args.repeats, device)
    results["quantization"] = bench_quantization(model, val, args.repeats, device)
    print(json.dumps(results["quantization"]))
    (args.run / "bench.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {args.run / 'bench.json'}")


if __name__ == "__main__":
    main()
