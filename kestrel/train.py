"""Reproducible training run.

    python -m kestrel.train --out runs/shakespeare
    python -m kestrel.train --precision bf16 --grad-checkpoint --out runs/bf16

Writes to --out: config.json, metrics.jsonl, final.json, ckpt.pt, sample.txt
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch

from .data import CharTokenizer, download_tinyshakespeare, get_batch, load_splits
from .model import GPT, ModelConfig

DEFAULT_DATA = "data/tinyshakespeare.txt"


@dataclass
class TrainConfig:
    data: str = DEFAULT_DATA
    steps: int = 3000
    batch: int = 32
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    dropout: float = 0.0
    lr: float = 1e-3
    min_lr_ratio: float = 0.1
    warmup: int = 100
    weight_decay: float = 0.1
    clip: float = 1.0
    precision: str = "fp32"  # fp32 | bf16
    grad_checkpoint: bool = False
    eval_every: int = 250
    eval_batches: int = 20
    seed: int = 0
    device: str = "cpu"  # cpu | cuda | auto
    threads: int = 0  # 0 = PyTorch default


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = min(1.0, (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup))
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def autocast_ctx(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    raise ValueError(f"unknown precision {precision!r}")


def make_optimizer(model: GPT, cfg: TrainConfig) -> torch.optim.AdamW:
    """Weight decay on matrices only, never on biases or LayerNorm parameters."""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95))


@torch.no_grad()
def estimate_loss(model: GPT, data: torch.Tensor, cfg: TrainConfig, device: torch.device) -> float:
    """Mean loss over a FIXED set of batches, so evals are comparable across steps and runs."""
    was_training = model.training
    model.eval()
    g = torch.Generator().manual_seed(12345)
    total = 0.0
    for _ in range(cfg.eval_batches):
        x, y = get_batch(data, cfg.block_size, cfg.batch, g)
        with autocast_ctx(device, cfg.precision):
            _, loss, _ = model(x.to(device), y.to(device))
        total += loss.item()
    model.train(was_training)
    return total / cfg.eval_batches


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train(cfg: TrainConfig, out: Path | None = None, text_path: Path | None = None, verbose: bool = True) -> dict:
    torch.manual_seed(cfg.seed)
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    device = resolve_device(cfg.device)

    path = Path(text_path or cfg.data)
    if not path.exists() and str(path) == DEFAULT_DATA:
        download_tinyshakespeare(path)
    tok, train_data, val_data = load_splits(path)

    mcfg = ModelConfig(tok.vocab_size, cfg.block_size, cfg.n_layer, cfg.n_head, cfg.d_model, cfg.dropout)
    model = GPT(mcfg).to(device)
    model.grad_checkpointing = cfg.grad_checkpoint
    opt = make_optimizer(model, cfg)
    g = torch.Generator().manual_seed(cfg.seed + 1)  # data order, independent of init

    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        meta = {
            "train": asdict(cfg), "model": mcfg.to_dict(), "num_params": model.num_params(),
            "torch": torch.__version__, "python": platform.python_version(),
            "threads": torch.get_num_threads(), "device": str(device),
            "train_tokens": len(train_data), "val_tokens": len(val_data),
        }
        (out / "config.json").write_text(json.dumps(meta, indent=2))
        tok.save(out / "tokenizer.json")
        log = (out / "metrics.jsonl").open("w")

    losses: list[float] = []
    model.train()
    t0 = time.perf_counter()
    for step in range(cfg.steps):
        lr = lr_at(step, cfg)
        for group in opt.param_groups:
            group["lr"] = lr
        x, y = get_batch(train_data, cfg.block_size, cfg.batch, g)
        with autocast_ctx(device, cfg.precision):
            _, loss, _ = model(x.to(device), y.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip).item()
        if not math.isfinite(gnorm):
            raise FloatingPointError(f"non-finite gradient norm at step {step}")
        opt.step()
        losses.append(loss.item())

        last = step == cfg.steps - 1
        if step % cfg.eval_every == 0 or last:
            rec = {"step": step, "train_loss": round(losses[-1], 5), "grad_norm": round(gnorm, 5), "lr": lr}
            if cfg.eval_batches:
                rec["val_loss"] = round(estimate_loss(model, val_data, cfg, device), 5)
            if out is not None:
                log.write(json.dumps(rec) + "\n")
                log.flush()
            if verbose:
                print("  ".join(f"{k} {v}" for k, v in rec.items()))
    seconds = time.perf_counter() - t0
    if out is not None:
        log.close()

    val = estimate_loss(model, val_data, cfg, device) if cfg.eval_batches else float("nan")
    result = {
        "val_loss": round(val, 5),
        "val_perplexity": round(math.exp(val), 4) if math.isfinite(val) else None,
        "final_train_loss": round(losses[-1], 5),
        "train_seconds": round(seconds, 2),
        "train_tokens_per_sec": round(cfg.steps * cfg.batch * cfg.block_size / seconds),
        "num_params": model.num_params(),
    }
    if out is not None:
        (out / "final.json").write_text(json.dumps(result, indent=2))
        torch.save({"model": model.state_dict(), "model_cfg": mcfg.to_dict(), "train_cfg": asdict(cfg),
                    "chars": tok.chars}, out / "ckpt.pt")
        model.eval()
        prompt = torch.tensor([tok.encode("\n")], device=device)
        sample = model.generate(prompt, 600, use_cache=False, temperature=0.8,
                                generator=torch.Generator(device=device).manual_seed(0))
        (out / "sample.txt").write_text(tok.decode(sample[0]), encoding="utf-8")
    if verbose:
        print(json.dumps(result, indent=2))
    result["losses"] = losses
    result["model"] = model
    result["tokenizer"] = tok
    return result


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[GPT, CharTokenizer, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GPT(ModelConfig(**ckpt["model_cfg"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, CharTokenizer(ckpt["chars"]), ckpt["train_cfg"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(TrainConfig):
        flag = "--" + f.name.replace("_", "-")
        if f.type in ("bool", bool):
            parser.add_argument(flag, action="store_true")
        else:
            caster = {"int": int, "float": float, "str": str}.get(f.type, type(f.default))
            parser.add_argument(flag, type=caster, default=f.default)
    parser.add_argument("--out", type=Path, default=Path("runs/shakespeare"))
    args = vars(parser.parse_args(argv))
    out = args.pop("out")
    train(TrainConfig(**args), out=out)


if __name__ == "__main__":
    main()
