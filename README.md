# Kestrel

A PyTorch training and inference pipeline for a GPT-style language model, built around one rule: **every efficiency technique must be proven equivalent before it is measured.** Gradient checkpointing must produce the same gradients, the KV cache must produce the same tokens, and bf16 and int8 must stay within a measured error bound. Only then do their speed and memory numbers count.

**Measured results are in [RESULTS.md](RESULTS.md).** That file is generated from the run's output files, so no number in it is typed by hand.

## Quick start (CPU)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pytest

pytest                                            # full test suite
python -m kestrel.train --out runs/shakespeare    # downloads TinyShakespeare, trains, evaluates
python -m kestrel.bench --run runs/shakespeare    # speed, memory, KV cache, int8
python -m kestrel.report --run runs/shakespeare   # writes RESULTS.md
```

The default model has about 0.8M parameters (4 layers, 4 heads, d=128, context 128) and trains for 3,000 steps on character-level TinyShakespeare. On a typical laptop CPU that takes roughly 10–20 minutes.

## What's inside

| Component | What it does | How it's verified |
|---|---|---|
| **GPT model** (`model.py`) | Pre-norm decoder, causal attention via `scaled_dot_product_attention`, GPT-2 residual init scaling | Causality test: changing token *t* never changes logits before *t* |
| **KV cache** | Each decoding step processes 1 new token and reuses cached keys and values, instead of re-running the whole prefix | Logits match a full forward pass at every position; generated tokens are **identical** with and without the cache; multi-token chunks use a correctly offset causal mask |
| **Gradient checkpointing** | Frees each block's activations after forward and recomputes them during backward, trading compute for memory | Loss and **every parameter gradient** match the non-checkpointed model |
| **bf16 autocast** | Mixed precision on CPU; the loss stays in float32 | Loss within 2% of fp32, all gradients finite |
| **Int8 quantization** (`quant.py`) | Per-output-channel symmetric weight quantization, implemented directly (not a framework call) | Every weight's error ≤ half a quantization step (the theoretical bound); validation-loss change is measured |
| **Activation memory** (`memory.py`) | Counts the exact bytes autograd saves for backward, using `saved_tensors_hooks` | Exact for a single linear layer (equals input bytes); confirms checkpointing more than halves activation memory |
| **Training loop** (`train.py`) | AdamW (no decay on biases or LayerNorm), warmup + cosine schedule, gradient clipping, a fixed evaluation set | Same seed gives **bitwise-identical** losses and weights; a different seed gives a different run; the model memorizes periodic text |

## Why measure memory this way

On CPU there's no `torch.cuda.max_memory_allocated`, and process RSS is noisy. Autograd's saved-tensor hooks see every tensor kept for the backward pass, which is exactly the activation memory that checkpointing and lower precision are meant to reduce. Storages are deduplicated and parameters excluded. For non-reentrant checkpointing, PyTorch applies only the innermost hooks, so tensors inside a checkpointed block are correctly *not* counted. The block inputs the checkpoint keeps alive are added back explicitly, so the comparison stays fair.

## Reproducibility

Every run writes `config.json` (all hyperparameters plus the PyTorch, Python, and thread-count versions), `metrics.jsonl`, `final.json`, `ckpt.pt`, `tokenizer.json`, and `sample.txt`. Model initialization and data order use separate seeds. Validation uses the last 10% of the text, which is never seen in training, and is evaluated on the same fixed batches every time, so numbers are comparable across steps and runs.

## CI

Every push runs the full test suite on Python 3.11 and 3.12, then does an end-to-end smoke run on real TinyShakespeare (train → bench → report). The smoke run's `RESULTS.md` is uploaded as a build artifact.

## Layout

```
kestrel/
  model.py    GPT, KV cache, gradient checkpointing, generation
  data.py     TinyShakespeare download, char tokenizer, seeded batching
  train.py    reproducible training CLI (fp32 / bf16, checkpointing)
  quant.py    int8 weight-only quantization
  memory.py   exact activation-memory measurement
  bench.py    training-step, generation, and quantization benchmarks
  report.py   RESULTS.md generator
tests/        equivalence, numerics, quantization bounds, reproducibility, end-to-end
```
