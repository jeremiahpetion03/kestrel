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

The default model has 826K parameters (4 layers, 4 heads, d=128, context 128) and trains for 3,000 steps on character-level TinyShakespeare. On the 2-thread laptop CPU used for [RESULTS.md](RESULTS.md), training took about 43 minutes and the benchmarks about 15. Faster CPUs will finish sooner.

## Headline results

Measured on a 2-thread laptop CPU. Full tables are in [RESULTS.md](RESULTS.md).

| Technique | Effect | Equivalence check |
|---|---|---|
| Training | **1.58** validation loss (perplexity 4.86) on held-out text | Bitwise-reproducible from the seed |
| KV cache | **2.8x** faster generation | Token-for-token identical output |
| Gradient checkpointing | **6.3x** less activation memory (133.6 → 21.1 MB), 14% slower step | Identical loss and gradients |
| Int8 quantization | **3.5x** smaller model (3.15 → 0.90 MB) | Validation loss +0.0004 |
| bf16 autocast | 42% less activation memory, but **16.5x slower** step | Loss within 2% of fp32 |

### Finding: mixed precision is a hardware decision, not a default

bf16 is usually described as a free speedup. On this CPU it made each training step **16.5x slower** (719 ms → 11,879 ms) while saving 42% of activation memory. The CPU has no native bfloat16 arithmetic, so the low-precision math is emulated, and the cost of converting values back and forth far outweighs the memory savings. On GPUs, or on CPUs with native bf16 support (e.g. AVX-512 BF16 or AMX), the same code is expected to speed up. The lesson: measure on the target hardware before assuming an optimization helps.

The same benchmark shows the opposite trade for int8: weight-only quantization makes the model 3.5x smaller with negligible quality loss, but the forward pass is 7% *slower* (57 → 61 ms), because weights are dequantized on every call. Int8 here saves storage and memory bandwidth, not compute. A real speedup would need int8 matrix-multiply kernels.

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
