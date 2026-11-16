import json

import torch

from kestrel.data import CharTokenizer, get_batch, load_splits
from kestrel.train import TrainConfig, load_checkpoint, train

QUICK = dict(steps=15, batch=8, block_size=32, n_layer=2, n_head=2, d_model=32, lr=3e-3, warmup=2,
             eval_every=1000, eval_batches=2)


def test_tokenizer_round_trip_and_split(text_file):
    tok, tr, va = load_splits(text_file)
    text = text_file.read_text(encoding="utf-8")
    assert tok.decode(torch.cat([tr, va])) == text
    assert len(va) == len(text) - int(len(text) * 0.9)
    tok2 = CharTokenizer(tok.chars)
    assert tok2.encode("to be") == tok.encode("to be")


def test_targets_are_inputs_shifted_by_one():
    data = torch.arange(100)
    x, y = get_batch(data, 10, 4, torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (4, 10)
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert torch.equal(y - x, torch.ones_like(x))


def test_same_seed_is_bitwise_reproducible(text_file):
    a = train(TrainConfig(seed=3, **QUICK), text_path=text_file, verbose=False)
    b = train(TrainConfig(seed=3, **QUICK), text_path=text_file, verbose=False)
    assert a["losses"] == b["losses"]
    for (n, pa), (_, pb) in zip(a["model"].named_parameters(), b["model"].named_parameters(), strict=True):
        assert torch.equal(pa, pb), n


def test_different_seed_gives_a_different_run(text_file):
    a = train(TrainConfig(seed=1, **QUICK), text_path=text_file, verbose=False)
    b = train(TrainConfig(seed=2, **QUICK), text_path=text_file, verbose=False)
    assert a["losses"] != b["losses"]


def test_model_can_memorize_a_repeating_text(text_file):
    """Sanity check that the full training loop (optimizer, schedule, clipping)
    actually learns: periodic text is almost perfectly predictable. (Not zero:
    the first few positions of each window have too little context.)"""
    cfg = TrainConfig(steps=300, batch=16, block_size=32, n_layer=2, n_head=2, d_model=64,
                      lr=3e-3, warmup=20, eval_every=1000, eval_batches=4)
    r = train(cfg, text_path=text_file, verbose=False)
    assert r["losses"][0] > 2.5  # starts near log(vocab)
    assert r["final_train_loss"] < 0.4


def test_bf16_with_checkpointing_trains(text_file):
    r = train(TrainConfig(precision="bf16", grad_checkpoint=True, **QUICK), text_path=text_file, verbose=False)
    assert all(torch.isfinite(torch.tensor(r["losses"])))
    assert r["losses"][-1] < r["losses"][0]


def test_artifacts_and_checkpoint_round_trip(text_file, tmp_path):
    out = tmp_path / "run"
    cfg = TrainConfig(data=str(text_file), **QUICK)
    r = train(cfg, out=out, verbose=False)
    for name in ("config.json", "metrics.jsonl", "final.json", "ckpt.pt", "sample.txt", "tokenizer.json"):
        assert (out / name).exists(), name
    meta = json.loads((out / "config.json").read_text())
    assert meta["train"]["seed"] == cfg.seed and "torch" in meta

    model, tok, _ = load_checkpoint(out / "ckpt.pt")
    x = torch.randint(tok.vocab_size, (2, 16), generator=torch.Generator().manual_seed(0))
    r["model"].eval()
    with torch.no_grad():
        assert torch.equal(model(x)[0], r["model"](x)[0])
