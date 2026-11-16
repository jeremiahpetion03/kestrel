"""End to end: train a tiny run, benchmark it, and render the report."""

import json

from kestrel import bench, report
from kestrel.train import TrainConfig, train


def test_bench_and_report_end_to_end(text_file, tmp_path):
    run = tmp_path / "run"
    cfg = TrainConfig(data=str(text_file), steps=20, batch=8, block_size=32, n_layer=2, n_head=2,
                      d_model=32, eval_every=1000, eval_batches=2)
    train(cfg, out=run, verbose=False)

    bench.main(["--run", str(run), "--repeats", "3", "--batch", "8"])
    b = json.loads((run / "bench.json").read_text())

    assert len(b["training"]) == 4
    by = {(r["precision"], r["grad_checkpoint"]): r for r in b["training"]}
    assert by[("fp32", True)]["saved_activation_mb"] < by[("fp32", False)]["saved_activation_mb"]
    assert by[("fp32", False)]["initial_loss"] == by[("fp32", True)]["initial_loss"]
    assert b["generation"]["identical_output"] is True
    assert b["quantization"]["size_reduction"] > 2

    md = report.render(run)
    assert "# Kestrel results" in md and "KV cache" in md and "int8" in md
