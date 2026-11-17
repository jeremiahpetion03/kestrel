"""Device handling. CI has no GPU, so these check the CPU-side logic of the
GPU code paths; the CUDA test runs only where a GPU is present."""

import json

import pytest
import torch

from kestrel.bench import bench_training, precisions_for
from kestrel.report import _hardware
from kestrel.train import autocast_ctx, hardware_info, resolve_device

from conftest import tiny_cfg


def test_precisions_by_device():
    assert precisions_for(torch.device("cpu")) == ["fp32", "bf16"]
    assert precisions_for(torch.device("cuda")) == ["fp32", "bf16", "fp16"]


def test_unknown_precision_is_rejected():
    with pytest.raises(ValueError):
        autocast_ctx(torch.device("cpu"), "fp8")


def test_cuda_request_without_cuda_explains_the_fix():
    if torch.cuda.is_available():
        pytest.skip("GPU present")
    with pytest.raises(RuntimeError, match="CUDA build"):
        resolve_device("cuda")
    assert resolve_device("auto").type == "cpu"


def test_hardware_info_is_json_serializable():
    info = hardware_info(torch.device("cpu"))
    assert info["device"] == "cpu"
    json.dumps(info)


def test_report_names_the_gpu_when_present():
    bench = {"device": "cuda", "gpu": "NVIDIA GeForce RTX 3060", "gpu_memory_gb": 12.0,
             "compute_capability": "8.6", "cuda": "12.4"}
    assert _hardware(bench, {"threads": 8}) == \
        "NVIDIA GeForce RTX 3060 (12.0 GB, compute capability 8.6, CUDA 12.4)"
    assert _hardware({"device": "cpu"}, {"threads": 8}) == "CPU, 8 threads"
    assert _hardware({}, {"threads": 2}) == "CPU, 2 threads"  # bench.json from older runs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an NVIDIA GPU")
def test_cuda_training_bench_matches_cpu_loss():
    cfg = tiny_cfg(vocab_size=40)
    cpu = bench_training(cfg, 4, "fp32", False, 2, torch.device("cpu"))
    gpu = bench_training(cfg, 4, "fp32", False, 2, torch.device("cuda"))
    assert abs(cpu["initial_loss"] - gpu["initial_loss"]) < 1e-3
    assert gpu["peak_gpu_mb"] is not None
