"""Small reproducible CPU forward/backward architecture smoke benchmarks."""

from __future__ import annotations

import time
import tracemalloc

import torch

from .gpt_config import GPTPreset
from .models.gpt import GPTLanguageModel


def benchmark_gpt(preset: GPTPreset, *, seed: int = 1337) -> dict[str, float | int | str]:
    torch.manual_seed(seed)
    start = time.perf_counter()
    model = GPTLanguageModel(preset.model)
    construction_seconds = time.perf_counter() - start
    inputs = torch.randint(preset.model.vocab_size, (preset.batch_size, preset.model.block_size))
    targets = torch.randint(preset.model.vocab_size, (preset.batch_size, preset.model.block_size))
    model.eval()
    tracemalloc.start()
    start = time.perf_counter()
    with torch.no_grad():
        model(inputs)
    forward_seconds = time.perf_counter() - start
    start = time.perf_counter()
    _, loss = model(inputs, targets)
    assert loss is not None
    loss.backward()
    backward_seconds = time.perf_counter() - start
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "parameter_count": model.parameter_count,
        "construction_seconds": construction_seconds,
        "forward_seconds": forward_seconds,
        "forward_backward_seconds": backward_seconds,
        "peak_python_memory_bytes": peak_memory,
        "device": "cpu",
        "batch_size": preset.batch_size,
        "sequence_length": preset.model.block_size,
        "pytorch_version": torch.__version__,
    }
