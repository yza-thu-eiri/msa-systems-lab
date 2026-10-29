from __future__ import annotations

import math

import pytest
import torch

from benchmarks.reference.benchmark_forward import (
    arithmetic_ledger,
    make_prefix_block_ids,
    summarize_samples,
)


def test_summary_preserves_raw_sample_statistics() -> None:
    summary = summarize_samples([0.1, 0.3, 0.2])
    assert summary == {
        "count": 3,
        "median_ms": 0.2,
        "min_ms": 0.1,
        "max_ms": 0.3,
    }


def test_summary_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        summarize_samples([])


@pytest.mark.parametrize("sample", [float("nan"), -0.1, float("inf")])
def test_summary_rejects_invalid_samples(sample: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        summarize_samples([sample])


def test_historical_arithmetic_ledger_is_reproducible() -> None:
    ledger = arithmetic_ledger(48.25)
    assert ledger["masked_dense_qk_pv_flops"] == 33_554_432
    assert ledger["gathered_qk_pv_flops"] == 12_648_448
    assert math.isclose(
        ledger["arithmetic_reduction_factor"],
        33_554_432 / 12_648_448,
        rel_tol=1e-12,
    )


def test_prefix_metadata_is_valid_on_cpu_without_running_cuda_benchmark() -> None:
    block_ids = make_prefix_block_ids(2, device=torch.device("cpu"))
    assert tuple(block_ids.shape) == (1, 128, 2, 2)
    assert torch.equal(block_ids[0, 0, 0], torch.tensor([0, 1]))
