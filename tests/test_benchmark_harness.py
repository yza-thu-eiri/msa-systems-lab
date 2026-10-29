from __future__ import annotations

import pytest

from benchmarks.reference.benchmark_forward import summarize_samples


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
