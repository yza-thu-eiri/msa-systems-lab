"""Recompute C07 logical-work proxies using a public geometry-only config."""

from pathlib import Path

import pytest

from msa_lab.config import ExperimentConfig
from scripts.analyze_c07_work_crossover import estimate_length, full_pairs, sparse_tokens_per_head


ROOT = Path(__file__).resolve().parents[1]


def test_full_pair_count_counts_causal_triangular_support():
    assert full_pairs(batch=1, sequence_length=4, heads=2) == 20
    assert sparse_tokens_per_head(sequence_length=4, block_size=2, top_k=2) == 8


@pytest.mark.parametrize(
    ("length", "recorded_ratio"),
    [(128, 0.9812), (512, 0.7552), (2048, 0.4304)],
)
def test_public_work_proxy_recomputes_source_rounded_ratios(length, recorded_ratio):
    cfg = ExperimentConfig.load(ROOT / "configs" / "c07_work_proxy.json")
    result = estimate_length(cfg, length)
    assert result["sequence_length"] == length
    assert result["main"]["sparse_pairs_per_layer"] < result["main"]["full_pairs_per_layer"]
    assert result["deployment_forward_proxy"]["ss_ratio_vs_ff"] == pytest.approx(
        recorded_ratio, abs=0.0001
    )


def test_public_work_proxy_rejects_too_few_blocks():
    cfg = ExperimentConfig.load(ROOT / "configs" / "c07_work_proxy.json")
    with pytest.raises(ValueError, match="fewer blocks"):
        estimate_length(cfg, 16)
