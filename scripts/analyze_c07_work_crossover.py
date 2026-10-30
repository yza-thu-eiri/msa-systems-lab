#!/usr/bin/env python3
"""Estimate logical work for four C07 execution modes, not GPU latency.

Adapted from the original private P2 analysis source (blob 25cb7352be5001a2451231ba17e4b77630cfb701).
The public default config is a model-geometry-only fixture; no checkpoint or
historical training artifact is required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from msa_lab.config import ExperimentConfig


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/c07_work_proxy.json"
DEFAULT_OUTPUT = ROOT / "artifacts/p2-c07-work-proxy.json"
DEFAULT_LENGTHS = (32, 48, 64, 96, 128, 256, 512, 1024, 2048, 4096)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(ROOT)))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT.relative_to(ROOT)))
    parser.add_argument("--lengths", type=int, nargs="+", default=DEFAULT_LENGTHS)
    return parser.parse_args()


def full_pairs(*, batch: int, sequence_length: int, heads: int) -> int:
    return batch * heads * sequence_length * (sequence_length + 1) // 2


def sparse_tokens_per_head(*, sequence_length: int, block_size: int, top_k: int) -> int:
    total = 0
    for query in range(sequence_length):
        local_block = query // block_size
        local_visible_tokens = query % block_size + 1
        selected_nonlocal_blocks = min(top_k - 1, local_block)
        total += local_visible_tokens + selected_nonlocal_blocks * block_size
    return total


def linear_flops(batch_tokens: int, in_features: int, out_features: int) -> int:
    # 一个 multiply-add 记作 2 FLOPs。
    return 2 * batch_tokens * in_features * out_features


def estimate_length(config: ExperimentConfig, sequence_length: int) -> dict[str, object]:
    model = config.model
    batch = 1
    tokens = batch * sequence_length
    head_dim = model.d_model // model.n_q_heads
    group_size = model.n_q_heads // model.n_kv_heads
    layers = len(model.resolved_msa_layer_ids())
    if layers != model.n_layers:
        raise ValueError("C07 work model expects every Transformer layer to be MSA")
    num_blocks = (sequence_length + model.msa_block_size - 1) // model.msa_block_size
    if model.msa_top_k > num_blocks:
        raise ValueError("sequence length produces fewer blocks than msa_top_k")

    full_main_pairs = full_pairs(
        batch=batch,
        sequence_length=sequence_length,
        heads=model.n_q_heads,
    )
    sparse_main_pairs = (
        batch
        * model.n_q_heads
        * sparse_tokens_per_head(
            sequence_length=sequence_length,
            block_size=model.msa_block_size,
            top_k=model.msa_top_k,
        )
    )
    full_index_pairs = full_pairs(
        batch=batch,
        sequence_length=sequence_length,
        heads=model.n_kv_heads,
    )
    sparse_index_pairs = sparse_main_pairs // group_size

    q_out = model.n_q_heads * head_dim
    kv_out = model.n_kv_heads * head_dim
    attention_projection_flops = sum(
        linear_flops(tokens, model.d_model, width)
        for width in (q_out, kv_out, kv_out)
    ) + linear_flops(tokens, q_out, model.d_model)
    ffn_flops = (
        linear_flops(tokens, model.d_model, model.d_ff)
        + linear_flops(tokens, model.d_ff, model.d_model)
    )
    shared_block_flops = attention_projection_flops + ffn_flops
    lm_head_flops = linear_flops(tokens, model.d_model, model.vocab_size)

    full_main_flops = 4 * head_dim * full_main_pairs
    sparse_main_flops = 4 * head_dim * sparse_main_pairs

    index_projection_flops = (
        linear_flops(tokens, model.d_model, model.n_kv_heads * model.msa_index_dim)
        + linear_flops(tokens, model.d_model, model.msa_index_dim)
    )
    router_logical_score_flops = 2 * model.msa_index_dim * full_index_pairs
    router_reference_score_flops = (
        2
        * model.msa_index_dim
        * batch
        * model.n_kv_heads
        * sequence_length**2
    )
    router_logical_flops = index_projection_flops + router_logical_score_flops
    router_reference_flops = index_projection_flops + router_reference_score_flops

    # Routing-KL 复算 Main teacher QK 和 Index student logits，不含 sort/gather/softmax。
    full_routing_kl_dot_flops = (
        2 * head_dim * full_main_pairs
        + 2 * model.msa_index_dim * full_index_pairs
    )
    sparse_routing_kl_dot_flops = (
        2 * head_dim * sparse_main_pairs
        + 2 * model.msa_index_dim * sparse_index_pairs
    )

    common_model_flops = layers * shared_block_flops + lm_head_flops

    def deployment_total(sparse_layers: int, *, logical_router: bool) -> int:
        full_layers = layers - sparse_layers
        router = router_logical_flops if logical_router else router_reference_flops
        return (
            common_model_flops
            + full_layers * full_main_flops
            + sparse_layers * (sparse_main_flops + router)
        )

    def experiment_total(sparse_layers: int) -> int:
        # 当前训练中两层始终计算 Index；KL support 随该层 Main mode 改变。
        full_layers = layers - sparse_layers
        return (
            common_model_flops
            + full_layers * (full_main_flops + router_reference_flops + full_routing_kl_dot_flops)
            + sparse_layers * (
                sparse_main_flops + router_reference_flops + sparse_routing_kl_dot_flops
            )
        )

    deployment_ff = deployment_total(0, logical_router=True)
    deployment = {
        "ff": deployment_ff,
        "sf_or_fs": deployment_total(1, logical_router=True),
        "ss": deployment_total(2, logical_router=True),
        "ss_with_current_reference_router": deployment_total(2, logical_router=False),
    }
    experiment_ff = experiment_total(0)
    experiment = {
        "ff": experiment_ff,
        "sf_or_fs": experiment_total(1),
        "ss": experiment_total(2),
    }
    return {
        "sequence_length": sequence_length,
        "main": {
            "full_pairs_per_layer": full_main_pairs,
            "sparse_pairs_per_layer": sparse_main_pairs,
            "sparse_density": sparse_main_pairs / full_main_pairs,
            "full_flops_per_layer": full_main_flops,
            "sparse_flops_per_layer": sparse_main_flops,
        },
        "mode_invariant": {
            "attention_projections_per_layer": attention_projection_flops,
            "ffn_per_layer": ffn_flops,
            "lm_head": lm_head_flops,
        },
        "router_per_sparse_layer": {
            "index_projections": index_projection_flops,
            "logical_causal_score": router_logical_score_flops,
            "current_reference_score": router_reference_score_flops,
            "logical_total": router_logical_flops,
            "current_reference_total": router_reference_flops,
        },
        "routing_kl_dot_per_layer": {
            "full_support": full_routing_kl_dot_flops,
            "sparse_support": sparse_routing_kl_dot_flops,
        },
        "deployment_forward_proxy": {
            **deployment,
            "sf_or_fs_ratio_vs_ff": deployment["sf_or_fs"] / deployment_ff,
            "ss_ratio_vs_ff": deployment["ss"] / deployment_ff,
            "ss_reference_router_ratio_vs_ff": (
                deployment["ss_with_current_reference_router"] / deployment_ff
            ),
        },
        "current_experiment_forward_proxy": {
            **experiment,
            "sf_or_fs_ratio_vs_ff": experiment["sf_or_fs"] / experiment_ff,
            "ss_ratio_vs_ff": experiment["ss"] / experiment_ff,
        },
    }


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.load(ROOT / args.config)
    lengths = sorted(set(args.lengths))
    records = [estimate_length(config, length) for length in lengths]
    crossover = next(
        (
            record["sequence_length"]
            for record in records
            if record["deployment_forward_proxy"]["ss_ratio_vs_ff"] < 1.0
        ),
        None,
    )
    minimum_valid_length = (config.model.msa_top_k - 1) * config.model.msa_block_size + 1
    exact_crossover = next(
        (
            length
            for length in range(minimum_valid_length, max(lengths) + 1)
            if estimate_length(config, length)["deployment_forward_proxy"]["ss_ratio_vs_ff"]
            < 1.0
        ),
        None,
    )
    result = {
        "model": config.model.__dict__,
        "flop_convention": "multiply-add = 2 FLOPs",
        "scope": {
            "included": [
                "Q/K/V/O projections",
                "FFN linear layers",
                "LM head",
                "Main QK and AV dots",
                "Index projections and Index score dots",
                "Routing-KL teacher/student dot products in experiment proxy",
            ],
            "excluded": [
                "LayerNorm, GELU, softmax, BlockMax, top-k, sort/gather",
                "memory traffic, launch/scheduling overhead, backward",
            ],
        },
        "deployment_contract": (
            "FF skips Router; each SPARSE layer pays a logical causal Router and no routing KL."
        ),
        "experiment_contract": (
            "Every layer pays the current materialized Router; routing KL dot work uses that layer's Main support."
        ),
        "first_sampled_ss_deployment_crossover": crossover,
        "first_integer_ss_deployment_crossover": exact_crossover,
        "length_sweep_boundary": (
            "Lengths above model.max_seq_len are hypothetical geometry extrapolations, not runnable evidence."
        ),
        "records": records,
    }
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
