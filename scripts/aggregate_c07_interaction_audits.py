#!/usr/bin/env python3
"""Aggregate C07 four-state audit JSON files, if supplied by the reader.

Adapted from frozen private P2 source blob 5d81ea9b0ea48b1f56921f53e110796d6bbe11c6. The original
historical audit input files are not included in this public repository.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def mean(values: list[float]) -> float:
    return statistics.mean(values)


def extract(record: dict[str, Any]) -> dict[str, float]:
    nodes = record["nodes"]
    interaction = record["four_state_non_additivity"]
    support = record["layer1_support_under_upstream_switch"]
    losses = record["losses"]
    token_outcomes = record["token_outcomes"]
    states = token_outcomes["states"]
    transitions = token_outcomes["transitions_from_ff"]
    return {
        "loss_ff": float(losses["ff"]),
        "loss_sf": float(losses["sf"]),
        "loss_fs": float(losses["fs"]),
        "loss_ss": float(losses["ss"]),
        "loss_interaction": float(losses["interaction_ss_minus_sf_minus_fs_plus_ff"]),
        "zero_step_sf_to_ss_loss_jump": float(losses["ss"]) - float(losses["sf"]),
        "zero_step_fs_to_ss_loss_jump": float(losses["ss"]) - float(losses["fs"]),
        "upstream_layer1_input_delta_rms": float(
            nodes["upstream_effect_ff_to_sf"]["block_input"]["delta_rms"]
        ),
        "upstream_centered_logit_delta_rms": float(
            nodes["upstream_effect_ff_to_sf"]["centered_logits"]["delta_rms"]
        ),
        "layer1_local_full_input_block_delta_rms": float(
            nodes["layer1_local_on_full_input_ff_to_fs"]["block_output"]["delta_rms"]
        ),
        "layer1_local_shifted_input_block_delta_rms": float(
            nodes["layer1_local_on_shifted_input_sf_to_ss"]["block_output"]["delta_rms"]
        ),
        "total_centered_logit_delta_rms": float(
            nodes["total_effect_ff_to_ss"]["centered_logits"]["delta_rms"]
        ),
        "centered_logit_interaction_rms": float(interaction["centered_logits"]["interaction_rms"]),
        "centered_logit_interaction_over_ff_rms": float(
            interaction["centered_logits"]["interaction_over_ff_rms"]
        ),
        "support_changed_row_rate": float(support["changed_row_rate"]),
        "support_mean_token_jaccard": float(support["mean_token_jaccard"]),
        "coverage_delta": float(support["coverage_delta"]),
        "token_accuracy_ff": float(states["ff"]["accuracy"]),
        "token_accuracy_sf": float(states["sf"]["accuracy"]),
        "token_accuracy_fs": float(states["fs"]["accuracy"]),
        "token_accuracy_ss": float(states["ss"]["accuracy"]),
        "token_nll_ff": float(states["ff"]["mean_nll"]),
        "token_nll_sf": float(states["sf"]["mean_nll"]),
        "token_nll_fs": float(states["fs"]["mean_nll"]),
        "token_nll_ss": float(states["ss"]["mean_nll"]),
        "ff_to_sf_top1_changed_rate": float(
            transitions["ff_to_sf"]["top1_changed_rate"]
        ),
        "ff_to_fs_top1_changed_rate": float(
            transitions["ff_to_fs"]["top1_changed_rate"]
        ),
        "ff_to_ss_top1_changed_rate": float(
            transitions["ff_to_ss"]["top1_changed_rate"]
        ),
        "ff_to_sf_correct_to_wrong_rate": float(
            transitions["ff_to_sf"]["correct_to_wrong_rate"]
        ),
        "ff_to_sf_wrong_to_correct_rate": float(
            transitions["ff_to_sf"]["wrong_to_correct_rate"]
        ),
        "ff_to_fs_correct_to_wrong_rate": float(
            transitions["ff_to_fs"]["correct_to_wrong_rate"]
        ),
        "ff_to_fs_wrong_to_correct_rate": float(
            transitions["ff_to_fs"]["wrong_to_correct_rate"]
        ),
        "ff_to_ss_correct_to_wrong_rate": float(
            transitions["ff_to_ss"]["correct_to_wrong_rate"]
        ),
        "ff_to_ss_wrong_to_correct_rate": float(
            transitions["ff_to_ss"]["wrong_to_correct_rate"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True,
                        help="directory containing per-checkpoint/batch_*.json")
    parser.add_argument("--precision", default="unspecified",
                        help="precision of supplied replay records; not inferred from JSON")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/c07_population_summary.json")
    args = parser.parse_args()
    if not args.audit_root.is_dir():
        parser.error("--audit-root must be an existing directory")
    summaries: dict[str, Any] = {}
    for checkpoint_dir in sorted(path for path in args.audit_root.iterdir() if path.is_dir()):
        files = sorted(checkpoint_dir.glob("batch_*.json"))
        if not files:
            continue
        records = [json.loads(path.read_text(encoding="utf-8")) for path in files]
        extracted = [extract(record) for record in records]
        keys = extracted[0].keys()
        summaries[checkpoint_dir.name] = {
            "checkpoint": records[0]["checkpoint"],
            "global_step": records[0]["global_step"],
            "batches": len(records),
            "sequences": sum(int(record["probe"]["batch_size"]) for record in records),
            "sparse_active_tokens": sum(
                int(record["token_outcomes"]["states"]["ff"]["active_tokens"])
                for record in records
            ),
            "metrics": {
                key: {
                    "mean": mean([row[key] for row in extracted]),
                    "min": min(row[key] for row in extracted),
                    "max": max(row[key] for row in extracted),
                }
                for key in keys
            },
        }
    if not summaries:
        raise RuntimeError("没有找到 population interaction audit records")
    result = {
        "precision": args.precision,
        "population": "Per-checkpoint held-out batches, see counts; checkpoints are not independent seeds",
        "checkpoints": summaries,
        "boundary": "descriptive fixed population; not a seed-level confidence interval",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
