"""Synthetic fixture for the published C07 population aggregation contract."""

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "aggregate_c07_interaction_audits.py"


def _record():
    return {
        "checkpoint": "synthetic_checkpoint_for_contract_test",
        "global_step": 20100,
        "probe": {"batch_size": 2},
        "nodes": {
            "upstream_effect_ff_to_sf": {
                "block_input": {"delta_rms": 0.1},
                "centered_logits": {"delta_rms": 0.2},
            },
            "layer1_local_on_full_input_ff_to_fs": {
                "block_output": {"delta_rms": 0.3},
            },
            "layer1_local_on_shifted_input_sf_to_ss": {
                "block_output": {"delta_rms": 0.4},
            },
            "total_effect_ff_to_ss": {
                "centered_logits": {"delta_rms": 0.5},
            },
        },
        "four_state_non_additivity": {
            "centered_logits": {
                "interaction_rms": 0.006,
                "interaction_over_ff_rms": 0.013,
            }
        },
        "layer1_support_under_upstream_switch": {
            "changed_row_rate": 0.004,
            "mean_token_jaccard": 0.998,
            "coverage_delta": 0.0003,
        },
        "losses": {
            "ff": 1.0, "sf": 1.1, "fs": 1.2, "ss": 1.32,
            "interaction_ss_minus_sf_minus_fs_plus_ff": 0.02,
        },
        "token_outcomes": {
            "states": {
                key: {"accuracy": 0.5, "mean_nll": 1.1, "active_tokens": 3}
                for key in ("ff", "sf", "fs", "ss")
            },
            "transitions_from_ff": {
                name: {
                    "top1_changed_rate": 0.1,
                    "correct_to_wrong_rate": 0.06,
                    "wrong_to_correct_rate": 0.04,
                }
                for name in ("ff_to_sf", "ff_to_fs", "ff_to_ss")
            },
        },
    }


def test_c07_cli_aggregates_reader_supplied_records(tmp_path):
    directory = tmp_path / "synthetic_population" / "checkpoint_0"
    directory.mkdir(parents=True)
    for i in range(2):
        (directory / f"batch_{i}.json").write_text(
            json.dumps(_record()), encoding="utf-8"
        )
    output = tmp_path / "summary.json"
    run = subprocess.run(
        [
            sys.executable, str(SCRIPT), "--audit-root",
            str(directory.parent), "--output", str(output),
        ],
        capture_output=True, text=True, check=True,
    )
    summary = json.loads(run.stdout)
    assert summary == json.loads(output.read_text(encoding="utf-8"))
    assert summary["precision"] == "unspecified"
    report = summary["checkpoints"]["checkpoint_0"]
    assert report["batches"] == 2
    assert report["sequences"] == 4
    assert report["sparse_active_tokens"] == 6
    assert report["metrics"]["support_changed_row_rate"]["mean"] == 0.004
    assert report["metrics"]["loss_interaction"]["mean"] == 0.02
    assert report["metrics"]["ff_to_ss_top1_changed_rate"]["max"] == 0.1


def test_c07_cli_rejects_missing_input_directory(tmp_path):
    run = subprocess.run(
        [
            sys.executable, str(SCRIPT), "--audit-root",
            str(tmp_path / "does-not-exist"),
        ],
        capture_output=True, text=True,
    )
    assert run.returncode != 0
    assert "existing directory" in run.stderr
