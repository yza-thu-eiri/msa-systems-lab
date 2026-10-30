"""Public analysis checks use synthetic paired tokens, not private run data."""

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "evidence" / "p2_historical_summary.json"
SCRIPT = ROOT / "scripts" / "audit_paired_predictions.py"


def test_historical_summary_internal_arithmetic_and_provenance():
    data = json.loads(LEDGER.read_text(encoding="utf-8"))
    assert data["status"] == "recorded_internal_summary_not_raw_measurements"
    assert data["provenance"]["artifacts_present_in_public_repo"] is False
    assert "uncommitted" in data["provenance"]["source_warning"]
    study = data["studies"]["layer1_heldout_paired"]
    n = study["positions"]
    transitions = [
        study["correct_to_wrong"],
        study["wrong_to_correct"],
        study["wrong_to_wrong_different"],
    ]
    assert sum(transitions) == study["changed_top1_predictions"] == 3951
    gap_from_counts_pp = 100 * (transitions[1] - transitions[0]) / n
    rounded_gap_pp = 100 * (
        study["sparse_top1_accuracy_rounded"] - study["full_top1_accuracy_rounded"]
    )
    # Accuracy was recorded as a rounded percentage; do not expect exact equality.
    assert gap_from_counts_pp == pytest.approx(rounded_gap_pp, abs=0.001)
    assert 100 * study["changed_top1_predictions"] / n == pytest.approx(4.725, abs=0.002)

    schedule = data["studies"]["c07_schedule"]
    assert {row["horizon_step"] for row in schedule["options"]} == {20100, 20200}
    assert all(row["main_pairs_ratio_vs_sync_ss"] > 1 for row in schedule["options"])
    population = data["studies"]["c07_population"]
    assert population["checkpoints"] * population["batches_per_checkpoint"] == 28


def test_paired_audit_cli_tracks_all_transition_classes(tmp_path):
    source = tmp_path / "paired.csv"
    output = tmp_path / "result.json"
    rows = [
        ("A", "B", "A"),  # correct -> wrong
        ("B", "A", "A"),  # wrong -> correct
        ("B", "C", "A"),  # wrong -> another wrong
        ("A", "A", "A"),  # unchanged correct
        ("B", "B", "A"),  # unchanged wrong
    ]
    with source.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("full_prediction", "sparse_prediction", "target_token"))
        writer.writerows(rows)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(source), "--output", str(output)],
        capture_output=True, text=True, check=True,
    )
    report = json.loads(result.stdout)
    assert report == json.loads(output.read_text(encoding="utf-8"))
    assert (report["positions"], report["changed_top1_predictions"]) == (5, 3)
    assert (report["correct_to_wrong"], report["wrong_to_correct"],
            report["wrong_to_wrong_different"]) == (1, 1, 1)
    assert report["full_correct"] == report["sparse_correct"] == 2
    assert report["accuracy_gap_pp"] == 0
    assert report["top1_changed_rate"] == pytest.approx(0.6)


def test_paired_audit_cli_rejects_missing_fields(tmp_path):
    source = tmp_path / "missing.csv"
    source.write_text("full_prediction,sparse_prediction\nA,B\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(source)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "target_token" in result.stderr
