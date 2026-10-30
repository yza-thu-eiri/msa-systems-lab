"""Audit paired FULL/SPARSE token predictions from a user-supplied CSV.

This is a standalone public analysis utility. It does not contain or reconstruct
the private C06 historical token-level audit and does not rerun model inference.

CSV columns: full_prediction, sparse_prediction, target_token.
All token identities are interpreted as strings, so integer IDs and text labels
are both valid if they are consistent within the CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


COLUMNS = ("full_prediction", "sparse_prediction", "target_token")


def summarize(rows: list[dict[str, str]]) -> dict[str, int | float]:
    if not rows:
        raise ValueError("paired prediction CSV is empty")
    n = len(rows)
    transitions = {
        "correct_to_wrong": 0,
        "wrong_to_correct": 0,
        "wrong_to_wrong_different": 0,
    }
    full_correct = 0
    sparse_correct = 0
    changed = 0
    for index, row in enumerate(rows):
        if any(k not in row or row[k] is None or row[k] == "" for k in COLUMNS):
            raise ValueError(f"missing paired prediction fields on data row {index + 1}")
        f, s, y = (row[name] for name in COLUMNS)
        fc, sc = f == y, s == y
        full_correct += int(fc)
        sparse_correct += int(sc)
        if f == s:
            continue
        changed += 1
        if fc:
            transitions["correct_to_wrong"] += 1
        elif sc:
            transitions["wrong_to_correct"] += 1
        else:
            transitions["wrong_to_wrong_different"] += 1
    assert sum(transitions.values()) == changed
    return {
        "positions": n,
        "full_correct": full_correct,
        "sparse_correct": sparse_correct,
        "changed_top1_predictions": changed,
        **transitions,
        "full_accuracy": full_correct / n,
        "sparse_accuracy": sparse_correct / n,
        "accuracy_gap_pp": 100.0 * (sparse_correct - full_correct) / n,
        "top1_changed_rate": changed / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="paired per-token CSV")
    parser.add_argument("--output", type=Path, help="optional JSON summary path")
    args = parser.parse_args()
    with args.input.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not set(COLUMNS) <= set(reader.fieldnames or ()):
            parser.error("CSV must contain: " + ", ".join(COLUMNS))
        result = summarize(list(reader))
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
