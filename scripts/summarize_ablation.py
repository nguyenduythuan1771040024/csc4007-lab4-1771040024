"""Build a validation-only comparison table from independent Lab 4 runs.

Do NOT use test metrics for choosing an architecture or hyperparameters.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def collect(root: Path) -> list[dict]:
    runs = []
    for path in sorted(root.glob("*/metrics/metrics_summary.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        validation = meta.get("val", {})
        if validation.get("macro_f1") is None:
            continue
        runs.append({
            "run_name": meta.get("run_name", path.parents[1].name),
            "dataset": meta.get("dataset"),
            "seed": meta.get("seed"),
            "model_type": meta.get("model_type"),
            "bidirectional": meta.get("bidirectional"),
            "num_layers": meta.get("num_layers"),
            "max_len": meta.get("max_len"),
            "hidden_dim": meta.get("hidden_dim"),
            "dropout": meta.get("dropout"),
            "val_accuracy": float(validation["accuracy"]),
            "val_macro_f1": float(validation["macro_f1"]),
            "val_loss": float(validation["loss"]),
            "source": str(path),
        })
    if not runs:
        raise ValueError(f"No completed run summaries in {root}; run run_lab4.py first")
    signatures = {(r["dataset"], r["seed"]) for r in runs}
    if len(signatures) > 1:
        raise ValueError("Runs use different dataset/seed; specify a folder with comparable experiments")
    best = max(runs, key=lambda r: (r["val_macro_f1"], -r["val_loss"]))
    for r in runs:
        r["selected_by_validation"] = r is best
    return runs


def save_table(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LSTM/GRU runs using validation only")
    parser.add_argument("--runs_dir", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/ablation_validation.csv"))
    args = parser.parse_args()
    rows = collect(args.runs_dir)
    save_table(rows, args.output)
    selected = next(r for r in rows if r["selected_by_validation"])
    print(f"Selected by validation: {selected['run_name']}; table: {args.output}")
    print("After model choice is fixed, report its test performance once; never rank by test metrics.")


if __name__ == "__main__":
    main()
