#!/usr/bin/env python3
"""Iteration 7: Gate-Based Hallucination Flagging and Calibration."""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache")))

import matplotlib.pyplot as plt

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.iter_pope_lite import (
    BASE_THRESHOLD,
    GROUND_THRESHOLD,
    OBJECT_POOL,
    HIDDEN_DIM,
    build_dataset,
    build_object_embeddings,
    evaluate_model,
)

OUTPUT_DIR = Path("outputs/iteration_gate_flagging")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

GATE_FLAG_THRESHOLD = 0.35
GATE_CONFIDENCE_BINS = 10


def threshold_metrics(records: List[Dict], key: str, threshold: float) -> Dict[str, float]:
    pos = [r for r in records if r["label"] == 1]
    neg = [r for r in records if r["label"] == 0]

    pos_correct = sum(1 for r in pos if r[key] >= threshold)
    neg_correct = sum(1 for r in neg if r[key] < threshold)

    tp = pos_correct
    fp = sum(1 for r in neg if r[key] >= threshold)
    fn = sum(1 for r in pos if r[key] < threshold)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "pos_accuracy": pos_correct / len(pos),
        "neg_accuracy": neg_correct / len(neg),
        "f1": f1,
        "threshold": threshold,
    }


def plot_gate_distribution(records: List[Dict]):
    pos = [r["gate_score"] for r in records if r["label"] == 1]
    neg = [r["gate_score"] for r in records if r["label"] == 0]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(pos, bins=GATE_CONFIDENCE_BINS, alpha=0.7, label="Positives", color="#2ca02c")
    ax.hist(neg, bins=GATE_CONFIDENCE_BINS, alpha=0.7, label="Negatives", color="#d62728")
    ax.axvline(GATE_FLAG_THRESHOLD, color="black", linestyle="--", label="Flag Threshold")
    ax.set_xlabel("Gate Score")
    ax.set_ylabel("Count")
    ax.set_title("Iteration 7: Gate Reliability Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "gate_score_distribution.png", dpi=150)
    plt.close(fig)


def main():
    embeddings = build_object_embeddings(OBJECT_POOL, HIDDEN_DIM)
    dataset = build_dataset()
    records = evaluate_model(dataset, embeddings)

    base_metrics = threshold_metrics(records, "base_score", BASE_THRESHOLD)
    grounded_metrics = threshold_metrics(records, "gate_score", GROUND_THRESHOLD)
    gate_flag_metrics = threshold_metrics(records, "gate_score", GATE_FLAG_THRESHOLD)

    table_lines = [
        "| Metric | Base Model | Grounded Gate | Gate Flagging |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for label, key in [
        ("Accuracy (Pos)", "pos_accuracy"),
        ("Accuracy (Neg)", "neg_accuracy"),
        ("Overall F1", "f1"),
    ]:
        table_lines.append(
            f"| {label} | {base_metrics[key]:.2f} | {grounded_metrics[key]:.2f} | {gate_flag_metrics[key]:.2f} |"
        )

    summary = {
        "metrics": {
            "base_model": base_metrics,
            "grounded_model": grounded_metrics,
            "gate_flag": gate_flag_metrics,
        },
        "thresholds": {
            "base": BASE_THRESHOLD,
            "grounded": GROUND_THRESHOLD,
            "gate_flag": GATE_FLAG_THRESHOLD,
        },
    }
    with open(OUTPUT_DIR / "gate_flagging_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    print("\nGate Flagging Threshold Table:")
    print("\n".join(table_lines))

    print("\nGate Flagging Stats:")
    pos = [r for r in records if r["label"] == 1]
    neg = [r for r in records if r["label"] == 0]
    print(f"  Positives with gate >= {GATE_FLAG_THRESHOLD:.2f}: {sum(1 for r in pos if r['gate_score'] >= GATE_FLAG_THRESHOLD)}/{len(pos)}")
    print(f"  Negatives with gate <  {GATE_FLAG_THRESHOLD:.2f}: {sum(1 for r in neg if r['gate_score'] < GATE_FLAG_THRESHOLD)}/{len(neg)}")

    plot_gate_distribution(records)
    print(f"\nSaved artifacts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
