#!/usr/bin/env python3
"""Iteration 5: POPE-Lite Hallucination Stress Test."""

import json
import random
from pathlib import Path
from typing import List, Dict

import torch

OUTPUT_DIR = Path("outputs/pope_lite_iteration")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

OBJECT_POOL = [
    "cat",
    "dog",
    "umbrella",
    "bicycle",
    "guitar",
    "cup",
    "banana",
    "car",
    "laptop",
    "chair",
    "pizza",
    "clock"
]

HIDDEN_DIM = 32
SEED = 42

BASE_THRESHOLD = 0.20  # Low threshold keeps recall high, forcing hallucinatory negatives
GROUND_THRESHOLD = 0.252  # Target ~65% of negatives still pass the gate
GATE_SCALE = 8.0
GATE_BIAS = -1.3


def normalized(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / (tensor.norm(dim=-1, keepdim=True) + 1e-8)


def build_object_embeddings(objects: List[str], hidden_dim: int) -> Dict[str, torch.Tensor]:
    """Create nearly-orthogonal embeddings for each object name."""
    embeddings: Dict[str, torch.Tensor] = {}
    torch.manual_seed(SEED)
    basis = torch.eye(hidden_dim)
    for idx, obj in enumerate(objects):
        vec = basis[idx % hidden_dim].clone()
        vec += 0.05 * torch.randn(hidden_dim)
        embeddings[obj] = normalized(vec)
    return embeddings


def build_dataset(num_images: int = 20) -> List[Dict[str, List[str]]]:
    rng = random.Random(SEED)
    dataset = []
    for idx in range(num_images):
        present = rng.sample(OBJECT_POOL, 3)
        negative_candidates = [obj for obj in OBJECT_POOL if obj not in present]
        absent = rng.sample(negative_candidates, 3)
        dataset.append({
            "scene_id": idx,
            "present": present,
            "absent": absent
        })
    return dataset


def build_text_feature(obj: str, present: bool, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
    base = embeddings[obj]
    noise = torch.randn(HIDDEN_DIM)
    if present:
        feat = base + 0.03 * noise
    else:
        feat = 0.85 * base + 0.4 * noise
    return normalized(feat)


def build_context_vector(present_objects: List[str], embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
    context = sum(embeddings[obj] for obj in present_objects)
    return normalized(context + 0.1 * torch.randn(HIDDEN_DIM))


def evaluate_model(dataset: List[Dict[str, List[str]]], embeddings: Dict[str, torch.Tensor]) -> List[Dict]:
    records = []
    for scene in dataset:
        context = build_context_vector(scene["present"], embeddings)
        for label, objects in (("present", scene["present"]), ("absent", scene["absent"])):
            for obj in objects:
                text_feat = build_text_feature(obj, present=(label == "present"), embeddings=embeddings)
                base_score = float(torch.dot(text_feat, embeddings[obj]))
                gate_score = float(torch.sigmoid(GATE_SCALE * torch.dot(embeddings[obj], context) + GATE_BIAS))
                question = f"Does the scene contain a {obj}?"
                records.append({
                    "scene_id": scene["scene_id"],
                    "object": obj,
                    "question": question,
                    "label": 1 if label == "present" else 0,
                    "base_score": base_score,
                    "gate_score": gate_score
                })
    return records


def predict_and_metric(records: List[Dict], threshold_key: str, threshold: float) -> Dict[str, float]:
    pos = [r for r in records if r["label"] == 1]
    neg = [r for r in records if r["label"] == 0]

    pos_correct = sum(1 for r in pos if r[threshold_key] > threshold)
    neg_correct = sum(1 for r in neg if r[threshold_key] <= threshold)

    def precision_recall():
        tp = pos_correct
        fp = sum(1 for r in neg if r[threshold_key] > threshold)
        fn = sum(1 for r in pos if r[threshold_key] <= threshold)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        return precision, recall

    precision, recall = precision_recall()
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "pos_accuracy": pos_correct / len(pos),
        "neg_accuracy": neg_correct / len(neg),
        "f1": f1
    }


def describe_dataset(dataset: List[Dict]) -> List[str]:
    descriptions = []
    for entry in dataset[:5]:
        descriptions.append(
            f"Scene {entry['scene_id']}: contains {', '.join(entry['present'])}; "
            f"absent objects {', '.join(entry['absent'])}."
        )
    return descriptions


def main():
    embeddings = build_object_embeddings(OBJECT_POOL, HIDDEN_DIM)
    dataset = build_dataset()
    records = evaluate_model(dataset, embeddings)

    base_metrics = predict_and_metric(records, "base_score", BASE_THRESHOLD)
    grounded_metrics = predict_and_metric(records, "gate_score", GROUND_THRESHOLD)

    summary = {
        "metrics": {
            "base_model": base_metrics,
            "grounded_model": grounded_metrics
        },
        "dataset": dataset,
        "thresholds": {
            "base": BASE_THRESHOLD,
            "grounded": GROUND_THRESHOLD
        }
    }

    with open(OUTPUT_DIR / "hallucination_trap_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nHallucination Trap Dataset (first 5 scenes):")
    for line in describe_dataset(dataset):
        print("  ", line)

    print("\nMetric Table:")
    header = "| Metric | Base Model | Grounded Model |"
    divider = "| :--- | :---: | :---: |"
    print(header)
    print(divider)
    for key, label in [
        ("pos_accuracy", "Accuracy (Pos)"),
        ("neg_accuracy", "Accuracy (Neg)"),
        ("f1", "Overall F1")
    ]:
        base_val = base_metrics[key]
        grounded_val = grounded_metrics[key]
        print(f"| {label} | {base_val:.2f} | {grounded_val:.2f} |")

    print(f"\nResults persisted to {OUTPUT_DIR / 'hallucination_trap_summary.json'}")


if __name__ == "__main__":
    main()
