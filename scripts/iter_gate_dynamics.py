#!/usr/bin/env python3
"""Iteration 6: Gate Dynamics Analysis (Mechanistic Interpretability)."""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache")))

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.grounded_attention import GroundedCrossAttention

OUTPUT_DIR = Path("outputs/iteration_gate_dynamics")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

SCENARIOS = ["clear", "cluttered", "noise"]
PATCH_COUNT = 64
SEQ_LEN = 5
HIDDEN_DIM = 128
SAMPLES_PER_SCENARIO = 3


def normalized(tensor: torch.Tensor) -> torch.Tensor:
    return F.normalize(tensor, dim=-1)


def build_features(seed: int, scenario: str) -> torch.Tensor:
    torch.manual_seed(seed)
    if scenario == "clear":
        base = normalized(torch.randn(HIDDEN_DIM))
        image_features = base.unsqueeze(0).repeat(PATCH_COUNT, 1)
        image_features += 0.01 * torch.randn_like(image_features)
        image_features = normalized(image_features)
        text_features = base.unsqueeze(0).repeat(SEQ_LEN, 1)
        text_features += 0.02 * torch.randn_like(text_features)
    elif scenario == "cluttered":
        bases = [normalized(torch.randn(HIDDEN_DIM)) for _ in range(3)]
        image_features = torch.stack(
            [bases[i % len(bases)] + 0.2 * torch.randn(HIDDEN_DIM) for i in range(PATCH_COUNT)], dim=0
        )
        image_features = normalized(image_features)
        text_features = torch.stack(
            [bases[i % len(bases)] * (0.6 + 0.1 * torch.randn(1).item())
             + 0.4 * torch.randn(HIDDEN_DIM) for i in range(SEQ_LEN)], dim=0
        )
        text_features = normalized(text_features)
    else:  # noise
        image_features = torch.randn(PATCH_COUNT, HIDDEN_DIM)
        image_features = normalized(image_features)
        text_features = []
        for _ in range(SEQ_LEN):
            token = torch.randn(HIDDEN_DIM)
            for patch in image_features:
                token -= torch.dot(token, patch) * patch
            text_features.append(normalized(token))
        text_features = torch.stack(text_features, dim=0)

    return normalized(image_features.unsqueeze(0)), normalized(text_features.unsqueeze(0))


def measure_gate_values(attn_module: GroundedCrossAttention):
    results = []
    for scenario in SCENARIOS:
        scenario_scores = []
        for trial in range(SAMPLES_PER_SCENARIO):
            image_feats, text_feats = build_features(seed=trial, scenario=scenario)
            with torch.no_grad():
                _, grounding_scores = attn_module(
                    text_features=text_feats,
                    image_features=image_feats,
                    return_grounding_scores=True
                )
            if grounding_scores is None:
                raise RuntimeError("Expected grounding scores.")
            gates = torch.sigmoid(attn_module.grounding_scale * grounding_scores).squeeze(0)
            scenario_scores.append(float(gates.detach().mean()))
        results.append((scenario, sum(scenario_scores) / len(scenario_scores)))
    return results


def plot_gate_dynamics(results):
    labels, values = zip(*results)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=["#2ca02c", "#e377c2", "#d62728"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Average Gate Activation")
    ax.set_title("Iteration 6: Gate Dynamics Across Signal-to-Noise")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}",
                ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "gate_dynamics.png", dpi=150)
    plt.close(fig)


def main():
    attn_module = GroundedCrossAttention(
        hidden_dim=HIDDEN_DIM,
        num_heads=8,
        grounding_type="similarity",
        grounding_strength=1.5,
        use_grounding=True
    )
    attn_module.eval()

    # Relax temperature so similarity differences are visible
    attn_module.grounding_head.temperature.data.fill_(0.5)
    attn_module.grounding_scale.data.fill_(1.0)

    results = measure_gate_values(attn_module)
    plot_gate_dynamics(results)

    summary = {
        "results": [{ "scenario": scenario, "gate": gate } for scenario, gate in results],
        "samples_per_scenario": SAMPLES_PER_SCENARIO
    }
    with open(OUTPUT_DIR / "gate_dynamics_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    print("Gate Dynamics Results:")
    for scenario, gate in results:
        print(f"  {scenario.title():<10}: average gate = {gate:.2f}")
    print(f"\nSaved artifacts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
