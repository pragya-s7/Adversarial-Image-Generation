#!/usr/bin/env python3
"""Iteration 4: Attention + Gate Visualization"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache")))

import torch
import matplotlib.pyplot as plt

# Ensure repo root in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.grounded_attention import GroundedCrossAttention

OUTPUT_PATH = Path("outputs/iteration_attention_viz.png")
OUTPUT_PATH.parent.mkdir(exist_ok=True, parents=True)


TOKEN_LABELS = [
    "bear", "sitting", "on", "grass", "text", "logo", "noise", "shelf", "unknown", "frame"
]


def build_synthetic_features(batch_size: int = 1,
                             token_count: int = len(TOKEN_LABELS),
                             patch_count: int = 64,
                             hidden_dim: int = 128):
    torch.manual_seed(42)
    image_features = torch.randn(batch_size, patch_count, hidden_dim)
    text_features = torch.randn(batch_size, token_count, hidden_dim)

    # Anchor the first few tokens on actual patches
    for i in range(4):
        patch_idx = (i * 5) % patch_count
        text_features[0, i] = image_features[0, patch_idx] + 0.02 * torch.randn(hidden_dim)

    # Distort the remainder so they look less grounded
    for i in range(4, token_count):
        text_features[0, i] = torch.randn(hidden_dim) * 2

    return image_features, text_features


def visualize_attention():
    image_features, text_features = build_synthetic_features()

    attn_module = GroundedCrossAttention(hidden_dim=text_features.size(-1),
                                         num_heads=4,
                                         grounding_type="similarity",
                                         grounding_strength=1.0,
                                         use_grounding=True)
    attn_module.eval()

    _, grounding_scores = attn_module(
        text_features=text_features,
        image_features=image_features,
        return_grounding_scores=True
    )

    if grounding_scores is None:
        raise RuntimeError("Grounding scores were not returned")

    gate_values = torch.sigmoid(attn_module.grounding_scale * grounding_scores)
    gate_values = gate_values.squeeze(0).detach().cpu().numpy()

    pre_attn = attn_module.last_attn_before_grounding
    post_attn = attn_module.last_attn_after_grounding

    if pre_attn is None or post_attn is None:
        raise RuntimeError("Attention snapshots were not captured")

    def mean_across_heads(attn_tensor):
        attn_np = attn_tensor[0].mean(axis=0)
        return attn_np

    pre_map = mean_across_heads(pre_attn)
    post_map = mean_across_heads(post_attn)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(pre_map, aspect="auto", cmap="viridis")
    axes[0].set_title("Softmax Attention (Before Gate)")
    axes[0].set_ylabel("Token")
    axes[0].set_xlabel("Patches")
    axes[0].set_yticks(range(len(TOKEN_LABELS)))
    axes[0].set_yticklabels(TOKEN_LABELS)
    plt.colorbar(im0, ax=axes[0], pad=0.02)

    im1 = axes[1].imshow(post_map, aspect="auto", cmap="magma")
    axes[1].set_title("Grounded Attention (After Gate)")
    axes[1].set_xlabel("Patches")
    axes[1].set_yticks(range(len(TOKEN_LABELS)))
    axes[1].set_yticklabels(TOKEN_LABELS)
    plt.colorbar(im1, ax=axes[1], pad=0.02)

    colors = ["tab:green" if i < 4 else "tab:red" for i in range(len(TOKEN_LABELS))]
    bars = axes[2].bar(range(len(TOKEN_LABELS)), gate_values, color=colors)
    axes[3-1].set_title("Gate Activation per Token")
    axes[2].set_ylim(0, 1.0)
    axes[2].set_xticks(range(len(TOKEN_LABELS)))
    axes[2].set_xticklabels(TOKEN_LABELS, rotation=45, ha="right")
    axes[2].set_ylabel("Sigmoid Gate")

    for rect, val in zip(bars, gate_values):
        axes[2].text(rect.get_x() + rect.get_width() / 2, val + 0.02, f"{val:.2f}",
                     ha="center", va="bottom", fontsize=9, rotation=90)

    fig.suptitle("Iteration 4: Grounded Attention Visualization", fontsize=16)
    plt.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved attention visualization to {OUTPUT_PATH}")


def main():
    visualize_attention()


if __name__ == "__main__":
    main()
