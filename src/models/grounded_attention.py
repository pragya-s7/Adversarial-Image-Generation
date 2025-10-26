"""
Grounded Attention Mechanism

This module implements the core grounding mechanism that can be integrated
into any vision-language transformer architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class GroundingHead(nn.Module):
    """
    Computes grounding scores for text tokens based on visual features.
    """

    def __init__(
        self,
        hidden_dim: int,
        grounding_type: str = "similarity",
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_dim: Dimension of hidden features
            grounding_type: Type of grounding computation
                - "similarity": Cosine similarity-based (recommended for MVP)
                - "attention_weighted": Weighted by attention scores
                - "learnable": Learnable MLP-based
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.grounding_type = grounding_type
        self.num_heads = num_heads

        if grounding_type == "learnable":
            self.grounding_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )

        # Learnable temperature for scaling similarities
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute grounding scores.

        Args:
            text_features: [batch_size, seq_len, hidden_dim]
            image_features: [batch_size, num_patches, hidden_dim]
            attention_weights: [batch_size, num_heads, seq_len, num_patches] (optional)

        Returns:
            grounding_scores: [batch_size, seq_len]
        """
        if self.grounding_type == "similarity":
            return self._similarity_grounding(text_features, image_features)
        elif self.grounding_type == "attention_weighted":
            assert attention_weights is not None, "attention_weights required for attention_weighted grounding"
            return self._attention_weighted_grounding(text_features, image_features, attention_weights)
        elif self.grounding_type == "learnable":
            return self._learnable_grounding(text_features, image_features, attention_weights)
        else:
            raise ValueError(f"Unknown grounding type: {self.grounding_type}")

    def _similarity_grounding(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute grounding based on maximum cosine similarity.
        """
        # Normalize features
        text_norm = F.normalize(text_features, p=2, dim=-1)  # [B, T, D]
        image_norm = F.normalize(image_features, p=2, dim=-1)  # [B, P, D]

        # Compute similarity matrix
        similarity = torch.matmul(text_norm, image_norm.transpose(-1, -2))  # [B, T, P]
        similarity = similarity / self.temperature

        # Take maximum similarity across all patches
        grounding_scores, _ = similarity.max(dim=-1)  # [B, T]

        return grounding_scores

    def _attention_weighted_grounding(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute grounding weighted by attention distribution.
        """
        # Normalize features
        text_norm = F.normalize(text_features, p=2, dim=-1)
        image_norm = F.normalize(image_features, p=2, dim=-1)

        # Compute similarity
        similarity = torch.matmul(text_norm, image_norm.transpose(-1, -2))  # [B, T, P]
        similarity = similarity / self.temperature

        # Average attention across heads
        attn_avg = attention_weights.mean(dim=1)  # [B, T, P]

        # Weighted grounding score
        grounding_scores = (similarity * attn_avg).sum(dim=-1)  # [B, T]

        return grounding_scores

    def _learnable_grounding(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute grounding using learnable MLP.
        """
        B, T, D = text_features.shape

        # Get attended image features (if attention weights available)
        if attention_weights is not None:
            attn_avg = attention_weights.mean(dim=1)  # [B, T, P]
            attended_image = torch.matmul(attn_avg, image_features)  # [B, T, D]
        else:
            # Use mean pooled image features as fallback
            attended_image = image_features.mean(dim=1, keepdim=True).expand(B, T, D)

        # Concatenate text and attended image features
        combined = torch.cat([text_features, attended_image], dim=-1)  # [B, T, 2D]

        # Predict grounding score
        grounding_scores = self.grounding_mlp(combined).squeeze(-1)  # [B, T]

        return grounding_scores


class GroundedCrossAttention(nn.Module):
    """
    Cross-attention layer with integrated grounding mechanism.

    This replaces standard cross-attention in vision-language transformers.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        grounding_type: str = "similarity",
        grounding_strength: float = 1.0,
        use_grounding: bool = True
    ):
        """
        Args:
            hidden_dim: Dimension of hidden features
            num_heads: Number of attention heads
            dropout: Dropout probability
            grounding_type: Type of grounding mechanism
            grounding_strength: Initial strength of grounding modulation
            use_grounding: Whether to use grounding (for ablation)
        """
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_grounding = use_grounding

        # Standard attention components
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # Grounding components
        if use_grounding:
            self.grounding_head = GroundingHead(
                hidden_dim=hidden_dim,
                grounding_type=grounding_type,
                num_heads=num_heads,
                dropout=dropout
            )

            # Learnable grounding strength
            self.grounding_scale = nn.Parameter(
                torch.tensor(grounding_strength, dtype=torch.float32)
            )

        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_grounding_scores: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional grounding.

        Args:
            text_features: [batch_size, seq_len, hidden_dim]
            image_features: [batch_size, num_patches, hidden_dim]
            attention_mask: [batch_size, seq_len, num_patches] (optional)
            return_grounding_scores: Whether to return grounding scores

        Returns:
            output: [batch_size, seq_len, hidden_dim]
            grounding_scores: [batch_size, seq_len] (if return_grounding_scores=True)
        """
        B, T, D = text_features.shape
        P = image_features.shape[1]

        # Project to Q, K, V
        Q = self.q_proj(text_features)  # [B, T, D]
        K = self.k_proj(image_features)  # [B, P, D]
        V = self.v_proj(image_features)  # [B, P, D]

        # Reshape for multi-head attention
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D/H]
        K = K.view(B, P, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, P, D/H]
        V = V.view(B, P, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, P, D/H]

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)  # [B, H, T, P]

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(
                attention_mask.unsqueeze(1) == 0,
                float('-inf')
            )

        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, T, P]
        attn_weights = self.attn_dropout(attn_weights)

        # Apply grounding modulation if enabled
        grounding_scores = None
        if self.use_grounding:
            # Compute grounding scores
            grounding_scores = self.grounding_head(
                text_features,
                image_features,
                attn_weights
            )  # [B, T]

            # Apply sigmoid and scale
            grounding_gate = torch.sigmoid(self.grounding_scale * grounding_scores)  # [B, T]

            # Expand to match attention shape
            grounding_gate = grounding_gate.unsqueeze(1).unsqueeze(-1)  # [B, 1, T, 1]

            # Modulate attention weights
            attn_weights = attn_weights * grounding_gate

            # Renormalize (crucial for stability)
            attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Apply attention to values
        attended = torch.matmul(attn_weights, V)  # [B, H, T, D/H]

        # Reshape and project output
        attended = attended.transpose(1, 2).contiguous().view(B, T, D)  # [B, T, D]
        output = self.out_proj(attended)
        output = self.dropout(output)

        # Layer norm with residual connection
        output = self.layer_norm(output + text_features)

        if return_grounding_scores:
            return output, grounding_scores
        return output, None


if __name__ == "__main__":
    # Test grounded attention layer
    print("Testing GroundedCrossAttention...")

    batch_size = 2
    seq_len = 10
    num_patches = 576  # 24x24 patches
    hidden_dim = 768

    # Create dummy inputs
    text_features = torch.randn(batch_size, seq_len, hidden_dim)
    image_features = torch.randn(batch_size, num_patches, hidden_dim)

    # Test with grounding
    grounded_attn = GroundedCrossAttention(
        hidden_dim=hidden_dim,
        num_heads=8,
        grounding_type="similarity",
        use_grounding=True
    )

    output, grounding_scores = grounded_attn(
        text_features,
        image_features,
        return_grounding_scores=True
    )

    print(f"✓ Output shape: {output.shape}")  # [2, 10, 768]
    print(f"✓ Grounding scores shape: {grounding_scores.shape}")  # [2, 10]
    print(f"✓ Grounding scores range: [{grounding_scores.min():.3f}, {grounding_scores.max():.3f}]")

    # Test without grounding (ablation)
    standard_attn = GroundedCrossAttention(
        hidden_dim=hidden_dim,
        num_heads=8,
        use_grounding=False
    )

    output_no_ground, _ = standard_attn(
        text_features,
        image_features,
        return_grounding_scores=False
    )

    print(f"✓ Standard attention output shape: {output_no_ground.shape}")
    print("\nGrounded attention module working correctly!")
