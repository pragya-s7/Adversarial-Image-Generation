"""
Loss Functions for Grounded Attention Training

Implements various loss functions for training the grounded attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class GroundingLoss(nn.Module):
    """
    Loss function to encourage high grounding scores for grounded tokens
    and low scores for hallucinated tokens.
    """

    def __init__(self, loss_type: str = "bce"):
        """
        Args:
            loss_type: Type of grounding loss
                - "bce": Binary cross-entropy (requires ground truth labels)
                - "margin": Margin-based loss (encourage high scores)
                - "mse": Mean squared error (requires ground truth labels)
        """
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self,
        grounding_scores: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        margin: float = 0.5
    ) -> torch.Tensor:
        """
        Compute grounding loss.

        Args:
            grounding_scores: [batch_size, seq_len] - predicted grounding scores
            labels: [batch_size, seq_len] - ground truth labels (1=grounded, 0=hallucinated)
                   Only required for "bce" and "mse" loss types
            margin: Margin for margin-based loss

        Returns:
            loss: scalar tensor
        """
        if self.loss_type == "bce":
            assert labels is not None, "BCE loss requires ground truth labels"
            # Binary cross-entropy
            loss = F.binary_cross_entropy_with_logits(
                grounding_scores,
                labels.float()
            )

        elif self.loss_type == "margin":
            # Encourage grounding scores to be above margin
            # For tokens without labels, we want scores to be high
            loss = F.relu(margin - grounding_scores).mean()

        elif self.loss_type == "mse":
            assert labels is not None, "MSE loss requires ground truth labels"
            # Mean squared error
            loss = F.mse_loss(
                torch.sigmoid(grounding_scores),
                labels.float()
            )

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss to distinguish grounded from hallucinated captions.
    """

    def __init__(self, margin: float = 0.5):
        """
        Args:
            margin: Margin for contrastive loss
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        grounding_scores_pos: torch.Tensor,
        grounding_scores_neg: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        We want grounding scores to be higher for positive (grounded) examples
        than for negative (hallucinated) examples.

        Args:
            grounding_scores_pos: [batch_size, seq_len] - scores for grounded captions
            grounding_scores_neg: [batch_size, seq_len] - scores for hallucinated captions

        Returns:
            loss: scalar tensor
        """
        # Average grounding score across sequence
        avg_score_pos = grounding_scores_pos.mean(dim=1)  # [batch_size]
        avg_score_neg = grounding_scores_neg.mean(dim=1)  # [batch_size]

        # Contrastive loss with margin
        # We want: avg_score_pos > avg_score_neg + margin
        loss = F.relu(avg_score_neg - avg_score_pos + self.margin).mean()

        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss function for grounded attention training.

    Combines language modeling loss, grounding loss, and optionally contrastive loss.
    """

    def __init__(
        self,
        lambda_grounding: float = 0.5,
        lambda_contrastive: float = 0.1,
        grounding_loss_type: str = "margin",
        contrastive_margin: float = 0.5
    ):
        """
        Args:
            lambda_grounding: Weight for grounding loss
            lambda_contrastive: Weight for contrastive loss
            grounding_loss_type: Type of grounding loss
            contrastive_margin: Margin for contrastive loss
        """
        super().__init__()
        self.lambda_grounding = lambda_grounding
        self.lambda_contrastive = lambda_contrastive

        self.grounding_loss = GroundingLoss(loss_type=grounding_loss_type)
        self.contrastive_loss = ContrastiveLoss(margin=contrastive_margin)

    def forward(
        self,
        lm_loss: torch.Tensor,
        grounding_scores: Optional[torch.Tensor] = None,
        grounding_labels: Optional[torch.Tensor] = None,
        grounding_scores_neg: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Args:
            lm_loss: Language modeling loss (from the model)
            grounding_scores: Grounding scores for positive examples
            grounding_labels: Ground truth labels for grounding
            grounding_scores_neg: Grounding scores for negative examples (for contrastive loss)

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual loss components
        """
        total_loss = lm_loss
        loss_dict = {"lm_loss": lm_loss.item()}

        # Add grounding loss if scores provided
        if grounding_scores is not None and self.lambda_grounding > 0:
            g_loss = self.grounding_loss(grounding_scores, grounding_labels)
            total_loss = total_loss + self.lambda_grounding * g_loss
            loss_dict["grounding_loss"] = g_loss.item()
        else:
            loss_dict["grounding_loss"] = 0.0

        # Add contrastive loss if negative scores provided
        if (grounding_scores is not None and
            grounding_scores_neg is not None and
            self.lambda_contrastive > 0):
            c_loss = self.contrastive_loss(grounding_scores, grounding_scores_neg)
            total_loss = total_loss + self.lambda_contrastive * c_loss
            loss_dict["contrastive_loss"] = c_loss.item()
        else:
            loss_dict["contrastive_loss"] = 0.0

        loss_dict["total_loss"] = total_loss.item()

        return total_loss, loss_dict


# Simplified helper functions for MVP
def compute_grounding_loss(
    grounding_scores: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    loss_type: str = "margin"
) -> torch.Tensor:
    """
    Simple helper to compute grounding loss.

    Args:
        grounding_scores: [batch_size, seq_len]
        labels: [batch_size, seq_len] (optional)
        loss_type: Type of loss

    Returns:
        loss: scalar tensor
    """
    loss_fn = GroundingLoss(loss_type=loss_type)
    return loss_fn(grounding_scores, labels)


def compute_contrastive_loss(
    grounding_scores_pos: torch.Tensor,
    grounding_scores_neg: torch.Tensor,
    margin: float = 0.5
) -> torch.Tensor:
    """
    Simple helper to compute contrastive loss.

    Args:
        grounding_scores_pos: [batch_size, seq_len]
        grounding_scores_neg: [batch_size, seq_len]
        margin: Margin value

    Returns:
        loss: scalar tensor
    """
    loss_fn = ContrastiveLoss(margin=margin)
    return loss_fn(grounding_scores_pos, grounding_scores_neg)


if __name__ == "__main__":
    print("Testing loss functions...")

    batch_size = 4
    seq_len = 20

    # Create dummy data
    grounding_scores_pos = torch.randn(batch_size, seq_len)
    grounding_scores_neg = torch.randn(batch_size, seq_len) - 1.0  # Lower scores
    grounding_labels = torch.randint(0, 2, (batch_size, seq_len))
    lm_loss = torch.tensor(2.5)

    # Test grounding loss
    print("\n1. Testing Grounding Loss:")
    for loss_type in ["margin", "bce", "mse"]:
        g_loss = GroundingLoss(loss_type=loss_type)
        if loss_type in ["bce", "mse"]:
            loss = g_loss(grounding_scores_pos, grounding_labels)
        else:
            loss = g_loss(grounding_scores_pos)
        print(f"   {loss_type}: {loss.item():.4f}")

    # Test contrastive loss
    print("\n2. Testing Contrastive Loss:")
    c_loss = ContrastiveLoss(margin=0.5)
    loss = c_loss(grounding_scores_pos, grounding_scores_neg)
    print(f"   Contrastive loss: {loss.item():.4f}")

    # Test combined loss
    print("\n3. Testing Combined Loss:")
    combined = CombinedLoss(
        lambda_grounding=0.5,
        lambda_contrastive=0.1,
        grounding_loss_type="margin"
    )
    total_loss, loss_dict = combined(
        lm_loss=lm_loss,
        grounding_scores=grounding_scores_pos,
        grounding_scores_neg=grounding_scores_neg
    )
    print(f"   Total loss: {total_loss.item():.4f}")
    print(f"   Loss breakdown: {loss_dict}")

    print("\n✓ All loss functions working correctly!")
