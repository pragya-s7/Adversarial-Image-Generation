"""
LLaVA with Grounded Attention

Integrates grounded attention into the LLaVA architecture.
This is a simplified MVP version for quick proof of concept.
"""

import torch
import torch.nn as nn
from transformers import LlavaForConditionalGeneration, LlavaConfig, AutoProcessor
from typing import Optional, Tuple, List, Dict, Any
import copy

from .grounded_attention import GroundedCrossAttention


class LlavaGroundedConfig:
    """Configuration for grounded LLaVA model."""

    def __init__(
        self,
        base_model_name: str = "llava-hf/llava-1.5-7b-hf",
        grounded_layer_indices: Optional[List[int]] = None,
        grounding_type: str = "similarity",
        grounding_strength: float = 1.0,
        use_grounding: bool = True
    ):
        self.base_model_name = base_model_name
        self.grounded_layer_indices = grounded_layer_indices
        self.grounding_type = grounding_type
        self.grounding_strength = grounding_strength
        self.use_grounding = use_grounding


def load_llava_with_grounding(
    model_name: str = "llava-hf/llava-1.5-7b-hf",
    grounded_layer_indices: Optional[List[int]] = None,
    grounding_type: str = "similarity",
    grounding_strength: float = 1.0,
    device: str = "cuda",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    torch_dtype: Optional[torch.dtype] = None
):
    """
    Load LLaVA model with grounded attention layers.

    This is a simplified version for MVP. For the final version, we'll need
    to properly integrate grounding into the decoder architecture.

    Args:
        model_name: HuggingFace model ID
        grounded_layer_indices: Which decoder layers to add grounding to (None = last 4 layers)
        grounding_type: Type of grounding mechanism ("similarity", "attention_weighted", "learnable")
        grounding_strength: Initial strength of grounding modulation
        device: Device to load model on
        load_in_8bit: Whether to use 8-bit quantization
        load_in_4bit: Whether to use 4-bit quantization
        torch_dtype: Torch dtype for model weights

    Returns:
        model, processor, grounded_config
    """
    print(f"Loading base LLaVA model from {model_name}...")

    # Prepare loading kwargs
    loading_kwargs = {
        "device_map": "auto" if (load_in_8bit or load_in_4bit) else device,
    }

    if torch_dtype is not None:
        loading_kwargs["torch_dtype"] = torch_dtype

    if load_in_8bit:
        loading_kwargs["load_in_8bit"] = True
    elif load_in_4bit:
        loading_kwargs["load_in_4bit"] = True

    # Load base model
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        **loading_kwargs
    )

    # Load processor
    processor = AutoProcessor.from_pretrained(model_name)

    # Determine which layers to modify
    if grounded_layer_indices is None:
        # Default: last 4 layers
        num_layers = model.config.text_config.num_hidden_layers
        grounded_layer_indices = list(range(num_layers - 4, num_layers))

    print(f"Adding grounding to layers: {grounded_layer_indices}")
    print(f"Grounding type: {grounding_type}")

    # Store grounding configuration
    grounded_config = LlavaGroundedConfig(
        base_model_name=model_name,
        grounded_layer_indices=grounded_layer_indices,
        grounding_type=grounding_type,
        grounding_strength=grounding_strength,
        use_grounding=True
    )

    # Add grounding config to model for tracking
    model.grounded_config = grounded_config
    model.grounding_enabled = True

    print("Model loaded successfully!")
    print("Note: Full grounding integration requires modifying decoder layers.")
    print("For MVP, use this model to test the grounding mechanism independently.")

    return model, processor, grounded_config


def create_grounded_layer_wrapper(
    original_layer: nn.Module,
    hidden_dim: int,
    num_heads: int,
    grounding_type: str = "similarity",
    grounding_strength: float = 1.0
) -> nn.Module:
    """
    Wrap an existing layer with grounded attention.

    This is a helper function for adding grounding to existing layers.
    For MVP, we'll implement this as a proof of concept.

    Args:
        original_layer: Original transformer layer
        hidden_dim: Hidden dimension size
        num_heads: Number of attention heads
        grounding_type: Type of grounding mechanism
        grounding_strength: Initial grounding strength

    Returns:
        Modified layer with grounded attention
    """
    # Create grounded attention module
    grounded_attn = GroundedCrossAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        grounding_type=grounding_type,
        grounding_strength=grounding_strength,
        use_grounding=True
    )

    # For MVP, return the grounded attention module
    # In full implementation, we would properly integrate this into the layer
    return grounded_attn


class GroundedLLaVAWrapper(nn.Module):
    """
    Wrapper class that adds grounding capabilities to LLaVA.

    This is an MVP implementation. For production, we need to modify
    the actual decoder architecture.
    """

    def __init__(
        self,
        base_model: LlavaForConditionalGeneration,
        grounded_config: LlavaGroundedConfig
    ):
        super().__init__()
        self.base_model = base_model
        self.grounded_config = grounded_config

        # Store grounding modules (for MVP testing)
        self.grounding_modules = nn.ModuleDict()

        # Initialize grounding modules for specified layers
        hidden_dim = base_model.config.text_config.hidden_size
        num_heads = base_model.config.text_config.num_attention_heads

        for layer_idx in grounded_config.grounded_layer_indices:
            self.grounding_modules[f"layer_{layer_idx}"] = GroundedCrossAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                grounding_type=grounded_config.grounding_type,
                grounding_strength=grounded_config.grounding_strength,
                use_grounding=True
            )

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_grounding_scores: bool = False,
        **kwargs
    ):
        """
        Forward pass with optional grounding score extraction.

        For MVP: This delegates to base model. Full integration requires
        modifying the forward pass to use grounding modules.
        """
        outputs = self.base_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

        if return_grounding_scores:
            # For MVP, return None for grounding scores
            # Full implementation will extract scores from grounding modules
            return outputs, None

        return outputs

    def generate(self, *args, **kwargs):
        """Delegate generation to base model."""
        return self.base_model.generate(*args, **kwargs)


# Helper function for inference
def run_grounded_inference(
    model,
    processor,
    image,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    return_grounding_scores: bool = False
):
    """
    Run inference with the grounded model.

    Args:
        model: Grounded LLaVA model
        processor: LLaVA processor
        image: PIL Image or path to image
        prompt: Text prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        return_grounding_scores: Whether to return grounding scores

    Returns:
        Generated text (and optionally grounding scores)
    """
    from PIL import Image

    # Load image if path provided
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')

    # Prepare inputs
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False
        )

    # Decode
    generated_text = processor.decode(outputs[0], skip_special_tokens=True)

    if return_grounding_scores:
        # For MVP, return None for grounding scores
        # Full implementation will extract scores
        return generated_text, None

    return generated_text


if __name__ == "__main__":
    print("Testing LLaVA Grounded integration...")
    print("\nNote: This is an MVP implementation for proof of concept.")
    print("Full integration requires modifying the decoder architecture.")

    # Test configuration
    config = LlavaGroundedConfig(
        base_model_name="llava-hf/llava-1.5-7b-hf",
        grounded_layer_indices=[28, 29, 30, 31],
        grounding_type="similarity",
        grounding_strength=1.0
    )

    print(f"\nConfiguration:")
    print(f"  Base model: {config.base_model_name}")
    print(f"  Grounded layers: {config.grounded_layer_indices}")
    print(f"  Grounding type: {config.grounding_type}")
    print(f"  Grounding strength: {config.grounding_strength}")

    print("\n✓ LLaVA grounded module structure defined successfully!")
    print("\nTo load the model:")
    print("  model, processor, config = load_llava_with_grounding()")
