#!/usr/bin/env python3
"""
Simple Evaluation Script for Grounded Attention MVP

Quick evaluation script for testing the model on sample images.
For full benchmarks (POPE, CHAIR, MME), see evaluation/ directory.

Usage:
    python scripts/evaluate_simple.py --model_path outputs/checkpoint --image_path test.jpg
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.llava_grounded import load_llava_with_grounding, run_grounded_inference


def parse_args():
    parser = argparse.ArgumentParser(description="Simple evaluation script")
    parser.add_argument("--model_path", type=str, help="Path to saved model checkpoint")
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf", help="Base model name")
    parser.add_argument("--image_path", type=str, help="Path to test image")
    parser.add_argument("--image_dir", type=str, help="Directory of test images")
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.", help="Prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_8bit", action="store_true", help="Use 8-bit quantization")
    return parser.parse_args()


def evaluate_single_image(model, processor, image_path: str, prompt: str, args):
    """Evaluate on a single image."""
    print(f"\n{'='*60}")
    print(f"Image: {image_path}")
    print(f"{'='*60}")

    # Load image
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"Image size: {image.size}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Run inference
    print(f"\nPrompt: {prompt}")
    print("\nGenerating response...")

    try:
        response = run_grounded_inference(
            model=model,
            processor=processor,
            image=image,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            return_grounding_scores=False
        )

        print(f"\n{'-'*60}")
        print("Response:")
        print(f"{'-'*60}")
        print(response)
        print(f"{'-'*60}\n")

    except Exception as e:
        print(f"Error during generation: {e}")


def evaluate_directory(model, processor, image_dir: str, prompt: str, args):
    """Evaluate on all images in a directory."""
    image_dir = Path(image_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    # Find all images
    image_files = [
        f for f in image_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    print(f"\nFound {len(image_files)} images in {image_dir}")

    # Evaluate each image
    for i, image_file in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}]")
        evaluate_single_image(model, processor, str(image_file), prompt, args)


def main():
    args = parse_args()

    print("="*60)
    print("Grounded Attention Simple Evaluation")
    print("="*60)

    # Determine model path
    if args.model_path:
        model_name = args.model_path
        print(f"\nLoading model from: {model_name}")
    else:
        model_name = args.model_name
        print(f"\nLoading base model: {model_name}")

    # Load model
    print("\n1. Loading model...")
    try:
        model, processor, config = load_llava_with_grounding(
            model_name=model_name,
            grounding_type="similarity",
            device=args.device,
            load_in_8bit=args.use_8bit
        )
        model.eval()
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   Error loading model: {e}")
        return

    # Run evaluation
    print("\n2. Running evaluation...")

    if args.image_path:
        # Evaluate single image
        evaluate_single_image(model, processor, args.image_path, args.prompt, args)

    elif args.image_dir:
        # Evaluate directory
        evaluate_directory(model, processor, args.image_dir, args.prompt, args)

    else:
        print("\nError: Must provide either --image_path or --image_dir")
        print("\nExample usage:")
        print("  python scripts/evaluate_simple.py --image_path test.jpg")
        print("  python scripts/evaluate_simple.py --image_dir test_images/")

    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
