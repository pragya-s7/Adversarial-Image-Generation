#!/usr/bin/env python3
"""
Minimal Training Script for Grounded Attention MVP

This is a simplified training script for quick proof of concept.
For production training, use train.py with full configuration.

Usage:
    python scripts/train_minimal.py --data_root /path/to/coco --output_dir outputs/test
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.llava_grounded import load_llava_with_grounding
from src.data.datasets import SimpleCaptioningDataset, DataCollator
from src.training.losses import CombinedLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal training script for MVP")
    parser.add_argument("--data_root", type=str, required=True, help="Path to image directory")
    parser.add_argument("--annotation_file", type=str, required=True, help="Path to COCO annotation file")
    parser.add_argument("--output_dir", type=str, default="outputs/mvp", help="Output directory")
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf", help="Base model name")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--max_samples", type=int, default=100, help="Max samples for testing")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--use_8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    print("="*60)
    print("Grounded Attention MVP Training")
    print("="*60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\n1. Loading model: {args.model_name}")
    model, processor, config = load_llava_with_grounding(
        model_name=args.model_name,
        grounding_type="similarity",
        device=args.device,
        load_in_8bit=args.use_8bit
    )

    # Freeze vision encoder for faster training
    print("   Freezing vision encoder...")
    for param in model.vision_tower.parameters():
        param.requires_grad = False

    # Create dataset
    print(f"\n2. Loading dataset from: {args.data_root}")
    dataset = SimpleCaptioningDataset(
        data_root=args.data_root,
        annotation_file=args.annotation_file,
        split='train',
        max_samples=args.max_samples,
        processor=processor
    )

    # Create dataloader
    collator = DataCollator(processor=processor)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging
        collate_fn=collator
    )

    print(f"   Dataset size: {len(dataset)}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Number of batches: {len(dataloader)}")

    # Setup optimizer
    print(f"\n3. Setting up optimizer")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01
    )

    # Setup scheduler
    num_training_steps = len(dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )

    # Setup loss function
    criterion = CombinedLoss(
        lambda_grounding=0.5,
        lambda_contrastive=0.0,  # No contrastive for MVP
        grounding_loss_type="margin"
    )

    # Training loop
    print(f"\n4. Starting training for {args.num_epochs} epoch(s)")
    model.train()

    global_step = 0
    for epoch in range(args.num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"{'='*60}")

        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Training")

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(args.device)
            pixel_values = batch['pixel_values'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device)

            # Forward pass
            try:
                outputs = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    labels=labels
                )

                # Compute loss
                lm_loss = outputs.loss
                total_loss, loss_dict = criterion(lm_loss=lm_loss)

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # Update metrics
                epoch_loss += total_loss.item()
                global_step += 1

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })

                # Log every 10 steps
                if global_step % 10 == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    print(f"\n   Step {global_step}: Loss = {avg_loss:.4f}")

            except Exception as e:
                print(f"\n   Error in batch {batch_idx}: {e}")
                continue

        # Epoch summary
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"\n   Epoch {epoch + 1} Average Loss: {avg_epoch_loss:.4f}")

        # Save checkpoint
        checkpoint_dir = output_dir / f"checkpoint-epoch-{epoch+1}"
        checkpoint_dir.mkdir(exist_ok=True)
        print(f"\n   Saving checkpoint to {checkpoint_dir}")

        try:
            model.save_pretrained(checkpoint_dir)
            processor.save_pretrained(checkpoint_dir)
            print("   ✓ Checkpoint saved")
        except Exception as e:
            print(f"   Error saving checkpoint: {e}")

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
