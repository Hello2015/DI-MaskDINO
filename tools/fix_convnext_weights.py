#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------
# Fix ConvNeXt weights that were saved incorrectly with pickle
# This script converts improperly formatted checkpoint files to correct format
# ------------------------------------------------------------------------
# Usage:
#   python tools/fix_convnext_weights.py \
#       --input convnext_tiny_1k_224_d2.pkl \
#       --output convnext_tiny_1k_224_d2_fixed.pkl
# ------------------------------------------------------------------------

import argparse
import pickle
import torch


def fix_convnext_weights(input_path, output_path):
    """
    Fix ConvNeXt weights that were saved with pickle in wrong format.
    
    Args:
        input_path: Path to improperly formatted checkpoint
        output_path: Path to save corrected checkpoint
    """
    print(f"Loading checkpoint from: {input_path}")
    
    # Try loading as pickle first
    try:
        with open(input_path, 'rb') as f:
            checkpoint = pickle.load(f)
        print("✓ Loaded as pickle format")
    except:
        # Try loading as torch
        try:
            checkpoint = torch.load(input_path, map_location="cpu")
            print("✓ Loaded as torch format")
        except Exception as e:
            print(f"✗ Failed to load checkpoint: {e}")
            return False
    
    # Extract model state dict
    if "model" in checkpoint:
        model_weights = checkpoint["model"]
    else:
        model_weights = checkpoint
    
    print(f"Found {len(model_weights)} weight entries")
    
    # Convert to proper format - ensure all values are torch tensors
    fixed_state_dict = {}
    
    for key, value in model_weights.items():
        # Convert to tensor if needed
        if isinstance(value, torch.Tensor):
            fixed_state_dict[key] = value
        elif isinstance(value, dict):
            # This shouldn't happen, skip
            print(f"Warning: Skipping dict value at key {key}")
            continue
        else:
            try:
                fixed_state_dict[key] = torch.tensor(value)
            except Exception as e:
                print(f"Warning: Failed to convert {key}: {e}")
                continue
    
    # Create proper checkpoint
    fixed_checkpoint = {"model": fixed_state_dict}
    
    # Save using torch.save
    print(f"Saving fixed checkpoint to: {output_path}")
    torch.save(fixed_checkpoint, output_path)
    
    print(f"✓ Successfully fixed and saved {len(fixed_state_dict)} weights")
    
    # Verify the saved file can be loaded
    print("\nVerifying saved checkpoint...")
    try:
        test_load = torch.load(output_path, map_location="cpu")
        if "model" in test_load:
            print(f"✓ Verification passed! Contains {len(test_load['model'])} weights")
            return True
        else:
            print("✗ Verification failed: 'model' key not found")
            return False
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Fix ConvNeXt weights saved with incorrect format"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to incorrectly formatted checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save fixed checkpoint"
    )
    args = parser.parse_args()
    
    success = fix_convnext_weights(args.input, args.output)
    
    if success:
        print("\n" + "="*60)
        print("✓ Checkpoint successfully fixed!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("✗ Failed to fix checkpoint")
        print("="*60)


if __name__ == "__main__":
    main()
