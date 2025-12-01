#!/usr/bin/env python
# ------------------------------------------------------------------------
# Convert ConvNeXt pretrained weights to Detectron2 format
# ------------------------------------------------------------------------
# Usage:
#   python tools/convert_convnext_to_d2.py \
#       --source convnext_tiny_1k_224_ema.pth \
#       --output convnext_tiny_1k_224_d2.pkl
# ------------------------------------------------------------------------

import argparse
import pickle
import torch


def convert_convnext_weights(src_path, dst_path):
    """
    Convert ConvNeXt weights from timm/official format to Detectron2 format.
    
    Args:
        src_path: Path to source ConvNeXt checkpoint (.pth file)
        dst_path: Path to save converted checkpoint (.pkl file)
    """
    # Load source checkpoint
    checkpoint = torch.load(src_path, map_location="cpu")
    
    # Handle different checkpoint formats
    if "model" in checkpoint:
        model_weights = checkpoint["model"]
    elif "state_dict" in checkpoint:
        model_weights = checkpoint["state_dict"]
    else:
        model_weights = checkpoint
    
    # Create new state dict with Detectron2 naming convention
    new_state_dict = {}
    
    for key, value in model_weights.items():
        # Skip classifier head (we don't need it for detection)
        if key.startswith("head."):
            continue
        
        # ConvNeXt layer name mapping
        new_key = key
        
        # Handle downsample_layers (stem + 3 downsampling layers)
        if "downsample_layers" in key:
            new_key = key  # Keep as is, matches our implementation
        
        # Handle stages (feature extraction blocks)
        elif "stages" in key:
            new_key = key  # Keep as is, matches our implementation
        
        # Handle layer norms
        elif key.startswith("norm"):
            new_key = key  # Keep as is
        
        # Add backbone prefix for Detectron2
        new_key = "backbone." + new_key
        
        new_state_dict[new_key] = value
    
    # Print conversion summary
    print(f"Converted {len(model_weights)} source keys to {len(new_state_dict)} D2 keys")
    print(f"Skipped {len(model_weights) - len(new_state_dict)} keys (classifier head, etc.)")
    
    # Print some example keys for verification
    print("\nExample converted keys:")
    for i, key in enumerate(list(new_state_dict.keys())[:5]):
        print(f"  {key}: {new_state_dict[key].shape}")
    
    # Save in Detectron2 format (pickle)
    # Note: Detectron2 expects the checkpoint to be a dict with 'model' key
    # containing state_dict, but when loading it uses DetectionCheckpointer
    # which needs proper format with torch tensors
    checkpoint_dict = {"model": new_state_dict}
    
    # Convert numpy arrays to torch tensors if needed
    for key in checkpoint_dict["model"]:
        if not isinstance(checkpoint_dict["model"][key], torch.Tensor):
            checkpoint_dict["model"][key] = torch.tensor(checkpoint_dict["model"][key])
    
    # Save using torch.save for proper tensor serialization
    torch.save(checkpoint_dict, dst_path)
    
    print(f"\nSuccessfully converted and saved to: {dst_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert ConvNeXt weights to Detectron2 format")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to source ConvNeXt checkpoint (.pth file)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save converted checkpoint (.pkl file)"
    )
    args = parser.parse_args()
    
    convert_convnext_weights(args.source, args.output)


if __name__ == "__main__":
    main()
