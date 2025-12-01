#!/usr/bin/env python
# ------------------------------------------------------------------------
# Test ConvNeXt Backbone Integration
# ------------------------------------------------------------------------
# This script tests if the ConvNeXt backbone is correctly integrated.
# Usage: python tools/test_convnext_backbone.py
# ------------------------------------------------------------------------

import sys
import torch
from detectron2.config import get_cfg
from dimaskdino import add_dimaskdino_config
from detectron2.modeling import build_backbone


def test_convnext_backbone():
    """Test ConvNeXt backbone initialization and forward pass."""
    
    print("=" * 70)
    print("Testing ConvNeXt Backbone Integration")
    print("=" * 70)
    
    # Setup config
    cfg = get_cfg()
    add_dimaskdino_config(cfg)
    
    # Configure ConvNeXt-Tiny
    cfg.MODEL.BACKBONE.NAME = "D2ConvNeXt"
    cfg.MODEL.CONVNEXT.DEPTHS = [3, 3, 9, 3]
    cfg.MODEL.CONVNEXT.DIMS = [96, 192, 384, 768]
    cfg.MODEL.CONVNEXT.DROP_PATH_RATE = 0.1
    cfg.MODEL.CONVNEXT.LAYER_SCALE_INIT_VALUE = 1e-6
    cfg.MODEL.CONVNEXT.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    
    print("\n1. Building ConvNeXt backbone...")
    try:
        backbone = build_backbone(cfg)
        print("   ✓ Backbone built successfully!")
        print(f"   Backbone type: {type(backbone).__name__}")
    except Exception as e:
        print(f"   ✗ Failed to build backbone: {e}")
        return False
    
    # Test output shape
    print("\n2. Testing output shapes...")
    try:
        dummy_input = torch.randn(2, 3, 1024, 1024)  # Batch size 2, 1024x1024 images
        print(f"   Input shape: {dummy_input.shape}")
        
        with torch.no_grad():
            outputs = backbone(dummy_input)
        
        print("   Output feature maps:")
        for name, tensor in outputs.items():
            print(f"     {name}: {tensor.shape}")
        
        # Verify output shapes
        expected_shapes = {
            "res2": (2, 96, 256, 256),    # stride 4
            "res3": (2, 192, 128, 128),   # stride 8
            "res4": (2, 384, 64, 64),     # stride 16
            "res5": (2, 768, 32, 32),     # stride 32
        }
        
        all_correct = True
        for name, expected_shape in expected_shapes.items():
            actual_shape = tuple(outputs[name].shape)
            if actual_shape == expected_shape:
                print(f"   ✓ {name} shape correct: {actual_shape}")
            else:
                print(f"   ✗ {name} shape mismatch: expected {expected_shape}, got {actual_shape}")
                all_correct = False
        
        if not all_correct:
            return False
            
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test output_shape() method
    print("\n3. Testing output_shape() method...")
    try:
        output_shapes = backbone.output_shape()
        print("   Output shape specifications:")
        for name, shape_spec in output_shapes.items():
            print(f"     {name}: channels={shape_spec.channels}, stride={shape_spec.stride}")
        
        # Verify shape specs
        expected_specs = {
            "res2": (96, 4),
            "res3": (192, 8),
            "res4": (384, 16),
            "res5": (768, 32),
        }
        
        all_correct = True
        for name, (exp_channels, exp_stride) in expected_specs.items():
            spec = output_shapes[name]
            if spec.channels == exp_channels and spec.stride == exp_stride:
                print(f"   ✓ {name} spec correct")
            else:
                print(f"   ✗ {name} spec mismatch: expected (channels={exp_channels}, stride={exp_stride}), "
                      f"got (channels={spec.channels}, stride={spec.stride})")
                all_correct = False
        
        if not all_correct:
            return False
            
    except Exception as e:
        print(f"   ✗ output_shape() failed: {e}")
        return False
    
    # Count parameters
    print("\n4. Model statistics...")
    try:
        total_params = sum(p.numel() for p in backbone.parameters())
        trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    except Exception as e:
        print(f"   ✗ Failed to count parameters: {e}")
        return False
    
    # Test different model sizes
    print("\n5. Testing different ConvNeXt variants...")
    variants = {
        "ConvNeXt-Tiny": {"depths": [3, 3, 9, 3], "dims": [96, 192, 384, 768]},
        "ConvNeXt-Small": {"depths": [3, 3, 27, 3], "dims": [96, 192, 384, 768]},
        "ConvNeXt-Base": {"depths": [3, 3, 27, 3], "dims": [128, 256, 512, 1024]},
    }
    
    for variant_name, variant_cfg in variants.items():
        try:
            test_cfg = get_cfg()
            add_dimaskdino_config(test_cfg)
            test_cfg.MODEL.BACKBONE.NAME = "D2ConvNeXt"
            test_cfg.MODEL.CONVNEXT.DEPTHS = variant_cfg["depths"]
            test_cfg.MODEL.CONVNEXT.DIMS = variant_cfg["dims"]
            test_cfg.MODEL.CONVNEXT.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
            
            test_backbone = build_backbone(test_cfg)
            params = sum(p.numel() for p in test_backbone.parameters())
            print(f"   ✓ {variant_name}: {params:,} parameters")
        except Exception as e:
            print(f"   ✗ {variant_name} failed: {e}")
    
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("ConvNeXt backbone is correctly integrated.")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    success = test_convnext_backbone()
    sys.exit(0 if success else 1)
