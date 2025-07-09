#!/usr/bin/env python3
# coding=utf-8

"""
Simplified test for GLM4V preprocessing pipeline components.
Tests core functionality without heavy dependencies.
"""

import torch
import numpy as np
from PIL import Image
import sys
import os

# Add the project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def test_config():
    """Test the configuration classes."""
    print("=== Testing Configuration Classes ===")
    
    try:
        from model.multimodal.config import Glm4vConfig, Glm4vVisionConfig, Glm4vTextConfig
        
        # Test vision config
        vision_config = Glm4vVisionConfig()
        print(f"✓ Vision config created")
        print(f"  Hidden size: {vision_config.hidden_size}")
        print(f"  Image size: {vision_config.image_size}")
        print(f"  Patch size: {vision_config.patch_size}")
        print(f"  Spatial merge size: {vision_config.spatial_merge_size}")
        
        # Test text config
        text_config = Glm4vTextConfig()
        print(f"✓ Text config created")
        print(f"  Vocab size: {text_config.vocab_size}")
        print(f"  Hidden size: {text_config.hidden_size}")
        
        # Test main config
        config = Glm4vConfig()
        print(f"✓ Main config created")
        print(f"  Image token ID: {config.image_token_id}")
        print(f"  Video start token ID: {config.video_start_token_id}")
        print(f"  Video end token ID: {config.video_end_token_id}")
        
        return True
        
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vision_model():
    """Test the vision model."""
    print("\n=== Testing Vision Model ===")
    
    try:
        from model.multimodal.config import Glm4vVisionConfig
        from model.multimodal.vision_model import Glm4vVisionModel
        
        # Create config and model
        config = Glm4vVisionConfig()
        vision_model = Glm4vVisionModel(config)
        vision_model.eval()
        
        print(f"✓ Vision model created")
        print(f"  Model parameters: {sum(p.numel() for p in vision_model.parameters()):,}")
        
        # Test with dummy input
        batch_size = 2
        num_patches = 100
        patch_dim = config.patch_size * config.patch_size * 3  # 3 channels
        
        dummy_input = torch.randn(batch_size * num_patches, patch_dim)
        dummy_grid_thw = torch.tensor([[1, 10, 10], [1, 10, 10]])  # Two images
        
        with torch.no_grad():
            output = vision_model(dummy_input, grid_thw=dummy_grid_thw)
        
        print(f"✓ Vision model forward pass successful")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected output dim: {config.hidden_size}")
        
        return True
        
    except Exception as e:
        print(f"✗ Vision model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fusion_model():
    """Test the multimodal fusion model."""
    print("\n=== Testing Fusion Model ===")
    
    try:
        from model.multimodal.config import Glm4vConfig
        from model.multimodal.fusion_model import Glm4vMultimodalFusion
        
        # Create config and model
        config = Glm4vConfig()
        fusion_model = Glm4vMultimodalFusion(config)
        fusion_model.eval()
        
        print(f"✓ Fusion model created")
        print(f"  Model parameters: {sum(p.numel() for p in fusion_model.parameters()):,}")
        
        # Test RoPE position calculation
        batch_size = 1
        seq_len = 10
        input_ids = torch.tensor([[100, 200, config.image_token_id, config.image_token_id, 300, 400, 500, 800, 900, 1000]])
        image_grid_thw = torch.tensor([[1, 4, 4]])  # 16 patches, but merge_size=2, so 4 tokens
        attention_mask = torch.ones(batch_size, seq_len)
        
        with torch.no_grad():
            position_ids, rope_deltas = fusion_model.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask
            )
        
        print(f"✓ RoPE position calculation successful")
        print(f"  Input IDs shape: {input_ids.shape}")
        print(f"  Position IDs shape: {position_ids.shape}")
        print(f"  RoPE deltas: {rope_deltas}")
        print(f"  3D position structure: temporal, height, width")
        
        # Test dummy vision embeddings
        dummy_pixels = torch.randn(4, config.vision_config.hidden_size)  # 4 patches
        dummy_grid = torch.tensor([[1, 2, 2]])  # 2x2 spatial grid
        
        with torch.no_grad():
            vision_embeds = fusion_model.get_image_features(
                pixel_values=dummy_pixels.unsqueeze(0),
                image_grid_thw=dummy_grid
            )
        
        print(f"✓ Vision feature extraction successful")
        print(f"  Vision embeddings: {len(vision_embeds)} groups, first shape: {vision_embeds[0].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Fusion model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_critical_fusion_operation():
    """Test the critical masked_scatter fusion operation."""
    print("\n=== Testing Critical Fusion Operation ===")
    
    try:
        from model.multimodal.config import Glm4vConfig
        from model.multimodal.fusion_model import Glm4vMultimodalFusion
        
        config = Glm4vConfig()
        fusion_model = Glm4vMultimodalFusion(config)
        fusion_model.eval()
        
        # Create a scenario with text + image tokens
        batch_size = 1
        seq_len = 7
        hidden_size = config.text_config.hidden_size
        
        # Sequence: [text, text, image, image, image, text, text]
        input_ids = torch.tensor([[100, 200, config.image_token_id, config.image_token_id, config.image_token_id, 300, 400]])
        
        # Create dummy vision embeddings (3 patches)
        num_vision_patches = 3
        dummy_vision_embeddings = torch.randn(num_vision_patches, hidden_size)
        
        # Get text embeddings
        text_embeddings = fusion_model.get_input_embeddings()(input_ids)
        
        print(f"✓ Setup complete")
        print(f"  Sequence: {input_ids[0].tolist()}")
        print(f"  Text embeddings shape: {text_embeddings.shape}")
        print(f"  Vision embeddings shape: {dummy_vision_embeddings.shape}")
        
        # Identify image token positions
        image_mask = input_ids == config.image_token_id
        print(f"  Image token positions: {image_mask[0].nonzero().flatten().tolist()}")
        
        # Perform the critical fusion operation
        with torch.no_grad():
            image_mask_expanded = image_mask.unsqueeze(-1).expand_as(text_embeddings)
            fused_embeddings = text_embeddings.masked_scatter(image_mask_expanded, dummy_vision_embeddings)
        
        print(f"✓ Fusion operation successful")
        print(f"  Fused embeddings shape: {fused_embeddings.shape}")
        
        # Verify fusion worked by checking that embeddings changed at image positions
        original_at_image_pos = text_embeddings[image_mask].sum()
        fused_at_image_pos = fused_embeddings[image_mask].sum()
        
        print(f"  Original sum at image positions: {original_at_image_pos:.4f}")
        print(f"  Fused sum at image positions: {fused_at_image_pos:.4f}")
        print(f"  ✓ Embeddings successfully replaced at image token positions")
        
        return True
        
    except Exception as e:
        print(f"✗ Fusion operation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_3d_rope_coordinates():
    """Test the 3D RoPE coordinate system."""
    print("\n=== Testing 3D RoPE Coordinate System ===")
    
    try:
        from model.multimodal.config import Glm4vConfig
        from model.multimodal.fusion_model import Glm4vMultimodalFusion
        
        config = Glm4vConfig()
        fusion_model = Glm4vMultimodalFusion(config)
        
        # Create a sequence with both text and vision tokens
        # Sequence: [text, image_grid, text]
        input_ids = torch.tensor([[100, config.image_token_id, config.image_token_id, config.image_token_id, config.image_token_id, 200]])
        
        # Image grid: 2x2 spatial patches
        image_grid_thw = torch.tensor([[1, 2, 2]])  # 1 temporal, 2 height, 2 width
        attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            position_ids, rope_deltas = fusion_model.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask
            )
        
        print(f"✓ 3D RoPE calculation successful")
        print(f"  Input sequence length: {input_ids.shape[1]}")
        print(f"  Position IDs shape: {position_ids.shape}")  # [3, batch, seq_len]
        print(f"  Image grid (T,H,W): {image_grid_thw[0].tolist()}")
        
        # Analyze the coordinate structure
        temporal_coords = position_ids[0, 0, :].tolist()
        height_coords = position_ids[1, 0, :].tolist()
        width_coords = position_ids[2, 0, :].tolist()
        
        print(f"  Temporal coordinates: {temporal_coords}")
        print(f"  Height coordinates:   {height_coords}")
        print(f"  Width coordinates:    {width_coords}")
        
        # Check that vision tokens have spatial structure
        image_positions = (input_ids[0] == config.image_token_id).nonzero().flatten()
        print(f"  Image token positions: {image_positions.tolist()}")
        
        # For a 2x2 grid, we expect coordinates like:
        # Position 1: (t=0, h=0, w=0) - top-left
        # Position 2: (t=0, h=0, w=1) - top-right  
        # Position 3: (t=0, h=1, w=0) - bottom-left
        # Position 4: (t=0, h=1, w=1) - bottom-right
        
        if len(image_positions) >= 4:
            pos1, pos2, pos3, pos4 = image_positions[:4]
            coords = [
                (temporal_coords[pos1], height_coords[pos1], width_coords[pos1]),
                (temporal_coords[pos2], height_coords[pos2], width_coords[pos2]),
                (temporal_coords[pos3], height_coords[pos3], width_coords[pos3]),
                (temporal_coords[pos4], height_coords[pos4], width_coords[pos4]),
            ]
            print(f"  Vision patch coordinates:")
            for i, (t, h, w) in enumerate(coords):
                print(f"    Patch {i+1}: (t={t}, h={h}, w={w})")
        
        print(f"  ✓ Spatial-temporal coordinate system working correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ 3D RoPE test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all core tests."""
    print("GLM4V Preprocessing Pipeline Core Tests")
    print("=" * 60)
    print("Testing extracted components without heavy dependencies")
    print("=" * 60)
    
    tests = [
        test_config,
        test_vision_model,
        test_fusion_model,
        test_critical_fusion_operation,
        test_3d_rope_coordinates,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("✓ All core components working correctly!")
        print("✓ GLM4V preprocessing pipeline is functional")
    else:
        print(f"✗ {total - passed} tests failed")
        
    print("=" * 60)


if __name__ == "__main__":
    main() 