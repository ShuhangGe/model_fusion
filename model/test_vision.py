#!/usr/bin/env python3
# coding=utf-8
# Copyright 2025 The ZhipuAI Inc. team and HuggingFace Inc. team. All rights reserved.

"""
Test script for GLM4V Vision Encoder

This script tests the standalone GLM4V vision encoder with sample data to ensure
everything works correctly after extraction.
"""

import torch
import numpy as np
from PIL import Image

def test_vision_model_basic():
    """Test basic vision model functionality."""
    print("Testing GLM4V Vision Model - Basic Functionality")
    print("=" * 50)
    
    try:
        from . import create_glm4v_vision_encoder, Glm4vVisionConfig
        
        # Create a smaller model for testing
        config = Glm4vVisionConfig(
            hidden_size=256,
            depth=2,
            num_heads=4,
            image_size=224,
            patch_size=16,
            out_hidden_size=512,
            spatial_merge_size=2,
        )
        
        vision_model = create_glm4v_vision_encoder(
            hidden_size=256,
            depth=2,
            num_heads=4,
            image_size=224,
            patch_size=16,
            out_hidden_size=512,
        )
        
        print(f"✓ Vision model created successfully")
        print(f"  - Parameters: {sum(p.numel() for p in vision_model.parameters()):,}")
        print(f"  - Config: {config.hidden_size}D hidden, {config.depth} layers")
        
        return True
        
    except Exception as e:
        print(f"✗ Vision model creation failed: {e}")
        return False


def test_image_processing():
    """Test image processing functionality."""
    print("\nTesting GLM4V Image Processing")
    print("=" * 50)
    
    try:
        from . import create_glm4v_image_processor
        
        # Create image processor
        image_processor = create_glm4v_image_processor(
            patch_size=16,
            temporal_patch_size=1,
            merge_size=2,
        )
        
        print("✓ Image processor created successfully")
        
        # Create fake image data
        fake_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pil_image = Image.fromarray(fake_image)
        
        # Process image
        inputs = image_processor.preprocess([pil_image], return_tensors="pt")
        
        print(f"✓ Image processed successfully")
        print(f"  - Input shape: {inputs['pixel_values'].shape}")
        print(f"  - Grid THW: {inputs['image_grid_thw']}")
        
        return inputs
        
    except Exception as e:
        print(f"✗ Image processing failed: {e}")
        return None


def test_video_processing():
    """Test video processing functionality."""
    print("\nTesting GLM4V Video Processing")
    print("=" * 50)
    
    try:
        from . import create_glm4v_video_processor, VideoMetadata
        
        # Create video processor
        video_processor = create_glm4v_video_processor(
            patch_size=16,
            temporal_patch_size=2,
            merge_size=2,
            fps=2.0,
        )
        
        print("✓ Video processor created successfully")
        
        # Create fake video data (8 frames, 3 channels, 224x224)
        fake_video = torch.randn(8, 3, 224, 224)
        metadata = VideoMetadata(fps=30.0, total_num_frames=8, duration=8/30)
        
        # Process video
        inputs = video_processor.preprocess(
            [fake_video], 
            video_metadata=[metadata], 
            return_tensors="pt"
        )
        
        print(f"✓ Video processed successfully")
        print(f"  - Input shape: {inputs['pixel_values_videos'].shape}")
        print(f"  - Grid THW: {len(inputs['video_grid_thw'])} frames")
        print(f"  - Timestamps: {inputs['timestamps']}")
        
        return inputs
        
    except Exception as e:
        print(f"✗ Video processing failed: {e}")
        return None


def test_end_to_end_image():
    """Test end-to-end image processing and vision encoding."""
    print("\nTesting End-to-End Image Pipeline")
    print("=" * 50)
    
    try:
        from . import create_complete_glm4v_vision_pipeline
        
        # Create complete pipeline with smaller model
        vision_model, image_processor, _ = create_complete_glm4v_vision_pipeline(
            hidden_size=256,
            depth=2,
            num_heads=4,
            image_size=224,
            patch_size=16,
            out_hidden_size=512,
        )
        
        print("✓ Complete pipeline created")
        
        # Create and process fake image
        fake_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pil_image = Image.fromarray(fake_image)
        
        inputs = image_processor.preprocess([pil_image], return_tensors="pt")
        print("✓ Image preprocessed")
        
        # Run through vision model
        vision_model.eval()
        with torch.no_grad():
            embeddings = vision_model(inputs["pixel_values"], inputs["image_grid_thw"])
        
        print(f"✓ Vision encoding successful")
        print(f"  - Input patches: {inputs['pixel_values'].shape}")
        print(f"  - Output embeddings: {embeddings.shape}")
        print(f"  - Expected output dim: 512")
        
        return embeddings
        
    except Exception as e:
        print(f"✗ End-to-end image pipeline failed: {e}")
        return None


def test_end_to_end_video():
    """Test end-to-end video processing and vision encoding."""
    print("\nTesting End-to-End Video Pipeline")
    print("=" * 50)
    
    try:
        from . import create_complete_glm4v_vision_pipeline, VideoMetadata
        
        # Create complete pipeline
        vision_model, _, video_processor = create_complete_glm4v_vision_pipeline(
            hidden_size=256,
            depth=2,
            num_heads=4,
            image_size=224,
            patch_size=16,
            out_hidden_size=512,
        )
        
        print("✓ Complete pipeline created")
        
        # Create and process fake video
        fake_video = torch.randn(8, 3, 224, 224)
        metadata = VideoMetadata(fps=30.0, total_num_frames=8)
        
        inputs = video_processor.preprocess(
            [fake_video], 
            video_metadata=[metadata], 
            return_tensors="pt"
        )
        print("✓ Video preprocessed")
        
        # Run through vision model
        vision_model.eval()
        with torch.no_grad():
            embeddings = vision_model(inputs["pixel_values_videos"], inputs["video_grid_thw"])
        
        print(f"✓ Video vision encoding successful")
        print(f"  - Input video patches: {inputs['pixel_values_videos'].shape}")
        print(f"  - Output embeddings: {embeddings.shape}")
        
        return embeddings
        
    except Exception as e:
        print(f"✗ End-to-end video pipeline failed: {e}")
        return None


def test_custom_configuration():
    """Test custom configuration creation."""
    print("\nTesting Custom Configuration")
    print("=" * 50)
    
    try:
        from . import Glm4vVisionConfig, Glm4vVisionModel
        
        # Test various configurations
        configs = [
            {
                "name": "Tiny",
                "config": {
                    "hidden_size": 128,
                    "depth": 1,
                    "num_heads": 2,
                    "image_size": 224,
                    "patch_size": 32,
                    "out_hidden_size": 256,
                }
            },
            {
                "name": "Small", 
                "config": {
                    "hidden_size": 384,
                    "depth": 6,
                    "num_heads": 6,
                    "image_size": 224,
                    "patch_size": 16,
                    "out_hidden_size": 768,
                }
            },
            {
                "name": "Medium",
                "config": {
                    "hidden_size": 768,
                    "depth": 12,
                    "num_heads": 12,
                    "image_size": 336,
                    "patch_size": 14,
                    "out_hidden_size": 1024,
                }
            }
        ]
        
        for config_info in configs:
            config = Glm4vVisionConfig(**config_info["config"])
            model = Glm4vVisionModel(config)
            param_count = sum(p.numel() for p in model.parameters())
            
            print(f"✓ {config_info['name']} model: {param_count:,} parameters")
        
        return True
        
    except Exception as e:
        print(f"✗ Custom configuration test failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("GLM4V Vision Encoder - Standalone Test Suite")
    print("=" * 60)
    
    results = []
    
    # Test basic functionality
    results.append(("Basic Vision Model", test_vision_model_basic()))
    
    # Test processing
    image_inputs = test_image_processing()
    results.append(("Image Processing", image_inputs is not None))
    
    video_inputs = test_video_processing() 
    results.append(("Video Processing", video_inputs is not None))
    
    # Test end-to-end
    image_embeddings = test_end_to_end_image()
    results.append(("End-to-End Image", image_embeddings is not None))
    
    video_embeddings = test_end_to_end_video()
    results.append(("End-to-End Video", video_embeddings is not None))
    
    # Test configurations
    results.append(("Custom Configurations", test_custom_configuration()))
    
    # Summary
    print("\nTest Results Summary")
    print("=" * 60)
    
    passed = 0
    for test_name, passed_test in results:
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"{test_name:<25} {status}")
        if passed_test:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! GLM4V Vision Encoder is working correctly.")
        return True
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1) 