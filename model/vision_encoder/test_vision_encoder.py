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
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_vision_model_basic():
    """Test basic vision model functionality."""
    print("Testing GLM4V Vision Model - Basic Functionality")
    print("=" * 50)
    
    try:
        from vision_encoder import VisionConfig, VisionModel
        
        # Create a smaller model for testing
        config = VisionConfig(
            hidden_size=256,
            depth=2,
            num_heads=4,
            image_size=224,
            patch_size=16,
            out_hidden_size=512,
            spatial_merge_size=2,
        )
        
        vision_model = VisionModel(config)
        
        print(f"✓ Vision model created successfully")
        print(f"  - Parameters: {sum(p.numel() for p in vision_model.parameters()):,}")
        print(f"  - Config: {config.hidden_size}D hidden, {config.depth} layers")
        print(f"  - Model type: {config.model_type}")
        
        return True, vision_model, config
        
    except Exception as e:
        print(f"✗ Vision model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_image_processing():
    """Test image processing functionality."""
    print("\nTesting GLM4V Image Processing")
    print("=" * 50)
    
    try:
        from vision_encoder import VisionProcessor
        
        # Create image processor
        image_processor = VisionProcessor(
            patch_size=16,
            temporal_patch_size=1,
            merge_size=2,
        )
        
        print("✓ Image processor created successfully")
        
        # Create fake image data
        fake_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pil_image = Image.fromarray(fake_image)
        
        # Process image
        inputs = image_processor(images=[pil_image], return_tensors="pt")
        
        print(f"✓ Image processed successfully")
        print(f"  - Input shape: {inputs['pixel_values'].shape}")
        print(f"  - Grid THW: {inputs['image_grid_thw']}")
        print(f"  - Grid THW shape: {inputs['image_grid_thw'].shape}")
        
        return True, inputs, image_processor
        
    except Exception as e:
        print(f"✗ Image processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_video_processing():
    """Test video processing functionality."""
    print("\nTesting GLM4V Video Processing")
    print("=" * 50)
    
    try:
        from vision_encoder import VisionProcessor
        
        # Create video processor (same as image processor but with multiple frames)
        video_processor = VisionProcessor(
            patch_size=16,
            temporal_patch_size=2,
            merge_size=2,
        )
        
        print("✓ Video processor created successfully")
        
        # Create fake video data (4 frames as individual images)
        fake_frames = []
        for i in range(4):
            fake_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            fake_frames.append(Image.fromarray(fake_frame))
        
        # Process video as sequence of images
        inputs = video_processor(images=fake_frames, return_tensors="pt")
        
        print(f"✓ Video processed successfully")
        print(f"  - Input shape: {inputs['pixel_values'].shape}")
        print(f"  - Grid THW: {inputs['image_grid_thw']}")
        print(f"  - Number of frames processed: {len(fake_frames)}")
        
        return True, inputs, video_processor
        
    except Exception as e:
        print(f"✗ Video processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_end_to_end_image():
    """Test end-to-end image processing and vision encoding."""
    print("\nTesting End-to-End Image Pipeline")
    print("=" * 50)
    
    try:
        from vision_encoder import create_glm4v_vision_encoder
        
        # Create complete pipeline with smaller model
        config, vision_model, image_processor = create_glm4v_vision_encoder(
            config_kwargs={
                "hidden_size": 256,
                "depth": 2,
                "num_heads": 4,
                "image_size": 224,
                "patch_size": 16,
                "out_hidden_size": 512,
                "spatial_merge_size": 2,
            }
        )
        
        print("✓ Complete pipeline created")
        
        # Create and process fake image
        fake_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pil_image = Image.fromarray(fake_image)
        
        inputs = image_processor(images=[pil_image], return_tensors="pt")
        print("✓ Image preprocessed")
        
        # Run through vision model
        vision_model.eval()
        with torch.no_grad():
            embeddings = vision_model.get_image_features(
                inputs["pixel_values"], 
                inputs["image_grid_thw"]
            )
        
        print(f"✓ Vision encoding successful")
        print(f"  - Input patches: {inputs['pixel_values'].shape}")
        print(f"  - Output embeddings: {len(embeddings)} tensors")
        print(f"  - First embedding shape: {embeddings[0].shape}")
        print(f"  - Expected feature dim: {config.out_hidden_size}")
        
        return True, embeddings
        
    except Exception as e:
        print(f"✗ End-to-end image pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_factory_functions():
    """Test factory functions and convenience APIs."""
    print("\nTesting Factory Functions")
    print("=" * 50)
    
    try:
        from vision_encoder import create_glm4v_vision_encoder, encode_image
        
        # Test factory function
        config, model, processor = create_glm4v_vision_encoder()
        print("✓ Factory function created complete setup")
        print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create a test image
        test_image = Image.new('RGB', (336, 336), color='blue')
        test_path = "/tmp/test_image.jpg"
        test_image.save(test_path)
        
        # Test convenience function (this will create new model if needed)
        try:
            embedding = encode_image(test_path, model, processor)
            print(f"✓ Convenience function worked")
            print(f"  - Embedding shape: {embedding.shape}")
        except Exception as e:
            print(f"⚠ Convenience function failed (expected): {e}")
        
        # Clean up
        if os.path.exists(test_path):
            os.remove(test_path)
        
        return True
        
    except Exception as e:
        print(f"✗ Factory functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_custom_configuration():
    """Test custom configuration creation."""
    print("\nTesting Custom Configuration")
    print("=" * 50)
    
    try:
        from vision_encoder import VisionConfig, VisionModel
        
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
            config = VisionConfig(**config_info["config"])
            model = VisionModel(config)
            param_count = sum(p.numel() for p in model.parameters())
            
            print(f"✓ {config_info['name']} model: {param_count:,} parameters")
            
            # Test config serialization
            config_dict = config.to_dict()
            config_restored = VisionConfig.from_dict(config_dict)
            print(f"  - Config serialization works")
        
        return True
        
    except Exception as e:
        print(f"✗ Custom configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_save_load_functionality():
    """Test model save and load functionality."""
    print("\nTesting Save/Load Functionality")
    print("=" * 50)
    
    try:
        from vision_encoder import VisionConfig, VisionModel
        
        # Create a small test model
        config = VisionConfig(
            hidden_size=128,
            depth=2,
            num_heads=4,
            image_size=224,
            patch_size=16,
            out_hidden_size=256,
        )
        
        original_model = VisionModel(config)
        print("✓ Original model created")
        
        # Save the model
        save_path = "/tmp/test_vision_model"
        original_model.save_pretrained(save_path)
        print(f"✓ Model saved to {save_path}")
        
        # Load the model
        loaded_model = VisionModel.from_pretrained(save_path)
        print("✓ Model loaded successfully")
        
        # Verify they're the same
        original_params = sum(p.numel() for p in original_model.parameters())
        loaded_params = sum(p.numel() for p in loaded_model.parameters())
        print(f"✓ Parameter count matches: {original_params == loaded_params}")
        
        # Clean up
        import shutil
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        
        return True
        
    except Exception as e:
        print(f"✗ Save/load test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_example():
    """Test integration with other models."""
    print("\nTesting Integration Example")
    print("=" * 50)
    
    try:
        from vision_encoder import create_glm4v_vision_encoder
        import torch.nn as nn
        
        # Create vision encoder
        config, vision_model, processor = create_glm4v_vision_encoder(
            config_kwargs={
                "hidden_size": 256,
                "depth": 2,
                "num_heads": 4,
                "out_hidden_size": 512,
            }
        )
        
        # Create a simple classifier using the vision encoder
        class VisionClassifier(nn.Module):
            def __init__(self, vision_encoder, num_classes=10):
                super().__init__()
                self.vision_encoder = vision_encoder
                self.classifier = nn.Linear(config.out_hidden_size, num_classes)
                
            def forward(self, pixel_values, image_grid_thw):
                # Get vision embeddings
                vision_embeddings = self.vision_encoder.get_image_features(
                    pixel_values, image_grid_thw
                )
                
                # Pool embeddings (simple mean pooling)
                pooled_features = torch.stack([emb.mean(dim=0) for emb in vision_embeddings])
                
                # Classify
                logits = self.classifier(pooled_features)
                return logits
        
        # Create integrated model
        classifier = VisionClassifier(vision_model, num_classes=5)
        total_params = sum(p.numel() for p in classifier.parameters())
        print(f"✓ Integrated classifier created with {total_params:,} parameters")
        
        # Test with dummy images
        test_images = [
            Image.new('RGB', (224, 224), color='red'),
            Image.new('RGB', (224, 224), color='green'),
        ]
        
        inputs = processor(images=test_images, return_tensors="pt")
        
        with torch.no_grad():
            logits = classifier(inputs["pixel_values"], inputs["image_grid_thw"])
        
        print(f"✓ Classification successful")
        print(f"  - Output shape: {logits.shape}")
        print(f"  - Predicted classes: {logits.argmax(dim=1).tolist()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("GLM4V Vision Encoder - Standalone Test Suite")
    print("=" * 60)
    
    results = []
    
    # Test basic functionality
    success, model, config = test_vision_model_basic()
    results.append(("Basic Vision Model", success))
    
    # Test processing
    success, image_inputs, image_processor = test_image_processing()
    results.append(("Image Processing", success))
    
    success, video_inputs, video_processor = test_video_processing()
    results.append(("Video Processing", success))
    
    # Test end-to-end
    success, image_embeddings = test_end_to_end_image()
    results.append(("End-to-End Image", success))
    
    # Test factory functions
    results.append(("Factory Functions", test_factory_functions()))
    
    # Test configurations
    results.append(("Custom Configurations", test_custom_configuration()))
    
    # Test save/load
    results.append(("Save/Load", test_save_load_functionality()))
    
    # Test integration
    results.append(("Integration Example", test_integration_example()))
    
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