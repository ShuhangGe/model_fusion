#!/usr/bin/env python3

"""
Basic Usage Example for GLM-4.1V-Thinking Model

This script demonstrates how to load and use the GLM-4V model components.
"""

import torch
from PIL import Image
import numpy as np

# Import GLM-4V components
try:
    from glm4v import (
        Glm4vConfig,
        Glm4vForConditionalGeneration,
        Glm4vImageProcessor,
        Glm4vVideoProcessor,
    )
    print("✓ Successfully imported GLM-4V components")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure you're in the GLM-4.1V-Thinking_model directory and dependencies are installed")
    exit(1)


def test_configuration():
    """Test configuration creation and access."""
    print("\n=== Testing Configuration ===")
    
    # Create default configuration
    config = Glm4vConfig()
    print(f"Model type: {config.model_type}")
    print(f"Vision hidden size: {config.vision_config.hidden_size}")
    print(f"Text hidden size: {config.text_config.hidden_size}")
    print(f"Image token ID: {config.image_token_id}")
    print(f"Video token ID: {config.video_token_id}")
    
    # Create custom configuration
    custom_config = Glm4vConfig(
        vision_config={
            "hidden_size": 1024,
            "depth": 12,
            "num_heads": 8,
        },
        text_config={
            "hidden_size": 2048,
            "num_hidden_layers": 20,
        }
    )
    print(f"\nCustom vision hidden size: {custom_config.vision_config.hidden_size}")
    print(f"Custom text hidden size: {custom_config.text_config.hidden_size}")


def test_model_creation():
    """Test model instantiation."""
    print("\n=== Testing Model Creation ===")
    
    # Create a smaller configuration for testing
    config = Glm4vConfig(
        vision_config={
            "hidden_size": 256,
            "depth": 2,
            "num_heads": 4,
            "image_size": 224,
            "patch_size": 16,
        },
        text_config={
            "hidden_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 512,
        }
    )
    
    try:
        # Create model
        model = Glm4vForConditionalGeneration(config)
        print(f"✓ Model created successfully")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test model components
        print(f"Vision model type: {type(model.model.visual).__name__}")
        print(f"Language model type: {type(model.model.language_model).__name__}")
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")


def test_image_processing():
    """Test image processing functionality."""
    print("\n=== Testing Image Processing ===")
    
    try:
        # Create image processor
        image_processor = Glm4vImageProcessor(
            patch_size=16,
            temporal_patch_size=2,
            merge_size=2,
        )
        print("✓ Image processor created")
        
        # Create a dummy image
        dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        print(f"Dummy image size: {dummy_image.size}")
        
        # Process the image
        processed = image_processor(images=dummy_image, return_tensors="pt")
        print(f"✓ Image processed successfully")
        print(f"Pixel values shape: {processed['pixel_values'].shape}")
        print(f"Image grid THW: {processed['image_grid_thw']}")
        
    except Exception as e:
        print(f"✗ Image processing failed: {e}")


def test_video_processing():
    """Test video processing functionality."""
    print("\n=== Testing Video Processing ===")
    
    try:
        # Create video processor
        video_processor = Glm4vVideoProcessor(
            patch_size=16,
            temporal_patch_size=2,
            merge_size=2,
        )
        print("✓ Video processor created")
        
        # Note: Video processing typically requires more complex setup
        # This is a simplified test
        print("Note: Full video processing requires video metadata and proper tensor setup")
        print("See the video_processing.py example for more details")
        
    except Exception as e:
        print(f"✗ Video processor creation failed: {e}")


def test_smart_resize():
    """Test the smart resize functionality."""
    print("\n=== Testing Smart Resize ===")
    
    try:
        from glm4v.image_processing_glm4v import smart_resize
        
        # Test smart resize with different dimensions
        test_cases = [
            (480, 640),
            (224, 224),
            (720, 1280),
            (336, 336),
        ]
        
        for height, width in test_cases:
            new_h, new_w = smart_resize(
                num_frames=2,
                height=height,
                width=width,
                temporal_factor=2,
                factor=32,  # patch_size * merge_size
            )
            print(f"{height}x{width} → {new_h}x{new_w}")
            
    except Exception as e:
        print(f"✗ Smart resize test failed: {e}")


def main():
    """Run all tests."""
    print("GLM-4.1V-Thinking Model - Basic Usage Test")
    print("=" * 50)
    
    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    # Run tests
    test_configuration()
    test_model_creation()
    test_image_processing()
    test_video_processing()
    test_smart_resize()
    
    print("\n" + "=" * 50)
    print("Basic usage test completed!")
    print("\nNext steps:")
    print("1. Install a compatible tokenizer for text processing")
    print("2. Load pre-trained weights for actual inference")
    print("3. See other examples for more advanced usage")


if __name__ == "__main__":
    main() 