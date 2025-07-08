# coding=utf-8
# Copyright 2025 The ZhipuAI Inc. team and HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example Usage of the Standalone GLM4V Vision Encoder

This file demonstrates various ways to use the extracted GLM4V vision encoder
for processing images and videos independently of the text model.
"""

import torch
import numpy as np
from PIL import Image
import time
import sys
import os

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision_encoder import (
    VisionConfig,
    VisionModel,
    VisionProcessor,
    create_glm4v_vision_encoder,
    encode_image,
    encode_images,
)


def example_basic_usage():
    """
    Demonstrate basic usage of the vision encoder.
    """
    print("=" * 50)
    print("BASIC USAGE EXAMPLE")
    print("=" * 50)
    
    # Create configuration
    config = VisionConfig(
        hidden_size=1536,
        depth=24,
        num_heads=12,
        patch_size=14,
        image_size=336
    )
    print(f"Created config: {config.model_type}")
    print(f"Hidden size: {config.hidden_size}, Depth: {config.depth}")
    
    # Create vision model
    model = VisionModel(config)
    print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create processor
    processor = VisionProcessor(
        patch_size=config.patch_size,
        temporal_patch_size=config.temporal_patch_size,
        merge_size=config.spatial_merge_size
    )
    print(f"Created processor with patch size: {processor.patch_size}")
    
    # Create a dummy image for testing
    dummy_image = Image.new('RGB', (336, 336), color='red')
    print(f"Created dummy image: {dummy_image.size}")
    
    # Process the image
    inputs = processor(images=[dummy_image], return_tensors="pt")
    print(f"Processed inputs keys: {list(inputs.keys())}")
    print(f"Pixel values shape: {inputs['pixel_values'].shape}")
    print(f"Image grid thw: {inputs['image_grid_thw']}")
    
    # Generate embeddings
    with torch.no_grad():
        embeddings = model.get_image_features(
            inputs["pixel_values"],
            inputs["image_grid_thw"]
        )
    
    print(f"Generated embeddings: {len(embeddings)} tensors")
    print(f"First embedding shape: {embeddings[0].shape}")
    print()


def example_factory_functions():
    """
    Demonstrate using factory functions for simplified setup.
    """
    print("=" * 50)
    print("FACTORY FUNCTIONS EXAMPLE")
    print("=" * 50)
    
    # Create complete vision encoder setup
    config, model, processor = create_glm4v_vision_encoder()
    print("Created complete vision encoder setup using factory function")
    
    # Create dummy images
    images = [
        Image.new('RGB', (224, 224), color='red'),
        Image.new('RGB', (336, 336), color='green'),
        Image.new('RGB', (448, 448), color='blue'),
    ]
    print(f"Created {len(images)} test images with different sizes")
    
    # Process multiple images
    inputs = processor(images=images, return_tensors="pt")
    print(f"Processed {len(images)} images")
    print(f"Pixel values shape: {inputs['pixel_values'].shape}")
    print(f"Image grid thw shape: {inputs['image_grid_thw'].shape}")
    
    # Generate embeddings for all images
    with torch.no_grad():
        embeddings = model.get_image_features(
            inputs["pixel_values"],
            inputs["image_grid_thw"]
        )
    
    print(f"Generated {len(embeddings)} embedding tensors")
    for i, emb in enumerate(embeddings):
        print(f"  Image {i+1} embedding shape: {emb.shape}")
    print()


def example_custom_configuration():
    """
    Demonstrate creating custom configurations for different use cases.
    """
    print("=" * 50)
    print("CUSTOM CONFIGURATION EXAMPLE")
    print("=" * 50)
    
    # Create a smaller model for faster inference
    small_config = VisionConfig(
        hidden_size=768,  # Smaller hidden size
        depth=12,         # Fewer layers
        num_heads=8,      # Fewer attention heads
        patch_size=16,    # Larger patches for faster processing
        image_size=224,   # Smaller input size
    )
    print("Created small configuration:")
    print(f"  Hidden size: {small_config.hidden_size}")
    print(f"  Depth: {small_config.depth}")
    print(f"  Patch size: {small_config.patch_size}")
    
    small_model = VisionModel(small_config)
    param_count_small = sum(p.numel() for p in small_model.parameters())
    print(f"Small model parameters: {param_count_small:,}")
    
    # Create a larger model for better quality
    large_config = VisionConfig(
        hidden_size=2048,  # Larger hidden size
        depth=32,          # More layers
        num_heads=16,      # More attention heads
        patch_size=12,     # Smaller patches for finer detail
        image_size=448,    # Larger input size
    )
    print("\nCreated large configuration:")
    print(f"  Hidden size: {large_config.hidden_size}")
    print(f"  Depth: {large_config.depth}")
    print(f"  Patch size: {large_config.patch_size}")
    
    large_model = VisionModel(large_config)
    param_count_large = sum(p.numel() for p in large_model.parameters())
    print(f"Large model parameters: {param_count_large:,}")
    
    print(f"\nParameter ratio (large/small): {param_count_large / param_count_small:.2f}x")
    print()


def example_save_and_load():
    """
    Demonstrate saving and loading models.
    """
    print("=" * 50)
    print("SAVE AND LOAD EXAMPLE")
    print("=" * 50)
    
    # Create and configure a model
    config = VisionConfig(hidden_size=512, depth=6)
    model = VisionModel(config)
    print("Created test model")
    
    # Save the model
    save_path = "/tmp/test_vision_model"
    model.save_pretrained(save_path)
    print(f"Saved model to: {save_path}")
    
    # Load the model
    loaded_model = VisionModel.from_pretrained(save_path)
    print("Loaded model from disk")
    
    # Verify they're the same
    original_params = sum(p.numel() for p in model.parameters())
    loaded_params = sum(p.numel() for p in loaded_model.parameters())
    print(f"Original model parameters: {original_params}")
    print(f"Loaded model parameters: {loaded_params}")
    print(f"Parameters match: {original_params == loaded_params}")
    
    # Clean up
    import shutil
    shutil.rmtree(save_path)
    print("Cleaned up temporary files")
    print()


def example_performance_benchmark():
    """
    Demonstrate performance characteristics of the vision encoder.
    """
    print("=" * 50)
    print("PERFORMANCE BENCHMARK")
    print("=" * 50)
    
    config, model, processor = create_glm4v_vision_encoder()
    
    # Test with different image sizes
    image_sizes = [(224, 224), (336, 336), (448, 448)]
    
    for width, height in image_sizes:
        print(f"\nTesting with image size: {width}x{height}")
        
        # Create test image
        image = Image.new('RGB', (width, height), color='blue')
        
        # Time the preprocessing
        start_time = time.time()
        inputs = processor(images=[image], return_tensors="pt")
        preprocess_time = time.time() - start_time
        
        # Time the model inference
        start_time = time.time()
        with torch.no_grad():
            embeddings = model.get_image_features(
                inputs["pixel_values"],
                inputs["image_grid_thw"]
            )
        inference_time = time.time() - start_time
        
        # Calculate number of patches
        patches_h = height // (processor.patch_size * processor.merge_size) 
        patches_w = width // (processor.patch_size * processor.merge_size)
        total_patches = patches_h * patches_w
        
        print(f"  Preprocessing time: {preprocess_time:.3f}s")
        print(f"  Inference time: {inference_time:.3f}s")
        print(f"  Total patches: {total_patches}")
        print(f"  Embedding shape: {embeddings[0].shape}")
        print(f"  Memory usage: {embeddings[0].numel() * 4 / 1024 / 1024:.2f} MB")
    print()


def example_integration_with_other_models():
    """
    Demonstrate how to integrate the vision encoder with other models.
    """
    print("=" * 50)
    print("INTEGRATION EXAMPLE")
    print("=" * 50)
    
    config, vision_model, processor = create_glm4v_vision_encoder()
    
    # Simulate a simple classifier that uses vision embeddings
    class SimpleClassifier(torch.nn.Module):
        def __init__(self, vision_encoder, num_classes=10):
            super().__init__()
            self.vision_encoder = vision_encoder
            self.classifier = torch.nn.Linear(config.out_hidden_size, num_classes)
            
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
    classifier = SimpleClassifier(vision_model, num_classes=5)
    total_params = sum(p.numel() for p in classifier.parameters())
    print(f"Created integrated classifier with {total_params:,} parameters")
    
    # Test with dummy images
    test_images = [
        Image.new('RGB', (336, 336), color='red'),
        Image.new('RGB', (336, 336), color='green'),
    ]
    
    inputs = processor(images=test_images, return_tensors="pt")
    
    with torch.no_grad():
        logits = classifier(inputs["pixel_values"], inputs["image_grid_thw"])
    
    print(f"Classification output shape: {logits.shape}")
    print(f"Predicted classes: {logits.argmax(dim=1).tolist()}")
    print()


def main():
    """
    Run all examples.
    """
    print("GLM4V Standalone Vision Encoder Examples")
    print("========================================")
    print()
    
    try:
        example_basic_usage()
        example_factory_functions()
        example_custom_configuration()
        example_save_and_load()
        example_performance_benchmark()
        example_integration_with_other_models()
        
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 