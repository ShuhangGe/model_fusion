#!/usr/bin/env python3
# coding=utf-8
# Copyright 2025 The ZhipuAI Inc. team and HuggingFace Inc. team. All rights reserved.

"""
Example usage of the GLM4V Embedding Pipeline.

This script demonstrates how to use the extracted GLM4V embedding pipeline
to process images and videos and extract embeddings.
"""

import torch
import numpy as np
from PIL import Image

# Import the embedding pipeline
from . import GLM4VEmbeddingPipeline, Glm4vVisionConfig


def example_image_processing():
    """Example of processing images through the embedding pipeline."""
    print("=== Image Processing Example ===")
    
    # Initialize the pipeline
    pipeline = GLM4VEmbeddingPipeline()
    pipeline.eval()  # Set to evaluation mode
    
    # Create a dummy image (you would load real images in practice)
    dummy_image = Image.new('RGB', (336, 336), color='red')
    
    # Process the image
    try:
        result = pipeline.extract_image_embeddings([dummy_image], return_dict=True)
        
        print(f"✓ Image processed successfully!")
        print(f"  Embeddings shape: {result['embeddings'].shape}")
        print(f"  Number of patches: {result['num_patches']}")
        print(f"  Embedding dimension: {result['embedding_dim']}")
        print(f"  Image grid (t,h,w): {result['image_grid_thw']}")
        
    except Exception as e:
        print(f"✗ Error processing image: {e}")


def example_video_processing():
    """Example of processing videos through the embedding pipeline."""
    print("\n=== Video Processing Example ===")
    
    # Initialize the pipeline
    pipeline = GLM4VEmbeddingPipeline()
    pipeline.eval()
    
    # Create dummy video data (you would load real video in practice)
    video_tensor = torch.randn(16, 3, 224, 224)  # 16 frames, 3 channels, 224x224
    video_metadata = {
        "fps": 30.0,
        "duration": 0.533,
        "total_num_frames": 16
    }
    
    try:
        result = pipeline.extract_video_embeddings(
            video_tensor,
            video_metadata=[video_metadata],
            return_dict=True
        )
        
        print(f"✓ Video processed successfully!")
        print(f"  Embeddings shape: {result['embeddings'].shape}")
        print(f"  Number of patches: {result['num_patches']}")
        print(f"  Embedding dimension: {result['embedding_dim']}")
        print(f"  Video grid (t,h,w): {len(result['video_grid_thw'])}")
        print(f"  Timestamps: {result['timestamps']}")
        
    except Exception as e:
        print(f"✗ Error processing video: {e}")


def example_custom_configuration():
    """Example of using custom configuration."""
    print("\n=== Custom Configuration Example ===")
    
    # Create custom configuration
    config = Glm4vVisionConfig(
        hidden_size=1536,
        depth=24,
        num_heads=12,
        patch_size=14,
        spatial_merge_size=2,
        out_hidden_size=4096,
    )
    
    # Initialize pipeline with custom config
    pipeline = GLM4VEmbeddingPipeline(config=config)
    pipeline.eval()
    
    print(f"✓ Custom pipeline initialized!")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Depth: {config.depth}")
    print(f"  Number of heads: {config.num_heads}")
    print(f"  Patch size: {config.patch_size}")
    print(f"  Output dimension: {config.out_hidden_size}")


def example_batch_processing():
    """Example of processing multiple images in batch."""
    print("\n=== Batch Processing Example ===")
    
    pipeline = GLM4VEmbeddingPipeline()
    pipeline.eval()
    
    # Create multiple dummy images
    images = [
        Image.new('RGB', (224, 224), color='red'),
        Image.new('RGB', (336, 336), color='green'),
        Image.new('RGB', (448, 448), color='blue'),
    ]
    
    try:
        results = []
        for i, image in enumerate(images):
            result = pipeline.extract_image_embeddings([image], return_dict=True)
            results.append(result)
            print(f"  Image {i+1}: {result['embeddings'].shape} embeddings")
        
        print(f"✓ Processed {len(images)} images in batch!")
        
    except Exception as e:
        print(f"✗ Error in batch processing: {e}")


def example_device_management():
    """Example of managing GPU/CPU devices."""
    print("\n=== Device Management Example ===")
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize pipeline
    pipeline = GLM4VEmbeddingPipeline()
    
    # Move to device
    pipeline = pipeline.to(device)
    pipeline.eval()
    
    print(f"✓ Pipeline moved to {pipeline.device}")
    
    # Create dummy image
    dummy_image = Image.new('RGB', (224, 224), color='purple')
    
    try:
        result = pipeline.extract_image_embeddings([dummy_image])
        print(f"✓ Processing successful on {device}")
        print(f"  Output device: {result.device}")
        
    except Exception as e:
        print(f"✗ Error with device {device}: {e}")


def main():
    """Run all examples."""
    print("GLM4V Embedding Pipeline Examples")
    print("="*50)
    
    # Run examples
    example_image_processing()
    example_video_processing()
    example_custom_configuration()
    example_batch_processing()
    example_device_management()
    
    print("\n" + "="*50)
    print("All examples completed!")


if __name__ == "__main__":
    main() 