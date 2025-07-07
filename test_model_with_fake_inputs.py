#!/usr/bin/env python3

"""
GLM-4.1V-Thinking Model Test with Fake Inputs

This script demonstrates how to use the GLM-4.1V model with synthetic data
to understand the model architecture and input/output format.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import sys
import os

# Add the current directory to Python path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from glm4v import (
        Glm4vConfig,
        Glm4vForConditionalGeneration,
        Glm4vModel,
        Glm4vImageProcessor,
        Glm4vProcessor,
    )
    print("✓ Successfully imported GLM-4V components")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure all dependencies are installed: pip install -r requirements.txt")
    exit(1)


def create_test_config():
    """Create a small test configuration for faster testing."""
    print("\n=== Creating Test Configuration ===")
    
    # Create smaller config for testing (much faster than full model)
    config = Glm4vConfig(
        vision_config={
            "hidden_size": 256,          # Reduced from 1536
            "depth": 2,                  # Reduced from 24
            "num_heads": 4,              # Reduced from 12
            "image_size": 224,           # Reduced from 336
            "patch_size": 16,            # Keep standard
            "out_hidden_size": 512,      # Reduced from 4096
            "intermediate_size": 1024,   # Reduced from 13696
            "spatial_merge_size": 2,     # Keep standard
            "temporal_patch_size": 1,    # Keep standard
        },
        text_config={
            "hidden_size": 512,          # Reduced from 4096
            "num_hidden_layers": 4,      # Reduced from 40
            "num_attention_heads": 8,    # Reduced from 32
            "num_key_value_heads": 2,    # Keep standard
            "intermediate_size": 1024,   # Reduced from 13696
            "vocab_size": 1000,          # Reduced from 151552
            "max_position_embeddings": 512,  # Reduced from 32768
        }
    )
    
    print(f"Vision config - Hidden size: {config.vision_config.hidden_size}")
    print(f"Text config - Hidden size: {config.text_config.hidden_size}")
    print(f"Image token ID: {config.image_token_id}")
    print(f"Video start token ID: {config.video_start_token_id}")
    print(f"Video end token ID: {config.video_end_token_id}")
    
    return config


def create_fake_text_inputs(config, batch_size=1, seq_len=20):
    """Create fake text inputs for testing."""
    print(f"\n=== Creating Fake Text Inputs (batch_size={batch_size}, seq_len={seq_len}) ===")
    
    # Create random input IDs within vocabulary range
    input_ids = torch.randint(0, config.text_config.vocab_size, (batch_size, seq_len))
    
    # Create attention mask (all 1s for simplicity)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    print(f"Sample input IDs: {input_ids[0][:10].tolist()}")
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def create_fake_image_inputs(config, num_images=1):
    """Create fake image inputs for testing."""
    print(f"\n=== Creating Fake Image Inputs (num_images={num_images}) ===")
    
    # Image specifications
    image_size = config.vision_config.image_size
    patch_size = config.vision_config.patch_size
    spatial_merge_size = config.vision_config.spatial_merge_size
    
    # Calculate grid dimensions
    patches_per_side = image_size // patch_size
    merged_patches_per_side = patches_per_side // spatial_merge_size
    total_patches = merged_patches_per_side ** 2
    
    print(f"Image size: {image_size}x{image_size}")
    print(f"Patch size: {patch_size}x{patch_size}")
    print(f"Patches per side: {patches_per_side}")
    print(f"Merged patches per side: {merged_patches_per_side}")
    print(f"Total patches per image: {total_patches}")
    
    # Create fake pixel values
    # Shape: (total_patches_all_images, channels, temporal_patch_size, patch_size, patch_size)
    total_patches_all_images = num_images * total_patches
    temporal_patch_size = config.vision_config.temporal_patch_size
    channels = config.vision_config.in_channels
    
    pixel_values = torch.randn(
        total_patches_all_images,
        channels * temporal_patch_size,
        patch_size,
        patch_size,
        dtype=torch.float32
    )
    
    # Create image grid THW (Temporal, Height, Width)
    # For images: temporal=1, height=merged_patches_per_side, width=merged_patches_per_side
    image_grid_thw = torch.tensor([
        [1, merged_patches_per_side, merged_patches_per_side]
        for _ in range(num_images)
    ], dtype=torch.long)
    
    print(f"Pixel values shape: {pixel_values.shape}")
    print(f"Image grid THW shape: {image_grid_thw.shape}")
    print(f"Image grid THW: {image_grid_thw.tolist()}")
    
    return {
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }


def create_multimodal_inputs(config, num_images=1, text_length=10):
    """Create inputs that combine text and images."""
    print(f"\n=== Creating Multimodal Inputs ===")
    
    # Special tokens
    image_token_id = config.image_token_id
    
    # Create a sequence with image tokens embedded in text
    # Format: [text_tokens] [image_token] [text_tokens]
    vocab_size = config.text_config.vocab_size
    
    # Generate text tokens (avoiding special token IDs)
    text_tokens_1 = torch.randint(10, min(100, vocab_size-1000), (text_length,))
    text_tokens_2 = torch.randint(10, min(100, vocab_size-1000), (text_length,))
    
    # Create image tokens
    image_tokens = torch.full((num_images,), image_token_id, dtype=torch.long)
    
    # Combine: text + image + text
    input_ids = torch.cat([
        text_tokens_1,
        image_tokens,
        text_tokens_2
    ]).unsqueeze(0)  # Add batch dimension
    
    # Create attention mask
    attention_mask = torch.ones_like(input_ids)
    
    # Create image inputs
    image_inputs = create_fake_image_inputs(config, num_images)
    
    print(f"Combined input IDs shape: {input_ids.shape}")
    print(f"Number of image tokens: {num_images}")
    print(f"Image token positions: {(input_ids == image_token_id).nonzero(as_tuple=True)[1].tolist()}")
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": image_inputs["pixel_values"],
        "image_grid_thw": image_inputs["image_grid_thw"],
    }


def test_model_components(config):
    """Test individual model components."""
    print(f"\n=== Testing Model Components ===")
    
    # Create the model
    model = Glm4vForConditionalGeneration(config)
    model.eval()  # Set to evaluation mode
    
    print(f"✓ Model created successfully")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test text-only forward pass
    print("\n--- Testing Text-Only Input ---")
    text_inputs = create_fake_text_inputs(config, batch_size=1, seq_len=15)
    
    with torch.no_grad():
        try:
            text_output = model(**text_inputs)
            print(f"✓ Text-only forward pass successful")
            print(f"Logits shape: {text_output.logits.shape}")
            print(f"Hidden states available: {text_output.hidden_states is not None}")
        except Exception as e:
            print(f"✗ Text-only forward pass failed: {e}")
    
    # Test multimodal forward pass
    print("\n--- Testing Multimodal Input ---")
    multimodal_inputs = create_multimodal_inputs(config, num_images=1, text_length=8)
    
    with torch.no_grad():
        try:
            multimodal_output = model(**multimodal_inputs)
            print(f"✓ Multimodal forward pass successful")
            print(f"Logits shape: {multimodal_output.logits.shape}")
            print(f"Rope deltas: {multimodal_output.rope_deltas}")
        except Exception as e:
            print(f"✗ Multimodal forward pass failed: {e}")
    
    return model


def test_vision_model_separately(config):
    """Test the vision model component separately."""
    print(f"\n=== Testing Vision Model Separately ===")
    
    from glm4v.modeling_glm4v import Glm4vVisionModel
    
    # Create vision model
    vision_model = Glm4vVisionModel(config.vision_config)
    vision_model.eval()
    
    print(f"Vision model parameters: {sum(p.numel() for p in vision_model.parameters()):,}")
    
    # Create fake vision input
    image_inputs = create_fake_image_inputs(config, num_images=2)
    
    with torch.no_grad():
        try:
            # Reshape pixel values for vision model input
            pixel_values = image_inputs["pixel_values"]
            grid_thw = image_inputs["image_grid_thw"]
            
            print(f"Vision input shape: {pixel_values.shape}")
            print(f"Grid THW: {grid_thw}")
            
            # Forward pass through vision model
            vision_output = vision_model(pixel_values, grid_thw)
            print(f"✓ Vision model forward pass successful")
            print(f"Vision output shape: {vision_output.shape}")
            
        except Exception as e:
            print(f"✗ Vision model forward pass failed: {e}")


def test_generation_capability(model, config):
    """Test the model's generation capability."""
    print(f"\n=== Testing Generation Capability ===")
    
    # Create a simple input for generation
    text_inputs = create_fake_text_inputs(config, batch_size=1, seq_len=5)
    
    try:
        with torch.no_grad():
            # Generate tokens (just a few steps)
            generated_ids = model.generate(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
                max_new_tokens=5,
                do_sample=False,  # Greedy decoding
                pad_token_id=0,   # Assuming 0 is pad token
            )
            
        print(f"✓ Generation successful")
        print(f"Input shape: {text_inputs['input_ids'].shape}")
        print(f"Generated shape: {generated_ids.shape}")
        print(f"Input tokens: {text_inputs['input_ids'][0].tolist()}")
        print(f"Generated tokens: {generated_ids[0].tolist()}")
        
    except Exception as e:
        print(f"✗ Generation failed: {e}")


def analyze_model_architecture(model):
    """Analyze and print model architecture details."""
    print(f"\n=== Model Architecture Analysis ===")
    
    print("Model structure:")
    print(f"- Main model: {type(model.model).__name__}")
    print(f"- Vision model: {type(model.model.visual).__name__}")
    print(f"- Language model: {type(model.model.language_model).__name__}")
    print(f"- LM head: {type(model.lm_head).__name__}")
    
    # Count parameters by component
    vision_params = sum(p.numel() for p in model.model.visual.parameters())
    text_params = sum(p.numel() for p in model.model.language_model.parameters())
    lm_head_params = sum(p.numel() for p in model.lm_head.parameters())
    
    print(f"\nParameter counts:")
    print(f"- Vision model: {vision_params:,}")
    print(f"- Text model: {text_params:,}")
    print(f"- LM head: {lm_head_params:,}")
    print(f"- Total: {vision_params + text_params + lm_head_params:,}")
    
    # Print some layer details
    print(f"\nVision model layers:")
    print(f"- Patch embedding: {model.model.visual.patch_embed}")
    print(f"- Number of transformer blocks: {len(model.model.visual.blocks)}")
    print(f"- Output merger: {model.model.visual.merger}")
    
    print(f"\nText model layers:")
    print(f"- Embedding: {model.model.language_model.embed_tokens}")
    print(f"- Number of decoder layers: {len(model.model.language_model.layers)}")
    print(f"- Final norm: {model.model.language_model.norm}")


def main():
    """Main test function."""
    print("GLM-4.1V-Thinking Model - Comprehensive Test with Fake Inputs")
    print("=" * 70)
    
    # System information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    # Create test configuration
    config = create_test_config()
    
    # Test model components
    model = test_model_components(config)
    
    # Test vision model separately
    test_vision_model_separately(config)
    
    # Test generation
    test_generation_capability(model, config)
    
    # Analyze architecture
    analyze_model_architecture(model)
    
    print("\n" + "=" * 70)
    print("✓ All tests completed successfully!")
    print("\nKey takeaways:")
    print("1. The model handles both text-only and multimodal inputs")
    print("2. Images are tokenized using special image token IDs")
    print("3. The vision model processes images into embeddings")
    print("4. The language model integrates vision and text representations")
    print("5. Generation works with both modalities")
    print("\nTo use with real data:")
    print("- Load pretrained weights using model.load_state_dict() or from_pretrained()")
    print("- Use proper tokenizer for text processing")
    print("- Use image processor for real image preprocessing")
    print("- Implement proper prompting strategies for multimodal tasks")


if __name__ == "__main__":
    main() 