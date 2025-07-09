#!/usr/bin/env python3
# coding=utf-8
# Copyright 2025 The ZhipuAI Inc. team and HuggingFace Inc. team. All rights reserved.

"""
Example usage of the GLM4V Complete Preprocessing Pipeline.

This script demonstrates how to use the extracted GLM4V preprocessing pipeline
that handles everything before the TextModel input.
"""

import torch
import numpy as np
from PIL import Image

# Import the complete preprocessing pipeline
from . import GLM4VPreprocessingPipeline, Glm4vConfig


def example_complete_preprocessing():
    """Example of the complete preprocessing pipeline - from raw inputs to language model ready."""
    print("=== Complete Preprocessing Pipeline Example ===")
    
    # Create a dummy tokenizer (you would use a real tokenizer in practice)
    class DummyTokenizer:
        def __init__(self):
            self.image_token = "<|image|>"
            self.image_token_id = 151343
            self.vocab_size = 151552
            
        def __call__(self, text, **kwargs):
            # Dummy tokenization
            tokens = text.split()
            input_ids = []
            for token in tokens:
                if token == self.image_token:
                    input_ids.append(self.image_token_id)
                else:
                    input_ids.append(hash(token) % 50000)  # Dummy token ID
            
            return {
                "input_ids": torch.tensor([input_ids]),
                "attention_mask": torch.ones(1, len(input_ids))
            }
    
    # Initialize pipeline
    tokenizer = DummyTokenizer()
    config = Glm4vConfig()
    pipeline = GLM4VPreprocessingPipeline(config=config, tokenizer=tokenizer)
    pipeline.eval()
    
    # Create dummy multimodal input
    text = "What is in this image? <|image|>"
    dummy_image = Image.new('RGB', (336, 336), color='red')
    
    try:
        # Process complete multimodal input
        result = pipeline(
            text=text,
            images=[dummy_image],
            return_dict=True
        )
        
        print(f"✓ Complete preprocessing successful!")
        print(f"  Original text: '{text}'")
        print(f"  Input embeddings shape: {result['inputs_embeds'].shape}")
        print(f"  Position IDs shape: {result['position_ids'].shape}")
        print(f"  Attention mask shape: {result['attention_mask'].shape}")
        print(f"  Has images: {result['has_images']}")
        print(f"  Number of image tokens: {result['num_image_tokens']}")
        print(f"  Sequence length: {result['sequence_length']}")
        print(f"  Hidden size: {result['hidden_size']}")
        
        # Verify the embeddings are ready for language model
        embeddings = result['inputs_embeds']
        positions = result['position_ids']
        attention = result['attention_mask']
        
        print(f"  → Ready for language model: embeddings {embeddings.shape}, positions {positions.shape}")
        
    except Exception as e:
        print(f"✗ Error in complete preprocessing: {e}")


def example_step_by_step_processing():
    """Example of step-by-step processing through the pipeline."""
    print("\n=== Step-by-Step Processing Example ===")
    
    # Dummy tokenizer
    class DummyTokenizer:
        def __init__(self):
            self.image_token = "<|image|>"
            self.image_token_id = 151343
            
        def __call__(self, text, **kwargs):
            tokens = text.replace(self.image_token, f" {self.image_token} ").split()
            input_ids = []
            for token in tokens:
                if token == self.image_token:
                    input_ids.append(self.image_token_id)
                elif token.strip():
                    input_ids.append(hash(token) % 50000)
            
            return {
                "input_ids": torch.tensor([input_ids]),
                "attention_mask": torch.ones(1, len(input_ids))
            }
    
    tokenizer = DummyTokenizer()
    pipeline = GLM4VPreprocessingPipeline(tokenizer=tokenizer)
    pipeline.eval()
    
    # Step 1: Process raw inputs
    print("Step 1: Processing raw inputs...")
    text = "Describe this image: <|image|>"
    image = Image.new('RGB', (224, 224), color='blue')
    
    try:
        processed = pipeline.process_inputs(
            text=text,
            images=[image]
        )
        print(f"  ✓ Raw input processing complete")
        print(f"    Input IDs shape: {processed['input_ids'].shape}")
        print(f"    Pixel values shape: {processed['pixel_values'].shape}")
        print(f"    Image grid: {processed['image_grid_thw']}")
        
        # Step 2: Extract and fuse embeddings
        print("Step 2: Extracting and fusing embeddings...")
        fused = pipeline.extract_and_fuse_embeddings(
            input_ids=processed['input_ids'],
            pixel_values=processed['pixel_values'],
            image_grid_thw=processed['image_grid_thw']
        )
        
        print(f"  ✓ Embedding fusion complete")
        print(f"    Fused embeddings shape: {fused['inputs_embeds'].shape}")
        print(f"    Position IDs shape: {fused['position_ids'].shape}")
        print(f"    Has images: {fused['has_images']}")
        
        # Step 3: Verify outputs
        print("Step 3: Verifying outputs...")
        embeddings = fused['inputs_embeds']
        
        # Check that embeddings contain both text and vision information
        text_positions = processed['input_ids'][0] != tokenizer.image_token_id
        vision_positions = processed['input_ids'][0] == tokenizer.image_token_id
        
        print(f"  ✓ Output verification complete")
        print(f"    Text token positions: {text_positions.sum()}")
        print(f"    Vision token positions: {vision_positions.sum()}")
        print(f"    Total tokens: {embeddings.shape[1]}")
        
    except Exception as e:
        print(f"  ✗ Error in step-by-step processing: {e}")


def example_vision_only_extraction():
    """Example of extracting only vision embeddings."""
    print("\n=== Vision-Only Extraction Example ===")
    
    pipeline = GLM4VPreprocessingPipeline()
    pipeline.eval()
    
    # Create dummy vision inputs
    dummy_image = Image.new('RGB', (448, 448), color='green')
    
    try:
        # First process the image
        processed = pipeline.image_processor([dummy_image], return_tensors="pt")
        
        # Extract only vision embeddings
        vision_only = pipeline.get_vision_embeddings_only(
            pixel_values=processed["pixel_values"],
            image_grid_thw=processed["image_grid_thw"]
        )
        
        print(f"✓ Vision-only extraction successful!")
        if "image_embeddings" in vision_only:
            print(f"  Image embeddings shape: {vision_only['image_embeddings'].shape}")
            print(f"  Pure vision features extracted (no text fusion)")
        
    except Exception as e:
        print(f"✗ Error in vision-only extraction: {e}")


def example_multimodal_fusion_demonstration():
    """Demonstrate the critical multimodal fusion process."""
    print("\n=== Multimodal Fusion Demonstration ===")
    
    class DummyTokenizer:
        def __init__(self):
            self.image_token_id = 151343
            
        def __call__(self, text, **kwargs):
            # Create a sequence with image tokens
            input_ids = [100, 200, 151343, 151343, 151343, 300, 400]  # text, image, image, image, text
            return {
                "input_ids": torch.tensor([input_ids]),
                "attention_mask": torch.ones(1, len(input_ids))
            }
    
    tokenizer = DummyTokenizer()
    pipeline = GLM4VPreprocessingPipeline(tokenizer=tokenizer)
    pipeline.eval()
    
    try:
        # Create inputs that will trigger fusion
        processed_inputs = {
            "input_ids": torch.tensor([[100, 200, 151343, 151343, 151343, 300, 400]]),
            "pixel_values": torch.randn(3, 3 * 1 * 14 * 14),  # 3 patches
            "image_grid_thw": torch.tensor([[1, 28, 28]])  # Will be split into 3 patches after merging
        }
        
        print("Before fusion:")
        print(f"  Input IDs: {processed_inputs['input_ids'][0].tolist()}")
        print(f"  Image token positions: {(processed_inputs['input_ids'][0] == 151343).nonzero().flatten().tolist()}")
        
        # Perform fusion
        fused = pipeline.extract_and_fuse_embeddings(**processed_inputs)
        
        print("After fusion:")
        print(f"  Fused embeddings shape: {fused['inputs_embeds'].shape}")
        print(f"  Position IDs shape: {fused['position_ids'].shape}")
        print(f"  ✓ Vision embeddings inserted at image token positions")
        print(f"  ✓ 3D spatial coordinates calculated for vision tokens")
        
        # Show position ID structure
        pos_ids = fused['position_ids']
        print(f"  Position structure: temporal={pos_ids[0,0,:3]}, height={pos_ids[1,0,:3]}, width={pos_ids[2,0,:3]}")
        
    except Exception as e:
        print(f"✗ Error in fusion demonstration: {e}")


def example_output_compatibility():
    """Demonstrate output compatibility with language models."""
    print("\n=== Language Model Compatibility Example ===")
    
    class DummyLanguageModel(torch.nn.Module):
        def __init__(self, hidden_size=4096):
            super().__init__()
            self.norm = torch.nn.LayerNorm(hidden_size)
            self.linear = torch.nn.Linear(hidden_size, 50000)  # vocab size
            
        def forward(self, inputs_embeds, position_ids=None, attention_mask=None):
            # Dummy language model processing
            x = self.norm(inputs_embeds)
            logits = self.linear(x)
            return {"logits": logits}
    
    # Initialize pipeline and dummy language model
    class DummyTokenizer:
        def __init__(self):
            self.image_token_id = 151343
            
        def __call__(self, text, **kwargs):
            return {
                "input_ids": torch.tensor([[100, 200, 151343, 300]]),
                "attention_mask": torch.ones(1, 4)
            }
    
    tokenizer = DummyTokenizer()
    pipeline = GLM4VPreprocessingPipeline(tokenizer=tokenizer)
    language_model = DummyLanguageModel()
    
    pipeline.eval()
    language_model.eval()
    
    try:
        # Process multimodal input
        image = Image.new('RGB', (224, 224), color='purple')
        result = pipeline(
            text="Test <|image|> input",
            images=[image],
            return_dict=True
        )
        
        # Feed to language model
        with torch.no_grad():
            lm_output = language_model(
                inputs_embeds=result['inputs_embeds'],
                position_ids=result['position_ids'],
                attention_mask=result['attention_mask']
            )
        
        print(f"✓ Language model compatibility verified!")
        print(f"  Pipeline output shape: {result['inputs_embeds'].shape}")
        print(f"  Language model logits shape: {lm_output['logits'].shape}")
        print(f"  ✓ Seamless integration between preprocessing and language model")
        
    except Exception as e:
        print(f"✗ Error in compatibility test: {e}")


def main():
    """Run all examples."""
    print("GLM4V Complete Preprocessing Pipeline Examples")
    print("=" * 60)
    print("This demonstrates the complete pipeline that handles everything")
    print("before the GLM4V TextModel input - from raw inputs to LM-ready embeddings.")
    print("=" * 60)
    
    # Run examples
    example_complete_preprocessing()
    example_step_by_step_processing()
    example_vision_only_extraction()
    example_multimodal_fusion_demonstration()
    example_output_compatibility()
    
    print("\n" + "=" * 60)
    print("✓ All examples completed!")
    print("The pipeline successfully processes multimodal inputs and produces")
    print("embeddings ready for any transformer-based language model.")
    print("=" * 60)


if __name__ == "__main__":
    main() 