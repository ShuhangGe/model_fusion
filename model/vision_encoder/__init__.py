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
Standalone GLM4V Vision Encoder

This module provides a complete, standalone implementation of the GLM4V vision encoder
that is independent of text model components. It includes:

- Vision configuration management
- Vision transformer layers and components  
- Complete vision model for processing images and videos
- Image and video preprocessing utilities
- Simple APIs for integration with other models

Example Usage:

    ```python
    from model.vision_encoder import VisionConfig, VisionModel, VisionProcessor
    
    # Create configuration
    config = VisionConfig()
    
    # Create vision model
    model = VisionModel(config)
    
    # Create processor
    processor = VisionProcessor()
    
    # Process an image
    from PIL import Image
    image = Image.open("path/to/image.jpg")
    inputs = processor(images=[image], return_tensors="pt")
    
    # Get vision embeddings
    embeddings = model.get_image_features(
        inputs["pixel_values"], 
        inputs["image_grid_thw"]
    )
    ```

Factory Functions:

    ```python
    # Create a complete vision encoder setup
    config, model, processor = create_glm4v_vision_encoder()
    
    # Process and encode an image in one step
    embeddings = encode_image("path/to/image.jpg")
    ```
"""

from .config import VisionConfig
from .model import VisionModel  
from .processing import VisionProcessor, create_vision_processor, BatchFeature
from .layers import (
    RMSNorm,
    VisionMlp,
    VisionPatchEmbed,
    VisionRotaryEmbedding,
    VisionPatchMerger,
    VisionEmbeddings,
    VisionAttention,
    VisionBlock,
)
from .utils import (
    ACT2FN,
    apply_rotary_pos_emb_vision,
    repeat_kv,
    eager_attention_forward,
    BaseLayer,
    init_weights,
    get_activation_fn,
)

import torch
from PIL import Image
from typing import Union, List, Optional, Tuple


def create_glm4v_vision_encoder(
    config_kwargs=None,
    model_kwargs=None, 
    processor_kwargs=None
) -> Tuple[VisionConfig, VisionModel, VisionProcessor]:
    """
    Create a complete GLM4V vision encoder setup with default configurations.
    
    Args:
        config_kwargs: Optional config overrides
        model_kwargs: Optional model creation overrides  
        processor_kwargs: Optional processor overrides
        
    Returns:
        Tuple of (config, model, processor)
    """
    config = VisionConfig(**(config_kwargs or {}))
    model = VisionModel(config, **(model_kwargs or {}))
    processor = VisionProcessor(**(processor_kwargs or {}))
    
    return config, model, processor


def encode_image(
    image_path: str,
    model: Optional[VisionModel] = None,
    processor: Optional[VisionProcessor] = None,
    return_grid: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Convenience function to encode a single image.
    
    Args:
        image_path: Path to the image file
        model: Optional pre-created vision model
        processor: Optional pre-created processor
        return_grid: Whether to return grid information
        
    Returns:
        Image embeddings tensor, optionally with grid info
    """
    if model is None or processor is None:
        _, model, processor = create_glm4v_vision_encoder()
    
    # Load and process image
    image = Image.open(image_path)
    inputs = processor(images=[image], return_tensors="pt")
    
    # Generate embeddings
    with torch.no_grad():
        embeddings = model.get_image_features(
            inputs["pixel_values"],
            inputs["image_grid_thw"]
        )
    
    if return_grid:
        return embeddings[0], inputs["image_grid_thw"]
    else:
        return embeddings[0]


def encode_images(
    image_paths: List[str],
    model: Optional[VisionModel] = None,
    processor: Optional[VisionProcessor] = None,
    batch_size: int = 1
) -> List[torch.Tensor]:
    """
    Encode multiple images.
    
    Args:
        image_paths: List of image file paths
        model: Optional pre-created vision model
        processor: Optional pre-created processor
        batch_size: Processing batch size
        
    Returns:
        List of embedding tensors
    """
    if model is None or processor is None:
        _, model, processor = create_glm4v_vision_encoder()
    
    embeddings = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = [Image.open(path) for path in batch_paths]
        
        inputs = processor(images=images, return_tensors="pt")
        
        with torch.no_grad():
            batch_embeddings = model.get_image_features(
                inputs["pixel_values"],
                inputs["image_grid_thw"]
            )
        
        embeddings.extend(batch_embeddings)
    
    return embeddings


def create_vision_config(**kwargs) -> VisionConfig:
    """
    Create a vision configuration with custom parameters.
    
    Args:
        **kwargs: Configuration parameters to override
        
    Returns:
        VisionConfig instance
    """
    return VisionConfig(**kwargs)


def load_vision_model(model_path: str) -> VisionModel:
    """
    Load a pre-trained vision model from disk.
    
    Args:
        model_path: Path to the saved model directory
        
    Returns:
        Loaded VisionModel instance
    """
    return VisionModel.from_pretrained(model_path)


def save_vision_model(model: VisionModel, save_path: str):
    """
    Save a vision model to disk.
    
    Args:
        model: VisionModel to save
        save_path: Directory path to save the model
    """
    model.save_pretrained(save_path)


# Export all public components
__all__ = [
    # Core classes
    "VisionConfig",
    "VisionModel", 
    "VisionProcessor",
    "BatchFeature",
    
    # Layer components
    "RMSNorm",
    "VisionMlp", 
    "VisionPatchEmbed",
    "VisionRotaryEmbedding",
    "VisionPatchMerger",
    "VisionEmbeddings",
    "VisionAttention", 
    "VisionBlock",
    
    # Utility functions
    "ACT2FN",
    "apply_rotary_pos_emb_vision",
    "repeat_kv",
    "eager_attention_forward",
    "BaseLayer",
    "init_weights",
    "get_activation_fn",
    
    # Factory and convenience functions
    "create_glm4v_vision_encoder",
    "create_vision_processor",
    "create_vision_config",
    "encode_image",
    "encode_images",
    "load_vision_model",
    "save_vision_model",
]


# Version information
__version__ = "1.0.0"
__author__ = "GLM4V Vision Encoder Team"
__description__ = "Standalone GLM4V Vision Encoder extracted from the original GLM4V model" 