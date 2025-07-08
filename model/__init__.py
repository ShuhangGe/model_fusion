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
GLM4V Vision Encoder - Standalone Package

This package provides a complete standalone implementation of the GLM4V vision encoder,
including image and video processing capabilities.

Usage:
    ```python
    from model import create_glm4v_vision_encoder, create_glm4v_image_processor
    
    # Create vision model and processor
    vision_model = create_glm4v_vision_encoder()
    image_processor = create_glm4v_image_processor()
    
    # Process images
    from PIL import Image
    image = Image.open("image.jpg")
    inputs = image_processor.preprocess([image], return_tensors="pt")
    
    # Get vision embeddings
    embeddings = vision_model(inputs["pixel_values"], inputs["image_grid_thw"])
    ```
"""

from .vision_embedding import (
    # Configuration
    Glm4vVisionConfig,
    
    # Core model components
    Glm4vVisionModel,
    Glm4vVisionEmbeddings,
    Glm4vVisionPatchEmbed,
    Glm4vVisionRotaryEmbedding,
    Glm4vVisionPatchMerger,
    Glm4vVisionAttention,
    Glm4vVisionBlock,
    Glm4VisionMlp,
    Glm4vRMSNorm,
    
    # Factory functions
    create_glm4v_vision_encoder,
    create_glm4v_vision_encoder_default,
    
    # Utility functions
    rotate_half,
    apply_rotary_pos_emb_vision,
    repeat_kv,
    eager_attention_forward,
    smart_resize,
    get_activation_fn,
)

from .image_processing import (
    # Image processor
    Glm4vImageProcessor,
    BatchFeature,
    
    # Factory functions
    create_glm4v_image_processor,
    
    # Image processing utilities
    resize_image,
    convert_to_rgb,
    to_channel_dimension_format,
    normalize_image,
    rescale_image,
    
    # Constants
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
)

from .video_processing import (
    # Video processor
    Glm4vVideoProcessor,
    VideoMetadata,
    
    # Factory functions
    create_glm4v_video_processor,
    
    # Video processing utilities
    group_videos_by_shape,
    reorder_videos,
    get_image_size,
)

__version__ = "1.0.0"

__all__ = [
    # Configuration
    "Glm4vVisionConfig",
    
    # Core model components
    "Glm4vVisionModel",
    "Glm4vVisionEmbeddings", 
    "Glm4vVisionPatchEmbed",
    "Glm4vVisionRotaryEmbedding",
    "Glm4vVisionPatchMerger",
    "Glm4vVisionAttention",
    "Glm4vVisionBlock",
    "Glm4VisionMlp",
    "Glm4vRMSNorm",
    
    # Processing classes
    "Glm4vImageProcessor",
    "Glm4vVideoProcessor",
    "BatchFeature",
    "VideoMetadata",
    
    # Factory functions
    "create_glm4v_vision_encoder",
    "create_glm4v_vision_encoder_default",
    "create_glm4v_image_processor",
    "create_glm4v_video_processor",
    
    # Utility functions
    "rotate_half",
    "apply_rotary_pos_emb_vision",
    "repeat_kv",
    "eager_attention_forward",
    "smart_resize",
    "get_activation_fn",
    "resize_image",
    "convert_to_rgb",
    "to_channel_dimension_format", 
    "normalize_image",
    "rescale_image",
    "group_videos_by_shape",
    "reorder_videos",
    "get_image_size",
    
    # Constants
    "OPENAI_CLIP_MEAN",
    "OPENAI_CLIP_STD",
]


# =============================================================================
# Convenience Factory Functions
# =============================================================================

def create_complete_glm4v_vision_pipeline(
    hidden_size: int = 1536,
    depth: int = 24,
    num_heads: int = 12,
    image_size: int = 336,
    patch_size: int = 14,
    out_hidden_size: int = 4096,
    **kwargs
):
    """
    Create a complete GLM4V vision pipeline with model and processors.
    
    Args:
        hidden_size: Vision model hidden size
        depth: Number of transformer layers
        num_heads: Number of attention heads
        image_size: Input image size
        patch_size: Patch size
        out_hidden_size: Output hidden size
        **kwargs: Additional configuration parameters
        
    Returns:
        tuple: (vision_model, image_processor, video_processor)
    """
    # Create vision model
    vision_model = create_glm4v_vision_encoder(
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        image_size=image_size,
        patch_size=patch_size,
        out_hidden_size=out_hidden_size,
        **kwargs
    )
    
    # Create processors
    image_processor = create_glm4v_image_processor(
        patch_size=patch_size,
        **kwargs
    )
    
    video_processor = create_glm4v_video_processor(
        patch_size=patch_size,
        **kwargs
    )
    
    return vision_model, image_processor, video_processor


def load_glm4v_vision_from_config(config_dict: dict):
    """
    Load GLM4V vision model from configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        GLM4V vision model
    """
    config = Glm4vVisionConfig(**config_dict)
    return Glm4vVisionModel(config)


# =============================================================================
# Examples and Documentation
# =============================================================================

def example_image_processing():
    """
    Example of how to process images with the GLM4V vision encoder.
    
    Returns:
        Example usage code as string
    """
    return '''
    from model import create_complete_glm4v_vision_pipeline
    from PIL import Image
    import torch
    
    # Create complete pipeline
    vision_model, image_processor, _ = create_complete_glm4v_vision_pipeline()
    
    # Load and process image
    image = Image.open("path/to/image.jpg")
    inputs = image_processor.preprocess([image], return_tensors="pt")
    
    # Get vision embeddings
    with torch.no_grad():
        embeddings = vision_model(inputs["pixel_values"], inputs["image_grid_thw"])
    
    print(f"Vision embeddings shape: {embeddings.shape}")
    '''


def example_video_processing():
    """
    Example of how to process videos with the GLM4V vision encoder.
    
    Returns:
        Example usage code as string
    """
    return '''
    from model import create_complete_glm4v_vision_pipeline, VideoMetadata
    import torch
    
    # Create complete pipeline
    vision_model, _, video_processor = create_complete_glm4v_vision_pipeline()
    
    # Load video tensor (example shape: [num_frames, channels, height, width])
    video = torch.randn(16, 3, 224, 224)  # 16 frames, 3 channels, 224x224
    metadata = VideoMetadata(fps=30.0, total_num_frames=16)
    
    # Process video
    inputs = video_processor.preprocess([video], video_metadata=[metadata], return_tensors="pt")
    
    # Get vision embeddings
    with torch.no_grad():
        embeddings = vision_model(inputs["pixel_values_videos"], inputs["video_grid_thw"])
    
    print(f"Video vision embeddings shape: {embeddings.shape}")
    '''


def example_custom_configuration():
    """
    Example of creating custom GLM4V vision configuration.
    
    Returns:
        Example usage code as string
    """
    return '''
    from model import Glm4vVisionConfig, Glm4vVisionModel, create_glm4v_image_processor
    
    # Create custom configuration
    config = Glm4vVisionConfig(
        hidden_size=768,      # Smaller model
        depth=12,             # Fewer layers
        num_heads=12,         # Adjust heads
        image_size=224,       # Different input size
        patch_size=16,        # Larger patches
        out_hidden_size=2048, # Different output size
    )
    
    # Create model with custom config
    vision_model = Glm4vVisionModel(config)
    image_processor = create_glm4v_image_processor(patch_size=16)
    
    print(f"Custom model parameters: {sum(p.numel() for p in vision_model.parameters()):,}")
    ''' 