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

"""Utility functions for GLM4V embedding pipeline."""

from typing import Union, List, Tuple, Any, Optional
import torch
import numpy as np
from PIL import Image


def format_embeddings_for_language_model(
    embeddings: torch.Tensor,
    batch_size: int = 1,
    sequence_length: Optional[int] = None,
) -> torch.Tensor:
    """
    Format GLM4V embeddings for use with language models.
    
    Args:
        embeddings: Raw embeddings from GLM4V vision model
        batch_size: Target batch size
        sequence_length: Target sequence length (if None, uses embedding length)
        
    Returns:
        torch.Tensor: Formatted embeddings with shape (batch_size, sequence_length, hidden_size)
    """
    if embeddings.dim() == 2:
        # Shape: (num_patches, hidden_size) -> (batch_size, sequence_length, hidden_size)
        num_patches, hidden_size = embeddings.shape
        
        if sequence_length is None:
            sequence_length = num_patches
        
        if sequence_length > num_patches:
            # Pad with zeros if needed
            padding = torch.zeros(sequence_length - num_patches, hidden_size, 
                                device=embeddings.device, dtype=embeddings.dtype)
            embeddings = torch.cat([embeddings, padding], dim=0)
        elif sequence_length < num_patches:
            # Truncate if needed
            embeddings = embeddings[:sequence_length]
        
        # Reshape to batch format
        embeddings = embeddings.unsqueeze(0).expand(batch_size, -1, -1)
    
    return embeddings


def calculate_patch_count(
    image_size: Union[int, Tuple[int, int]],
    patch_size: int = 14,
    merge_size: int = 2,
) -> Tuple[int, int]:
    """
    Calculate the number of patches for given image dimensions.
    
    Args:
        image_size: Image size (height, width) or single int for square images
        patch_size: Size of each patch
        merge_size: Spatial merge factor
        
    Returns:
        Tuple[int, int]: (height_patches, width_patches)
    """
    if isinstance(image_size, int):
        height, width = image_size, image_size
    else:
        height, width = image_size
    
    # Calculate effective patch size after merging
    effective_patch_size = patch_size * merge_size
    
    height_patches = height // effective_patch_size
    width_patches = width // effective_patch_size
    
    return height_patches, width_patches


def prepare_image_input(
    images: Union[Image.Image, np.ndarray, torch.Tensor, List[Any]],
    target_device: torch.device = None,
) -> Union[Image.Image, List[Image.Image]]:
    """
    Prepare image input for processing.
    
    Args:
        images: Input images in various formats
        target_device: Target device for tensor operations
        
    Returns:
        Image(s) in PIL format ready for processing
    """
    if isinstance(images, (list, tuple)):
        return [prepare_single_image(img) for img in images]
    else:
        return prepare_single_image(images)


def prepare_single_image(image: Union[Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:
    """
    Convert a single image to PIL format.
    
    Args:
        image: Input image in various formats
        
    Returns:
        PIL.Image: Image in PIL format
    """
    if isinstance(image, Image.Image):
        return image
    elif isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[0] in [1, 3, 4]:  # CHW format
            image = image.transpose(1, 2, 0)
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        return Image.fromarray(image)
    elif isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] in [1, 3, 4]:  # CHW format
            image = image.permute(1, 2, 0)
        if image.dtype in [torch.float32, torch.float64]:
            image = (image * 255).to(torch.uint8)
        return Image.fromarray(image.cpu().numpy())
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")


def create_attention_mask(
    embeddings: torch.Tensor,
    padding_length: int = 0,
) -> torch.Tensor:
    """
    Create attention mask for embeddings.
    
    Args:
        embeddings: Input embeddings
        padding_length: Number of padding tokens
        
    Returns:
        torch.Tensor: Attention mask (1 for valid tokens, 0 for padding)
    """
    if embeddings.dim() == 2:
        seq_length = embeddings.shape[0]
        batch_size = 1
    else:
        batch_size, seq_length = embeddings.shape[:2]
    
    # Create mask
    mask = torch.ones(batch_size, seq_length + padding_length, 
                     device=embeddings.device, dtype=torch.long)
    
    if padding_length > 0:
        mask[:, -padding_length:] = 0
    
    return mask


def validate_input_shapes(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
) -> bool:
    """
    Validate that input shapes are compatible.
    
    Args:
        pixel_values: Processed pixel values
        grid_thw: Grid temporal, height, width information
        
    Returns:
        bool: True if shapes are valid
    """
    try:
        # Check basic shapes
        if pixel_values.dim() != 2:
            return False
        
        if grid_thw.dim() != 2 or grid_thw.shape[1] != 3:
            return False
        
        # Calculate expected patches
        expected_patches = sum(t * h * w for t, h, w in grid_thw)
        actual_patches = pixel_values.shape[0]
        
        return expected_patches == actual_patches
    except Exception:
        return False


def get_embedding_info(embeddings: torch.Tensor) -> dict:
    """
    Get information about embeddings.
    
    Args:
        embeddings: Input embeddings
        
    Returns:
        dict: Information about the embeddings
    """
    info = {
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "device": str(embeddings.device),
        "requires_grad": embeddings.requires_grad,
        "memory_usage_mb": embeddings.numel() * embeddings.element_size() / (1024 * 1024),
    }
    
    if embeddings.dim() >= 2:
        info["num_tokens"] = embeddings.shape[-2] if embeddings.dim() > 2 else embeddings.shape[0]
        info["hidden_size"] = embeddings.shape[-1]
    
    return info


__all__ = [
    "format_embeddings_for_language_model",
    "calculate_patch_count",
    "prepare_image_input",
    "prepare_single_image",
    "create_attention_mask",
    "validate_input_shapes",
    "get_embedding_info",
] 