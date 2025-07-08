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
GLM4V Image Processing - Standalone Implementation

This module contains the image processing pipeline for the GLM4V vision encoder.
"""

import math
from typing import Optional, Union, List, Any
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

# =============================================================================
# Constants
# =============================================================================

OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# =============================================================================
# Utility Functions
# =============================================================================

def smart_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 2,
    factor: int = 28,
    min_pixels: int = 112 * 112,
    max_pixels: int = 14 * 14 * 2 * 2 * 2 * 6144,
):
    """Smart resize function for images."""
    if num_frames < temporal_factor:
        raise ValueError(f"t:{num_frames} must be larger than temporal_factor:{temporal_factor}")
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = round(num_frames / temporal_factor) * temporal_factor

    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


def resize_image(image: np.ndarray, size: tuple, resample_method: str = "bicubic") -> np.ndarray:
    """Resize image using PIL."""
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[0] in [1, 3]:  # CHW format
            image = image.transpose(1, 2, 0)  # Convert to HWC
        pil_image = Image.fromarray(image.astype(np.uint8))
    else:
        pil_image = image
    
    resample_map = {
        "bicubic": Image.BICUBIC,
        "bilinear": Image.BILINEAR,
        "nearest": Image.NEAREST
    }
    
    resized = pil_image.resize(size, resample=resample_map.get(resample_method, Image.BICUBIC))
    return np.array(resized)


def convert_to_rgb(image):
    """Convert image to RGB format."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    elif isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 4:  # RGBA
            pil_image = Image.fromarray(image).convert("RGB")
            return np.array(pil_image)
        elif image.ndim == 2:  # Grayscale
            return np.stack([image] * 3, axis=-1)
    return image


def to_channel_dimension_format(image: np.ndarray, channel_dim: str) -> np.ndarray:
    """Convert image to specified channel dimension format."""
    if channel_dim == "channels_first":
        if image.ndim == 3 and image.shape[-1] in [1, 3, 4]:  # HWC -> CHW
            return image.transpose(2, 0, 1)
    elif channel_dim == "channels_last":
        if image.ndim == 3 and image.shape[0] in [1, 3, 4]:  # CHW -> HWC
            return image.transpose(1, 2, 0)
    return image


def normalize_image(image: np.ndarray, mean: List[float], std: List[float]) -> np.ndarray:
    """Normalize image with mean and std."""
    image = image.astype(np.float32)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    
    if image.ndim == 3:
        if image.shape[0] in [1, 3]:  # CHW format
            mean = mean.reshape(-1, 1, 1)
            std = std.reshape(-1, 1, 1)
        else:  # HWC format
            mean = mean.reshape(1, 1, -1)
            std = std.reshape(1, 1, -1)
    
    return (image - mean) / std


def rescale_image(image: np.ndarray, scale: float) -> np.ndarray:
    """Rescale image pixels by a factor."""
    return image.astype(np.float32) * scale


# =============================================================================
# Batch Feature Class
# =============================================================================

@dataclass
class BatchFeature:
    """Container for batched features."""
    data: dict
    tensor_type: Optional[str] = None
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, value):
        self.data[key] = value
        
    def get(self, key, default=None):
        return self.data.get(key, default)
    
    def keys(self):
        return self.data.keys()
    
    def values(self):
        return self.data.values()
    
    def items(self):
        return self.data.items()
    
    def to_tensor_type(self, tensor_type: str):
        """Convert data to specified tensor type."""
        if tensor_type == "pt":
            for key, value in self.data.items():
                if isinstance(value, np.ndarray):
                    self.data[key] = torch.from_numpy(value)
        elif tensor_type == "np":
            for key, value in self.data.items():
                if isinstance(value, torch.Tensor):
                    self.data[key] = value.numpy()
        return self


# =============================================================================
# Image Processor
# =============================================================================

class Glm4vImageProcessor:
    """
    GLM4V image processor for standalone vision encoder.
    
    This processor handles image preprocessing including resizing, normalization,
    and patch extraction for the GLM4V vision model.
    """
    
    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[dict] = None,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        do_convert_rgb: bool = True,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        **kwargs,
    ):
        if size is None:
            size = {"shortest_edge": 112 * 112, "longest_edge": 28 * 28 * 15000}
        self.size = size
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.do_convert_rgb = do_convert_rgb

    def _preprocess_single_image(
        self,
        image,
        do_resize: bool = True,
        do_rescale: bool = True,
        do_normalize: bool = True,
        do_convert_rgb: bool = True,
    ):
        """Preprocess a single image."""
        # Convert to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to RGB
        if do_convert_rgb:
            image = convert_to_rgb(image)
        
        # Get original dimensions
        if image.ndim == 3 and image.shape[0] in [1, 3]:  # CHW
            height, width = image.shape[1], image.shape[2]
        else:  # HWC
            height, width = image.shape[0], image.shape[1]
        
        # Resize
        resized_height, resized_width = height, width
        if do_resize:
            resized_height, resized_width = smart_resize(
                num_frames=self.temporal_patch_size,
                height=height,
                width=width,
                temporal_factor=self.temporal_patch_size,
                factor=self.patch_size * self.merge_size,
            )
            image = resize_image(image, (resized_width, resized_height))
        
        # Convert to channel-first format
        image = to_channel_dimension_format(image, "channels_first")
        
        # Rescale
        if do_rescale:
            image = rescale_image(image, self.rescale_factor)
        
        # Normalize
        if do_normalize:
            image = normalize_image(image, self.image_mean, self.image_std)
        
        return image, (resized_height, resized_width)

    def _create_patches(self, images: List[np.ndarray], resized_height: int, resized_width: int):
        """Create patches from processed images."""
        # Stack images
        patches = np.array(images)
        
        # Ensure temporal dimension compatibility
        if patches.shape[0] % self.temporal_patch_size != 0:
            repeats = np.repeat(
                patches[-1][np.newaxis], self.temporal_patch_size - (patches.shape[0] % self.temporal_patch_size), axis=0
            )
            patches = np.concatenate([patches, repeats], axis=0)
        
        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
        
        # Reshape for patch extraction
        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )
        
        # Rearrange dimensions
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        
        # Flatten patches
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * self.temporal_patch_size * self.patch_size * self.patch_size
        )
        
        return flatten_patches, (grid_t, grid_h, grid_w)

    def preprocess(
        self,
        images,
        do_resize: Optional[bool] = None,
        do_rescale: Optional[bool] = None,
        do_normalize: Optional[bool] = None,
        do_convert_rgb: Optional[bool] = None,
        return_tensors: Optional[str] = None,
    ) -> BatchFeature:
        """
        Preprocess images for the GLM4V vision model.
        
        Args:
            images: Single image or list of images to preprocess
            do_resize: Whether to resize images
            do_rescale: Whether to rescale pixel values
            do_normalize: Whether to normalize images
            do_convert_rgb: Whether to convert to RGB
            return_tensors: Type of tensors to return ("pt" for PyTorch, "np" for NumPy)
            
        Returns:
            BatchFeature containing processed images and metadata
        """
        # Set defaults
        do_resize = do_resize if do_resize is not None else self.do_resize
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        
        # Ensure images is a list
        if not isinstance(images, list):
            images = [images]
        
        # Process each image
        processed_images = []
        all_patches = []
        vision_grid_thws = []
        
        for image in images:
            processed_image, (resized_height, resized_width) = self._preprocess_single_image(
                image, do_resize, do_rescale, do_normalize, do_convert_rgb
            )
            processed_images.append(processed_image)
            
            # Create patches for this image
            patches, grid_thw = self._create_patches([processed_image], resized_height, resized_width)
            all_patches.extend(patches)
            vision_grid_thws.append(grid_thw)
        
        # Convert to arrays
        pixel_values = np.array(all_patches)
        image_grid_thw = np.array(vision_grid_thws)
        
        # Create batch feature
        data = {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
        
        result = BatchFeature(data=data, tensor_type=return_tensors)
        
        # Convert to tensors if requested
        if return_tensors:
            result.to_tensor_type(return_tensors)
        
        return result

    def get_number_of_image_patches(self, height: int, width: int) -> int:
        """
        Get the number of patches for an image of given dimensions.
        
        Args:
            height: Image height
            width: Image width
            
        Returns:
            Number of patches
        """
        factor = self.patch_size * self.merge_size
        resized_height, resized_width = smart_resize(
            num_frames=self.temporal_patch_size,
            height=height,
            width=width,
            temporal_factor=self.temporal_patch_size,
            factor=factor,
        )
        grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
        return grid_h * grid_w


# =============================================================================
# Factory Functions
# =============================================================================

def create_glm4v_image_processor(
    patch_size: int = 14,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    **kwargs
) -> Glm4vImageProcessor:
    """
    Create a GLM4V image processor with custom parameters.
    
    Args:
        patch_size: Size of image patches
        temporal_patch_size: Size of temporal patches
        merge_size: Spatial merge size
        **kwargs: Additional processor parameters
        
    Returns:
        Configured image processor
    """
    return Glm4vImageProcessor(
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        merge_size=merge_size,
        **kwargs
    ) 