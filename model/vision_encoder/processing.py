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
Simplified Image and Video Processing for GLM4V Vision Encoder
Extracted from GLM4V processors to be independent of transformers.
"""

import math
import numpy as np
import torch
from typing import Optional, Union, List, Any, Dict
from PIL import Image

# Constants from CLIP
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def smart_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 2,
    factor: int = 28,
    min_pixels: int = 112 * 112,
    max_pixels: int = 14 * 14 * 2 * 2 * 2 * 6144,
):
    """
    Smart resize function from GLM4V to determine optimal image dimensions.
    
    Args:
        num_frames: Number of frames (for videos)
        height: Original height
        width: Original width
        temporal_factor: Temporal patch size
        factor: Spatial patch factor
        min_pixels: Minimum pixel count
        max_pixels: Maximum pixel count
        
    Returns:
        Tuple of (new_height, new_width)
    """
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


def convert_to_rgb(image):
    """Convert PIL image to RGB format."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return image


def to_numpy_array(image):
    """Convert image to numpy array."""
    if isinstance(image, Image.Image):
        return np.array(image)
    elif torch.is_tensor(image):
        return image.numpy()
    return np.array(image)


def resize_image(image, size, resample=Image.BICUBIC):
    """Resize image to specified size."""
    if isinstance(image, Image.Image):
        return image.resize(size, resample)
    elif isinstance(image, np.ndarray):
        # Convert to PIL, resize, then back to numpy
        pil_image = Image.fromarray(image)
        resized = pil_image.resize(size, resample)
        return np.array(resized)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")


def normalize_image(image, mean, std):
    """Normalize image with mean and std."""
    image = image.astype(np.float32)
    mean = np.array(mean).reshape(1, 1, -1)
    std = np.array(std).reshape(1, 1, -1)
    return (image - mean) / std


class BatchFeature:
    """Simple batch feature container."""
    
    def __init__(self, data, tensor_type=None):
        self.data = data
        if tensor_type == "pt" or tensor_type == "torch":
            # Convert to PyTorch tensors
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    self.data[key] = torch.from_numpy(value)
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, value):
        self.data[key] = value
    
    def keys(self):
        return self.data.keys()
    
    def values(self):
        return self.data.values()
    
    def items(self):
        return self.data.items()


class VisionProcessor:
    """
    Simplified Vision Processor
    Extracted from Glm4vImageProcessor to be independent of transformers.
    """
    
    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        **kwargs,
    ) -> None:
        if size is not None and ("shortest_edge" not in size or "longest_edge" not in size):
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
        else:
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

    def _preprocess_single(
        self,
        images,
        do_resize: Optional[bool] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        do_convert_rgb: Optional[bool] = None,
    ):
        """
        Preprocess a single image or batch of images.
        
        Args:
            images: Image or list of images to preprocess
            do_resize: Whether to resize images
            do_rescale: Whether to rescale pixel values
            rescale_factor: Factor for rescaling
            do_normalize: Whether to normalize
            image_mean: Mean for normalization
            image_std: Std for normalization
            patch_size: Spatial patch size
            temporal_patch_size: Temporal patch size
            merge_size: Merge size
            do_convert_rgb: Whether to convert to RGB
            
        Returns:
            Tuple of (flattened_patches, grid_shape)
        """
        # Set defaults
        do_resize = do_resize if do_resize is not None else self.do_resize
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        patch_size = patch_size if patch_size is not None else self.patch_size
        temporal_patch_size = temporal_patch_size if temporal_patch_size is not None else self.temporal_patch_size
        merge_size = merge_size if merge_size is not None else self.merge_size
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        # Ensure images is a list
        if not isinstance(images, list):
            images = [images]

        # Convert to RGB if needed
        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # Convert to numpy arrays
        images = [to_numpy_array(image) for image in images]

        # Get dimensions from first image
        if len(images[0].shape) == 3:
            height, width, channels = images[0].shape
        else:
            height, width = images[0].shape
            channels = 1

        resized_height, resized_width = height, width
        processed_images = []
        
        for image in images:
            if do_resize:
                resized_height, resized_width = smart_resize(
                    num_frames=temporal_patch_size,
                    height=height,
                    width=width,
                    temporal_factor=temporal_patch_size,
                    factor=patch_size * merge_size,
                )
                if isinstance(image, np.ndarray) and len(image.shape) == 3:
                    image = Image.fromarray(image)
                    image = image.resize((resized_width, resized_height), Image.BICUBIC)
                    image = np.array(image)

            if do_rescale:
                image = image.astype(np.float32) * rescale_factor

            if do_normalize:
                image = normalize_image(image, image_mean, image_std)

            # Ensure channel-first format (C, H, W)
            if len(image.shape) == 3 and image.shape[-1] in [1, 3]:
                image = image.transpose(2, 0, 1)
            elif len(image.shape) == 2:
                image = image[np.newaxis, :, :]

            processed_images.append(image)

        patches = np.array(processed_images)
        
        # Pad frames if needed
        if patches.shape[0] % temporal_patch_size != 0:
            repeats = np.repeat(
                patches[-1][np.newaxis], temporal_patch_size - (patches.shape[0] % temporal_patch_size), axis=0
            )
            patches = np.concatenate([patches, repeats], axis=0)

        channel = patches.shape[1]
        grid_t = patches.shape[0] // temporal_patch_size
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

        # Reshape patches for vision transformer
        patches = patches.reshape(
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
        )

        return flatten_patches, (grid_t, grid_h, grid_w)

    def __call__(
        self,
        images=None,
        videos=None,
        return_tensors=None,
        **kwargs
    ):
        """
        Main preprocessing method.
        
        Args:
            images: Image or list of images to process
            videos: Video or list of videos to process  
            return_tensors: Format for returned tensors ("pt" for PyTorch)
            **kwargs: Additional preprocessing arguments
            
        Returns:
            BatchFeature containing processed data
        """
        data = {}
        
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            
            pixel_values, vision_grid_thws = [], []
            for image in images:
                patches, image_grid_thw = self._preprocess_single(image, **kwargs)
                pixel_values.extend(patches)
                vision_grid_thws.append(image_grid_thw)
            
            pixel_values = np.array(pixel_values)
            vision_grid_thws = np.array(vision_grid_thws)
            data.update({"pixel_values": pixel_values, "image_grid_thw": vision_grid_thws})

        if videos is not None:
            if not isinstance(videos, list):
                videos = [videos]
            
            pixel_values_videos, video_grid_thws = [], []
            for video in videos:
                # Treat video as sequence of images
                patches, video_grid_thw = self._preprocess_single(video, **kwargs)
                pixel_values_videos.extend(patches)
                video_grid_thws.append(video_grid_thw)
            
            pixel_values_videos = np.array(pixel_values_videos)
            video_grid_thws = np.array(video_grid_thws)
            data.update({"pixel_values_videos": pixel_values_videos, "video_grid_thw": video_grid_thws})

        return BatchFeature(data=data, tensor_type=return_tensors)

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None):
        """
        Calculate number of image patches for given dimensions.
        
        Args:
            height: Image height
            width: Image width
            images_kwargs: Optional processing arguments
            
        Returns:
            Number of patches
        """
        patch_size = (images_kwargs or {}).get("patch_size", self.patch_size)
        merge_size = (images_kwargs or {}).get("merge_size", self.merge_size)

        factor = patch_size * merge_size
        resized_height, resized_width = smart_resize(
            num_frames=self.temporal_patch_size,
            height=height,
            width=width,
            temporal_factor=self.temporal_patch_size,
            factor=factor,
        )
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        return grid_h * grid_w


# Factory function for creating processors
def create_vision_processor(**kwargs):
    """
    Create a vision processor with default GLM4V settings.
    
    Args:
        **kwargs: Override default processor settings
        
    Returns:
        VisionProcessor instance
    """
    return VisionProcessor(**kwargs) 