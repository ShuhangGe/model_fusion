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
GLM4V Video Processing - Standalone Implementation

This module contains the video processing pipeline for the GLM4V vision encoder.
"""

import math
from typing import Optional, Union, List, Any, Dict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .image_processing import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, smart_resize, BatchFeature

# =============================================================================
# Video Metadata
# =============================================================================

@dataclass
class VideoMetadata:
    """Metadata for video files."""
    fps: float = 2.0
    total_num_frames: int = 16
    duration: Optional[float] = None


# =============================================================================
# Video Processing Utilities
# =============================================================================

def group_videos_by_shape(videos: List[torch.Tensor]) -> tuple[Dict[tuple, torch.Tensor], List[int]]:
    """Group videos by their shape for batch processing."""
    shape_to_videos = {}
    shape_to_indices = {}
    
    for i, video in enumerate(videos):
        shape = video.shape
        if shape not in shape_to_videos:
            shape_to_videos[shape] = []
            shape_to_indices[shape] = []
        shape_to_videos[shape].append(video)
        shape_to_indices[shape].append(i)
    
    # Stack videos with same shape
    grouped_videos = {}
    video_indices = []
    
    for shape, video_list in shape_to_videos.items():
        stacked = torch.stack(video_list, dim=0)
        grouped_videos[shape] = stacked
        video_indices.extend(shape_to_indices[shape])
    
    return grouped_videos, video_indices


def reorder_videos(grouped_videos: Dict[tuple, torch.Tensor], original_indices: List[int]) -> List[torch.Tensor]:
    """Reorder grouped videos back to original order."""
    # Flatten all videos
    all_videos = []
    current_indices = []
    
    for shape, stacked_videos in grouped_videos.items():
        for i in range(stacked_videos.shape[0]):
            all_videos.append(stacked_videos[i])
            current_indices.append(len(all_videos) - 1)
    
    # Create mapping from original to current positions
    reorder_map = {}
    flat_idx = 0
    for shape, stacked_videos in grouped_videos.items():
        for i in range(stacked_videos.shape[0]):
            reorder_map[original_indices[flat_idx]] = all_videos[flat_idx]
            flat_idx += 1
    
    # Reorder according to original indices
    reordered = []
    for i in range(len(original_indices)):
        reordered.append(reorder_map[i])
    
    return reordered


def get_image_size(video: torch.Tensor, channel_dim: str = "channels_first") -> tuple[int, int]:
    """Get spatial dimensions of video tensor."""
    if channel_dim == "channels_first":
        return video.shape[-2], video.shape[-1]  # H, W
    else:
        return video.shape[-3], video.shape[-2]  # H, W


# =============================================================================
# Video Processor
# =============================================================================

class Glm4vVideoProcessor:
    """
    GLM4V video processor for standalone vision encoder.
    
    This processor handles video preprocessing including frame sampling, resizing,
    normalization, and patch extraction for the GLM4V vision model.
    """
    
    def __init__(
        self,
        size: Optional[Dict[str, int]] = None,
        max_image_size: Optional[Dict[str, int]] = None,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        do_resize: bool = True,
        do_rescale: bool = True,
        do_normalize: bool = True,
        do_convert_rgb: bool = True,
        do_sample_frames: bool = True,
        max_duration: int = 300,
        fps: float = 2.0,
        num_frames: int = 16,
        **kwargs,
    ):
        self.size = size or {"shortest_edge": 112 * 112, "longest_edge": 28 * 28 * 2 * 30000}
        self.max_image_size = max_image_size or {"longest_edge": 28 * 28 * 2 * 30000}
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        self.do_convert_rgb = do_convert_rgb
        self.do_sample_frames = do_sample_frames
        self.max_duration = max_duration
        self.fps = fps
        self.num_frames = num_frames

    def sample_frames(
        self,
        video: torch.Tensor,
        metadata: Union[VideoMetadata, Dict],
    ) -> tuple[torch.Tensor, List[int]]:
        """Sample frames from video tensor."""
        total_frames = video.shape[0]
        
        if isinstance(metadata, dict):
            video_fps = metadata.get("fps", 2.0)
            meta_frames = metadata.get("total_num_frames", total_frames)
            duration = metadata.get("duration", None)
        else:
            video_fps = getattr(metadata, "fps", 2.0)
            meta_frames = getattr(metadata, "total_num_frames", total_frames)
            duration = getattr(metadata, "duration", None)
        
        max_frame_idx = meta_frames - 1
        if duration is None:
            duration = round(max_frame_idx / video_fps) + 1

        if duration <= self.max_duration:
            n = int(math.floor(duration * self.fps))
            frame_indices = [min(max_frame_idx, int(math.ceil(i * video_fps / self.fps))) for i in range(n)]
        else:
            num_samples = int(self.max_duration * self.fps)
            if num_samples >= meta_frames:
                frame_indices = list(range(meta_frames))
            else:
                target_seconds = np.linspace(0, duration, num_samples, endpoint=True)
                frame_indices = [min(max_frame_idx, int(math.ceil(t * video_fps))) for t in target_seconds]

        # Remove duplicates while preserving order
        seen, uniq = set(), []
        for idx in frame_indices:
            if idx not in seen:
                seen.add(idx)
                uniq.append(idx)

        # Ensure even number of frames
        if len(uniq) & 1:
            uniq.append(uniq[-1])

        frame_indices = uniq
        sampled_video = video[frame_indices]
        full_second_idxs = [int(idx / video_fps) for idx in frame_indices]
        second_idxs = full_second_idxs[::2]  # mrope
        
        return sampled_video, second_idxs

    def rescale_and_normalize(
        self,
        videos: torch.Tensor,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: List[float],
        image_std: List[float],
    ) -> torch.Tensor:
        """Apply rescaling and normalization to video tensors."""
        if do_rescale:
            videos = videos * rescale_factor
        
        if do_normalize:
            mean = torch.tensor(image_mean, device=videos.device, dtype=videos.dtype)
            std = torch.tensor(image_std, device=videos.device, dtype=videos.dtype)
            
            # Reshape for broadcasting
            mean = mean.view(1, 1, -1, 1, 1)  # (1, 1, C, 1, 1)
            std = std.view(1, 1, -1, 1, 1)   # (1, 1, C, 1, 1)
            
            videos = (videos - mean) / std
        
        return videos

    def _preprocess(
        self,
        videos: List[torch.Tensor],
        video_metadata: Optional[List[Union[VideoMetadata, Dict]]] = None,
        do_convert_rgb: bool = True,
        do_resize: bool = True,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        do_sample_frames: bool = True,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> BatchFeature:
        """Internal preprocessing method."""
        timestamps_list = []
        
        if do_sample_frames:
            if video_metadata is None or (isinstance(video_metadata, list) and video_metadata[0] is None):
                raise ValueError(
                    "Frame sampling is enabled but no video metadata was found. "
                    "Please pass in `VideoMetadata` object per each input video or set `do_sample_frames=False`"
                )
            processed_videos = []
            for video, metadata in zip(videos, video_metadata):
                video, timestamps = self.sample_frames(video, metadata)
                timestamps_list.append(timestamps)
                processed_videos.append(video)
        else:
            processed_videos = videos
            timestamps_list = [[] for _ in videos]

        # Group videos by shape for batch processing
        grouped_videos, grouped_video_indices = group_videos_by_shape(processed_videos)
        resized_videos_grouped = {}

        # Resize videos
        for shape, stacked_videos in grouped_videos.items():
            B, T, C, H, W = stacked_videos.shape
            num_frames, height, width = T, H, W
            
            if do_resize:
                resized_height, resized_width = smart_resize(
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    temporal_factor=self.temporal_patch_size,
                    factor=self.patch_size * self.merge_size,
                    max_pixels=self.max_image_size["longest_edge"],
                )
                stacked_videos = stacked_videos.view(B * T, C, H, W)
                stacked_videos = F.interpolate(
                    stacked_videos, size=(resized_height, resized_width), mode="bicubic", align_corners=False
                )
                stacked_videos = stacked_videos.view(B, T, C, resized_height, resized_width)
            
            resized_videos_grouped[shape] = stacked_videos

        # Reorder videos back to original order
        resized_videos = reorder_videos(resized_videos_grouped, grouped_video_indices)

        # Group again for further processing
        grouped_videos, grouped_video_indices = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_grids = {}

        for shape, stacked_videos in grouped_videos.items():
            resized_height, resized_width = get_image_size(stacked_videos[0])

            # Apply rescaling and normalization
            stacked_videos = self.rescale_and_normalize(
                stacked_videos, do_rescale, rescale_factor, do_normalize, 
                image_mean or self.image_mean, image_std or self.image_std
            )
            patches = stacked_videos

            # Ensure frames are divisible by temporal_patch_size
            if patches.shape[1] % self.temporal_patch_size != 0:
                repeats = patches[:, -1:].repeat(1, self.temporal_patch_size - 1, 1, 1, 1)
                patches = torch.cat([patches, repeats], dim=1)

            batch_size, grid_t, channel = patches.shape[:3]
            grid_t = grid_t // self.temporal_patch_size
            grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size

            # Reshape for patch extraction
            patches = patches.view(
                batch_size,
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
            patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * self.temporal_patch_size * self.patch_size * self.patch_size,
            )

            processed_videos_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        # Reorder back and create final output
        processed_videos = reorder_videos(processed_videos_grouped, grouped_video_indices)
        processed_grids = reorder_videos(processed_grids, grouped_video_indices)
        
        pixel_values_videos = torch.cat(processed_videos, dim=0)
        video_grid_thw = torch.tensor(processed_grids)
        
        total_frames = video_grid_thw[0][0].item()
        h = video_grid_thw[0][1].item()
        w = video_grid_thw[0][2].item()
        video_grid_thw = [[1, h, w] for _ in range(total_frames)]

        data = {
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
            "timestamps": timestamps_list,
        }

        return BatchFeature(data=data, tensor_type=return_tensors)

    def preprocess(
        self,
        videos: Union[torch.Tensor, List[torch.Tensor]],
        video_metadata: Optional[List[Union[VideoMetadata, Dict]]] = None,
        do_sample_frames: Optional[bool] = None,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess videos for the GLM4V vision model.
        
        Args:
            videos: Video tensor(s) to preprocess
            video_metadata: Metadata for each video
            do_sample_frames: Whether to sample frames
            return_tensors: Type of tensors to return ("pt" for PyTorch)
            **kwargs: Additional processing parameters
            
        Returns:
            BatchFeature containing processed videos and metadata
        """
        # Ensure videos is a list
        if not isinstance(videos, list):
            videos = [videos]
        
        # Set defaults
        do_sample_frames = do_sample_frames if do_sample_frames is not None else self.do_sample_frames
        
        # If no metadata provided and sampling is enabled, create default metadata
        if do_sample_frames and video_metadata is None:
            video_metadata = [VideoMetadata() for _ in videos]
        
        return self._preprocess(
            videos=videos,
            video_metadata=video_metadata,
            do_sample_frames=do_sample_frames,
            return_tensors=return_tensors,
            **kwargs,
        )

    def get_number_of_video_patches(self, num_frames: int, height: int, width: int) -> int:
        """
        Get the number of patches for a video of given dimensions.
        
        Args:
            num_frames: Number of frames
            height: Video height
            width: Video width
            
        Returns:
            Number of patches
        """
        factor = self.patch_size * self.merge_size
        resized_height, resized_width = smart_resize(
            num_frames=num_frames,
            height=height,
            width=width,
            temporal_factor=self.temporal_patch_size,
            factor=factor,
            max_pixels=self.max_image_size["longest_edge"],
        )
        grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
        grid_t = num_frames // self.temporal_patch_size
        return grid_t * grid_h * grid_w


# =============================================================================
# Factory Functions
# =============================================================================

def create_glm4v_video_processor(
    patch_size: int = 14,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    fps: float = 2.0,
    **kwargs
) -> Glm4vVideoProcessor:
    """
    Create a GLM4V video processor with custom parameters.
    
    Args:
        patch_size: Size of spatial patches
        temporal_patch_size: Size of temporal patches
        merge_size: Spatial merge size
        fps: Target frames per second for sampling
        **kwargs: Additional processor parameters
        
    Returns:
        Configured video processor
    """
    return Glm4vVideoProcessor(
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        merge_size=merge_size,
        fps=fps,
        **kwargs
    ) 