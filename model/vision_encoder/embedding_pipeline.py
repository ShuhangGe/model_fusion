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

"""GLM4V Embedding Pipeline for extracting visual embeddings."""

from typing import Optional, Union, List, Dict, Any
import torch
import torch.nn as nn

from .config import Glm4vVisionConfig
from .vision_model import Glm4vVisionModel
from .processing import Glm4vImageProcessor, Glm4vVideoProcessor


class GLM4VEmbeddingPipeline(nn.Module):
    """
    Complete GLM4V embedding pipeline for extracting visual embeddings from images and videos.
    
    This pipeline combines the GLM4V vision model with image/video processors to provide
    a unified interface for extracting embeddings that can be used as input to language models.
    """
    
    def __init__(self, config: Optional[Glm4vVisionConfig] = None):
        """
        Initialize the GLM4V embedding pipeline.
        
        Args:
            config (Glm4vVisionConfig, optional): Vision configuration. If None, uses default config.
        """
        super().__init__()
        
        if config is None:
            config = Glm4vVisionConfig()
        
        self.config = config
        self.vision_model = Glm4vVisionModel(config)
        self.image_processor = Glm4vImageProcessor(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            merge_size=config.spatial_merge_size,
        )
        self.video_processor = Glm4vVideoProcessor(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            merge_size=config.spatial_merge_size,
        )
    
    def extract_image_embeddings(
        self,
        images,
        return_dict: bool = True,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Extract embeddings from images.
        
        Args:
            images: Input images (PIL.Image, numpy array, or torch.Tensor)
            return_dict (bool): Whether to return a dictionary with additional info
            **kwargs: Additional arguments for image processing
            
        Returns:
            torch.Tensor or dict: Image embeddings or dictionary containing embeddings and metadata
        """
        # Process images
        processed_data = self.image_processor(
            images=images,
            return_tensors="pt",
            **kwargs
        )
        
        pixel_values = processed_data["pixel_values"]
        image_grid_thw = processed_data["image_grid_thw"]
        
        # Convert to tensors if needed
        if not isinstance(pixel_values, torch.Tensor):
            pixel_values = torch.tensor(pixel_values)
        if not isinstance(image_grid_thw, torch.Tensor):
            image_grid_thw = torch.tensor(image_grid_thw)
        
        # Move to same device as model
        device = next(self.vision_model.parameters()).device
        pixel_values = pixel_values.to(device)
        image_grid_thw = image_grid_thw.to(device)
        
        # Extract embeddings through vision model
        with torch.no_grad():
            embeddings = self.vision_model(pixel_values, grid_thw=image_grid_thw)
        
        if return_dict:
            return {
                "embeddings": embeddings,
                "image_grid_thw": image_grid_thw,
                "num_patches": embeddings.shape[0],
                "embedding_dim": embeddings.shape[1],
            }
        else:
            return embeddings
    
    def extract_video_embeddings(
        self,
        videos,
        video_metadata=None,
        return_dict: bool = True,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Extract embeddings from videos.
        
        Args:
            videos: Input videos (torch.Tensor or list of tensors)
            video_metadata: Video metadata for frame sampling
            return_dict (bool): Whether to return a dictionary with additional info
            **kwargs: Additional arguments for video processing
            
        Returns:
            torch.Tensor or dict: Video embeddings or dictionary containing embeddings and metadata
        """
        # Process videos
        processed_data = self.video_processor._preprocess(
            videos=[videos] if not isinstance(videos, list) else videos,
            video_metadata=video_metadata,
            return_tensors="pt",
            **kwargs
        )
        
        pixel_values_videos = processed_data["pixel_values_videos"]
        video_grid_thw = processed_data["video_grid_thw"]
        timestamps = processed_data.get("timestamps", [])
        
        # Convert to tensors if needed
        if not isinstance(pixel_values_videos, torch.Tensor):
            pixel_values_videos = torch.tensor(pixel_values_videos)
        if not isinstance(video_grid_thw, torch.Tensor):
            video_grid_thw = torch.tensor(video_grid_thw)
        
        # Move to same device as model
        device = next(self.vision_model.parameters()).device
        pixel_values_videos = pixel_values_videos.to(device)
        video_grid_thw = video_grid_thw.to(device)
        
        # Extract embeddings through vision model
        with torch.no_grad():
            embeddings = self.vision_model(pixel_values_videos, grid_thw=video_grid_thw)
        
        if return_dict:
            return {
                "embeddings": embeddings,
                "video_grid_thw": video_grid_thw,
                "timestamps": timestamps,
                "num_patches": embeddings.shape[0],
                "embedding_dim": embeddings.shape[1],
            }
        else:
            return embeddings
    
    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.
        
        This method is compatible with the GLM4V model interface.
        
        Args:
            pixel_values (`torch.FloatTensor`): The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor`, optional): The temporal, height and width of feature shape.
        
        Returns:
            List[torch.Tensor]: List of image embeddings split by image.
        """
        pixel_values = pixel_values.type(self.vision_model.dtype)
        image_embeds = self.vision_model(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.vision_model.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds
    
    def get_video_features(self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model.
        
        This method is compatible with the GLM4V model interface.
        
        Args:
            pixel_values_videos (`torch.FloatTensor`): The tensors corresponding to the input videos.
            video_grid_thw (`torch.LongTensor`, optional): The temporal, height and width of feature shape.
        
        Returns:
            List[torch.Tensor]: List of video embeddings split by video.
        """
        pixel_values_videos = pixel_values_videos.type(self.vision_model.dtype)
        video_embeds = self.vision_model(pixel_values_videos, grid_thw=video_grid_thw)
        split_sizes = (video_grid_thw.prod(-1) // self.vision_model.spatial_merge_size**2).tolist()
        video_embeds = torch.split(video_embeds, split_sizes)
        return video_embeds
    
    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the vision model.
        
        Args:
            pixel_values: Processed pixel values
            grid_thw: Grid temporal, height, width information
            
        Returns:
            torch.Tensor: Vision embeddings
        """
        return self.vision_model(pixel_values, grid_thw)
    
    def to(self, device):
        """Move the pipeline to specified device."""
        self.vision_model = self.vision_model.to(device)
        return self
    
    def eval(self):
        """Set the pipeline to evaluation mode."""
        self.vision_model.eval()
        return self
    
    def train(self, mode: bool = True):
        """Set the pipeline to training mode."""
        self.vision_model.train(mode)
        return self
    
    @property
    def device(self):
        """Get the device of the vision model."""
        return next(self.vision_model.parameters()).device
    
    @property
    def dtype(self):
        """Get the dtype of the vision model."""
        return next(self.vision_model.parameters()).dtype


__all__ = ["GLM4VEmbeddingPipeline"] 