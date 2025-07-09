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
GLM4V Preprocessing Pipeline

This module provides the complete preprocessing pipeline that handles all steps
before the GLM4V TextModel input, including:

1. Multimodal input processing (text + images/videos)
2. Vision embedding extraction
3. Text-vision fusion via masked_scatter
4. 3D RoPE position embedding calculation
5. Preparation of unified multimodal embeddings

The output of this pipeline is ready to be fed directly into a language model.
"""

from typing import Optional, Union, List, Dict, Any
import torch
import torch.nn as nn

from .config import Glm4vConfig, Glm4vVisionConfig, Glm4vTextConfig
from .processor import Glm4vProcessor
from .image_processing import Glm4vImageProcessor
from .video_processing import Glm4vVideoProcessor
from .fusion_model import Glm4vMultimodalFusion


class GLM4VPreprocessingPipeline(nn.Module):
    """
    Complete GLM4V preprocessing pipeline that handles everything before TextModel input.
    
    This pipeline combines:
    - Glm4vProcessor: Unified text+vision processing
    - Glm4vMultimodalFusion: Vision embedding extraction and fusion
    - 3D RoPE: Spatial-temporal position embedding calculation
    
    The output is ready for direct input to a language model.
    """
    
    def __init__(
        self, 
        config: Optional[Glm4vConfig] = None,
        tokenizer=None,
        image_processor: Optional[Glm4vImageProcessor] = None,
        video_processor: Optional[Glm4vVideoProcessor] = None
    ):
        """
        Initialize the complete preprocessing pipeline.
        
        Args:
            config: GLM4V configuration
            tokenizer: Text tokenizer (required for processing)
            image_processor: Image processor (will create default if None)
            video_processor: Video processor (will create default if None)
        """
        super().__init__()
        
        if config is None:
            config = Glm4vConfig()
        
        self.config = config
        
        # Initialize processors
        if image_processor is None:
            image_processor = Glm4vImageProcessor(
                patch_size=config.vision_config.patch_size,
                temporal_patch_size=config.vision_config.temporal_patch_size,
                merge_size=config.vision_config.spatial_merge_size,
            )
        
        if video_processor is None:
            video_processor = Glm4vVideoProcessor(
                patch_size=config.vision_config.patch_size,
                temporal_patch_size=config.vision_config.temporal_patch_size,
                merge_size=config.vision_config.spatial_merge_size,
            )
        
        # Main processor for unified text+vision handling
        if tokenizer is not None:
            self.processor = Glm4vProcessor(
                image_processor=image_processor,
                tokenizer=tokenizer,
                video_processor=video_processor
            )
        else:
            self.processor = None
            
        # Multimodal fusion model
        self.fusion_model = Glm4vMultimodalFusion(config)
        
        # Store individual processors for direct access
        self.image_processor = image_processor
        self.video_processor = video_processor
        self.tokenizer = tokenizer
    
    def process_inputs(
        self,
        text: Union[str, List[str]] = None,
        images = None,
        videos = None,
        return_tensors: str = "pt",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process multimodal inputs using the unified processor.
        
        Args:
            text: Input text (can include image/video tokens)
            images: Input images
            videos: Input videos  
            return_tensors: Format for returned tensors
            **kwargs: Additional processing arguments
            
        Returns:
            Dict containing processed inputs ready for fusion
        """
        if self.processor is None:
            raise ValueError("Tokenizer is required for input processing")
            
        return self.processor(
            text=text,
            images=images,
            videos=videos,
            return_tensors=return_tensors,
            **kwargs
        )
    
    def extract_and_fuse_embeddings(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Extract vision embeddings and fuse with text embeddings.
        
        This method handles:
        1. Vision embedding extraction via GLM4V vision model
        2. Text-vision fusion via masked_scatter operation
        3. 3D RoPE position embedding calculation
        
        Args:
            input_ids: Text token IDs
            attention_mask: Attention mask
            position_ids: Position IDs (will be calculated if None)
            inputs_embeds: Pre-computed input embeddings
            pixel_values: Processed image data
            pixel_values_videos: Processed video data
            image_grid_thw: Image spatial layout info
            video_grid_thw: Video spatial layout info
            rope_deltas: RoPE delta values
            cache_position: Cache position for generation
            **kwargs: Additional arguments
            
        Returns:
            Dict containing fused embeddings and metadata ready for language model
        """
        return self.fusion_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            **kwargs
        )
    
    def forward(
        self,
        text: Union[str, List[str]] = None,
        images = None,
        videos = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        return_dict: bool = True,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Complete preprocessing pipeline from raw inputs to language model ready embeddings.
        
        Args:
            text: Input text with image/video tokens
            images: Raw images
            videos: Raw videos
            input_ids: Pre-tokenized input IDs (if text not provided)
            attention_mask: Attention mask
            pixel_values: Pre-processed image data
            pixel_values_videos: Pre-processed video data  
            image_grid_thw: Image grid layout
            video_grid_thw: Video grid layout
            return_dict: Whether to return dictionary or tensor
            **kwargs: Additional arguments
            
        Returns:
            Either fused embeddings tensor or dict with detailed outputs
        """
        
        # Step 1: Process raw inputs if provided
        if text is not None or images is not None or videos is not None:
            if self.processor is None:
                raise ValueError("Tokenizer is required for processing raw text/images/videos")
                
            processed_inputs = self.process_inputs(
                text=text,
                images=images, 
                videos=videos,
                return_tensors="pt",
                **kwargs
            )
            
            # Extract processed data
            input_ids = processed_inputs.get("input_ids", input_ids)
            attention_mask = processed_inputs.get("attention_mask", attention_mask)
            pixel_values = processed_inputs.get("pixel_values", pixel_values)
            pixel_values_videos = processed_inputs.get("pixel_values_videos", pixel_values_videos)
            image_grid_thw = processed_inputs.get("image_grid_thw", image_grid_thw)
            video_grid_thw = processed_inputs.get("video_grid_thw", video_grid_thw)
        
        # Step 2: Extract and fuse embeddings
        fusion_outputs = self.extract_and_fuse_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            **kwargs
        )
        
        if return_dict:
            return {
                "inputs_embeds": fusion_outputs["inputs_embeds"],
                "position_ids": fusion_outputs["position_ids"],
                "attention_mask": fusion_outputs["attention_mask"],
                "rope_deltas": fusion_outputs["rope_deltas"],
                # Additional metadata
                "has_images": fusion_outputs["has_images"],
                "has_videos": fusion_outputs["has_videos"],
                "image_grid_thw": fusion_outputs["image_grid_thw"],
                "video_grid_thw": fusion_outputs["video_grid_thw"],
                # Input processing metadata
                "input_ids": input_ids,
                "original_text": text,
                "num_image_tokens": (input_ids == self.config.image_token_id).sum() if input_ids is not None else 0,
                "sequence_length": fusion_outputs["inputs_embeds"].shape[1],
                "hidden_size": fusion_outputs["inputs_embeds"].shape[2],
            }
        else:
            return fusion_outputs["inputs_embeds"]
    
    def get_text_embeddings_only(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Extract text embeddings only (no vision fusion)."""
        return self.fusion_model.get_input_embeddings()(input_ids)
    
    def get_vision_embeddings_only(
        self, 
        pixel_values: torch.Tensor = None,
        pixel_values_videos: torch.Tensor = None,
        image_grid_thw: torch.LongTensor = None,
        video_grid_thw: torch.LongTensor = None
    ) -> Dict[str, torch.Tensor]:
        """Extract vision embeddings only (no text fusion)."""
        results = {}
        
        if pixel_values is not None:
            image_embeds = self.fusion_model.get_image_features(pixel_values, image_grid_thw)
            results["image_embeddings"] = torch.cat(image_embeds, dim=0) if image_embeds else None
            
        if pixel_values_videos is not None:
            video_embeds = self.fusion_model.get_video_features(pixel_values_videos, video_grid_thw)
            results["video_embeddings"] = torch.cat(video_embeds, dim=0) if video_embeds else None
            
        return results
    
    def to(self, device):
        """Move the pipeline to specified device."""
        self.fusion_model = self.fusion_model.to(device)
        return self
    
    def eval(self):
        """Set the pipeline to evaluation mode."""
        self.fusion_model.eval()
        return self
    
    def train(self, mode: bool = True):
        """Set the pipeline to training mode."""
        self.fusion_model.train(mode)
        return self
    
    @property
    def device(self):
        """Get the device of the fusion model."""
        return next(self.fusion_model.parameters()).device
    
    @property
    def dtype(self):
        """Get the dtype of the fusion model."""
        return next(self.fusion_model.parameters()).dtype


__all__ = ["GLM4VPreprocessingPipeline"] 