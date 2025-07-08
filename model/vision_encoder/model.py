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
Standalone Vision Model for GLM4V Vision Encoder
Extracted from GLM4V model to be independent of text components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union

from .config import VisionConfig
from .layers import (
    VisionEmbeddings,
    VisionPatchEmbed, 
    VisionRotaryEmbedding,
    VisionBlock,
    VisionPatchMerger,
    RMSNorm
)
from .utils import init_weights


class VisionModel(nn.Module):
    """
    Standalone Vision Model
    Extracted from Glm4vVisionModel to be independent of text components.
    
    This model processes vision inputs (images/videos) and outputs continuous embeddings.
    """
    
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size

        self.embeddings = VisionEmbeddings(config)
        self.patch_embed = VisionPatchEmbed(config)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([VisionBlock(config) for _ in range(config.depth)])
        self.merger = VisionPatchMerger(
            dim=config.out_hidden_size, context_dim=config.intermediate_size, hidden_act=config.hidden_act
        )

        self.post_conv_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.downsample = nn.Conv2d(
            in_channels=config.hidden_size,
            out_channels=config.out_hidden_size,
            kernel_size=config.spatial_merge_size,
            stride=config.spatial_merge_size,
        )
        self.post_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        
        # Initialize weights
        self.apply(lambda module: init_weights(module, config))

    def rot_pos_emb(self, grid_thw):
        """
        Generate rotary position embeddings for spatial coordinates.
        
        Args:
            grid_thw: Tensor of shape (num_images_or_videos, 3) representing temporal, height, width
            
        Returns:
            Tuple of (rotary_pos_emb, pos_ids)
        """
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb, pos_ids

    def _prepare_attention_mask(self, inputs_tensor: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        """
        Prepare attention mask for vision transformer blocks.
        
        Args:
            inputs_tensor: Input tensor
            cu_seqlens: Cumulative sequence lengths
            
        Returns:
            Attention mask or None for flash attention
        """
        # Flash Attention 2 doesn't need a 4D mask and relies on `cu_seqlens/max_seqlen`
        if self.config._attn_implementation == "flash_attention_2":
            return None

        seq_length = inputs_tensor.shape[0]
        attention_mask = torch.full(
            [1, 1, seq_length, seq_length],
            torch.finfo(inputs_tensor.dtype).min,
            device=inputs_tensor.device,
            dtype=inputs_tensor.dtype,
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
        return attention_mask

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the vision model.
        
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: Processed hidden states with vision embeddings.
        """
        # Apply patch embedding
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = self.post_conv_layernorm(hidden_states)

        # Generate rotary position embeddings
        rotary_pos_emb, image_type_ids = self.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # Prepare cumulative sequence lengths for attention
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        
        # Apply position embeddings
        hidden_states = self.embeddings(hidden_states, seqlens, grid_thw, image_type_ids[:, 0], image_type_ids[:, 1])
        
        # Prepare attention mask
        attention_mask = self._prepare_attention_mask(hidden_states, cu_seqlens=cu_seqlens)

        # Pass through transformer blocks
        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    blk,
                    hidden_states,
                    cu_seqlens,
                    position_embeddings,
                    attention_mask,
                )
            else:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                )

        # Final layer normalization
        hidden_states = self.post_layernorm(hidden_states)

        # Apply spatial merging through downsampling
        batch_size = len(seqlens)
        output_hidden_states = []
        start_idx = 0
        
        for i, (t, h, w) in enumerate(grid_thw):
            seq_len = seqlens[i]
            seq_hidden_states = hidden_states[start_idx:start_idx + seq_len]
            
            # Reshape for spatial operations
            # From (seq_len, hidden_size) to (t, h//merge, w//merge, hidden_size)
            merge_h = h // self.spatial_merge_size
            merge_w = w // self.spatial_merge_size
            seq_hidden_states = seq_hidden_states.view(t, merge_h, merge_w, -1)
            
            # Reshape for conv2d: (t, hidden_size, merge_h, merge_w)
            seq_hidden_states = seq_hidden_states.permute(0, 3, 1, 2)
            
            # Apply downsampling
            seq_hidden_states = seq_hidden_states.view(-1, seq_hidden_states.shape[1], merge_h, merge_w)
            seq_hidden_states = self.downsample(seq_hidden_states)
            
            # Reshape back to sequence format
            out_h = seq_hidden_states.shape[2]
            out_w = seq_hidden_states.shape[3]
            seq_hidden_states = seq_hidden_states.view(t, -1, out_h, out_w)
            seq_hidden_states = seq_hidden_states.permute(0, 2, 3, 1).contiguous()
            seq_hidden_states = seq_hidden_states.view(-1, seq_hidden_states.shape[-1])
            
            output_hidden_states.append(seq_hidden_states)
            start_idx += seq_len

        # Concatenate all processed sequences
        hidden_states = torch.cat(output_hidden_states, dim=0)
        
        return hidden_states

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        """
        Encodes images into continuous embeddings.

        Args:
            pixel_values (`torch.FloatTensor`): The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor`, *optional*): The temporal, height and width of feature shape.

        Returns:
            List of image embeddings.
        """
        pixel_values = pixel_values.to(dtype=next(self.parameters()).dtype)
        image_embeds = self.forward(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds

    def get_video_features(self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None):
        """
        Encodes videos into continuous embeddings.

        Args:
            pixel_values_videos (`torch.FloatTensor`): The tensors corresponding to the input videos.
            video_grid_thw (`torch.LongTensor`, *optional*): The temporal, height and width of feature shape.

        Returns:
            List of video embeddings.
        """
        pixel_values_videos = pixel_values_videos.to(dtype=next(self.parameters()).dtype)
        video_embeds = self.forward(pixel_values_videos, grid_thw=video_grid_thw)
        split_sizes = (video_grid_thw.prod(-1) // self.spatial_merge_size**2).tolist()
        video_embeds = torch.split(video_embeds, split_sizes)
        return video_embeds

    @classmethod
    def from_config(cls, config: VisionConfig):
        """
        Create a VisionModel from a configuration.
        
        Args:
            config: Vision configuration
            
        Returns:
            VisionModel instance
        """
        return cls(config)

    def save_pretrained(self, save_directory: str):
        """
        Save the model and configuration to a directory.
        
        Args:
            save_directory: Directory to save the model
        """
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Save model weights
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)

    @classmethod
    def from_pretrained(cls, model_directory: str):
        """
        Load a model from a directory.
        
        Args:
            model_directory: Directory containing the model
            
        Returns:
            VisionModel instance
        """
        import os
        import json
        
        # Load configuration
        config_path = os.path.join(model_directory, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = VisionConfig.from_dict(config_dict)
        
        # Create model
        model = cls(config)
        
        # Load weights
        model_path = os.path.join(model_directory, "pytorch_model.bin")
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        
        return model 