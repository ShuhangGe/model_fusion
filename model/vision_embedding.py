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
GLM4V Vision Encoder - Standalone Implementation

This module contains the complete GLM4V vision encoding pipeline extracted as a standalone component.
It includes the vision model, image/video processing, and all necessary utility functions.
"""

import itertools
import math
from typing import Callable, Optional, Union, Any, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import LayerNorm

# =============================================================================
# Utility Functions
# =============================================================================

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def smart_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 2,
    factor: int = 28,
    min_pixels: int = 112 * 112,
    max_pixels: int = 14 * 14 * 2 * 2 * 2 * 6144,
):
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


# =============================================================================
# Configuration Classes  
# =============================================================================

@dataclass
class Glm4vVisionConfig:
    """
    Configuration class for GLM4V vision model.
    
    This is the configuration class to store the configuration of a [`Glm4vVisionModel`]. It is used to instantiate an Glm4vVisionModel
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield
    a similar configuration to that of GLM-4.1V-9B-Thinking.

    Args:
        hidden_size (`int`, *optional*, defaults to 1536):
            Dimensionality of the encoder layers and the pooler layer.
        depth (`int`, *optional*, defaults to 24):
            Number of layers (depth) in the model.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to add a bias to the queries, keys and values.
        intermediate_size (`int`, *optional*, defaults to 13696):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for attention weights.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        image_size (`int`, *optional*, defaults to 336):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_hidden_size (`int`, *optional*, defaults to 4096):
            The output hidden size of the vision model.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        spatial_merge_size (`int`, *optional*, defaults to 2):
            The size used for merging spatial dimensions.
        temporal_patch_size (`int`, *optional*, defaults to 1):
            The size used for patches along the temporal dimension.
    """
    depth: int = 24
    hidden_size: int = 1536
    hidden_act: str = "silu"
    attention_bias: bool = False
    attention_dropout: float = 0.0
    num_heads: int = 12
    in_channels: int = 3
    image_size: int = 336
    patch_size: int = 14
    rms_norm_eps: float = 1e-05
    spatial_merge_size: int = 2
    temporal_patch_size: int = 1
    out_hidden_size: int = 4096
    intermediate_size: int = 13696
    initializer_range: float = 0.02
    _attn_implementation: str = "eager"


# =============================================================================
# Activation Functions
# =============================================================================

def get_activation_fn(activation: str):
    """Get activation function by name."""
    if activation == "silu":
        return F.silu
    elif activation == "gelu":
        return F.gelu
    elif activation == "relu":
        return F.relu
    elif activation == "swish":
        return F.silu  # swish is same as silu
    else:
        raise ValueError(f"Unsupported activation: {activation}")


# =============================================================================
# Model Components
# =============================================================================

class Glm4vRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Glm4vRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Glm4VisionMlp(nn.Module):
    def __init__(self, config, bias: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.out_hidden_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = get_activation_fn(config.hidden_act)

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class Glm4vVisionPatchEmbed(nn.Module):
    def __init__(self, config: Glm4vVisionConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class Glm4vVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Glm4vVisionPatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, hidden_act: str, bias: bool = False) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.post_projection_norm = LayerNorm(dim)
        self.gate_proj = nn.Linear(dim, context_dim, bias=bias)
        self.up_proj = nn.Linear(dim, context_dim, bias=bias)
        self.down_proj = nn.Linear(context_dim, dim, bias=bias)
        self.act1 = nn.GELU()
        self.act_fn = get_activation_fn(hidden_act)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.proj(hidden_state)
        hidden_state = self.act1(self.post_projection_norm(hidden_state))
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class Glm4vVisionEmbeddings(nn.Module):
    def __init__(self, config: Glm4vVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, embeddings, lengths, image_shapes, h_coords, w_coords) -> torch.Tensor:
        """
        Forward pass with integrated position encoding adaptation using 2D interpolation.

        Args:
            embeddings: Input embeddings tensor
            lengths (torch.Tensor): Sequence lengths for each image in the batch.
            image_shapes (torch.Tensor): Tensor of shape [batch_size, 3] representing the image shapes (t, h, w).
            h_coords (torch.Tensor): Tensor of shape [total_seq] representing the h coordinate for each patch.
            w_coords (torch.Tensor): Tensor of shape [total_seq] representing the w coordinate for each patch.

        Returns:
            torch.Tensor: Embeddings with adapted position encoding added.
        """
        # Get position embedding parameters
        pos_embed_weight = self.position_embedding.weight
        hidden_size = pos_embed_weight.shape[1]
        total_seq = h_coords.shape[0]
        device = pos_embed_weight.device

        # Move coordinates to correct device
        h_coords, w_coords = h_coords.to(device), w_coords.to(device)

        # Handle empty sequence case
        if total_seq == 0:
            adapted_pos_embed = torch.empty(0, hidden_size, device=device, dtype=pos_embed_weight.dtype)
        else:
            # Convert inputs to tensors if needed
            if isinstance(lengths, list):
                lengths = torch.tensor(lengths, device=device, dtype=torch.long)
            if not isinstance(image_shapes, torch.Tensor):
                image_shapes = torch.tensor(image_shapes, device=device, dtype=torch.long)

            # Prepare 2D position embedding
            orig_size_sq = pos_embed_weight.shape[0]
            orig_size = int(orig_size_sq**0.5)
            pos_embed_2d = (
                pos_embed_weight.view(orig_size, orig_size, hidden_size)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device=device, dtype=torch.float32)
            )

            # Calculate target dimensions for each patch
            target_h = torch.cat([image_shapes[i, 1].repeat(lengths[i]) for i in range(len(lengths))]).to(
                device=device, dtype=torch.float32
            )
            target_w = torch.cat([image_shapes[i, 2].repeat(lengths[i]) for i in range(len(lengths))]).to(
                device=device, dtype=torch.float32
            )

            # Normalize coordinates to [-1, 1] range for grid_sample
            h_coords = h_coords.to(device=device, dtype=torch.float32)
            w_coords = w_coords.to(device=device, dtype=torch.float32)
            norm_w = ((w_coords + 0.5) / target_w) * 2 - 1
            norm_h = ((h_coords + 0.5) / target_h) * 2 - 1

            # Create sampling grid
            grid = torch.stack((norm_w, norm_h), dim=-1).unsqueeze(0).unsqueeze(2)

            # Perform bicubic interpolation
            interpolated_embed_fp32 = F.grid_sample(
                pos_embed_2d, grid, mode="bicubic", align_corners=False, padding_mode="border"
            )

            # Reshape and convert back to original dtype
            adapted_pos_embed_fp32 = interpolated_embed_fp32.squeeze(0).squeeze(-1).permute(1, 0)
            adapted_pos_embed = adapted_pos_embed_fp32.to(pos_embed_weight.dtype).to(embeddings.device)

        # Add adapted position encoding to embeddings
        embeddings = embeddings + adapted_pos_embed
        return embeddings


class Glm4vVisionAttention(nn.Module):
    def __init__(self, config: Glm4vVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=config.attention_bias)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        if position_embeddings is None:
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        attention_interface: Callable = eager_attention_forward

        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class Glm4vVisionBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.norm1 = Glm4vRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = Glm4vRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = Glm4vVisionAttention(config)
        self.mlp = Glm4VisionMlp(config, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


# =============================================================================
# Main Vision Model
# =============================================================================

class Glm4vVisionModel(nn.Module):
    """
    GLM4V standalone vision encoder model.
    
    This model can process images and videos independently and output vision embeddings
    that can be used with any language model.
    """
    
    def __init__(self, config: Glm4vVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size

        self.embeddings = Glm4vVisionEmbeddings(config)
        self.patch_embed = Glm4vVisionPatchEmbed(config)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Glm4vVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([Glm4vVisionBlock(config) for _ in range(config.depth)])
        self.merger = Glm4vVisionPatchMerger(
            dim=config.out_hidden_size, context_dim=config.intermediate_size, hidden_act=config.hidden_act
        )

        self.post_conv_layernorm = Glm4vRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.downsample = nn.Conv2d(
            in_channels=config.hidden_size,
            out_channels=config.out_hidden_size,
            kernel_size=config.spatial_merge_size,
            stride=config.spatial_merge_size,
        )
        self.post_layernorm = Glm4vRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, Glm4vRMSNorm):
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.LayerNorm):
                module.weight.data.fill_(1.0)
                module.bias.data.zero_()

    def rot_pos_emb(self, grid_thw):
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
        """Prepare attention mask for vision attention."""
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
            `torch.Tensor`: Processed vision embeddings.
        """
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = self.post_conv_layernorm(hidden_states)

        rotary_pos_emb, image_type_ids = self.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        hidden_states = self.embeddings(hidden_states, seqlens, grid_thw, image_type_ids[:, 0], image_type_ids[:, 1])
        attention_mask = self._prepare_attention_mask(hidden_states, cu_seqlens=cu_seqlens)

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )

        hidden_states = self.post_layernorm(hidden_states)

        # Merge patches and create final embeddings
        split_sizes = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        hidden_states = torch.split(hidden_states, split_sizes)
        
        outputs = []
        for i, hidden_state in enumerate(hidden_states):
            h = int(grid_thw[i, 1].item())
            w = int(grid_thw[i, 2].item())
            
            # Reshape to spatial format
            hidden_state = hidden_state.view(h, w, -1).permute(2, 0, 1).unsqueeze(0)
            
            # Apply downsampling
            hidden_state = self.downsample(hidden_state)
            
            # Flatten back
            hidden_state = hidden_state.squeeze(0).permute(1, 2, 0).flatten(0, 1)
            
            # Apply merger
            hidden_state = self.merger(hidden_state)
            outputs.append(hidden_state)
        
        return torch.cat(outputs, dim=0)


# =============================================================================
# Factory Functions
# =============================================================================

def create_glm4v_vision_encoder(
    hidden_size: int = 1536,
    depth: int = 24,
    num_heads: int = 12,
    image_size: int = 336,
    patch_size: int = 14,
    out_hidden_size: int = 4096,
    **kwargs
) -> Glm4vVisionModel:
    """
    Factory function to create a GLM4V vision encoder with custom parameters.
    
    Args:
        hidden_size: Dimensionality of the encoder layers
        depth: Number of transformer layers
        num_heads: Number of attention heads
        image_size: Input image size
        patch_size: Patch size for vision transformer
        out_hidden_size: Output hidden size
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured GLM4V vision model
    """
    config = Glm4vVisionConfig(
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        image_size=image_size,
        patch_size=patch_size,
        out_hidden_size=out_hidden_size,
        **kwargs
    )
    return Glm4vVisionModel(config)


def create_glm4v_vision_encoder_default() -> Glm4vVisionModel:
    """
    Create GLM4V vision encoder with default GLM-4.1V-9B-Thinking configuration.
    
    Returns:
        GLM4V vision model with default configuration
    """
    return create_glm4v_vision_encoder() 