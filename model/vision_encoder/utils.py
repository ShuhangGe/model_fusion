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
Utilities for the standalone GLM4V Vision Encoder
Contains activation functions, attention utilities, and helper functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable


def silu(x):
    """
    SiLU activation function (Swish)
    """
    return x * torch.sigmoid(x)


def gelu(x):
    """
    GELU activation function
    """
    return F.gelu(x)


def relu(x):
    """
    ReLU activation function
    """
    return F.relu(x)


def gelu_new(x):
    """
    GELU activation function (new implementation)
    """
    return 0.5 * x * (1.0 + torch.tanh(0.79788456 * x * (1.0 + 0.044715 * x * x)))


# Activation function mapping
ACT2FN = {
    "gelu": gelu,
    "gelu_new": gelu_new,
    "relu": relu,
    "silu": silu,
    "swish": silu,
}


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to query and key tensors.
    
    Args:
        q: Query tensor
        k: Key tensor
        cos: Cosine components of rotary embeddings
        sin: Sine components of rotary embeddings
    
    Returns:
        Tuple of (rotated_q, rotated_k)
    """
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
    """
    Eager attention implementation for vision transformer.
    
    Args:
        module: The attention module
        query: Query tensor
        key: Key tensor  
        value: Value tensor
        attention_mask: Optional attention mask
        scaling: Scaling factor for attention scores
        dropout: Dropout probability
        
    Returns:
        Tuple of (attention_output, attention_weights)
    """
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


class BaseLayer(nn.Module):
    """
    Base layer class that provides common functionality for vision layers.
    Replaces functionality from transformers.modeling_layers.GradientCheckpointingLayer
    """
    
    def __init__(self):
        super().__init__()
        self.gradient_checkpointing = False

    def _gradient_checkpointing_func(self, *args, **kwargs):
        """
        Simple gradient checkpointing wrapper.
        """
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(self.forward, *args, **kwargs)
        else:
            return self.forward(*args, **kwargs)


def init_weights(module, config):
    """
    Initialize weights for vision model components.
    
    Args:
        module: The module to initialize
        config: Vision configuration containing initializer_range
    """
    std = config.initializer_range
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif hasattr(module, 'weight') and hasattr(module.weight, 'data'):
        # For RMSNorm and LayerNorm
        if hasattr(module, 'variance_epsilon'):  # RMSNorm
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()


def create_causal_mask(seq_length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Create a causal attention mask.
    
    Args:
        seq_length: Length of the sequence
        device: Device to create the mask on
        dtype: Data type for the mask
        
    Returns:
        Causal attention mask
    """
    mask = torch.full((seq_length, seq_length), float('-inf'), dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions


def get_activation_fn(activation_string: str):
    """
    Get activation function by name.
    
    Args:
        activation_string: Name of the activation function
        
    Returns:
        Activation function
    """
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}") 