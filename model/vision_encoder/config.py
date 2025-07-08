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
Standalone Vision Configuration for GLM4V Vision Encoder
Extracted from GLM4V model to be independent of text components.
"""


class VisionConfig:
    r"""
    This is the configuration class to store the configuration of a Vision Model. It is used to instantiate a Vision
    model according to the specified arguments, defining the model architecture.

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
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for attention weights.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        image_size (`int` or `list[int]`, *optional*, defaults to 336):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_hidden_size (`int`, *optional*, defaults to 4096):
            The output hidden size of the vision model.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        spatial_merge_size (`int`, *optional*, defaults to 2):
            The size used for merging spatial dimensions.
        temporal_patch_size (`int`, *optional*, defaults to 1):
            The size used for patches along the temporal dimension.
        num_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        _attn_implementation (`str`, *optional*, defaults to "eager"):
            The attention implementation to use. Can be "eager", "flash_attention_2", or "sdpa".

    Example:

    ```python
    >>> from model.vision_encoder import VisionConfig, VisionModel

    >>> # Initializing a VisionConfig with GLM-4V style configuration
    >>> configuration = VisionConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = VisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "glm4v_vision"

    def __init__(
        self,
        depth=24,
        hidden_size=1536,
        hidden_act="silu",
        attention_bias=False,
        attention_dropout=0.0,
        num_heads=12,
        in_channels=3,
        image_size=336,
        patch_size=14,
        rms_norm_eps=1e-05,
        spatial_merge_size=2,
        temporal_patch_size=1,
        out_hidden_size=4096,
        intermediate_size=13696,
        initializer_range=0.02,
        _attn_implementation="eager",
        **kwargs,
    ):
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.intermediate_size = intermediate_size
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self._attn_implementation = _attn_implementation

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.
        """
        output = {}
        for key, value in self.__dict__.items():
            output[key] = value
        return output

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """
        Instantiates a VisionConfig from a Python dictionary of parameters.
        """
        return cls(**config_dict, **kwargs) 