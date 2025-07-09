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

from ...transformers.configuration_utils import PretrainedConfig
from ...transformers.modeling_rope_utils import rope_config_validation


class Glm4vVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Glm4vVisionModel`]. It is used to instantiate an Glm4vVisionModel
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield
    a similar configuration to that of
    GLM-4.1V-9B-Thinking [THUDM/GLM-4.1V-9B-Thinking](https://huggingface.co/THUDM/GLM-4.1V-9B-Thinking).

    Args:
        hidden_size (`int`, *optional*, defaults to 1536):
            Dimensionality of the encoder layers and the pooler layer.
        depth (`int`, *optional*, defaults to 24):
            Number of layers (depth) in the model.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to add a bias to the queries, keys and values.
        intermediate_size (`int`, *optional*, defaults to 13696):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"selu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for attention weights.
        projection_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for the projection layer.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        image_size (`int` or `list[int]`, *optional*, defaults to `[336, 336]`):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to `14`):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_hidden_size (`int`, *optional*, defaults to 4096):
            The output hidden size of the vision model.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        spatial_merge_size (`int`, *optional*, defaults to 2):
            The size used for merging spatial dimensions.
        temporal_patch_size (`int`, *optional*, defaults to 2):
            The size used for patches along the temporal dimension.
    Example:

    ```python
    >>> from transformers import Glm4vVisionConfig, Glm4vVisionModel

    >>> # Initializing a Glm4vVisionConfig GLM-4.1V-9B style configuration
    >>> configuration = Glm4vVisionConfig()

    >>> # Initializing a model (with random weights) from the GLM-4.1V-9B configuration
    >>> model = Glm4vVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "glm4v"
    base_config_key = "vision_config"

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
        **kwargs,
    ):
        super().__init__(**kwargs)

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


__all__ = ["Glm4vVisionConfig"] 