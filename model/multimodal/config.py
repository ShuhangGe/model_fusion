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

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation


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
    """

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


class Glm4vTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Glm4vModel`]. It is used to instantiate a
    GLM-4.1V model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of
    GLM-4.1V-9B-Thinking [THUDM/GLM-4.1V-9B-Thinking](https://huggingface.co/THUDM/GLM-4.1V-9B-Thinking).
    """

    model_type = "glm4v_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151552,
        hidden_size=4096,
        intermediate_size=13696,
        num_hidden_layers=40,
        num_attention_heads=32,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        attention_dropout=0.0,
        rope_scaling=None,
        image_token_id=None,
        video_token_id=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.rope_scaling = rope_scaling

        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self, ignore_keys={"mrope_section"})
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class Glm4vConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Glm4vModel`]. It is used to instantiate a
    GLM-4.1V model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of
    GLM-4.1V-9B-Thinking [THUDM/GLM-4.1V-9B-Thinking](https://huggingface.co/THUDM/GLM-4.1V-9B-Thinking).

    Args:
        text_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Glm4vTextConfig`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PreTrainedConfig, dict]`,  *optional*, defaults to `Glm4vVisionConfig`):
            The config object or dictionary of the vision backbone.
        image_token_id (`int`, *optional*, defaults to 151343):
            The image token index to encode the image prompt.
        video_token_id (`int`, *optional*, defaults to 151344):
            The video token index to encode the image prompt.
        image_start_token_id (`int`, *optional*, defaults to 151339):
            The image start token index to encode the start of image.
        image_end_token_id (`int`, *optional*, defaults to 151340):
            The image end token index to encode the end of image.
        video_start_token_id (`int`, *optional*, defaults to 151341):
            The video start token index to encode the start of video.
        video_end_token_id (`int`, *optional*, defaults to 151342):
            The video end token index to encode the end of video.
    """

    model_type = "glm4v"
    sub_configs = {"vision_config": Glm4vVisionConfig, "text_config": Glm4vTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=151343,
        video_token_id=151344,
        image_start_token_id=151339,
        image_end_token_id=151340,
        video_start_token_id=151341,
        video_end_token_id=151342,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            # For BC use all kwargs to init `TextConfig`
            self.text_config = self.sub_configs["text_config"](**kwargs)

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id

    def get_text_config(self):
        return self.text_config

    def get_vision_config(self):
        return self.vision_config


__all__ = ["Glm4vConfig", "Glm4vTextConfig", "Glm4vVisionConfig"] 