# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
Qwen models for the GLM-4.1V-Thinking project.

This package contains Qwen3 and Qwen3-MoE models with fixed imports 
to work with the local transformers structure.
"""

# Qwen3 models
from .qwen3.configuration_qwen3 import Qwen3Config
from .qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM,
    Qwen3ForQuestionAnswering,
    Qwen3ForSequenceClassification,
    Qwen3ForTokenClassification,
    Qwen3Model,
    Qwen3PreTrainedModel,
)

# Qwen3-MoE models
from .qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from .qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeForCausalLM,
    Qwen3MoeForQuestionAnswering,
    Qwen3MoeForSequenceClassification,
    Qwen3MoeForTokenClassification,
    Qwen3MoeModel,
    Qwen3MoePreTrainedModel,
)

__all__ = [
    # Qwen3
    "Qwen3Config",
    "Qwen3ForCausalLM",
    "Qwen3ForQuestionAnswering", 
    "Qwen3ForSequenceClassification",
    "Qwen3ForTokenClassification",
    "Qwen3Model",
    "Qwen3PreTrainedModel",
    # Qwen3-MoE
    "Qwen3MoeConfig",
    "Qwen3MoeForCausalLM",
    "Qwen3MoeForQuestionAnswering",
    "Qwen3MoeForSequenceClassification", 
    "Qwen3MoeForTokenClassification",
    "Qwen3MoeModel",
    "Qwen3MoePreTrainedModel",
] 