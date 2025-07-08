# Copyright 2025 The HuggingFace Team. All rights reserved.
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
Configuration management for GLM-4V inference tools.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch


@dataclass
class InferenceConfig:
    """Configuration class for GLM-4V inference settings."""
    
    # Model settings
    model_path: str = "THUDM/GLM-4.1V-9B-Thinking"
    device_map: str = "auto"
    torch_dtype: torch.dtype = torch.bfloat16
    use_fast_processor: bool = True
    attn_implementation: str = "sdpa"
    
    # Generation settings
    max_new_tokens: int = 8192
    temperature: float = 1.0
    top_k: int = 2
    top_p: Optional[float] = None
    repetition_penalty: float = 1.0
    do_sample: bool = True
    
    # Special token settings
    extract_answer_tags: bool = True
    thinking_tag_start: str = "<think>"
    thinking_tag_end: str = "</think>"
    answer_tag_start: str = "<answer>"
    answer_tag_end: str = "</answer>"
    
    # Multimodal settings
    max_images: int = 10
    max_videos: int = 1
    allow_mixed_media: bool = False  # images and videos together
    
    # Robust inference settings (for benchmarking)
    first_max_tokens: int = 8192
    force_max_tokens: int = 8192
    enable_force_completion: bool = True
    
    # Web interface settings
    server_name: str = "127.0.0.1"
    server_port: int = 7860
    share: bool = False
    enable_mcp_server: bool = False
    
    # API settings
    api_base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    
    # Additional kwargs for flexibility
    additional_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def to_generation_kwargs(self) -> Dict[str, Any]:
        """Convert to kwargs suitable for model.generate()"""
        kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature if self.temperature > 0 else None,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.do_sample,
        }
        
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
            
        kwargs.update(self.additional_kwargs)
        return kwargs
    
    def get_special_tokens(self) -> Dict[str, str]:
        """Get special tokens for thinking/answer extraction"""
        return {
            "think_start": self.thinking_tag_start,
            "think_end": self.thinking_tag_end,
            "answer_start": self.answer_tag_start,
            "answer_end": self.answer_tag_end,
        }


# Default configurations for different use cases
DEFAULT_CLI_CONFIG = InferenceConfig(
    temperature=1.0,
    max_new_tokens=8192,
)

DEFAULT_WEB_CONFIG = InferenceConfig(
    temperature=1.0,
    max_new_tokens=8192,
    server_name="127.0.0.1",
    server_port=7860,
)

DEFAULT_BENCHMARK_CONFIG = InferenceConfig(
    temperature=0.1,
    do_sample=True,
    enable_force_completion=True,
    first_max_tokens=8192,
    force_max_tokens=8192,
)

DEFAULT_API_CONFIG = InferenceConfig(
    temperature=1.0,
    max_new_tokens=25000,
    api_base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
) 