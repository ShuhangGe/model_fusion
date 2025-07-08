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
GLM-4V Inference Tools

This module provides various inference interfaces for the GLM-4V model:
- CLI: Interactive command-line interface
- Web: Gradio-based web interface  
- API: vLLM API client tools
- Benchmark: Academic benchmarking tools
"""

from .engine import GLM4VInferenceEngine
from .base import BaseInference
from .config import InferenceConfig

# CLI Tools
from .cli.interactive import GLM4VCLIChat

# Web Interface
from .web.gradio_app import GLM4VGradioApp

# API Tools
from .api.vllm_client import GLM4VvLLMClient

# Benchmark Tools
from .benchmark.robust_inference import GLM4VRobustInference

__all__ = [
    # Core components
    "GLM4VInferenceEngine",
    "BaseInference", 
    "InferenceConfig",
    # CLI
    "GLM4VCLIChat",
    # Web
    "GLM4VGradioApp",
    # API
    "GLM4VvLLMClient",
    # Benchmark
    "GLM4VRobustInference",
] 