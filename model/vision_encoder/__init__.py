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
GLM4V Embedding Pipeline

This module provides a complete embedding pipeline extracted from GLM4V for processing
images and videos into embeddings that can be used with language models.

Key Components:
- GLM4VEmbeddingPipeline: Main interface for extracting embeddings
- Glm4vVisionModel: Core vision transformer model
- Glm4vImageProcessor: Image preprocessing pipeline
- Glm4vVideoProcessor: Video preprocessing pipeline
- Glm4vVisionConfig: Configuration for vision model

Usage:
    from model.encoder import GLM4VEmbeddingPipeline
    
    # Initialize pipeline
    pipeline = GLM4VEmbeddingPipeline()
    
    # Extract image embeddings
    image_embeddings = pipeline.extract_image_embeddings(images)
    
    # Extract video embeddings
    video_embeddings = pipeline.extract_video_embeddings(videos, video_metadata)
"""

# Configuration
from .config import Glm4vVisionConfig

# Vision Model Components
from .vision_model import (
    Glm4vRMSNorm,
    Glm4VisionMlp,
    Glm4vVisionPatchEmbed,
    Glm4vVisionRotaryEmbedding,
    Glm4vVisionPatchMerger,
    Glm4vVisionEmbeddings,
    Glm4vVisionAttention,
    Glm4vVisionBlock,
    Glm4vPreTrainedModel,
    Glm4vVisionModel,
)

# Processing Components
from .processing import (
    smart_resize,
    Glm4vImageProcessor,
    Glm4vVideoProcessor,
)

# Main Pipeline Interface
from .embedding_pipeline import GLM4VEmbeddingPipeline

__all__ = [
    # Configuration
    "Glm4vVisionConfig",
    
    # Vision Model Components
    "Glm4vRMSNorm",
    "Glm4VisionMlp",
    "Glm4vVisionPatchEmbed",
    "Glm4vVisionRotaryEmbedding",
    "Glm4vVisionPatchMerger",
    "Glm4vVisionEmbeddings",
    "Glm4vVisionAttention",
    "Glm4vVisionBlock",
    "Glm4vPreTrainedModel",
    "Glm4vVisionModel",
    
    # Processing Components
    "smart_resize",
    "Glm4vImageProcessor",
    "Glm4vVideoProcessor",
    
    # Main Pipeline Interface
    "GLM4VEmbeddingPipeline",
] 