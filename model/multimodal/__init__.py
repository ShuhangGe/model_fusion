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
GLM4V Complete Preprocessing Pipeline

This module contains the complete GLM4V preprocessing pipeline that handles
everything before the TextModel input, including:

1. **Multimodal Input Processing**: Raw text + images/videos
2. **Token Replacement**: Critical <|image|> expansion logic  
3. **Vision Processing**: Complete GLM4V vision transformer
4. **Multimodal Fusion**: masked_scatter operation inserting vision embeddings
5. **3D RoPE System**: Spatial-temporal coordinates for vision, 1D for text
6. **Output**: Unified embeddings ready for any language model

## Architecture Overview

```
Raw Inputs (text + images/videos)
    ↓
Glm4vProcessor (token replacement, image/video processing)
    ↓  
Glm4vMultimodalFusion (vision encoding + embedding fusion)
    ↓
GLM4VPreprocessingPipeline (3D RoPE + final preparation)
    ↓
Fused embeddings ready for TextModel
```

## Key Components

- **Glm4vConfig**: Complete configuration with vision, text, and fusion settings
- **Glm4vVisionModel**: Full GLM4V vision transformer for image/video encoding
- **Glm4vMultimodalFusion**: Critical fusion model with masked_scatter and 3D RoPE
- **GLM4VPreprocessingPipeline**: Main pipeline orchestrating the complete flow

## Usage

```python
from model.multimodal import GLM4VPreprocessingPipeline, Glm4vConfig

# Initialize pipeline
config = Glm4vConfig()
pipeline = GLM4VPreprocessingPipeline(config=config, tokenizer=tokenizer)

# Process multimodal inputs
result = pipeline(
    text="What is in this image? <|image|>",
    images=[pil_image],
    return_dict=True
)

# result contains:
# - inputs_embeds: Fused embeddings ready for language model
# - position_ids: 3D coordinates (temporal, height, width)
# - attention_mask: Attention mask for the sequence
# - Additional metadata for language model compatibility
```

## Integration Notes

This pipeline produces outputs that are fully compatible with any transformer-based
language model. The fused embeddings contain both textual and visual information,
with proper position embeddings and attention masks.

The 3D RoPE system ensures that vision tokens receive spatial-temporal coordinates
while text tokens maintain sequential 1D coordinates, enabling proper attention
computation in the language model.
"""

from typing import TYPE_CHECKING

# Core imports that should always work
from .config import (
    Glm4vConfig,
    Glm4vVisionConfig, 
    Glm4vTextConfig,
)

from .vision_model import Glm4vVisionModel

from .fusion_model import Glm4vMultimodalFusion

from .preprocessing_pipeline import GLM4VPreprocessingPipeline

# Optional imports for full functionality
# These may fail due to dependency issues
_OPTIONAL_IMPORTS = {}

try:
    from .processor import Glm4vProcessor
    _OPTIONAL_IMPORTS["processor"] = Glm4vProcessor
except ImportError as e:
    _OPTIONAL_IMPORTS["processor"] = None
    import warnings
    warnings.warn(f"Could not import Glm4vProcessor due to dependency issues: {e}")

try:
    from .image_processing import Glm4vImageProcessor
    _OPTIONAL_IMPORTS["image_processor"] = Glm4vImageProcessor
except ImportError as e:
    _OPTIONAL_IMPORTS["image_processor"] = None
    import warnings
    warnings.warn(f"Could not import Glm4vImageProcessor due to dependency issues: {e}")

try:
    from .video_processing import Glm4vVideoProcessor
    _OPTIONAL_IMPORTS["video_processor"] = Glm4vVideoProcessor
except ImportError as e:
    _OPTIONAL_IMPORTS["video_processor"] = None
    import warnings
    warnings.warn(f"Could not import Glm4vVideoProcessor due to dependency issues: {e}")

# Expose optional imports if available
if _OPTIONAL_IMPORTS["processor"]:
    Glm4vProcessor = _OPTIONAL_IMPORTS["processor"]
    
if _OPTIONAL_IMPORTS["image_processor"]:
    Glm4vImageProcessor = _OPTIONAL_IMPORTS["image_processor"]
    
if _OPTIONAL_IMPORTS["video_processor"]:
    Glm4vVideoProcessor = _OPTIONAL_IMPORTS["video_processor"]

# Always available exports
__all__ = [
    "Glm4vConfig",
    "Glm4vVisionConfig", 
    "Glm4vTextConfig",
    "Glm4vVisionModel",
    "Glm4vMultimodalFusion", 
    "GLM4VPreprocessingPipeline",
]

# Add optional exports if available
if _OPTIONAL_IMPORTS["processor"]:
    __all__.append("Glm4vProcessor")
if _OPTIONAL_IMPORTS["image_processor"]:
    __all__.append("Glm4vImageProcessor")
if _OPTIONAL_IMPORTS["video_processor"]:
    __all__.append("Glm4vVideoProcessor") 