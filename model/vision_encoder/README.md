# GLM4V Embedding Pipeline

This directory contains the complete GLM4V embedding pipeline extracted from the GLM4V multimodal model. The pipeline provides a unified interface for extracting visual embeddings from images and videos that can be used as input to language models.

## Overview

The GLM4V embedding pipeline consists of several key components:

- **Vision Model**: Transformer-based vision encoder that processes visual patches
- **Image Processor**: Handles image preprocessing and patch extraction
- **Video Processor**: Handles video preprocessing and temporal patch extraction
- **Embedding Pipeline**: Unified interface that combines all components

## Components

### Core Files

- `config.py` - Configuration classes for the vision model
- `vision_model.py` - Complete vision transformer implementation
- `processing.py` - Image and video processing pipelines
- `embedding_pipeline.py` - Main interface for embedding extraction
- `utils.py` - Utility functions for common operations
- `__init__.py` - Module exports and public API

### Key Classes

#### `GLM4VEmbeddingPipeline`
Main interface for extracting embeddings from images and videos.

```python
from model.encoder import GLM4VEmbeddingPipeline

# Initialize pipeline
pipeline = GLM4VEmbeddingPipeline()

# Extract image embeddings
image_embeddings = pipeline.extract_image_embeddings(images)

# Extract video embeddings 
video_embeddings = pipeline.extract_video_embeddings(videos, video_metadata)
```

#### `Glm4vVisionModel`
Core vision transformer that processes visual patches into embeddings.

#### `Glm4vImageProcessor` / `Glm4vVideoProcessor`
Handle preprocessing of images and videos respectively, including:
- Dynamic resizing based on content
- Patch extraction and tokenization
- Normalization and data format conversion

## Usage Examples

### Basic Image Processing

```python
from model.encoder import GLM4VEmbeddingPipeline
from PIL import Image

# Load pipeline
pipeline = GLM4VEmbeddingPipeline()

# Load and process image
image = Image.open("example.jpg")
result = pipeline.extract_image_embeddings([image], return_dict=True)

print(f"Embeddings shape: {result['embeddings'].shape}")
print(f"Number of patches: {result['num_patches']}")
print(f"Embedding dimension: {result['embedding_dim']}")
```

### Basic Video Processing

```python
import torch
from model.encoder import GLM4VEmbeddingPipeline

# Load pipeline
pipeline = GLM4VEmbeddingPipeline()

# Process video (assuming video is already loaded as tensor)
video_tensor = torch.randn(16, 3, 224, 224)  # 16 frames, 3 channels, 224x224
video_metadata = {"fps": 30.0, "duration": 0.533, "total_num_frames": 16}

result = pipeline.extract_video_embeddings(
    video_tensor, 
    video_metadata=[video_metadata],
    return_dict=True
)

print(f"Video embeddings shape: {result['embeddings'].shape}")
print(f"Timestamps: {result['timestamps']}")
```

### Advanced Usage with Custom Configuration

```python
from model.encoder import GLM4VEmbeddingPipeline, Glm4vVisionConfig

# Custom configuration
config = Glm4vVisionConfig(
    hidden_size=1536,
    depth=24,
    num_heads=12,
    patch_size=14,
    spatial_merge_size=2,
)

# Initialize with custom config
pipeline = GLM4VEmbeddingPipeline(config=config)

# Use pipeline...
```

## Output Format

### Image Embeddings
The pipeline produces embeddings with the following characteristics:

- **Shape**: `(num_patches, hidden_size)` where `hidden_size=4096` by default
- **Content**: Dense visual representations suitable for language model input
- **Grid Information**: Spatial layout information (`image_grid_thw`)

### Video Embeddings
Video embeddings include additional temporal information:

- **Shape**: `(num_patches, hidden_size)` 
- **Temporal Info**: Frame sampling timestamps
- **Grid Information**: Temporal, height, width layout (`video_grid_thw`)

## Configuration Options

The `Glm4vVisionConfig` class provides these key parameters:

- `hidden_size`: Dimension of hidden representations (default: 1536)
- `depth`: Number of transformer layers (default: 24)
- `num_heads`: Number of attention heads (default: 12)
- `patch_size`: Spatial patch size (default: 14)
- `spatial_merge_size`: Spatial merging factor (default: 2)
- `out_hidden_size`: Output embedding dimension (default: 4096)

## Performance Considerations

- The pipeline supports GPU acceleration when available
- Video processing includes frame sampling to manage computational load
- Images are dynamically resized to optimize patch count vs. quality tradeoff

## Integration Notes

This pipeline is designed to be compatible with:
- Language models expecting vision embeddings
- Multimodal architectures that combine vision and text
- Custom downstream tasks requiring visual representations

The embeddings produced are in the same format as the original GLM4V model, ensuring compatibility with existing workflows.

## Dependencies

- PyTorch
- Transformers library
- PIL (Python Imaging Library)
- NumPy
- Optional: torchvision (for video processing)

## Architecture Details

The vision model uses a standard Vision Transformer architecture with GLM4V-specific modifications:

1. **Patch Embedding**: 3D convolution for temporal-spatial patch extraction
2. **Position Embedding**: 2D interpolated position embeddings with rotary embeddings
3. **Transformer Blocks**: Standard multi-head attention with RMS normalization
4. **Spatial Merging**: Downsampling via 2D convolution before final projection
5. **Output Projection**: MLP projection to target embedding dimension

This extracted pipeline maintains full compatibility with the original GLM4V vision processing while providing a clean, standalone interface for embedding extraction. 