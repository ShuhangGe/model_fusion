# GLM4V Multimodal Preprocessing Pipeline

This directory contains the **complete GLM4V preprocessing pipeline** that handles all steps before the GLM4V TextModel input. This is everything shown in the black and white diagram - from raw multimodal inputs to unified embeddings ready for language model processing.

## 🎯 **What This Pipeline Does**

This pipeline extracts and implements **everything that happens before the GLM4V TextModel**, including:

1. **Input Processing**: Raw text + images/videos 
2. **Token Replacement**: Critical `<|image|>` token expansion
3. **Vision Processing**: Complete GLM4V vision transformer
4. **Multimodal Fusion**: `masked_scatter` operation that inserts vision embeddings
5. **3D RoPE**: Spatial-temporal position embedding calculation
6. **Output**: Unified multimodal embeddings ready for any language model

## 📋 **Pipeline Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT STAGE                             │
├─────────────────────────────────────────────────────────────────┤
│  Text: "What is in this image? <|image|>"                      │
│  Image: PIL.Image or Tensor                                     │
│  Video: Tensor (optional)                                       │
└─────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────┐
│                    PROCESSING STAGE                             │
├─────────────────────────────────────────────────────────────────┤
│  Glm4vProcessor:                                                │
│  • Text Processing: Tokenization                               │
│  • Vision Processing: Image/Video → patches                    │
│  • Token Replacement: <|image|> → N × <|image|> tokens         │
└─────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────┐
│                    EMBEDDING STAGE                              │
├─────────────────────────────────────────────────────────────────┤
│  Text Branch: input_ids → text_embeddings                      │
│  Vision Branch: pixel_values → vision_embeddings               │
└─────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────┐
│                      FUSION STAGE                               │
├─────────────────────────────────────────────────────────────────┤
│  masked_scatter Operation:                                      │
│  Insert vision_embeddings where input_ids == image_token_id    │
│  Result: [text_emb, text_emb, vision_emb, vision_emb, ...]     │
└─────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────┐
│                   3D RoPE STAGE                                 │
├─────────────────────────────────────────────────────────────────┤
│  Calculate 3D coordinates (temporal, height, width)            │
│  for vision tokens + 1D coordinates for text tokens           │
└─────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT                                     │
├─────────────────────────────────────────────────────────────────┤
│  • inputs_embeds: Fused multimodal embeddings                  │
│  • position_ids: 3D spatial-temporal coordinates               │
│  • attention_mask: Attention mask                              │
│  → READY FOR LANGUAGE MODEL INPUT                              │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 **Key Components**

### **1. Main Pipeline Interface**
- **`GLM4VPreprocessingPipeline`**: Complete end-to-end preprocessing

### **2. Processing Components**
- **`Glm4vProcessor`**: Unified text+vision processor (copied from GLM4V)
- **`Glm4vImageProcessor`**: Image preprocessing (copied from GLM4V)
- **`Glm4vVideoProcessor`**: Video preprocessing (copied from GLM4V)

### **3. Vision Model**
- **`Glm4vVisionModel`**: Complete vision transformer (copied from GLM4V)
- All supporting vision components (attention, embeddings, etc.)

### **4. Fusion System**
- **`Glm4vMultimodalFusion`**: Handles embedding fusion and 3D RoPE calculation
- Critical `masked_scatter` operation
- 3D spatial-temporal position embedding system

### **5. Configuration**
- **`Glm4vConfig`**: Complete GLM4V configuration
- **`Glm4vVisionConfig`**: Vision model configuration
- **`Glm4vTextConfig`**: Text model configuration

## 💻 **Usage Examples**

### **Basic Usage**
```python
from model.multimodal import GLM4VPreprocessingPipeline
from PIL import Image

# Initialize pipeline (requires tokenizer)
pipeline = GLM4VPreprocessingPipeline(tokenizer=tokenizer)

# Process multimodal input
image = Image.open("example.jpg")
result = pipeline(
    text="What is in this image? <|image|>",
    images=[image],
    return_dict=True
)

# Access the outputs ready for language model
embeddings = result['inputs_embeds']      # Shape: (batch_size, seq_len, hidden_size)
positions = result['position_ids']        # Shape: (3, batch_size, seq_len) - 3D coordinates
attention = result['attention_mask']      # Shape: (batch_size, seq_len)
```

### **Step-by-Step Processing**
```python
# Step 1: Process raw inputs
processed = pipeline.process_inputs(
    text="Describe this image: <|image|>",
    images=[image]
)

# Step 2: Extract and fuse embeddings
fused = pipeline.extract_and_fuse_embeddings(
    input_ids=processed['input_ids'],
    pixel_values=processed['pixel_values'],
    image_grid_thw=processed['image_grid_thw']
)

# Result: Ready for language model
embeddings = fused['inputs_embeds']
```

### **Vision-Only Processing**
```python
# Extract only vision embeddings (no text fusion)
vision_only = pipeline.get_vision_embeddings_only(
    pixel_values=processed['pixel_values'],
    image_grid_thw=processed['image_grid_thw']
)

image_embeddings = vision_only['image_embeddings']
```

## 📊 **Output Format**

The pipeline produces a dictionary with:

```python
{
    'inputs_embeds': torch.Tensor,      # Fused multimodal embeddings
    'position_ids': torch.Tensor,       # 3D position coordinates  
    'attention_mask': torch.Tensor,     # Attention mask
    'rope_deltas': torch.Tensor,        # RoPE delta values
    'has_images': bool,                 # Whether images were processed
    'has_videos': bool,                 # Whether videos were processed
    'image_grid_thw': torch.Tensor,     # Image spatial layout
    'video_grid_thw': torch.Tensor,     # Video spatial layout
    'input_ids': torch.Tensor,          # Original token IDs
    'num_image_tokens': int,            # Number of image tokens
    'sequence_length': int,             # Total sequence length
    'hidden_size': int,                 # Embedding dimension
}
```

## 🔍 **Key Innovations**

### **1. Token-Level Fusion**
The critical `masked_scatter` operation directly replaces image tokens with vision embeddings:
```python
image_mask = input_ids == image_token_id
inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
```

### **2. 3D Spatial-Temporal RoPE**
Vision tokens get 3D coordinates (time, height, width) while text tokens get 1D coordinates:
```python
# Vision tokens: [temporal_pos, height_pos, width_pos]
# Text tokens: [sequential_pos, sequential_pos, sequential_pos]
```

### **3. Dynamic Token Count**
The number of vision tokens depends on image resolution and is calculated automatically:
```python
num_image_tokens = (image_height * image_width) // (merge_size ** 2)
```

## 🎯 **Why This Matters**

This pipeline enables:
- **True Multimodal Understanding**: Text and vision inform each other
- **Spatial Awareness**: Vision tokens retain spatial relationships
- **Flexible Input**: Handle variable image sizes and video lengths
- **Language Model Ready**: Output can be fed to any transformer language model

## 🔗 **Integration with Language Models**

The output embeddings are designed to be compatible with any transformer-based language model:

```python
# After preprocessing
result = pipeline(text="...", images=[...])

# Feed to language model
language_model_output = your_language_model(
    inputs_embeds=result['inputs_embeds'],
    position_ids=result['position_ids'],
    attention_mask=result['attention_mask']
)
```

## 📁 **File Structure**

```
model/multimodal/
├── __init__.py                 # Module exports
├── config.py                  # All GLM4V configurations
├── processor.py               # Unified multimodal processor
├── image_processing.py        # Image preprocessing
├── video_processing.py        # Video preprocessing  
├── vision_model.py            # Complete vision transformer
├── fusion_model.py            # Multimodal fusion + 3D RoPE
├── preprocessing_pipeline.py  # Main pipeline interface
└── README.md                  # This file
```

This is the complete "everything before TextModel" pipeline as requested, implementing all the components shown in the diagram from raw multimodal inputs to language model ready embeddings. 