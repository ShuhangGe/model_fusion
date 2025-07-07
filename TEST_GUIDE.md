# GLM-4.1V-Thinking Model Test Guide

This guide helps you understand and test the GLM-4.1V-Thinking model using the provided test script with fake inputs.

## Quick Start

1. **Install dependencies:**
   ```bash
   cd GLM-4.1V-Thinking_model
   pip install -r requirements.txt
   ```

2. **Run the test:**
   ```bash
   python test_model_with_fake_inputs.py
   ```

## What the Test Script Does

### 1. Model Architecture Overview
The GLM-4.1V-Thinking model is a multimodal model with three main components:

- **Vision Model**: Processes images/videos into embeddings
- **Language Model**: Handles text understanding and generation
- **Integration Layer**: Combines vision and text representations

### 2. Key Components Tested

#### Configuration
- Creates a smaller test configuration for faster testing
- Shows how vision and text configs work together
- Demonstrates special token IDs for multimodal inputs

#### Text Processing
- Creates fake text inputs with proper input_ids and attention_mask
- Tests text-only forward pass through the model
- Shows how vocabulary size affects input generation

#### Image Processing
- Generates fake image data in the correct format
- Demonstrates patch-based image processing
- Shows how images are converted to embeddings

#### Multimodal Integration
- Combines text and image inputs
- Uses special image tokens to mark where images should be embedded
- Tests the model's ability to process mixed modalities

#### Generation Testing
- Tests the model's text generation capabilities
- Shows how to use the `generate()` method
- Demonstrates basic inference workflow

## Understanding the Output

### Model Structure
```
GLM-4V Model
├── Vision Model (Glm4vVisionModel)
│   ├── Patch Embedding
│   ├── Transformer Blocks (24 layers in full model)
│   └── Patch Merger
├── Language Model (Glm4vTextModel)
│   ├── Token Embedding
│   ├── Decoder Layers (40 layers in full model)
│   └── RMSNorm
└── LM Head (for token generation)
```

### Input Formats

#### Text Input
```python
{
    "input_ids": torch.tensor([[1, 234, 567, ...]]),      # Token IDs
    "attention_mask": torch.tensor([[1, 1, 1, ...]])       # Attention mask
}
```

#### Image Input
```python
{
    "pixel_values": torch.tensor(...),     # Shape: (total_patches, channels, patch_h, patch_w)
    "image_grid_thw": torch.tensor([...])  # Shape: (num_images, 3) - [temporal, height, width]
}
```

#### Multimodal Input
```python
{
    "input_ids": torch.tensor([[1, 234, 151343, 567, ...]]),  # 151343 = image token
    "attention_mask": torch.tensor([[1, 1, 1, 1, ...]]),
    "pixel_values": torch.tensor(...),
    "image_grid_thw": torch.tensor([...])
}
```

## Key Concepts

### 1. Special Tokens
- `image_token_id` (151343): Marks where image embeddings should be inserted
- `video_start_token_id` (151341): Marks start of video content
- `video_end_token_id` (151342): Marks end of video content

### 2. Image Processing Pipeline
```
Raw Image → Patches → Vision Transformer → Embeddings → Language Model
```

### 3. Position Encoding
The model uses 3D rotary position embeddings for:
- **Temporal dimension**: For video frames
- **Height dimension**: For spatial relationships
- **Width dimension**: For spatial relationships

### 4. Attention Mechanism
- Vision model uses bidirectional attention
- Language model uses causal (autoregressive) attention
- Multimodal attention allows cross-modal interactions

## Customizing the Test

### Modify Model Size
```python
config = Glm4vConfig(
    vision_config={
        "hidden_size": 512,     # Increase for more capacity
        "depth": 4,             # More transformer layers
        "num_heads": 8,         # More attention heads
    },
    text_config={
        "hidden_size": 1024,    # Larger text model
        "num_hidden_layers": 8, # More decoder layers
    }
)
```

### Test Different Inputs
```python
# Test with multiple images
multimodal_inputs = create_multimodal_inputs(config, num_images=3, text_length=20)

# Test longer sequences
text_inputs = create_fake_text_inputs(config, batch_size=2, seq_len=100)
```

## Real Usage Workflow

### 1. Load Pretrained Model
```python
from glm4v import Glm4vForConditionalGeneration
from transformers import AutoTokenizer

# Load pretrained model
model = Glm4vForConditionalGeneration.from_pretrained("THUDM/GLM-4.1V-9B-Thinking")
tokenizer = AutoTokenizer.from_pretrained("THUDM/GLM-4.1V-9B-Thinking")
```

### 2. Process Real Images
```python
from glm4v import Glm4vImageProcessor
from PIL import Image

processor = Glm4vImageProcessor.from_pretrained("THUDM/GLM-4.1V-9B-Thinking")
image = Image.open("your_image.jpg")
image_inputs = processor(images=image, return_tensors="pt")
```

### 3. Create Multimodal Prompt
```python
# Text with image placeholder
text = "What do you see in this image? <image>"
text_inputs = tokenizer(text, return_tensors="pt")

# Combine inputs
inputs = {
    **text_inputs,
    **image_inputs
}
```

### 4. Generate Response
```python
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're in the correct directory and dependencies are installed
2. **Memory issues**: Reduce model size in test config or use smaller batch sizes
3. **Shape mismatches**: Check that image_grid_thw matches pixel_values dimensions

### Performance Tips

1. **Use smaller models for testing**: The test script creates a mini version for speed
2. **Enable gradient checkpointing**: For memory efficiency during training
3. **Use mixed precision**: FP16 or BF16 for faster inference

## Next Steps

1. **Study the model architecture**: Look at `modeling_glm4v.py` for implementation details
2. **Try real images**: Use the image processor with actual images
3. **Implement fine-tuning**: Adapt the model for your specific tasks
4. **Explore generation strategies**: Different decoding methods and parameters

This test script gives you a solid foundation for understanding how the GLM-4.1V-Thinking model works with both text and visual inputs! 