# GLM-4V Inference Tools

This module provides comprehensive inference tools for the GLM-4.1V-9B-Thinking model, supporting multiple interfaces and use cases.

## Overview

The inference tools are organized into several categories:

- **CLI**: Interactive command-line interface
- **Web**: Gradio-based web interface
- **API**: vLLM API client tools
- **Benchmark**: Academic benchmarking tools
- **Core**: Unified inference engine and configuration

## Quick Start

### Basic Usage

```python
from model_fusion.glm4v import GLM4VInferenceEngine, InferenceConfig

# Create configuration
config = InferenceConfig(
    model_path="THUDM/GLM-4.1V-9B-Thinking",
    temperature=0.8,
    max_new_tokens=4096
)

# Initialize engine
engine = GLM4VInferenceEngine(config)
engine.load_model()

# Generate response
messages = [
    {
        "role": "user", 
        "content": [
            {"type": "image", "url": "/path/to/image.jpg"},
            {"type": "text", "text": "Describe this image"}
        ]
    }
]

response = engine.generate(messages)
print(response)
```

## CLI Interface

Interactive command-line chat with multimodal support.

### Usage

```bash
# Text-only chat
python -m model_fusion.glm4v.inference.cli.interactive

# Chat with images
python -m model_fusion.glm4v.inference.cli.interactive \
    --image_paths /path/to/img1.jpg /path/to/img2.png

# Chat with video
python -m model_fusion.glm4v.inference.cli.interactive \
    --video_path /path/to/video.mp4

# Custom parameters
python -m model_fusion.glm4v.inference.cli.interactive \
    --temperature 0.8 --max_tokens 4096
```

### Programmatic Usage

```python
from model_fusion.glm4v import GLM4VCLIChat, InferenceConfig

config = InferenceConfig(temperature=0.8)
chat = GLM4VCLIChat(config)
chat.run_interactive_chat(image_paths=["/path/to/image.jpg"])
```

## Web Interface

Gradio-based web interface with file upload support.

### Launch Server

```bash
python -m model_fusion.glm4v.inference.web.gradio_app \
    --server_name 0.0.0.0 --server_port 7860 --share
```

### Programmatic Usage

```python
from model_fusion.glm4v import GLM4VGradioApp, InferenceConfig

config = InferenceConfig(
    server_name="0.0.0.0",
    server_port=7860,
    share=True
)

app = GLM4VGradioApp(config)
app.launch()
```

### Features

- Upload images, videos, PDFs, PowerPoint files
- Real-time streaming responses
- Thinking process visualization
- System prompt customization
- Conversation history management

## API Client

vLLM-compatible API client for remote inference.

### Usage

```bash
# Single request
python -m model_fusion.glm4v.inference.api.vllm_client \
    --media-path "/path/to/video.mp4" \
    --text "Analyze this video" \
    --base-url "http://localhost:8000/v1"

# With retry logic
python -m model_fusion.glm4v.inference.api.vllm_client \
    --media-path "/path/to/image.jpg" \
    --text "Describe this image" \
    --max-retries 5 --retry-delay 2.0
```

### Programmatic Usage

```python
from model_fusion.glm4v import GLM4VvLLMClient, InferenceConfig

config = InferenceConfig(
    api_base_url="http://localhost:8000/v1",
    api_key="your-api-key"
)

client = GLM4VvLLMClient(config)
response = client.send_request("/path/to/media.mp4", "Analyze this")
```

## Benchmark Tools

Robust inference for academic benchmarking with force completion.

### Usage

```bash
python -m model_fusion.glm4v.inference.benchmark.robust_inference \
    --video_path "/path/to/video.mp4" \
    --prompt "Analyze this video in detail" \
    --first_max_tokens 4096 \
    --force_max_tokens 4096 \
    --temperature 0.1
```

### Programmatic Usage

```python
from model_fusion.glm4v import GLM4VRobustInference, InferenceConfig

config = InferenceConfig(
    temperature=0.1,
    enable_force_completion=True
)

runner = GLM4VRobustInference(config)
result = runner.run_single_inference(
    video_path="/path/to/video.mp4",
    prompt="Analyze this video",
    first_max_tokens=4096,
    force_max_tokens=4096
)

print(f"Complete: {result['complete']}")
print(f"Output: {result['output_text']}")
```

### Features

- Force completion for incomplete outputs
- Automatic thinking/answer tag handling
- Detailed generation metadata
- Batch processing support

## Configuration

The `InferenceConfig` class provides centralized configuration:

```python
from model_fusion.glm4v import InferenceConfig

config = InferenceConfig(
    # Model settings
    model_path="THUDM/GLM-4.1V-9B-Thinking",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    
    # Generation settings
    max_new_tokens=8192,
    temperature=1.0,
    top_k=2,
    repetition_penalty=1.0,
    
    # Special tokens
    thinking_tag_start="<think>",
    thinking_tag_end="</think>",
    answer_tag_start="<answer>",
    answer_tag_end="</answer>",
    
    # Multimodal constraints
    max_images=10,
    max_videos=1,
    allow_mixed_media=False,
    
    # Server settings (for web interface)
    server_name="127.0.0.1",
    server_port=7860,
    share=False,
    
    # API settings
    api_base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
)
```

### Default Configurations

Pre-configured settings for different use cases:

```python
from model_fusion.glm4v.inference.config import (
    DEFAULT_CLI_CONFIG,
    DEFAULT_WEB_CONFIG, 
    DEFAULT_BENCHMARK_CONFIG,
    DEFAULT_API_CONFIG
)
```

## Advanced Features

### Streaming Generation

```python
engine = GLM4VInferenceEngine()
engine.load_model()

for token in engine.generate_stream(messages):
    print(token, end="", flush=True)
```

### Force Completion

```python
result = engine.generate_with_force_completion(
    messages=messages,
    first_max_tokens=4096,
    force_max_tokens=4096
)

print(f"Generation rounds: {result['generation_rounds']}")
print(f"Complete: {result['complete']}")
```

### Thinking Process Extraction

```python
raw_output = engine.generate(messages)
answer = engine.extract_answer_from_output(raw_output)
formatted = engine.format_thinking_output(raw_output)
```

## File Format Support

### Images
- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- BMP (.bmp)
- TIFF (.tiff)
- WebP (.webp)

### Videos
- MP4 (.mp4)
- AVI (.avi)
- MKV (.mkv)
- MOV (.mov)
- WMV (.wmv)
- FLV (.flv)
- WebM (.webm)
- MPEG (.mpeg)
- M4V (.m4v)

### Documents
- PDF (.pdf) - converted to images
- PowerPoint (.ppt, .pptx) - converted to images via LibreOffice

## Error Handling

All tools include comprehensive error handling:

```python
try:
    response = engine.generate(messages)
except Exception as e:
    print(f"Generation failed: {e}")
```

For API clients, retry logic is built-in:

```python
response = client.send_request_with_retry(
    media_path="/path/to/file",
    text="prompt",
    max_retries=3,
    retry_delay=1.0
)
```

## Dependencies

Core dependencies:
- torch
- transformers
- local GLM4V model components

Optional dependencies for specific features:
- gradio (web interface)
- PyMuPDF (PDF processing)
- openai (API client)
- LibreOffice (PowerPoint conversion)

## Examples

See the `examples/` directory for more detailed usage examples and tutorials.

## Troubleshooting

### Common Issues

1. **Model loading fails**: Check model path and ensure sufficient GPU memory
2. **File not found**: Verify file paths and permissions
3. **Generation timeout**: Increase max_tokens or check device resources
4. **API connection fails**: Verify server URL and network connectivity

### Performance Optimization

- Use `torch_dtype=torch.bfloat16` for faster inference
- Set `device_map="auto"` for optimal GPU utilization
- Enable `use_fast_processor=True` for faster tokenization
- Use streaming for long responses to improve user experience

## Contributing

When adding new inference tools:

1. Follow the existing package structure
2. Inherit from `BaseInference` for consistency
3. Use the unified `InferenceConfig` system
4. Include comprehensive error handling
5. Add documentation and examples
6. Update this README 