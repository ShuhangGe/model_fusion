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
vLLM API client for GLM-4V multimodal inference.

Features:
- Auto-detects media type (video/image) based on file extension
- Supports both local files and web URLs
- Configurable API parameters via command line
- Single media file per request (video OR image, not both)
- Enhanced error handling and retry logic

Supported formats:
- Videos: .mp4, .avi, .mov, .mkv, .wmv, .flv, .webm, .mpeg, .m4v
- Images: .jpg, .png, .jpeg, .gif, .bmp, .tiff, .webp

Usage examples:
  python -m model_fusion.glm4v.inference.api.vllm_client --media-path "/path/video.mp4" --text "Analyze this video"
  python -m model_fusion.glm4v.inference.api.vllm_client --media-path "https://example.com/image.jpg" --text "Describe image"
  python -m model_fusion.glm4v.inference.api.vllm_client --media-path "/path/file.png" --text "What's this?" --temperature 0.5

Note: Only one media file (video OR image) can be processed per request.
"""

import argparse
import time
from typing import Dict, Any, Optional, List
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("OpenAI client is required. Install with: pip install openai")

from ..config import InferenceConfig, DEFAULT_API_CONFIG
from ..base import FileHandler


class GLM4VvLLMClient:
    """vLLM API client for GLM-4V multimodal inference."""
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        """Initialize the vLLM client."""
        self.config = config or DEFAULT_API_CONFIG
        self.client = None
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize the OpenAI client for vLLM."""
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.api_base_url
        )
        
    def create_content_item(self, media_path: str) -> Dict[str, Any]:
        """Create content item for media file."""
        # Determine media type
        file_type = FileHandler.get_file_type(media_path)
        
        if file_type not in ["image", "video"]:
            raise ValueError(f"Unsupported media type: {file_type}")
            
        # Handle URL vs local file
        if media_path.startswith(("http://", "https://")):
            url = media_path
        else:
            # Validate local file
            is_valid, error_msg = FileHandler.validate_file_path(media_path)
            if not is_valid:
                raise ValueError(f"Invalid media file: {error_msg}")
            url = "file://" + str(Path(media_path).absolute())
            
        # Create content item based on type
        if file_type == "video":
            return {"type": "video_url", "video_url": {"url": url}}
        else:  # image
            return {"type": "image_url", "image_url": {"url": url}}
            
    def send_request(self, media_path: str, text: str, 
                    temperature: Optional[float] = None,
                    max_tokens: Optional[int] = None,
                    top_p: Optional[float] = None,
                    repetition_penalty: Optional[float] = None,
                    **extra_kwargs) -> str:
        """
        Send a multimodal API request to vLLM.
        
        Args:
            media_path: Path to media file (image or video)
            text: Text prompt
            temperature: Generation temperature (overrides config)
            max_tokens: Maximum tokens to generate (overrides config)
            top_p: Top-p sampling parameter (overrides config)
            repetition_penalty: Repetition penalty (overrides config)
            **extra_kwargs: Additional parameters for the API call
            
        Returns:
            Generated response text
        """
        # Create content
        try:
            content_item = self.create_content_item(media_path)
        except ValueError as e:
            raise ValueError(f"Error processing media file: {e}")
            
        # Build messages
        messages = [
            {
                "role": "user",
                "content": [
                    content_item,
                    {"type": "text", "text": text}
                ]
            }
        ]
        
        # Prepare API parameters
        api_params = {
            "model": self.config.model_path,
            "messages": messages,
            "max_tokens": max_tokens or self.config.max_new_tokens,
            "temperature": temperature or self.config.temperature,
        }
        
        if top_p is not None or self.config.top_p is not None:
            api_params["top_p"] = top_p or self.config.top_p
            
        # Extra body parameters
        extra_body = {
            "skip_special_tokens": False,
            "repetition_penalty": repetition_penalty or self.config.repetition_penalty,
        }
        extra_body.update(extra_kwargs)
        api_params["extra_body"] = extra_body
        
        # Send request
        try:
            response = self.client.chat.completions.create(**api_params)
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"API request failed: {e}")
            
    def send_request_with_retry(self, media_path: str, text: str,
                              max_retries: int = 3,
                              retry_delay: float = 1.0,
                              **kwargs) -> str:
        """
        Send request with retry logic.
        
        Args:
            media_path: Path to media file
            text: Text prompt  
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries (seconds)
            **kwargs: Additional parameters for send_request
            
        Returns:
            Generated response text
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                return self.send_request(media_path, text, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    print(f"Request failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    break
                    
        raise RuntimeError(f"All retry attempts failed. Last error: {last_error}")
        
    def batch_request(self, requests: List[tuple[str, str]], **kwargs) -> List[Dict[str, Any]]:
        """
        Send multiple requests in batch.
        
        Args:
            requests: List of (media_path, text) tuples
            **kwargs: Common parameters for all requests
            
        Returns:
            List of results with metadata
        """
        results = []
        
        for i, (media_path, text) in enumerate(requests):
            print(f"Processing request {i+1}/{len(requests)}: {Path(media_path).name}")
            
            try:
                response = self.send_request_with_retry(media_path, text, **kwargs)
                results.append({
                    "media_path": media_path,
                    "text": text,
                    "response": response,
                    "success": True,
                    "error": None
                })
            except Exception as e:
                print(f"Failed to process {media_path}: {e}")
                results.append({
                    "media_path": media_path,
                    "text": text,
                    "response": None,
                    "success": False,
                    "error": str(e)
                })
                
        return results
        
    def print_request_info(self, media_path: str, text: str):
        """Print request information."""
        print("=" * 50)
        print("vLLM API Request")
        print("=" * 50)
        print(f"Model: {self.config.model_path}")
        print(f"Base URL: {self.config.api_base_url}")
        print(f"Media: {Path(media_path).name} ({FileHandler.get_file_type(media_path)})")
        print(f"Text: {text}")
        print(f"Max tokens: {self.config.max_new_tokens}")
        print(f"Temperature: {self.config.temperature}")
        print("=" * 50)


def main():
    """Main entry point for the vLLM client."""
    parser = argparse.ArgumentParser(description="GLM-4V vLLM API Client")
    
    # Required arguments
    parser.add_argument("--media-path", type=str, required=True,
                       help="Path to media file (image or video)")
    parser.add_argument("--text", type=str, required=True,
                       help="Text prompt for the model")
    
    # API configuration
    parser.add_argument("--api-key", type=str, default="EMPTY",
                       help="API key for authentication")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000/v1",
                       help="Base URL for the vLLM API")
    parser.add_argument("--model", type=str, default="THUDM/GLM-4.1V-9B-Thinking",
                       help="Model name to use")
    
    # Generation parameters
    parser.add_argument("--max-tokens", type=int, default=25000,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Generation temperature")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                       help="Repetition penalty")
    parser.add_argument("--top-p", type=float, default=None,
                       help="Top-p sampling parameter")
    
    # Retry configuration
    parser.add_argument("--max-retries", type=int, default=3,
                       help="Maximum number of retry attempts")
    parser.add_argument("--retry-delay", type=float, default=1.0,
                       help="Initial delay between retries (seconds)")
    
    # Display options
    parser.add_argument("--show-request", action="store_true",
                       help="Show request details before sending")
    
    args = parser.parse_args()
    
    # Create configuration
    config = InferenceConfig(
        model_path=args.model,
        api_base_url=args.base_url,
        api_key=args.api_key,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        top_p=args.top_p,
    )
    
    # Create client
    client = GLM4VvLLMClient(config)
    
    # Show request info if requested
    if args.show_request:
        client.print_request_info(args.media_path, args.text)
    
    # Send request
    try:
        print("Sending request to vLLM API...")
        response = client.send_request_with_retry(
            media_path=args.media_path,
            text=args.text,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay
        )
        
        print("\n" + "=" * 50)
        print("Response:")
        print("=" * 50)
        print(response)
        
    except Exception as e:
        print(f"\nError: {e}")
        exit(1)


if __name__ == "__main__":
    main() 