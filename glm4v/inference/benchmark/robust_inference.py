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
Robust inference benchmark tool for GLM-4V-Thinking model.

This script is designed for academic inference benchmarking with the GLM-4.1V-9B-Thinking model,
particularly in multi-modal settings involving video inputs. It ensures robust handling of structured
reasoning outputs such as `<think>...</think><answer>...</answer>`.

Core features:
- Automatically detects whether the model output includes a complete reasoning block.
- If the model reaches the `first_max_tokens` limit without emitting `</think>` or `<answer>`,
  it forcefully appends `</think><answer>` and re-generates to ensure complete output.
- Accepts video input via `video_url`, pointing to a local video file to simulate real multi-modal inputs.
- Designed for the local GLM4V model components with force completion capability.

Arguments:
- `--model_path`: Path to the model. Defaults to `THUDM/GLM-4.1V-9B-Thinking`.
- `--video_path`: Path to the input video file (required).
- `--prompt`: Text prompt for the model (required).
- `--first_max_tokens` / `--force_max_tokens`: Maximum tokens for initial generation and forced continuation.
- `--temperature`: Generation temperature (default: 0.1 for stability).
"""

import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..engine import GLM4VInferenceEngine
from ..config import InferenceConfig, DEFAULT_BENCHMARK_CONFIG
from ..base import FileHandler, MessageBuilder


class GLM4VRobustInference:
    """Robust inference engine for academic benchmarking with GLM-4V."""
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        """Initialize the robust inference engine."""
        self.config = config or DEFAULT_BENCHMARK_CONFIG
        self.engine = GLM4VInferenceEngine(self.config)
        
    def load_model(self):
        """Load the GLM-4V model and processor."""
        self.engine.load_model()
        
    def generate_with_force_completion(self, messages: List[Dict[str, Any]], 
                                     first_max_tokens: Optional[int] = None,
                                     force_max_tokens: Optional[int] = None,
                                     temperature: Optional[float] = None,
                                     do_sample: Optional[bool] = None) -> Dict[str, Any]:
        """
        Generate response with force completion to ensure complete output.
        
        Args:
            messages: List of conversation messages
            first_max_tokens: Maximum tokens for first generation round
            force_max_tokens: Maximum tokens for force completion round
            temperature: Generation temperature
            do_sample: Whether to use sampling
            
        Returns:
            Dictionary with generation results and metadata
        """
        if not self.engine._is_loaded:
            self.load_model()
            
        # Use provided parameters or fall back to config
        kwargs = {}
        if first_max_tokens is not None:
            # Temporarily update config for this generation
            original_first_max = self.config.first_max_tokens
            self.config.first_max_tokens = first_max_tokens
            
        if force_max_tokens is not None:
            original_force_max = self.config.force_max_tokens
            self.config.force_max_tokens = force_max_tokens
            
        if temperature is not None:
            kwargs["temperature"] = temperature
        if do_sample is not None:
            kwargs["do_sample"] = do_sample
            
        try:
            result = self.engine.generate_with_force_completion(messages, **kwargs)
            return result
        finally:
            # Restore original config values
            if first_max_tokens is not None:
                self.config.first_max_tokens = original_first_max
            if force_max_tokens is not None:
                self.config.force_max_tokens = original_force_max
                
    def run_single_inference(self, video_path: str, prompt: str, 
                           first_max_tokens: int = 8192,
                           force_max_tokens: int = 8192,
                           temperature: float = 0.1) -> Dict[str, Any]:
        """
        Run a single inference with video and prompt.
        
        Args:
            video_path: Path to the video file
            prompt: Text prompt for the model
            first_max_tokens: Maximum tokens for initial generation
            force_max_tokens: Maximum tokens for force completion
            temperature: Generation temperature
            
        Returns:
            Generation results with metadata
        """
        # Validate video file
        is_valid, error_msg = FileHandler.validate_file_path(video_path)
        if not is_valid:
            raise ValueError(f"Invalid video file: {error_msg}")
            
        if FileHandler.get_file_type(video_path) != "video":
            raise ValueError(f"File is not a video: {video_path}")
            
        # Build messages
        content = [
            {"type": "video", "url": video_path},
            {"type": "text", "text": prompt}
        ]
        messages = [MessageBuilder.create_user_message(content)]
        
        # Generate with force completion
        result = self.generate_with_force_completion(
            messages=messages,
            first_max_tokens=first_max_tokens,
            force_max_tokens=force_max_tokens,
            temperature=temperature,
            do_sample=True
        )
        
        return result
        
    def run_batch_inference(self, video_prompts: List[tuple[str, str]], 
                          **generation_kwargs) -> List[Dict[str, Any]]:
        """
        Run batch inference on multiple video-prompt pairs.
        
        Args:
            video_prompts: List of (video_path, prompt) tuples
            **generation_kwargs: Generation parameters
            
        Returns:
            List of generation results
        """
        results = []
        
        for i, (video_path, prompt) in enumerate(video_prompts):
            print(f"Processing {i+1}/{len(video_prompts)}: {Path(video_path).name}")
            
            try:
                result = self.run_single_inference(
                    video_path=video_path,
                    prompt=prompt,
                    **generation_kwargs
                )
                result["video_path"] = video_path
                result["prompt"] = prompt
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                results.append({
                    "video_path": video_path,
                    "prompt": prompt,
                    "error": str(e),
                    "complete": False
                })
                
        return results
        
    def print_results(self, result: Dict[str, Any]):
        """Print formatted results from generation."""
        print("=" * 50)
        print("Inference Results:")
        print(f"Complete: {result['complete']}")
        print(f"Reason: {result['reason']}")
        print(f"Generation rounds: {result['generation_rounds']}")
        print(f"Input length: {result['input_length']}")
        
        if "first_generation_length" in result:
            print(f"First generation length: {result['first_generation_length']}")
        if "second_generation_length" in result:
            print(f"Second generation length: {result['second_generation_length']}")
        if "total_generation_length" in result:
            print(f"Total generation length: {result['total_generation_length']}")
            
        print("=" * 50)
        print("Generated Output:")
        print(result["output_text"])
        print("=" * 50)


def main():
    """Main entry point for the robust inference benchmark."""
    parser = argparse.ArgumentParser(description="GLM-4V Robust Inference Benchmark")
    
    # Required arguments
    parser.add_argument("--video_path", type=str, required=True, 
                       help="Path to the input video file")
    parser.add_argument("--prompt", type=str, required=True,
                       help="Text prompt for the model")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, default="THUDM/GLM-4.1V-9B-Thinking",
                       help="Path to the model")
    
    # Generation arguments
    parser.add_argument("--first_max_tokens", type=int, default=8192,
                       help="Maximum tokens for first generation round")
    parser.add_argument("--force_max_tokens", type=int, default=8192,
                       help="Maximum tokens for force completion round")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Generation temperature")
    
    # Device arguments
    parser.add_argument("--device_map", type=str, default="auto",
                       help="Device mapping strategy")
    
    args = parser.parse_args()
    
    # Create configuration
    config = InferenceConfig(
        model_path=args.model_path,
        first_max_tokens=args.first_max_tokens,
        force_max_tokens=args.force_max_tokens,
        temperature=args.temperature,
        device_map=args.device_map,
        enable_force_completion=True,
    )
    
    # Create robust inference engine
    runner = GLM4VRobustInference(config)
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    runner.load_model()
    
    # Run inference
    print(f"Running inference on video: {Path(args.video_path).name}")
    print(f"Prompt: {args.prompt}")
    
    result = runner.run_single_inference(
        video_path=args.video_path,
        prompt=args.prompt,
        first_max_tokens=args.first_max_tokens,
        force_max_tokens=args.force_max_tokens,
        temperature=args.temperature
    )
    
    # Print results
    runner.print_results(result)


if __name__ == "__main__":
    main() 