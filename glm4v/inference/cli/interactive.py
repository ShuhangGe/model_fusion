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
Interactive command-line interface for GLM-4V-Thinking model supporting images and videos.

Examples:
    # Text-only chat
    python -m model_fusion.glm4v.inference.cli.interactive
    
    # Chat with single image
    python -m model_fusion.glm4v.inference.cli.interactive --image_paths /path/to/image.jpg
    
    # Chat with multiple images
    python -m model_fusion.glm4v.inference.cli.interactive --image_paths /path/to/img1.jpg /path/to/img2.png
    
    # Chat with single video
    python -m model_fusion.glm4v.inference.cli.interactive --video_path /path/to/video.mp4
    
    # Custom generation parameters
    python -m model_fusion.glm4v.inference.cli.interactive --temperature 0.8 --top_k 5 --max_tokens 4096

Notes:
    - Media files are loaded once at startup and persist throughout the conversation
    - Type 'exit' to quit the chat
    - Chat with images and video is NOT allowed
    - The model will remember the conversation history and can reference uploaded media in subsequent turns
"""

import argparse
import sys
from typing import List, Optional
from pathlib import Path

from ..engine import GLM4VInferenceEngine
from ..config import InferenceConfig, DEFAULT_CLI_CONFIG
from ..base import FileHandler, MessageBuilder


class GLM4VCLIChat:
    """Interactive CLI chat interface for GLM-4V model."""
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        """Initialize the CLI chat interface."""
        self.config = config or DEFAULT_CLI_CONFIG
        self.engine = GLM4VInferenceEngine(self.config)
        self.messages = []
        self.first_turn = True
        
    def setup_media(self, image_paths: Optional[List[str]] = None, 
                   video_path: Optional[str] = None) -> tuple[bool, str]:
        """Setup and validate media inputs."""
        # Validate media constraints
        if image_paths and video_path:
            return False, "Chat with images and video is NOT allowed. Please use either --image_paths or --video_path, not both."
        
        # Validate file paths
        all_files = []
        if image_paths:
            all_files.extend(image_paths)
        if video_path:
            all_files.append(video_path)
            
        for file_path in all_files:
            is_valid, error_msg = FileHandler.validate_file_path(file_path)
            if not is_valid:
                return False, error_msg
                
        return True, ""
    
    def run_interactive_chat(self, image_paths: Optional[List[str]] = None,
                           video_path: Optional[str] = None):
        """Run the interactive chat loop."""
        # Setup media
        is_valid, error_msg = self.setup_media(image_paths, video_path)
        if not is_valid:
            print(f"Error: {error_msg}")
            return
            
        # Load model
        print("Loading GLM-4V model...")
        self.engine.load_model()
        
        # Show startup info
        self._show_startup_info(image_paths, video_path)
        
        # Chat loop
        while True:
            try:
                question = input("\nUser: ").strip()
                if question.lower() == "exit":
                    print("Goodbye!")
                    break
                    
                if not question:
                    continue
                    
                # Build message content
                if self.first_turn:
                    content = self._build_first_turn_content(image_paths, video_path, question)
                    self.first_turn = False
                else:
                    content = [{"type": "text", "text": question}]
                
                # Add user message
                user_message = MessageBuilder.create_user_message(content)
                self.messages.append(user_message)
                
                # Generate response
                print("Assistant: ", end="", flush=True)
                try:
                    raw_output = self.engine.generate(self.messages)
                    
                    # Extract answer if configured
                    if self.config.extract_answer_tags:
                        answer = self.engine.extract_answer_from_output(raw_output)
                        display_output = answer if answer else raw_output
                    else:
                        display_output = raw_output
                    
                    print(display_output)
                    
                    # Add assistant message to history
                    assistant_message = MessageBuilder.create_assistant_message(display_output)
                    self.messages.append(assistant_message)
                    
                except Exception as e:
                    print(f"\nError during generation: {e}")
                    # Remove the user message if generation failed
                    self.messages.pop()
                    
            except KeyboardInterrupt:
                print("\n\nChat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nUnexpected error: {e}")
                break
    
    def _build_first_turn_content(self, image_paths: Optional[List[str]], 
                                 video_path: Optional[str], 
                                 text: str) -> List[dict]:
        """Build content for the first turn with media."""
        content = []
        
        if image_paths:
            for img_path in image_paths:
                content.append({"type": "image", "url": img_path})
                
        if video_path:
            content.append({"type": "video", "url": video_path})
            
        content.append({"type": "text", "text": text})
        return content
    
    def _show_startup_info(self, image_paths: Optional[List[str]], 
                          video_path: Optional[str]):
        """Show startup information."""
        print("\n" + "="*50)
        print("GLM-4V Interactive Chat")
        print("="*50)
        
        if image_paths:
            print(f"Images loaded: {len(image_paths)}")
            for i, path in enumerate(image_paths, 1):
                print(f"  {i}. {Path(path).name}")
                
        if video_path:
            print(f"Video loaded: {Path(video_path).name}")
            
        if not image_paths and not video_path:
            print("Text-only mode")
            
        print(f"Model: {self.config.model_path}")
        print(f"Max tokens: {self.config.max_new_tokens}")
        print(f"Temperature: {self.config.temperature}")
        print("\nType 'exit' to quit the chat")
        print("="*50)


def main():
    """Main entry point for the CLI chat."""
    parser = argparse.ArgumentParser(description="GLM-4V Interactive Chat CLI")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, default="THUDM/GLM-4.1V-9B-Thinking",
                       help="Path to the model")
    
    # Media arguments
    parser.add_argument("--image_paths", type=str, nargs="*", default=None,
                       help="Paths to image files")
    parser.add_argument("--video_path", type=str, default=None,
                       help="Path to video file")
    
    # Generation arguments
    parser.add_argument("--max_tokens", type=int, default=8192,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Generation temperature")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                       help="Repetition penalty")
    parser.add_argument("--top_k", type=int, default=2,
                       help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=None,
                       help="Top-p sampling parameter")
    
    # Device arguments
    parser.add_argument("--device_map", type=str, default="auto",
                       help="Device mapping strategy")
    
    args = parser.parse_args()
    
    # Create configuration
    config = InferenceConfig(
        model_path=args.model_path,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        top_k=args.top_k,
        top_p=args.top_p,
        device_map=args.device_map,
    )
    
    # Create and run chat
    chat = GLM4VCLIChat(config)
    chat.run_interactive_chat(args.image_paths, args.video_path)


if __name__ == "__main__":
    main() 