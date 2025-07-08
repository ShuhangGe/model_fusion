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
Base classes for GLM-4V inference implementations.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from .config import InferenceConfig


class BaseInference(ABC):
    """Base class for all GLM-4V inference implementations."""
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        """Initialize the inference instance with configuration."""
        self.config = config or InferenceConfig()
        
    @abstractmethod
    def load_model(self):
        """Load the model and processor. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """Generate response from messages. Must be implemented by subclasses."""
        pass
    
    def build_content(self, image_paths: Optional[List[str]] = None, 
                     video_path: Optional[str] = None, 
                     text: str = "") -> List[Dict[str, Any]]:
        """Build content list from media paths and text."""
        content = []
        
        if image_paths:
            for img_path in image_paths:
                content.append({"type": "image", "url": img_path})
                
        if video_path:
            content.append({"type": "video", "url": video_path})
            
        if text:
            content.append({"type": "text", "text": text})
            
        return content
    
    def validate_media_inputs(self, image_paths: Optional[List[str]] = None,
                            video_path: Optional[str] = None) -> tuple[bool, str]:
        """Validate media input constraints."""
        if image_paths and video_path and not self.config.allow_mixed_media:
            return False, "Chat with images and video is NOT allowed. Please use either images or video, not both."
        
        if image_paths and len(image_paths) > self.config.max_images:
            return False, f"Maximum {self.config.max_images} images allowed"
            
        return True, ""
    
    def extract_answer_from_output(self, raw_output: str) -> str:
        """Extract answer from model output with thinking tags."""
        if not self.config.extract_answer_tags:
            return raw_output
            
        # Try to extract content between <answer> tags
        answer_pattern = rf"{re.escape(self.config.answer_tag_start)}(.*?){re.escape(self.config.answer_tag_end)}"
        match = re.search(answer_pattern, raw_output, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        # If no answer tags found, return the original output
        return raw_output
    
    def strip_html_tags(self, text: str) -> str:
        """Remove HTML tags from text."""
        return re.sub(r"<[^>]+>", "", text).strip()
    
    def format_thinking_output(self, raw_output: str) -> str:
        """Format thinking output for display with HTML."""
        thinking_html = ""
        answer_html = ""
        
        # Extract thinking content
        think_pattern = rf"{re.escape(self.config.thinking_tag_start)}(.*?){re.escape(self.config.thinking_tag_end)}"
        think_match = re.search(think_pattern, raw_output, re.DOTALL)
        
        if think_match:
            thinking_content = think_match.group(1).strip().replace("\n", "<br>")
            thinking_html = (
                "<details open><summary style='cursor:pointer;font-weight:bold;color:#bbbbbb;'>💭 Thinking</summary>"
                "<div style='color:#cccccc;line-height:1.4;padding:10px;border-left:3px solid #666;margin:5px 0;background-color:rgba(128,128,128,0.1);'>"
                + thinking_content +
                "</div></details>"
            )
        
        # Extract answer content
        answer_pattern = rf"{re.escape(self.config.answer_tag_start)}(.*?){re.escape(self.config.answer_tag_end)}"
        answer_match = re.search(answer_pattern, raw_output, re.DOTALL)
        
        if answer_match:
            answer_html = answer_match.group(1).strip()
        
        # If no special tags found, return stripped HTML
        if not thinking_html and not answer_html:
            return self.strip_html_tags(raw_output)
            
        return thinking_html + answer_html


class FileHandler:
    """Utility class for handling file operations."""
    
    SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
    SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".mpeg", ".m4v"}
    SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".ppt", ".pptx"}
    
    @classmethod
    def get_file_type(cls, file_path: str) -> str:
        """Determine the type of file based on extension."""
        ext = Path(file_path).suffix.lower()
        
        if ext in cls.SUPPORTED_VIDEO_EXTENSIONS:
            return "video"
        elif ext in cls.SUPPORTED_IMAGE_EXTENSIONS:
            return "image"
        elif ext in cls.SUPPORTED_DOCUMENT_EXTENSIONS:
            return "document"
        else:
            return "unknown"
    
    @classmethod
    def validate_file_path(cls, file_path: str) -> tuple[bool, str]:
        """Validate if file path exists and is supported."""
        if not file_path:
            return False, "File path is empty"
            
        path = Path(file_path)
        if not path.exists():
            return False, f"File does not exist: {file_path}"
            
        file_type = cls.get_file_type(file_path)
        if file_type == "unknown":
            return False, f"Unsupported file type: {path.suffix}"
            
        return True, ""
    
    @classmethod
    def group_files_by_type(cls, file_paths: List[str]) -> Dict[str, List[str]]:
        """Group files by their type."""
        groups = {"image": [], "video": [], "document": []}
        
        for file_path in file_paths:
            file_type = cls.get_file_type(file_path)
            if file_type in groups:
                groups[file_type].append(file_path)
                
        return groups


class MessageBuilder:
    """Utility class for building conversation messages."""
    
    @staticmethod
    def create_user_message(content: Union[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Create a user message with content."""
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        return {"role": "user", "content": content}
    
    @staticmethod 
    def create_assistant_message(content: str) -> Dict[str, Any]:
        """Create an assistant message with text content."""
        return {"role": "assistant", "content": [{"type": "text", "text": content}]}
    
    @staticmethod
    def create_system_message(content: str) -> Dict[str, Any]:
        """Create a system message with text content."""
        return {"role": "system", "content": [{"type": "text", "text": content}]}
    
    @staticmethod
    def add_media_to_content(content: List[Dict[str, Any]], 
                           image_paths: Optional[List[str]] = None,
                           video_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Add media files to existing content."""
        if image_paths:
            for img_path in image_paths:
                content.insert(-1, {"type": "image", "url": img_path})
                
        if video_path:
            content.insert(-1, {"type": "video", "url": video_path})
            
        return content 