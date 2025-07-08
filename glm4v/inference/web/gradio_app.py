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
Gradio web interface for GLM-4V-Thinking model.

This module provides a web-based chat interface for the GLM-4V model with support for:
- Images, videos, PDFs, and PowerPoint files
- Real-time streaming responses with thinking visualization
- File upload and processing
- Interactive chat with conversation history
"""

import argparse
import copy
import os
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator

try:
    import gradio as gr
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("Gradio and PyMuPDF are required. Install with: pip install gradio PyMuPDF")

from ..engine import GLM4VInferenceEngine
from ..config import InferenceConfig, DEFAULT_WEB_CONFIG
from ..base import FileHandler, MessageBuilder


class GLM4VGradioApp:
    """Gradio web application for GLM-4V multimodal chat."""
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        """Initialize the Gradio application."""
        self.config = config or DEFAULT_WEB_CONFIG
        self.engine = GLM4VInferenceEngine(self.config)
        self.stop_generation = False
        
        # Load model on initialization
        self.engine.load_model()
        
    def strip_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        return re.sub(r"<[^>]+>", "", text).strip()
        
    def wrap_text(self, text: str) -> List[Dict[str, str]]:
        """Wrap text in message format."""
        return [{"type": "text", "text": text}]
        
    def pdf_to_images(self, pdf_path: str) -> List[str]:
        """Convert PDF pages to images."""
        doc = fitz.open(pdf_path)
        images = []
        
        for i in range(doc.page_count):
            pix = doc.load_page(i).get_pixmap(dpi=180)
            img_path = os.path.join(
                tempfile.gettempdir(), 
                f"{Path(pdf_path).stem}_{i}.png"
            )
            pix.save(img_path)
            images.append(img_path)
            
        doc.close()
        return images
        
    def ppt_to_images(self, ppt_path: str) -> List[str]:
        """Convert PowerPoint to images via PDF."""
        tmp_dir = tempfile.mkdtemp()
        try:
            # Convert to PDF using LibreOffice
            subprocess.run([
                "libreoffice", "--headless", "--convert-to", "pdf",
                "--outdir", tmp_dir, ppt_path
            ], check=True)
            
            pdf_path = os.path.join(tmp_dir, Path(ppt_path).stem + ".pdf")
            return self.pdf_to_images(pdf_path)
        except subprocess.CalledProcessError:
            # Fallback: return empty list if conversion fails
            return []
            
    def files_to_content(self, files: List[Any]) -> List[Dict[str, str]]:
        """Convert uploaded files to content format."""
        content = []
        
        for file in files or []:
            file_path = file.name
            ext = Path(file_path).suffix.lower()
            
            if ext in FileHandler.SUPPORTED_VIDEO_EXTENSIONS:
                content.append({"type": "video", "url": file_path})
            elif ext in FileHandler.SUPPORTED_IMAGE_EXTENSIONS:
                content.append({"type": "image", "url": file_path})
            elif ext in [".ppt", ".pptx"]:
                for img_path in self.ppt_to_images(file_path):
                    content.append({"type": "image", "url": img_path})
            elif ext == ".pdf":
                for img_path in self.pdf_to_images(file_path):
                    content.append({"type": "image", "url": img_path})
                    
        return content
        
    def stream_fragment(self, buffer: str) -> str:
        """Process streaming fragment with thinking visualization."""
        thinking_html = ""
        answer_html = ""
        
        # Extract thinking content
        if "<think>" in buffer:
            if "</think>" in buffer:
                match = re.search(r"<think>(.*?)</think>", buffer, re.DOTALL)
                if match:
                    thinking_content = match.group(1).strip().replace("\n", "<br>")
                    thinking_html = (
                        "<details open><summary style='cursor:pointer;font-weight:bold;color:#bbbbbb;'>💭 Thinking</summary>"
                        "<div style='color:#cccccc;line-height:1.4;padding:10px;border-left:3px solid #666;margin:5px 0;background-color:rgba(128,128,128,0.1);'>"
                        + thinking_content +
                        "</div></details>"
                    )
            else:
                # Partial thinking content
                part = buffer.split("<think>", 1)[1]
                thinking_html = (
                    "<details open><summary style='cursor:pointer;font-weight:bold;color:#bbbbbb;'>💭 Thinking</summary>"
                    "<div style='color:#cccccc;line-height:1.4;padding:10px;border-left:3px solid #666;margin:5px 0;background-color:rgba(128,128,128,0.1);'>"
                    + part.replace("\n", "<br>") +
                    "</div></details>"
                )
                
        # Extract answer content
        if "<answer>" in buffer:
            if "</answer>" in buffer:
                match = re.search(r"<answer>(.*?)</answer>", buffer, re.DOTALL)
                if match:
                    answer_html = match.group(1).strip()
            else:
                answer_html = buffer.split("<answer>", 1)[1]
                
        # Return formatted content
        if not thinking_html and not answer_html:
            return self.strip_html(buffer)
        return thinking_html + answer_html
        
    def build_messages(self, raw_history: List[Dict], system_prompt: str) -> List[Dict[str, Any]]:
        """Build message list from chat history."""
        messages = []
        
        # Add system prompt if provided
        if system_prompt.strip():
            messages.append(MessageBuilder.create_system_message(system_prompt.strip()))
            
        # Process history
        for entry in raw_history:
            if entry["role"] == "user":
                messages.append({"role": "user", "content": entry["content"]})
            else:
                # Clean assistant content
                content = entry["content"]
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
                content = re.sub(r"<details.*?</details>", "", content, flags=re.DOTALL)
                clean_content = self.strip_html(content).strip()
                messages.append(MessageBuilder.create_assistant_message(clean_content))
                
        return messages
        
    def stream_generate(self, raw_history: List[Dict], system_prompt: str) -> Iterator[str]:
        """Generate streaming response."""
        self.stop_generation = False
        messages = self.build_messages(raw_history, system_prompt)
        
        try:
            buffer = ""
            for token in self.engine.generate_stream(messages):
                if self.stop_generation:
                    break
                buffer += token
                yield self.stream_fragment(buffer)
        except Exception as e:
            yield f"Error during generation: {e}"
            
    def check_files(self, files: List[Any]) -> tuple[bool, str]:
        """Check file upload constraints."""
        if not files:
            return True, ""
            
        file_groups = {"video": 0, "image": 0, "document": 0}
        
        for file in files:
            ext = Path(file.name).suffix.lower()
            if ext in FileHandler.SUPPORTED_VIDEO_EXTENSIONS:
                file_groups["video"] += 1
            elif ext in FileHandler.SUPPORTED_IMAGE_EXTENSIONS:
                file_groups["image"] += 1
            elif ext in [".pdf", ".ppt", ".pptx"]:
                file_groups["document"] += 1
                
        # Check constraints
        if file_groups["video"] > 1 or file_groups["document"] > 1:
            return False, "Only one video or one document allowed"
        if file_groups["image"] > self.config.max_images:
            return False, f"Maximum {self.config.max_images} images allowed"
        if sum(file_groups.values()) > 1 and (file_groups["video"] > 0 or file_groups["document"] > 0):
            return False, "Cannot mix documents/videos with other files"
            
        return True, ""
        
    def format_display_content(self, content: Any) -> str:
        """Format content for display in chat."""
        if isinstance(content, list):
            text_parts = []
            file_count = 0
            
            for item in content:
                if item["type"] == "text":
                    text_parts.append(item["text"])
                else:
                    file_count += 1
                    
            display_text = " ".join(text_parts)
            if file_count > 0:
                return f"[{file_count} file(s) uploaded]\n{display_text}"
            return display_text
        return str(content)
        
    def create_display_history(self, raw_history: List[Dict]) -> List[Dict]:
        """Create display-friendly chat history."""
        display_history = []
        
        for entry in raw_history:
            if entry["role"] == "user":
                display_content = self.format_display_content(entry["content"])
                display_history.append({"role": "user", "content": display_content})
            else:
                display_history.append({"role": "assistant", "content": entry["content"]})
                
        return display_history
        
    def chat(self, files: List[Any], message: str, raw_history: List[Dict], 
            system_prompt: str) -> Iterator[tuple]:
        """Main chat function for Gradio interface."""
        self.stop_generation = False
        
        # Check file constraints
        is_valid, error_msg = self.check_files(files)
        if not is_valid:
            if raw_history is None:
                raw_history = []
            raw_history.append({"role": "assistant", "content": error_msg})
            display_history = self.create_display_history(raw_history)
            yield display_history, copy.deepcopy(raw_history), None, ""
            return
            
        # Build content
        content = self.files_to_content(files) if files else []
        if message.strip():
            content.append({"type": "text", "text": message.strip()})
            
        # Add user message
        if raw_history is None:
            raw_history = []
        user_entry = {"role": "user", "content": content if content else message.strip()}
        raw_history.append(user_entry)
        
        # Create placeholder for assistant response
        assistant_entry = {"role": "assistant", "content": ""}
        raw_history.append(assistant_entry)
        
        # Initial update
        display_history = self.create_display_history(raw_history)
        yield display_history, copy.deepcopy(raw_history), None, ""
        
        # Generate streaming response
        try:
            for chunk in self.stream_generate(raw_history[:-1], system_prompt):
                if self.stop_generation:
                    break
                assistant_entry["content"] = chunk
                display_history = self.create_display_history(raw_history)
                yield display_history, copy.deepcopy(raw_history), None, ""
        except Exception as e:
            assistant_entry["content"] = f"Error: {e}"
            display_history = self.create_display_history(raw_history)
            yield display_history, copy.deepcopy(raw_history), None, ""
            
    def reset(self) -> tuple:
        """Reset the chat interface."""
        self.stop_generation = True
        time.sleep(0.1)
        return [], [], None, ""
        
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        css = """.chatbot-container .message-wrap .message{font-size:14px!important}
        details summary{cursor:pointer;font-weight:bold}
        details[open] summary{margin-bottom:10px}"""
        
        with gr.Blocks(title="GLM-4V Chat", theme=gr.themes.Soft(), css=css) as demo:
            gr.Markdown("""
            <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
                GLM-4.1V-9B-Thinking Gradio Interface
            </div>
            <div style="text-align: center;">
                Powered by local GLM4V model components
            </div>
            """)
            
            raw_history = gr.State([])
            
            with gr.Row():
                with gr.Column(scale=7):
                    chatbox = gr.Chatbot(
                        label="Conversation",
                        type="messages",
                        height=800,
                        elem_classes="chatbot-container",
                    )
                    textbox = gr.Textbox(label="💭 Message")
                    with gr.Row():
                        send = gr.Button("Send", variant="primary")
                        clear = gr.Button("Clear")
                        
                with gr.Column(scale=3):
                    upload = gr.File(
                        label="📁 Upload",
                        file_count="multiple",
                        file_types=["file"],
                        type="filepath",
                    )
                    gr.Markdown("Supports images / videos / PPT / PDF")
                    gr.Markdown(
                        f"Maximum {self.config.max_images} images or 1 video/PPT/PDF. "
                        "Videos and images cannot be mixed."
                    )
                    system_prompt = gr.Textbox(label="⚙️ System Prompt", lines=6)
                    
            # Event handlers
            gr.on(
                triggers=[send.click, textbox.submit],
                fn=self.chat,
                inputs=[upload, textbox, raw_history, system_prompt],
                outputs=[chatbox, raw_history, upload, textbox],
            )
            clear.click(self.reset, outputs=[chatbox, raw_history, upload, textbox])
            
        return demo
        
    def launch(self, **kwargs) -> None:
        """Launch the Gradio interface."""
        demo = self.create_interface()
        
        launch_kwargs = {
            "server_name": self.config.server_name,
            "server_port": self.config.server_port,
            "share": self.config.share,
            "inbrowser": True,
        }
        launch_kwargs.update(kwargs)
        
        demo.launch(**launch_kwargs)


def main():
    """Main entry point for the Gradio application."""
    parser = argparse.ArgumentParser(description="GLM-4V Gradio Web Interface")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, default="THUDM/GLM-4.1V-9B-Thinking",
                       help="Path to the model")
    
    # Server arguments
    parser.add_argument("--server_name", type=str, default="127.0.0.1",
                       help="Server host (use 0.0.0.0 for LAN access)")
    parser.add_argument("--server_port", type=int, default=7860,
                       help="Server port")
    parser.add_argument("--share", action="store_true",
                       help="Enable Gradio sharing")
    
    # Generation arguments
    parser.add_argument("--max_tokens", type=int, default=8192,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Generation temperature")
    
    args = parser.parse_args()
    
    # Create configuration
    config = InferenceConfig(
        model_path=args.model_path,
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    
    # Create and launch app
    app = GLM4VGradioApp(config)
    app.launch()


if __name__ == "__main__":
    main() 