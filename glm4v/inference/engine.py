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
Core inference engine for GLM-4V with unified generation logic.
"""

import torch
from typing import List, Dict, Any, Optional, Iterator
from transformers import TextIteratorStreamer
import threading

from ..modeling_glm4v import Glm4vForConditionalGeneration
from ..processing_glm4v import Glm4vProcessor
from .base import BaseInference
from .config import InferenceConfig


class GLM4VInferenceEngine(BaseInference):
    """Core inference engine with unified generation logic for GLM-4V."""
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        """Initialize the inference engine."""
        super().__init__(config)
        self.processor = None
        self.model = None
        self.special_token_ids = {}
        self._is_loaded = False
        
    def load_model(self):
        """Load the GLM-4V model and processor."""
        if self._is_loaded:
            return
            
        print(f"Loading GLM-4V model from {self.config.model_path}...")
        
        # Load processor using local GLM4V components
        self.processor = Glm4vProcessor.from_pretrained(
            self.config.model_path, 
            use_fast=self.config.use_fast_processor
        )
        
        # Load model using local GLM4V components
        self.model = Glm4vForConditionalGeneration.from_pretrained(
            self.config.model_path,
            torch_dtype=self.config.torch_dtype,
            device_map=self.config.device_map,
            attn_implementation=self.config.attn_implementation,
        )
        
        # Get special token IDs for thinking/answer extraction
        self._initialize_special_tokens()
        
        self._is_loaded = True
        print("Model loaded successfully!")
        
    def _initialize_special_tokens(self):
        """Initialize special token IDs for robust inference."""
        special_tokens = self.config.get_special_tokens()
        
        self.special_token_ids = {
            "think_start": self.processor.tokenizer.convert_tokens_to_ids(special_tokens["think_start"]),
            "think_end": self.processor.tokenizer.convert_tokens_to_ids(special_tokens["think_end"]),
            "answer_start": self.processor.tokenizer.convert_tokens_to_ids(special_tokens["answer_start"]),
            "answer_end": self.processor.tokenizer.convert_tokens_to_ids(special_tokens["answer_end"]),
        }
        
    def generate(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """Generate response from messages."""
        if not self._is_loaded:
            self.load_model()
            
        # Prepare inputs
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)
        
        # Get generation kwargs
        gen_kwargs = self.config.to_generation_kwargs()
        gen_kwargs.update(kwargs)
        
        # Generate response
        with torch.no_grad():
            output = self.model.generate(**inputs, **gen_kwargs)
            
        # Decode response
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = output[0][input_length:-1]  # Remove EOS token
        raw_output = self.processor.decode(generated_tokens, skip_special_tokens=False)
        
        return raw_output
    
    def generate_stream(self, messages: List[Dict[str, Any]], **kwargs) -> Iterator[str]:
        """Generate streaming response from messages."""
        if not self._is_loaded:
            self.load_model()
            
        # Prepare inputs
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)
        
        # Setup streamer
        streamer = TextIteratorStreamer(
            self.processor.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=False
        )
        
        # Get generation kwargs
        gen_kwargs = self.config.to_generation_kwargs()
        gen_kwargs.update(kwargs)
        gen_kwargs["streamer"] = streamer
        
        # Start generation in a separate thread
        generation_thread = threading.Thread(
            target=self.model.generate, 
            kwargs={**inputs, **gen_kwargs}
        )
        generation_thread.start()
        
        # Yield tokens as they come
        for token in streamer:
            yield token
            
        generation_thread.join()
    
    def generate_with_force_completion(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Generate with force completion for robust academic benchmarking.
        Ensures complete thinking and answer output even if max tokens is reached.
        """
        if not self._is_loaded:
            self.load_model()
            
        # Prepare inputs
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)
        
        input_length = inputs["input_ids"].shape[1]
        
        # First generation round
        first_gen_kwargs = {
            "max_new_tokens": self.config.first_max_tokens,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "do_sample": kwargs.get("do_sample", self.config.do_sample),
        }
        
        first_output = self.model.generate(**inputs, **first_gen_kwargs)
        first_generated_tokens = first_output[0][input_length:]
        
        # Check if completion is needed
        needs_completion = self._check_needs_completion(
            first_generated_tokens, self.config.first_max_tokens
        )
        
        if not needs_completion:
            raw_output = self.processor.decode(first_generated_tokens, skip_special_tokens=False)
            return {
                "output_text": raw_output,
                "complete": True,
                "reason": "first_generation_complete",
                "input_length": input_length,
                "first_generation_length": len(first_generated_tokens),
                "generation_rounds": 1,
            }
        
        # Force completion if needed
        force_inputs = self._prepare_force_input(inputs["input_ids"], first_generated_tokens)
        
        force_input_dict = {
            "input_ids": force_inputs.to(self.model.device),
            "attention_mask": torch.ones_like(force_inputs).to(self.model.device),
        }
        
        # Copy multimodal inputs if present
        for key in ["pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"]:
            if key in inputs:
                force_input_dict[key] = inputs[key]
        
        # Second generation round
        second_gen_kwargs = {
            "max_new_tokens": self.config.force_max_tokens,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "do_sample": kwargs.get("do_sample", self.config.do_sample),
        }
        
        second_output = self.model.generate(**force_input_dict, **second_gen_kwargs)
        second_generated_tokens = second_output[0][force_inputs.shape[1]:]
        
        # Combine outputs
        added_tokens = force_inputs[0][input_length + len(first_generated_tokens):]
        complete_tokens = torch.cat([first_generated_tokens, added_tokens, second_generated_tokens], dim=0)
        complete_output = self.processor.decode(complete_tokens, skip_special_tokens=False)
        
        return {
            "output_text": complete_output,
            "complete": (self.special_token_ids["answer_end"] in complete_tokens.tolist()),
            "reason": "force_completion_success",
            "input_length": input_length,
            "first_generation_length": len(first_generated_tokens),
            "second_generation_length": len(second_generated_tokens),
            "total_generation_length": len(complete_tokens),
            "generation_rounds": 2,
        }
    
    def _check_needs_completion(self, output_tokens: torch.Tensor, max_tokens: int) -> bool:
        """Check if force completion is needed."""
        token_list = output_tokens.tolist()
        
        # Check conditions
        reached_max = len(token_list) >= max_tokens
        has_answer_end = self.special_token_ids["answer_end"] in token_list
        has_think_start = self.special_token_ids["think_start"] in token_list
        has_think_end = self.special_token_ids["think_end"] in token_list
        
        # If we have complete answer, no completion needed
        if has_answer_end:
            return False
            
        # If we reached max tokens or started thinking without completion
        if reached_max or (has_think_start and not has_think_end):
            return True
            
        return False
    
    def _prepare_force_input(self, original_input_ids: torch.Tensor, 
                           first_output_tokens: torch.Tensor) -> torch.Tensor:
        """Prepare input for force completion by adding missing tokens."""
        first_output_list = first_output_tokens.tolist()
        
        has_think_end = self.special_token_ids["think_end"] in first_output_list
        has_answer_start = self.special_token_ids["answer_start"] in first_output_list
        
        tokens_to_add = []
        
        # Add missing tokens to complete the structure
        if not has_think_end:
            tokens_to_add.extend([
                self.special_token_ids["think_end"], 
                self.special_token_ids["answer_start"]
            ])
        elif not has_answer_start:
            tokens_to_add.append(self.special_token_ids["answer_start"])
        
        # Build the force input
        if tokens_to_add:
            additional_tokens = torch.tensor(tokens_to_add).unsqueeze(0).to(self.model.device)
            force_inputs = torch.cat([
                original_input_ids, 
                first_output_tokens.unsqueeze(0), 
                additional_tokens
            ], dim=1)
        else:
            force_inputs = torch.cat([
                original_input_ids, 
                first_output_tokens.unsqueeze(0)
            ], dim=1)
            
        return force_inputs
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self._is_loaded:
            return {"status": "not_loaded"}
            
        return {
            "status": "loaded",
            "model_path": self.config.model_path,
            "device": str(self.model.device) if self.model else "unknown",
            "dtype": str(self.config.torch_dtype),
            "special_tokens": self.special_token_ids,
        } 