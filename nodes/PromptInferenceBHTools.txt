import torch
import json
import random
import re
import gc
import time
from typing import Tuple

try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, pipeline,
        BitsAndBytesConfig, GenerationConfig
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class PromptInferenceBHTools:
    """
    BHTools - Professional prompt enhancement using local LLM inference.
    Supports custom trigger words and ensures original prompt is embedded in output.
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_model_name = None
        
        # Model mappings - using actual HuggingFace model names
        self.model_mappings = {
            "Qwen2.5-1.5B [Best Quality]": "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen2.5-3B [High Quality]": "Qwen/Qwen2.5-3B-Instruct", 
            "Qwen2.5-7B [Maximum Quality]": "Qwen/Qwen2.5-7B-Instruct",
            "Llama-3.2-1B [Fast]": "meta-llama/Llama-3.2-1B-Instruct",
            "Llama-3.2-3B [Balanced]": "meta-llama/Llama-3.2-3B-Instruct"
        }
        
        # Quality enhancement words
        self.quality_words = {
            "photography": ["professional photography", "high resolution", "detailed", "sharp focus"],
            "artistic": ["artistic masterpiece", "beautiful", "creative", "expressive"],
            "cinematic": ["cinematic lighting", "dramatic", "epic", "movie scene"],
            "realistic": ["photorealistic", "ultra detailed", "lifelike", "high quality"]
        }
        
        # Enhancement templates
        self.enhancement_templates = {
            "photography": "Enhance this photography prompt with professional details: {prompt}",
            "artistic": "Create an artistic enhancement for: {prompt}",
            "cinematic": "Add cinematic elements to: {prompt}",
            "realistic": "Make this more realistic and detailed: {prompt}"
        }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A man in a monkey outfit",
                    "placeholder": "Enter your base prompt here."
                }),
                "trigger_words": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "e.g. lora:animeLora1:0.8, lora:lightingLora:0.5"
                }),
                "model_selection": ([
                    "Qwen2.5-1.5B [Best Quality]",
                    "Qwen2.5-3B [High Quality]", 
                    "Qwen2.5-7B [Maximum Quality]",
                    "Llama-3.2-1B [Fast]",
                    "Llama-3.2-3B [Balanced]",
                    "Fallback [No Model]"
                ], {"default": "Fallback [No Model]"}),
                "enhancement_style": ([
                    "photography",
                    "artistic", 
                    "cinematic",
                    "realistic"
                ], {"default": "photography"}),
                "creativity_level": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "max_length": ("INT", {
                    "default": 150,
                    "min": 50,
                    "max": 500,
                    "step": 5,
                    "display": "slider"
                })
            },
            "optional": {
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647
                }),
                "purge_cache": ("BOOLEAN", {
                    "default": False
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_prompt",)
    FUNCTION = "enhance_prompt"
    CATEGORY = "BHTools/conditioning"

    def get_model_name(self, model_selection):
        """Get actual model name from selection"""
        return self.model_mappings.get(model_selection, None)

    def load_model_optimized(self, model_name):
        """Load model with optimized settings"""
        if not TRANSFORMERS_AVAILABLE:
            print("⚠️ Transformers not available, using fallback enhancement")
            return False
            
        if self.current_model_name == model_name and self.model is not None:
            return True
            
        # Cleanup previous model
        self.cleanup_model()
        
        try:
            print(f"🔄 Loading model: {model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=True
            )
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            self.current_model_name = model_name
            print(f"✅ Successfully loaded model: {model_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model {model_name}: {str(e)}")
            print("🔄 Falling back to rule-based enhancement")
            return False
            
    def cleanup_model(self):
        """Clean up model resources"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.current_model_name = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def enhance_prompt(
        self,
        prompt: str,
        trigger_words: str,
        model_selection: str,
        enhancement_style: str,
        creativity_level: float,
        max_length: int,
        seed: int = -1,
        purge_cache: bool = False
    ) -> Tuple[str]:
        """
        Main enhancement function
        """
        # Set seed for reproducibility
        if seed != -1:
            torch.manual_seed(seed)
            random.seed(seed)

        # Clean inputs
        prompt = prompt.strip()
        trigger_words = trigger_words.strip()

        # Handle model loading
        model_loaded = False
        if model_selection != "Fallback [No Model]":
            model_name = self.get_model_name(model_selection)
            if model_name:
                model_loaded = self.load_model_optimized(model_name)

        # Generate enhanced prompt
        if model_loaded and self.model is not None:
            enhanced = self.generate_with_model(prompt, enhancement_style, creativity_level, max_length)
        else:
            enhanced = self.create_fallback_enhancement(prompt, enhancement_style, creativity_level)

        # Ensure original prompt is included
        if prompt.lower() not in enhanced.lower():
            enhanced = f"{prompt}, {enhanced}"

        # Prepend trigger words if provided
        if trigger_words:
            enhanced = f"{trigger_words}, {enhanced}"

        # Clean up if requested
        if purge_cache:
            self.cleanup_model()

        return (enhanced,)

    def generate_with_model(self, prompt, style, creativity_level, max_length):
        """Generate enhanced prompt using the loaded model"""
        try:
            # Create instruction
            template = self.enhancement_templates.get(style, self.enhancement_templates["photography"])
            instruction = template.format(prompt=prompt)
            
            # Format for chat models
            messages = [
                {"role": "system", "content": "You are a professional prompt engineer. Enhance the given prompt with relevant details while keeping it concise."},
                {"role": "user", "content": instruction}
            ]
            
            # Apply chat template if available
            if hasattr(self.tokenizer, 'apply_chat_template'):
                formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                formatted_prompt = instruction
            
            # Tokenize
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Move to device
            if torch.cuda.is_available() and self.model.device.type == 'cuda':
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=creativity_level,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract enhancement
            enhanced = self.extract_enhancement(generated_text, formatted_prompt)
            return self.clean_output(enhanced)
            
        except Exception as e:
            print(f"❌ Error during generation: {str(e)}")
            return self.create_fallback_enhancement(prompt, style, creativity_level)

    def extract_enhancement(self, generated_text, instruction):
        """Extract the enhancement from generated text"""
        # Try to find the assistant's response
        if "assistant" in generated_text.lower():
            parts = generated_text.split("assistant")
            if len(parts) > 1:
                enhanced = parts[-1].strip()
            else:
                enhanced = generated_text
        else:
            # Remove the instruction part
            enhanced = generated_text.replace(instruction, "").strip()
        
        return enhanced

    def clean_output(self, text):
        """Clean and format the output"""
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove common unwanted patterns
        patterns_to_remove = [
            r'^(Here\'s|Here is|The enhanced prompt is:?|Enhanced prompt:)',
            r'^(A|An)\s+',
            r'^\W+',
            r'\W+$'
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
        
        # Ensure it doesn't start with punctuation
        if text and text[0] in '.,!?;':
            text = text[1:].strip()
            
        return text

    def create_fallback_enhancement(self, prompt, style, creativity_level):
        """Create a fallback enhancement when model is not available"""
        quality_words = self.quality_words.get(style, self.quality_words["photography"])
        
        # Select enhancement words based on creativity level
        num_words = min(3, max(1, int(creativity_level * 4)))
        selected_words = random.sample(quality_words, min(num_words, len(quality_words)))
        
        enhancement = ", ".join(selected_words)
        
        # Add extra details for higher creativity
        if creativity_level > 0.7:
            extra_details = {
                "photography": "8K resolution, professional lighting",
                "artistic": "masterpiece, award winning",
                "cinematic": "dramatic composition, epic scale",
                "realistic": "hyperrealistic, ultra detailed"
            }
            enhancement += f", {extra_details.get(style, 'high quality')}"
        
        return enhancement

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "PromptInferenceBHTools": PromptInferenceBHTools,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptInferenceBHTools": "🎬 Prompt Inference | BH Tools",
}