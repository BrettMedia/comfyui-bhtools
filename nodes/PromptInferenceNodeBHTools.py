# comfyui-bhtools/nodes/PromptInferenceNodeBHTools.py

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
    print("Warning: 'transformers' library not found. PromptInferenceBHTools will use fallback enhancement.")


class PromptInferenceBHTools:
    """
    BHTools - Professional prompt enhancement using local LLM inference.
    Supports custom trigger words, negative prompt enhancement, customizable system prompts,
    and ensures original prompt is embedded in output.
    This node requires the 'transformers' library to be installed (pip install transformers).
    """

    # Model mappings - now a class-level attribute
    model_mappings = {
        "Qwen2.5-1.5B [Best Quality]": "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen2.5-3B [High Quality]": "Qwen/Qwen2.5-3B-Instruct",
        "Qwen2.5-7B [Maximum Quality]": "Qwen/Qwen2.5-7B-Instruct",
        "Llama-3.2-1B [Fast]": "meta-llama/Llama-3.2-1B-Instruct",
        "Llama-3.2-3B [Balanced]": "meta-llama/Llama-3.2-3B-Instruct"
    }

    # Quality enhancement words for fallback - now a class-level attribute
    quality_words = {
        "photography": ["professional photography", "high resolution", "detailed", "sharp focus", "natural lighting", "bokeh", "depth of field"],
        "artistic": ["artistic masterpiece", "beautiful composition", "creative vision", "expressive brushstrokes", "vibrant colors", "unique style"],
        "cinematic": ["cinematic lighting", "dramatic composition", "epic scale", "movie scene", "wide shot", "film grain", "anamorphic"],
        "realistic": ["photorealistic", "ultra detailed", "lifelike", "high quality textures", "accurate anatomy", "subtle imperfections"]
    }

    # Enhancement templates for LLM - now a class-level attribute
    enhancement_templates = {
        "positive": {
            "photography": "Enhance this photography prompt with professional details: {prompt}",
            "artistic": "Create an artistic enhancement for: {prompt}",
            "cinematic": "Add cinematic elements to: {prompt}",
            "realistic": "Make this more realistic and detailed: {prompt}"
        },
        "negative": {
            "general": "Suggest negative prompt elements to avoid common imperfections for: {prompt}",
            "photography": "Suggest negative prompt elements to avoid common photography flaws for: {prompt}",
            "artistic": "Suggest negative prompt elements to avoid common artistic flaws for: {prompt}",
            "cinematic": "Suggest negative prompt elements to avoid common cinematic flaws for: {prompt}",
            "realistic": "Suggest negative prompt elements to avoid common realism flaws for: {prompt}"
        }
    }

    # Pre-defined system prompts for selection - NEW CLASS-LEVEL ATTRIBUTE
    PREDEFINED_SYSTEM_PROMPTS = {
        "positive": {
            "Standard Enhancement": "You are a professional prompt engineer. Enhance the given prompt with relevant details while keeping it concise and focused on the core subject.",
            "Creative Expansion": "You are a highly creative prompt engineer. Expand on the given prompt with imaginative and unique details, aiming for a highly artistic and original output.",
            "Concise Refinement": "You are a minimalist prompt engineer. Refine the given prompt to its most essential and impactful elements, removing any redundancy while maintaining clarity.",
            "Technical Detail": "You are a technical prompt engineer. Add highly specific and technical details to the given prompt, focusing on camera settings, lighting, and artistic techniques.",
            "No Specific System Prompt": "" # Option for no specific system prompt
        },
        "negative": {
            "Standard Negative": "You are a professional prompt engineer. Suggest concise negative prompt elements to improve image quality by avoiding common flaws and undesirable elements. Focus on what to exclude.",
            "Quality Control": "You are a quality control expert. List negative prompt elements to eliminate common defects, blurriness, distortions, and low-quality aspects from the image.",
            "Artistic Flaws": "You are an art critic. Provide negative prompt elements to avoid typical artistic errors, poor composition, or unappealing aesthetics.",
            "No Specific System Prompt": "" # Option for no specific system prompt
        }
    }

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_model_name = None


    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A man in a monkey outfit",
                    "placeholder": "Enter your base positive prompt here."
                }),
                "trigger_words": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "e.g. lora:animeLora1:0.8, lora:lightingLora:0.5",
                    "tooltip": "Additional trigger words or LoRA tags to prepend to the enhanced positive prompt."
                }),
                "model_selection": ([
                    "Qwen2.5-1.5B [Best Quality]",
                    "Qwen2.5-3B [High Quality]",
                    "Qwen2.5-7B [Maximum Quality]",
                    "Llama-3.2-1B [Fast]",
                    "Llama-3.2-3B [Balanced]",
                    "Fallback [No Model]"
                ], {"default": "Fallback [No Model]", "tooltip": "Select a local HuggingFace LLM for prompt enhancement. 'Fallback' uses rule-based enhancement."}),
                "enhancement_style": ([
                    "photography",
                    "artistic",
                    "cinematic",
                    "realistic"
                ], {"default": "photography", "tooltip": "The stylistic direction for the LLM enhancement of the positive prompt."}),
                "creativity_level": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Controls the randomness and creativity of the LLM's output (higher = more creative)."
                }),
                "max_length": ("INT", {
                    "default": 150,
                    "min": 50,
                    "max": 500,
                    "step": 5,
                    "display": "slider",
                    "tooltip": "Maximum number of new tokens the LLM will generate for the enhancement."
                }),
                # New pulldown for system prompt method
                "system_prompt_method": (list(cls.PREDEFINED_SYSTEM_PROMPTS["positive"].keys()), {
                    "default": "Standard Enhancement",
                    "tooltip": "Choose a pre-defined system prompt strategy for positive prompt enhancement."
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "", # Default is now empty, relies on method selection or manual override
                    "placeholder": "Manual override for positive system prompt (optional).",
                    "tooltip": "Manually override the selected system prompt method. Leave blank to use the chosen method."
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter your base negative prompt here (optional).",
                    "tooltip": "Optional base negative prompt to be enhanced."
                }),
                "enhance_negative_prompt": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If checked, the LLM will also enhance the negative prompt."
                }),
                # New pulldown for negative system prompt method
                "negative_system_prompt_method": (list(cls.PREDEFINED_SYSTEM_PROMPTS["negative"].keys()), {
                    "default": "Standard Negative",
                    "tooltip": "Choose a pre-defined system prompt strategy for negative prompt enhancement."
                }),
                "negative_system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "", # Default is now empty, relies on method selection or manual override
                    "placeholder": "Manual override for negative system prompt (optional).",
                    "tooltip": "Manually override the selected negative system prompt method. Leave blank to use the chosen method."
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "tooltip": "Random seed for reproducible LLM generation. -1 for random."
                }),
                "purge_cache": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If checked, unloads the LLM model from VRAM after generation."
                })
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("enhanced_positive_prompt", "enhanced_negative_prompt",)
    FUNCTION = "enhance_prompt"
    CATEGORY = "BH Tools/Prompting" # Standardized category name for ComfyUI menu

    def get_model_name(self, model_selection):
        """Maps user-friendly model selection to HuggingFace model name."""
        # Access class-level attribute directly
        return self.model_mappings.get(model_selection, None)

    def load_model_optimized(self, model_name):
        """
        Loads the specified HuggingFace model and tokenizer with optimizations.
        Checks if the model is already loaded to avoid redundant loading.
        """
        if not TRANSFORMERS_AVAILABLE:
            print("⚠️ PromptInferenceBHTools: 'transformers' library not available, cannot load model. Using fallback enhancement.")
            return False

        if self.current_model_name == model_name and self.model is not None:
            print(f"✅ PromptInferenceBHTools: Model '{model_name}' already loaded. Skipping.")
            return True

        # Cleanup previous model before loading a new one
        self.cleanup_model()

        try:
            print(f"🔄 PromptInferenceBHTools: Loading model: {model_name}...")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=True
            )

            # Add pad token if missing (common for some models)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with optimizations (float16 for GPU, auto device mapping)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True # Reduces RAM usage during loading
            )

            self.current_model_name = model_name
            print(f"✅ PromptInferenceBHTools: Successfully loaded model: {model_name}")
            return True

        except Exception as e:
            print(f"❌ PromptInferenceBHTools: Error loading model {model_name}: {str(e)}")
            print("🔄 PromptInferenceBHTools: Falling back to rule-based enhancement.")
            self.cleanup_model() # Ensure resources are freed on error
            return False

    def cleanup_model(self):
        """
        Unloads the current model and tokenizer from memory and clears CUDA cache.
        This helps free up VRAM and RAM.
        """
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.current_model_name = None
        gc.collect() # Force Python garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache() # Clear PyTorch's CUDA cache
            print("✅ PromptInferenceBHTools: VRAM cache cleared.")

    def enhance_prompt(
        self,
        prompt: str,
        trigger_words: str,
        model_selection: str,
        enhancement_style: str,
        creativity_level: float,
        max_length: int,
        system_prompt_method: str, # New input for system prompt method
        system_prompt: str, # This is now the manual override
        negative_prompt: str = "",
        enhance_negative_prompt: bool = False,
        negative_system_prompt_method: str = "", # New input for negative system prompt method
        negative_system_prompt: str = "", # This is now the manual override
        seed: int = -1,
        purge_cache: bool = False
    ) -> Tuple[str, str]:
        """
        Main function to enhance the input prompt(s) using either a loaded LLM
        or a rule-based fallback.
        """
        # Set seed for reproducibility
        if seed != -1:
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed) # For all CUDA devices
            torch.manual_seed(seed) # For CPU operations
            random.seed(seed) # For Python's random module

        # Clean and strip input strings
        prompt = prompt.strip()
        trigger_words = trigger_words.strip()
        negative_prompt = negative_prompt.strip()

        # Determine the effective positive system prompt
        # If user provides a manual system_prompt, it overrides the method selection
        effective_system_prompt = system_prompt.strip()
        if not effective_system_prompt: # If manual override is empty, use the selected method
            effective_system_prompt = self.PREDEFINED_SYSTEM_PROMPTS["positive"].get(system_prompt_method, "")

        # Determine the effective negative system prompt
        effective_negative_system_prompt = negative_system_prompt.strip()
        if not effective_negative_system_prompt: # If manual override is empty, use the selected method
            effective_negative_system_prompt = self.PREDEFINED_SYSTEM_PROMPTS["negative"].get(negative_system_prompt_method, "")


        # Handle model loading based on user selection
        model_loaded = False
        if model_selection != "Fallback [No Model]":
            model_name = self.get_model_name(model_selection)
            if model_name:
                model_loaded = self.load_model_optimized(model_name)

        # --- Enhance Positive Prompt ---
        print(f"🚀 PromptInferenceBHTools: Enhancing positive prompt...")
        if model_loaded and self.model is not None:
            enhanced_positive = self.generate_with_model(
                prompt=prompt,
                style=enhancement_style,
                creativity_level=creativity_level,
                max_length=max_length,
                system_prompt=effective_system_prompt, # Use the effective system prompt
                prompt_type="positive"
            )
        else:
            enhanced_positive = self.create_fallback_enhancement(
                prompt=prompt,
                style=enhancement_style,
                creativity_level=creativity_level,
                prompt_type="positive"
            )

        # Ensure original positive prompt is included if it's not already part of the enhanced text
        if prompt and prompt.lower() not in enhanced_positive.lower() and not enhanced_positive.startswith(prompt):
            enhanced_positive = f"{prompt}, {enhanced_positive}"

        # Prepend trigger words if provided
        if trigger_words:
            enhanced_positive = f"{trigger_words}, {enhanced_positive}"

        # --- Enhance Negative Prompt (if enabled) ---
        enhanced_negative = negative_prompt # Start with original negative prompt
        if enhance_negative_prompt and negative_prompt:
            print(f"🚀 PromptInferenceBHTools: Enhancing negative prompt...")
            if model_loaded and self.model is not None:
                enhanced_negative_llm = self.generate_with_model(
                    prompt=negative_prompt,
                    style=enhancement_style, # Use same style for consistency, or a 'negative' specific one
                    creativity_level=creativity_level,
                    max_length=max_length,
                    system_prompt=effective_negative_system_prompt, # Use the effective negative system prompt
                    prompt_type="negative"
                )
                # Combine original negative prompt with LLM enhancement
                if negative_prompt.lower() not in enhanced_negative_llm.lower() and not enhanced_negative_llm.startswith(negative_prompt):
                    enhanced_negative = f"{negative_prompt}, {enhanced_negative_llm}"
                else:
                    enhanced_negative = enhanced_negative_llm
            else:
                # Fallback for negative prompt is simpler, usually just appending common negative terms
                enhanced_negative_fallback = self.create_fallback_enhancement(
                    prompt=negative_prompt,
                    style=enhancement_style,
                    creativity_level=creativity_level,
                    prompt_type="negative"
                )
                if negative_prompt.lower() not in enhanced_negative_fallback.lower() and not enhanced_negative_fallback.startswith(negative_prompt):
                    enhanced_negative = f"{negative_prompt}, {enhanced_negative_fallback}"
                else:
                    enhanced_negative = enhanced_negative_fallback
        elif enhance_negative_prompt and not negative_prompt:
            print("ℹ️ PromptInferenceBHTools: Negative prompt enhancement enabled but no base negative prompt provided. Skipping enhancement.")


        # Clean up model from VRAM if 'purge_cache' is enabled
        if purge_cache:
            self.cleanup_model()

        print(f"✅ PromptInferenceBHTools: Positive Prompt Output: {enhanced_positive}")
        if enhance_negative_prompt:
            print(f"✅ PromptInferenceBHTools: Negative Prompt Output: {enhanced_negative}")

        return (enhanced_positive, enhanced_negative,) # Return as a tuple as expected by ComfyUI

    def generate_with_model(self, prompt, style, creativity_level, max_length, system_prompt, prompt_type="positive"):
        """
        Generates an enhanced prompt using the loaded HuggingFace model.
        """
        try:
            # Select appropriate template based on prompt type and style
            template_dict = self.enhancement_templates.get(prompt_type, {})
            template = template_dict.get(style, template_dict.get("general", "{prompt}")) # Fallback to general if style specific not found
            instruction = template.format(prompt=prompt)

            # Format messages for chat-tuned models (e.g., Instruct models)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction}
            ]

            # Apply chat template if the tokenizer supports it, otherwise use raw instruction
            if hasattr(self.tokenizer, 'apply_chat_template'):
                formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                formatted_prompt = instruction

            # Tokenize the input prompt
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512, # Max input tokens for the model
                padding=True
            )

            # Move input tensors to the model's device (GPU if available)
            if torch.cuda.is_available() and self.model.device.type == 'cuda':
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate new tokens using the model (synchronous operation)
            with torch.no_grad(): # Disable gradient calculation for inference
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length, # Max tokens to generate
                    temperature=creativity_level, # Controls randomness
                    do_sample=True, # Enable sampling for temperature to work
                    pad_token_id=self.tokenizer.pad_token_id, # Ensure proper padding handling
                    eos_token_id=self.tokenizer.eos_token_id, # Stop generation at EOS token
                    repetition_penalty=1.1, # Penalize repetition
                    # Add more generation parameters for robustness if needed, e.g.:
                    # top_p=0.9,
                    # top_k=50,
                    # num_beams=1, # For greedy decoding
                )

            # Decode the generated tokens back to text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract and clean the relevant enhancement from the generated text
            enhanced = self.extract_enhancement(generated_text, formatted_prompt, system_prompt)
            return self.clean_output(enhanced)

        except Exception as e:
            print(f"❌ PromptInferenceBHTools: Error during model generation: {str(e)}")
            # Fallback to rule-based enhancement if model generation fails
            return self.create_fallback_enhancement(prompt, style, creativity_level, prompt_type)

    def extract_enhancement(self, generated_text, instruction, system_prompt):
        """
        Extracts the generated enhancement from the full LLM output.
        This handles cases where the LLM might echo the input or system messages.
        """
        # Try to remove the system prompt first
        if system_prompt and system_prompt in generated_text:
            generated_text = generated_text.replace(system_prompt, "").strip()

        # Try to remove the instruction (user prompt)
        if instruction and instruction in generated_text:
            generated_text = generated_text.replace(instruction, "").strip()

        # Look for common chat model delimiters for assistant's response
        # This handles cases where the model might re-include user/system turns
        patterns_to_remove = [
            r"\[INST\].*?\[/INST\]",  # Llama-style instruction tags
            r"<\|user\|>.*?<\|endoftext\|>", # Qwen-style user tags
            r"<\|system\|>.*?<\|endoftext\|>", # Qwen-style system tags
            r"user\n", # Common raw user turn indicator
            r"system\n", # Common raw system turn indicator
            r"assistant\n", # Common raw assistant turn indicator
            r"\[/INST\]", # Trailing instruction tag
            r"<\|endoftext\|>", # Common end of text token
        ]

        for pattern in patterns_to_remove:
            generated_text = re.sub(pattern, '', generated_text, flags=re.IGNORECASE | re.DOTALL).strip()

        # After removing known patterns, if the text still contains the original instruction
        # (e.g., if the model just repeated it without tags), remove it again.
        if instruction in generated_text:
            generated_text = generated_text.replace(instruction, "").strip()
        # The prompt_type variable is not available in this scope, so remove the condition
        # if prompt_type == "negative": # For negative prompts, sometimes models add "negative prompt:"
        #     generated_text = re.sub(r"^(negative prompt:?|negative:)", "", generated_text, flags=re.IGNORECASE).strip()

        return generated_text.strip()

    def clean_output(self, text):
        """
        Cleans and formats the final output string, removing unwanted patterns
        and ensuring proper spacing.
        """
        # Remove multiple whitespaces and leading/trailing whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Define common unwanted patterns to remove from the start/end of the prompt
        patterns_to_remove = [
            r'^(Here\'s|Here is|The enhanced prompt is:?|Enhanced prompt:?|Output:?|Result:?|Generated:?)\s*', # Common LLM intros
            r'^(A|An)\s+', # Leading articles
            r'^\W+', # Any leading non-alphanumeric characters (e.g., punctuation, newlines)
            r'\W+$',  # Any trailing non-alphanumeric characters
            r'^\s*,\s*', # Leading comma and space
            r'\s*,\s*$' # Trailing comma and space
        ]

        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()

        # Remove any remaining leading/trailing commas or periods
        text = re.sub(r'^[.,\s]+|[.,\s]+$', '', text).strip()

        # Ensure consistent comma-space separation
        text = re.sub(r'\s*,\s*', ', ', text)

        return text.strip()

    def create_fallback_enhancement(self, prompt, style, creativity_level, prompt_type="positive"):
        """
        Generates a rule-based enhancement when LLM inference is not available
        or fails.
        """
        if prompt_type == "positive":
            # Access class-level attribute directly
            quality_words = self.quality_words.get(style, self.quality_words["photography"])

            # Determine number of words to select based on creativity level
            num_words = min(len(quality_words), max(1, int(creativity_level * len(quality_words))))
            selected_words = random.sample(quality_words, num_words)

            enhancement = ", ".join(selected_words)

            # Add extra details for higher creativity levels
            if creativity_level > 0.7:
                extra_details = {
                    "photography": "8K resolution, professional lighting, cinematic detail",
                    "artistic": "masterpiece, award winning, highly detailed, trending on artstation",
                    "cinematic": "dramatic composition, epic scale, volumetric lighting, film noir",
                    "realistic": "hyperrealistic, ultra detailed, lifelike textures, intricate details"
                }
                enhancement += f", {extra_details.get(style, 'high quality')}"

            return f"{prompt}, {enhancement}".strip()
        else: # Negative prompt fallback
            # Simple fallback for negative prompts
            common_negative_terms = [
                "blurry", "deformed", "bad anatomy", "ugly", "disfigured",
                "low quality", "poorly drawn", "mutated", "extra limbs",
                "text", "signature", "watermark", "cropped", "out of frame"
            ]
            # Select a subset based on creativity_level (more creative = more terms)
            num_terms = min(len(common_negative_terms), max(1, int(creativity_level * len(common_negative_terms) * 0.5)))
            selected_terms = random.sample(common_negative_terms, num_terms)
            return f"{prompt}, {', '.join(selected_terms)}".strip()

