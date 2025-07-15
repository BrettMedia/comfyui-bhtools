# 🎬 BHTools

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ComfyUI Compatible](https://img.shields.io/badge/ComfyUI-Compatible-brightgreen)](https://github.com/comfyanonymous/ComfyUI)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)

This repository contains a collection of advanced custom nodes for ComfyUI, designed to enhance your workflow for cinematic scene generation, intelligent prompt enhancement, and robust image/video saving with integrated memory management.

## 💖 Donations

* **Patreon:** https://www.patreon.com/YourPatreonPage (Temporary Link)
* **Buy Me a Coffee:** https://www.buymeacoffee.com/YourCoffeePage (Temporary Link)

---

## 🚀 Overview

The BHTools offer a suite of powerful nodes to streamline and expand your ComfyUI capabilities, bringing a touch of Hollywood magic to your generations:

* **Cinematic Scene Director:** Generate sophisticated cinematic prompts with extensive controls over camera, lighting, environment, and visual style, optionally enhanced by local LLMs.
* **Prompt Inference:** Professionally enhance your positive and negative prompts using local LLM inference, with options for custom trigger words and system prompt strategies.
* **Save Image/Video:** A versatile node for saving individual images or combining image sequences into various video formats, featuring audio integration, dynamic naming, and workflow metadata embedding.
* **End of Workflow Clearing:** Optimize resource usage by performing comprehensive cleanup of GPU VRAM, system RAM, and temporary files at the end of your workflow.

## Table of Contents

* [⬇️ Installation](#installation)
* [🛠️ Nodes Overview](#nodes-overview)
    * [1. CinematicSceneDirectorNodeBHTools](#1-cinematicscenedirectornodebhtools)
    * [2. PromptInferenceNodeBHTools](#2-promptinferencenodebhtools)
    * [3. SaveImageVideoBHTools](#3-saveimagevideobhtools)
    * [4. EndOfWorkflowClearingNodeBHTools](#4-endofworkflowclearingnodebhtools)
* [🤝 Contributing](#contributing)
* [📄 License](#license)

## ⬇️ Installation

1.  **Navigate to your ComfyUI `custom_nodes` directory:**

    ```bash
    cd ComfyUI/custom_nodes
    ```

2.  **Clone this repository:**

    ```bash
    git clone [https://github.com/your-username/ComfyUI-BH-Tools.git](https://github.com/your-username/ComfyUI-BH-Tools.git)
    ```

3.  **Install dependencies:**
    Some nodes require additional Python libraries. It's recommended to install them in your ComfyUI Python environment.

    ```bash
    pip install transformers torch accelerate bitsandbytes sentencepiece scipy psutil
    ```

    * `transformers`, `torch`, `accelerate`, `bitsandbytes`, `sentencepiece`: Required for LLM-powered prompt enhancement in `CinematicSceneDirectorNodeBHTools.py` and `PromptInferenceNodeBHTools.py`. `bitsandbytes` is optional but highly recommended for VRAM efficiency with LLMs.
    * `scipy`: Required for audio handling in `SaveVideoOrImageNodeBHTools.py`.
    * `psutil`: Required for memory monitoring in `EndOfWorkflowClearingNodeBHTools.py`.

4.  **FFmpeg (for video features):**
    Ensure FFmpeg is installed and accessible in your system's PATH, or place the `ffmpeg` executable within `ComfyUI/ffmpeg/` or `ComfyUI/custom_nodes/ffmpeg/`.

## 🛠️ Nodes Overview

### 1. CinematicSceneDirectorNodeBHTools

**Category:** `BH Tools`

An advanced node for generating sophisticated cinematic prompts. It allows users to cycle through master prompts and apply comprehensive cinematic parameters, including composition, lighting, atmosphere, and post-production effects.

**Key Features:**

* **Master Prompts System:** Define multiple base prompts and cycle through them.
* **Comprehensive Cinematic Controls:** Fine-tune various aspects like shot type, camera angle/movement/lens, aspect ratio, depth of field, composition rules, lighting quality/setup/color, character attributes, visual style, image quality, environment, atmospheric effects, time of day, weather, season, post-processing, lens effects, motion blur, cinematic techniques, genre influence, mood, and animation style.
* **Weighted Parameter System:** Apply weights to individual parameters for nuanced control over their influence on the final prompt.
* **Preset Overrides:** Quickly apply predefined stylistic presets (e.g., "Cinematic Drama," "Golden Hour Magic," "Sci-Fi Futuristic," "Film Noir," "Hentai").
* **Optional LLM-Powered Prompt Enhancement:** Utilize local HuggingFace LLMs (Qwen, Llama, Florence-2) to intelligently enhance or generate prompts based on your inputs, with memory management and creativity level controls.
* **Negative Prompt Presets:** Select from various negative prompt presets to easily filter out undesirable elements.
* **Dynamic UI:** Input fields dynamically appear/hide based on LLM enablement and preset override settings.

**How to Use:**

1.  **Start with `master_prompts`:** Enter a list of base ideas, one per line (e.g., "A futuristic city at sunset," "A lone knight in a mystical forest").
2.  **Select `prompt_index`:** Choose which of your master prompts to use for the current generation.
3.  **Enable LLM (Optional):** If `enable_llm_inference` is checked, select an `llm_model_selection` and `llm_enhancement_style` to have the LLM expand on your base prompt. Experiment with `llm_creativity_level` for varied results.
4.  **Apply Presets:** Choose from `cinematic_preset`, `genre_preset`, etc., to quickly set a mood or style. If `enable_preset_override_button` is checked, only the first selected preset will apply.
5.  **Fine-tune with Adjustable Details:** Use the individual `camera_controls`, `lighting_controls`, etc., to add specific details. Adjust their corresponding `_weight` inputs for more or less influence.
6.  **Set Negative Prompt:** Select a `negative_prompt_preset` to automatically add common undesirable terms.
7.  **Connect Outputs:** The `cinematic_prompt` and `negative_prompt` outputs can be connected directly to your sampler or other prompt processing nodes.

**Inputs:**

* `master_prompts` (STRING, multiline): Base prompts to cycle through.
* `prompt_index` (INT): Selects the current prompt from `master_prompts`.
* **LLM Positive Prompt Enhancement:**
    * `enable_llm_inference` (BOOLEAN): Activates LLM enhancement.
    * `llm_model_selection` (STRING): Choose from available local LLMs (e.g., Qwen2.5-1.5B, Llama-3.2-1B).
    * `llm_enhancement_style` (STRING): Guides LLM enhancement (e.g., "photography", "cinematic").
    * `llm_creativity_level` (FLOAT): Controls LLM output randomness.
    * `llm_max_length` (INT): Maximum length of LLM-generated text.
    * `llm_purge_cache` (BOOLEAN): Unloads LLM from VRAM after use.
    * `llm_regenerate_on_each_run` (BOOLEAN): Generates new LLM prompt each run or reuses last.
    * `incorporate_adjustments_into_llm` (BOOLEAN): Feeds selected presets/adjustments into LLM.
* `enable_preset_override_button` (BOOLEAN): Uses only the first selected preset.
* **Negative Prompt Controls:**
    * `negative_prompt_preset` (STRING): Selects a predefined negative prompt.
* **Categorized Presets:**
    * `cinematic_preset`, `genre_preset`, `visual_style_preset`, `lighting_preset`, `environment_preset`, `character_preset`, `vfx_preset`, `humor_preset`, `horror_preset`, `nsfw_preset`, `hentai_preset`, `domain_specific_preset` (STRING): Dropdowns for various presets.
* **Categorized Adjustable Details:**
    * Various STRING and FLOAT inputs for granular control over camera, lighting, character, visual style, environment, time/weather, post/VFX, and miscellaneous parameters, each with an associated weight.

**Outputs:**

* `cinematic_prompt` (STRING): The final, combined positive prompt.
* `negative_prompt` (STRING): The generated negative prompt.

### 2. PromptInferenceNodeBHTools

**Category:** `BH Tools/Prompting`

A dedicated node for professional prompt enhancement using local LLM inference. It supports custom trigger words, negative prompt enhancement, and customizable system prompts.

**Key Features:**

* **LLM-Powered Enhancement:** Utilizes local HuggingFace LLMs (Qwen, Llama) to expand and refine prompts.
* **Custom Trigger Words:** Prepend specific trigger words or LoRA tags.
* **Negative Prompt Enhancement:** Optionally enhance negative prompts to avoid common imperfections.
* **Customizable System Prompts:** Choose from predefined system prompt strategies or provide a manual override for both positive and negative prompts.
* **Stylistic Control:** Guide the LLM with styles like "photography," "artistic," "cinematic," or "realistic."
* **Creativity and Length Control:** Adjust the randomness and maximum length of LLM-generated output.
* **VRAM Management:** Option to purge LLM from VRAM after use.
* **Reproducibility:** Supports setting a random seed for consistent LLM output.

**How to Use:**

1.  **Input Your Prompt:** Connect your base prompt to the `prompt` input.
2.  **Add Trigger Words (Optional):** Use `trigger_words` for LoRAs or specific keywords (e.g., `lora:myLora:0.7, cinematic`).
3.  **Select LLM:** Choose a `model_selection` or "Fallback [No Model]" if you don't have transformers installed or prefer rule-based enhancement.
4.  **Define Enhancement Style:** Pick an `enhancement_style` (e.g., "cinematic") to guide the LLM's output.
5.  **Customize System Prompt:** Select a `system_prompt_method` or provide your own `system_prompt` override to instruct the LLM on how to enhance.
6.  **Enhance Negative Prompt (Optional):** If you have a `negative_prompt` and check `enhance_negative_prompt`, the LLM will also refine your negative prompt. You can similarly customize its `negative_system_prompt_method`.
7.  **Connect Outputs:** The `enhanced_positive_prompt` and `enhanced_negative_prompt` can be fed into your text encoder or sampler nodes.

**Inputs:**

* `prompt` (STRING, multiline): Your base positive prompt.
* `trigger_words` (STRING, multiline): Additional trigger words or LoRA tags.
* `model_selection` (STRING): Choose from available local LLMs or "Fallback [No Model]".
* `enhancement_style` (STRING): Stylistic direction for LLM enhancement.
* `creativity_level` (FLOAT): Controls LLM creativity.
* `max_length` (INT): Max tokens for LLM generation.
* `system_prompt_method` (STRING): Selects a predefined positive system prompt.
* `system_prompt` (STRING, multiline): Manual override for positive system prompt.
* `negative_prompt` (STRING, multiline, optional): Base negative prompt.
* `enhance_negative_prompt` (BOOLEAN): Enables LLM enhancement for negative prompt.
* `negative_system_prompt_method` (STRING): Selects a predefined negative system prompt.
* `negative_system_prompt` (STRING, multiline, optional): Manual override for negative system prompt.
* `seed` (INT): Random seed for LLM generation (-1 for random).
* `purge_cache` (BOOLEAN): Unloads LLM model after generation.

**Outputs:**

* `enhanced_positive_prompt` (STRING): The enhanced positive prompt.
* `enhanced_negative_prompt` (STRING): The enhanced negative prompt.

### 3. SaveImageVideoBHTools

**Category:** `BH Tools`

A versatile node for saving and previewing generated images and videos. It offers extensive control over output formats, filenames, directories, and video encoding parameters.

**Key Features:**

* **Flexible Saving:** Save individual images (PNG, JPG, WEBP, GIF) or combine image sequences into videos (MP4, WebM, GIF, PNG sequence).
* **Intelligent Preview:** Always displays a preview in ComfyUI's Extra Outputs panel, either as individual images or an animated video.
* **Audio Integration:** Optionally add an audio track to your videos and animated previews (requires `scipy`).
* **Custom Output Paths:** Define subfolders within ComfyUI's output directory.
* **Dynamic Filenaming:** Control filename prefixes, append timestamps for uniqueness, and manage file overwriting.
* **Metadata Embedding:** Automatically embeds workflow metadata into PNGs for easy reloading.
* **Video Encoding Control:** Adjust FPS, CRF (Constant Rate Factor) for quality, loop count, and audio synchronization for video outputs.
* **Workflow Thumbnail Export:** Option to save a separate PNG image with workflow metadata alongside videos.
* **Verbose Logging:** Detailed console output for debugging.
* **VAE Decoding:** Supports decoding latent inputs if a VAE is provided.
* **Dynamic UI:** Input fields dynamically adjust visibility based on whether image or video output is selected.

**How to Use:**

1.  **Connect Images:** Connect the `images` output from your KSampler or image processing nodes.
2.  **Choose Save or Preview:** Set `save_output` to `True` to save files to disk, or `False` to only generate previews.
3.  **Select Output Type:**
    * For **individual images**, ensure `combine_video` is `False`. Choose your `image_file_extension` (e.g., "png", "jpeg") and `jpeg_quality` if applicable.
    * For **videos**, set `combine_video` to `True`. Select your `video_format` (e.g., "video/h264-mp4"), `video_fps`, and `video_crf`.
4.  **Add Audio (Optional):** Connect an `AUDIO` output from an audio processing node if you want sound in your video.
5.  **Customize Filenames:** Use `filename_prefix` to name your files. Enable `append_timestamp` for unique filenames, or disable it and use `overwrite_existing_files` to replace existing files.
6.  **Set Output Directory:** Specify a `output_dir` (e.g., "my_renders/videos") to organize your output.
7.  **Monitor Preview:** The generated images or videos will appear in ComfyUI's Extra Outputs panel. For image batches, `preview_batch` controls whether all images or just the last one are shown in the preview.

**Inputs:**

* `images` (IMAGE): The image or image sequence to save/preview.
* `filename_prefix` (STRING): Prefix for output filenames.
* `save_output` (BOOLEAN): If true, saves files to disk; otherwise, only previews.
* `audio` (AUDIO, optional): Optional audio input for videos.
* `output_dir` (STRING): Subfolder for output files.
* `combine_video` (BOOLEAN): If true, combines images into a video.
* `video_fps` (INT): Frames per second for video/animated previews.
* **Image Specific Options (visible when `combine_video` is False):**
    * `image_file_extension` (STRING): "png", "jpeg", "webp", "gif".
    * `jpeg_quality` (INT): Quality for JPEG/WebP (1-100).
    * `preview_batch` (BOOLEAN): If true, previews all images in batch; else, only the last.
* **Video Specific Options (visible when `combine_video` is True):**
    * `video_format` (STRING): "image/gif", "video/h264-mp4", etc.
    * `video_crf` (INT): CRF for video quality (0-51).
    * `video_loop_count` (INT): Number of times video loops (0 for no loop).
    * `video_sync_audio` (BOOLEAN): Syncs video duration to audio.
    * `export_workflow_image_with_video` (BOOLEAN): Saves workflow PNG with video.
* **General Options:**
    * `append_timestamp` (BOOLEAN): Appends timestamp to filename.
    * `overwrite_existing_files` (BOOLEAN): Overwrites existing files (if no timestamp).
    * `verbose` (BOOLEAN): Enables detailed logging.
    * `vae` (VAE, optional): Optional VAE for latent decoding.
* `prompt` (PROMPT, hidden): Hidden input for workflow metadata.
* `extra_pnginfo` (EXTRA_PNGINFO, hidden): Hidden input for additional PNG metadata.

**Outputs:**

* `filename` (STRING): The filename of the primary saved/previewed output.
* `width` (INT): Width of the output.
* `height` (INT): Height of the output.

### 4. EndOfWorkflowClearingNodeBHTools

**Category:** `BH Tools`

A utility node designed to perform a comprehensive cleanup at the end of your ComfyUI workflow execution. This helps optimize resource usage by freeing up system RAM, GPU VRAM, clearing temporary files, and unloading models.

**Key Features:**

* **Memory Management:** Clears GPU VRAM, PyTorch's internal CUDA cache, and forces Python's garbage collection.
* **Model Unloading:** Attempts to unload ComfyUI models from memory.
* **System Cache Clearing:** Option to clear system-level file caches (Linux/macOS specific).
* **Temporary File Removal:** Clears temporary files created by ComfyUI.
* **Threshold-Based Cleanup:** Optionally trigger cleanup only if RAM/VRAM usage exceeds specified thresholds.
* **Run Once Per Session:** Option to ensure cleanup executes only once per ComfyUI session to prevent redundant operations.
* **Verbose Reporting:** Provides detailed cleanup status in the console.

**How to Use:**

1.  **Connect to Workflow End:** Connect the `trigger` input of this node to the output of the *last* node in your ComfyUI workflow (e.g., a `Save Image` node or another final processing node). This ensures cleanup runs after your main generation is complete.
2.  **Configure Cleanup Actions:** Select which resources to clear:
    * Check `clear_vram` to free up GPU memory.
    * Check `clear_models` to unload ComfyUI models.
    * Check `clear_torch_cache` for PyTorch's internal cache.
    * Check `clear_temp_files` to remove temporary ComfyUI files.
    * Check `force_gc` for Python's garbage collection.
3.  **Set Thresholds (Optional):** If `only_if_above_threshold` is checked, set `vram_threshold_gb` and `ram_threshold_gb` to only trigger cleanup when memory usage is high.
4.  **Run Once:** Keep `only_run_once` checked to prevent repeated cleanup if your workflow loops within the same ComfyUI session.
5.  **Monitor Report:** The `cleanup_report` output will provide a summary of the actions taken in your ComfyUI console.

**Inputs:**

* `trigger` (\*): Connect any node output here to trigger the cleanup.
* `clear_vram` (BOOLEAN): Clear GPU VRAM.
* `clear_models` (BOOLEAN): Unload ComfyUI models.
* `clear_torch_cache` (BOOLEAN): Clear PyTorch's CUDA cache.
* `clear_system_cache` (BOOLEAN): Clear system-level file caches (Linux/macOS).
* `clear_temp_files` (BOOLEAN): Remove temporary files.
* `force_gc` (BOOLEAN): Force Python garbage collection.
* `vram_clear_method` (STRING): "soft", "hard", "complete" for VRAM clearing aggressiveness.
* `only_if_above_threshold` (BOOLEAN): Only clean if memory usage exceeds thresholds.
* `only_run_once` (BOOLEAN): Execute cleanup only once per session.
* `verbose` (BOOLEAN): Enable detailed status reports.
* `vram_threshold_gb` (FLOAT, optional): VRAM usage (GB) to trigger cleanup.
* `ram_threshold_gb` (FLOAT, optional): RAM usage (GB) to trigger cleanup.

**Outputs:**

* `cleanup_report` (STRING): A detailed report of the cleanup actions performed.

## 🤝 Contributing

Feel free to open issues or pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
