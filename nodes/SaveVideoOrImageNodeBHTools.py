# comfyui-bhtools/nodes/SaveVideoOrImageNodeBHTools.py

import os
import torch
from PIL import Image, PngImagePlugin, ExifTags
from PIL.PngImagePlugin import PngInfo # Explicitly import PngInfo
import numpy as np
import json
from datetime import datetime
import re
import itertools
import functools
import subprocess
import sys
import copy
import logging
import shutil # For shutil.which
import tempfile # For temporary file creation

# Attempt to import scipy for WAV writing, otherwise warn
try:
    from scipy.io import wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("\n--- WARNING: 'scipy' library not found. Audio saving functionality may be limited. ---\n")
    print("--- Please install it with: pip install scipy ---\n")


# This code was generated with the help of a large language model (LLM)
# for "vibe coding" and is intended for open-source use.\
# Feel free to modify and distribute it under appropriate open-source licenses.\

# Assume these are available from ComfyUI environment
import folder_paths
from comfy.utils import ProgressBar

# --- Custom logging for BH Tools ---
class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[0;36m",  # CYAN
        "INFO": "\033[0;32m",  # GREEN
        "WARNING": "\033[0;33m",  # YELLOW
        "ERROR": "\033[0;31m",  # RED
        "CRITICAL": "\033[0;37;41m",  # WHITE ON RED
        "RESET": "\033[0m",  # RESET COLOR
    }

    def format(self, record):
        log_message = super().format(record)
        return f"{self.COLORS.get(record.levelname, self.COLORS['RESET'])}{log_message}{self.COLORS['RESET']}"

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Default to INFO, change to DEBUG for more verbosity

# Check if handlers already exist to prevent duplicate output
if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(ColoredFormatter("%(levelname)s: %(message)s"))
    logger.addHandler(ch)

# --- Helper functions (re-implemented for self-contained node) ---

def get_ffmpeg_path():
    """Attempts to find the ffmpeg executable."""
    # Check if ffmpeg is in PATH
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    
    # Check common ComfyUI custom_nodes/ffmpeg directory
    ffmpeg_path_candidates = [
        os.path.join(folder_paths.base_path, "ffmpeg", "ffmpeg"),
        os.path.join(folder_paths.base_path, "ffmpeg", "ffmpeg.exe"),
        os.path.join(folder_paths.base_path, "ComfyUI", "ffmpeg", "ffmpeg"), # For portable installs
        os.path.join(folder_paths.base_path, "ComfyUI", "ffmpeg", "ffmpeg.exe"),
        os.path.join(folder_paths.base_path, "custom_nodes", "ffmpeg", "ffmpeg"),
        os.path.join(folder_paths.base_path, "custom_nodes", "ffmpeg", "ffmpeg.exe"),
    ]
    
    for path in ffmpeg_path_candidates:
        if os.path.exists(path):
            return path
            
    logger.warning("FFmpeg not found in system PATH or common ComfyUI locations. Video generation may fail.")
    return None

FFMPEG_PATH = get_ffmpeg_path()

def _get_media_duration(file_path, verbose=False):
    """
    Uses ffprobe to get the duration of a media file.
    Returns duration in seconds (float) or None if an error occurs.
    """
    ffprobe_path = shutil.which("ffprobe")
    if not ffprobe_path:
        logger.error("FFprobe executable not found. Please ensure FFmpeg (and thus ffprobe) is installed and accessible in your system's PATH, or placed alongside ffmpeg.")
        return None
    
    # Construct ffprobe command
    command = [
        ffprobe_path,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        if verbose:
            logger.debug(f"BH Tools Debug: Duration of '{os.path.basename(file_path)}': {duration:.2f} seconds")
        return duration
    except FileNotFoundError:
        logger.error("FFprobe executable not found. This should not happen if shutil.which('ffprobe') succeeded.")
        return None
    except subprocess.CalledProcessError as e:
        logger.error(f"BH Tools ERROR: FFprobe failed for '{os.path.basename(file_path)}': {e.stderr.strip()}")
        return None
    except ValueError:
        logger.error(f"BH Tools ERROR: Could not parse duration from ffprobe output for '{os.path.basename(file_path)}'. Output: '{result.stdout.strip()}'")
        return None
    except Exception as e:
        logger.error(f"BH Tools ERROR: An unexpected error occurred while getting duration for '{os.path.basename(file_path)}': {e}")
        return None


def _save_audio_to_temp_wav(audio_tensor, sample_rate, verbose=False):
    """
    Saves an audio tensor to a temporary WAV file.
    Audio tensor is expected to be [batch_size, num_samples] or [num_samples].
    Returns (temp_audio_path, channels) or (None, None) if an error occurs.
    """
    if not SCIPY_AVAILABLE:
        logger.error("SciPy not available. Cannot save audio to WAV for FFmpeg.")
        return None, None

    if audio_tensor is None:
        return None, None

    # Ensure audio_tensor is on CPU
    audio_np = audio_tensor.cpu().numpy()

    # Determine channels based on shape
    channels = 1
    if audio_np.ndim == 2:
        # Assuming ComfyUI's AUDIO output is [channels, num_samples] or [batch_size, num_samples]
        # If the first dimension is small (e.g., 1 or 2 for channels) and second is large (samples)
        if audio_np.shape[0] < 3 and audio_np.shape[1] > 100: # Heuristic for [channels, samples]
            channels = audio_np.shape[0]
            audio_np = audio_np.T # Transpose to (num_samples, num_channels)
        elif audio_np.shape[0] == 1: # Mono audio in batch dim or single channel
            audio_np = audio_np[0] # Remove the single channel/batch dim
            channels = 1
        else: # Fallback, assume mono if ambiguous, or if it's already (samples,)
            channels = 1
            if audio_np.ndim == 2: # If still 2D, take first row
                audio_np = audio_np[0]


    # Normalize to int16 range for WAV
    if audio_np.dtype != np.int16:
        audio_np = (audio_np * 32767).astype(np.int16)

    temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=folder_paths.get_temp_directory())
    temp_audio_path = temp_audio_file.name
    temp_audio_file.close()

    try:
        wavfile.write(temp_audio_path, sample_rate, audio_np)
        if verbose:
            logger.debug(f"BH Tools Debug: Saved temporary audio to: {temp_audio_path} with SR={sample_rate}, Channels={channels}")
        return temp_audio_path, channels
    except Exception as e:
        logger.error(f"BH Tools ERROR: Failed to save temporary audio file: {e}")
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        return None, None

# New helper function to check if a path is absolute and outside ComfyUI's output directory
def _is_absolute_path_outside_comfyui_output(path_str):
    """
    Checks if the given path string is an absolute path and if it's outside
    ComfyUI's default output directory.
    """
    if not os.path.isabs(path_str):
        return False # Not an absolute path

    # Normalize paths to handle different separators and redundancies
    normalized_path = os.path.normpath(path_str)
    normalized_comfyui_output_dir = os.path.normpath(folder_paths.get_output_directory())

    # Check if the normalized path starts with the normalized ComfyUI output directory
    # This implies it's either inside or is the output directory itself
    return not normalized_path.startswith(normalized_comfyui_output_dir)


# --- Main Node Class ---
class SaveImageVideoBHTools:
    """
    BH Tools - Advanced Image and Video Saving Node for ComfyUI.
    
    Provides comprehensive options for saving images, generating videos from image sequences,
    and creating video previews, with detailed control over filenames, metadata,
    and video encoding parameters.
    
    Features:
    - Save individual images (PNG, JPG, WEBP, GIF)
    - Generate MP4/WebM/GIF videos from image sequences
    - Optional audio track for videos (direct AUDIO input)
    - Always-on intelligent video/image preview (displayed in ComfyUI's Extra Outputs panel)
    - Custom filename prefixes and output directories (supports absolute paths for saving, with temporary previews)
    - PNG metadata saving (prompt, workflow)
    - Automatic subfolder creation
    - Verbose logging for debugging
    """
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        # Initialize preview_results here. It will persist across runs if the node instance is reused.
        self.preview_results = [] 
        self.ffmpeg_path = FFMPEG_PATH

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "save_output": ("BOOLEAN", {"default": True, "tooltip": "If true, saves images/videos to disk. If false, only processes and previews."}),
            },
            "optional": {
                "audio": ("AUDIO", {"optional": True, "tooltip": "Optional audio to add to the main video and animated previews. Connect an audio output here."}), # Unified audio input - MOVED TO OPTIONAL
                "output_dir": ("STRING", {
                    "default": "BH_Tools", # Changed default output directory
                    "placeholder": "output or custom_folder/subfolder or C:/absolute/path",
                    "tooltip": "Subfolder within ComfyUI's output directory, or an absolute path. Absolute paths will save externally, but previews will be temporary copies."
                }),
                "combine_video": ("BOOLEAN", {"default": False, "tooltip": "If true, combine images into a video. The output format is determined by the 'video_format' input."}),
                "video_fps": ("INT", {"default": 24, "min": 1, "max": 60, "step": 1, "tooltip": "Frames per second for the output video and animated previews."}),
                
                # --- Image Specific Options (visible when combine_video is False) ---
                "image_file_extension": (["png", "jpeg", "webp", "gif"], {"default": "png", "tooltip": "File extension for individual image saves."}), # Correct tuple format
                "jpeg_quality": ("INT", { # Only applies to JPEG/WebP image_file_extension
                    "default": 90, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Quality setting for JPEG and WebP images (1-100). Only applicable to .jpg, .jpeg, and .webp files."
                }),
                "preview_batch": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If checked, all images in the batch will be kept in the preview. If unchecked, only the last image will be shown."
                }),

                # --- Video Specific Options (visible when combine_video is True) ---
                "video_format": ([ # Correct tuple format
                    "image/gif", # Animated GIF via FFmpeg/PIL
                    "image/webp", # Animated WebP via PIL
                    "video/16bit-png", # PNG sequence
                    "video/8bit-png",  # PNG sequence
                    "video/av1-webm",
                    "video/ffmpeg-gif",
                    "video/h264-mp4",
                    "video/h265-mp4",
                    "video/nvenc_av1-mp4",
                    "video/nvenc_h264-mp4",
                    "video/nvenc_hevc-mp4",
                    "video/ProRes",
                    "video/webm"
                ], {"default": "video/h264-mp4", "tooltip": "Select the video or animated image output format. Requires FFmpeg for most video formats."}),
                "video_crf": ("INT", {"default": 23, "min": 0, "max": 51, "step": 1, "tooltip": "Constant Rate Factor (CRF) for video quality. Lower is higher quality (larger file). 0 is lossless. Not applicable for GIF."}),
                # "video_preset": (["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"], {"default": "veryslow", "tooltip": "Encoding preset for video. Faster presets mean larger files/lower quality. 'veryslow' for maximum quality. Not applicable for GIF."}), # REMOVED
                "video_loop_count": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "tooltip": "Number of times the video should loop. 0 for no loop (single play). Only applies if audio is shorter than video."}),
                "video_sync_audio": ("BOOLEAN", {"default": False, "tooltip": "If true, video duration will match audio duration. If audio is longer, video will cut short or loop to match."}),
                "export_workflow_image_with_video": ("BOOLEAN", {"default": True, "tooltip": "If true, a separate PNG image with workflow metadata will be saved alongside the video. This only applies when combining images into a video."}), # New option with clarified tooltip
                
                # --- General Options (always visible) ---
                "append_timestamp": ("BOOLEAN", {"default": True, "tooltip": "If true, appends date and time to the filename for uniqueness."}), # New option
                "overwrite_existing_files": ("BOOLEAN", {"default": False, "tooltip": "If true, overwrites existing files with the same name instead of creating new ones. Only effective if 'append_timestamp' is false."}), # New option
                "verbose": ("BOOLEAN", {"default": False, "tooltip": "Enable verbose logging for debugging. Messages will appear in the ComfyUI console."}), # Re-added verbose input
                "vae": ("VAE", {"optional": True, "tooltip": "Optional VAE for latent input. If connected, 'images' input is treated as latents and decoded before saving."}),
            },
            "hidden": {
                "prompt": "PROMPT", # Hidden input for workflow metadata
                "extra_pnginfo": "EXTRA_PNGINFO", # Hidden input for additional PNG metadata
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT") # Added "STRING" for filename
    RETURN_NAMES = ("filename", "width", "height") # Added "filename"
    FUNCTION = "save_content_bh"
    CATEGORY = "BH Tools"
    DESCRIPTION = "A powerful node to save or preview generated images or videos. It allows saving to any specified absolute or relative path. Images/Videos are always previewed in the ComfyUI output panel. Offers flexible naming with optional date-based timestamps and prefixes, automatically handles overwriting to prevent loss, applies quality settings to JPEG/WebP, and always embeds workflow data into PNGs for easy reloading. Supports various image and video formats with relevant encoding options. Outputs the dimensions of the last processed image."
    OUTPUT_NODE = True

    @classmethod
    def get_input_property(cls, values, property_name):
        """
        Dynamically controls the visibility of input widgets based on other input values.
        """
        combine_video = values.get("combine_video")
        image_file_extension = values.get("image_file_extension")
        append_timestamp = values.get("append_timestamp") # Get value for conditional visibility

        # Common Image/Video options (always visible)
        if property_name in ["save_output", "output_dir", "filename_prefix", 
                            "video_fps", "audio", "verbose", "vae", "append_timestamp", "combine_video"]:
            return {"hidden": False}
        
        # Overwrite option visibility: only visible if append_timestamp is false
        if property_name == "overwrite_existing_files":
            return {"hidden": append_timestamp}

        # Image-specific options: visible only if combine_video is False
        image_specific_inputs = ["image_file_extension", "jpeg_quality", "preview_batch"]
        if property_name in image_specific_inputs:
            return {"hidden": combine_video}
        
        # Conditional visibility for jpeg_quality
        if property_name == "jpeg_quality":
            if not combine_video and image_file_extension in ["jpeg", "webp"]:
                return {"hidden": False}
            return {"hidden": True}

        # Video-specific options: visible only if combine_video is True
        video_specific_inputs = [
            "video_format", "video_crf", "video_loop_count", "video_sync_audio", "export_workflow_image_with_video"
        ]
        if property_name in video_specific_inputs:
            return {"hidden": not combine_video}

        # Default to not hidden if not explicitly handled above
        return {"hidden": False}


    def get_full_save_path(self, user_output_dir_input, filename, extension):
        """
        Constructs the full output path for a file, handling absolute vs. relative paths.
        Returns (full_filepath, is_external_absolute_path).
        """
        is_external_absolute_path = _is_absolute_path_outside_comfyui_output(user_output_dir_input)
        
        if is_external_absolute_path:
            # If it's an absolute path outside ComfyUI's output, use it directly
            base_folder = user_output_dir_input
            logger.debug(f"BH Tools Debug: Detected external absolute output path: {base_folder}")
        else:
            # Otherwise, treat it as a subfolder relative to ComfyUI's output directory
            base_folder = os.path.join(folder_paths.get_output_directory(), user_output_dir_input)
            logger.debug(f"BH Tools Debug: Detected relative output path: {base_folder}")

        os.makedirs(base_folder, exist_ok=True)
        
        full_filepath = os.path.join(base_folder, f"{filename}.{extension}")
        
        return full_filepath, is_external_absolute_path

    def tensor_to_pil(self, tensor):
        """Convert a tensor to PIL Image, handling ComfyUI's typical IMAGE formats (BHWC or BCHW)."""
        if isinstance(tensor, torch.Tensor):
            # Handle potential batch dimension (ComfyUI IMAGE is typically BHWC)
            if tensor.ndim == 4:
                # Check if it's BCHW and permute to BHWC
                # Assuming channels are 3 (RGB) or 4 (RGBA)
                if tensor.shape[1] in [3, 4] and tensor.shape[2] > 4: # Heuristic: if channels is 3/4 and height is much larger
                    tensor = tensor.permute(0, 2, 3, 1) # BCHW to BHWC
                tensor = tensor[0] # Remove batch dimension
            
            # Convert from tensor to numpy array
            image_np = tensor.cpu().numpy()

            # Normalize to 0-255 and convert to uint8 if float
            if image_np.dtype == np.float32 or image_np.dtype == np.float64:
                image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
            
            return Image.fromarray(image_np)
        return tensor

    def generate_video(self, image_paths, output_path_base, output_extension, fps, codec, crf, audio_src_path=None, audio_sample_rate=None, audio_channels=None, loop_count=0, sync_audio=False, verbose=False):
        """Generates a video from a list of image paths using image2 demuxer."""
        if not self.ffmpeg_path:
            logger.error("FFmpeg not found. Cannot generate video.")
            return False, None # Return False and None for path

        if not image_paths:
            logger.warning("No images provided for video generation.")
            return False, None

        # Get the first image to determine resolution
        try:
            with Image.open(image_paths[0]) as img:
                width, height = img.size
        except Exception as e:
            logger.error(f"Failed to read first image for video resolution: {e}")
            return False, None

        output_path = f"{output_path_base}.{output_extension}"
        temp_dir = os.path.dirname(image_paths[0]) # Directory where temporary images are saved

        # Use image2 demuxer for sequential image input
        # Assumes images are sequentially numbered (e.g., 00000.png, 00001.png)
        # and are all in the same directory.
        first_image_filename = os.path.basename(image_paths[0])
        image_pattern = "%05d." + first_image_filename.split('.')[-1] # e.g., %05d.png

        command = [
            self.ffmpeg_path,
            "-y",  # Overwrite output files without asking
            "-framerate", str(fps), # Input framerate
        ]
        
        # Calculate video duration from image count and fps
        video_duration = len(image_paths) / fps
        audio_duration = 0 # Initialize audio duration

        # Add audio input if provided
        if audio_src_path and os.path.exists(audio_src_path):
            audio_duration = _get_media_duration(audio_src_path, verbose=verbose)
            if audio_duration is None:
                logger.warning("BH Tools Warning: Could not determine audio duration. Proceeding without specific audio sync logic.")
                audio_duration = 0 # Treat as no audio for duration comparison

            logger.debug(f"BH Tools Debug: Video Duration: {video_duration:.2f}s, Audio Duration: {audio_duration:.2f}s (from ffprobe if available), Using SR={audio_sample_rate}, Channels={audio_channels} from input.")

            # Add video input (images)
            command.extend(["-i", os.path.join(temp_dir, image_pattern)]) # Video input

            # Add audio input
            command.extend(["-i", audio_src_path])
            
            # Map streams explicitly: video from first input (images), audio from second input (audio file)
            command.extend(["-map", "0:v", "-map", "1:a"])
            
            # Explicitly set audio sample rate and channels for the output
            if audio_sample_rate is not None and audio_channels is not None:
                 command.extend(["-ar", str(audio_sample_rate), "-ac", str(audio_channels)])

            # Handle audio sync and looping
            if sync_audio:
                # If sync_audio is True, video is the master duration.
                # If audio is longer than video, it will be trimmed by -shortest.
                # If audio is shorter than video, it will play once and then silence (video continues).
                command.extend(["-shortest"])  
            else:
                # Audio is not synced to video duration. Audio loops if loop_count > 0.
                if loop_count > 0:
                    command.extend(["-stream_loop", str(loop_count)]) # Apply loop to audio stream
            
            # Audio codec
            command.extend(["-c:a", "aac", "-b:a", "128k"])
        else:
            # No audio, only map video stream
            command.extend(["-i", os.path.join(temp_dir, image_pattern)]) # Video input
            command.extend(["-map", "0:v"])

        # Video codec and settings
        command.extend(["-c:v", codec])
        
        # Add codec-specific options
        if codec == "gif":
            # For GIF, use specific options for palette generation and looping
            command.extend([
                "-vf", f"fps={fps},scale={width}:{height}:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
                "-loop", "0"  # Infinite loop for GIF
            ])
        elif codec == "png":
            # For PNG sequence, no additional options needed (raw image output)
            pass
        else:
            # For other video codecs (h264, webm, etc.), use CRF, preset, and pixel format
            command.extend([
                "-crf", str(crf),
                "-preset", "veryslow", # Using 'veryslow' for max quality, can be changed for speed
                "-pix_fmt", "yuv420p", # Standard pixel format for wide compatibility
                "-vf", f"scale={width}:{height}" # Ensure output resolution matches input images
            ])

        command.append(output_path)
        
        if verbose:
            logger.debug(f"BH Tools Debug: FFmpeg command: {' '.join(command)}")

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            if verbose:
                logger.debug(f"FFmpeg stdout:\n{result.stdout}")
                logger.debug(f"FFmpeg stderr:\n{result.stderr}")
            logger.info(f"Video successfully generated: {output_path}")
            
            return True, output_path # Return True and the path
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg video generation failed: {e}")
            logger.error(f"FFmpeg stdout:\n{e.stdout}")
            logger.error(f"FFmpeg stderr:\n{e.stderr}")
            return False, None
        except FileNotFoundError:
            logger.error("FFmpeg executable not found. Please ensure FFmpeg is installed and accessible in your system's PATH, or placed in 'ComfyUI/ffmpeg/' or 'ComfyUI/custom_nodes/ffmpeg/'.")
            return False, None
        except Exception as e:
            logger.error(f"An unexpected error occurred during video generation: {e}")
            return False, None

    def save_images(self, images, filename_prefix, user_output_dir_input, output_extension, prompt=None, extra_pnginfo=None, verbose=False, pbar=None, save_only_first_image=False, jpeg_quality=90, append_timestamp=True, overwrite_existing_files=False):
        """
        Saves a list of images to the specified output directory.
        If save_only_first_image is True, only the first image will be saved.
        Handles saving to external absolute paths and creating temporary previews.
        Returns a list of dictionaries for preview_results.
        """
        preview_info_list = []
        
        # Prepare metadata for PNG (always embed workflow metadata)
        info = PngInfo()
        if prompt is not None:
            info.add_text("prompt", json.dumps(prompt))
        if extra_pnginfo is not None:
            for k, v in extra_pnginfo.items():
                info.add_text(k, json.dumps(v))

        images_to_process = images
        if save_only_first_image and len(images) > 0:
            images_to_process = [images[0]]
        elif save_only_first_image and len(images) == 0:
            logger.warning("BH Tools Warning: save_only_first_image is True but no images provided. Skipping image save.")
            return []

        for i, image in enumerate(images_to_process):
            # Generate the filename part (without directory)
            filename_part = self._generate_output_filename(
                filename_prefix, i, output_extension, append_timestamp, overwrite_existing_files,
                is_single_image_mode=save_only_first_image
            )
            
            # Get the full save path and check if it's an external absolute path
            full_save_filepath, is_external_absolute_path = self.get_full_save_path(
                user_output_dir_input, filename_part, output_extension
            )
            
            # Convert tensor to PIL Image
            image_pil = self.tensor_to_pil(image)

            try:
                # Save the image to the intended location (absolute or relative)
                if output_extension == "png":
                    image_pil.save(full_save_filepath, pnginfo=info)
                elif output_extension == "jpg":
                    if image_pil.mode != "RGB":
                        image_pil = image_pil.convert("RGB")
                    image_pil.save(full_save_filepath, quality=jpeg_quality, optimize=True)
                elif output_extension == "webp":
                    if image_pil.mode not in ['RGBA', 'RGB']:
                        image_pil = image_pil.convert('RGBA')
                    image_pil.save(full_save_filepath, quality=jpeg_quality, method=6, lossless=False)
                elif output_extension == "gif":
                    image_pil.save(full_save_filepath, save_all=True, append_images=[], duration=100, loop=0)
                
                logger.debug(f"BH Tools Debug: Saved image to: {full_save_filepath}")
                
                # --- Prepare for UI Preview ---
                preview_filename = os.path.basename(full_save_filepath)
                preview_subfolder = ""
                preview_type = "output" # Default to output type

                if is_external_absolute_path:
                    # If saving to an external absolute path, create a temporary copy for preview
                    temp_preview_dir = os.path.join(folder_paths.get_temp_directory(), "bh_tools_preview")
                    os.makedirs(temp_preview_dir, exist_ok=True)
                    temp_preview_filepath = os.path.join(temp_preview_dir, preview_filename)
                    
                    shutil.copy2(full_save_filepath, temp_preview_filepath) # Copy with metadata
                    logger.debug(f"BH Tools Debug: Created temporary preview copy for external save: {temp_preview_filepath}")
                    
                    preview_subfolder = os.path.relpath(temp_preview_dir, folder_paths.get_temp_directory())
                    preview_type = "temp"
                else:
                    # If saving to a relative path within ComfyUI's output, use its relative path
                    preview_subfolder = os.path.relpath(os.path.dirname(full_save_filepath), folder_paths.get_output_directory())

                preview_info_list.append({
                    "filename": preview_filename,
                    "subfolder": preview_subfolder,
                    "type": preview_type,
                    "format": f"image/{output_extension}",
                    "width": image_pil.width,
                    "height": image_pil.height
                })
                
                if pbar:
                    pbar.update(1)

            except Exception as e:
                logger.error(f"BH Tools ERROR: Failed to save or preview image {full_save_filepath}: {e}")
                import traceback
                traceback.print_exc()

        return preview_info_list


    def _generate_output_filename(self, filename_prefix, index, output_extension, append_timestamp, overwrite_existing_files, is_single_image_mode=False):
        """
        Generates a unique or overwrite-friendly filename part (without directory).
        
        Args:
            filename_prefix (str): The user-defined prefix for the filename.
            index (int): The index of the current image in a batch.
            output_extension (str): The desired file extension (e.g., "png", "mp4").
            append_timestamp (bool): Whether to append a timestamp to the filename.
            overwrite_existing_files (bool): Whether to overwrite existing files (only effective if no timestamp).
            is_single_image_mode (bool): True if only one image is being processed (e.g., for workflow thumbnail or single image save).

        Returns:
            str: The generated filename part (e.g., "ComfyUI_2024-07-15_140000_00000").
        """
        safe_prefix = re.sub(r'[^\w_.-]', '', filename_prefix).rstrip('.')
        
        parts = [safe_prefix]
        
        # Special case: Overwriting a single file without timestamp
        # In this scenario, the filename is simply the prefix, and index is ignored.
        if is_single_image_mode and overwrite_existing_files and not append_timestamp:
            return f"{safe_prefix}" # Return just the prefix, extension added by caller
        
        # Append timestamp if enabled
        if append_timestamp:
            # Simplified timestamp format: YYYY-MM-DD_HHMMSS
            parts.append(datetime.now().strftime('%Y-%m-%d_%H%M%S'))
        
        # Always append index for uniqueness, unless it's the specific single-image-overwrite case
        # For batches, or if timestamp is appended, index ensures uniqueness
        # If it's a single image mode, and not the overwrite case, still add index for uniqueness (e.g., workflow thumb)
        if not (is_single_image_mode and overwrite_existing_files and not append_timestamp):
            parts.append(f"{index:05d}")
        
        return '_'.join(parts) # Return filename part, extension added by caller


    def get_output_details_from_format(self, selected_format):
        """
        Maps user-friendly format selection to internal parameters.
        Returns (is_video_output, ffmpeg_codec, output_extension).
        """
        if selected_format.startswith("video/"):
            # Video formats
            codec_map = {
                "video/h264-mp4": ("libx264", "mp4"),
                "video/h265-mp4": ("libx265", "mp4"),
                "video/av1-webm": ("libaom-av1", "webm"),
                "video/ffmpeg-gif": ("gif", "gif"), # FFmpeg's GIF encoder
                "video/nvenc_av1-mp4": ("av1_nvenc", "mp4"),
                "video/nvenc_h264-mp4": ("h264_nvenc", "mp4"),
                "video/nvenc_hevc-mp4": ("hevc_nvenc", "mp4"),
                "video/ProRes": ("prores_ks", "mov"),
                "video/webm": ("libvpx-vp9", "webm"),
                "video/16bit-png": ("png", "png"), # Special case: outputs PNG sequence, handled as video combine
                "video/8bit-png": ("png", "png"), # Special case: outputs PNG sequence, handled as video combine
            }
            codec, ext = codec_map.get(selected_format, ("libx264", "mp4"))
            return (True, codec, ext)
        elif selected_format.startswith("image/"):
            # Animated image formats handled by video logic
            ext = selected_format.split('/')[1]
            if ext == "jpeg": # map to jpg for file extension
                ext = "jpg"
            # If it's an image format that can be animated (gif, webp), treat as video for internal processing
            # This allows generate_video to handle animated image formats if combine_video is true
            if ext in ["gif", "webp"]:
                return (True, None, ext) 
            return (False, None, ext) # Standard static image format
        
        # Fallback for unknown formats (should not happen with proper dropdown)
        logger.warning(f"Unknown format selected: {selected_format}. Defaulting to image/png.")
        return (False, None, "png")


    def save_content_bh(self, images, audio=None, filename_prefix="ComfyUI", save_output=True,
                        output_dir="BH_Tools", combine_video=False, video_fps=24, 
                        image_file_extension="png", jpeg_quality=90, preview_batch=False,
                        video_format="video/h264-mp4", video_crf=23,
                        video_loop_count=0, video_sync_audio=False, export_workflow_image_with_video=True, 
                        append_timestamp=True, overwrite_existing_files=False,
                        verbose=False, vae=None,
                        prompt=None, extra_pnginfo=None):

        # Set logger level based on verbose input
        if verbose:
            logger.setLevel(logging.DEBUG)
            logger.debug("BH Tools Debug: Starting save_content_bh operation.")
            if torch.cuda.is_available():
                initial_vram = torch.cuda.memory_allocated() / 1024**3
                logger.debug(f"BH Tools Debug: Initial VRAM usage: {initial_vram:.2f} GB")
        else:
            logger.setLevel(logging.INFO)

        # Handle empty image input early
        if not isinstance(images, torch.Tensor) or images.numel() == 0:
            logger.error("BH Tools ERROR: No images provided for processing. Returning empty results.")
            # If no images, ensure preview_results is empty to avoid showing old previews
            self.preview_results = []
            return {"ui": {"images": []}, "result": ("", 0, 0)}

        # Determine dimensions for return values
        # After tensor_to_pil, the tensor will be HWC, so shape[1] is H, shape[2] is W
        height, width = images.shape[1], images.shape[2]
        logger.debug(f"BH Tools Debug: Image dimensions: Width={width}, Height={height}")
        logger.debug(f"BH Tools Debug: preview_batch setting: {preview_batch}") # Debug log for preview_batch

        # Handle VAE decoding if provided
        if vae is not None:
            logger.debug("BH Tools Debug: VAE provided, decoding latents to images.")
            try:
                images = vae.decode(images)
            except Exception as e:
                logger.error(f"BH Tools ERROR: Failed to decode latents with VAE: {e}")
                self.preview_results = [] # Clear on VAE error too
                return {"ui": {"images": []}, "result": ("", 0, 0)}

        pbar = ProgressBar(len(images))
        
        # Determine output details based on combine_video and selected formats
        final_output_is_video = False
        ffmpeg_codec = None
        output_ext = None
        
        if combine_video:
            final_output_is_video, ffmpeg_codec, output_ext = self.get_output_details_from_format(video_format)
        else:
            # User wants individual images
            final_output_is_video = False
            ffmpeg_codec = None
            output_ext = image_file_extension

        # Determine whether to combine video
        _effective_combine_video = final_output_is_video and combine_video and len(images) > 1

        # Determine if we should accumulate results or reset for this run
        # Accumulate only if preview_batch is True AND we are NOT combining into a single video
        should_accumulate = preview_batch and not _effective_combine_video

        if not should_accumulate:
            self.preview_results = []
        # else: if should_accumulate is True, self.preview_results retains its state from previous calls

        main_video_temp_audio_path = None
        main_video_audio_channels = None # New variable
        output_filename_to_return = "" # Initialize here

        try:
            # Check if FFmpeg is available if video combining is intended
            if _effective_combine_video and not FFMPEG_PATH:
                logger.critical("BH Tools ERROR: FFmpeg is required for video generation but was not found. Falling back to image save.")
                _effective_combine_video = False
                output_ext = image_file_extension # Use image extension for fallback
                # Ensure only first image is saved on fallback if it was intended to be a video
                _save_only_first_image_for_save = True 


            # Handle audio input
            is_valid_audio_input = False
            audio_tensor = None
            sample_rate = None

            # Debugging: Print raw audio input to understand its type and value
            logger.debug(f"BH Tools Debug: Raw audio input type: {type(audio)}, value: {audio}")

            if isinstance(audio, (list, tuple)) and len(audio) == 2 and isinstance(audio[0], torch.Tensor) and isinstance(audio[1], int):
                audio_tensor, sample_rate = audio
                is_valid_audio_input = True
                logger.debug(f"BH Tools Debug: Valid audio input (tensor, sample_rate) received with sample rate: {sample_rate}")
            elif isinstance(audio, dict) and 'waveform' in audio and 'sample_rate' in audio and isinstance(audio['waveform'], torch.Tensor) and isinstance(audio['sample_rate'], int):
                # Handle audio from custom nodes that might output a dict with 'waveform' and 'sample_rate'
                audio_tensor = audio['waveform']
                sample_rate = audio['sample_rate']
                is_valid_audio_input = True
                logger.debug(f"BH Tools Debug: Valid audio input (dict with waveform/sample_rate) received with sample rate: {sample_rate}")
            elif audio is None or (isinstance(audio, dict) and not audio): # Handles None or empty dict {}
                logger.debug("BH Tools Debug: Audio input is None or an empty dict. No audio will be processed.")
                is_valid_audio_input = False
            else: # Catches anything else that is not the expected format
                logger.warning(f"BH Tools Warning: Audio input provided but not in expected (tensor, sample_rate) or (dict with waveform/sample_rate) format. Received type: {type(audio)}. Skipping audio processing.")
                is_valid_audio_input = False

            if is_valid_audio_input:
                main_video_temp_audio_path, main_video_audio_channels = _save_audio_to_temp_wav(audio_tensor, sample_rate, verbose=verbose)
                if main_video_temp_audio_path is None:
                    logger.warning("BH Tools Warning: Failed to prepare audio. Video will be generated without audio.")

            if save_output:
                # Check if the user's output_dir is an external absolute path
                is_external_absolute_save_path = _is_absolute_path_outside_comfyui_output(output_dir)

                if _effective_combine_video:
                    # Generate video
                    main_video_filename_part = self._generate_output_filename(
                        filename_prefix, 0, output_ext, append_timestamp, overwrite_existing_files,
                        is_single_image_mode=True # This ensures the main video output adheres to overwrite logic
                    )
                    
                    # Get the full path where the video will be saved
                    full_video_save_filepath, _ = self.get_full_save_path(
                        output_dir, main_video_filename_part, output_ext
                    )
                    
                    # Save all images to a temporary folder for video generation
                    temp_dir = os.path.join(folder_paths.get_temp_directory(), f"temp_video_images_{datetime.now().strftime('%Y%m%d%H%M%S')}")
                    os.makedirs(temp_dir, exist_ok=True)
                    logger.debug(f"BH Tools Debug: Created temporary directory for video images: {temp_dir}")
                    
                    temp_image_paths = []
                    for i, image in enumerate(images):
                        temp_filename = os.path.join(temp_dir, f"{i:05d}.png") # Always save as PNG for temp
                        image_pil = self.tensor_to_pil(image) # Use helper function
                        image_pil.save(temp_filename)
                        temp_image_paths.append(temp_filename)
                        if verbose:
                            logger.debug(f"BH Tools Debug: Saved temp image for video: {temp_filename}")
                        pbar.update(1)

                    # Generate the main video
                    video_generated, generated_video_path = self.generate_video(
                        temp_image_paths, os.path.splitext(full_video_save_filepath)[0], output_ext, video_fps, ffmpeg_codec, video_crf,
                        audio_src_path=main_video_temp_audio_path, audio_sample_rate=sample_rate, audio_channels=main_video_audio_channels,
                        loop_count=video_loop_count, sync_audio=video_sync_audio, verbose=verbose
                    )

                    if video_generated and generated_video_path:
                        logger.info(f"BH Tools: Successfully generated video: {generated_video_path}")
                        output_filename_to_return = os.path.basename(generated_video_path)
                        
                        # --- Prepare for UI Preview ---
                        preview_filename = os.path.basename(generated_video_path)
                        preview_subfolder = ""
                        preview_type = "output" # Default to output type

                        if is_external_absolute_save_path:
                            # If saving to an external absolute path, create a temporary copy for preview
                            temp_preview_dir = os.path.join(folder_paths.get_temp_directory(), "bh_tools_preview")
                            os.makedirs(temp_preview_dir, exist_ok=True)
                            temp_preview_filepath = os.path.join(temp_preview_dir, preview_filename)
                            
                            shutil.copy2(generated_video_path, temp_preview_filepath) # Copy with metadata
                            logger.debug(f"BH Tools Debug: Created temporary video preview copy for external save: {temp_preview_filepath}")
                            
                            preview_subfolder = os.path.relpath(temp_preview_dir, folder_paths.get_temp_directory())
                            preview_type = "temp"
                        else:
                            # If saving to a relative path within ComfyUI's output, use its relative path
                            preview_subfolder = os.path.relpath(os.path.dirname(generated_video_path), folder_paths.get_output_directory())

                        self.preview_results.append({
                            "filename": preview_filename,
                            "subfolder": preview_subfolder,
                            "type": preview_type,
                            "format": f"video/{output_ext}",
                            "width": width,
                            "height": height
                        })
                        logger.debug(f"BH Tools Debug: Added main video to preview_results: {self.preview_results[-1]}")
                        
                        # Conditionally save workflow image
                        if export_workflow_image_with_video: # Only export if this option is True
                            logger.debug("BH Tools Debug: Exporting workflow image with video.")
                            workflow_thumb_info = self.save_images(
                                images=[images[0]],
                                filename_prefix=filename_prefix + "_workflow_thumb",
                                user_output_dir_input=output_dir, # Pass original output_dir for workflow thumb
                                output_extension="png",
                                prompt=prompt,
                                extra_pnginfo=extra_pnginfo,
                                verbose=verbose,
                                pbar=None,
                                save_only_first_image=True,
                                append_timestamp=append_timestamp,
                                overwrite_existing_files=overwrite_existing_files
                            )
                            if workflow_thumb_info:
                                logger.info(f"BH Tools: Successfully saved workflow thumbnail: {workflow_thumb_info[0]['filename']}")
                            else:
                                logger.warning("BH Tools Warning: Failed to save workflow thumbnail.")
                        else:
                            logger.info("BH Tools: Skipping workflow thumbnail export as 'export_workflow_image_with_video' is disabled.")
                    else:
                        logger.error("BH Tools ERROR: Video generation failed. Workflow thumbnail will not be saved.")
                    
                    # Clean up temporary images
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                        if verbose:
                            logger.debug(f"BH Tools Debug: Cleaned up temporary directory: {temp_dir}")

                else: # Image mode (not combining into video)
                    logger.debug("BH Tools Debug: Saving in image mode.")
                    # For image saving, if preview_batch is False and there are multiple images, only the first image is saved and previewed.
                    # Otherwise, all images in the batch are saved and previewed.
                    _save_only_first_image_for_save = (len(images) > 1 and not preview_batch)
                    saved_image_info = self.save_images(
                        images, filename_prefix, output_dir, output_ext, prompt, extra_pnginfo, 
                        verbose, pbar, save_only_first_image=_save_only_first_image_for_save, 
                        jpeg_quality=jpeg_quality, append_timestamp=append_timestamp, 
                        overwrite_existing_files=overwrite_existing_files
                    )
                    
                    # Extend preview_results if should_accumulate is True, otherwise replace
                    if should_accumulate:
                        self.preview_results.extend(saved_image_info)
                    else:
                        self.preview_results = saved_image_info

                    if self.preview_results: # Check self.preview_results after potential extension/replacement
                        output_filename_to_return = self.preview_results[0]['filename']
                        logger.info(f"BH Tools: Successfully saved image(s). First preview file: {os.path.join(self.preview_results[0]['subfolder'], self.preview_results[0]['filename'])}")
                        if verbose:
                            logger.debug(f"BH Tools Debug: Added saved image(s) to preview_results: {self.preview_results}")
                    else:
                        logger.warning("BH Tools Warning: No images were saved.")
            else: # save_output is False, only generate previews
                logger.info("BH Tools: Skipping save as 'save_output' is disabled. Generating preview only.")
                pbar.update(len(images))

                if _effective_combine_video:
                    logger.debug("BH Tools Debug: Generating temporary video for preview.")
                    temp_dir = os.path.join(folder_paths.get_temp_directory(), f"temp_preview_video_images_{datetime.now().strftime('%Y%m%d%H%M%S')}")
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_image_paths = []
                    for i, image in enumerate(images):
                        temp_filename = os.path.join(temp_dir, f"{i:05d}.png")
                        image_pil = self.tensor_to_pil(image) # Use helper function
                        image_pil.save(temp_filename)
                        temp_image_paths.append(temp_filename)
                    
                    preview_video_filename_part = self._generate_output_filename(
                        filename_prefix + "_preview", 0, output_ext, True, False, is_single_image_mode=True
                    )
                    # For preview-only mode, the base path is always a temporary ComfyUI path
                    preview_video_filepath_base_no_ext = os.path.join(folder_paths.get_temp_directory(), "bh_tools_preview", preview_video_filename_part)
                    os.makedirs(os.path.dirname(preview_video_filepath_base_no_ext), exist_ok=True)


                    # Ensure the preview codec is compatible and sensible for temporary preview
                    # For video formats, use h264 for preview unless it's gif/webp.
                    preview_codec = "libx264" 
                    if output_ext == "gif":
                        preview_codec = "gif"
                    elif output_ext == "webp":
                        preview_codec = None # PIL handles animated webp, not ffmpeg for preview
                    
                    preview_crf = 28 # Higher CRF for smaller preview size
                    if output_ext == "gif":
                        preview_crf = 0 # CRF not applicable for GIF

                    # Only attempt video generation if a valid ffmpeg_path exists and it's a video format
                    preview_generated = False
                    generated_preview_video_path = None
                    if self.ffmpeg_path and (preview_codec or output_ext in ["gif", "webp"]): # Check for ffmpeg path and valid output type
                        preview_generated, generated_preview_video_path = self.generate_video(
                            temp_image_paths, preview_video_filepath_base_no_ext, output_ext, video_fps, preview_codec, preview_crf,
                            audio_src_path=main_video_temp_audio_path, audio_sample_rate=sample_rate, audio_channels=main_video_audio_channels,
                            loop_count=video_loop_count, sync_audio=video_sync_audio, verbose=verbose
                        )
                    else:
                        logger.warning("BH Tools Warning: Skipping video preview generation due to missing FFmpeg or unsupported preview format.")

                    if preview_generated and generated_preview_video_path:
                        # For temp previews, filename should INCLUDE extension, and subfolder relative to ComfyUI's temp directory
                        self.preview_results.append({
                            "filename": os.path.basename(generated_preview_video_path),
                            "subfolder": os.path.relpath(os.path.dirname(generated_preview_video_path), folder_paths.get_temp_directory()),
                            "type": "temp",
                            "format": f"video/{output_ext}",
                            "width": width,
                            "height": height
                        })
                        output_filename_to_return = os.path.basename(generated_preview_video_path)
                        logger.debug(f"BH Tools Debug: Video preview generated: {generated_preview_video_path}. Preview results: {self.preview_results[-1]}")
                    else:
                        logger.warning("BH Tools Warning: Failed to generate video preview.")
                    
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                
                else: # Generate temporary images for preview (combine_video is False, save_output is False)
                    logger.debug("BH Tools Debug: Generating temporary images for preview.")
                    preview_temp_dir = os.path.join(folder_paths.get_temp_directory(), f"temp_batch_preview_{datetime.now().strftime('%Y%m%d%H%M%S')}")
                    os.makedirs(preview_temp_dir, exist_ok=True)
                    
                    images_to_preview = images
                    if not preview_batch and len(images) > 1:
                        images_to_preview = [images[-1]]

                    # If not accumulating, clear previous results before adding current
                    if not should_accumulate:
                         self.preview_results = []

                    for i, image in enumerate(images_to_preview):
                        temp_filename_part = self._generate_output_filename(
                            filename_prefix + "_preview", i, "png", True, False,
                            is_single_image_mode=(len(images_to_preview) == 1)
                        )
                        temp_filename_with_ext = os.path.join(preview_temp_dir, f"{temp_filename_part}.png")
                        image_pil = self.tensor_to_pil(image) # Use helper function
                        image_pil.save(temp_filename_with_ext)

                        # For temp previews, filename should INCLUDE extension, and subfolder relative to ComfyUI's temp directory
                        self.preview_results.append({
                            "filename": os.path.basename(temp_filename_with_ext),
                            "subfolder": os.path.relpath(os.path.dirname(temp_filename_with_ext), folder_paths.get_temp_directory()),
                            "type": "temp",
                            "format": "image/png",
                            "width": image_pil.width,
                            "height": image_pil.height
                        })
                        output_filename_to_return = os.path.basename(temp_filename_with_ext)
                        logger.debug(f"BH Tools Debug: Generated temp image for preview: {temp_filename_with_ext}. Preview results: {self.preview_results[-1]}")

        except Exception as e:
            logger.error(f"BH Tools ERROR: An unhandled critical error occurred in save_content_bh: {e}")
            import traceback
            traceback.print_exc()
            self.preview_results = []
            width, height = 0, 0
        finally:
            if main_video_temp_audio_path and os.path.exists(main_video_temp_audio_path):
                os.remove(main_video_temp_audio_path)
                if verbose:
                    logger.debug(f"BH Tools Debug: Cleaned up temporary main video audio file: {main_video_temp_audio_path}")


        logger.debug(f"BH Tools Debug: Final preview_results before UI return: {self.preview_results}")
        if self.preview_results and self.preview_results[0].get("format", "").startswith("video/"):
            logger.debug("BH Tools Debug: Returning UI data as 'gifs'.")
            return {
                "ui": {"gifs": self.preview_results},
                "result": (output_filename_to_return, width, height)
            }
        else:
            logger.debug("BH Tools Debug: Returning UI data as 'images'.")
            return {
                "ui": {"images": self.preview_results},
                "result": (output_filename_to_return, width, height)
            }
