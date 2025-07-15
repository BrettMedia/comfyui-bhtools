# comfyui-bhtools/nodes/EndOfWorkflowClearingNodeBHTools.py

import gc
import os
import logging
import shutil
import tempfile
from pathlib import Path

import psutil
import torch

logger = logging.getLogger(__name__)

def clear_temp_dir(pattern="*comfy*") -> int:
    """Clears temporary files matching a pattern."""
    count = 0
    temp_dir = Path(tempfile.gettempdir())
    for p in temp_dir.glob(pattern):
        try:
            if p.is_file():
                p.unlink()
            else:
                shutil.rmtree(p)
            count += 1
        except Exception as e:
            logger.warning(f"Failed to remove {p}: {e}")
    return count

def unload_comfyui_models() -> bool:
    """Attempts to unload ComfyUI models from memory."""
    try:
        import comfy.model_management as mm
        if hasattr(mm, 'unload_all_models'):
            mm.unload_all_models()
            return True
        if hasattr(mm, 'cleanup_models'): # Fallback for older ComfyUI versions
            mm.cleanup_models()
            return True
    except Exception as e:
        logger.warning(f"Model unload error: {e}")
    return False

def gpu_clear(method: str, reset_peak: bool = False) -> None:
    """Clears GPU memory based on the specified method."""
    torch.cuda.empty_cache() # Always clear cache
    if method in ("hard", "complete"):
        torch.cuda.ipc_collect() # Aggressive collection
        torch.cuda.synchronize() # Wait for GPU operations to complete
    if method == "complete" and reset_peak:
        # Reset peak memory stats for accurate monitoring
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()

class EndOfWorkflowClearingBHTools:
    """
    ComfyUI node: Performs a final cleanup at the end of a workflow execution.
    This node helps free up system RAM and GPU VRAM, clear temporary files,
    and unload models to optimize resource usage.
    """
    def __init__(self):
        self._has_run = False # Internal flag to ensure 'only_run_once' functionality

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("*", {"tooltip": "Connect any node output here to trigger the cleanup."}),
                "clear_vram": ("BOOLEAN", {"default": True, "tooltip": "If checked, attempts to clear GPU VRAM."}),
                "clear_models": ("BOOLEAN", {"default": True, "tooltip": "If checked, attempts to unload ComfyUI models from VRAM."}),
                "clear_torch_cache": ("BOOLEAN", {"default": False, "tooltip": "If checked, clears PyTorch's internal CUDA cache."}),
                "clear_system_cache": ("BOOLEAN", {"default": False, "tooltip": "If checked, attempts to clear system-level file caches (Linux/macOS 'sync')."}),
                "clear_temp_files": ("BOOLEAN", {"default": False, "tooltip": "If checked, removes temporary files created by ComfyUI."}),
                "force_gc": ("BOOLEAN", {"default": True, "tooltip": "If checked, forces Python's garbage collector to run."}),
                "vram_clear_method": (["soft", "hard", "complete"], {"default": "hard", "tooltip": "Method for VRAM clearing: 'soft' (empty cache), 'hard' (ipc_collect), 'complete' (hard + reset peak stats)."}),
                "only_if_above_threshold": ("BOOLEAN", {"default": False, "tooltip": "If checked, cleanup only runs if RAM/VRAM usage exceeds thresholds."}),
                "only_run_once": ("BOOLEAN", {"default": True, "tooltip": "If checked, the cleanup process will only execute once per ComfyUI session."}),
                "verbose": ("BOOLEAN", {"default": True, "tooltip": "If checked, prints detailed cleanup status to the console."}),
            },
            "optional": {
                "vram_threshold_gb": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 64.0, "step": 0.1,
                    "display": "slider", "tooltip": "Minimum VRAM usage (GB) to trigger cleanup if 'only_if_above_threshold' is true."
                }),
                "ram_threshold_gb": ("FLOAT", {
                    "default": 4.0, "min": 0.0, "max": 128.0, "step": 0.1,
                    "display": "slider", "tooltip": "Minimum RAM usage (GB) to trigger cleanup if 'only_if_above_threshold' is true."
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("cleanup_report",)
    FUNCTION = "end_of_workflow_cleanup"
    CATEGORY = "BH Tools" # Node category for ComfyUI menu

    def end_of_workflow_cleanup(
        self,
        trigger, # This input is just to trigger execution, its value is not used
        clear_vram,
        clear_models,
        clear_torch_cache,
        clear_system_cache,
        clear_temp_files,
        force_gc,
        vram_clear_method,
        only_if_above_threshold,
        only_run_once,
        verbose,
        vram_threshold_gb=2.0,
        ram_threshold_gb=4.0,
    ):
        # Guard to ensure cleanup runs only once per ComfyUI session if specified
        if only_run_once and self._has_run:
            return ("Cleanup already executed, skipping.",)
        if only_run_once:
            self._has_run = True

        status = [] # List to collect status messages for the report

        # Validate threshold values
        if vram_threshold_gb < 0 or ram_threshold_gb < 0:
            status.append("Error: Threshold values must be non-negative. Cleanup aborted.")
            report = "\n".join(status)
            print(report)
            return (report,)

        # Record initial memory usage
        vm = psutil.virtual_memory()
        initial_ram = vm.used / 1024**3 # Convert bytes to GB
        initial_vram = (
            torch.cuda.memory_allocated() / 1024**3
            if torch.cuda.is_available() else 0.0
        )

        if verbose:
            msg = f"Cleanup Start — RAM: {initial_ram:.2f} GB"
            if torch.cuda.is_available():
                msg += f", VRAM: {initial_vram:.2f} GB"
            status.append(msg)

        # Conditional cleanup based on thresholds
        if only_if_above_threshold:
            ram_cond = initial_ram >= ram_threshold_gb
            vram_cond = initial_vram >= vram_threshold_gb
            if not (ram_cond or vram_cond):
                status.append("✓ Usage below thresholds, cleanup skipped.")
                report = "\n".join(status)
                print(report)
                return (report,)
            if ram_cond:
                status.append(f"⚠ RAM usage ({initial_ram:.2f} GB) ≥ threshold ({ram_threshold_gb:.1f} GB).")
            if vram_cond:
                status.append(f"⚠ VRAM usage ({initial_vram:.2f} GB) ≥ threshold ({vram_threshold_gb:.1f} GB).")

        # Perform cleanup actions
        # Unload ComfyUI models
        if clear_models and torch.cuda.is_available():
            if unload_comfyui_models():
                status.append("✓ Models unloaded.")
            else:
                status.append("⚠ Model unload skipped or failed.")

        # Force Python garbage collection
        if force_gc:
            collected = gc.collect()
            status.append(f"✓ GC freed {collected} objects.")

        # Clear VRAM
        if clear_vram and torch.cuda.is_available():
            try:
                gpu_clear(vram_clear_method, reset_peak=(vram_clear_method == "complete"))
                status.append(f"✓ VRAM {vram_clear_method} clear performed.")
            except Exception as e:
                status.append(f"⚠ VRAM clear error: {e}")

        # Clear PyTorch CUDA cache
        if clear_torch_cache:
            try:
                torch.cuda.empty_cache()
                status.append("✓ PyTorch CUDA cache cleared.")
            except Exception as e:
                status.append(f"⚠ PyTorch cache error: {e}")

        # Clear system cache (Linux/macOS specific)
        if clear_system_cache:
            try:
                if os.name == "posix": # For Linux/macOS
                    ret = os.system("sync")
                    status.append("✓ System sync" if ret == 0 else f"⚠ Sync returned {ret}")
                else: # For Windows, this is a no-op as Windows manages cache differently
                    status.append("✓ Windows system cache clear (no-op).")
            except Exception as e:
                status.append(f"⚠ System cache clear error: {e}")

        # Clear temporary files
        if clear_temp_files:
            removed = clear_temp_dir()
            status.append(f"✓ Temp files removed: {removed} items.")

        # Record final memory usage and report freed memory
        if verbose:
            vm2 = psutil.virtual_memory()
            final_ram = vm2.used / 1024**3
            freed_ram = max(0.0, initial_ram - final_ram) # Ensure non-negative
            status.append(f"Cleanup End — RAM freed {freed_ram:.2f} GB.")
            if torch.cuda.is_available():
                final_vram = torch.cuda.memory_allocated() / 1024**3
                freed_vram = max(0.0, initial_vram - final_vram) # Ensure non-negative
                status.append(f"VRAM freed {freed_vram:.2f} GB.")

        status.append("🎉 Cleanup complete!")
        report = "\n".join(status)
        print(report) # Print the report to the ComfyUI console
        return (report,)

