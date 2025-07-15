
from .nodes.CinematicSceneDirectorNodeBHTools import CinematicSceneDirectorTools
from .nodes.EndOfWorkflowClearingNodeBHTools import EndOfWorkflowClearingBHTools
from .nodes.PromptInferenceNodeBHTools import PromptInferenceBHTools
from .nodes.SaveVideoOrImageNodeBHTools import SaveImageVideoBHTools


NODE_CLASS_MAPPINGS = {
    "CinematicSceneDirectorTools": CinematicSceneDirectorTools,
    "EndOfWorkflowClearingBHTools": EndOfWorkflowClearingBHTools,
    "PromptInferenceBHTools": PromptInferenceBHTools,
    "SaveImageVideoBHTools": SaveImageVideoBHTools,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CinematicSceneDirectorTools": "🎬 Cinematic Scene Director | BH Tools",
    "EndOfWorkflowClearingBHTools": "🎬 End Workflow Cleanup | BH Tools",
    "PromptInferenceBHTools": "🎬 Prompt Inference | BH Tools",
    "SaveImageVideoBHTools": "🎬 Save Image/Video | BH Tools",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "get_web_extensions"]
