from .nodes.CinematicSceneDirectorBHTools import CinematicSceneDirector
from .nodes.EndOfWorkflowClearingNodeBHTools import EndOfWorkflowClearingNodeBHTools
from .nodes.PromptInferenceBHTools import PromptInferenceBHTools

# Set categories for each node class
CinematicSceneDirector.CATEGORY = "BHTools"
EndOfWorkflowClearingNodeBHTools.CATEGORY = "BHTools"
PromptInferenceBHTools.CATEGORY = "BHTools"

NODE_CLASS_MAPPINGS = {
    "CinematicSceneDirector|BHTools": CinematicSceneDirector,
    "EndOfWorkflowClearingNodeBHTools|BHTools": EndOfWorkflowClearingNodeBHTools,
    "PromptInferenceBHTools|BHTools": PromptInferenceBHTools
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CinematicSceneDirector|BHTools": "🎬 Cinematic Scene Director | BH Tools",
    "EndOfWorkflowClearingNodeBHTools|BHTools": "🎬 End Of Workflow Clearing Node | BH Tools",
    "PromptInferenceBHTools|BHTools": "🎬 Prompt Inference | BH Tools"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]