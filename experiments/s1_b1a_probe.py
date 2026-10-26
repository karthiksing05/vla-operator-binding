"""PLAN_SOLVE S1: can this env load GroundingDINO (transformers) from the HF hub?  Prints versions only."""
import transformers
print("transformers", transformers.__version__, flush=True)
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base")
print("grounding-dino-base loaded", flush=True)
