# test_models.py — verify all three models load and fit in VRAM
import torch
from pipeline.models import get_whisper_model, get_align_model, get_vad_model, cache_status

vad = get_vad_model()
print("VAD loaded")

whisper = get_whisper_model()
print(f"Whisper loaded  VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

align, meta = get_align_model()
print(f"Align loaded    VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

print(cache_status())