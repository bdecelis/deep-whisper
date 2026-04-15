"""
deep-whisper · test_env.py
==========================
Environment smoke test. Run this after install.ps1 to confirm all
dependencies are correctly installed and the GPU is accessible.

Usage:
    python test_env.py

All checks must pass before running the pipeline tests.
"""

import sys


def check(label: str, fn, expected=None):
    """Run fn(), print result, return True if no exception."""
    try:
        result = fn()
        if expected is not None and result != expected:
            print(f"  WARN  {label:<30s} got {result!r}, expected {expected!r}")
            return False
        print(f"  OK    {label:<30s} {result}")
        return True
    except Exception as exc:
        print(f"  FAIL  {label:<30s} {exc}")
        return False


print()
print("=== deep-whisper environment check ===")
print()

failures = 0

# ── PyTorch ──────────────────────────────────────────────────────────────────
print("── PyTorch ───────────────────────────────────────────")

def torch_version():
    import torch
    return torch.__version__

def cuda_available():
    import torch
    return torch.cuda.is_available()

def cuda_version():
    import torch
    v = torch.version.cuda
    if v is None:
        raise RuntimeError(
            "torch is installed but has no CUDA build (torch.version.cuda is None). "
            "Reinstall torch with a CUDA index URL — see README §Installation."
        )
    return f"{v}  ← this is the CUDA version deep-whisper will use"

def gpu_name():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    return torch.cuda.get_device_name(0)

def vram_gb():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    return f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"

def torchaudio_version():
    import torchaudio
    return torchaudio.__version__

if not check("torch installed",      torch_version): failures += 1
if not check("CUDA version (env)",   cuda_version): failures += 1
if not check("CUDA available",       cuda_available, True): failures += 1
if not check("GPU name",             gpu_name): failures += 1
if not check("VRAM",                 vram_gb): failures += 1
if not check("torchaudio installed", torchaudio_version): failures += 1

# ── CTranslate2 / cuDNN ───────────────────────────────────────────────────────
print()
print("── CTranslate2 / cuDNN ───────────────────────────────")

def ct2_version():
    import ctranslate2
    return ctranslate2.__version__

def cudnn_dlls_present():
    """
    Verify the three cuDNN 8 DLLs are findable by CTranslate2 on Windows.
    Checks the ctranslate2 package directory first (where install.ps1 copies
    them), then falls back to checking PATH entries.
    """
    import sys
    if sys.platform != "win32":
        return "skipped (non-Windows)"

    import os, ctranslate2
    required = [
        "cudnn64_8.dll",
        "cudnn_ops_infer64_8.dll",
        "cudnn_cnn_infer64_8.dll",
    ]
    ct2_dir      = os.path.dirname(ctranslate2.__file__)
    path_dirs    = os.environ.get("PATH", "").split(os.pathsep)
    search_dirs  = [ct2_dir] + path_dirs

    missing = []
    for dll in required:
        found = any(
            os.path.exists(os.path.join(d, dll))
            for d in search_dirs if d
        )
        if not found:
            missing.append(dll)

    if missing:
        raise RuntimeError(
            f"Missing cuDNN 8 DLLs: {', '.join(missing)}\n"
            f"  Run install.ps1 to copy them automatically, or see README §Installation."
        )
    return f"all 3 DLLs present in {ct2_dir}"

if not check("ctranslate2 installed", ct2_version): failures += 1
if not check("cuDNN 8 DLLs present",  cudnn_dlls_present): failures += 1



def fw_version():
    from faster_whisper import WhisperModel
    import faster_whisper
    return faster_whisper.__version__

def fw_load_model():
    """Load tiny model to verify CTranslate2 CUDA works end-to-end."""
    from faster_whisper import WhisperModel
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Skipped — CUDA not available")
    # Use 'tiny' for the smoke test — fast download, minimal VRAM
    model = WhisperModel("tiny", device="cuda", compute_type="int8")
    return "loaded (tiny)"

if not check("faster-whisper installed", fw_version): failures += 1
if not check("CTranslate2 CUDA load",    fw_load_model): failures += 1

# ── whisperx ─────────────────────────────────────────────────────────────────
print()
print("── whisperx ──────────────────────────────────────────")

def wx_version():
    # whisperx does not expose __version__ — use importlib.metadata instead
    from importlib.metadata import version, PackageNotFoundError
    try:
        return version("whisperx")
    except PackageNotFoundError:
        # Some installs register under a different name
        import whisperx  # noqa — confirms importable at minimum
        return "installed (version unknown)"

def wx_load_align():
    """Load English wav2vec2-base alignment model."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Skipped — CUDA not available")
    import whisperx
    model, meta = whisperx.load_align_model(language_code="en", device="cuda")
    return "alignment model loaded"

if not check("whisperx installed",   wx_version): failures += 1
if not check("alignment model load", wx_load_align): failures += 1

# ── silero-vad ────────────────────────────────────────────────────────────────
print()
print("── silero-vad ────────────────────────────────────────")

def vad_load():
    from silero_vad import load_silero_vad
    model = load_silero_vad()
    return "VAD model loaded"

def vad_run():
    """Run VAD on a 1-second silent array."""
    import torch
    from silero_vad import load_silero_vad, get_speech_timestamps
    model = load_silero_vad()
    audio = torch.zeros(16_000)
    ts = get_speech_timestamps(audio, model, sampling_rate=16_000, return_seconds=True)
    return f"VAD ran OK (segments: {len(ts)})"

if not check("silero-vad loaded", vad_load): failures += 1
if not check("VAD inference",     vad_run): failures += 1

# ── Audio libraries ───────────────────────────────────────────────────────────
print()
print("── Audio libraries ───────────────────────────────────")

def librosa_version():
    import librosa
    return librosa.__version__

def soundfile_version():
    import soundfile
    return soundfile.__version__

def librosa_load():
    """Verify librosa can generate and process a tone."""
    import numpy as np, librosa
    tone = (0.5 * np.sin(
        2 * np.pi * 440 * np.arange(16_000, dtype=np.float32) / 16_000
    ))
    rms = librosa.feature.rms(y=tone)[0]
    return f"RMS computed (frames: {len(rms)})"

if not check("librosa installed",   librosa_version): failures += 1
if not check("soundfile installed", soundfile_version): failures += 1
if not check("librosa processing",  librosa_load): failures += 1

# ── Text utilities ────────────────────────────────────────────────────────────
print()
print("── Text utilities ────────────────────────────────────")

def num2words_check():
    import num2words
    result = num2words.num2words(2024, to="year")
    assert "twenty" in result, f"Unexpected: {result}"
    return result

def dmp_check():
    import diff_match_patch as dmp_mod
    dmp = dmp_mod.diff_match_patch()
    diffs = dmp.diff_main("hello world", "hello there")
    return f"diff OK ({len(diffs)} ops)"

if not check("num2words",          num2words_check): failures += 1
if not check("diff-match-patch",   dmp_check): failures += 1

# ── VRAM summary ─────────────────────────────────────────────────────────────
print()
print("── VRAM usage after all loads ────────────────────────")

def vram_used():
    import torch
    if not torch.cuda.is_available():
        return "N/A (CUDA not available)"
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved  = torch.cuda.memory_reserved()  / 1e9
    total     = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"{allocated:.2f} GB allocated  /  {total:.1f} GB total"

check("VRAM usage", vram_used)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("─" * 52)
if failures == 0:
    print(f"  All checks passed. Environment is ready.")
    print(f"  Next step: run test_pipeline.py with a real audio file.")
else:
    print(f"  {failures} check(s) failed.")
    print(f"  Fix the failures above before proceeding.")
    print()
    print(f"  Common fixes:")
    print(f"    CUDA not available  → re-run install.ps1 with correct -CudaTag")
    print(f"    Model load failed   → check internet connection (first run downloads weights)")
    print(f"    Library not found   → pip install -r requirements.txt")
print()

sys.exit(0 if failures == 0 else 1)
