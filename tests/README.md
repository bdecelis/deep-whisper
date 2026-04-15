# deep-whisper · Tests

This folder contains two types of tests that serve different purposes and
are run at different points in the development workflow.

---

## Overview

| File | Type | Needs GPU | Run with |
|---|---|---|---|
| `test_env.py` | Environment check | Yes | `python tests/test_env.py` |
| `test_pipeline.py` | Integration test | Yes | `python tests/test_pipeline.py --audio ...` |
| `test_*.py` *(future)* | Unit tests | No | `pytest tests/` |

---

## `test_env.py` — Environment verification

**Run this first, before anything else.**

Checks that every dependency is correctly installed and the GPU is accessible.
Does not require an audio file. Loads small versions of the models to confirm
the full CUDA stack works end-to-end.

```powershell
python tests/test_env.py
```

**What it checks:**
- PyTorch version and CUDA availability
- CUDA version in use by this environment (`torch.version.cuda`)
- GPU name and VRAM
- torchaudio
- CTranslate2 and cuDNN 8 DLL placement
- faster-whisper (loads `whisper-tiny` on GPU)
- whisperx (loads `wav2vec2-base` alignment model)
- Silero VAD (loads and runs on a silent tensor)
- librosa (runs an RMS computation)
- soundfile, num2words, diff-match-patch
- VRAM usage after all models loaded

All checks must pass before running `test_pipeline.py`.

**Common failures and fixes:**

| Error | Fix |
|---|---|
| `CUDA not available` | Re-run `install.ps1` with the correct `-CudaTag` |
| `Could not load cudnn_ops_infer64_8.dll` | cuDNN DLLs not copied — re-run step 3 of `install.ps1` |
| `Numba needs NumPy 2.2 or less` | `pip install "numpy>=1.24,<2.3"` |
| `whisperx` import fails | Re-install: `pip install whisperx>=3.0.0` after torchaudio |
| Model download hangs | First run downloads ~4 GB of weights — give it time |

---

## `test_pipeline.py` — End-to-end integration test

Runs all seven pipeline stages against a real audio file, printing the
output of each stage before proceeding to the next. Designed to be run
interactively so failures are easy to localise.

```powershell
# Basic run
python tests/test_pipeline.py --audio path\to\audio.wav

# With a context prompt (improves Whisper's domain vocabulary)
python tests/test_pipeline.py --audio path\to\audio.wav ^
    --prompt "A tutorial about Python programming."

# Test the user-transcript path (exercises reconcile_segments)
python tests/test_pipeline.py --audio path\to\audio.wav ^
    --transcript "The exact words you expect to be spoken."

# Save the final JSON output to a file
python tests/test_pipeline.py --audio path\to\audio.wav ^
    --output result.json

# Fast run for quick iteration (greedy decode, no boundary snapping)
python tests/test_pipeline.py --audio path\to\audio.wav ^
    --quality fast --no-snap
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--audio` | *(required)* | Path to the audio file to transcribe |
| `--prompt` | `""` | Context prompt to seed Whisper's vocabulary |
| `--transcript` | `""` | User-provided transcript — activates the reconcile path |
| `--quality` | `balanced` | Whisper preset: `fast`, `balanced`, or `accurate` |
| `--output` | `""` | Path to write the final JSON output |
| `--no-snap` | off | Disable energy boundary snapping |

**What a successful run looks like:**

```
=== deep-whisper pipeline test ===
  audio:      my_audio.wav
  quality:    balanced
  ...

────────────────────────────────────────────────────────
  Stage 1 — load_audio + normalize_audio
────────────────────────────────────────────────────────
  OK   load_audio                         shape=(480000,)  dtype=float32
  OK   normalize                          peak=1.0000
  OK   duration                           30.00s
  OK   load time                          0.18s

────────────────────────────────────────────────────────
  Stage 2 — get_speech_chunks  (VAD)
────────────────────────────────────────────────────────
  OK   chunks detected                    2
  ...
  chunk  0:  offset=  0.50s  len=14.30s
  chunk  1:  offset= 16.20s  len=12.80s
...
```

**What to look for at each stage:**

| Stage | Healthy | Investigate if |
|---|---|---|
| 1 — Load | peak = 1.0, duration matches file | RuntimeError on load |
| 2 — VAD | 1–5 chunks for 30–120s audio | 0 chunks (all filtered as silence) or 20+ tiny chunks |
| 3 — Transcribe | Text matches audio reasonably | Empty segments, repeated hallucinations |
| 4 — Reconcile | Changed segment texts match user transcript vocabulary | No changes when differences were expected |
| 5 — Normalise | Numbers/abbreviations expanded in output | `Dr.` still present, `$200` not expanded |
| 6 — Align | Word timestamps spaced plausibly | All words at `0.000` (alignment failed silently) |
| 7 — Output | Valid JSON, transcript makes sense | Serialisation error, empty transcript |

**Recommended test audio:**
- 30–120 seconds of clean spoken English
- A voice memo, recorded narration, or exported podcast clip works well
- Avoid music, background noise, or multiple overlapping speakers
- The quality preset comparison (`fast` vs `balanced` vs `accurate`) is most useful on audio with domain-specific vocabulary or fast speech

---

## Unit tests *(future)*

Module-level unit tests for the pipeline stages live here as `test_*.py`
files and are discovered automatically by pytest:

```powershell
pytest tests/
```

These tests stub GPU dependencies and run without a GPU, making them
suitable for CI. The `conftest.py` at the project root and `pythonpath = ["."]`
in `pyproject.toml` ensure `from pipeline.x import y` resolves correctly
regardless of which directory pytest is invoked from.

---

## Running everything

```powershell
# Step 1 — confirm environment is healthy
python tests/test_env.py

# Step 2 — confirm pipeline works end-to-end
python tests/test_pipeline.py --audio path\to\audio.wav

# Step 3 — run unit tests (no GPU needed)
pytest tests/
```
