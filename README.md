# deep-whisper

> GPU-accelerated local audio transcription with precise word-level timestamps.

**deep-whisper** is a Python pipeline that takes an audio file and returns a structured transcript — with timestamps accurate to the millisecond at both word and sentence level — running entirely on your local machine. No API keys, no data leaving your machine, no usage costs.

It is built to run efficiently on a consumer CUDA GPU (8 GB VRAM is sufficient) and is designed for clean spoken-word audio such as voiceovers, interviews, lectures, and narration.

---

## Table of Contents

1. [What it does](#what-it-does)
2. [How it works](#how-it-works)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Quick start](#quick-start)
6. [Output format](#output-format)
7. [Configuration options](#configuration-options)
8. [The user transcript path](#the-user-transcript-path)
9. [Architecture overview](#architecture-overview)
10. [Performance notes](#performance-notes)
11. [Limitations](#limitations)
12. [ComfyUI integration](#comfyui-integration)
13. [Development](#development)
14. [Acknowledgements](#acknowledgements)

---

## What it does

Given an audio file, deep-whisper produces:

- **A full text transcript** of the spoken content
- **Sentence-level timestamps** — when each sentence starts and ends
- **Word-level timestamps** — when each individual word starts and ends
- **Confidence scores** — how certain the pipeline is about each word and sentence
- **Flags** for words or segments where the pipeline is uncertain

All of this is returned as structured JSON, making it straightforward to consume from any downstream tool — subtitle generators, video editors, language model pipelines, or custom applications.

---

## How it works

deep-whisper chains five specialised stages, each responsible for one aspect of the transcription problem:

```
Audio file
    │
    ▼
┌─────────────────────────────────┐
│  Voice Activity Detection        │  Silero VAD (CPU)
│  Find where speech occurs        │  Splits audio into speech chunks,
│  and chunk it intelligently      │  sized for optimal Whisper accuracy
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Transcription                   │  faster-whisper (GPU)
│  Convert speech to text          │  OpenAI Whisper large-v3-turbo
│  with rolling context            │  with rolling prompt chaining
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Text Normalisation              │  num2words + regex (CPU)
│  Expand numbers and symbols      │  "$200" → "two hundred dollars"
│  to their spoken form            │  "Dr." → "Doctor", "AI" → "A I"
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Forced Alignment                │  WhisperX / wav2vec2 (GPU)
│  Pin each word to an exact       │  CTC alignment is more accurate
│  point in the audio              │  than Whisper's built-in timestamps
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Post-processing                 │  CPU
│  Assemble, annotate, and         │  Confidence scoring, flagging,
│  serialise the final output      │  and JSON assembly
└────────────────┴────────────────┘
                 │
                 ▼
           JSON output
```

### Why these choices?

**Whisper** is the most capable open speech recognition model available. OpenAI's original implementation is slow; **faster-whisper** reimplements it using CTranslate2 and quantisation, giving ~4–6× faster inference at equivalent accuracy. The `large-v3-turbo` variant is a distilled model that is approximately 6× faster than `large-v3` with minimal accuracy loss on clean speech.

**Forced CTC alignment** (via WhisperX's wav2vec2 integration) produces more precise word timestamps than Whisper's built-in attention-weight timestamps. Whisper guesses word timing from its attention patterns; forced alignment solves a constrained problem — given this text and this audio, find the exact frame where each phoneme occurs.

**Energy trough boundary snapping** further tightens word boundaries by locating the nearest local energy minimum around each timestamp — word transitions almost always coincide with reduced acoustic energy, so this is a low-cost refinement that adds real precision.

**Text normalisation before alignment** prevents a common silent failure: if the text says "$200" but the speaker said "two hundred dollars", the aligner cannot match phonemes correctly and timestamps drift. Normalising to spoken form first ensures text and audio match as closely as possible.

---

## Requirements

| Requirement | Minimum | Notes |
|---|---|---|
| Python | 3.10+ | |
| CUDA GPU | 8 GB VRAM | NVIDIA only |
| CUDA Driver | 11.8+ | Check with `nvidia-smi` |
| OS | Windows 11 / Linux | macOS not supported (no CUDA) |
| Disk space | ~6 GB | For model weights (downloaded on first run) |

> **No internet connection is required after first run.** Model weights are cached locally by faster-whisper and whisperx.

---

## Installation

### The short version

```powershell
pip install deep-whisper
deep-whisper-setup
```

That's it. `deep-whisper-setup` is a command that becomes available the
moment `pip install` finishes. It detects your CUDA version automatically,
installs PyTorch, faster-whisper, whisperx, and cuDNN (Windows), and handles
all the known issues described below. Run `python test_env.py` afterward to
confirm everything is working.

> **ComfyUI users:** If you are installing via `ComfyUI-BDC_DeepWhisper`,
> you do not need to do any of this — `install.py` in that package handles
> everything automatically when ComfyUI Manager installs the node.

---

### What `deep-whisper-setup` does

The GPU setup cannot be done safely by `pip install` alone because
`whisperx` — one of the required packages — pulls in a CPU-only PyTorch
as a transitive dependency, silently overwriting any CUDA-capable build
already in your environment. `deep-whisper-setup` handles this by:

1. Fixing any packages with broken pip metadata before resolution starts
2. Detecting the CUDA version in use by your Python environment
3. Installing PyTorch with the correct CUDA build if needed
4. Installing faster-whisper
5. Installing whisperx, then automatically detecting and repairing any
   torch CUDA overwrite
6. Copying cuDNN 8 DLLs into the CTranslate2 package directory (Windows only)
7. Running a final verification pass

If anything goes wrong, the output tells you exactly what the problem is
and what command to run to fix it.

---

### Installing from GitHub (latest commit)

```powershell
pip install git+https://github.com/bdecelis/deep-whisper.git
deep-whisper-setup
```

### Local development install

```powershell
git clone https://github.com/bdecelis/deep-whisper.git
cd deep-whisper
pip install -e .
deep-whisper-setup
```

The editable install (`-e .`) means changes to `pipeline/` source are
reflected immediately without reinstalling.

### Targeting a specific Python environment

If you have multiple Python environments (conda, venv, ComfyUI portable),
always run both commands with the same Python:

```powershell
# ComfyUI portable — both commands must use the same Python
.\python_embeded\python.exe -m pip install deep-whisper
.\python_embeded\python.exe -m pipeline.setup_gpu

# venv — activate first, then use the defaults
pip install deep-whisper
deep-whisper-setup

# Conda
conda activate your_env
pip install deep-whisper
deep-whisper-setup
```

---

### Verifying the installation

```powershell
python tests/test_env.py
```

All checks must pass before running the pipeline. On first use, model
weights (~4 GB) are downloaded and cached locally.

> **If you see `Numba needs NumPy 2.2 or less`:**
> `pip install "numpy>=1.24,<2.3"`

---

### Known issues and manual fixes

These are handled automatically by `deep-whisper-setup`. This section is
for anyone who encounters them in other contexts.

#### pytorch-lightning invalid requirement

```
error: invalid-installed-package
Cannot process installed package pytorch-lightning 1.7.7 ...
  torch (>=1.9.*)
  .* suffix can only be used with `==` or `!=` operators
```

`pytorch-lightning < 2.0.0` has a malformed version specifier that pip 24.1+
rejects. This can persist even with `--no-user` when the broken package is in
user site-packages that has been explicitly added to `sys.path`.

**Fix:**
```powershell
# Set before any pip call to isolate pip from user site-packages
$env:PYTHONNOUSERSITE = "1"

# Then upgrade or remove the broken package
python -m pip install "pytorch-lightning>=2.0.0"
# or: python -m pip uninstall pytorch-lightning -y
```

#### whisperx overwrites CUDA torch

If `torch.cuda.is_available()` returns `False` after installation:

```powershell
# Check what CUDA version your environment expects
python -c "import torch; print(torch.version.cuda)"

# Force-reinstall with the matching tag (replace cu128 with your version)
pip install --force-reinstall --no-user ^
    --index-url https://download.pytorch.org/whl/cu128 ^
    torch torchaudio
```

#### cuDNN not found (Windows)

If you see `Could not load library cudnn_ops_infer64_8.dll`:
CTranslate2 requires cuDNN 8 DLLs to be present in its package directory.
Run `deep-whisper-setup` — it places them automatically.

### Step 4 — Install remaining pipeline dependencies

```powershell
pip install -r requirements.txt
```

### Step 5 — Verify the installation

```powershell
python test_env.py
```

All checks should pass before using the pipeline. On first run, model weights (~4 GB total) will be downloaded and cached.

> **Windows note:** If you see `Numba needs NumPy 2.2 or less`, run:
> `pip install "numpy>=1.24,<2.3"`

---

### Using the automated installer

`install.ps1` handles steps 1–4 automatically — CUDA detection, PyTorch, cuDNN, and all remaining dependencies — in the correct order:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\install.ps1
```

**Installing into an existing ComfyUI environment?**  
Pass the path to ComfyUI's Python explicitly so the installer targets the right environment. If you use the wrong Python, deep-whisper will be installed into a different environment than the one ComfyUI runs and nothing will work.

```powershell
# Windows portable (most common install type)
.\install.ps1 -PythonExe "C:\ComfyUI_windows_portable\python_embeded\python.exe"

# venv (activate the venv first, then just use the default)
.\install.ps1

# Conda
.\install.ps1 -PythonExe "C:\Users\you\miniconda3\envs\comfyui\python.exe"
```

To find ComfyUI's Python path if you are not sure which type of install you have:

```powershell
# From inside ComfyUI's terminal / embedded console, or from a terminal
# where the correct environment is already active:
python -c "import sys; print(sys.executable)"
```

> **Note for ComfyUI users:** If you are installing deep-whisper via the
> `ComfyUI-BDC_DeepWhisper` node, `install.py` in that repo handles this
> automatically — you do not need to run `install.ps1` at all. This manual
> process is only needed when installing `deep-whisper` as a standalone
> package outside of ComfyUI.

---

## Quick start

### Single-call entry point

```python
import json
from pipeline import run

result = run("my_audio.wav")
print(result["transcript"])
print(json.dumps(result, indent=2))
```

### Step-by-step (more control)

```python
from pipeline.audio       import load_audio, normalize_audio
from pipeline.vad         import get_speech_chunks
from pipeline.transcribe  import transcribe_chunks
from pipeline.normalise   import normalise_segments
from pipeline.align       import align_segments
from pipeline.postprocess import build_output, serialise

# Load and prepare audio
audio  = normalize_audio(load_audio("my_audio.wav"))

# Detect speech and split into chunks
chunks = get_speech_chunks(audio)

# Transcribe with optional context prompt
segments = transcribe_chunks(
    chunks,
    initial_prompt="A lecture about machine learning and neural networks.",
    quality="balanced",   # "fast", "balanced", or "accurate"
    language="en",
)

# Normalise text to spoken form for accurate alignment
segments = normalise_segments(segments)

# Align words to audio with precision timestamps
segments = align_segments(segments, audio)

# Assemble the final output
output = build_output(
    segments, audio,
    whisper_model    = "large-v3-turbo",
    alignment_model  = "wav2vec2-base-960h",
    timestamp_level  = "both",   # "both", "segment", "word", or "none"
)

# Save to file
with open("transcript.json", "w") as f:
    f.write(serialise(output))
```

---

## Output format

The output is a JSON object with three main sections:

```json
{
  "schema_version": "1.0",

  "metadata": {
    "duration_seconds": 125.4,
    "language": "en",
    "whisper_model": "large-v3-turbo",
    "alignment_model": "wav2vec2-base-960h",
    "prompt": "The context prompt used.",
    "user_transcript_provided": false,
    "timestamp_level": "both",
    "timestamp_utc": "2026-04-14T10:00:00Z"
  },

  "transcript": "Hello, this is a test of the pipeline.",

  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 4.85,
      "text": "Hello, this is a test of the pipeline.",
      "confidence": 0.96,
      "flagged": false,
      "words": [
        { "word": "Hello",    "start": 0.00, "end": 0.42, "confidence": 0.99, "low_confidence": false },
        { "word": "this",     "start": 0.55, "end": 0.73, "confidence": 0.97, "low_confidence": false },
        { "word": "is",       "start": 0.74, "end": 0.85, "confidence": 0.96, "low_confidence": false },
        { "word": "a",        "start": 0.86, "end": 0.91, "confidence": 0.95, "low_confidence": false },
        { "word": "test",     "start": 0.92, "end": 1.20, "confidence": 0.98, "low_confidence": false },
        { "word": "of",       "start": 1.21, "end": 1.33, "confidence": 0.94, "low_confidence": false },
        { "word": "the",      "start": 1.34, "end": 1.45, "confidence": 0.96, "low_confidence": false },
        { "word": "pipeline", "start": 1.46, "end": 2.10, "confidence": 0.51, "low_confidence": true }
      ]
    }
  ],

  "words": [
    { "word": "Hello", "start": 0.00, "end": 0.42, "confidence": 0.99, "low_confidence": false },
    ...
  ]
}
```

### Key fields

| Field | Description |
|---|---|
| `transcript` | Full plain-text transcript — always present regardless of timestamp level |
| `segments[].text` | Text of each sentence or clause |
| `segments[].start/end` | Sentence-level timestamps in seconds |
| `segments[].confidence` | Mean word confidence for the segment (0–1) |
| `segments[].flagged` | `true` if Whisper detected possible silence or hallucination |
| `words[].start/end` | Word-level timestamps in seconds |
| `words[].low_confidence` | `true` if confidence < 0.6 |
| `words` (top-level) | Flat chronological list — only present for `timestamp_level` `"both"` or `"word"` |

### Timestamp levels

Control how much timing information is included via the `timestamp_level` parameter:

| Level | Segment timing | Word timing | Use when |
|---|---|---|---|
| `"both"` | ✅ | ✅ | Default — subtitles, word highlighting |
| `"segment"` | ✅ | ❌ | Simpler subtitles, speaker attribution |
| `"word"` | ✅ | ✅ | Same as `"both"` + flat word list |
| `"none"` | ❌ | ❌ | Plain transcript only |

---

## Configuration options

All configurable parameters have defaults that work well out of the box. Override them when needed:

### Whisper model

| Model | VRAM | Speed | Accuracy |
|---|---|---|---|
| `large-v3-turbo` | ~3.5 GB | ⚡⚡⚡ | ★★★★ — **default** |
| `large-v2` | ~3.5 GB | ⚡⚡ | ★★★★ |
| `large-v3` | ~4.5 GB | ⚡ | ★★★★★ |

### Quality preset

Controls Whisper's beam search. On clean speech the difference is rarely audible:

| Preset | Speed | Use when |
|---|---|---|
| `"fast"` | ⚡⚡⚡ | Prototyping, long files, clean audio |
| `"balanced"` | ⚡⚡ | **Default** — best general choice |
| `"accurate"` | ⚡ | Maximum accuracy, short files |

### Alignment model

| Model | VRAM | Notes |
|---|---|---|
| `"wav2vec2-base-960h"` | ~0.5 GB | **Default** — sufficient for clean speech |
| `"wav2vec2-large-960h-lv60"` | ~1.5 GB | Better on fast or complex speech |

### Context prompt

Always provide a prompt when you have domain context. It anchors Whisper's vocabulary to your specific audio:

```python
transcribe_chunks(
    chunks,
    initial_prompt="A product demo for an AI image generation tool called ComfyUI.",
)
```

If you have your own transcript of the audio, pass it as the prompt — Whisper will use it to guide its vocabulary choices.

---

## The user transcript path

If you already have a transcript and only need timestamps added, deep-whisper has a dedicated path for this. Whisper still runs on the audio (for acoustic grounding), but your transcript wins on vocabulary, capitalisation, and phrasing:

```python
from pipeline.reconcile  import reconcile_segments

# Run transcription as normal first
segments = transcribe_chunks(chunks, initial_prompt=your_transcript)

# Reconcile: your vocabulary + Whisper's acoustic grounding
segments = reconcile_segments(your_transcript, segments)

# Continue the pipeline as usual
segments = normalise_segments(segments)
segments = align_segments(segments, audio)
output   = build_output(segments, audio, user_transcript_provided=True)
```

**What reconciliation does:**
- Your capitalisation, hyphenation, and domain terms are preserved (`"GPT-4"`, `"Dr. Smith"`, `"Retrieval-Augmented Generation"`)
- Words Whisper heard that you didn't write are included (acoustic evidence of spoken content)
- The reconciled text is what gets aligned, so timestamps reflect the acoustic reality

---

## Architecture overview

The pipeline is split into discrete, independently testable modules. Each module has a single responsibility and clean input/output contracts:

| Module | Responsibility | Compute |
|---|---|---|
| `config.py` | All constants, thresholds, and presets | — |
| `utils.py` | Shared stateless helpers | CPU |
| `audio.py` | Audio loading, normalisation, energy analysis | CPU |
| `models.py` | Model loading and session-lifetime cache | GPU / CPU |
| `vad.py` | Voice activity detection and chunk merging | CPU |
| `transcribe.py` | Whisper transcription with rolling prompt | GPU |
| `reconcile.py` | User transcript reconciliation | CPU |
| `normalise.py` | Spoken-form text expansion | CPU |
| `align.py` | Forced CTC alignment + boundary snapping | GPU |
| `postprocess.py` | Confidence annotation and JSON assembly | CPU |

Models are loaded once and cached for the session. On a typical run, peak VRAM usage is approximately:

| Component | VRAM |
|---|---|
| `large-v3-turbo` (int8_float16) | ~3.5 GB |
| wav2vec2-base alignment | ~0.5 GB |
| PyTorch runtime | ~0.5 GB |
| **Total** | **~4.5 GB** |

This leaves ~3.5 GB of headroom on an 8 GB card — sufficient to upgrade to `wav2vec2-large` (~1.5 GB) or `large-v3` (~4.5 GB) if needed.

---

## Performance notes

- **Files of any length** are handled by splitting audio into 20–28 second chunks at natural speech pauses before transcription. VRAM usage does not increase with file length.
- **`"fast"` quality** uses greedy decoding (no beam search) and is typically 90–95% as accurate as `"accurate"` on clean speech, at significantly higher speed.
- **Model weights are downloaded on first run** (~4 GB total) and cached locally. Subsequent runs are instant.
- **Rolling prompt chaining** — the last ~224 tokens of each chunk's transcript are fed as context to the next chunk, significantly reducing errors at chunk boundaries where most transcription mistakes occur.

---

## Limitations

- **Clean speech only** — background music, overlapping speakers, and significant noise will degrade accuracy. Deep-whisper does not include audio enhancement or speaker diarisation.
- **CUDA required** — CPU inference is not supported. The pipeline is designed around GPU memory management.
- **English optimised** — the pipeline works for other languages supported by Whisper, but the text normalisation rules (`normalise.py`) are English-only. Numbers and abbreviations in other languages will not be expanded before alignment, which may reduce timestamp accuracy on numeric or abbreviated content.
- **No real-time processing** — the pipeline processes complete files, not streams.

---

## ComfyUI integration

deep-whisper is the backend package for the **ComfyUI-BDC_DeepWhisper** custom node collection, which exposes the full pipeline as composable nodes within a ComfyUI graph.

→ [ComfyUI-BDC_DeepWhisper](https://github.com/YOUR_USERNAME/ComfyUI-BDC_DeepWhisper)

If you are using deep-whisper through ComfyUI, follow the installation instructions in that repository instead.

---

## Development

### Running tests

```powershell
pip install pytest pytest-timeout
pytest tests/
```

All pipeline modules have full test coverage using stubbed GPU dependencies — tests run without a GPU.

### Environment verification

```powershell
python test_env.py
```

Verifies PyTorch CUDA access, loads all three models, and checks all dependencies.

### Editable install

```powershell
pip install -e .
```

Changes to `pipeline/` are reflected immediately without reinstalling.

### Project structure

```
deep-whisper/
├── pipeline/
│   ├── __init__.py       ← public API + run() entry point
│   ├── config.py
│   ├── utils.py
│   ├── models.py
│   ├── audio.py
│   ├── vad.py
│   ├── transcribe.py
│   ├── reconcile.py
│   ├── normalise.py
│   ├── align.py
│   └── postprocess.py
├── tests/
├── test_env.py
├── requirements.txt
├── pyproject.toml
└── install.ps1
```

---

## License and Legal

deep-whisper is released under the **Apache License 2.0** — see
[LICENSE.md](LICENSE.md) for the full text and a plain-language summary.

deep-whisper is built on several open-source projects, some of which carry
more restrictive terms. Before deploying or distributing software built on
deep-whisper, please read [LEGAL.md](LEGAL.md), which covers:

- Third-party software licenses (WhisperX, PyTorch, num2words, and others)
- Terms of use — including consent obligations and lawful use requirements
- Misuse disclaimer

---

## Acknowledgements


deep-whisper is built on the following open-source projects:

- [**faster-whisper**](https://github.com/SYSTRAN/faster-whisper) — CTranslate2 reimplementation of OpenAI Whisper
- [**WhisperX**](https://github.com/m-bain/whisperX) — forced alignment and wav2vec2 integration
- [**Silero VAD**](https://github.com/snakers4/silero-vad) — voice activity detection
- [**OpenAI Whisper**](https://github.com/openai/whisper) — the underlying speech recognition model
- [**wav2vec2**](https://huggingface.co/facebook/wav2vec2-base-960h) — Facebook AI's CTC acoustic model
- [**librosa**](https://librosa.org) — audio analysis and feature extraction
- [**num2words**](https://github.com/savoirfairelinux/num2words) — number to words conversion
