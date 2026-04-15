"""
deep-whisper · config.py
========================
Single source of truth for all constants, thresholds, presets, and defaults.
No other module hardcodes a number or model name — everything lives here.

Import pattern:
    from deep_whisper.pipeline.config import SAMPLE_RATE, QUALITY_PRESETS, ...
"""

from typing import Final

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

# Whisper and wav2vec2 both expect 16 kHz mono float32
SAMPLE_RATE: Final[int] = 16_000


# ---------------------------------------------------------------------------
# VAD & chunking
# ---------------------------------------------------------------------------

# Whisper accuracy sweet spot is 20–30 s. Chunks are greedily merged up to
# this limit, then split at the next natural pause.
VAD_CHUNK_MAX_S:    Final[float] = 28.0

# Segments shorter than this are merged into an adjacent segment rather than
# processed alone (too little context degrades Whisper accuracy).
VAD_SEGMENT_MIN_S:  Final[float] = 2.0

# Pauses longer than this always trigger a chunk split, regardless of the
# current window length. Preserves natural sentence/clause boundaries.
VAD_PAUSE_SPLIT_S:  Final[float] = 1.5


# ---------------------------------------------------------------------------
# Whisper model options
# ---------------------------------------------------------------------------

WHISPER_MODELS: Final[list[str]] = [
    "large-v3-turbo",   # Recommended default — distilled, ~6× large-v3 speed
    "large-v2",         # Alternative if turbo underperforms on target audio
    "large-v3",         # Maximum accuracy fallback; ~4.5 GB VRAM
]
WHISPER_MODEL_DEFAULT: Final[str] = "large-v3-turbo"

# CTranslate2 compute types. int8_float16 is the best balance for 8 GB VRAM:
# weights stored as int8 (small), activations as float16 (accurate).
COMPUTE_TYPES: Final[list[str]] = [
    "int8_float16",     # Recommended — best speed/accuracy on 8 GB
    "float16",          # Higher accuracy, higher VRAM usage
    "int8",             # Lowest VRAM, slight accuracy trade-off
]
COMPUTE_TYPE_DEFAULT: Final[str] = "int8_float16"


# ---------------------------------------------------------------------------
# Quality presets  (beam search parameters)
# ---------------------------------------------------------------------------
# On clean, noise-free speech, greedy decoding (beam_size=1) is typically
# 90–95 % as accurate as beam_size=5 at meaningfully higher speed. Expose
# all three tiers and let the user decide.

QUALITY_PRESETS: Final[dict] = {
    "fast": {
        "beam_size":   1,
        "best_of":     1,
        "temperature": 0.0,
    },
    "balanced": {
        "beam_size":   3,
        "best_of":     1,
        "temperature": 0.0,
    },
    "accurate": {
        "beam_size":   5,
        "best_of":     1,
        "temperature": 0.0,
    },
}
QUALITY_DEFAULT: Final[str] = "balanced"


# ---------------------------------------------------------------------------
# Hallucination & confidence thresholds
# ---------------------------------------------------------------------------

# Segments with no_speech_prob above this are almost certainly silence or
# hallucination and are dropped entirely before alignment.
NO_SPEECH_HARD_THRESHOLD: Final[float] = 0.8

# Segments with no_speech_prob above this (but below HARD) are included in
# output but marked flagged=true for downstream consumers to handle.
NO_SPEECH_SOFT_THRESHOLD: Final[float] = 0.4

# Individual words with confidence below this are marked low_confidence=true.
WORD_LOW_CONFIDENCE_THRESHOLD: Final[float] = 0.6


# ---------------------------------------------------------------------------
# Rolling prompt
# ---------------------------------------------------------------------------

# Maximum number of tokens carried forward as context when chaining Whisper
# calls across chunks. Whisper's context window is 448 tokens; 224 leaves
# comfortable headroom for the new chunk.
ROLLING_PROMPT_MAX_TOKENS: Final[int] = 224


# ---------------------------------------------------------------------------
# Alignment model options
# ---------------------------------------------------------------------------

# wav2vec2-base is fast and sufficient for clean speech.
# wav2vec2-large gives ~2× better timestamp accuracy on fast or complex
# speech at ~3× the VRAM cost (~1.5 GB vs ~0.5 GB).
ALIGN_MODELS: Final[list[str]] = [
    "WAV2VEC2_ASR_BASE_960H",   # maps to wav2vec2-base-960h in WhisperX
    "WAV2VEC2_ASR_LARGE_LV60_SELFTRAINING_960H",   # wav2vec2-large-960h-lv60
]

# Human-readable labels for the ComfyUI COMBO input (parallel to ALIGN_MODELS)
ALIGN_MODEL_LABELS: Final[list[str]] = [
    "wav2vec2-base-960h",
    "wav2vec2-large-960h-lv60",
]
ALIGN_MODEL_DEFAULT: Final[str] = "wav2vec2-base-960h"


# ---------------------------------------------------------------------------
# Energy-based boundary snapping
# ---------------------------------------------------------------------------

# After CTC alignment, word boundaries are snapped to the nearest local
# energy trough within this window. Tightens timestamps by ~10–30 ms on
# typical speech without any model inference cost.
BOUNDARY_SNAP_WINDOW_MS: Final[int] = 40

# RMS frame and hop lengths used when computing the energy envelope for
# snapping. Short frames preserve temporal resolution at 16 kHz.
BOUNDARY_SNAP_FRAME_LENGTH: Final[int] = 64
BOUNDARY_SNAP_HOP_LENGTH:   Final[int] = 16


# ---------------------------------------------------------------------------
# Timestamp output levels
# ---------------------------------------------------------------------------

TIMESTAMP_LEVELS: Final[list[str]] = [
    "both",       # word-level timestamps nested inside sentence segments
    "segment",    # sentence-level only
    "word",       # flat word list only
    "none",       # plain text transcript, no timestamps
]
TIMESTAMP_LEVEL_DEFAULT: Final[str] = "both"


# ---------------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------------

LANGUAGE_DEFAULT: Final[str] = "en"


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

# Increment the minor version when the schema gains new optional fields.
# Increment the major version on any breaking change.
SCHEMA_VERSION: Final[str] = "1.0"
