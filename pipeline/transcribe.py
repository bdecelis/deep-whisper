"""
deep-whisper · transcribe.py
============================
Whisper transcription with rolling prompt chaining and hallucination filtering.

Accepts the chunk list produced by vad.py and returns a flat list of segment
dicts with timestamps expressed in absolute seconds (relative to the start of
the original audio, not the start of the chunk).

Rolling prompt chaining
-----------------------
Chunk boundaries are the primary source of transcription errors because
Whisper's context window resets between chunks. To mitigate this, the last
~ROLLING_PROMPT_MAX_TOKENS of the previous chunk's transcript are fed as the
initial_prompt for the next chunk. When the user provides their own transcript
it is used to seed the first prompt, immediately anchoring Whisper's vocabulary
to the domain and speaker style.

Hallucination filtering
-----------------------
Whisper is prone to hallucinating on silence or very short segments.
faster-whisper exposes no_speech_prob per segment:
  - Segments above NO_SPEECH_HARD_THRESHOLD are dropped entirely.
  - Segments above NO_SPEECH_SOFT_THRESHOLD are kept but marked flagged=True.

Output structure
----------------
Each returned segment dict has the shape required by normalise.py and align.py:
{
    "id":            int,      # sequential, 0-based across all chunks
    "start":         float,    # absolute start in the original audio (s)
    "end":           float,    # absolute end in the original audio (s)
    "text":          str,      # raw Whisper text, including punctuation
    "no_speech_prob": float,   # raw value from faster-whisper
    "flagged":       bool,     # True if no_speech_prob > SOFT threshold
}

This module intentionally does NOT produce word-level timestamps — that is
the responsibility of align.py. Whisper's built-in word timestamps are
attention-weight based and less accurate than forced CTC alignment.

Usage
-----
    from deep_whisper.pipeline.transcribe import transcribe_chunks

    segments = transcribe_chunks(chunks, initial_prompt="context here")
"""

from __future__ import annotations
import logging
from typing import Iterator

import numpy as np

from deep_whisper.pipeline.config import (
    WHISPER_MODEL_DEFAULT,
    COMPUTE_TYPE_DEFAULT,
    QUALITY_PRESETS,
    QUALITY_DEFAULT,
    NO_SPEECH_HARD_THRESHOLD,
    NO_SPEECH_SOFT_THRESHOLD,
    ROLLING_PROMPT_MAX_TOKENS,
)
from deep_whisper.pipeline.models import get_whisper_model
from deep_whisper.pipeline.utils import truncate_to_tokens

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transcribe_chunks(
    chunks:          list[tuple[np.ndarray, float]],
    initial_prompt:  str  = "",
    model_name:      str  = WHISPER_MODEL_DEFAULT,
    compute_type:    str  = COMPUTE_TYPE_DEFAULT,
    quality:         str  = QUALITY_DEFAULT,
    language:        str  = "en",
) -> list[dict]:
    """
    Transcribe a list of audio chunks and return flat absolute-time segments.

    Iterates over each (audio_slice, start_offset_s) chunk, runs
    faster-whisper with the current rolling prompt, filters hallucinations,
    adjusts timestamps to absolute time, and accumulates results into a
    single flat list.

    Args:
        chunks:         List of (audio_slice, start_offset_s) tuples as
                        produced by vad.get_speech_chunks().
        initial_prompt: Seed text for the rolling prompt. Pass the user's
                        transcript here when available — anchors Whisper's
                        vocabulary to the domain from the first chunk.
        model_name:     Whisper model identifier. Defaults to
                        WHISPER_MODEL_DEFAULT.
        compute_type:   CTranslate2 compute type. Defaults to
                        COMPUTE_TYPE_DEFAULT.
        quality:        Quality preset name from config.QUALITY_PRESETS.
                        Defaults to QUALITY_DEFAULT.
        language:       ISO 639-1 language code. Defaults to "en".

    Returns:
        List of segment dicts with absolute timestamps, ordered
        chronologically. Empty list if no speech is found after filtering.
    """
    if not chunks:
        logger.warning("transcribe: received empty chunk list.")
        return []

    model        = get_whisper_model(model_name, compute_type)
    decode_opts  = _resolve_quality(quality)
    segments_out: list[dict] = []
    segment_id   = 0
    rolling_prompt = _prepare_initial_prompt(initial_prompt)

    for chunk_idx, (audio_slice, start_offset_s) in enumerate(chunks):
        logger.debug(
            "transcribe: chunk %d/%d  offset=%.2fs  len=%.2fs",
            chunk_idx + 1, len(chunks),
            start_offset_s, len(audio_slice) / 16_000,
        )

        raw_segments, rolling_prompt, n_dropped, n_flagged = _transcribe_one(
            model=model,
            audio=audio_slice,
            start_offset_s=start_offset_s,
            rolling_prompt=rolling_prompt,
            decode_opts=decode_opts,
            language=language,
            start_id=segment_id,
        )

        if n_dropped:
            logger.debug(
                "transcribe: chunk %d — dropped %d hallucinated segment(s).",
                chunk_idx + 1, n_dropped,
            )
        if n_flagged:
            logger.debug(
                "transcribe: chunk %d — flagged %d uncertain segment(s).",
                chunk_idx + 1, n_flagged,
            )

        segments_out.extend(raw_segments)
        segment_id += len(raw_segments)

    logger.info(
        "transcribe: %d chunk(s) → %d segment(s) retained.",
        len(chunks), len(segments_out),
    )
    return segments_out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_quality(quality: str) -> dict:
    """
    Look up decode parameters for *quality*, falling back to QUALITY_DEFAULT.

    Args:
        quality: A key in config.QUALITY_PRESETS (e.g. "fast", "balanced").

    Returns:
        Dict with keys beam_size, best_of, temperature.
    """
    if quality not in QUALITY_PRESETS:
        logger.warning(
            "transcribe: unknown quality '%s' — using '%s'.",
            quality, QUALITY_DEFAULT,
        )
        return QUALITY_PRESETS[QUALITY_DEFAULT]
    return QUALITY_PRESETS[quality]


def _prepare_initial_prompt(prompt: str) -> str:
    """
    Truncate the user-supplied seed prompt to the rolling prompt budget.

    This ensures a very long user transcript doesn't flood the entire prompt
    window, leaving no room for chunk-to-chunk context.

    Args:
        prompt: Raw prompt string from the caller.

    Returns:
        Truncated prompt string within ROLLING_PROMPT_MAX_TOKENS.
    """
    return truncate_to_tokens(prompt.strip(), ROLLING_PROMPT_MAX_TOKENS)


def _transcribe_one(
    model,
    audio:          np.ndarray,
    start_offset_s: float,
    rolling_prompt: str,
    decode_opts:    dict,
    language:       str,
    start_id:       int,
) -> tuple[list[dict], str, int, int]:
    """
    Transcribe a single audio chunk and return filtered, offset-adjusted segments.

    Runs faster-whisper on *audio*, applies hallucination filtering, adjusts
    all timestamps by *start_offset_s*, and updates the rolling prompt.

    Args:
        model:          Loaded WhisperModel from models.get_whisper_model().
        audio:          1-D float32 audio array for this chunk.
        start_offset_s: Absolute start time of this chunk in the source audio.
        rolling_prompt: Current rolling prompt string (previous chunk tail).
        decode_opts:    Dict of beam_size / best_of / temperature from quality preset.
        language:       ISO 639-1 language code.
        start_id:       Starting segment id for this chunk (for sequential IDs).

    Returns:
        Tuple of:
          - list[dict]:  Filtered segments with absolute timestamps.
          - str:         Updated rolling prompt (for the next chunk).
          - int:         Number of segments hard-dropped (hallucinations).
          - int:         Number of segments soft-flagged (uncertain).
    """
    raw_iter: Iterator = model.transcribe(
        audio,
        language=language,
        initial_prompt=rolling_prompt or None,
        beam_size=decode_opts["beam_size"],
        best_of=decode_opts["best_of"],
        temperature=decode_opts["temperature"],
        word_timestamps=False,   # alignment stage handles this more accurately
    )

    # faster-whisper returns a generator + TranscriptionInfo tuple
    # Unpack carefully — the iterator must be consumed before we can read info
    segments_gen, _info = raw_iter

    kept:    list[dict] = []
    dropped: int = 0
    flagged: int = 0
    chunk_texts: list[str] = []

    for seg in segments_gen:
        nsp = float(getattr(seg, "no_speech_prob", 0.0))

        # Hard filter — almost certainly silence or hallucination
        if nsp > NO_SPEECH_HARD_THRESHOLD:
            dropped += 1
            continue

        is_flagged = nsp > NO_SPEECH_SOFT_THRESHOLD
        if is_flagged:
            flagged += 1

        kept.append({
            "id":             start_id + len(kept),
            "start":          round(float(seg.start) + start_offset_s, 4),
            "end":            round(float(seg.end)   + start_offset_s, 4),
            "text":           seg.text.strip(),
            "no_speech_prob": round(nsp, 4),
            "flagged":        is_flagged,
        })
        chunk_texts.append(seg.text)

    # Update rolling prompt: append this chunk's text, then trim to budget
    chunk_transcript = " ".join(chunk_texts)
    updated_prompt   = truncate_to_tokens(
        (rolling_prompt + " " + chunk_transcript).strip(),
        ROLLING_PROMPT_MAX_TOKENS,
    )

    return kept, updated_prompt, dropped, flagged
