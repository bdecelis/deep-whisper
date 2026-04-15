"""
deep-whisper · postprocess.py
==============================
Final pipeline stage: assemble, annotate, and serialise the output.

Takes aligned segments from align.py and produces the canonical deep-whisper
output dict, ready for JSON serialisation and consumption by downstream nodes
or format converters (SRT, VTT, plain text).

Responsibilities
----------------
1. Compute segment-level confidence from word scores.
2. Apply low_confidence and flagged annotations.
3. Filter output fields according to timestamp_level.
4. Assemble the full output dict (metadata + segments).
5. Optionally serialise to a JSON string.

Output schema (timestamp_level="both")
---------------------------------------
{
    "schema_version": "1.0",
    "metadata": {
        "duration_seconds":        float,
        "language":                str,
        "whisper_model":           str,
        "alignment_model":         str,
        "prompt":                  str,
        "user_transcript_provided":bool,
        "timestamp_level":         str,
        "timestamp_utc":           str,
    },
    "segments": [
        {
            "id":             int,
            "start":          float,
            "end":            float,
            "text":           str,
            "confidence":     float,
            "flagged":        bool,
            "words": [
                {
                    "word":           str,
                    "start":          float,
                    "end":            float,
                    "confidence":     float,
                    "low_confidence": bool,
                },
                ...
            ]
        },
        ...
    ],
    "transcript": str          # always present: full plain-text join
}

timestamp_level variants
------------------------
"both"    — full schema above
"segment" — segments without "words" key
"word"    — full schema above PLUS top-level "words" flat list
"none"    — segments with "id" and "text" only; no timing or confidence

Usage
-----
    from deep_whisper.pipeline.postprocess import build_output, serialise

    output = build_output(segments, audio, ...)
    json_str = serialise(output)
"""

from __future__ import annotations
import json
import logging
from datetime import datetime, timezone

import numpy as np

from deep_whisper.pipeline.config import (
    SCHEMA_VERSION,
    TIMESTAMP_LEVEL_DEFAULT,
    TIMESTAMP_LEVELS,
    WORD_LOW_CONFIDENCE_THRESHOLD,
)
from deep_whisper.pipeline.utils import (
    merge_word_confidences,
    safe_confidence,
    flatten_words,
)
from deep_whisper.pipeline.audio import get_duration

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_output(
    segments:                list[dict],
    audio:                   np.ndarray,
    language:                str   = "en",
    whisper_model:           str   = "",
    alignment_model:         str   = "",
    prompt:                  str   = "",
    user_transcript_provided:bool  = False,
    timestamp_level:         str   = TIMESTAMP_LEVEL_DEFAULT,
) -> dict:
    """
    Assemble the final deep-whisper output dict from aligned segments.

    Computes segment-level confidence, applies low_confidence annotations,
    filters fields according to timestamp_level, and builds the complete
    output structure including metadata.

    Args:
        segments:                 Aligned segment dicts from align.py.
        audio:                    Full audio array (used for duration only).
        language:                 ISO 639-1 language code.
        whisper_model:            Model identifier string for metadata.
        alignment_model:          Alignment model label string for metadata.
        prompt:                   Seed prompt / user transcript for metadata.
        user_transcript_provided: Whether the user supplied their own text.
        timestamp_level:          One of config.TIMESTAMP_LEVELS.

    Returns:
        Output dict matching the schema described in the module docstring.
    """
    if timestamp_level not in TIMESTAMP_LEVELS:
        logger.warning(
            "postprocess: unknown timestamp_level '%s' — using '%s'.",
            timestamp_level, TIMESTAMP_LEVEL_DEFAULT,
        )
        timestamp_level = TIMESTAMP_LEVEL_DEFAULT

    processed  = [_process_segment(s) for s in segments]
    filtered   = _apply_timestamp_level(processed, timestamp_level)
    transcript = _build_transcript(processed)

    output: dict = {
        "schema_version": SCHEMA_VERSION,
        "metadata": _build_metadata(
            audio                   = audio,
            language                = language,
            whisper_model           = whisper_model,
            alignment_model         = alignment_model,
            prompt                  = prompt,
            user_transcript_provided= user_transcript_provided,
            timestamp_level         = timestamp_level,
        ),
        "segments":   filtered,
        "transcript": transcript,
    }

    # Flat word list — only present for levels that include word timestamps
    if timestamp_level in ("both", "word"):
        output["words"] = flatten_words(processed)

    logger.info(
        "postprocess: built output  "
        "segments=%d  words=%d  level=%s",
        len(filtered),
        len(output.get("words", [])),
        timestamp_level,
    )
    return output


def serialise(output: dict, indent: int = 2) -> str:
    """
    Serialise a deep-whisper output dict to a JSON string.

    Handles numpy scalar types that json.dumps would otherwise reject.

    Args:
        output: Output dict from build_output().
        indent: JSON indentation level. Defaults to 2.

    Returns:
        JSON string.
    """
    return json.dumps(output, indent=indent, default=_json_default)


# ---------------------------------------------------------------------------
# Segment processing
# ---------------------------------------------------------------------------

def _process_segment(seg: dict) -> dict:
    """
    Compute and attach segment-level confidence; clean up internal fields.

    Segment confidence is the mean of its word-level confidence scores.
    Falls back to (1.0 - no_speech_prob) when no words are available,
    giving a rough proxy derived from Whisper's own uncertainty estimate.

    The internal "no_speech_prob" field is retained for potential downstream
    use but is not part of the public output schema — _apply_timestamp_level
    will strip it along with other internal keys.

    Args:
        seg: Aligned segment dict from align.py.

    Returns:
        New segment dict with "confidence" key attached.
    """
    words = seg.get("words", [])

    if words:
        confidence = merge_word_confidences(words)
    else:
        nsp        = safe_confidence(seg.get("no_speech_prob", 0.0))
        confidence = max(0.0, 1.0 - nsp)

    result              = dict(seg)
    result["confidence"] = round(float(confidence), 4)
    return result


# ---------------------------------------------------------------------------
# Timestamp level filtering
# ---------------------------------------------------------------------------

def _apply_timestamp_level(
    segments:        list[dict],
    timestamp_level: str,
) -> list[dict]:
    """
    Strip fields from each segment according to timestamp_level.

    "both"    — id, start, end, text, confidence, flagged, words
    "segment" — id, start, end, text, confidence, flagged  (no words)
    "word"    — same as "both"  (flat top-level list added by build_output)
    "none"    — id, text only

    Internal pipeline fields (no_speech_prob) are always stripped.

    Args:
        segments:        Processed segment dicts.
        timestamp_level: One of config.TIMESTAMP_LEVELS.

    Returns:
        List of cleaned segment dicts.
    """
    result = []
    for seg in segments:
        if timestamp_level == "none":
            result.append({
                "id":   seg["id"],
                "text": seg.get("text", ""),
            })

        elif timestamp_level == "segment":
            result.append({
                "id":         seg["id"],
                "start":      seg.get("start"),
                "end":        seg.get("end"),
                "text":       seg.get("text", ""),
                "confidence": seg.get("confidence"),
                "flagged":    seg.get("flagged", False),
            })

        else:  # "both" or "word"
            result.append({
                "id":         seg["id"],
                "start":      seg.get("start"),
                "end":        seg.get("end"),
                "text":       seg.get("text", ""),
                "confidence": seg.get("confidence"),
                "flagged":    seg.get("flagged", False),
                "words":      _clean_words(seg.get("words", [])),
            })

    return result


def _clean_words(words: list[dict]) -> list[dict]:
    """
    Return word dicts containing only the public schema fields.

    Strips any internal keys that may have been added during processing.

    Args:
        words: Raw word dicts from align.py.

    Returns:
        Word dicts with only: word, start, end, confidence, low_confidence.
    """
    return [
        {
            "word":           w.get("word", ""),
            "start":          w.get("start"),
            "end":            w.get("end"),
            "confidence":     w.get("confidence"),
            "low_confidence": w.get("low_confidence", False),
        }
        for w in words
    ]


# ---------------------------------------------------------------------------
# Metadata and transcript
# ---------------------------------------------------------------------------

def _build_metadata(
    audio:                   np.ndarray,
    language:                str,
    whisper_model:           str,
    alignment_model:         str,
    prompt:                  str,
    user_transcript_provided:bool,
    timestamp_level:         str,
) -> dict:
    """
    Build the metadata dict for the output.

    Args:
        audio:                   Full audio array (for duration).
        language:                ISO 639-1 language code.
        whisper_model:           Whisper model identifier.
        alignment_model:         Alignment model label.
        prompt:                  Seed prompt or user transcript.
        user_transcript_provided:Whether user supplied their own text.
        timestamp_level:         Selected timestamp granularity.

    Returns:
        Metadata dict.
    """
    duration = round(get_duration(audio), 3) if len(audio) > 0 else 0.0
    return {
        "duration_seconds":         duration,
        "language":                 language,
        "whisper_model":            whisper_model,
        "alignment_model":          alignment_model,
        "prompt":                   prompt,
        "user_transcript_provided": user_transcript_provided,
        "timestamp_level":          timestamp_level,
        "timestamp_utc":            datetime.now(timezone.utc).strftime(
                                        "%Y-%m-%dT%H:%M:%SZ"
                                    ),
    }


def _build_transcript(segments: list[dict]) -> str:
    """
    Join all segment texts into a single plain-text transcript string.

    Always present in the output regardless of timestamp_level, providing
    a clean plain-text fallback for any consumer that doesn't need timing.

    Args:
        segments: Processed segment dicts.

    Returns:
        Single string with one space between segments.
    """
    return " ".join(
        seg.get("text", "").strip()
        for seg in segments
        if seg.get("text", "").strip()
    )


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------

def _json_default(obj):
    """
    JSON serialisation fallback for non-standard types.

    Handles numpy scalar types (int32, float32, etc.) that json.dumps
    rejects by default.

    Args:
        obj: Object that json.dumps could not serialise.

    Returns:
        A JSON-serialisable Python primitive.

    Raises:
        TypeError: For types not explicitly handled here.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")
