"""
deep-whisper · align.py
=======================
Forced CTC alignment and energy-based word boundary refinement.

Takes normalised segment dicts (from normalise.py) and the full original
audio array, and returns segments enriched with precise word-level timestamps.

Two-stage refinement
--------------------
1. WhisperX forced alignment (wav2vec2 / MMS CTC model)
   Aligns each segment's text to its audio window. Accurate to ±20–40 ms —
   better than Whisper's attention-weight timestamps, which are the baseline.

2. Energy-based boundary snapping (audio.get_energy_trough)
   Tightens each word's start and end to the nearest local RMS energy trough
   within a ±BOUNDARY_SNAP_WINDOW_MS window. Word boundaries almost always
   coincide with reduced acoustic energy. Costs nothing — pure numpy.

Words without timestamps
------------------------
WhisperX occasionally fails to align individual words (very short words,
homophones, or low-confidence regions). These words are included in the
output with start/end copied from the surrounding context (nearest
aligned neighbour), marked with confidence 0.0 and low_confidence=True.
The segment-level timestamps are not affected.

Usage
-----
    from pipeline.align import align_segments

    aligned = align_segments(segments, audio)
"""

from __future__ import annotations
import logging

import numpy as np

from pipeline.config import (
    ALIGN_MODEL_DEFAULT,
    LANGUAGE_DEFAULT,
    WORD_LOW_CONFIDENCE_THRESHOLD,
)
from pipeline.models import get_align_model
from pipeline.utils import safe_confidence
from pipeline.audio import get_energy_trough

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def align_segments(
    segments:     list[dict],
    audio:        np.ndarray,
    model_label:  str  = ALIGN_MODEL_DEFAULT,
    language:     str  = LANGUAGE_DEFAULT,
    snap_enabled: bool = True,
) -> list[dict]:
    """
    Align segment text to audio and return word-level timestamps.

    Runs WhisperX forced CTC alignment, then optionally refines each word
    boundary using acoustic energy trough snapping.

    Args:
        segments:     Normalised segment dicts from normalise.py.
                      Each must have "text", "start", "end" keys.
        audio:        Full original audio as 1-D float32 array at SAMPLE_RATE.
                      Must span the full duration covered by all segments.
        model_label:  Human-readable alignment model label from
                      config.ALIGN_MODEL_LABELS. Defaults to ALIGN_MODEL_DEFAULT.
        language:     ISO 639-1 language code. Defaults to LANGUAGE_DEFAULT.
        snap_enabled: Whether to apply energy trough boundary snapping.
                      Defaults to True. Disable for fast/debug runs.

    Returns:
        List of segment dicts (copies of inputs) enriched with a "words" key.
        Each word dict has: word, start, end, confidence, low_confidence.
        Segments that fail alignment are returned with an empty "words" list
        rather than raising an exception.
    """
    if not segments:
        logger.warning("align: received empty segment list.")
        return []

    if len(audio) == 0:
        logger.warning("align: received empty audio — returning segments without word timestamps.")
        return [_segment_without_words(s) for s in segments]

    align_model, metadata = get_align_model(model_label, language)

    raw_result = _run_alignment(segments, audio, align_model, metadata)

    enriched = _build_output(
        raw_result   = raw_result,
        orig_segments = segments,
        audio         = audio,
        snap_enabled  = snap_enabled,
    )

    total_words = sum(len(s.get("words", [])) for s in enriched)
    logger.info(
        "align: %d segment(s), %d word(s) aligned.",
        len(enriched), total_words,
    )
    return enriched


# ---------------------------------------------------------------------------
# Step 1 — WhisperX alignment
# ---------------------------------------------------------------------------

def _run_alignment(
    segments:    list[dict],
    audio:       np.ndarray,
    align_model,
    metadata:    dict,
) -> dict:
    """
    Call whisperx.align() and return its raw result dict.

    Falls back to an empty result dict on failure so the pipeline can
    continue rather than crashing on a single bad segment.

    Args:
        segments:    Normalised segment list.
        audio:       Full audio array.
        align_model: Loaded wav2vec2 / MMS model.
        metadata:    Language metadata dict from whisperx.load_align_model().

    Returns:
        WhisperX result dict with a "segments" key, or {"segments": []} on
        failure.
    """
    try:
        import whisperx
        result = whisperx.align(
            segments,
            align_model,
            metadata,
            audio,
            device="cuda",
            return_char_alignments=False,
        )
        return result
    except Exception as exc:
        logger.error("align: whisperx.align() failed — %s", exc)
        return {"segments": []}


# ---------------------------------------------------------------------------
# Step 2 — Build enriched output
# ---------------------------------------------------------------------------

def _build_output(
    raw_result:    dict,
    orig_segments: list[dict],
    audio:         np.ndarray,
    snap_enabled:  bool,
) -> list[dict]:
    """
    Merge WhisperX word timestamps back onto the original segment dicts.

    WhisperX may return fewer segments than were passed in (e.g. if a
    segment was empty after normalisation). Original segments that have
    no corresponding WhisperX output are returned with empty word lists.

    Args:
        raw_result:    Return value of whisperx.align().
        orig_segments: Original segment dicts passed into alignment.
        audio:         Full audio array (used for boundary snapping).
        snap_enabled:  Whether to apply energy trough snapping.

    Returns:
        List of enriched segment dicts.
    """
    # Index aligned segments by their start time for O(1) lookup
    aligned_by_start: dict[float, dict] = {}
    for seg in raw_result.get("segments", []):
        start_key = round(float(seg.get("start", 0.0)), 3)
        aligned_by_start[start_key] = seg

    result = []
    for seg in orig_segments:
        start_key = round(float(seg.get("start", 0.0)), 3)
        aligned   = aligned_by_start.get(start_key)

        if aligned is None:
            logger.debug(
                "align: no WhisperX result for segment at %.3fs — "
                "returning without word timestamps.",
                seg.get("start", 0.0),
            )
            result.append(_segment_without_words(seg))
            continue

        raw_words  = aligned.get("words", [])
        word_dicts = _extract_words(raw_words, seg)

        if snap_enabled and len(audio) > 0:
            word_dicts = _snap_boundaries(word_dicts, audio)

        enriched = dict(seg)
        enriched["words"] = word_dicts
        result.append(enriched)

    return result


def _extract_words(
    raw_words:   list[dict],
    parent_seg:  dict,
) -> list[dict]:
    """
    Convert WhisperX word dicts to the deep-whisper word schema.

    Words missing start/end (alignment failure) receive timestamps
    interpolated from their neighbours; confidence is set to 0.0 and
    low_confidence to True.

    WhisperX word schema:
        {"word": str, "start": float|missing, "end": float|missing, "score": float}

    Output word schema:
        {"word": str, "start": float, "end": float,
         "confidence": float, "low_confidence": bool}

    Args:
        raw_words:  Raw word list from WhisperX.
        parent_seg: Parent segment dict (provides fallback start/end times).

    Returns:
        List of word dicts in the deep-whisper schema.
    """
    if not raw_words:
        return []

    # First pass: extract what WhisperX gave us
    words: list[dict] = []
    for w in raw_words:
        conf = safe_confidence(w.get("score"))
        words.append({
            "word":           w.get("word", "").strip(),
            "start":          w.get("start"),    # may be None
            "end":            w.get("end"),       # may be None
            "confidence":     conf,
            "low_confidence": conf < WORD_LOW_CONFIDENCE_THRESHOLD,
        })

    # Second pass: fill missing timestamps by interpolation
    seg_start = float(parent_seg.get("start", 0.0))
    seg_end   = float(parent_seg.get("end",   0.0))
    words = _fill_missing_timestamps(words, seg_start, seg_end)

    return words


def _fill_missing_timestamps(
    words:     list[dict],
    seg_start: float,
    seg_end:   float,
) -> list[dict]:
    """
    Fill None timestamps by interpolating between known neighbours.

    Strategy:
      - Scan forward to find the next word with a known start.
      - Distribute the gap evenly across words with missing timestamps.
      - Words before the first known timestamp use seg_start as anchor.
      - Words after the last known timestamp use seg_end as anchor.

    Args:
        words:     List of word dicts, possibly with None start/end.
        seg_start: Fallback start time from the parent segment (s).
        seg_end:   Fallback end time from the parent segment (s).

    Returns:
        Same list with all None timestamps replaced by float values.
        Words filled by interpolation have low_confidence forced to True.
    """
    n = len(words)
    if n == 0:
        return words

    # Replace None with sentinel so we can do numeric ops
    for w in words:
        if w["start"] is None:
            w["start"] = float("nan")
        if w["end"] is None:
            w["end"] = float("nan")
        else:
            w["end"] = float(w["end"])
        w["start"] = float(w["start"])

    import math

    # Forward pass: fill missing start/end by linear interpolation
    # Collect anchor points: (index, time) for known starts
    anchors = [(i, w["start"]) for i, w in enumerate(words)
               if not math.isnan(w["start"])]

    # Add boundary anchors
    anchors = [(-1, seg_start)] + anchors + [(n, seg_end)]

    for k in range(len(anchors) - 1):
        left_idx,  left_t  = anchors[k]
        right_idx, right_t = anchors[k + 1]

        # Fill words between left and right anchors
        gap_count = right_idx - left_idx - 1
        if gap_count <= 0:
            continue

        step = (right_t - left_t) / (gap_count + 1)
        for j, idx in enumerate(range(left_idx + 1, right_idx), start=1):
            if math.isnan(words[idx]["start"]):
                words[idx]["start"]          = round(left_t + j * step, 4)
                words[idx]["low_confidence"] = True  # interpolated

    # Fill missing end times: use next word's start, or seg_end for last
    for i, w in enumerate(words):
        if math.isnan(w["end"]):
            if i + 1 < n:
                w["end"]          = words[i + 1]["start"]
            else:
                w["end"]          = seg_end
            w["low_confidence"]   = True
        w["start"] = round(float(w["start"]), 4)
        w["end"]   = round(float(w["end"]),   4)

    return words


# ---------------------------------------------------------------------------
# Step 3 — Energy boundary snapping
# ---------------------------------------------------------------------------

def _snap_boundaries(
    words: list[dict],
    audio: np.ndarray,
) -> list[dict]:
    """
    Refine word start and end times using acoustic energy trough snapping.

    Calls get_energy_trough() for each word boundary. If the refined time
    would invert the word's start/end ordering, the original value is kept.

    Args:
        words: Word dicts with float start/end times.
        audio: Full audio array.

    Returns:
        Words with tightened start/end timestamps (in-place update of copies).
    """
    snapped = []
    for w in words:
        word = dict(w)
        orig_start = word["start"]
        orig_end   = word["end"]

        snapped_start = get_energy_trough(audio, orig_start)
        snapped_end   = get_energy_trough(audio, orig_end)

        # Guard: never let snapping invert or collapse a word window
        if snapped_start < snapped_end:
            word["start"] = round(snapped_start, 4)
            word["end"]   = round(snapped_end,   4)

        snapped.append(word)

    return snapped


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _segment_without_words(seg: dict) -> dict:
    """Return a copy of *seg* with an empty "words" list added."""
    result       = dict(seg)
    result["words"] = []
    return result
