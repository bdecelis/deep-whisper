"""
deep-whisper · vad.py
=====================
Voice Activity Detection and smart chunk merging.

Runs Silero VAD over the full audio to find speech segments, then applies
a greedy merge pass to produce chunks sized for optimal Whisper performance
(20–28 s). The result is a list of (audio_slice, start_offset_s) tuples
that downstream modules treat as the canonical unit of work.

Chunk contract
--------------
Every yielded chunk is:
  - A 1-D contiguous float32 numpy array at SAMPLE_RATE Hz
  - Between VAD_SEGMENT_MIN_S and VAD_CHUNK_MAX_S seconds long*
  - Aligned to a natural speech pause wherever possible
  - Accompanied by its absolute start offset in the original audio (seconds)

  * Edge case: a single VAD segment longer than VAD_CHUNK_MAX_S is yielded
    as-is rather than hard-cut mid-word. Whisper handles over-long segments
    gracefully; hard cuts do not.

Usage
-----
    from deep_whisper.pipeline.vad import get_speech_chunks

    chunks = get_speech_chunks(audio)
    for audio_slice, start_s in chunks:
        ...
"""

from __future__ import annotations
import logging

import numpy as np

from deep_whisper.pipeline.config import (
    SAMPLE_RATE,
    VAD_CHUNK_MAX_S,
    VAD_SEGMENT_MIN_S,
    VAD_PAUSE_SPLIT_S,
)
from deep_whisper.pipeline.models import get_vad_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_speech_chunks(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> list[tuple[np.ndarray, float]]:
    """
    Detect speech in *audio* and return a list of optimally-sized chunks.

    Pipeline:
      1. Run Silero VAD → raw speech segment timestamps
      2. Filter out sub-minimum segments (< VAD_SEGMENT_MIN_S)
      3. Greedy-merge adjacent segments into chunks up to VAD_CHUNK_MAX_S,
         splitting at pauses longer than VAD_PAUSE_SPLIT_S regardless
      4. Slice the audio array and package each chunk with its start offset

    Args:
        audio:       1-D float32 array at *sample_rate* Hz, as returned by
                     audio.load_audio().
        sample_rate: Sample rate of *audio*. Defaults to SAMPLE_RATE.

    Returns:
        List of (audio_slice, start_offset_s) tuples, ordered chronologically.
        Returns an empty list if no speech is detected.
    """
    if len(audio) == 0:
        logger.warning("vad: received empty audio array — returning no chunks.")
        return []

    raw_segments = _run_vad(audio, sample_rate)
    logger.info("vad: %d raw speech segments detected.", len(raw_segments))

    if not raw_segments:
        logger.warning("vad: no speech detected in audio.")
        return []

    merged = _merge_segments(raw_segments)
    logger.info("vad: %d chunks after merge.", len(merged))

    chunks = _slice_audio(audio, merged, sample_rate)
    return chunks


# ---------------------------------------------------------------------------
# Step 1 — VAD
# ---------------------------------------------------------------------------

def _run_vad(
    audio: np.ndarray,
    sample_rate: int,
) -> list[dict]:
    """
    Run Silero VAD and return a list of raw speech segment dicts.

    Each dict has keys:
        "start" (float): segment start in seconds
        "end"   (float): segment end in seconds

    Args:
        audio:       1-D float32 audio array.
        sample_rate: Sample rate of *audio*.

    Returns:
        List of segment dicts, sorted by start time.
    """
    from silero_vad import get_speech_timestamps

    model = get_vad_model()

    # get_speech_timestamps expects a torch tensor
    import torch
    audio_tensor = torch.from_numpy(audio)

    raw = get_speech_timestamps(
        audio_tensor,
        model,
        sampling_rate=sample_rate,
        return_seconds=True,
    )

    # Normalise to plain dicts with float values
    segments = [
        {"start": float(s["start"]), "end": float(s["end"])}
        for s in raw
    ]
    return sorted(segments, key=lambda s: s["start"])


# ---------------------------------------------------------------------------
# Step 2 & 3 — Merge logic
# ---------------------------------------------------------------------------

def _merge_segments(
    segments: list[dict],
) -> list[dict]:
    """
    Merge VAD segments into Whisper-optimal chunks using greedy windowing.

    Rules applied in order:
      1. Segments shorter than VAD_SEGMENT_MIN_S are absorbed into an
         adjacent segment (prefer the following segment; fall back to
         the preceding one). Applied before windowing.
      2. Segments are greedily merged into a growing window. The window
         closes (chunk is emitted) when:
           a. Adding the next segment would exceed VAD_CHUNK_MAX_S, OR
           b. The pause between the current end and the next start
              exceeds VAD_PAUSE_SPLIT_S.
      3. A segment that is itself longer than VAD_CHUNK_MAX_S is emitted
         as its own chunk without splitting.

    Args:
        segments: List of {"start": float, "end": float} dicts,
                  sorted by start time.

    Returns:
        List of merged chunk dicts with the same structure.
    """
    if not segments:
        return []

    # --- Pass 1: absorb sub-minimum segments --------------------------------
    filtered = _absorb_short_segments(segments)

    if not filtered:
        return []

    # --- Pass 2: greedy window merge ----------------------------------------
    chunks: list[dict] = []
    window_start = filtered[0]["start"]
    window_end   = filtered[0]["end"]

    for seg in filtered[1:]:
        seg_duration   = seg["end"] - seg["start"]
        window_duration = seg["end"] - window_start
        pause_before    = seg["start"] - window_end

        force_split = pause_before >= VAD_PAUSE_SPLIT_S
        would_overflow = window_duration > VAD_CHUNK_MAX_S

        if force_split or would_overflow:
            # Emit current window as a chunk
            chunks.append({"start": window_start, "end": window_end})
            window_start = seg["start"]
            window_end   = seg["end"]
        else:
            # Extend the window to include this segment
            window_end = seg["end"]

    # Emit the final window
    chunks.append({"start": window_start, "end": window_end})

    return chunks


def _absorb_short_segments(
    segments: list[dict],
) -> list[dict]:
    """
    Remove segments shorter than VAD_SEGMENT_MIN_S by merging them into
    an adjacent segment.

    Preference order:
      - Merge into the NEXT segment (extend its start backwards), if one exists
      - Otherwise merge into the PREVIOUS segment (extend its end forwards)
      - If neither exists (single isolated short segment), keep it as-is
        rather than silently dropping audio

    Args:
        segments: List of segment dicts, sorted by start time.

    Returns:
        Filtered/merged list of segment dicts.
    """
    if not segments:
        return []

    result = [dict(s) for s in segments]   # work on a copy
    i = 0
    while i < len(result):
        duration = result[i]["end"] - result[i]["start"]
        if duration < VAD_SEGMENT_MIN_S:
            if i + 1 < len(result):
                # Merge into next: pull next segment's start back
                result[i + 1]["start"] = result[i]["start"]
                result.pop(i)
                # Don't advance i — the merged segment at position i
                # might itself now be below the threshold
            elif i > 0:
                # Merge into previous: push previous segment's end forward
                result[i - 1]["end"] = result[i]["end"]
                result.pop(i)
                i = max(0, i - 1)
            else:
                # Only segment and it's short — keep it
                i += 1
        else:
            i += 1

    return result


# ---------------------------------------------------------------------------
# Step 4 — Slice audio
# ---------------------------------------------------------------------------

def _slice_audio(
    audio: np.ndarray,
    chunks: list[dict],
    sample_rate: int,
) -> list[tuple[np.ndarray, float]]:
    """
    Convert a list of time-range dicts into (audio_slice, start_s) tuples.

    Args:
        audio:       Full audio array at *sample_rate* Hz.
        chunks:      List of {"start": float, "end": float} dicts.
        sample_rate: Sample rate of *audio*.

    Returns:
        List of (audio_slice, start_offset_s) tuples.
        Each audio_slice is a contiguous float32 array.
    """
    result = []
    total_samples = len(audio)

    for chunk in chunks:
        start_sample = int(chunk["start"] * sample_rate)
        end_sample   = min(int(chunk["end"] * sample_rate), total_samples)

        if start_sample >= end_sample:
            logger.debug("vad: skipping zero-length chunk at %.3fs", chunk["start"])
            continue

        sliced = np.ascontiguousarray(audio[start_sample:end_sample], dtype=np.float32)
        result.append((sliced, float(chunk["start"])))

    return result
