"""
deep-whisper · audio.py
=======================
Audio I/O and signal processing utilities.
No model or pipeline dependencies — pure librosa / soundfile / numpy.

All functions operate on 1-D float32 numpy arrays at SAMPLE_RATE Hz.
Use load_audio() to get audio into this canonical form before passing
it to any other pipeline module.
"""

from __future__ import annotations

import numpy as np
import librosa

from pipeline.config import (
    SAMPLE_RATE,
    BOUNDARY_SNAP_WINDOW_MS,
    BOUNDARY_SNAP_FRAME_LENGTH,
    BOUNDARY_SNAP_HOP_LENGTH,
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_audio(path: str) -> np.ndarray:
    """
    Load an audio file, downmix to mono, and resample to SAMPLE_RATE.

    Uses librosa as the primary loader — handles WAV, FLAC, OGG, MP3, AIFF,
    and most other formats via soundfile / audioread backends. Returns a
    contiguous float32 array in the range [-1.0, 1.0].

    Args:
        path: Absolute or relative path to the audio file.

    Returns:
        1-D contiguous float32 numpy array at SAMPLE_RATE Hz.

    Raises:
        FileNotFoundError: If *path* does not exist.
        RuntimeError:      If the file cannot be decoded.
    """
    try:
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, dtype=np.float32)
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed to load audio from '{path}': {exc}") from exc

    return np.ascontiguousarray(audio, dtype=np.float32)


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Peak-normalize audio to the range [-1.0, 1.0].

    If the signal is effectively silent (peak amplitude < 1e-6), the array
    is returned unchanged to avoid division by near-zero.

    Args:
        audio: 1-D float32 numpy array.

    Returns:
        Peak-normalized float32 array of the same shape.
    """
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-6:
        return audio
    return (audio / peak).astype(np.float32)


def get_duration(audio: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """
    Return the duration of an audio array in seconds.

    Args:
        audio: 1-D audio array.
        sr:    Sample rate in Hz. Defaults to SAMPLE_RATE (16 000).

    Returns:
        Duration as a float in seconds.
    """
    return len(audio) / sr


def get_energy_trough(
    audio: np.ndarray,
    approx_time_s: float,
    window_ms: int = BOUNDARY_SNAP_WINDOW_MS,
    sr: int = SAMPLE_RATE,
) -> float:
    """
    Refine a word boundary by locating the nearest local energy trough.

    After CTC alignment, word boundaries are accurate to roughly ±20–40 ms.
    This function tightens them by finding the minimum RMS energy frame
    within a small window around the approximate boundary — word boundaries
    almost always coincide with reduced acoustic energy (the brief
    silence or transition between phonemes).

    Uses BOUNDARY_SNAP_FRAME_LENGTH and BOUNDARY_SNAP_HOP_LENGTH from
    config for RMS computation, giving sub-millisecond energy resolution
    at 16 kHz.

    Args:
        audio:         Full audio array at *sr* Hz.
        approx_time_s: Approximate word boundary in seconds, as returned
                       by the alignment stage.
        window_ms:     Half-width of the search window in milliseconds.
                       Defaults to BOUNDARY_SNAP_WINDOW_MS (±40 ms).
        sr:            Sample rate. Defaults to SAMPLE_RATE (16 000).

    Returns:
        Refined boundary time in seconds. Returns *approx_time_s* unchanged
        if the window falls outside the audio or the segment is silent.
    """
    window_samples = int((window_ms / 1000) * sr)
    center_sample  = int(approx_time_s * sr)
    start = max(0, center_sample - window_samples)
    end   = min(len(audio), center_sample + window_samples)

    if end <= start:
        return approx_time_s

    segment = audio[start:end]
    energy  = librosa.feature.rms(
        y=segment,
        frame_length=BOUNDARY_SNAP_FRAME_LENGTH,
        hop_length=BOUNDARY_SNAP_HOP_LENGTH,
    )[0]

    if len(energy) == 0:
        return approx_time_s

    trough_frame  = int(np.argmin(energy))
    trough_sample = start + trough_frame * BOUNDARY_SNAP_HOP_LENGTH
    return float(trough_sample / sr)
