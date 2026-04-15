"""
deep-whisper · models.py
========================
Model loading and session-lifetime singleton cache.

All three models (Whisper, alignment, VAD) are loaded once on first request
and held in module-level state for the duration of the process. Subsequent
calls to any get_*() function return the cached instance with no I/O cost.

Cache invalidation
------------------
The Whisper and alignment caches are keyed on their configuration parameters.
If the user changes model name, compute type, or language between executions,
the affected model is reloaded automatically and the old instance released.
The VAD model is configuration-free and never needs reloading.

GPU memory
----------
All three models coexist comfortably within 8 GB VRAM (see strategy doc §5).
Models are never explicitly unloaded — they remain resident for the session.

Usage
-----
    from deep_whisper.pipeline.models import get_whisper_model, get_align_model, get_vad_model

    whisper   = get_whisper_model()
    aligner, meta = get_align_model()
    vad       = get_vad_model()
"""

from __future__ import annotations
import logging

from deep_whisper.pipeline.config import (
    WHISPER_MODEL_DEFAULT,
    COMPUTE_TYPE_DEFAULT,
    ALIGN_MODEL_DEFAULT,
    ALIGN_MODELS,
    ALIGN_MODEL_LABELS,
    LANGUAGE_DEFAULT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal cache state
# ---------------------------------------------------------------------------

# Each cache entry is a dict so we can store both the object and the
# parameters it was loaded with — allowing clean invalidation.

_whisper_cache: dict = {
    "model":        None,
    "model_name":   None,
    "compute_type": None,
}

_align_cache: dict = {
    "model":        None,
    "metadata":     None,
    "model_label":  None,
    "language":     None,
}

_vad_cache: dict = {
    "model": None,
}


# ---------------------------------------------------------------------------
# Whisper
# ---------------------------------------------------------------------------

def get_whisper_model(
    model_name:   str = WHISPER_MODEL_DEFAULT,
    compute_type: str = COMPUTE_TYPE_DEFAULT,
):
    """
    Return a cached WhisperModel, reloading only if configuration has changed.

    The first call with a given (model_name, compute_type) pair loads the
    model from disk and caches it. Subsequent calls with identical parameters
    return the cached instance immediately. If either parameter differs from
    the cached version, the old model is released and the new one loaded.

    Args:
        model_name:   faster-whisper model identifier string.
                      Must be a value from config.WHISPER_MODELS.
                      Defaults to WHISPER_MODEL_DEFAULT.
        compute_type: CTranslate2 compute type string.
                      Must be a value from config.COMPUTE_TYPES.
                      Defaults to COMPUTE_TYPE_DEFAULT.

    Returns:
        A loaded faster_whisper.WhisperModel instance on CUDA.

    Raises:
        RuntimeError: If the model cannot be loaded (e.g. CUDA unavailable,
                      model files missing).
    """
    cache = _whisper_cache

    if (
        cache["model"] is not None
        and cache["model_name"]   == model_name
        and cache["compute_type"] == compute_type
    ):
        return cache["model"]

    if cache["model"] is not None:
        logger.info(
            "Whisper model config changed (%s %s → %s %s) — reloading.",
            cache["model_name"], cache["compute_type"],
            model_name, compute_type,
        )
        cache["model"] = None   # release reference; GC will handle VRAM

    logger.info("Loading Whisper model '%s' (%s) on CUDA …", model_name, compute_type)
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(model_name, device="cuda", compute_type=compute_type)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load Whisper model '{model_name}' ({compute_type}): {exc}"
        ) from exc

    cache["model"]        = model
    cache["model_name"]   = model_name
    cache["compute_type"] = compute_type
    logger.info("Whisper model ready.")
    return model


# ---------------------------------------------------------------------------
# Alignment model
# ---------------------------------------------------------------------------

def _label_to_whisperx_key(label: str) -> str:
    """
    Resolve a human-readable alignment model label to its WhisperX key.

    ALIGN_MODEL_LABELS and ALIGN_MODELS are index-matched in config.py.
    Falls back to passing the label through unchanged so callers can also
    pass WhisperX keys directly if needed.

    Args:
        label: A value from config.ALIGN_MODEL_LABELS, or a raw WhisperX key.

    Returns:
        The corresponding WhisperX model key string.
    """
    try:
        idx = ALIGN_MODEL_LABELS.index(label)
        return ALIGN_MODELS[idx]
    except ValueError:
        return label   # already a raw key, or unknown — let WhisperX handle it


def get_align_model(
    model_label: str = ALIGN_MODEL_DEFAULT,
    language:    str = LANGUAGE_DEFAULT,
) -> tuple:
    """
    Return a cached (alignment_model, metadata) tuple, reloading if needed.

    The alignment model is language-specific — a different language code
    triggers a reload. A different model label also triggers a reload.

    Args:
        model_label: Human-readable label from config.ALIGN_MODEL_LABELS,
                     e.g. "wav2vec2-base-960h".
                     Defaults to ALIGN_MODEL_DEFAULT.
        language:    ISO 639-1 language code, e.g. "en".
                     Defaults to LANGUAGE_DEFAULT.

    Returns:
        Tuple of (alignment_model, metadata) as returned by
        whisperx.load_align_model(). Both objects are needed by
        whisperx.align() in align.py.

    Raises:
        RuntimeError: If the alignment model cannot be loaded.
    """
    cache = _align_cache

    if (
        cache["model"] is not None
        and cache["model_label"] == model_label
        and cache["language"]    == language
    ):
        return cache["model"], cache["metadata"]

    if cache["model"] is not None:
        logger.info(
            "Alignment model config changed (%s/%s → %s/%s) — reloading.",
            cache["model_label"], cache["language"],
            model_label, language,
        )
        cache["model"]    = None
        cache["metadata"] = None

    whisperx_key = _label_to_whisperx_key(model_label)
    logger.info(
        "Loading alignment model '%s' (key: %s) for language '%s' on CUDA …",
        model_label, whisperx_key, language,
    )
    try:
        import whisperx
        align_model, metadata = whisperx.load_align_model(
            language_code=language,
            device="cuda",
            model_name=whisperx_key if whisperx_key != model_label else None,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load alignment model '{model_label}' "
            f"for language '{language}': {exc}"
        ) from exc

    cache["model"]       = align_model
    cache["metadata"]    = metadata
    cache["model_label"] = model_label
    cache["language"]    = language
    logger.info("Alignment model ready.")
    return align_model, metadata


# ---------------------------------------------------------------------------
# VAD model (Silero)
# ---------------------------------------------------------------------------

def get_vad_model():
    """
    Return a cached Silero VAD model, loading it on first call.

    Silero VAD is configuration-free — there is only one model variant and
    it runs on CPU, so this cache never needs invalidation.

    Returns:
        A loaded silero_vad model instance (CPU).

    Raises:
        RuntimeError: If the VAD model cannot be loaded.
    """
    cache = _vad_cache

    if cache["model"] is not None:
        return cache["model"]

    logger.info("Loading Silero VAD model (CPU) …")
    try:
        from silero_vad import load_silero_vad
        model = load_silero_vad()
    except Exception as exc:
        raise RuntimeError(f"Failed to load Silero VAD model: {exc}") from exc

    cache["model"] = model
    logger.info("VAD model ready.")
    return model


# ---------------------------------------------------------------------------
# Utility: cache status (useful for debugging and test assertions)
# ---------------------------------------------------------------------------

def cache_status() -> dict:
    """
    Return a snapshot of what is currently loaded in each cache slot.

    Intended for logging, debugging, and test assertions — not for use
    in the main pipeline flow.

    Returns:
        Dict with keys "whisper", "alignment", "vad", each containing
        a dict of the relevant cached parameters and a boolean "loaded".
    """
    return {
        "whisper": {
            "loaded":       _whisper_cache["model"] is not None,
            "model_name":   _whisper_cache["model_name"],
            "compute_type": _whisper_cache["compute_type"],
        },
        "alignment": {
            "loaded":      _align_cache["model"] is not None,
            "model_label": _align_cache["model_label"],
            "language":    _align_cache["language"],
        },
        "vad": {
            "loaded": _vad_cache["model"] is not None,
        },
    }
