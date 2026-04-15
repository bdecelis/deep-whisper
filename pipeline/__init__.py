"""
deep-whisper · pipeline/__init__.py
=====================================
Public API and single-call entry point.

After pip install deep-whisper, run the GPU stack installer:

    deep-whisper-setup

This installs PyTorch (CUDA build), faster-whisper, whisperx, and cuDNN
(Windows) with all known-issue mitigations. pip install alone is not
sufficient because whisperx overwrites the CUDA torch build as a side effect.
"""

from __future__ import annotations


def run(
    audio_path:      str,
    *,
    prompt:          str  = "",
    language:        str  = "en",
    whisper_model:   str  = "large-v3-turbo",
    compute_type:    str  = "int8_float16",
    quality:         str  = "balanced",
    alignment_model: str  = "wav2vec2-base-960h",
    timestamp_level: str  = "both",
    user_transcript: str  = "",
    snap_enabled:    bool = True,
) -> dict:
    """
    Transcribe an audio file and return the deep-whisper output dict.

    Requires the GPU stack to be installed first:
        deep-whisper-setup      (after pip install deep-whisper)

    Args:
        audio_path:       Path to audio file (WAV, FLAC, MP3, etc.)
        prompt:           Context prompt to anchor Whisper's vocabulary.
        language:         ISO 639-1 language code. Default "en".
        whisper_model:    Whisper model: "large-v3-turbo", "large-v2", "large-v3".
        compute_type:     CTranslate2 type: "int8_float16", "float16", "int8".
        quality:          Beam search: "fast", "balanced", "accurate".
        alignment_model:  Alignment model label. Default "wav2vec2-base-960h".
        timestamp_level:  "both", "segment", "word", or "none".
        user_transcript:  User-provided transcript — activates the align-only
                          path where your text wins on vocabulary.
        snap_enabled:     Apply energy trough boundary snapping. Default True.

    Returns:
        deep-whisper output dict. Use pipeline.postprocess.serialise() to
        convert to a JSON string.

    Example:
        from pipeline import run
        import json

        result = run("my_audio.wav", prompt="A lecture about machine learning.")
        print(result["transcript"])
        with open("transcript.json", "w") as f:
            f.write(json.dumps(result, indent=2))
    """
    # GPU dependency check — deferred to here so that importing individual
    # pipeline modules (e.g. pipeline.normalise) never fails on a machine
    # that has deep-whisper installed but hasn't run deep-whisper-setup yet.
    _check_ready()

    from pipeline.audio       import load_audio, normalize_audio
    from pipeline.vad         import get_speech_chunks
    from pipeline.transcribe  import transcribe_chunks
    from pipeline.reconcile   import reconcile_segments
    from pipeline.normalise   import normalise_segments
    from pipeline.align       import align_segments
    from pipeline.postprocess import build_output

    audio    = normalize_audio(load_audio(audio_path))
    chunks   = get_speech_chunks(audio)

    initial_prompt = user_transcript or prompt
    segments = transcribe_chunks(
        chunks,
        initial_prompt = initial_prompt,
        model_name     = whisper_model,
        compute_type   = compute_type,
        quality        = quality,
        language       = language,
    )

    if user_transcript:
        segments = reconcile_segments(user_transcript, segments)

    segments = normalise_segments(segments, language=language)
    segments = align_segments(
        segments, audio,
        model_label  = alignment_model,
        language     = language,
        snap_enabled = snap_enabled,
    )

    return build_output(
        segments, audio,
        language                 = language,
        whisper_model            = whisper_model,
        alignment_model          = alignment_model,
        prompt                   = initial_prompt,
        user_transcript_provided = bool(user_transcript),
        timestamp_level          = timestamp_level,
    )


def _check_ready() -> None:
    """
    Verify the GPU stack is installed and CUDA is available.
    Raises RuntimeError with a clear, actionable message if not.
    """
    missing = []
    for pkg in ("faster_whisper", "whisperx"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg.replace("_", "-"))

    if missing:
        raise RuntimeError(
            f"\n\ndeep-whisper: GPU dependencies not installed: {', '.join(missing)}\n\n"
            f"  Run the setup command (available after pip install deep-whisper):\n\n"
            f"    deep-whisper-setup\n\n"
            f"  This installs faster-whisper, whisperx, and cuDNN safely\n"
            f"  without overwriting your existing PyTorch CUDA build.\n"
        )

    try:
        import torch
        if not torch.cuda.is_available():
            cuda_ver = torch.version.cuda
            if cuda_ver is None:
                raise RuntimeError(
                    f"\n\ndeep-whisper: torch has no CUDA build "
                    f"(torch.version.cuda is None).\n\n"
                    f"  whisperx likely overwrote your CUDA torch. Fix it by running:\n\n"
                    f"    deep-whisper-setup\n\n"
                    f"  This will detect your CUDA version and reinstall torch correctly.\n"
                )
    except ImportError:
        raise RuntimeError(
            f"\n\ndeep-whisper: torch is not installed.\n\n"
            f"  Run:  deep-whisper-setup\n\n"
            f"  This will install PyTorch with the correct CUDA build automatically.\n"
        )
