# pipeline/__init__.py
from pipeline.audio       import load_audio, normalize_audio
from pipeline.vad         import get_speech_chunks
from pipeline.transcribe  import transcribe_chunks
from pipeline.reconcile   import reconcile_segments
from pipeline.normalise   import normalise_segments
from pipeline.align       import align_segments
from pipeline.postprocess import build_output, serialise

def run(audio_path, *, prompt="", language="en", **kwargs) -> dict:
    """Single-call entry point for the full pipeline."""
    audio    = normalize_audio(load_audio(audio_path))
    chunks   = get_speech_chunks(audio)
    segments = transcribe_chunks(chunks, initial_prompt=prompt,
                                 language=language, **kwargs)
    segments = normalise_segments(segments, language=language)
    segments = align_segments(segments, audio, language=language)
    return build_output(segments, audio, prompt=prompt,
                        language=language, **kwargs)