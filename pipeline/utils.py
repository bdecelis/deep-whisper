"""
deep-whisper · utils.py
=======================
Stateless helper functions shared across pipeline modules.
No pipeline logic lives here — only pure, reusable utilities.

All functions are side-effect free and independently testable.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Rolling prompt management
# ---------------------------------------------------------------------------

def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate *text* to approximately *max_tokens* Whisper tokens, keeping
    the most recent (rightmost) content.

    Uses a character-based estimate of ~4 chars per token — accurate enough
    for rolling-prompt budgeting in English and most Latin-script languages.
    The truncation point is snapped to the nearest word boundary so Whisper
    never receives a prompt that starts mid-word.

    Args:
        text:       The full rolling prompt string.
        max_tokens: Maximum token budget (e.g. config.ROLLING_PROMPT_MAX_TOKENS).

    Returns:
        The truncated string, or the original string if it is already within
        the token budget.
    """
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text

    # Keep the tail (most recent context), then snap to the first space so we
    # don't hand Whisper a prompt that starts mid-word.
    tail = text[-max_chars:]
    first_space = tail.find(" ")
    if first_space != -1:
        return tail[first_space + 1:]
    return tail


# ---------------------------------------------------------------------------
# Timestamp formatting
# ---------------------------------------------------------------------------

def seconds_to_srt_timestamp(seconds: float) -> str:
    """
    Convert a float number of seconds to SRT timestamp format.

    Converts to integer milliseconds first to avoid the floating-point edge
    case where ``round((secs % 1) * 1000)`` produces 1000 (e.g. 0.9999s),
    which would yield an invalid four-digit millisecond field.

    Example:
        >>> seconds_to_srt_timestamp(3723.456)
        '01:02:03,456'
        >>> seconds_to_srt_timestamp(0.9999)
        '00:00:01,000'

    Args:
        seconds: Non-negative float timestamp in seconds.

    Returns:
        String in the format ``HH:MM:SS,mmm``.
    """
    total_ms = round(max(0.0, seconds) * 1000)
    hours, r   = divmod(total_ms, 3_600_000)
    minutes, r = divmod(r, 60_000)
    secs, ms   = divmod(r, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def seconds_to_vtt_timestamp(seconds: float) -> str:
    """
    Convert a float number of seconds to WebVTT timestamp format.

    Identical to SRT except the sub-second separator is ``.`` not ``,``.

    Example:
        >>> seconds_to_vtt_timestamp(3723.456)
        '01:02:03.456'

    Args:
        seconds: Non-negative float timestamp in seconds.

    Returns:
        String in the format ``HH:MM:SS.mmm``.
    """
    return seconds_to_srt_timestamp(seconds).replace(",", ".")


# ---------------------------------------------------------------------------
# Confidence helpers
# ---------------------------------------------------------------------------

def safe_confidence(value: object) -> float:
    """
    Coerce *value* to a float confidence score in [0.0, 1.0].

    WhisperX and faster-whisper occasionally return ``None`` for word
    confidence on boundary words or short segments. This normalises those
    cases to 0.0 rather than propagating None through the pipeline.

    Args:
        value: A float, int, None, or anything else.

    Returns:
        A float clamped to [0.0, 1.0].
    """
    try:
        return max(0.0, min(1.0, float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def merge_word_confidences(words: list[dict]) -> float:
    """
    Compute a segment-level confidence score as the mean of its word scores.

    Args:
        words: List of word dicts, each expected to have a ``"confidence"``
               key (as produced by ``postprocess.py``).

    Returns:
        Mean confidence in [0.0, 1.0], or 0.0 for an empty list.
    """
    if not words:
        return 0.0
    return sum(safe_confidence(w.get("confidence")) for w in words) / len(words)


# ---------------------------------------------------------------------------
# Segment / word structure helpers
# ---------------------------------------------------------------------------

def flatten_words(segments: list[dict]) -> list[dict]:
    """
    Flatten the nested word lists from a list of segments into a single
    chronologically sorted list.

    Useful when ``timestamp_level == "word"`` and the caller wants a flat
    word timeline rather than sentence-grouped output.

    Args:
        segments: List of segment dicts with a ``"words"`` key, as produced
                  by ``postprocess.py``.

    Returns:
        Flat list of word dicts sorted by ``"start"`` time.
    """
    words: list[dict] = []
    for segment in segments:
        words.extend(segment.get("words", []))
    return sorted(words, key=lambda w: w.get("start", 0.0))


def words_to_text(words: list[dict]) -> str:
    """
    Join a list of word dicts into a plain text string.

    Strips leading/trailing whitespace and normalises internal spacing.

    Args:
        words: List of word dicts, each with a ``"word"`` key.

    Returns:
        Single string of space-separated words.
    """
    return " ".join(w.get("word", "").strip() for w in words if w.get("word"))
