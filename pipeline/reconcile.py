"""
deep-whisper · reconcile.py
============================
Reconcile a user-provided transcript with Whisper's output.

When the user supplies their own transcript, Whisper still runs on the audio
to produce acoustically-grounded text. This module merges the two versions
so that the result is suitable for forced CTC alignment:

  - User's version wins on vocabulary, capitalisation, hyphenation, and
    domain-specific terminology (e.g. "Retrieval-Augmented Generation",
    "GPT-4", "Dr. Smith").
  - Whisper's version contributes acoustic grounding — words Whisper heard
    that the user did not write are included in the output.
  - If the two texts diverge so completely that alignment is impossible
    (similarity ratio < MIN_SIMILARITY_RATIO), the user's text is returned
    unchanged as the safest fallback.

The reconciled text is then passed to normalise.py before alignment.

Word-level diffing
------------------
Character-level diffs (as used by diff-match-patch natively) are too fine-
grained for our purpose — a capitalisation change appears as two separate
edits rather than a single token substitution. Instead, we operate at word
level using difflib.SequenceMatcher on normalised (lowercase, punctuation-
stripped) comparison tokens, applying results to the original word forms.

Opcodes and their handling:
  equal   → use user's token (preserves capitalisation / punctuation)
  replace → use user's token (user's vocabulary wins)
  delete  → use user's token (user-only words kept)
  insert  → use Whisper's token (Whisper heard something the user missed)

Usage
-----
    from deep_whisper.pipeline.reconcile import reconcile_segments

    updated_segments = reconcile_segments(user_text, whisper_segments)
"""

from __future__ import annotations
import difflib
import logging
import re

logger = logging.getLogger(__name__)

# If word-level similarity falls below this, skip reconciliation entirely
# and return the user's text verbatim. Prevents garbage output when Whisper
# completely misheard a segment (should be rare on clean speech).
MIN_SIMILARITY_RATIO: float = 0.3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reconcile_segments(
    user_text: str,
    segments:  list[dict],
) -> list[dict]:
    """
    Apply reconciled text to Whisper segments, preserving their timing.

    Joins the segment texts to get a full Whisper transcript, reconciles it
    against the user's transcript, then redistributes the reconciled words
    back across segments proportionally by their original character lengths.

    Timing (start / end) on each segment is never modified.

    Args:
        user_text: The user's full transcript as a plain text string.
        segments:  Whisper segment dicts as produced by transcribe.py,
                   each with at minimum "start", "end", and "text" keys.

    Returns:
        New list of segment dicts (copies) with updated "text" values.
        Returns a copy of the original segments unchanged if user_text is
        empty or all segments are empty.
    """
    if not user_text.strip():
        logger.warning("reconcile: user_text is empty — returning segments unchanged.")
        return [dict(s) for s in segments]

    if not segments:
        return []

    whisper_text = " ".join(s.get("text", "").strip() for s in segments)

    if not whisper_text.strip():
        logger.warning("reconcile: all Whisper segments are empty — returning segments unchanged.")
        return [dict(s) for s in segments]

    reconciled_text = reconcile_transcripts(user_text, whisper_text)
    reconciled_words = reconciled_text.split()

    updated_segments = _distribute_words(reconciled_words, segments)

    logger.info(
        "reconcile: %d segment(s) updated  "
        "(user words: %d  whisper words: %d  reconciled words: %d).",
        len(updated_segments),
        len(user_text.split()),
        len(whisper_text.split()),
        len(reconciled_words),
    )
    return updated_segments


def reconcile_transcripts(user_text: str, whisper_text: str) -> str:
    """
    Reconcile two transcript strings at word level, returning a single string.

    User's vocabulary and form wins for matching and substituted tokens.
    Whisper-only tokens (acoustic evidence of spoken words not in the user
    transcript) are included in the output.

    Falls back to returning *user_text* verbatim if the two texts are too
    dissimilar to reconcile reliably (word-level ratio < MIN_SIMILARITY_RATIO).

    Args:
        user_text:    The user's transcript (ground truth for vocabulary).
        whisper_text: Whisper's transcript (ground truth for what was heard).

    Returns:
        Reconciled text string.
    """
    user_text    = user_text.strip()
    whisper_text = whisper_text.strip()

    if not user_text:
        return whisper_text
    if not whisper_text:
        return user_text
    if user_text == whisper_text:
        return user_text

    u_tokens = user_text.split()
    w_tokens = whisper_text.split()
    u_norm   = [_normalize_token(t) for t in u_tokens]
    w_norm   = [_normalize_token(t) for t in w_tokens]

    matcher = difflib.SequenceMatcher(None, u_norm, w_norm, autojunk=False)
    ratio   = matcher.ratio()

    if ratio < MIN_SIMILARITY_RATIO:
        logger.warning(
            "reconcile: low similarity (%.2f < %.2f) — "
            "returning user text unchanged.",
            ratio, MIN_SIMILARITY_RATIO,
        )
        return user_text

    out: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("equal", "replace", "delete"):
            out.extend(u_tokens[i1:i2])   # user's form always wins
        elif tag == "insert":
            out.extend(w_tokens[j1:j2])   # Whisper-only words included

    return " ".join(out)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_token(token: str) -> str:
    """
    Produce a lowercase, punctuation-stripped comparison key for a word token.

    Used only for SequenceMatcher comparison — never written to output.

    Args:
        token: A single word, possibly with attached punctuation.

    Returns:
        Lowercase alphanumeric string, or empty string for punctuation-only tokens.
    """
    return re.sub(r"[^\w]", "", token.lower())


def _distribute_words(
    words:    list[str],
    segments: list[dict],
) -> list[dict]:
    """
    Distribute a flat word list across segments proportionally by original
    character length.

    Ensures every segment receives at least one word. The last segment
    always receives all remaining words so no words are lost to rounding.
    Timing values (start, end) on segments are never modified.

    Args:
        words:    Flat list of word strings to distribute.
        segments: Original segment dicts (read for char lengths and timing).

    Returns:
        List of new segment dicts (copies) with updated "text" values.
    """
    if not segments:
        return []
    if not words:
        return [dict(s) for s in segments]

    n_segs  = len(segments)
    n_words = len(words)
    char_lengths = [max(len(s.get("text", "").strip()), 1) for s in segments]
    total_chars  = sum(char_lengths)

    result: list[dict] = []
    pos = 0

    for i, seg in enumerate(segments):
        updated = dict(seg)
        if i == n_segs - 1:
            updated["text"] = " ".join(words[pos:])
        else:
            proportion    = char_lengths[i] / total_chars
            remaining_segs = n_segs - i - 1
            max_words     = n_words - pos - remaining_segs
            n             = max(1, min(round(proportion * n_words), max_words))
            updated["text"] = " ".join(words[pos : pos + n])
            pos += n
        result.append(updated)

    return result
