"""
deep-whisper · normalise.py
============================
Expand numbers, abbreviations, and symbols to their spoken form.

The CTC alignment model (wav2vec2) matches phonemes — if the text contains
"$200" but the speaker said "two hundred dollars", the aligner silently
drifts. This module closes that gap by converting text to the form most
likely to match what was actually spoken.

Both pipeline paths converge here: Whisper output and user-provided (or
reconciled) transcripts both pass through before reaching align.py.

Substitution order
------------------
Order is load-bearing. Each step assumes previous steps have already fired.

  1. Acronym-number compounds   "GPT-4"      → "GPT four"
  2. Titles & abbreviations     "Dr."        → "Doctor"
  3. Currency                   "$200"       → "two hundred dollars"
                                "$3.50"      → "three dollars fifty cents"
  4. Percentages                "15%"        → "fifteen percent"
  5. Ordinals                   "1st"        → "first"
  6. Years (1800–2099)          "2024"       → "twenty twenty-four"
  7. Remaining numbers          "3.14"       → "three point one four"
                                "1,000,000"  → "one million"
  8. Acronyms                   "AI"         → "A I"
  9. Whitespace cleanup

Usage
-----
    from deep_whisper.pipeline.normalise import normalise_segments, normalise_text

    segments = normalise_segments(segments, language="en")
    text     = normalise_text("Dr. Smith used GPT-4", language="en")
"""

from __future__ import annotations
import logging
import re

import num2words as nw

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Title / abbreviation map
# ---------------------------------------------------------------------------
# Keys are raw strings (not regex) — the expansion function builds word-
# boundary regexes from them. Ordered longest-first so "Mrs." doesn't match
# before "Mr." on a two-pass scan.

TITLE_MAP: dict[str, str] = {
    "Prof.":  "Professor",
    "Mrs.":   "Missus",
    "Mr.":    "Mister",
    "Ms.":    "Miss",
    "Dr.":    "Doctor",
    "Rev.":   "Reverend",
    "Sgt.":   "Sergeant",
    "Cpl.":   "Corporal",
    "Lt.":    "Lieutenant",
    "Capt.":  "Captain",
    "Cmdr.":  "Commander",
    "Col.":   "Colonel",
    "Gen.":   "General",
    "St.":    "Saint",
    "vs.":    "versus",
    "etc.":   "et cetera",
    "e.g.":   "for example",
    "i.e.":   "that is",
}

# Currency symbols to (singular, plural, cent-singular, cent-plural)
CURRENCY_MAP: dict[str, tuple[str, str, str, str]] = {
    "$":  ("dollar",  "dollars",  "cent",    "cents"),
    "£":  ("pound",   "pounds",   "penny",   "pence"),
    "€":  ("euro",    "euros",    "cent",    "cents"),
    "¥":  ("yen",     "yen",      "sen",     "sen"),
}

# Year range treated as a year rather than a cardinal number
YEAR_MIN: int = 1800
YEAR_MAX: int = 2099


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalise_segments(
    segments: list[dict],
    language: str = "en",
) -> list[dict]:
    """
    Apply spoken-form normalisation to each segment's text field.

    All other segment fields (start, end, id, flagged, …) are preserved
    unchanged. Returns new dicts — original segments are not mutated.

    Args:
        segments: List of segment dicts, each with at minimum a "text" key.
        language: BCP-47 / ISO 639-1 language code for num2words.
                  Currently only "en" has full rule coverage; other codes
                  are passed through to num2words for numeric conversion
                  but the title/acronym rules are English-only.

    Returns:
        New list of segment dicts with normalised "text" values.
    """
    result = []
    for seg in segments:
        updated = dict(seg)
        raw     = seg.get("text", "")
        updated["text"] = normalise_text(raw, language=language)
        result.append(updated)
    return result


def normalise_text(text: str, language: str = "en") -> str:
    """
    Expand numbers, abbreviations, and symbols to their spoken form.

    Applies all substitution rules in the correct order. Safe to call on
    an already-normalised string — idempotent for fully-textual input.

    Args:
        text:     Input text, possibly containing numerals, symbols, etc.
        language: Language code forwarded to num2words for numeric conversion.

    Returns:
        Text with all expandable tokens replaced by their spoken equivalents.
    """
    if not text.strip():
        return text

    text = _expand_acronym_numbers(text)
    text = _expand_titles(text)
    text = _expand_currency(text, language)
    text = _expand_percentages(text, language)
    text = _expand_ordinals(text, language)
    text = _expand_years(text, language)
    text = _expand_numbers(text, language)
    text = _expand_acronyms(text)
    text = _clean_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# Step 1 — Acronym-number compounds
# ---------------------------------------------------------------------------

# Matches ALLCAPS-digit patterns like "GPT-4", "H2O", "B-52"
_ACRONYM_NUMBER_RE = re.compile(r"\b([A-Z]{1,})-(\d+)\b")


def _expand_acronym_numbers(text: str) -> str:
    """
    Separate acronym-number compounds: "GPT-4" → "GPT 4".

    The hyphen is replaced with a space so subsequent rules can expand
    "GPT" (acronym) and "4" (number) independently.

    Args:
        text: Input text.

    Returns:
        Text with CAPS-digit hyphens replaced by spaces.
    """
    return _ACRONYM_NUMBER_RE.sub(r"\1 \2", text)


# ---------------------------------------------------------------------------
# Step 2 — Titles and abbreviations
# ---------------------------------------------------------------------------

def _expand_titles(text: str) -> str:
    """
    Expand common titles and abbreviations using TITLE_MAP.

    Applies word-boundary matches in longest-key-first order so that
    "Mrs." is expanded before "Mr." is attempted.

    Args:
        text: Input text.

    Returns:
        Text with recognised abbreviations replaced by their spoken forms.
    """
    for abbrev, expansion in sorted(TITLE_MAP.items(), key=lambda kv: -len(kv[0])):
        # Escape the abbreviation for use in regex (e.g. the "." in "Dr.")
        pattern = re.escape(abbrev)
        text = re.sub(rf"\b{pattern}", expansion, text)
    return text


# ---------------------------------------------------------------------------
# Step 3 — Currency
# ---------------------------------------------------------------------------

# Matches optional symbol + integer part + optional cents: $3.50, £100, €0.99
_CURRENCY_RE = re.compile(
    r"([$£€¥])\s*(\d[\d,]*)(?:\.(\d{1,2}))?",
)


def _expand_currency(text: str, language: str = "en") -> str:
    """
    Expand currency amounts to spoken form.

    Examples:
        "$200"   → "two hundred dollars"
        "$3.50"  → "three dollars fifty cents"
        "£0.99"  → "zero pounds ninety-nine pence"

    Args:
        text:     Input text.
        language: Language code for num2words.

    Returns:
        Text with currency tokens expanded.
    """
    def _replace(m: re.Match) -> str:
        symbol      = m.group(1)
        int_str     = m.group(2).replace(",", "")
        cents_str   = m.group(3)

        names = CURRENCY_MAP.get(symbol, ("unit", "units", "cent", "cents"))
        int_val = int(int_str)

        try:
            int_words  = nw.num2words(int_val, lang=language)
        except Exception:
            int_words  = str(int_val)

        unit_name = names[0] if int_val == 1 else names[1]
        result    = f"{int_words} {unit_name}"

        if cents_str:
            cents_val   = int(cents_str.ljust(2, "0"))
            cent_name   = names[2] if cents_val == 1 else names[3]
            try:
                cent_words = nw.num2words(cents_val, lang=language)
            except Exception:
                cent_words = str(cents_val)
            result += f" {cent_words} {cent_name}"

        return result

    return _CURRENCY_RE.sub(_replace, text)


# ---------------------------------------------------------------------------
# Step 4 — Percentages
# ---------------------------------------------------------------------------

_PERCENTAGE_RE = re.compile(r"(\d[\d,]*)%")


def _expand_percentages(text: str, language: str = "en") -> str:
    """
    Expand percentage expressions: "15%" → "fifteen percent".

    Args:
        text:     Input text.
        language: Language code for num2words.

    Returns:
        Text with percentage tokens expanded.
    """
    def _replace(m: re.Match) -> str:
        val = int(m.group(1).replace(",", ""))
        try:
            words = nw.num2words(val, lang=language)
        except Exception:
            words = m.group(1)
        return f"{words} percent"

    return _PERCENTAGE_RE.sub(_replace, text)


# ---------------------------------------------------------------------------
# Step 5 — Ordinals
# ---------------------------------------------------------------------------

# Matches "1st", "2nd", "3rd", "4th" … "101st" etc.
_ORDINAL_RE = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.IGNORECASE)


def _expand_ordinals(text: str, language: str = "en") -> str:
    """
    Expand ordinal numbers: "1st" → "first", "42nd" → "forty-second".

    Args:
        text:     Input text.
        language: Language code for num2words.

    Returns:
        Text with ordinal tokens expanded.
    """
    def _replace(m: re.Match) -> str:
        val = int(m.group(1))
        try:
            return nw.num2words(val, to="ordinal", lang=language)
        except Exception:
            return m.group(0)

    return _ORDINAL_RE.sub(_replace, text)


# ---------------------------------------------------------------------------
# Step 6 — Years
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"\b(\d{4})\b")


def _expand_years(text: str, language: str = "en") -> str:
    """
    Expand 4-digit numbers in YEAR_MIN–YEAR_MAX as years.

    "2024" → "twenty twenty-four", "1999" → "nineteen ninety-nine".
    Numbers outside the year range are left for _expand_numbers.

    Args:
        text:     Input text.
        language: Language code for num2words (year format is English-only
                  in num2words; non-English codes fall back to cardinal).

    Returns:
        Text with year tokens expanded.
    """
    def _replace(m: re.Match) -> str:
        val = int(m.group(1))
        if YEAR_MIN <= val <= YEAR_MAX:
            try:
                return nw.num2words(val, to="year", lang=language)
            except Exception:
                # num2words may not support to='year' for all languages
                try:
                    return nw.num2words(val, lang=language)
                except Exception:
                    return m.group(0)
        return m.group(0)   # leave non-year numbers for next step

    return _YEAR_RE.sub(_replace, text)


# ---------------------------------------------------------------------------
# Step 7 — Remaining numbers (cardinals and decimals)
# ---------------------------------------------------------------------------

# Matches integers (with optional thousands commas) or decimals
_NUMBER_RE = re.compile(r"\b(\d[\d,]*(?:\.\d+)?)\b")


def _expand_numbers(text: str, language: str = "en") -> str:
    """
    Expand all remaining numeric tokens (cardinals and decimals).

    "42"        → "forty-two"
    "3.14"      → "three point one four"
    "1,000,000" → "one million"

    num2words handles decimals natively with digit-by-digit expansion
    (e.g. 3.14 → "three point one four").

    Args:
        text:     Input text.
        language: Language code for num2words.

    Returns:
        Text with numeric tokens expanded.
    """
    def _replace(m: re.Match) -> str:
        raw = m.group(1).replace(",", "")
        try:
            val = float(raw) if "." in raw else int(raw)
            return nw.num2words(val, lang=language)
        except Exception:
            return m.group(0)

    return _NUMBER_RE.sub(_replace, text)


# ---------------------------------------------------------------------------
# Step 8 — Acronyms
# ---------------------------------------------------------------------------

# Matches 2+ consecutive uppercase ASCII letters as a standalone word.
# Requires no adjacent lowercase letter (avoids expanding sentence-starting
# words captured mid-sentence where one word happens to be capitalised).
_ACRONYM_RE = re.compile(r"\b([A-Z]{2,})\b")


def _expand_acronyms(text: str) -> str:
    """
    Spell out all-caps acronyms letter by letter.

    "AI" → "A I", "NASA" → "N A S A", "GPT" → "G P T".

    Only matches tokens that are entirely uppercase ASCII letters (2+).
    Single capitals (proper names, "I") and mixed-case words are not affected.

    Args:
        text: Input text.

    Returns:
        Text with recognised acronyms spelled out with spaces.
    """
    def _replace(m: re.Match) -> str:
        return " ".join(m.group(1))

    return _ACRONYM_RE.sub(_replace, text)


# ---------------------------------------------------------------------------
# Step 9 — Whitespace cleanup
# ---------------------------------------------------------------------------

def _clean_whitespace(text: str) -> str:
    """
    Collapse multiple spaces and strip leading/trailing whitespace.

    Args:
        text: Input text, possibly with extra spaces from substitutions.

    Returns:
        Text with normalised single-space separators.
    """
    return re.sub(r" {2,}", " ", text).strip()
