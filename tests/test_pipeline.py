"""
deep-whisper · tests/test_pipeline.py
======================================
Stage-by-stage integration test for the full pipeline against real audio.

Unlike the unit tests in tests/, this script exercises the actual GPU models
end-to-end. It is intentionally a standalone script (not a pytest test) so
it can be run directly and inspected interactively:

    python tests/test_pipeline.py --audio path/to/file.wav

Requirements
------------
- A real audio file of clean spoken English (30–120 seconds recommended)
- All pipeline dependencies installed and GPU available
- test_env.py passing before running this

What it tests
-------------
Each pipeline stage is run and its output printed before the next stage
begins. If something goes wrong you can see exactly which stage failed
and what the data looked like going in.

  Stage 1  load_audio + normalize_audio      → shape, dtype, duration
  Stage 2  get_speech_chunks (VAD)            → chunk count and offsets
  Stage 3  transcribe_chunks                 → raw Whisper segments
  Stage 4  reconcile_segments (optional)     → reconciled text
  Stage 5  normalise_segments                → spoken-form text
  Stage 6  align_segments                    → word timestamps
  Stage 7  build_output + serialise          → final JSON

Usage
-----
  # Basic (transcription only)
  python tests/test_pipeline.py --audio my_audio.wav

  # With a context prompt
  python tests/test_pipeline.py --audio my_audio.wav \\
      --prompt "A lecture about machine learning."

  # With a user-provided transcript (exercises the reconcile path)
  python tests/test_pipeline.py --audio my_audio.wav \\
      --transcript "The exact words spoken in the audio."

  # Save output to file
  python tests/test_pipeline.py --audio my_audio.wav --output out.json

  # Use a specific quality preset
  python tests/test_pipeline.py --audio my_audio.wav --quality fast
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def banner(title: str) -> None:
    print(f"\n{'─' * 56}")
    print(f"  {title}")
    print(f"{'─' * 56}")


def ok(label: str, value: str = "") -> None:
    pad = f"  {label:<34s}"
    print(f"  OK   {label:<34s} {value}")


def info(msg: str) -> None:
    print(f"       {msg}")


def fail(label: str, exc: Exception) -> None:
    print(f"  FAIL {label}")
    print(f"       {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="deep-whisper end-to-end pipeline test"
    )
    parser.add_argument(
        "--audio", required=True,
        help="Path to a WAV/FLAC/MP3 audio file of clean spoken English",
    )
    parser.add_argument(
        "--prompt", default="",
        help="Optional context prompt to seed Whisper's vocabulary",
    )
    parser.add_argument(
        "--transcript", default="",
        help="Optional user transcript — exercises the reconcile path",
    )
    parser.add_argument(
        "--quality", default="balanced",
        choices=["fast", "balanced", "accurate"],
        help="Whisper quality preset (default: balanced)",
    )
    parser.add_argument(
        "--output", default="",
        help="Optional path to write the final JSON output",
    )
    parser.add_argument(
        "--no-snap", action="store_true",
        help="Disable energy boundary snapping (faster, slightly less accurate)",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"ERROR: audio file not found: {audio_path}")
        return 1

    failures = 0
    t_start  = time.perf_counter()

    print()
    print("=== deep-whisper pipeline test ===")
    print(f"  audio:      {audio_path}")
    print(f"  quality:    {args.quality}")
    print(f"  prompt:     {args.prompt[:60] + '...' if len(args.prompt) > 60 else args.prompt or '(none)'}")
    print(f"  transcript: {'provided' if args.transcript else '(none)'}")
    print(f"  snap:       {'disabled' if args.no_snap else 'enabled'}")

    # ── Stage 1: Load audio ──────────────────────────────────────────────────
    banner("Stage 1 — load_audio + normalize_audio")
    try:
        t = time.perf_counter()
        from deep_whisper.pipeline.audio import load_audio, normalize_audio, get_duration
        audio = normalize_audio(load_audio(str(audio_path)))
        elapsed = time.perf_counter() - t
        ok("load_audio",   f"shape={audio.shape}  dtype={audio.dtype}")
        ok("normalize",    f"peak={float(abs(audio).max()):.4f}")
        ok("duration",     f"{get_duration(audio):.2f}s")
        ok("load time",    f"{elapsed:.2f}s")
    except Exception as exc:
        fail("load_audio", exc)
        failures += 1
        print("\n  Cannot continue without audio. Exiting.")
        return 1

    # ── Stage 2: VAD + chunking ──────────────────────────────────────────────
    banner("Stage 2 — get_speech_chunks  (VAD)")
    try:
        t = time.perf_counter()
        from deep_whisper.pipeline.vad import get_speech_chunks
        chunks = get_speech_chunks(audio)
        elapsed = time.perf_counter() - t
        ok("chunks detected", str(len(chunks)))
        ok("vad time",        f"{elapsed:.2f}s")
        for i, (chunk, offset) in enumerate(chunks):
            info(f"  chunk {i:2d}:  offset={offset:7.2f}s  len={len(chunk)/16000:.2f}s")
        if not chunks:
            print("\n  WARNING: no speech detected. Check your audio file.")
            return 1
    except Exception as exc:
        fail("get_speech_chunks", exc)
        failures += 1

    # ── Stage 3: Transcription ───────────────────────────────────────────────
    banner("Stage 3 — transcribe_chunks  (GPU)")
    segments = []
    try:
        t = time.perf_counter()
        from deep_whisper.pipeline.transcribe import transcribe_chunks
        initial_prompt = args.transcript or args.prompt
        segments = transcribe_chunks(
            chunks,
            initial_prompt = initial_prompt,
            quality        = args.quality,
            language       = "en",
        )
        elapsed = time.perf_counter() - t
        ok("segments",        str(len(segments)))
        ok("transcribe time", f"{elapsed:.2f}s")
        dropped = sum(1 for s in segments if s.get("flagged"))
        if dropped:
            info(f"  {dropped} segment(s) soft-flagged (low no_speech confidence)")
        print()
        for s in segments:
            flag = " [flagged]" if s.get("flagged") else ""
            print(f"  [{s['start']:6.2f} -> {s['end']:6.2f}]  {s['text']}{flag}")
        if not segments:
            print("\n  WARNING: no segments produced. All audio may have been filtered.")
            return 1
    except Exception as exc:
        fail("transcribe_chunks", exc)
        failures += 1

    # ── Stage 4: Reconcile (user transcript path) ────────────────────────────
    if args.transcript:
        banner("Stage 4 — reconcile_segments  (user transcript path)")
        try:
            t = time.perf_counter()
            from deep_whisper.pipeline.reconcile import reconcile_segments
            segments = reconcile_segments(args.transcript, segments)
            elapsed  = time.perf_counter() - t
            ok("reconcile time", f"{elapsed:.2f}s")
            print()
            for s in segments:
                print(f"  [{s['start']:6.2f} -> {s['end']:6.2f}]  {s['text']}")
        except Exception as exc:
            fail("reconcile_segments", exc)
            failures += 1
    else:
        banner("Stage 4 — reconcile_segments  (skipped — no user transcript)")
        info("Pass --transcript to test the user-provided transcript path.")

    # ── Stage 5: Text normalisation ──────────────────────────────────────────
    banner("Stage 5 — normalise_segments")
    try:
        t = time.perf_counter()
        from deep_whisper.pipeline.normalise import normalise_segments
        norm_segments = normalise_segments(segments)
        elapsed = time.perf_counter() - t
        ok("normalise time", f"{elapsed:.2f}s")
        changed = sum(
            1 for a, b in zip(segments, norm_segments)
            if a["text"] != b["text"]
        )
        ok("segments changed", f"{changed} / {len(norm_segments)}")
        if changed:
            print()
            for a, b in zip(segments, norm_segments):
                if a["text"] != b["text"]:
                    info(f"  before: {a['text']}")
                    info(f"  after:  {b['text']}")
        segments = norm_segments
    except Exception as exc:
        fail("normalise_segments", exc)
        failures += 1

    # ── Stage 6: Forced alignment ────────────────────────────────────────────
    banner("Stage 6 — align_segments  (GPU)")
    aligned = []
    try:
        t = time.perf_counter()
        from deep_whisper.pipeline.align import align_segments
        aligned = align_segments(
            segments,
            audio,
            snap_enabled = not args.no_snap,
        )
        elapsed = time.perf_counter() - t
        total_words   = sum(len(s.get("words", [])) for s in aligned)
        low_conf_words = sum(
            1 for s in aligned
            for w in s.get("words", [])
            if w.get("low_confidence")
        )
        ok("align time",       f"{elapsed:.2f}s")
        ok("total words",      str(total_words))
        ok("low confidence",   f"{low_conf_words} / {total_words}")
        print()
        for s in aligned:
            print(f"  [{s['start']:6.2f} -> {s['end']:6.2f}]  {s['text']}")
            for w in s.get("words", [])[:5]:    # first 5 words per segment
                flag = " *" if w.get("low_confidence") else ""
                print(f"    {w['word']:20s} {w['start']:.3f} -> {w['end']:.3f}  "
                      f"conf={w['confidence']:.2f}{flag}")
            if len(s.get("words", [])) > 5:
                info(f"    ... ({len(s['words']) - 5} more words)")
    except Exception as exc:
        fail("align_segments", exc)
        failures += 1

    # ── Stage 7: Post-processing + serialisation ─────────────────────────────
    banner("Stage 7 — build_output + serialise")
    try:
        t = time.perf_counter()
        from deep_whisper.pipeline.postprocess import build_output, serialise
        output  = build_output(
            aligned if aligned else segments,
            audio,
            whisper_model            = "large-v3-turbo",
            alignment_model          = "wav2vec2-base-960h",
            prompt                   = args.prompt,
            user_transcript_provided = bool(args.transcript),
            timestamp_level          = "both",
        )
        json_str = serialise(output)
        elapsed  = time.perf_counter() - t
        ok("build time",    f"{elapsed:.2f}s")
        ok("schema version",output["schema_version"])
        ok("duration",      f"{output['metadata']['duration_seconds']:.2f}s")
        ok("segments",      str(len(output["segments"])))
        ok("words (flat)",  str(len(output.get("words", []))))
        ok("json size",     f"{len(json_str):,} chars")
        print()
        print("  Transcript:")
        print(f"  {output['transcript'][:200]}"
              + ("..." if len(output["transcript"]) > 200 else ""))

        if args.output:
            out_path = Path(args.output)
            out_path.write_text(json_str, encoding="utf-8")
            print(f"\n  Output written to: {out_path}")

    except Exception as exc:
        fail("build_output", exc)
        failures += 1

    # ── Summary ──────────────────────────────────────────────────────────────
    t_total = time.perf_counter() - t_start
    banner("Summary")
    print(f"  Total time:  {t_total:.2f}s")
    print(f"  Failures:    {failures}")
    if failures == 0:
        print()
        print("  All stages passed.")
        print("  Word timestamps marked * have low confidence (< 0.6).")
    else:
        print()
        print(f"  {failures} stage(s) failed — check output above.")
    print()
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
