"""
deep-whisper · deep_whisper/__init__.py
========================================
Top-level package. Re-exports the public API from deep_whisper.pipeline
so both import styles work:

    from deep_whisper import run          # short form
    from deep_whisper.pipeline import run  # explicit form
"""

from deep_whisper.pipeline import run  # noqa: F401

__all__ = ["run"]
