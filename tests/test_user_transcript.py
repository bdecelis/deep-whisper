# test_user_transcript.py
from pipeline.reconcile  import reconcile_segments

# Use the segments from your Stage 3 run above
# Simulate a user transcript that's slightly different
user_transcript = "..."  # type or paste what you think the audio says

reconciled = reconcile_segments(user_transcript, segments)
norm_segs  = normalise_segments(reconciled)
aligned    = align_segments(norm_segs, audio)
# ...