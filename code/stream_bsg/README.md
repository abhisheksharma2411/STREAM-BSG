# STREAM-BSG: Main Implementation

**Status: WIP — coming in v0.2 / camera-ready version**

This directory will contain the main STREAM-BSG implementation:

- `state.py` — Streaming buyer-supplier graph state (Redis-backed)
- `features.py` — Incremental computation of the 47 graph features
- `classifier.py` — XGBoost classifier over engineered features
- `pipeline.py` — End-to-end streaming pipeline
- `tests/` — Unit tests

The implementation is being finalized for the camera-ready version of the paper. Track progress on the [issues page](../../issues).
