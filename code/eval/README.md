# Evaluation

**Status: WIP — coming in v0.2 / camera-ready version**

End-to-end evaluation harness:

- `run_all.py` — Trains every baseline plus STREAM-BSG on the same split, writes results to `results/table_accuracy.csv` and `results/table_latency.csv`
- `metrics.py` — Metric computation (ROC-AUC, PR-AUC, F1, recall at FPR threshold)
- `latency_harness.py` — Measures per-component p50/p95/p99 latency for the streaming pipeline
- `cold_start.py` — Cold-start sub-experiment (10% held-out buyer-supplier pairs)

The evaluation harness is being finalized for the camera-ready version of the paper.
