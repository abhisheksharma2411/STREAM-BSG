# Baselines

**Status: WIP — coming in v0.2 / camera-ready version**

Reference baseline implementations for comparison against STREAM-BSG:

- `lr_baseline.py` — Logistic Regression on raw transaction features (sanity baseline)
- `xgb_tabular.py` — XGBoost over raw tabular features only (no graph structure)
- `graphsage_baseline.py` — GraphSAGE with 2-hop sampling on the buyer-supplier graph

All baselines train on the same train/test split as STREAM-BSG and report ROC-AUC, PR-AUC, F1, and recall at 5% false-positive rate.

Implementations are being finalized for the camera-ready version of the paper.
