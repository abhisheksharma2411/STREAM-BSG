# Data

This directory is the expected location for input datasets. **Dataset files are not committed to this repository** (see `.gitignore`) because they are subject to their original licenses.

## IEEE-CIS Fraud Detection (primary dataset)

The IEEE-CIS Fraud Detection dataset is the base data for STREAM-BSG's evaluation. It must be downloaded directly from Kaggle.

1. Visit <https://www.kaggle.com/c/ieee-fraud-detection/data> (Kaggle account required).
2. Accept the competition terms.
3. Download `train_transaction.csv` and `train_identity.csv`.
4. Place them in this `data/` directory.

The full IEEE-CIS dataset is approximately 590,000 transactions with binary fraud labels at a base rate of roughly 3.5%. STREAM-BSG layers synthetic B2B fraud injection on top of this real consumer transaction data.

## Output of the synthetic injection script

When you run `code/synth/synth_b2b_injection.py`, the labeled output is written to `data/ieee_cis_with_synthetic_b2b.parquet` by default. The output schema preserves all original IEEE-CIS columns and adds:

- `fraud_injected` (binary, 0 or 1) — STREAM-BSG's training label
- `fraud_topology` (categorical: T1, T2, T3, T4, T5, or NULL) — for per-topology analysis
