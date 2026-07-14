# Reproducing STREAM-BSG (IEEE InC4 2026, Paper 3092)

This document is the step-by-step guide for reproducing every number in the
camera-ready paper. For a one-shot end-to-end pipeline see
[`./reproduce.sh`](reproduce.sh). This document additionally covers the
individual ablations (WI1, WI3, WI5, WI8) and the seed sweep.

## 1. Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| Python | 3.10 | 3.13 used for the camera-ready |
| Disk | ~8 GB | ~1.3 GB Kaggle CSVs + ~500 MB intermediate parquet + venv |
| RAM | 8 GB | feature extractor peaks at ~5.9 GB |
| Time (end-to-end) | ~2 hours | dominated by the 49-feature streaming extractor (~85 min for the main run; add ~85 min per ablation that recomputes features) |

## 2. Data acquisition (IEEE-CIS Fraud Detection)

The dataset is **not** redistributed in this repository (Kaggle license).
Two ways to obtain it:

**A. Kaggle CLI (automated):**

```bash
pip install kaggle
# Configure Kaggle credentials once: place kaggle.json at ~/.kaggle/kaggle.json
# See https://github.com/Kaggle/kaggle-api#api-credentials
kaggle competitions download -c ieee-fraud-detection -p data/ieee-cis/
unzip data/ieee-cis/ieee-fraud-detection.zip -d data/ieee-cis/
```

You must accept the competition rules at
<https://www.kaggle.com/c/ieee-fraud-detection> once (from a browser signed
into a Kaggle account) before the CLI download will succeed.

**B. Manual:** download `train_transaction.csv` from
<https://www.kaggle.com/c/ieee-fraud-detection/data> and place it at
`data/ieee-cis/train_transaction.csv`.

Only `train_transaction.csv` is required for the paper's results.
`train_identity.csv` is not consumed.

## 3. Environment setup with pinned versions

The paper's numbers are **bit-for-bit reproducible only** under the pinned
`scikit-learn==1.8.0` and `xgboost==3.2.0`. Other pins are recommended for
consistency.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If PyTorch / torch-geometric wheels are unavailable for your platform, GraphSAGE
will fail to import; you can still reproduce STREAM-BSG, LR, and XGB-tabular.

## 4. End-to-end reproduction (one command)

```bash
./reproduce.sh
```

Produces `results/streambsg_results.json`, `results/lr_results.json`,
`results/xgb_tabular_results.json`, and (when torch-geometric is installed)
`results/graphsage_results.json`.

## 5. Reproducing individual work items

### Base run — Table II main numbers (seed=42)

```bash
# 5.1 Inject fraud (1 % rate)
python3 code/synth/synth_b2b_injection.py \
    --input data/ieee-cis/train_transaction.csv \
    --output data/ieee_cis_with_synthetic_b2b.parquet \
    --inject-rate 0.01 --seed 42
# ~40 s

# 5.2 Extract 49 streaming features
python3 code/stream_bsg/features.py \
    --input data/ieee_cis_with_synthetic_b2b.parquet \
    --output data/ieee_cis_with_features.parquet
# ~85 minutes (the slow step)

# 5.3 STREAM-BSG classifier — Table II headline row
python3 code/stream_bsg/classifier.py \
    --input data/ieee_cis_with_features.parquet \
    --output results/streambsg_results.json --seed 42
# ~30 s

# 5.4 Baselines
python3 code/baselines/lr_baseline.py \
    --input data/ieee_cis_with_synthetic_b2b.parquet \
    --output results/lr_results.json --seed 42
python3 code/baselines/xgb_tabular.py \
    --input data/ieee_cis_with_synthetic_b2b.parquet \
    --output results/xgb_tabular_results.json --seed 42
python3 code/baselines/graphsage_baseline.py \
    --input data/ieee_cis_with_synthetic_b2b.parquet \
    --output results/graphsage_results.json --seed 42
```

### WI1 — Feature ablation (drop 11 topology-targeted features)

Uses the same features parquet as the base run; only the classifier is retrained
on a restricted column set. See
[`results/ablation_a_no_topology_feats.json`](results/ablation_a_no_topology_feats.json)
for the shipped result. The dropped feature list and rerun instructions are in
[`results/all_experiments_summary.md`](results/all_experiments_summary.md) § 5.

### WI3 — 0.5 % injection-rate ablation

Full pipeline rerun with a lower injection rate:

```bash
python3 code/synth/synth_b2b_injection.py \
    --input data/ieee-cis/train_transaction.csv \
    --output data/ablation_0p5pct/synth_0p5pct.parquet \
    --inject-rate 0.005 --seed 42
python3 code/stream_bsg/features.py \
    --input data/ablation_0p5pct/synth_0p5pct.parquet \
    --output data/ablation_0p5pct/features_0p5pct.parquet
python3 code/stream_bsg/classifier.py \
    --input data/ablation_0p5pct/features_0p5pct.parquet \
    --output results/ablation_0p5pct/streambsg_0p5pct.json \
    --seed 42
```

Wall time ≈ 90 min (dominated by feature extraction).

### WI5 — Out-of-distribution structural attacks

Uses [`code/synth/synth_b2b_injection_ood.py`](code/synth/synth_b2b_injection_ood.py)
for structurally-reshaped T1–T5 variants (2-buyer shell, 14-day dup window,
3-tx payment-term gradient, ring size 4, 2-tx bank transit). The paper's
canonical STREAM-BSG model is scored against the OOD test split **without
retraining** — the reviewer-defense generalization test.

```bash
python3 -m code.synth.synth_b2b_injection_ood \
    --input data/ieee-cis/train_transaction.csv \
    --output data/ood_structural/synth_ood.parquet \
    --inject-rate 0.01 --seed 42
python3 code/stream_bsg/features.py \
    --input data/ood_structural/synth_ood.parquet \
    --output data/ood_structural/features_ood.parquet
# Cross-eval script: see results/ablation_c_ood_injection.json for the shipped
# result and results/all_experiments_summary.md § 7 for the eval procedure.
```

Wall time ≈ 90 min.

### WI8 — Leakage analysis (no synth-derived features)

Drops `synth_payment_term_days` and `synth_supplier_bank` from the tabular
baselines, and the 5 STREAM-BSG features derived from them
(`feat_payment_term_{mean,std,zscore_current}`,
`feat_bank_change_{recent,on_this_row}`). Uses the same feature parquet from
the base run; only the classifier retrains. See
[`results/leakage_analysis_report.md`](results/leakage_analysis_report.md) for
Table V and interpretation, and JSONs
[`results/lr_results_no_synth_leak.json`](results/lr_results_no_synth_leak.json),
[`results/xgb_tabular_no_synth_leak.json`](results/xgb_tabular_no_synth_leak.json),
[`results/streambsg_no_synth_leak.json`](results/streambsg_no_synth_leak.json).

### WI2 — Seed stability

```bash
# STREAM-BSG: 5 seeds (~25 s each on the pre-computed features parquet)
for s in 42 123 456 789 2026; do
  python3 code/stream_bsg/classifier.py \
      --input data/ieee_cis_with_features.parquet \
      --output results/streambsg_seed${s}.json --seed $s
done

# GraphSAGE: 7 seeds (~35 s each)
for s in 42 123 456 789 2026 314 1729; do
  python3 code/baselines/graphsage_baseline.py \
      --input data/ieee_cis_with_synthetic_b2b.parquet \
      --output results/graphsage_seed${s}.json --seed $s
done
```

Committed multi-seed summary:
[`results/seed_stability.json`](results/seed_stability.json) (STREAM-BSG) and
`methods.graphsage.seed_stability` in
[`results/all_baselines_comparison.json`](results/all_baselines_comparison.json).

## 6. Expected numbers (seed=42, pinned env)

Under the pinned environment
(`scikit-learn==1.8.0`, `xgboost==3.2.0`, `numpy==2.5.1`, `pandas==3.0.3`,
`pyarrow==24.0.0`), the paper's headline STREAM-BSG numbers reproduce
**bit-for-bit** — `|Δ| = 0.0000` on all four metrics:

| Metric | Expected | Tolerance under pinned env |
|---|---:|---|
| ROC-AUC | 0.8885 | ± 0.0000 |
| PR-AUC | 0.6778 | ± 0.0000 |
| F1_best | 0.7767 | ± 0.0000 |
| Recall@5%FPR | 0.7111 | ± 0.0000 |

If your environment has different XGBoost or scikit-learn versions, expect
sub-percent drift (e.g. `xgboost 3.3.x` was measured at ROC-AUC 0.8867 vs. paper
0.8885 — a 0.002 delta from split-tie-breaking changes).

Baseline expected numbers (same seed, pinned env):

| Method | ROC-AUC | PR-AUC | F1_best | R@5%FPR |
|---|---:|---:|---:|---:|
| LR (tabular) | 0.6519 | 0.0492 | 0.1597 | 0.3125 |
| XGB (tabular) | 0.6862 | 0.1252 | 0.2468 | 0.3487 |
| GraphSAGE (seed 42) | 0.6872 | 0.0864 | 0.2391 | 0.4013 |

Full per-topology and per-ablation numbers:
[`results/all_experiments_summary.md`](results/all_experiments_summary.md).

## 7. Troubleshooting

**`ModuleNotFoundError: No module named 'torch_geometric'`** — install per your
platform: `pip install torch torch-geometric`. On Apple Silicon,
`pip install torch torch-geometric` from PyPI works with the pinned
`torch>=2.0`. On Linux/CUDA, follow the torch-geometric installation guide for
your CUDA version.

**`XGBoostError: Library not loaded: @rpath/libomp.dylib`** (macOS) — install
OpenMP: `brew install libomp`. Required by XGBoost's `hist` tree method.

**`FileNotFoundError: data/ieee-cis/train_transaction.csv`** — the Kaggle
dataset is not committed. See §2 above.

**Feature extractor OOM (< 8 GB RAM)** — the pipeline holds the full ~591 K
row dataframe plus the running graph state. Split the input into a subset for
smoke tests: `python3 code/synth/synth_b2b_injection.py --input … --output …
--inject-rate 0.01 --seed 42 --max-rows 50000`.

**Feature extractor much slower than 85 min** — the 2-hop subgraph traversal
scales super-linearly as the graph densifies (final 10 % of rows takes ~15 %
of runtime). If it's spent hours below 90 % complete, check that your Python
build uses the platform-native NumPy wheels (macOS Rosetta emulation is
particularly slow).

**GraphSAGE PR-AUC differs from the paper** — GraphSAGE PR-AUC is bimodal
across seeds (0.07–0.29 range across 7 seeds); a single-seed run may land in
either mode. Compare against the range, not a single-seed number. See
`results/all_experiments_summary.md` § 4.2.

## 8. Consolidated reference

For a single-file walk-through of every table (I / II / III / IV / V / VII)
and every source JSON, see
[`results/all_experiments_summary.md`](results/all_experiments_summary.md).
