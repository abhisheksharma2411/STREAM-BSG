# STREAM-BSG Camera-Ready Experimental Results — Consolidated Reference

Single-file index for the seven reviewer work items requested for InC4 2026
Paper 3092. Every numeric result in this document is loaded verbatim from a
committed JSON artifact under `results/`; source paths are listed alongside
each table and in the index at §8.

## Contents

1. Environment & reproducibility
2. ID canonical results (with-synth) — Tables I, II, III
3. ID results without synthetic-attribute leakage — Table VII
4. Seed stability
5. WI1 — feature ablation without topology-targeted features
6. WI3 — reduced injection rate (0.5 %)
7. WI5 — out-of-distribution structural attacks
8. Source-file index

---

## 1. Environment & reproducibility

All results reproducible via pinned versions in [`requirements.txt`](../requirements.txt):

```
xgboost==3.2.0
scikit-learn==1.8.0
numpy==2.5.1
pandas==3.0.3
pyarrow==24.0.0
torch>=2.0
torch-geometric>=2.4
```

- Python 3.13
- Random seed **42** unless otherwise noted
- Chronological 70 / 15 / 15 train/val/test split by `TransactionDT` (stable
  mergesort); train ≈ 414,204 rows, val ≈ 88,758, test ≈ 88,759
- 49 STREAM-BSG features; documented in [`paper/feature_table.md`](../paper/feature_table.md)
- 5 injection topologies; documented in [`paper/topology_definitions.md`](../paper/topology_definitions.md)

**Reproducibility check (§Work Item 2, third clarification):** the seed=42 rerun
under pinned versions matches the camera-ready canonical STREAM-BSG headlines
**bit-for-bit**:

| Metric | Pinned rerun | Paper canonical | \|Δ\| |
|---|---:|---:|---:|
| ROC-AUC | 0.8885 | 0.8885 | 0.0000 |
| PR-AUC | 0.6778 | 0.6778 | 0.0000 |
| F1_best | 0.7767 | 0.7767 | 0.0000 |
| Recall@5%FPR | 0.7111 | 0.7111 | 0.0000 |

Feature-importance top-20 ranks and gains match to |Δgain| = 0.0000.

---

## 2. ID canonical results (with-synth) — paper Tables I / II / III

### 2.1 Aggregate accuracy — **Table I** (paper source of truth)

| Method | ROC-AUC | PR-AUC | F1_best | Recall@5%FPR |
|---|---:|---:|---:|---:|
| LR (tabular) | 0.6519 | 0.0492 | 0.1597 | 0.3125 |
| XGB (tabular) | 0.6862 | 0.1252 | 0.2468 | 0.3487 |
| GraphSAGE | 0.6872 | 0.0864 | 0.2391 | 0.4013 |
| **STREAM-BSG** | **0.8885** | **0.6778** | **0.7767** | **0.7111** |

Sources: [`lr_results.json`](lr_results.json), [`xgb_tabular_results.json`](xgb_tabular_results.json),
[`graphsage_results.json`](graphsage_results.json), [`streambsg_results.json`](streambsg_results.json),
consolidated in [`all_baselines_comparison.json`](all_baselines_comparison.json).

### 2.2 Per-row inference latency — **Table II**

Per-transaction predict on the test split after 100-prediction warmup; `time.perf_counter()`.

| Method | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---:|---:|---:|
| LR | — | — | — |
| XGB (tabular) | 0.19 | 0.22 | 0.34 |
| GraphSAGE | 0.25 | 0.46 | 1.74 |
| **STREAM-BSG** | **0.20** | **0.23** | **0.37** |

Sources: same as §2.1. GraphSAGE latency assumes pre-computed transductive
node embeddings; only the classification head forward is timed per row.

### 2.3 Per-topology recall @ each method's own 5 %-FPR threshold — **Table III**

| Topology | n | LR | XGB-tab | GraphSAGE | STREAM-BSG |
|---|---:|---:|---:|---:|---:|
| T1 vendor_injection | 227 | 0.216 | 0.414 | **1.000** | **1.000** |
| T2 invoice_cycling (row)†   | 360 | 0.042 | 0.011 | 0.061 | **0.153** |
| T2 invoice_cycling (event)† | 180 | 0.067 | 0.011 | 0.061 | **0.267** |
| T3 payment_term_manip‡  | 142 | 0.965 | 0.894 | 0.014 | **0.958** |
| T4 shell_supplier_ring | 186 | 0.038 | 0.043 | **1.000** | **1.000** |
| T5 wire_redirection‡ | 189 | 0.725 | 0.804 | 0.032 | **0.958** |

† T2 is reported at two granularities. Each T2 injection emits an original + a duplicate row and labels both. Row-level recall counts each labeled row independently; event-level recall groups by `(buyer_id, supplier_id, base_TransactionID)` and counts the pair detected if either row is flagged.

‡ LR and XGB benefit from direct visibility of `synth_payment_term_days` (T3) — see §3 for the leakage-controlled result. LR/XGB T5 recall is **not** driven by leakage (see §3).

Source: [`all_baselines_comparison.json`](all_baselines_comparison.json) key `methods.<m>.operating_points.fpr_5pct.per_topology`.

---

## 3. ID results without synthetic-attribute leakage — **Table VII**

Removes the following inputs from each method (see [`leakage_analysis_report.md`](leakage_analysis_report.md) for the full audit):

- **LR, XGB-tabular:** dropped raw feature `synth_payment_term_days`. Note that `synth_supplier_bank` was already dropped by the tabular pipelines' cardinality filter (`MAX_CAT_CARDINALITY = 50`); it was never actually leaking.
- **STREAM-BSG:** dropped the 5 `feat_*` that transitively depend on synth_* inputs — `feat_payment_term_mean`, `feat_payment_term_std`, `feat_payment_term_zscore_current`, `feat_bank_change_recent`, `feat_bank_change_on_this_row`. STREAM-BSG's classifier never sees raw `synth_*` columns directly.

### 3.1 Aggregate with-synth vs. no-synth

| Method | ROC-AUC (with) | ROC-AUC (no) | Δ | PR-AUC (with) | PR-AUC (no) | Δ | F1_best (with) | F1_best (no) | Δ | R@5%FPR (with) | R@5%FPR (no) | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LR | 0.6519 | **0.5979** | −0.054 | 0.0492 | 0.0304 | −0.019 | 0.1597 | 0.0850 | −0.075 | 0.3125 | 0.1957 | −0.117 |
| XGB-tabular | 0.6862 | **0.6299** | −0.056 | 0.1252 | 0.0802 | −0.045 | 0.2468 | 0.1589 | −0.088 | 0.3487 | 0.2400 | −0.109 |
| STREAM-BSG | 0.8885 | **0.7906** | −0.098 | 0.6778 | 0.4267 | −0.251 | 0.7767 | 0.5396 | −0.237 | 0.7111 | 0.4837 | −0.227 |

### 3.2 Per-topology recall @ 5 %-FPR — with-synth vs. no-synth

| Topology | n | LR (with) | LR (no) | XGB (with) | XGB (no) | STREAM-BSG (with) | STREAM-BSG (no) |
|---|---:|---:|---:|---:|---:|---:|---:|
| T1 vendor_injection | 227 | 0.216 | 0.225 | 0.414 | 0.410 | **1.000** | **1.000** |
| T2 invoice_cycling (row) | 360 | 0.042 | 0.044 | 0.011 | 0.014 | 0.153 | 0.086 |
| T2 invoice_cycling (event) | 180 | 0.067 | 0.044 | 0.011 | 0.017 | 0.267 | 0.150 |
| T3 payment_term_manip | 142 | 0.965 | **0.014** | 0.894 | **0.042** | 0.958 | **0.099** |
| T4 shell_supplier_ring | 186 | 0.038 | 0.027 | 0.043 | 0.048 | **1.000** | **1.000** |
| T5 wire_redirection | 189 | 0.725 | 0.751 | 0.804 | 0.804 | 0.958 | **0.402** |

**Key facts for the paper:**
- T1 / T4 are unchanged for STREAM-BSG (**1.000 in both configurations**) — pure structural wins.
- T3 collapses for all three methods when the synth column is removed — the paper should qualify T3 as *demonstrating the method*, not as an operational claim without payment-term inputs.
- T5 recall of LR/XGB was **never** from leakage (LR: +0.026, XGB: 0.000) — driven by `TransactionAmt` inflation. STREAM-BSG's T5 does depend on the derived bank features and drops from 0.958 to 0.402.

Sources: [`lr_results_no_synth_leak.json`](lr_results_no_synth_leak.json),
[`xgb_tabular_no_synth_leak.json`](xgb_tabular_no_synth_leak.json),
[`streambsg_no_synth_leak.json`](streambsg_no_synth_leak.json),
[`leakage_analysis_report.md`](leakage_analysis_report.md).

---

## 4. Seed stability — Work Item 2

### 4.1 STREAM-BSG (5 seeds: 42, 123, 456, 789, 2026) — pinned canonical env

| Metric | Mean ± Std | Min | Max | Range |
|---|---:|---:|---:|---:|
| ROC-AUC | **0.8894 ± 0.0011** | 0.8883 | 0.8909 | 0.0026 |
| PR-AUC | **0.6752 ± 0.0020** | 0.6730 | 0.6778 | 0.0048 |
| F1_best | **0.7760 ± 0.0011** | 0.7743 | 0.7771 | 0.0028 |
| Recall@5%FPR | **0.7022 ± 0.0079** | 0.6938 | 0.7111 | 0.0173 |

**Recommended paper quote:** ROC-AUC 0.889 ± 0.001 across 5 seeds.

Source: [`seed_stability.json`](seed_stability.json).

### 4.2 GraphSAGE (7 seeds: 42, 123, 456, 789, 2026, 314, 1729)

| Metric | Mean ± Std | Min | Max |
|---|---:|---:|---:|
| ROC-AUC | 0.6899 ± 0.0033 | 0.6849 | 0.6944 |
| PR-AUC | **0.1563 ± 0.0989** | **0.0698** | **0.2933** |
| F1_best | 0.2779 ± 0.0652 | 0.2036 | 0.3783 |
| Recall@5%FPR | 0.3940 ± 0.0191 | 0.3569 | 0.4103 |

**Note the PR-AUC bimodality**: three seeds (456, 1729, 2026) landed in a favorable early-stop epoch (PR-AUC ≥ 0.18); the other four in a poor local optimum (~0.07–0.10). The paper should quote **ROC-AUC 0.690 ± 0.003** as the stable metric and PR-AUC as a range [0.070, 0.293] rather than a mean ± std.

Source: [`all_baselines_comparison.json`](all_baselines_comparison.json) key `methods.graphsage.seed_stability`; per-seed JSONs `graphsage_seed{123,456,789,2026,314,1729}.json`.

---

## 5. WI1 — feature ablation (no topology-targeted features)

Drops 11 topology-targeted features from STREAM-BSG training input:

`feat_bank_change_on_this_row`, `feat_supplier_attr_change_on_this_row`,
`feat_bank_change_recent`, `feat_supplier_attribute_change_recent`,
`feat_duplicate_invoice_count_in_window`, `feat_invoice_amount_collision_in_window`,
`feat_payment_term_zscore_current`, `feat_ring_detection_score`,
`feat_shell_supplier_density`, `feat_has_cycle_2hop`, `feat_amount_above_p95`.

Result: **38 general-purpose features** trained under the same hyperparameters and seed.

### 5.1 Aggregate — full 49 features vs. WI1 38 features

| Metric | Full (49) | Ablation (38) | Δ |
|---|---:|---:|---:|
| ROC-AUC | 0.8885 | **0.7881** | −0.100 |
| PR-AUC | 0.6778 | **0.4253** | −0.253 |
| F1_best | 0.7767 | 0.5399 | −0.237 |
| Recall@5%FPR | 0.7111 | 0.4728 | −0.238 |

### 5.2 Per-topology recall @ 5 %-FPR — full vs. ablation

| Topology | n | Full | Ablation | Δ |
|---|---:|---:|---:|---:|
| T1 vendor_injection | 227 | 1.000 | 1.000 | 0.000 |
| T2 invoice_cycling (row) | 360 | 0.153 | 0.100 | −0.053 |
| T2 invoice_cycling (event) | 180 | 0.267 | 0.172 | −0.094 |
| **T3 payment_term_manip** | 142 | 0.958 | **0.077** | **−0.881** |
| T4 shell_supplier_ring | 186 | 1.000 | 1.000 | 0.000 |
| **T5 wire_redirection** | 189 | 0.958 | **0.328** | **−0.630** |

**Interpretation:** T1 and T4 survive intact — general-purpose graph features are sufficient. T3 and T5 collapse — the 11 topology-targeted features carry ~10 pts aggregate ROC-AUC and ~25 pts PR-AUC, almost entirely on those two topologies. Clean justification for the taxonomy-driven feature design.

Source: [`ablation_a_no_topology_feats.json`](ablation_a_no_topology_feats.json).

---

## 6. WI3 — reduced injection rate (0.5 %)

Runs the full pipeline (synth → features → classifier) with `--inject-rate 0.005` instead of 0.01, keeping every other setting identical. Train positive rate drops from 1.19 % to 0.59 %; `scale_pos_weight` doubles from ≈ 83 to ≈ 168.

### 6.1 Aggregate — 1.0 % (paper) vs. 0.5 %

| Metric | 1.0 % headline | 0.5 % (WI3) | Δ |
|---|---:|---:|---:|
| ROC-AUC | 0.8885 | **0.8757** | −0.013 |
| PR-AUC | 0.6778 | **0.6515** | −0.026 |
| F1_best | 0.7767 | 0.7790 | +0.002 |
| Recall@5%FPR | 0.7111 | **0.6866** | −0.025 |
| p99 latency | 0.37 ms | 0.38 ms | — |

### 6.2 Per-topology recall @ 5 %-FPR — 1.0 % vs. 0.5 %

| Topology | n@1.0% | n@0.5% | r@1.0% | r@0.5% |
|---|---:|---:|---:|---:|
| T1 vendor_injection | 227 | 109 | 1.000 | **1.000** |
| T2 invoice_cycling (row) | 360 | 184 | 0.153 | 0.087 |
| T2 invoice_cycling (event) | 180 | 92 | 0.267 | 0.141 |
| T3 payment_term_manip | 142 | 70 | 0.958 | **0.943** |
| T4 shell_supplier_ring | 186 | 96 | 1.000 | **1.000** |
| T5 wire_redirection | 189 | 93 | 0.958 | **0.989** |

**Interpretation:** the class is twice as rare and metrics degrade by only 1–3 aggregate points. T1, T3, T4, T5 all remain at ≥ 0.94 recall; T5 actually gains slightly. T2 halves (already the weakest topology). Addresses the reviewer concern about whether the paper's numbers are an artifact of the 1 % injection rate — they are not.

Source: [`ablation_0p5pct/streambsg_0p5pct.json`](ablation_0p5pct/streambsg_0p5pct.json).

---

## 7. WI5 — out-of-distribution structural attacks

Structural OOD variants of each topology, generated by
[`code/synth/synth_b2b_injection_ood.py`](../code/synth/synth_b2b_injection_ood.py).
The paper's canonical STREAM-BSG model (seed = 42, 49 features, trained on the
ID injection distribution) is applied *without retraining* to the OOD test
split. Sanity check: the same retrained-from-scratch model reproduces the
paper's ID headlines bit-for-bit (|Δ| = 0.0000 on all four aggregates) before
being scored against OOD.

OOD variants:

- **T1** 2-buyer shell (was 1-buyer)
- **T2** 14-day dup window, 5 % amount tolerance (was 300 s, 1 %)
- **T3** 3-tx gradient over ~20 d / ~10 d / ~1 d (was single-row abrupt shift)
- **T4** ring size 4 (was 3)
- **T5** bank transit across 2 tx (was single-row abrupt change)

Total row count is identical to ID (591,721) since T3/T5's extra labels come from *retro-mutating existing rows*, not inserting new ones (Option A). 13 rows (0.14 % of the 9,626 OOD-labeled rows) have their topology label overwritten by a later-processed topology's retro-mutation.

### 7.1 Aggregate — ID paper vs. OOD (frozen ID model)

| Metric | ID (paper) | OOD | Δ |
|---|---:|---:|---:|
| ROC-AUC | 0.8885 | **0.7572** | −0.131 |
| PR-AUC | 0.6778 | **0.4577** | −0.220 |
| F1_best | 0.7767 | **0.5634** | −0.213 |
| Recall@5%FPR | 0.7111 | **0.5074** | −0.204 |
| p99 latency | 0.37 ms | 0.48 ms | — |

**STREAM-BSG on OOD still beats every ID-tuned baseline:** aggregate ROC-AUC 0.757 (OOD) > XGB-tabular 0.686 (ID) > GraphSAGE 0.687 (ID) > LR 0.652 (ID).

### 7.2 Per-topology recall — OOD, two operating points

| Topology | n@ID | n@OOD | r@ID_5%FPR | r@OOD_5%FPR | Δ | r@OOD_F1opt |
|---|---:|---:|---:|---:|---:|---:|
| T1 vendor_injection | 227 | 227 | 1.000 | 0.529 | −0.471 | 0.141 |
| T2 invoice_cycling (row) | 360 | 397 | 0.153 | 0.018 | −0.135 | 0.000 |
| T2 invoice_cycling (event) | 180 | 217 | 0.267 | 0.032 | −0.234 | 0.000 |
| T3 payment_term_manip | 142 | 356 | 0.958 | 0.584 | −0.373 | 0.517 |
| T4 shell_supplier_ring | 186 | 184 | 1.000 | **0.995** | −0.005 | 0.984 |
| T5 wire_redirection | 189 | 332 | 0.958 | 0.726 | −0.232 | 0.361 |

n@OOD is 1.6× – 2.7× n@ID for T2/T3/T5 because those OOD variants label multiple rows per event; per-topology *recall* is still a directly comparable proportion.

**Threshold columns:** `r@OOD_5%FPR` recalibrates the operating threshold on OOD scores (defender adjusts to the new attack distribution). `r@OOD_F1opt` uses the ID-val-selected threshold unchanged (deployed-model, no recalibration).

**Interpretation for the paper:**

- **T4 generalizes exactly** — ring size 4 is invariant under STREAM-BSG's structural signatures. Recall 0.995 without any exposure to the OOD variant during training.
- **T5 degrades modestly** — the transit-bank OOD variant dilutes the abrupt-change signal but the final transition still fires `feat_bank_change_on_this_row`.
- **T3 halves** — the gradient dilutes each individual row's z-score anomaly; the model still catches the final ramp step.
- **T1 halves** — 2-buyer shell weakens `feat_supplier_unique_buyers`; still 0.53 recall on an unseen coordinated-shell variant.
- **T2 collapses** — floor-level on both operating points, consistent with T2 being STREAM-BSG's weakest topology at ID.

Source: [`ablation_c_ood_injection.json`](ablation_c_ood_injection.json).

---

## 8. Source-file index

Every table above pulls numbers from files in this directory. Full list:

### Canonical ID results
| Content | File |
|---|---|
| LR (tabular) with-synth | [`lr_results.json`](lr_results.json) |
| XGB-tabular with-synth | [`xgb_tabular_results.json`](xgb_tabular_results.json) |
| GraphSAGE (seed 42) with-synth | [`graphsage_results.json`](graphsage_results.json) |
| STREAM-BSG headline (49 features, seed 42) | [`streambsg_results.json`](streambsg_results.json) |
| Master consolidated comparison | [`all_baselines_comparison.json`](all_baselines_comparison.json) |
| CSV: Table I accuracy | [`table_accuracy.csv`](table_accuracy.csv) |
| CSV: Table II latency | [`table_latency.csv`](table_latency.csv) |
| CSV: Table III per-topology | [`table_per_topology.csv`](table_per_topology.csv) |

### No-synth-leakage results
| Content | File |
|---|---|
| Leakage analysis report (audit + Table VII writeup) | [`leakage_analysis_report.md`](leakage_analysis_report.md) |
| LR no-synth | [`lr_results_no_synth_leak.json`](lr_results_no_synth_leak.json) |
| XGB-tab no-synth | [`xgb_tabular_no_synth_leak.json`](xgb_tabular_no_synth_leak.json) |
| STREAM-BSG no-synth-derived | [`streambsg_no_synth_leak.json`](streambsg_no_synth_leak.json) |

### Seed stability
| Content | File |
|---|---|
| STREAM-BSG 5-seed | [`seed_stability.json`](seed_stability.json) |
| GraphSAGE seed 123 | [`graphsage_seed123.json`](graphsage_seed123.json) |
| GraphSAGE seed 456 | [`graphsage_seed456.json`](graphsage_seed456.json) |
| GraphSAGE seed 789 | [`graphsage_seed789.json`](graphsage_seed789.json) |
| GraphSAGE seed 2026 | [`graphsage_seed2026.json`](graphsage_seed2026.json) |
| GraphSAGE seed 314 | [`graphsage_seed314.json`](graphsage_seed314.json) |
| GraphSAGE seed 1729 | [`graphsage_seed1729.json`](graphsage_seed1729.json) |

### Ablations
| Content | File |
|---|---|
| WI1 — no topology-targeted features | [`ablation_a_no_topology_feats.json`](ablation_a_no_topology_feats.json) |
| WI3 — 0.5 % injection rate | [`ablation_0p5pct/streambsg_0p5pct.json`](ablation_0p5pct/streambsg_0p5pct.json) |
| WI5 — OOD structural (cross-eval) | [`ablation_c_ood_injection.json`](ablation_c_ood_injection.json) |

### Smoke-test artifacts (pipeline validation, not paper source)
| Content | File |
|---|---|
| LR smoke | [`lr_smoketest.json`](lr_smoketest.json) |
| XGB-tabular smoke | [`xgb_tabular_smoketest.json`](xgb_tabular_smoketest.json) |

### Documentation
| Content | File |
|---|---|
| Topology definitions (WI6) | [`../paper/topology_definitions.md`](../paper/topology_definitions.md) |
| 49-feature table (WI7) | [`../paper/feature_table.md`](../paper/feature_table.md) |

---

*Generated 2026-07-07 following completion of all seven camera-ready work items. All experimental compute complete; safe to freeze for submission.*
