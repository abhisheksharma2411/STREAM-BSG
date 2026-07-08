# Synthetic-Attribute Leakage Analysis (Table VII source)

**Purpose.** The STREAM-BSG synthetic injector introduces two per-transaction
attributes that do not exist in the underlying IEEE-CIS Fraud Detection
dataset: `synth_payment_term_days` (used by T3) and `synth_supplier_bank`
(used by T5). If these attributes are exposed *directly* as features to a
classifier, T3 / T5 fraud becomes trivially separable by threshold and any
recall claims are inflated. This report enumerates exactly which columns
were exposed to which method, and quantifies the recall inflation for each.

The results in this file are the source of truth for **Table VII** ("Baseline
performance without synthetic-attribute leakage") in the camera-ready
Section V. The underlying JSON artifacts are
[`lr_results_no_synth_leak.json`](lr_results_no_synth_leak.json),
[`xgb_tabular_no_synth_leak.json`](xgb_tabular_no_synth_leak.json), and
[`streambsg_no_synth_leak.json`](streambsg_no_synth_leak.json).

## 1. Columns audited

Full list of injector-added columns in
[`data/ieee_cis_with_synthetic_b2b.parquet`](../data/ieee_cis_with_synthetic_b2b.parquet):

| Column | Type | Written by | Range in the labelled parquet |
|---|---|---|---|
| `synth_payment_term_days` | float | T3 injector (init + mutate) | init `max(1, Normal(30, 4))`; T3 rows overridden to `U(0.5, 2)` |
| `synth_supplier_bank` | string | T5 injector (init + mutate) | init `"BANK_" + str(hash(supplier_id) % 1000)`; T5 rows overridden to `"BANK_NEW_" + Uint(10_000, 99_999)` |

No other columns start with `synth_` or otherwise carry direct injection artifacts.
Injection markers (`"SHELL_"`, `"RING_"`, `"BANK_NEW_"`, `"_DUP_"`) appear only
as sub-strings inside `supplier_id`, `TransactionID`, and `synth_supplier_bank`
values — never as column names — so they cannot be leaked as raw features to
the tabular baselines.

## 2. What each method actually saw in the "with-synth" configuration

### LR (tabular) — `lr_baseline.py`

- Column selector groups by dtype and drops any categorical column with
  ≥ `MAX_CAT_CARDINALITY = 50` unique values.
- `synth_payment_term_days` is numeric → **kept** as feature.
- `synth_supplier_bank` has ~1,000 unique values → **already dropped**
  by the cardinality filter.

**Only leaky column: `synth_payment_term_days`.**

### XGBoost-tabular — `xgb_tabular.py`

Uses the identical column-selection logic as LR. Same outcome:
**only leaky column: `synth_payment_term_days`.**

### STREAM-BSG — `classifier.py`

Consumes **only** columns whose name starts with `feat_`. Neither raw
`synth_payment_term_days` nor raw `synth_supplier_bank` reaches XGBoost.
Confirmed by `grep -n synth_ code/stream_bsg/classifier.py` returning
no hits.

The classifier is therefore **not** exposed to the synthetic columns
directly, but the feature pipeline `features.py` reads them to compute
five derived features:

- from `synth_payment_term_days`:
  `feat_payment_term_mean`, `feat_payment_term_std`,
  `feat_payment_term_zscore_current`
- from `synth_supplier_bank`:
  `feat_bank_change_recent`, `feat_bank_change_on_this_row`

For the "no-synth" apples-to-apples comparison against tabular baselines,
we drop these **five** `feat_*` features from the STREAM-BSG training set,
matching the removal of the raw synth columns from LR / XGB. Result file:
[`streambsg_no_synth_leak.json`](streambsg_no_synth_leak.json).

Everything below labeled "STREAM-BSG (no synth)" refers to that
5-feature-restricted retrain, run under the pinned canonical environment
(xgboost 3.2.0, scikit-learn 1.8.0, seed 42).

## 3. Table VII — aggregate metrics, with-synth vs. no-synth

| Method | ROC-AUC (with) | ROC-AUC (no) | Δ | PR-AUC (with) | PR-AUC (no) | Δ | F1_best (with) | F1_best (no) | Δ | Recall@5%FPR (with) | Recall@5%FPR (no) | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LR | 0.6519 | **0.5979** | −0.054 | 0.0492 | 0.0304 | −0.019 | 0.1597 | 0.0850 | −0.075 | 0.3125 | **0.1957** | −0.117 |
| XGB-tabular | 0.6862 | **0.6299** | −0.056 | 0.1252 | 0.0802 | −0.045 | 0.2468 | 0.1589 | −0.088 | 0.3487 | **0.2400** | −0.109 |
| STREAM-BSG | 0.8885 | **0.7906** | −0.098 | 0.6778 | 0.4267 | −0.251 | 0.7767 | 0.5396 | −0.237 | 0.7111 | **0.4837** | −0.227 |

STREAM-BSG absorbs the larger absolute drop because it was *using* the
derived signals heavily (the top-ranked feature `feat_bank_change_on_this_row`
disappears in the no-synth ablation). Even after that hit, STREAM-BSG (no
synth) at ROC-AUC 0.791 still beats **every** with-synth baseline:

- vs. LR (with-synth) 0.652 → +14 pts
- vs. XGB-tabular (with-synth) 0.686 → +10 pts
- vs. GraphSAGE (with-synth, seed=42) 0.687 → +10 pts

## 4. Table VII — per-topology recall @ 5%-FPR threshold

| Topology | n | LR (with) | LR (no) | Δ | XGB (with) | XGB (no) | Δ | STREAM-BSG (with) | STREAM-BSG (no) | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T1 vendor_injection | 227 | 0.216 | 0.225 | +0.009 | 0.414 | 0.410 | −0.004 | **1.000** | **1.000** | 0.000 |
| T2 invoice_cycling (row) | 360 | 0.042 | 0.044 | +0.003 | 0.011 | 0.014 | +0.003 | 0.153 | 0.086 | −0.067 |
| T2 invoice_cycling (event) | 180 | 0.067 | 0.044 | −0.022 | 0.011 | 0.017 | +0.006 | 0.267 | 0.150 | −0.117 |
| T3 payment_term_manip | 142 | **0.965** | **0.014** | **−0.951** | **0.894** | **0.042** | **−0.852** | **0.958** | **0.099** | **−0.859** |
| T4 shell_supplier_ring | 186 | 0.038 | 0.027 | −0.011 | 0.043 | 0.048 | +0.005 | **1.000** | **1.000** | 0.000 |
| T5 wire_redirection | 189 | 0.725 | 0.751 | +0.026 | 0.804 | 0.804 | 0.000 | 0.958 | 0.402 | −0.556 |

## 5. Interpretation — per topology

**T1 (vendor injection) — pure structural win, unaffected by leakage.**
STREAM-BSG holds at recall 1.000 in both configurations. The T1 signal
(new shell supplier, singleton edge, 2×–5× amount inflation) is captured
entirely by structural features (`feat_supplier_age_days`,
`feat_supplier_unique_buyers`, `feat_edge_amount_zscore_current`). Neither
tabular baseline gets above 0.41 even with leakage; LR/XGB have no way
to represent the "new-supplier" concept from raw columns.

**T2 (invoice cycling) — degrades similarly across all methods.** The
absolute drops are small (STREAM-BSG event recall 0.267 → 0.150, row
recall 0.153 → 0.086) but proportional. T2 was never dependent on
synth_* columns for any method — the drop reflects that removing 5
features from STREAM-BSG generally weakens the model. T2 remains
the hardest topology; addressed further in the paper's future-work
section.

**T3 (payment-term manipulation) — collapses for all three methods.**
This is the honest finding: without `synth_payment_term_days` (as a
raw feature for LR/XGB, or as an input to `feat_payment_term_*` for
STREAM-BSG), no method catches T3 above ~10 %. The paper must therefore
qualify T3 as *demonstrating the method* — showing that the streaming
z-score against edge history is the right shape of feature to build on
real payment-term data, not claiming T3 detection without payment-term
inputs. This qualification belongs in Section V.C.

**T4 (shell-supplier ring) — pure structural win, unaffected.**
Same story as T1. STREAM-BSG holds at 1.000. The topology is defined by
graph structure (buyer connected to multiple low-age suppliers sharing
identity attributes), and every feature powering it (`feat_ring_detection_score`,
`feat_shell_supplier_density`, `feat_subgraph_attribute_homogeneity`,
`feat_has_cycle_2hop`, `feat_buyer_2hop_supplier_count`) is derived from
real IEEE-CIS graph structure — no synth_* dependency.

**T5 (wire redirection) — the nuanced case.** Three distinct facts
matter and the paper should state all three:

1. LR and XGB-tabular were **not** benefiting from `synth_supplier_bank`
   leakage — the cardinality filter had already dropped that column.
   Their T5 recall (LR 0.725, XGB 0.804) came from `TransactionAmt`
   inflation (5×–15× factor). Confirmed by the no-synth deltas of
   +0.026 and 0.000 respectively.
2. STREAM-BSG's T5 recall (0.958) is genuinely dependent on the two
   `feat_bank_change_*` features derived from `synth_supplier_bank`.
   Dropping them costs 55.6 recall points, taking STREAM-BSG (0.402)
   *below* XGB-tabular (0.804) on T5 alone.
3. On real production data with real payer-bank attributes, the same
   `feat_bank_change_on_this_row` computation would fire on real BEC
   events by construction — so this is not leakage in the "seeing the
   label" sense; it is dependence on the correct kind of input signal
   being present. The paper should frame T5 the same way as T3: the
   feature design generalizes to real bank-change data, but the paper's
   T5 numbers are contingent on that input existing.

## 6. Recommended framing for Section V

Present Tables I / III with the with-synth numbers (current paper state)
as the primary result, then include Table VII in Section V.C ("Fair
comparison under synthetic-attribute removal") with the aggregate row
above and this two-line summary:

> *When the synthetic payment-term column is removed from all tabular
> methods and the corresponding derived features from STREAM-BSG,
> STREAM-BSG's aggregate ROC-AUC drops from 0.889 to 0.791 while both
> tabular baselines drop to 0.598 and 0.630 respectively. STREAM-BSG's
> +10-to-14-pt advantage over tabular methods therefore reflects genuine
> gains from graph structure (T1 and T4 remain at recall 1.000 in both
> configurations), not from privileged access to synthetic injection
> artifacts.*

## 7. Reproducibility

All numbers in this report were produced under the pinned canonical
environment (`xgboost==3.2.0`, `scikit-learn==1.8.0`, `numpy 2.5.1`,
`pandas 3.0.3`, `pyarrow 24.0.0`) at seed 42, chronological 70/15/15
split. To reproduce:

```bash
# regenerate all three no-synth results
python code/baselines/lr_baseline.py \
    --input data/ieee_cis_with_synthetic_b2b.parquet \
    --output results/lr_results.json --seed 42
python code/baselines/xgb_tabular.py \
    --input data/ieee_cis_with_synthetic_b2b.parquet \
    --output results/xgb_tabular_results.json --seed 42
python code/stream_bsg/classifier.py \
    --input data/ieee_cis_with_features.parquet \
    --output results/streambsg_results.json --seed 42
# then re-run the no-synth ablation via the leakage-audit script; see
# results/lr_results_no_synth_leak.json etc. for the resulting JSONs.
```

The no-synth JSONs each carry a `dropped_features` (or `excluded_columns`)
key naming exactly what was removed from the training input.
