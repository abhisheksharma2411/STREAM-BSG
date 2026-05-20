# STREAM-BSG — Project Context for Claude Code

## What this project is

STREAM-BSG (Streaming Buyer-Supplier Graph) is a real-time fraud detection system for B2B payment networks. It maintains an incremental buyer-supplier graph as transactions stream in, computes 47 graph features online, and classifies fraud with XGBoost. Target latency: sub-100ms p99 end-to-end.

This repo accompanies a paper submitted to IEEE InC4 2026. The paper makes three claims this code must support:
1. A taxonomy of 5 B2B fraud topologies (formalized below)
2. A streaming graph architecture with sub-100ms p99 latency
3. Empirical evaluation showing STREAM-BSG beats tabular baselines and matches GraphSAGE accuracy at lower inference latency

## The 5 fraud topologies (T1–T5)

**T1 — Vendor Injection:**
A new supplier receives concentrated payments from a single buyer, then disappears. Detection signature: `supplier_age` low, `supplier_edge_count == 1`, payment volume concentrated.

**T2 — Invoice Cycling:**
Same buyer-supplier pair sees duplicate-attribute invoices within a short window. Detection signature: `duplicate_invoice_count_in_window > 0`, `invoice_amount_collision_in_window > 0`.

**T3 — Payment-Term Manipulation:**
Abrupt shift in payment terms on an established buyer-supplier edge. Detection signature: `payment_term_zscore` vs edge history exceeds threshold.

**T4 — Shell-Supplier Ring:**
Buyer connected to multiple low-history suppliers sharing structural properties. Detection signature: `subgraph_size`, `supplier_age_distribution` low, cycle in 2-hop neighborhood.

**T5 — Wire Redirection (BEC):**
Established edge sees abrupt change in supplier banking attributes, followed by high-value payment. Detection signature: `supplier_attribute_change_recent == 1 AND amount > p95 of edge history`.

## The 47-feature design

**15 node features** (computed for each transaction's buyer and supplier):
- `buyer_age_days`, `supplier_age_days`
- `buyer_total_volume_30d`, `supplier_total_volume_30d`
- `buyer_tx_count_30d`, `supplier_tx_count_30d`
- `buyer_unique_suppliers`, `supplier_unique_buyers`
- `buyer_avg_amount`, `supplier_avg_amount`
- `buyer_amount_std`, `supplier_amount_std`
- `buyer_attribute_stability`, `supplier_attribute_stability`
- `buyer_first_seen_recency`

**18 edge features** (the buyer-supplier edge's history):
- `edge_age_days`, `edge_tx_count`, `edge_total_volume`
- `edge_avg_amount`, `edge_amount_std`, `edge_amount_zscore_current`
- `payment_term_mean`, `payment_term_std`, `payment_term_zscore_current`
- `duplicate_invoice_count_in_window`, `invoice_amount_collision_in_window`
- `supplier_attribute_change_count`, `supplier_attribute_change_recent` (binary)
- `bank_change_recent` (binary)
- `days_since_last_tx`, `tx_cadence_score`
- `amount_p95_history`, `amount_above_p95` (binary)

**14 subgraph features** (1-hop and 2-hop neighborhood):
- `buyer_1hop_supplier_count`, `buyer_2hop_supplier_count`
- `buyer_avg_supplier_age`, `buyer_supplier_age_std`, `buyer_supplier_age_min`
- `buyer_2hop_supplier_age_min`
- `has_cycle_2hop`, `max_path_length_2hop`
- `shell_supplier_density` (fraction of buyer's suppliers with age < threshold)
- `ring_detection_score`
- `subgraph_amount_concentration`, `subgraph_volume_velocity`
- `subgraph_attribute_homogeneity`, `subgraph_density`

**Feature implementation rules:**
- Each feature must be computable in O(1) or O(degree) — never a full graph scan.
- For batch evaluation, group-by aggregations on the dataframe are fine.
- For streaming (production), state lives in Redis; for evaluation, in-memory state is OK.
- All features computed using ONLY data available at transaction time (no future leakage).

## Common interface for all baselines and STREAM-BSG

```python
def train_and_evaluate(
    data_path: str,                    # path to labeled parquet
    output_path: str,                  # path to write results JSON
    seed: int = 42,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> dict:
    """Returns metrics dict and writes JSON to output_path."""
```

**Results JSON schema:**

```json
{
  "method": "lr | xgb_tabular | graphsage | streambsg",
  "metrics": {
    "roc_auc": 0.XXX,
    "pr_auc": 0.XXX,
    "f1_best": 0.XX,
    "recall_at_5pct_fpr": 0.XX
  },
  "latency_ms": {
    "p50": null,
    "p95": null,
    "p99": null
  },
  "n_train": 0,
  "n_test": 0,
  "config": {},
  "timestamp": "ISO-8601"
}
```

The label column is `fraud_injected` (binary 0/1). The topology column is `fraud_topology` (T1–T5 or NULL) — used for per-topology analysis, NOT for training.

## Data flow

```
IEEE-CIS CSV (Kaggle download → data/)
  └─→ synth_b2b_injection.py → ieee_cis_with_synthetic_b2b.parquet
        ├─→ lr_baseline.py        → results/lr_results.json
        ├─→ xgb_tabular.py        → results/xgb_tabular_results.json
        ├─→ graphsage_baseline.py → results/graphsage_results.json
        └─→ features.py → 47-feature parquet
              └─→ classifier.py   → results/streambsg_results.json
```

## File structure (what's done, what's TODO)

```
code/
├── synth/synth_b2b_injection.py    # DONE
├── figures/fig1_taxonomy.py         # DONE — do not modify
├── figures/fig2_architecture.py     # DONE — do not modify
├── stream_bsg/
│   ├── features.py                  # TODO
│   ├── classifier.py                # TODO
│   └── pipeline.py                  # TODO (optional, streaming demo)
├── baselines/
│   ├── lr_baseline.py               # TODO
│   ├── xgb_tabular.py               # TODO
│   └── graphsage_baseline.py        # TODO (optional, complex)
└── eval/
    ├── metrics.py                   # TODO
    └── run_all.py                   # TODO
```

## Train/test split

Chronological split based on `TransactionDT`:
- First 70% → train
- Next 15% → validation (for early stopping / threshold tuning)
- Last 15% → test (held out)

The chronological split is important — random splits leak future information about (buyer, supplier) pairs.

## Coding conventions

- Python 3.10+, type hints on all public functions
- Docstrings (Google style)
- Random seed must be settable for reproducibility (`seed: int = 42`)
- Metrics via `sklearn.metrics`
- No global state; pass config via dataclasses or dicts
- All scripts accept `--input`, `--output`, `--seed` as CLI args

## Latency measurement

For models supporting it (XGBoost-tabular, STREAM-BSG, GraphSAGE), measure inference latency by running prediction on each test row INDIVIDUALLY (not batched), in a tight loop, recording the time. Report p50/p95/p99 in milliseconds. Use `time.perf_counter()` for measurement. Run a warm-up of 100 predictions first.

For Logistic Regression, latency is irrelevant — set the latency fields to null.

## Constraints — do not do these things

- Do NOT change the data schema produced by `synth_b2b_injection.py`
- Do NOT introduce new heavy dependencies without asking (no TensorFlow, no LightGBM — we're standardizing on PyTorch for GNNs, XGBoost for trees, scikit-learn for the rest)
- Do NOT redistribute the IEEE-CIS dataset
- Do NOT hard-code dataset paths; accept them as CLI args
- Do NOT modify `code/figures/` (already used in the paper)
- Do NOT modify `CLAUDE.md`, `README.md`, or `LICENSE` without explicit user approval
- Do NOT use random splits — use chronological splits
- Do NOT leak future information into features

## Testing protocol

For each new file, before running on full data:
1. Run on a smoke-test subset (5,000 rows)
2. Print feature distributions / metric values for sanity check
3. Verify JSON output conforms to the schema above
4. Verify metrics are deterministic across two runs with the same seed
5. Only then run on the full dataset
