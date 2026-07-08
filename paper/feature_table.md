# STREAM-BSG Feature Table — All 49 Streaming Graph Features

Reference specification of every feature emitted by
[`code/stream_bsg/features.py`](../code/stream_bsg/features.py) at commit
`f78b111`.

The paper originally specified 47 features. The camera-ready adds two
row-level change flags (`feat_bank_change_on_this_row`,
`feat_supplier_attr_change_on_this_row`) that fix a latent off-by-one in
the "recent change" features — see the T5 diagnosis note at the end. Total
is now **49 features**.

## Design invariants (apply to every feature)

- **No future leakage.** For a transaction at time `t`, every feature is
  computed from state accumulated by transactions with `TransactionDT < t`
  only (rows tied at the same `TransactionDT` are processed in
  chronological input order and earlier-tied rows do leak into later-tied
  rows at sub-second resolution — accepted as a de-minimis relaxation).
- **O(1) or O(degree) per row.** Every feature update is either a
  constant-time scalar operation or scales only with the current node's
  degree; no full-graph scans.
- **Streaming semantics.** State is maintained per-buyer, per-supplier,
  and per-edge in in-memory dictionaries; the extractor is a single pass
  over the chronologically sorted dataframe.

## Constants

| Symbol | Value | Meaning |
|---|---|---|
| `WINDOW_30D` | 30 × 86400 s | 30-day rolling window (buyer/supplier volume, tx-count) |
| `RECENT_SHORT` | 7 × 86400 s | "Recent" window for attribute/bank change flags |
| `SHELL_SUPPLIER_AGE` | 7 × 86400 s | Age threshold below which a supplier counts as "shell" |
| `EDGE_AMOUNTS_KEEP` | 200 | Ring-buffer size for per-edge amount history (used for p95) |
| `INVOICE_DUPLICATE_TOL` | 0.02 | Relative amount tolerance (±2 %) for T2 duplicate detection |

## 15 Node features (7 buyer + 7 supplier + 1 shared)

| # | Feature | Scope | Formula / definition | Primary target |
|---:|---|---|---|---|
| 1 | `feat_buyer_age_days` | buyer | `(dt − buyer.first_seen_dt) / 86400`; 0 if first tx | general |
| 2 | `feat_supplier_age_days` | supplier | `(dt − supplier.first_seen_dt) / 86400`; 0 if first tx | **T1, T4** |
| 3 | `feat_buyer_total_volume_30d` | buyer | Σ `TransactionAmt` over buyer's transactions in `[dt − WINDOW_30D, dt)` | general |
| 4 | `feat_supplier_total_volume_30d` | supplier | Σ `TransactionAmt` over supplier's transactions in `[dt − WINDOW_30D, dt)` | general |
| 5 | `feat_buyer_tx_count_30d` | buyer | \|{buyer's prior tx in the 30d window}\| | general |
| 6 | `feat_supplier_tx_count_30d` | supplier | \|{supplier's prior tx in the 30d window}\| | general |
| 7 | `feat_buyer_unique_suppliers` | buyer | \|{unique supplier_ids buyer has transacted with, all-time before `dt`}\| | **T4** |
| 8 | `feat_supplier_unique_buyers` | supplier | \|{unique buyer_ids supplier has transacted with, all-time before `dt`}\| | **T1, T4** |
| 9 | `feat_buyer_avg_amount` | buyer | Welford mean of `TransactionAmt` over buyer's prior tx | general |
| 10 | `feat_supplier_avg_amount` | supplier | Welford mean of `TransactionAmt` over supplier's prior tx | general |
| 11 | `feat_buyer_amount_std` | buyer | Welford std (N−1) of `TransactionAmt` over buyer's prior tx | general |
| 12 | `feat_supplier_amount_std` | supplier | Welford std (N−1) of `TransactionAmt` over supplier's prior tx | general |
| 13 | `feat_buyer_attribute_stability` | buyer | mean over tracked attrs `A ∈ {addr1, ProductCD}` of `1 / max(1, \|distinct values seen for A on this buyer\|)`; 1.0 if buyer has no history | general |
| 14 | `feat_supplier_attribute_stability` | supplier | same shape as buyer, over `A ∈ {addr2, P_emaildomain, card1, R_emaildomain}` | **T5** (weak) |
| 15 | `feat_buyer_first_seen_recency` | buyer | `exp(−age_seconds / RECENT_SHORT)`; 1.0 for first-time buyer | general |

## 18 Edge features (per `(buyer, supplier)` pair)

| # | Feature | Formula / definition | Primary target |
|---:|---|---|---|
| 16 | `feat_edge_age_days` | `(dt − edge.first_seen_dt) / 86400`; 0 if first tx on edge | general |
| 17 | `feat_edge_tx_count` | count of prior tx on this edge | general |
| 18 | `feat_edge_total_volume` | Σ `TransactionAmt` on this edge, all prior tx | general |
| 19 | `feat_edge_avg_amount` | Welford mean of edge amounts, prior tx only | general |
| 20 | `feat_edge_amount_std` | Welford std (N−1) of edge amounts, prior tx only | general |
| 21 | `feat_edge_amount_zscore_current` | `(TransactionAmt[i] − edge.mean) / edge.std` if edge.std > 0, else 0 | **T1, T5** |
| 22 | `feat_payment_term_mean` | Welford mean of `synth_payment_term_days` over edge history | general |
| 23 | `feat_payment_term_std` | Welford std of same | general |
| 24 | `feat_payment_term_zscore_current` | `(term[i] − edge.term.mean) / edge.term.std` if term.std > 0, else 0 | **T3** |
| 25 | `feat_duplicate_invoice_count_in_window` | count of prior edge tx amounts within `± INVOICE_DUPLICATE_TOL · \|current_amt\|` of `TransactionAmt[i]`, restricted to the last `WINDOW_DAYS` seconds (CLI arg `--window-days`, default 90) | **T2** |
| 26 | `feat_invoice_amount_collision_in_window` | `1.0` if feature 25 > 0, else `0.0` | **T2** |
| 27 | `feat_supplier_attribute_change_count` | number of times a tracked supplier-side edge attribute (see feature 30) differed from the previous value seen on this edge | **T5** (weak) |
| 28 | `feat_supplier_attribute_change_recent` | `1.0` if `edge.last_attr_change_dt ≥ 0` and `(dt − last_attr_change_dt) < RECENT_SHORT`, else `0.0`. **See "known issue" below** | (superseded by feat 29) |
| 29 | `feat_supplier_attr_change_on_this_row` **(added camera-ready)** | `1.0` if any tracked supplier-side attr on the current row differs from the last value seen on this edge (evaluated BEFORE state update); else `0.0`. Tracked attrs: `{addr2, P_emaildomain}` | supplier-side drift (general) |
| 30 | `feat_bank_change_recent` | `1.0` if `edge.last_bank_change_dt ≥ 0` and `(dt − last_bank_change_dt) < RECENT_SHORT`, else `0.0`. **See "known issue" below** | (superseded by feat 31) |
| 31 | `feat_bank_change_on_this_row` **(added camera-ready)** | `1.0` if `edge.last_bank is not None and edge.last_bank ≠ current_bank` (evaluated BEFORE state update); else `0.0` | **T5** |
| 32 | `feat_days_since_last_tx` | `(dt − edge.last_seen_dt) / 86400`; 0 if first on edge | general |
| 33 | `feat_tx_cadence_score` | `1 / (1 + CV)` where `CV = interarrival.std / interarrival.mean` on this edge; 0 if no history. High = regular cadence | general |
| 34 | `feat_amount_p95_history` | 95th percentile of the last `EDGE_AMOUNTS_KEEP` amounts on this edge (0 if empty) | general |
| 35 | `feat_amount_above_p95` | `1.0` if `TransactionAmt[i] > feat_amount_p95_history and p95 > 0`, else `0.0` | **T5** |

## 14 Subgraph features (1-hop and 2-hop neighborhood of the buyer)

Let `B(b) = set of buyer b's known suppliers` (all-time, prior).
Let `B₂(b) = set of buyers that share ≥1 supplier with b (excluding b itself)`.

| # | Feature | Formula / definition | Primary target |
|---:|---|---|---|
| 36 | `feat_buyer_1hop_supplier_count` | count of `s ∈ B(b)` where `(dt − supplier_s.last_seen_dt) < WINDOW_30D` — active-in-30d suppliers | general |
| 37 | `feat_buyer_2hop_supplier_count` | count of unique suppliers `s' ∉ B(b)` reachable via `B₂(b)` — suppliers of buyers that share a supplier with `b` | **T4** |
| 38 | `feat_buyer_avg_supplier_age` | mean of `(dt − supplier_s.first_seen_dt) / 86400` over `s ∈ B(b)` | **T1, T4** |
| 39 | `feat_buyer_supplier_age_std` | std (population) of same | general |
| 40 | `feat_buyer_supplier_age_min` | min of `(dt − supplier_s.first_seen_dt) / 86400` over `s ∈ B(b)` | **T1, T4** |
| 41 | `feat_buyer_2hop_supplier_age_min` | min supplier age (days) over all suppliers in the 2-hop set (feature 37 domain) | **T4** |
| 42 | `feat_has_cycle_2hop` | `1.0` if any `s ∈ B(b)` has `\|supplier_s.neighbors\| > 1` (some other buyer shares supplier `s`); else `0.0` | **T4** |
| 43 | `feat_max_path_length_2hop` | `2` if `B₂(b) ≠ ∅`; `1` if `B(b) ≠ ∅ ∧ B₂(b) = ∅`; else `0` | general |
| 44 | `feat_shell_supplier_density` | fraction of `B(b)` with `supplier_age_days < SHELL_SUPPLIER_AGE / 86400 = 7 days` | **T4** |
| 45 | `feat_ring_detection_score` | `feat_shell_supplier_density × log(1 + \|B₂(b)\|)` | **T4** |
| 46 | `feat_subgraph_amount_concentration` | Herfindahl-Hirschman index over `b`'s per-supplier cumulative volume: `Σᵢ (volᵢ / Σⱼ volⱼ)²` | **T1** |
| 47 | `feat_subgraph_volume_velocity` | `buyer.win30_total / max(1, buyer_age_days)` | general |
| 48 | `feat_subgraph_attribute_homogeneity` | plurality share of `R_emaildomain` values across `B(b)`: `top_domain_count / Σ_domain_counts` | **T4** |
| 49 | `feat_subgraph_density` | `min(1.0, edges_in_subgraph / (\|B(b)\| · (\|B₂(b)\| + 1)))` where `edges_in_subgraph = Σ_{s ∈ B(b)} \|supplier_s.neighbors\|` | **T4** |

## Feature importance in the trained model (XGBoost gain, seed=42)

Top 20 of 49 by mean gain. `*` = added in camera-ready.

| Rank | Feature | Gain | Primary target |
|---:|---|---:|---|
| 1  | `feat_bank_change_on_this_row` * | 1941.3 | T5 |
| 2  | `feat_supplier_total_volume_30d` | 1374.6 | general |
| 3  | `feat_supplier_unique_buyers` | 839.1 | T1, T4 |
| 4  | `feat_supplier_tx_count_30d` | 667.5 | general |
| 5  | `feat_payment_term_zscore_current` | 501.7 | T3 |
| 6  | `feat_duplicate_invoice_count_in_window` | 453.2 | T2 |
| 7  | `feat_invoice_amount_collision_in_window` | 356.0 | T2 |
| 8  | `feat_amount_above_p95` | 344.4 | T5 |
| 9  | `feat_days_since_last_tx` | 330.8 | general |
| 10 | `feat_edge_tx_count` | 240.4 | general |
| 11 | `feat_supplier_attr_change_on_this_row` * | 230.9 | supplier drift |
| 12 | `feat_supplier_attribute_change_count` | 214.7 | T5 (weak) |
| 13 | `feat_edge_amount_zscore_current` | 198.4 | T1, T5 |
| 14 | `feat_edge_age_days` | 178.7 | general |
| 15 | `feat_payment_term_std` | 173.8 | general |
| 16 | `feat_supplier_avg_amount` | 168.4 | general |
| 17 | `feat_shell_supplier_density` | 167.9 | T4 |
| 18 | `feat_buyer_unique_suppliers` | 162.0 | T4 |
| 19 | `feat_subgraph_attribute_homogeneity` | 158.7 | T4 |
| 20 | `feat_edge_total_volume` | 157.2 | general |

The two features added in the camera-ready (`*`) rank **#1** and **#11**
respectively — the two originally-lagging change flags (features 28 and
30 in this table) fell to rank **47** and **46** out of 49, effectively
dead. See the note below for why.

## Known issue: features 28 & 30 (superseded)

The original paper features `feat_bank_change_recent` and
`feat_supplier_attribute_change_recent` (features 28 and 30 above) fire
`1.0` when a change occurred *previously* on this edge within the last
7 days. They are read from `edge.last_bank_change_dt` /
`edge.last_attr_change_dt` **before** the current row's bank/attributes
are folded into edge state.

This introduced an off-by-one for one-shot events like T5: the change is
recorded *by* the T5 row, so the flag would fire only on the next tx
on that edge — which the synth-injection design never produces. Result:
`feat_bank_change_recent` fired on 4.2% of T5 rows (barely above the 3.9%
non-fraud baseline) despite T5 being defined by construction as a bank
change.

The camera-ready adds features **29** (`feat_supplier_attr_change_on_this_row`)
and **31** (`feat_bank_change_on_this_row`), which compare current-row
values to prior edge state *before* the state update, firing on **100 %**
of T5 rows that hit an established edge (92.6 % of T5 rows in the test
set; the remainder are first-of-edge T5s that have no prior bank to
compare against).

At the 5%-FPR operating point this fix moves T5 recall from **36.5 %**
(47-feature version) to **95.8 %** (49-feature version) and lifts aggregate
ROC-AUC by **2.7 pts**. See §V.C of the camera-ready.
