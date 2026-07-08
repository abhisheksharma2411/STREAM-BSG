# STREAM-BSG Fraud Topologies — Formal Definitions

Reference specification of the five B2B fraud topologies used throughout the
paper. All rules below match the implementation in
[`code/synth/synth_b2b_injection.py`](../code/synth/synth_b2b_injection.py) at
commit `f78b111` (initial camera-ready snapshot).

## Notation

- `TransactionAmt` — original transaction amount (USD, from IEEE-CIS).
- `TransactionDT` — transaction timestamp (seconds since dataset epoch).
- `TransactionID` — original per-row identifier.
- `buyer_id = "B::" + card1 + "|" + card2 + "|" + addr1` (composite string key).
- `supplier_id = "S::" + ProductCD + "|" + R_emaildomain` (composite string key).
- `synth_payment_term_days`, `synth_supplier_bank` — synthetic per-transaction
  attributes added by the injector (see initialization notes below).
- `U(a, b)` — uniform sample on `[a, b)`; `Uint(a, b)` — uniform integer on
  `[a, b]` inclusive.
- `Normal(μ, σ)` — normal sample.

## Global injection budget

For a dataset of `N` transactions and an overall injection rate `r`
(default `r = 0.01`), the injector selects `⌊N · r⌋` row indices uniformly
without replacement from `[0, N)` and partitions them across topologies
using the fixed weights below. Row selection uses `random.Random(seed)`.

| Topology | Weight | Approx count at r=0.01, N≈590K |
|---|---:|---:|
| T1 Vendor Injection         | 0.25 | ≈ 1,476 |
| T2 Invoice Cycling          | 0.20 | ≈ 1,181 (each pair emits an additional dup row) |
| T3 Payment-Term Manipulation | 0.15 | ≈ 885   |
| T4 Shell-Supplier Ring      | 0.20 | ≈ 1,181 |
| T5 Wire Redirection         | 0.20 | ≈ 1,181 |

Weights sum to 1.0; the injector asserts this.

Every mutated row gets `fraud_injected = 1` and `fraud_topology =` the
label listed below. Non-injected rows have `fraud_injected = 0` and
`fraud_topology = ""` (empty string).

## Synthetic-attribute initialization (before any injection)

Two columns do not exist in IEEE-CIS and are synthesized once before
per-row mutation:

- **`synth_payment_term_days`** — initialized (only if the T3 injector runs)
  as `max(1, Normal(30.0, 4.0))` per row. Represents legitimate NET-30 terms
  with modest noise.
- **`synth_supplier_bank`** — initialized (only if the T5 injector runs) as
  `"BANK_" + str(abs(hash(supplier_id)) % 1000)` — a per-supplier bank ID
  taking one of 1,000 possible values, deterministic in `supplier_id`
  (so every transaction with the same supplier gets the same initial bank).

---

## T1 — Vendor Injection

**Pattern.** A brand-new supplier appears, receives concentrated payments
from a single buyer, then never appears again.

**Detection signature (paper).**
`supplier_age_days` very low, `supplier_edge_count == 1`, payment volume in
recent window concentrated on the shell node.

**Per-row mutation rule.** For each target row *i*:

| Field | New value |
|---|---|
| `supplier_id` | `"S::SHELL\|SHELL_" + Uint(100_000, 999_999)` — fresh, one-time-use ID |
| `TransactionAmt` | `TransactionAmt[i] × U(2.0, 5.0)` (2×–5× amount inflation) |
| `fraud_injected` | `1` |
| `fraud_topology` | `"T1_vendor_injection"` |

**Notes.** The shell suffix uses 6-digit uniform integers; collision
probability across ~1.5K injections is negligible. `buyer_id` is not
altered, so the shell supplier appears connected to the original buyer.
Every injected `supplier_id` is therefore both new and singleton.

---

## T2 — Invoice Cycling

**Pattern.** The same `(buyer, supplier)` pair sees duplicate-attribute
invoices within a short time window.

**Detection signature (paper).**
`duplicate_invoice_count_in_window > 0`,
`invoice_amount_collision_in_window > 0`.

**Per-row mutation rule.** For each target row *i*, produce a new
duplicate row `i'` and label BOTH rows as fraud:

Original row *i* (in-place mutation):

| Field | New value |
|---|---|
| `fraud_injected` | `1` |
| `fraud_topology` | `"T2_invoice_cycling"` |

Duplicate row *i'* (appended to dataframe):

| Field | Value |
|---|---|
| `TransactionAmt` | `TransactionAmt[i] × U(0.99, 1.01)` (±1% jitter) |
| `TransactionDT` | `TransactionDT[i] + Uint(1, 300)` (1–300 s later) |
| `TransactionID` | `str(TransactionID[i]) + "_DUP_" + Uint(1000, 9999)` |
| `invoice_id` | equal to the new `TransactionID` above |
| all other fields | copied from row *i*, including `buyer_id`, `supplier_id` |
| `fraud_injected` | `1` |
| `fraud_topology` | `"T2_invoice_cycling"` |

**Notes.** Because both rows are labeled as fraud, this topology
contributes 2× rows per event. The paper reports T2 recall at
**row level** and at **event level** (a T2 event is detected if
either row of the pair is flagged; pairs are grouped by
`(buyer_id, supplier_id, base_TransactionID)` where the base ID is the
prefix before `_DUP_`).

---

## T3 — Payment-Term Manipulation

**Pattern.** Abrupt shift in payment-term distribution for an established
`(buyer, supplier)` edge — e.g., NET-1 against historical NET-30.

**Detection signature (paper).**
`payment_term_zscore` vs edge history exceeds threshold.

**Per-row mutation rule.**

| Field | New value |
|---|---|
| `synth_payment_term_days` | `U(0.5, 2.0)` (anomalously short term) |
| `fraud_injected` | `1` |
| `fraud_topology` | `"T3_payment_term_manipulation"` |

**Notes.** Baseline `synth_payment_term_days` is `max(1, Normal(30, 4))`
(see init above). Injected rows land 15–60 σ below the mean — the
edge z-score is the discriminating signal.

---

## T4 — Shell-Supplier Ring

**Pattern.** A buyer connected to multiple low-history supplier nodes
that share structural properties (same domain pattern, same address, etc.)
suggesting a coordinated shell ring.

**Detection signature (paper).**
Small subgraph size around the buyer, low
`supplier_age_distribution`, cycle in 2-hop neighborhood.

**Per-row mutation rule.** Target rows are grouped by their `buyer_id`;
for each buyer group *G*, a fresh 5-digit `ring_id ← Uint(10_000, 99_999)`
and a pool of `ring_size = 3` synthetic ring suppliers is instantiated:

```
ring_suppliers = [
    "S::RING" + ring_id + "|m0",
    "S::RING" + ring_id + "|m1",
    "S::RING" + ring_id + "|m2",
]
```

Then the *k*-th target row in *G* is assigned:

| Field | New value |
|---|---|
| `supplier_id` | `ring_suppliers[k mod 3]` |
| `fraud_injected` | `1` |
| `fraud_topology` | `"T4_shell_supplier_ring"` |

**Notes.** All rows within the same buyer group share the same three
ring suppliers, producing a 4-cycle in the bipartite graph
(buyer → ring_supplier₀ → buyer → ring_supplier₁ → …). `TransactionAmt`
is not mutated for T4. Ring identity is not shared across buyers —
each buyer with T4-marked rows spawns its own private ring — so the
distinguishing signature is *shell-density around the buyer*, not
cross-buyer collusion.

---

## T5 — Wire Redirection (BEC pattern)

**Pattern.** An established `(buyer, supplier)` edge sees an abrupt
change in supplier banking attributes, followed by a high-value payment.

**Detection signature (paper).**
`supplier_attribute_change_recent == 1 AND amount > p95 of edge history`.

**Per-row mutation rule.**

| Field | New value |
|---|---|
| `synth_supplier_bank` | `"BANK_NEW_" + Uint(10_000, 99_999)` (fresh, one-time-use bank ID) |
| `TransactionAmt` | `TransactionAmt[i] × U(5.0, 15.0)` (5×–15× amount inflation) |
| `fraud_injected` | `1` |
| `fraud_topology` | `"T5_wire_redirection"` |

**Notes.** Because `synth_supplier_bank` is initialized deterministically
from `supplier_id`, the injected `"BANK_NEW_…"` value differs from every
prior bank value on the same edge with probability ≈ 1. The
`edge.last_bank ≠ current_bank` check in
`code/stream_bsg/features.py` (feature
`feat_bank_change_on_this_row`) therefore fires on all T5 rows that hit
an established edge — which is 92.6% of them by construction, since
row indices are drawn uniformly across the full dataset and only ~7% of
uniform-random rows are the first occurrence of their `(buyer, supplier)`
edge in chronological order.

---

## Summary table

| Topology | Rows mutated | `TransactionAmt` factor | New/mutated fields | Fires structural signal? | Fires attribute signal? |
|---|---|---|---|:---:|:---:|
| **T1** Vendor Injection      | in-place  | ×`U(2, 5)` | `supplier_id` → one-time shell | ✓ (age=0, degree=1) | via amount inflation |
| **T2** Invoice Cycling       | in-place + emit dup | dup ×`U(0.99, 1.01)` | dup: `TransactionID`, `TransactionAmt`, `TransactionDT` | — | ✓ (duplicate within window) |
| **T3** Payment-Term Manip    | in-place | unchanged | `synth_payment_term_days ← U(0.5, 2.0)` | — | ✓ (payment-term z-score) |
| **T4** Shell-Supplier Ring   | in-place  | unchanged | `supplier_id` → shared 3-supplier ring per buyer | ✓ (cycle, shell density) | — |
| **T5** Wire Redirection      | in-place | ×`U(5, 15)` | `synth_supplier_bank ← "BANK_NEW_…"` | — | ✓ (bank change + amount) |

Rows added to the dataframe: **T2 only** (one extra row per selected T2
index). Row count post-injection is `N + ⌊N · r⌋ · w_{T2}`.

## Reproducibility

Fixed by `--seed` (default 42). RNG state is a single
`random.Random(seed)` shared across all injectors, plus `np.random.seed(seed)`
for the T3 initializer's `np.random.normal(30, 4, size=N)`. Given the same
input CSV, the same `--inject-rate`, and the same seed, the output parquet is
bitwise identical.
