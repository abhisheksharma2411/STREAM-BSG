"""
synth_b2b_injection_ood.py
==========================

Out-of-distribution (OOD) variant of the STREAM-BSG synthetic B2B fraud
injector — same 5 topologies, structurally reshaped so the paper's model
(trained on the in-distribution injector) has not seen them.

OOD variants (specified by reviewer):

  T1  Vendor Injection      : 2 buyers share the same shell supplier
                              (instead of a single-buyer shell) — increases
                              supplier_edge_count to 2, weakening
                              `supplier_unique_buyers` signal.
  T2  Invoice Cycling       : dup delay Uint(1, 14 days) instead of Uint(1, 300s);
                              amount tolerance ±5 % instead of ±1 %.
  T3  Payment-Term Manip    : Gradual shift over 3 consecutive tx on the same
                              edge (~20d, ~10d, ~1d) instead of one abrupt jump.
                              All 3 rows labeled T3.
  T4  Shell-Supplier Ring   : Ring size 4 (instead of 3).
  T5  Wire Redirection      : Bank change spread across 2 consecutive tx on the
                              same edge — the prior tx switches to an
                              intermediate `"BANK_TRANSIT_..."` bank, then
                              the target row completes the transition to
                              `"BANK_NEW_..."`. Both rows labeled T5.

The base module ``synth_b2b_injection.py`` is *not* modified. This script
monkey-patches `base.INJECTORS` and shares its RNG plumbing / dataframe
handling for reproducibility.
"""

from __future__ import annotations

import argparse
import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from code.synth import synth_b2b_injection as base
from code.synth.synth_b2b_injection import FraudTopology

logger = logging.getLogger(__name__)

# Prior-tx-on-edge lookup, indexed by the current dataframe's row ordering.
# Populated once at the start of inject_all_ood; used by T3 and T5 which
# need to find nearest-prior transactions on the same (buyer, supplier) edge.
_EDGE_HISTORY: dict[tuple[str, str], list[int]] = {}


def _build_edge_history(df: pd.DataFrame) -> None:
    """Populate the module-level edge history in chronological order.

    Called once per inject_all_ood run. Assumes df already has buyer_id /
    supplier_id columns (set by build_graph_columns before injection).
    """
    global _EDGE_HISTORY
    _EDGE_HISTORY = defaultdict(list)
    # Preserve current dataframe indices; sort by TransactionDT to establish
    # chronological order on each edge, then append.
    order = df["TransactionDT"].argsort(kind="mergesort")
    buyers = df["buyer_id"].to_numpy()
    suppliers = df["supplier_id"].to_numpy()
    for i in order:
        _EDGE_HISTORY[(buyers[i], suppliers[i])].append(int(i))


# ---------------------------------------------------------------------------
# T1 OOD — 2-buyer shell signature
# ---------------------------------------------------------------------------

def inject_vendor_injection_ood(df: pd.DataFrame, rows: list[int], rng: random.Random) -> pd.DataFrame:
    """T1 OOD: pair target rows across DIFFERENT buyers so each shell
    supplier ends up connected to 2 buyers. Odd row leftover (if any)
    falls back to single-buyer T1 to avoid dropping the injection.
    """
    df = df.copy()
    unpaired: tuple[str, list[int]] | None = None
    for idx in rows:
        b = df.at[idx, "buyer_id"]
        if unpaired is None or unpaired[0] == b:
            # buffer this row; we need a DIFFERENT-buyer row to pair with
            if unpaired is None:
                unpaired = (b, [idx])
            else:
                # same-buyer collision — add to the buffered group so it
                # doesn't get orphaned; may result in 3-tx-per-shell if the
                # next row completes the pair. That's fine.
                unpaired[1].append(idx)
            continue
        # Different buyer — form a 2-buyer shell
        pair_b, pair_indices = unpaired
        shell_suffix = f"SHELL_{rng.randint(100000, 999999)}"
        shell_sid = f"S::SHELL|{shell_suffix}"
        for i in pair_indices + [idx]:
            df.at[i, "supplier_id"] = shell_sid
            df.at[i, "TransactionAmt"] = df.at[i, "TransactionAmt"] * rng.uniform(2.0, 5.0)
            df.at[i, "fraud_injected"] = 1
            df.at[i, "fraud_topology"] = FraudTopology.VENDOR_INJECTION.value
        unpaired = None
    if unpaired is not None:
        _, pair_indices = unpaired
        shell_suffix = f"SHELL_{rng.randint(100000, 999999)}"
        shell_sid = f"S::SHELL|{shell_suffix}"
        for i in pair_indices:
            df.at[i, "supplier_id"] = shell_sid
            df.at[i, "TransactionAmt"] = df.at[i, "TransactionAmt"] * rng.uniform(2.0, 5.0)
            df.at[i, "fraud_injected"] = 1
            df.at[i, "fraud_topology"] = FraudTopology.VENDOR_INJECTION.value
    return df


# ---------------------------------------------------------------------------
# T2 OOD — 14-day dup window, 5 % amount tolerance
# ---------------------------------------------------------------------------

_T2_WINDOW_SECONDS = 14 * 86400
_T2_AMT_LOW, _T2_AMT_HIGH = 0.95, 1.05


def inject_invoice_cycling_ood(df: pd.DataFrame, rows: list[int], rng: random.Random) -> pd.DataFrame:
    """T2 OOD: dup emitted anywhere within the next 14 days on the same
    edge; amount jittered by ±5 %.
    """
    df = df.copy()
    new_rows = []
    for idx in rows:
        dup = df.loc[idx].copy()
        dup["TransactionAmt"] = dup["TransactionAmt"] * rng.uniform(_T2_AMT_LOW, _T2_AMT_HIGH)
        dup["TransactionDT"] = dup["TransactionDT"] + rng.randint(1, _T2_WINDOW_SECONDS)
        dup["TransactionID"] = f"{dup['TransactionID']}_DUP_{rng.randint(1000, 9999)}"
        dup["invoice_id"] = str(dup["TransactionID"])
        dup["fraud_injected"] = 1
        dup["fraud_topology"] = FraudTopology.INVOICE_CYCLING.value
        new_rows.append(dup)
        df.at[idx, "fraud_injected"] = 1
        df.at[idx, "fraud_topology"] = FraudTopology.INVOICE_CYCLING.value
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# T3 OOD — gradual payment-term shift over 3 consecutive edge tx
# ---------------------------------------------------------------------------

_T3_GRADIENT = [20.0, 10.0, 1.0]  # terms for (t-2), (t-1), t


def inject_payment_term_manipulation_ood(df: pd.DataFrame, rows: list[int], rng: random.Random) -> pd.DataFrame:
    """T3 OOD: for each target row, override its payment-term day AND the
    two most-recent prior tx on the same edge to produce a monotonic ramp
    from ~20d → ~10d → ~1d. Fewer than 3 prior tx on the edge falls back
    to as many as are available.
    """
    df = df.copy()
    if "synth_payment_term_days" not in df.columns:
        df["synth_payment_term_days"] = np.random.normal(30.0, 4.0, size=len(df)).clip(min=1)
    for idx in rows:
        b = df.at[idx, "buyer_id"]
        s = df.at[idx, "supplier_id"]
        hist = _EDGE_HISTORY.get((b, s), [])
        try:
            pos = hist.index(idx)
        except ValueError:
            # target row not in the pre-injection history (shouldn't happen);
            # fall back to single-row modification
            df.at[idx, "synth_payment_term_days"] = _T3_GRADIENT[-1] + rng.uniform(-0.5, 0.5)
            df.at[idx, "fraud_injected"] = 1
            df.at[idx, "fraud_topology"] = FraudTopology.PAYMENT_TERM_MANIP.value
            continue
        # Take up to 3 rows ending at pos
        window = hist[max(0, pos - 2):pos + 1]
        grad = _T3_GRADIENT[-len(window):]
        for row_i, target_day in zip(window, grad):
            df.at[row_i, "synth_payment_term_days"] = target_day + rng.uniform(-0.5, 0.5)
            df.at[row_i, "fraud_injected"] = 1
            df.at[row_i, "fraud_topology"] = FraudTopology.PAYMENT_TERM_MANIP.value
    return df


# ---------------------------------------------------------------------------
# T4 OOD — ring size 4 instead of 3
# ---------------------------------------------------------------------------

def inject_shell_supplier_ring_ood(df: pd.DataFrame, rows: list[int], rng: random.Random) -> pd.DataFrame:
    return base.inject_shell_supplier_ring(df, rows, rng, ring_size=4)


# ---------------------------------------------------------------------------
# T5 OOD — bank change across 2 consecutive tx on the same edge
# ---------------------------------------------------------------------------

def inject_wire_redirection_ood(df: pd.DataFrame, rows: list[int], rng: random.Random) -> pd.DataFrame:
    """T5 OOD: prior tx on the same edge is retro-mutated to an intermediate
    `"BANK_TRANSIT_..."` value, and the target tx completes the transition
    to `"BANK_NEW_..."` with the high-amount signature. Both rows are
    labeled T5. If there is no prior tx on the edge, fall back to a
    single-row T5 injection.
    """
    df = df.copy()
    if "synth_supplier_bank" not in df.columns:
        df["synth_supplier_bank"] = "BANK_" + df["supplier_id"].apply(lambda x: str(abs(hash(x)) % 1000))
    for idx in rows:
        b = df.at[idx, "buyer_id"]
        s = df.at[idx, "supplier_id"]
        hist = _EDGE_HISTORY.get((b, s), [])
        try:
            pos = hist.index(idx)
        except ValueError:
            pos = -1
        new_bank = f"BANK_NEW_{rng.randint(10000, 99999)}"
        df.at[idx, "synth_supplier_bank"] = new_bank
        df.at[idx, "TransactionAmt"] = df.at[idx, "TransactionAmt"] * rng.uniform(5.0, 15.0)
        df.at[idx, "fraud_injected"] = 1
        df.at[idx, "fraud_topology"] = FraudTopology.WIRE_REDIRECTION.value
        # Retro-mutate the immediately prior edge tx to a transit bank
        if pos > 0:
            transit_idx = hist[pos - 1]
            transit_bank = f"BANK_TRANSIT_{rng.randint(10000, 99999)}"
            df.at[transit_idx, "synth_supplier_bank"] = transit_bank
            df.at[transit_idx, "fraud_injected"] = 1
            df.at[transit_idx, "fraud_topology"] = FraudTopology.WIRE_REDIRECTION.value
    return df


# ---------------------------------------------------------------------------
# Custom orchestrator (mirrors base.inject_all, plus edge-history precompute)
# ---------------------------------------------------------------------------

INJECTORS_OOD = {
    FraudTopology.VENDOR_INJECTION:    inject_vendor_injection_ood,
    FraudTopology.INVOICE_CYCLING:     inject_invoice_cycling_ood,
    FraudTopology.PAYMENT_TERM_MANIP:  inject_payment_term_manipulation_ood,
    FraudTopology.SHELL_SUPPLIER_RING: inject_shell_supplier_ring_ood,
    FraudTopology.WIRE_REDIRECTION:    inject_wire_redirection_ood,
}


def inject_all_ood(df: pd.DataFrame, config: base.InjectionConfig) -> pd.DataFrame:
    """OOD orchestrator: same row-selection logic as base.inject_all, but
    builds an edge-history index before invoking any injector and dispatches
    through INJECTORS_OOD.
    """
    rng = random.Random(config.seed)
    np.random.seed(config.seed)

    df = base.build_graph_columns(df)
    df["fraud_injected"] = 0
    df["fraud_topology"] = ""

    logger.info("Building edge-history index for T3/T5 gradual OOD injections...")
    _build_edge_history(df)

    n_total = len(df)
    n_to_inject = int(n_total * config.overall_rate)
    logger.info(f"OOD injection: {n_to_inject} target rows out of {n_total} at rate={config.overall_rate:.4f}")

    candidate_indices = list(df.index)
    rng.shuffle(candidate_indices)
    chosen = candidate_indices[:n_to_inject]

    cursor = 0
    for topo, weight in config.topology_weights.items():
        n_this = int(n_to_inject * weight)
        slice_ = chosen[cursor:cursor + n_this]
        cursor += n_this
        if not slice_:
            continue
        logger.info(f"  {topo.value}: {len(slice_)} rows")
        df = INJECTORS_OOD[topo](df, slice_, rng)

    n_inj = int(df["fraud_injected"].sum())
    logger.info(f"OOD final: {n_inj} injected fraud, {len(df)} total rows after any T2 duplicates")
    return df


def main():
    p = argparse.ArgumentParser(description="OOD (structural) synthetic B2B fraud injection")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--inject-rate", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rows", type=int, default=None,
                   help="Optional cap on input rows (smoke tests).")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    logger.info(f"Loading {args.input}")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df):,} transactions, {df.shape[1]} columns")

    if args.max_rows is not None and args.max_rows < len(df):
        if "TransactionDT" in df.columns:
            df = df.sort_values("TransactionDT").head(args.max_rows).reset_index(drop=True)
        else:
            df = df.head(args.max_rows).reset_index(drop=True)

    cfg = base.InjectionConfig(overall_rate=args.inject_rate, seed=args.seed)
    df = inject_all_ood(df, cfg)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    logger.info(f"Wrote {args.output} ({len(df):,} rows)")

    print("\n=== OOD injection summary ===")
    print(df["fraud_topology"].value_counts())
    print(f"\nInjected fraud (fraud_injected=1): {int(df['fraud_injected'].sum())}")


if __name__ == "__main__":
    main()
