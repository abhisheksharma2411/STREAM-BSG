"""
synth_b2b_injection.py
======================

Synthetic B2B fraud pattern injection over the IEEE-CIS Fraud Detection dataset
(or any tabular transaction dataset with buyer/merchant identifiers).

Re-casts a B2C transaction stream as a buyer-supplier graph and injects synthetic
fraud patterns matching the five topologies in the STREAM-BSG taxonomy:

  T1. Vendor Injection
  T2. Invoice Cycling
  T3. Payment-Term Manipulation
  T4. Shell-Supplier Ring
  T5. Wire Redirection (BEC pattern)

USAGE:
    python synth_b2b_injection.py \
        --input data/ieee-cis/train_transaction.csv \
        --output data/synth_b2b.parquet \
        --inject-rate 0.01 \
        --seed 42

The injected fraud is *labeled*. Original fraud labels are preserved; injected
fraud receives label fraud_injected=1. Models are evaluated on injected fraud,
on original fraud, or jointly.

ASSUMPTIONS DOCUMENTED IN THE PAPER:
  - Buyer node = card1 || card2 || addr1 (composite key)
  - Supplier node = ProductCD || R_emaildomain (composite key)
  - Invoice node = TransactionID
  - Edge (buyer, supplier) carries: total_volume, count, first_seen, last_seen,
    payment_term_dist (proxy: TransactionDT delta from order to settlement)

OPEN-SOURCE: This file will ship with the github.com/{user}/stream-bsg release.
LICENSE: MIT
"""

import argparse
import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Fraud topology types (mirror Section III taxonomy in the paper)
# -----------------------------------------------------------------------------

class FraudTopology(str, Enum):
    VENDOR_INJECTION = "T1_vendor_injection"
    INVOICE_CYCLING = "T2_invoice_cycling"
    PAYMENT_TERM_MANIP = "T3_payment_term_manipulation"
    SHELL_SUPPLIER_RING = "T4_shell_supplier_ring"
    WIRE_REDIRECTION = "T5_wire_redirection"


@dataclass
class InjectionConfig:
    """Configuration for synthetic fraud injection."""
    overall_rate: float = 0.01  # fraction of transactions to be marked injected
    topology_weights: dict = field(default_factory=lambda: {
        FraudTopology.VENDOR_INJECTION: 0.25,
        FraudTopology.INVOICE_CYCLING: 0.20,
        FraudTopology.PAYMENT_TERM_MANIP: 0.15,
        FraudTopology.SHELL_SUPPLIER_RING: 0.20,
        FraudTopology.WIRE_REDIRECTION: 0.20,
    })
    seed: int = 42

    def __post_init__(self):
        total = sum(self.topology_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Topology weights must sum to 1.0, got {total}")


# -----------------------------------------------------------------------------
# Graph construction from tabular transactions
# -----------------------------------------------------------------------------

def derive_buyer_id(row: pd.Series) -> str:
    """Buyer identity = card1 + card2 + addr1 (composite for stability)."""
    parts = [
        str(row.get("card1", "X")),
        str(row.get("card2", "X")),
        str(row.get("addr1", "X")),
    ]
    return "B::" + "|".join(parts)


def derive_supplier_id(row: pd.Series) -> str:
    """Supplier identity = ProductCD + R_emaildomain (composite)."""
    parts = [
        str(row.get("ProductCD", "X")),
        str(row.get("R_emaildomain", "X")),
    ]
    return "S::" + "|".join(parts)


def build_graph_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add buyer_id, supplier_id, invoice_id columns to the dataframe in place."""
    logger.info("Constructing buyer-supplier graph mapping...")
    df = df.copy()
    # Cast TransactionID to string up front: T2 invoice-cycling injects rows with
    # composite string IDs (e.g. "3001173_DUP_7814"), and a mixed int/str column
    # cannot be serialized to parquet.
    df["TransactionID"] = df["TransactionID"].astype(str)
    df["buyer_id"] = df.apply(derive_buyer_id, axis=1)
    df["supplier_id"] = df.apply(derive_supplier_id, axis=1)
    df["invoice_id"] = df["TransactionID"]
    return df


# -----------------------------------------------------------------------------
# Injection routines (one per topology)
#
# Each routine:
#   - takes the dataframe + a target row index set
#   - mutates rows to look like the named fraud topology
#   - sets fraud_injected=1 and fraud_topology=<topology label>
#
# NB: For InC4-Lite paper we use SIMPLE perturbations sufficient to demonstrate
# topology-aware features beat tabular features. Extended journal version (BigData)
# will use a richer behavioral simulator.
# -----------------------------------------------------------------------------

def inject_vendor_injection(df: pd.DataFrame, rows: list[int], rng: random.Random) -> pd.DataFrame:
    """T1 — Vendor Injection.
    Pattern: a brand-new supplier appears, receives concentrated payments from a
    single buyer, then never appears again. Detection signature: supplier_age_days
    very low, supplier_edge_count==1, payment_volume_in_window concentrated.

    Implementation: clone target rows, rewrite supplier_id to a synthetic 'shell'
    supplier ID with a one-time-use suffix; bump TransactionAmt by 2-5x.
    """
    df = df.copy()
    for idx in rows:
        shell_suffix = f"SHELL_{rng.randint(100000, 999999)}"
        df.at[idx, "supplier_id"] = f"S::SHELL|{shell_suffix}"
        df.at[idx, "TransactionAmt"] = df.at[idx, "TransactionAmt"] * rng.uniform(2.0, 5.0)
        df.at[idx, "fraud_injected"] = 1
        df.at[idx, "fraud_topology"] = FraudTopology.VENDOR_INJECTION.value
    return df


def inject_invoice_cycling(df: pd.DataFrame, rows: list[int], rng: random.Random) -> pd.DataFrame:
    """T2 — Invoice Cycling.
    Pattern: same (buyer, supplier) pair sees duplicate-attribute invoices within
    a short time window. Detection signature: dup_invoice_count_in_window > 0,
    invoice_amount_collision_in_window > 0.

    Implementation: duplicate target rows with near-identical amounts (+/- 1%)
    and TransactionDT shifted by 1-300 seconds.
    """
    df = df.copy()
    new_rows = []
    for idx in rows:
        dup = df.loc[idx].copy()
        # Tiny amount perturbation to look like an "adjusted" duplicate
        dup["TransactionAmt"] = dup["TransactionAmt"] * rng.uniform(0.99, 1.01)
        dup["TransactionDT"] = dup["TransactionDT"] + rng.randint(1, 300)
        dup["TransactionID"] = f"{dup['TransactionID']}_DUP_{rng.randint(1000, 9999)}"
        dup["invoice_id"] = str(dup["TransactionID"])
        dup["fraud_injected"] = 1
        dup["fraud_topology"] = FraudTopology.INVOICE_CYCLING.value
        new_rows.append(dup)
        # Also mark original as part of the cycling pattern
        df.at[idx, "fraud_injected"] = 1
        df.at[idx, "fraud_topology"] = FraudTopology.INVOICE_CYCLING.value
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return df


def inject_payment_term_manipulation(df: pd.DataFrame, rows: list[int], rng: random.Random) -> pd.DataFrame:
    """T3 — Payment-Term Manipulation.
    Pattern: abrupt shift in payment-term distribution for an established
    (buyer, supplier) edge. Detection signature: payment_term_zscore vs edge history.

    Implementation: we don't have native payment-term fields in IEEE-CIS, so we
    synthesize one (synth_payment_term_days) and inject anomalously short terms
    (e.g., NET-1 against historical NET-30) for the target rows.
    """
    df = df.copy()
    if "synth_payment_term_days" not in df.columns:
        # Default: NET-30 with normal noise
        df["synth_payment_term_days"] = np.random.normal(30.0, 4.0, size=len(df)).clip(min=1)
    for idx in rows:
        df.at[idx, "synth_payment_term_days"] = rng.uniform(0.5, 2.0)
        df.at[idx, "fraud_injected"] = 1
        df.at[idx, "fraud_topology"] = FraudTopology.PAYMENT_TERM_MANIP.value
    return df


def inject_shell_supplier_ring(df: pd.DataFrame, rows: list[int], rng: random.Random, ring_size: int = 3) -> pd.DataFrame:
    """T4 — Shell-Supplier Ring.
    Pattern: a buyer connected to multiple low-history supplier nodes that share
    structural properties (same domain pattern, same address, etc.) suggesting
    a coordinated shell ring.

    Implementation: group target rows by buyer; rewrite supplier_ids to a small
    pool of synthetic 'ring' suppliers per buyer.
    """
    df = df.copy()
    # Group target rows by buyer
    by_buyer = {}
    for idx in rows:
        b = df.at[idx, "buyer_id"]
        by_buyer.setdefault(b, []).append(idx)
    for buyer, indices in by_buyer.items():
        ring_id = rng.randint(10000, 99999)
        ring_suppliers = [f"S::RING{ring_id}|m{j}" for j in range(ring_size)]
        for i, idx in enumerate(indices):
            df.at[idx, "supplier_id"] = ring_suppliers[i % ring_size]
            df.at[idx, "fraud_injected"] = 1
            df.at[idx, "fraud_topology"] = FraudTopology.SHELL_SUPPLIER_RING.value
    return df


def inject_wire_redirection(df: pd.DataFrame, rows: list[int], rng: random.Random) -> pd.DataFrame:
    """T5 — Wire Redirection (BEC pattern).
    Pattern: an established (buyer, supplier) edge sees an abrupt change in
    supplier banking attributes followed by a high-value payment. Detection
    signature: supplier_attribute_change_recent==1 AND amount > p95 of edge history.

    Implementation: we don't have bank-account fields in IEEE-CIS, so we synthesize
    a synth_supplier_bank field; injected rows get a new bank value and a high
    transaction amount.
    """
    df = df.copy()
    if "synth_supplier_bank" not in df.columns:
        df["synth_supplier_bank"] = "BANK_" + df["supplier_id"].apply(lambda s: str(abs(hash(s)) % 1000))
    for idx in rows:
        new_bank = f"BANK_NEW_{rng.randint(10000, 99999)}"
        df.at[idx, "synth_supplier_bank"] = new_bank
        # Inflate amount to top-percentile
        df.at[idx, "TransactionAmt"] = df.at[idx, "TransactionAmt"] * rng.uniform(5.0, 15.0)
        df.at[idx, "fraud_injected"] = 1
        df.at[idx, "fraud_topology"] = FraudTopology.WIRE_REDIRECTION.value
    return df


INJECTORS = {
    FraudTopology.VENDOR_INJECTION: inject_vendor_injection,
    FraudTopology.INVOICE_CYCLING: inject_invoice_cycling,
    FraudTopology.PAYMENT_TERM_MANIP: inject_payment_term_manipulation,
    FraudTopology.SHELL_SUPPLIER_RING: inject_shell_supplier_ring,
    FraudTopology.WIRE_REDIRECTION: inject_wire_redirection,
}


# -----------------------------------------------------------------------------
# Main injection orchestrator
# -----------------------------------------------------------------------------

def inject_all(df: pd.DataFrame, config: InjectionConfig) -> pd.DataFrame:
    """Inject fraud patterns across the dataframe per config."""
    rng = random.Random(config.seed)
    np.random.seed(config.seed)

    df = build_graph_columns(df)
    df["fraud_injected"] = 0
    df["fraud_topology"] = ""

    n_total = len(df)
    n_to_inject = int(n_total * config.overall_rate)
    logger.info(f"Injecting {n_to_inject} fraud events (rate={config.overall_rate:.4f}) across {n_total} transactions")

    # Choose row indices uniformly at random
    candidate_indices = list(df.index)
    rng.shuffle(candidate_indices)
    chosen = candidate_indices[:n_to_inject]

    # Split among topologies per weights
    cursor = 0
    for topo, weight in config.topology_weights.items():
        n_this = int(n_to_inject * weight)
        slice_ = chosen[cursor:cursor + n_this]
        cursor += n_this
        if not slice_:
            continue
        logger.info(f"  {topo.value}: {len(slice_)} rows")
        df = INJECTORS[topo](df, slice_, rng)

    n_injected = int(df["fraud_injected"].sum())
    logger.info(f"Final injection counts: {n_injected} injected fraud, {len(df)} total rows after duplicates")
    return df


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Inject synthetic B2B fraud patterns into IEEE-CIS data")
    parser.add_argument("--input", required=True, type=Path, help="Path to IEEE-CIS train_transaction.csv")
    parser.add_argument("--output", required=True, type=Path, help="Output parquet path")
    parser.add_argument("--inject-rate", type=float, default=0.01, help="Overall injection rate (0.005-0.02 recommended)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on input rows loaded from --input (chronological prefix by TransactionDT if present, else first N rows). Use for smoke tests.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    logger.info(f"Loading {args.input}")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df):,} transactions, {df.shape[1]} columns")

    if args.max_rows is not None and args.max_rows < len(df):
        if "TransactionDT" in df.columns:
            df = df.sort_values("TransactionDT").head(args.max_rows).reset_index(drop=True)
        else:
            df = df.head(args.max_rows).reset_index(drop=True)
        logger.info(f"Capped to {len(df):,} rows (--max-rows={args.max_rows})")

    config = InjectionConfig(overall_rate=args.inject_rate, seed=args.seed)
    df = inject_all(df, config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    logger.info(f"Wrote {args.output} ({len(df):,} rows)")

    # Quick summary
    print("\n=== Injection summary ===")
    print(df["fraud_topology"].value_counts())
    print(f"\nOriginal fraud (isFraud=1): {int(df.get('isFraud', pd.Series([0])).sum())}")
    print(f"Injected fraud (fraud_injected=1): {int(df['fraud_injected'].sum())}")


if __name__ == "__main__":
    main()
