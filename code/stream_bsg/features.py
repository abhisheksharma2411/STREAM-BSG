"""
features.py
===========

STREAM-BSG 47-feature extractor.

Single chronological pass over a labeled IEEE-CIS-shaped dataframe. For each
transaction, computes the 47 buyer / supplier / edge / subgraph features
defined in CLAUDE.md using ONLY transactions with an earlier TransactionDT
(no future leakage), then updates streaming state with the current row.

USAGE:
    python features.py \\
        --input  data/ieee_cis_with_synthetic_b2b.parquet \\
        --output data/ieee_cis_with_features.parquet
    python features.py \\
        --input  data/ieee_cis_with_synthetic_b2b.parquet \\
        --output data/ieee_cis_with_features_smoke.parquet \\
        --smoke-test
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WINDOW_30D_SECONDS = 30 * 86400
RECENT_SHORT_SECONDS = 7 * 86400        # "recent" attribute changes / shell supplier age
SHELL_SUPPLIER_AGE_SECONDS = 7 * 86400  # age threshold for "shell" classification
EDGE_AMOUNTS_KEEP = 200                 # bounded ring buffer for p95 history
INVOICE_DUPLICATE_TOL = 0.02            # within 2% of current amount counts as duplicate
SMOKE_N_DEFAULT = 5000


FEATURE_COLS: list[str] = [
    # --- 15 NODE features --------------------------------------------------
    "feat_buyer_age_days",
    "feat_supplier_age_days",
    "feat_buyer_total_volume_30d",
    "feat_supplier_total_volume_30d",
    "feat_buyer_tx_count_30d",
    "feat_supplier_tx_count_30d",
    "feat_buyer_unique_suppliers",
    "feat_supplier_unique_buyers",
    "feat_buyer_avg_amount",
    "feat_supplier_avg_amount",
    "feat_buyer_amount_std",
    "feat_supplier_amount_std",
    "feat_buyer_attribute_stability",
    "feat_supplier_attribute_stability",
    "feat_buyer_first_seen_recency",
    # --- 18 EDGE features --------------------------------------------------
    "feat_edge_age_days",
    "feat_edge_tx_count",
    "feat_edge_total_volume",
    "feat_edge_avg_amount",
    "feat_edge_amount_std",
    "feat_edge_amount_zscore_current",
    "feat_payment_term_mean",
    "feat_payment_term_std",
    "feat_payment_term_zscore_current",
    "feat_duplicate_invoice_count_in_window",
    "feat_invoice_amount_collision_in_window",
    "feat_supplier_attribute_change_count",
    "feat_supplier_attribute_change_recent",
    "feat_supplier_attr_change_on_this_row",
    "feat_bank_change_recent",
    "feat_bank_change_on_this_row",
    "feat_days_since_last_tx",
    "feat_tx_cadence_score",
    "feat_amount_p95_history",
    "feat_amount_above_p95",
    # --- 14 SUBGRAPH features ---------------------------------------------
    "feat_buyer_1hop_supplier_count",
    "feat_buyer_2hop_supplier_count",
    "feat_buyer_avg_supplier_age",
    "feat_buyer_supplier_age_std",
    "feat_buyer_supplier_age_min",
    "feat_buyer_2hop_supplier_age_min",
    "feat_has_cycle_2hop",
    "feat_max_path_length_2hop",
    "feat_shell_supplier_density",
    "feat_ring_detection_score",
    "feat_subgraph_amount_concentration",
    "feat_subgraph_volume_velocity",
    "feat_subgraph_attribute_homogeneity",
    "feat_subgraph_density",
]
assert len(FEATURE_COLS) == 49, f"Expected 49 features (47 spec + 2 row-level change flags), have {len(FEATURE_COLS)}"


# ---------------------------------------------------------------------------
# State containers — small, __slots__-based, O(1) per-row updates
# ---------------------------------------------------------------------------

class Welford:
    __slots__ = ("count", "mean", "M2", "total")

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.total = 0.0

    def update(self, x: float) -> None:
        self.count += 1
        self.total += x
        delta = x - self.mean
        self.mean += delta / self.count
        self.M2 += delta * (x - self.mean)

    def std(self) -> float:
        if self.count < 2:
            return 0.0
        var = self.M2 / (self.count - 1)
        return math.sqrt(var) if var > 0 else 0.0


class NodeState:
    __slots__ = (
        "first_seen_dt",
        "last_seen_dt",
        "amount",
        "neighbors",
        "win30",
        "win30_total",
        "attr_vals",
        "vol_by_neighbor",
    )

    def __init__(self) -> None:
        self.first_seen_dt: int = -1
        self.last_seen_dt: int = -1
        self.amount = Welford()
        self.neighbors: set[str] = set()
        self.win30: deque = deque()  # (dt, amt)
        self.win30_total: float = 0.0
        # attribute drift trackers: column -> set of values seen
        self.attr_vals: dict[str, set] = defaultdict(set)
        # for buyers: supplier_id -> cumulative volume (Herfindahl)
        self.vol_by_neighbor: dict[str, float] = defaultdict(float)

    def expire_window(self, current_dt: int) -> None:
        cutoff = current_dt - WINDOW_30D_SECONDS
        win = self.win30
        while win and win[0][0] < cutoff:
            _, old_amt = win.popleft()
            self.win30_total -= old_amt


class EdgeState:
    __slots__ = (
        "first_seen_dt",
        "last_seen_dt",
        "amount",
        "term",
        "amounts",
        "win_invoices",
        "last_bank",
        "bank_change_count",
        "last_bank_change_dt",
        "attr_change_count",
        "last_attr_change_dt",
        "interarrival",
        "last_attrs",
    )

    def __init__(self) -> None:
        self.first_seen_dt: int = -1
        self.last_seen_dt: int = -1
        self.amount = Welford()
        self.term = Welford()
        self.amounts: deque = deque(maxlen=EDGE_AMOUNTS_KEEP)
        self.win_invoices: deque = deque()  # (dt, amt) — for duplicate detection
        self.last_bank: Optional[str] = None
        self.bank_change_count: int = 0
        self.last_bank_change_dt: int = -1
        self.attr_change_count: int = 0
        self.last_attr_change_dt: int = -1
        self.interarrival = Welford()
        # Per-edge most-recently-seen values for tracked supplier-side attrs.
        # A change vs the previous edge value bumps attr_change_count.
        self.last_attrs: dict[str, str] = {}

    def expire_invoice_window(self, current_dt: int, window_seconds: int) -> None:
        cutoff = current_dt - window_seconds
        win = self.win_invoices
        while win and win[0][0] < cutoff:
            win.popleft()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _h16(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _derive_ids(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, str]:
    """Use existing buyer_id/supplier_id columns if present (they encode the
    synth-injection signal for T1/T4); otherwise derive sha1[:16] hashes from
    (card1, addr1) and (ProductCD, R_emaildomain) per the user spec.

    Returns (buyer_ids, supplier_ids, source_str).
    """
    if "buyer_id" in df.columns and "supplier_id" in df.columns:
        return (
            df["buyer_id"].astype(str).to_numpy(),
            df["supplier_id"].astype(str).to_numpy(),
            "existing buyer_id/supplier_id columns",
        )
    bid = (df["card1"].astype(str) + "|" + df["addr1"].astype(str)).map(_h16).to_numpy()
    sid = (df["ProductCD"].astype(str) + "|" + df["R_emaildomain"].astype(str)).map(_h16).to_numpy()
    return bid, sid, "sha1[:16] of (card1,addr1) and (ProductCD,R_emaildomain)"


def _percentile(values: deque, q: float) -> float:
    """p-quantile of a (small) deque of floats."""
    n = len(values)
    if n == 0:
        return 0.0
    arr = np.fromiter(values, dtype=np.float32, count=n)
    return float(np.percentile(arr, q))


# ---------------------------------------------------------------------------
# Main compute
# ---------------------------------------------------------------------------

def compute_features(df: pd.DataFrame, window_days: int = 90) -> pd.DataFrame:
    """Single chronological pass; computes 47 streaming features per row.

    For each row, features are derived from state that includes ONLY
    transactions with strictly earlier TransactionDT (rows with equal
    TransactionDT are processed in input order; earlier-tied rows do leak
    into later-tied rows in the same tick — acceptable at sub-second
    resolution).
    """
    if "TransactionDT" not in df.columns:
        raise ValueError("Input must contain TransactionDT")
    if "TransactionAmt" not in df.columns:
        raise ValueError("Input must contain TransactionAmt")

    df = df.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)

    bids, sids, id_source = _derive_ids(df)
    logger.info("Buyer/supplier IDs: %s", id_source)

    n = len(df)
    window_seconds = window_days * 86400

    has_bank = "synth_supplier_bank" in df.columns
    has_term = "synth_payment_term_days" in df.columns

    dts = df["TransactionDT"].to_numpy(dtype=np.int64)
    amts = df["TransactionAmt"].to_numpy(dtype=np.float64)
    if has_term:
        terms = df["synth_payment_term_days"].to_numpy(dtype=np.float64)
    else:
        terms = np.full(n, np.nan, dtype=np.float64)
    if has_bank:
        banks = df["synth_supplier_bank"].astype(str).to_numpy()
    else:
        banks = np.empty(n, dtype=object)
        banks[:] = ""

    # Attribute-drift trackers — non-synthetic fields we expect to be largely
    # stable per buyer / per supplier. Note: supplier_id is derived from
    # (ProductCD, R_emaildomain), so those two fields are constant per
    # supplier by construction — pick OTHER fields for supplier stability.
    buyer_attr_cols = [c for c in ("addr1", "ProductCD") if c in df.columns]
    # supplier_attr_cols feeds two features:
    #   - feat_supplier_attribute_stability: needs drifty fields (addr2, P_emaildomain, card1)
    #   - feat_subgraph_attribute_homogeneity: needs the supplier's R_emaildomain so we can
    #     compare distributions across the buyer's neighbors. R_emaildomain is constant per
    #     supplier_id by construction but that's fine for the cross-supplier comparison.
    supplier_attr_cols = [c for c in ("addr2", "P_emaildomain", "card1", "R_emaildomain") if c in df.columns]
    edge_supplier_attr_cols = [c for c in ("addr2", "P_emaildomain") if c in df.columns]
    buyer_attr_data = {c: df[c].astype(str).to_numpy() for c in buyer_attr_cols}
    supplier_attr_data = {c: df[c].astype(str).to_numpy() for c in supplier_attr_cols}
    edge_attr_data = {c: df[c].astype(str).to_numpy() for c in edge_supplier_attr_cols}

    buyers: dict[str, NodeState] = {}
    suppliers: dict[str, NodeState] = {}
    edges: dict[tuple, EdgeState] = {}
    supplier_first_seen: dict[str, int] = {}

    out: dict[str, np.ndarray] = {c: np.zeros(n, dtype=np.float32) for c in FEATURE_COLS}

    log_every = max(1, n // 10)

    for i in range(n):
        if i and i % log_every == 0:
            logger.info("  ... row %d / %d (%.0f%%)", i, n, 100 * i / n)

        dt = int(dts[i])
        amt = float(amts[i]) if not math.isnan(amts[i]) else 0.0
        bid = bids[i]
        sid = sids[i]
        term = float(terms[i]) if has_term and not math.isnan(terms[i]) else float("nan")
        bank = banks[i] if has_bank else ""

        buyer = buyers.get(bid)
        supplier = suppliers.get(sid)
        edge = edges.get((bid, sid))

        # ===== NODE features (read from prior state) =======================
        if buyer is not None:
            buyer.expire_window(dt)
            out["feat_buyer_age_days"][i] = (dt - buyer.first_seen_dt) / 86400.0
            out["feat_buyer_total_volume_30d"][i] = buyer.win30_total
            out["feat_buyer_tx_count_30d"][i] = len(buyer.win30)
            out["feat_buyer_unique_suppliers"][i] = len(buyer.neighbors)
            out["feat_buyer_avg_amount"][i] = buyer.amount.mean
            out["feat_buyer_amount_std"][i] = buyer.amount.std()
            if buyer.attr_vals:
                stab = sum(1.0 / max(1, len(v)) for v in buyer.attr_vals.values()) / len(buyer.attr_vals)
                out["feat_buyer_attribute_stability"][i] = stab
            else:
                out["feat_buyer_attribute_stability"][i] = 1.0
            age_s = dt - buyer.first_seen_dt
            out["feat_buyer_first_seen_recency"][i] = math.exp(-age_s / RECENT_SHORT_SECONDS)
        else:
            out["feat_buyer_attribute_stability"][i] = 1.0
            out["feat_buyer_first_seen_recency"][i] = 1.0  # brand new buyer

        if supplier is not None:
            supplier.expire_window(dt)
            out["feat_supplier_age_days"][i] = (dt - supplier.first_seen_dt) / 86400.0
            out["feat_supplier_total_volume_30d"][i] = supplier.win30_total
            out["feat_supplier_tx_count_30d"][i] = len(supplier.win30)
            out["feat_supplier_unique_buyers"][i] = len(supplier.neighbors)
            out["feat_supplier_avg_amount"][i] = supplier.amount.mean
            out["feat_supplier_amount_std"][i] = supplier.amount.std()
            if supplier.attr_vals:
                stab = sum(1.0 / max(1, len(v)) for v in supplier.attr_vals.values()) / len(supplier.attr_vals)
                out["feat_supplier_attribute_stability"][i] = stab
            else:
                out["feat_supplier_attribute_stability"][i] = 1.0
        else:
            out["feat_supplier_attribute_stability"][i] = 1.0

        # ===== EDGE features (read from prior state) =======================
        if edge is not None:
            edge.expire_invoice_window(dt, window_seconds)
            out["feat_edge_age_days"][i] = (dt - edge.first_seen_dt) / 86400.0
            out["feat_edge_tx_count"][i] = edge.amount.count
            out["feat_edge_total_volume"][i] = edge.amount.total
            out["feat_edge_avg_amount"][i] = edge.amount.mean
            edge_std = edge.amount.std()
            out["feat_edge_amount_std"][i] = edge_std
            if edge_std > 0:
                out["feat_edge_amount_zscore_current"][i] = (amt - edge.amount.mean) / edge_std

            out["feat_payment_term_mean"][i] = edge.term.mean
            term_std = edge.term.std()
            out["feat_payment_term_std"][i] = term_std
            if has_term and not math.isnan(term) and term_std > 0:
                out["feat_payment_term_zscore_current"][i] = (term - edge.term.mean) / term_std

            # T2: duplicate invoice detection in window_days window
            dup_count = 0
            tol = max(1e-6, INVOICE_DUPLICATE_TOL * abs(amt))
            for _, prev_amt in edge.win_invoices:
                if abs(prev_amt - amt) <= tol:
                    dup_count += 1
            out["feat_duplicate_invoice_count_in_window"][i] = dup_count
            out["feat_invoice_amount_collision_in_window"][i] = 1.0 if dup_count > 0 else 0.0

            out["feat_supplier_attribute_change_count"][i] = edge.attr_change_count
            out["feat_supplier_attribute_change_recent"][i] = (
                1.0
                if edge.last_attr_change_dt >= 0
                and (dt - edge.last_attr_change_dt) < RECENT_SHORT_SECONDS
                else 0.0
            )
            out["feat_bank_change_recent"][i] = (
                1.0
                if edge.last_bank_change_dt >= 0
                and (dt - edge.last_bank_change_dt) < RECENT_SHORT_SECONDS
                else 0.0
            )
            # Row-level change flags: detect change at THIS row by comparing
            # the current row's bank / supplier-side attrs against the edge's
            # last-seen values BEFORE the state update overwrites them.
            # Necessary because T5 / supplier-attribute drift are one-shot
            # events; the "recent" flags above only fire on the SUBSEQUENT row
            # which often never arrives.
            if has_bank and bank and edge.last_bank is not None and edge.last_bank != bank:
                out["feat_bank_change_on_this_row"][i] = 1.0
            for col in edge_supplier_attr_cols:
                prev = edge.last_attrs.get(col)
                if prev is not None and prev != edge_attr_data[col][i]:
                    out["feat_supplier_attr_change_on_this_row"][i] = 1.0
                    break

            out["feat_days_since_last_tx"][i] = (dt - edge.last_seen_dt) / 86400.0
            if edge.interarrival.count >= 1:
                ia_std = edge.interarrival.std()
                # cadence_score: 1 / (1 + CV of inter-arrival), 0..1; high = regular
                if edge.interarrival.mean > 0:
                    cv = ia_std / edge.interarrival.mean
                    out["feat_tx_cadence_score"][i] = 1.0 / (1.0 + cv)
            p95 = _percentile(edge.amounts, 95)
            out["feat_amount_p95_history"][i] = p95
            out["feat_amount_above_p95"][i] = 1.0 if (p95 > 0 and amt > p95) else 0.0

        # ===== SUBGRAPH features (read from prior state) ===================
        if buyer is not None and buyer.neighbors:
            buyer_suppliers = buyer.neighbors
            # 1-hop: active in last 30 days
            active_1hop = sum(
                1
                for s in buyer_suppliers
                if s in suppliers and (dt - suppliers[s].last_seen_dt) < WINDOW_30D_SECONDS
            )
            out["feat_buyer_1hop_supplier_count"][i] = active_1hop

            # supplier ages over buyer's neighbors
            ages: list[float] = []
            min_age = float("inf")
            for s in buyer_suppliers:
                sst = suppliers.get(s)
                if sst is not None:
                    a = (dt - sst.first_seen_dt) / 86400.0
                    ages.append(a)
                    if a < min_age:
                        min_age = a
            if ages:
                arr = np.asarray(ages, dtype=np.float32)
                out["feat_buyer_avg_supplier_age"][i] = float(arr.mean())
                out["feat_buyer_supplier_age_std"][i] = float(arr.std(ddof=0))
                out["feat_buyer_supplier_age_min"][i] = float(min_age)
                shell_age_days = SHELL_SUPPLIER_AGE_SECONDS / 86400.0
                shell_density = float((arr < shell_age_days).sum()) / len(arr)
                out["feat_shell_supplier_density"][i] = shell_density

            # 2-hop: other buyers who share any supplier with this buyer,
            #        then THEIR suppliers
            two_hop_buyers: set[str] = set()
            for s in buyer_suppliers:
                sst = suppliers.get(s)
                if sst is None:
                    continue
                for b2 in sst.neighbors:
                    if b2 != bid:
                        two_hop_buyers.add(b2)
            two_hop_suppliers: set[str] = set()
            min_age_2hop = float("inf")
            for b2 in two_hop_buyers:
                bst = buyers.get(b2)
                if bst is None:
                    continue
                for s2 in bst.neighbors:
                    if s2 not in buyer_suppliers:
                        two_hop_suppliers.add(s2)
                        sst2 = suppliers.get(s2)
                        if sst2 is not None:
                            a2 = (dt - sst2.first_seen_dt) / 86400.0
                            if a2 < min_age_2hop:
                                min_age_2hop = a2
            out["feat_buyer_2hop_supplier_count"][i] = len(two_hop_suppliers)
            if min_age_2hop < float("inf"):
                out["feat_buyer_2hop_supplier_age_min"][i] = min_age_2hop
            # cycle: any supplier of ours has another buyer too -> 4-cycle exists
            has_cycle = any(
                len(suppliers[s].neighbors) > 1
                for s in buyer_suppliers
                if s in suppliers
            )
            out["feat_has_cycle_2hop"][i] = 1.0 if has_cycle else 0.0
            out["feat_max_path_length_2hop"][i] = 2.0 if two_hop_buyers else (1.0 if buyer_suppliers else 0.0)

            # ring_detection_score: shell density * log(1 + 2hop buyers)
            out["feat_ring_detection_score"][i] = out["feat_shell_supplier_density"][i] * math.log1p(
                len(two_hop_buyers)
            )

            # subgraph_amount_concentration: Herfindahl over supplier volumes
            if buyer.vol_by_neighbor:
                total = sum(buyer.vol_by_neighbor.values())
                if total > 0:
                    hhi = sum((v / total) ** 2 for v in buyer.vol_by_neighbor.values())
                    out["feat_subgraph_amount_concentration"][i] = hhi

            # subgraph_volume_velocity: 30d volume / max(1, buyer_age_days)
            age_days = (dt - buyer.first_seen_dt) / 86400.0
            denom = age_days if age_days >= 1.0 else 1.0
            out["feat_subgraph_volume_velocity"][i] = buyer.win30_total / denom

            # attribute homogeneity: how concentrated are buyer-supplier-shared
            # attribute values across the buyer's neighbors. Use plurality
            # share of the supplier's R_emaildomain.
            domain_counts: dict[str, int] = defaultdict(int)
            for s in buyer_suppliers:
                sst = suppliers.get(s)
                if sst is None:
                    continue
                vals = sst.attr_vals.get("R_emaildomain")
                if vals:
                    for v in vals:
                        domain_counts[v] += 1
            if domain_counts:
                top = max(domain_counts.values())
                total_dom = sum(domain_counts.values())
                out["feat_subgraph_attribute_homogeneity"][i] = top / total_dom

            # subgraph_density: edges_in_subgraph / max possible. The induced
            # subgraph is (buyer's suppliers ∪ 2-hop buyers ∪ buyer). We
            # approximate edges as sum of degrees of buyer's suppliers, max
            # possible as |suppliers| * (|2-hop buyers| + 1).
            edges_in = sum(
                len(suppliers[s].neighbors)
                for s in buyer_suppliers
                if s in suppliers
            )
            max_edges = len(buyer_suppliers) * (len(two_hop_buyers) + 1)
            if max_edges > 0:
                out["feat_subgraph_density"][i] = min(1.0, edges_in / max_edges)

        # ===== UPDATE STATE (after computing features) =====================
        if buyer is None:
            buyer = NodeState()
            buyer.first_seen_dt = dt
            buyers[bid] = buyer
        buyer.last_seen_dt = dt
        buyer.amount.update(amt)
        buyer.neighbors.add(sid)
        buyer.win30.append((dt, amt))
        buyer.win30_total += amt
        for col in buyer_attr_cols:
            buyer.attr_vals[col].add(buyer_attr_data[col][i])
        buyer.vol_by_neighbor[sid] += amt

        if supplier is None:
            supplier = NodeState()
            supplier.first_seen_dt = dt
            suppliers[sid] = supplier
            supplier_first_seen[sid] = dt
        supplier.last_seen_dt = dt
        supplier.amount.update(amt)
        supplier.neighbors.add(bid)
        supplier.win30.append((dt, amt))
        supplier.win30_total += amt
        for col in supplier_attr_cols:
            supplier.attr_vals[col].add(supplier_attr_data[col][i])

        if edge is None:
            edge = EdgeState()
            edge.first_seen_dt = dt
            edges[(bid, sid)] = edge
        else:
            if edge.last_seen_dt >= 0:
                edge.interarrival.update((dt - edge.last_seen_dt) / 86400.0)
        # T5: synthetic supplier bank change tracking
        if has_bank and bank:
            if edge.last_bank is not None and edge.last_bank != bank:
                edge.bank_change_count += 1
                edge.last_bank_change_dt = dt
            edge.last_bank = bank
        # Generic supplier-attribute change tracking on this edge: any
        # tracked supplier-side attribute whose value differs from the
        # previous value seen on THIS edge bumps the change counter.
        for col in edge_supplier_attr_cols:
            v = edge_attr_data[col][i]
            prev = edge.last_attrs.get(col)
            if prev is not None and prev != v:
                edge.attr_change_count += 1
                edge.last_attr_change_dt = dt
            edge.last_attrs[col] = v
        edge.last_seen_dt = dt
        edge.amount.update(amt)
        edge.amounts.append(amt)
        edge.win_invoices.append((dt, amt))
        if has_term and not math.isnan(term):
            edge.term.update(term)

    # Attach features
    for col, arr in out.items():
        df[col] = arr

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_feature_summary(df: pd.DataFrame) -> None:
    rows = []
    for col in FEATURE_COLS:
        s = df[col]
        n_null = int(s.isna().sum()) + int((~np.isfinite(s.to_numpy(dtype=np.float64))).sum() - s.isna().sum())
        # n_null counts NaN+Inf together
        pct_null = 100.0 * n_null / len(s)
        finite = s.to_numpy(dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            rows.append((col, float("nan"), float("nan"), float("nan"), float("nan"), pct_null))
            continue
        rows.append(
            (
                col,
                float(finite.mean()),
                float(finite.std()),
                float(finite.min()),
                float(finite.max()),
                pct_null,
            )
        )
    summary = pd.DataFrame(rows, columns=["feature", "mean", "std", "min", "max", "pct_null_or_inf"])
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\n=== Feature summary ({} rows) ===".format(len(df)))
    print(summary.to_string(index=False))

    # also stratify by fraud_injected if present
    if "fraud_injected" in df.columns and df["fraud_injected"].sum() > 0:
        print("\n=== Mean by fraud_injected (sanity check for signal) ===")
        means = df.groupby("fraud_injected")[FEATURE_COLS].mean().T
        means["delta"] = means.get(1, 0) - means.get(0, 0)
        print(means.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="STREAM-BSG 47-feature extractor")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--window-days", type=int, default=90)
    parser.add_argument("--smoke-test", action="store_true",
                        help=f"Process only first {SMOKE_N_DEFAULT} rows (chronological) and print feature summary.")
    parser.add_argument("--smoke-n", type=int, default=SMOKE_N_DEFAULT)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("Loading %s", args.input)
    df = pd.read_parquet(args.input)
    logger.info("Loaded %d rows, %d columns", len(df), df.shape[1])

    if args.smoke_test:
        df_sorted = df.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)
        df = df_sorted.head(args.smoke_n).copy()
        logger.info("Smoke test: processing first %d rows (chronological)", len(df))

    out = compute_features(df, window_days=args.window_days)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    logger.info("Wrote %s (%d rows, %d cols)", args.output, len(out), out.shape[1])

    if args.smoke_test:
        _print_feature_summary(out)


if __name__ == "__main__":
    main()
