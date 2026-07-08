"""
graphsage_baseline.py
=====================

GraphSAGE baseline for STREAM-BSG comparison.

Design (validated with user before implementation):
  * Bipartite buyer <-> supplier graph, represented as a homogeneous
    undirected graph with node-type one-hot as a node feature.
  * Transductive: node set is fixed. Graph is pruned per split:
      - train:  edges from train transactions only
      - val:    edges from train + val transactions
      - test:   edges from all transactions
    This mirrors STREAM-BSG's streaming semantics — no future edges leak
    into past message passing.
  * Per-transaction classification: for each transaction (buyer b, supplier s,
    tabular features x), the head predicts fraud from
    ``[GraphSAGE(g)[b] ; GraphSAGE(g)[s] ; x]``.

USAGE:
    python graphsage_baseline.py \\
        --input  data/ieee_cis_with_synthetic_b2b.parquet \\
        --output results/graphsage_results.json \\
        --seed 42
    python graphsage_baseline.py --input ... --output ... --smoke-test
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch_geometric.nn import SAGEConv

logger = logging.getLogger(__name__)

LABEL_COL = "fraud_injected"
TOPOLOGY_COL = "fraud_topology"
TIME_COL = "TransactionDT"
BUYER_COL = "buyer_id"
SUPPLIER_COL = "supplier_id"

TOPOLOGIES = [
    "T1_vendor_injection",
    "T2_invoice_cycling",
    "T3_payment_term_manipulation",
    "T4_shell_supplier_ring",
    "T5_wire_redirection",
]

NODE_FEATURE_NAMES = [
    "is_buyer",           # 1 if buyer, 0 if supplier
    "is_supplier",        # 1 if supplier, 0 if buyer
    "log_tx_count",       # log(1 + tx count endpointed at this node in the visible window)
    "log_total_volume",   # log(1 + total volume)
    "mean_amount",        # mean transaction amount at this node
    "std_amount",         # std of transaction amount
    "n_unique_counterparties",  # unique buyers-of-supplier / suppliers-of-buyer
    "first_seen_recency", # 1 - (dt - first_seen_dt) / max_dt_range, clipped [0,1]
]
N_NODE_FEATS = len(NODE_FEATURE_NAMES)

EDGE_FEATURE_NAMES = [
    "log_amount",         # log(1 + TransactionAmt)
    "dt_normalized",      # (TransactionDT - dt_min) / (dt_max - dt_min)
]
N_EDGE_FEATS = len(EDGE_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Node-id book-keeping
# ---------------------------------------------------------------------------

def build_node_index(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Assign a global node id to each unique buyer and supplier.

    Buyers get ids [0, n_buyers); suppliers get ids [n_buyers, n_buyers+n_suppliers).
    Returns (buyer_node_id_per_row, supplier_node_id_per_row, n_buyers, n_suppliers).
    """
    buyer_codes, buyer_uniques = pd.factorize(df[BUYER_COL].astype(str), sort=True)
    supplier_codes, supplier_uniques = pd.factorize(df[SUPPLIER_COL].astype(str), sort=True)
    n_buyers = len(buyer_uniques)
    n_suppliers = len(supplier_uniques)
    b_ids = buyer_codes.astype(np.int64)
    s_ids = (supplier_codes + n_buyers).astype(np.int64)
    return b_ids, s_ids, n_buyers, n_suppliers


# ---------------------------------------------------------------------------
# Per-split graph construction
# ---------------------------------------------------------------------------

@dataclass
class SplitGraph:
    """Everything the training loop needs for one split's graph."""
    node_x: torch.Tensor          # [N, N_NODE_FEATS], node features
    edge_index: torch.Tensor      # [2, E], undirected edges (both directions)
    edge_weight: torch.Tensor     # [E], transaction count per unique (buyer, supplier) pair
    n_nodes: int
    n_buyers: int
    n_suppliers: int
    n_unique_edges: int           # unique (buyer, supplier) pairs (before edge doubling)
    dt_min: int
    dt_max: int


def build_split_graph(
    visible: pd.DataFrame,
    buyer_ids: np.ndarray,
    supplier_ids: np.ndarray,
    n_nodes: int,
    n_buyers: int,
    n_suppliers: int,
) -> SplitGraph:
    """Build the pruned per-split graph from the transactions VISIBLE to this
    split (train-only for train phase, train+val for val phase, all for test).

    Node features are aggregated only over `visible`.
    """
    v_idx = visible.index.to_numpy()
    v_bid = buyer_ids[v_idx]     # buyer node id per visible row
    v_sid = supplier_ids[v_idx]  # supplier node id per visible row
    v_amt = visible["TransactionAmt"].to_numpy(dtype=np.float64)
    v_amt = np.nan_to_num(v_amt, nan=0.0)
    v_dt  = visible[TIME_COL].to_numpy(dtype=np.int64)

    dt_min = int(v_dt.min()) if len(v_dt) else 0
    dt_max = int(v_dt.max()) if len(v_dt) else 1
    dt_range = max(1, dt_max - dt_min)

    # Node aggregations
    node_first_dt   = np.full(n_nodes, np.iinfo(np.int64).max, dtype=np.int64)
    node_tx_count   = np.zeros(n_nodes, dtype=np.float64)
    node_total_vol  = np.zeros(n_nodes, dtype=np.float64)
    node_sum_amt    = np.zeros(n_nodes, dtype=np.float64)
    node_sum_amt_sq = np.zeros(n_nodes, dtype=np.float64)
    # unique counterparties: track set sizes cheaply via per-node counter dict
    # (simpler and fast enough with pandas groupby)
    for nid_arr in (v_bid, v_sid):
        # tx count / volume / amount stats
        np.add.at(node_tx_count,   nid_arr, 1.0)
        np.add.at(node_total_vol,  nid_arr, v_amt)
        np.add.at(node_sum_amt,    nid_arr, v_amt)
        np.add.at(node_sum_amt_sq, nid_arr, v_amt * v_amt)
        # first seen dt (min)
        np.minimum.at(node_first_dt, nid_arr, v_dt)

    # unique counterparties per node via pandas groupby (both directions)
    v_pairs = pd.DataFrame({"b": v_bid, "s": v_sid})
    b_uc = v_pairs.groupby("b")["s"].nunique()
    s_uc = v_pairs.groupby("s")["b"].nunique()
    n_unique_ct = np.zeros(n_nodes, dtype=np.float64)
    n_unique_ct[b_uc.index.to_numpy()] = b_uc.to_numpy()
    n_unique_ct[s_uc.index.to_numpy()] = s_uc.to_numpy()

    # Assemble node feature matrix
    x = np.zeros((n_nodes, N_NODE_FEATS), dtype=np.float32)
    x[:n_buyers, 0] = 1.0  # is_buyer
    x[n_buyers:, 1] = 1.0  # is_supplier

    valid = node_tx_count > 0
    x[:, 2] = np.log1p(node_tx_count)
    x[:, 3] = np.log1p(node_total_vol)
    mean_amt = np.zeros(n_nodes, dtype=np.float64)
    std_amt  = np.zeros(n_nodes, dtype=np.float64)
    mean_amt[valid] = node_sum_amt[valid] / node_tx_count[valid]
    var_amt = np.zeros(n_nodes, dtype=np.float64)
    var_amt[valid] = np.maximum(0.0, node_sum_amt_sq[valid] / node_tx_count[valid] - mean_amt[valid] ** 2)
    std_amt[valid] = np.sqrt(var_amt[valid])
    x[:, 4] = mean_amt.astype(np.float32)
    x[:, 5] = std_amt.astype(np.float32)
    x[:, 6] = n_unique_ct.astype(np.float32)
    first_seen_recency = np.zeros(n_nodes, dtype=np.float32)
    if valid.any():
        # 1 - (first_seen - dt_min) / range  → 1.0 for oldest node, 0.0 for newest
        first_seen_recency[valid] = 1.0 - (node_first_dt[valid] - dt_min) / dt_range
    x[:, 7] = first_seen_recency

    # Edge list: aggregate to unique (buyer, supplier) pairs to keep the graph
    # small; edge weight = tx count on that pair in the visible window.
    pair_counts = v_pairs.groupby(["b", "s"]).size().reset_index(name="w")
    src = pair_counts["b"].to_numpy(dtype=np.int64)
    dst = pair_counts["s"].to_numpy(dtype=np.int64)
    w   = pair_counts["w"].to_numpy(dtype=np.float32)
    # Undirected: add both directions
    edge_index = np.stack([np.concatenate([src, dst]),
                           np.concatenate([dst, src])], axis=0)
    edge_weight = np.concatenate([w, w])

    return SplitGraph(
        node_x=torch.from_numpy(x),
        edge_index=torch.from_numpy(edge_index).long(),
        edge_weight=torch.from_numpy(edge_weight).float(),
        n_nodes=n_nodes,
        n_buyers=n_buyers,
        n_suppliers=n_suppliers,
        n_unique_edges=len(src),
        dt_min=dt_min,
        dt_max=dt_max,
    )


def build_edge_features_for_rows(rows: pd.DataFrame, dt_min: int, dt_max: int) -> torch.Tensor:
    """Per-transaction tabular features consumed by the classifier head."""
    dt_range = max(1, dt_max - dt_min)
    amt = np.nan_to_num(rows["TransactionAmt"].to_numpy(dtype=np.float64), nan=0.0)
    dt  = rows[TIME_COL].to_numpy(dtype=np.int64)
    log_amt = np.log1p(amt).astype(np.float32)
    dt_norm = ((dt - dt_min) / dt_range).astype(np.float32)
    return torch.from_numpy(np.stack([log_amt, dt_norm], axis=1))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SAGEEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim, aggr="mean")
        self.conv2 = SAGEConv(hidden_dim, out_dim, aggr="mean")
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return h


class EdgeClassifier(nn.Module):
    def __init__(self, node_emb_dim: int, edge_feat_dim: int, hidden: int = 64, dropout: float = 0.5):
        super().__init__()
        in_dim = 2 * node_emb_dim + edge_feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, h_b: torch.Tensor, h_s: torch.Tensor, edge_x: torch.Tensor) -> torch.Tensor:
        z = torch.cat([h_b, h_s, edge_x], dim=1)
        return self.mlp(z).squeeze(-1)


class GraphSAGEFraudModel(nn.Module):
    def __init__(self, node_in: int, edge_in: int, hidden: int = 64, emb: int = 64, dropout: float = 0.5):
        super().__init__()
        self.encoder = SAGEEncoder(node_in, hidden, emb, dropout=dropout)
        self.head = EdgeClassifier(emb, edge_in, hidden=hidden, dropout=dropout)

    def forward(self, node_x, edge_index, b_idx, s_idx, edge_x):
        h = self.encoder(node_x, edge_index)
        return self.head(h[b_idx], h[s_idx], edge_x)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def aggregate_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    p, r, _ = precision_recall_curve(y_true, y_score)
    f1c = np.where(p + r > 0, 2 * p * r / np.where(p + r > 0, p + r, 1), 0.0)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    m = fpr <= 0.05
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc":  float(average_precision_score(y_true, y_score)),
        "f1_best": float(np.max(f1c)),
        "recall_at_5pct_fpr": float(tpr[m].max()) if m.any() else 0.0,
    }


def best_f1_threshold_from_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    p, r, thr = precision_recall_curve(y_true, y_score)
    denom = p + r
    f1 = np.where(denom > 0, 2 * p * r / np.where(denom > 0, denom, 1), 0.0)
    i = int(np.argmax(f1[:-1])) if len(f1) > 1 else 0
    return (float(thr[i]) if i < len(thr) else 0.5), float(f1[i])


def per_topology_recall(test_df: pd.DataFrame, scores: np.ndarray, threshold: float) -> dict:
    out = {}
    for topo in TOPOLOGIES:
        m = (test_df[TOPOLOGY_COL].astype(str) == topo).to_numpy()
        n_t = int(m.sum())
        det = int((scores[m] >= threshold).sum())
        key = topo.replace("T1_vendor_injection", "T1_vendor_injection") \
                   .replace("T2_invoice_cycling", "T2_invoice_cycling_row") \
                   .replace("T3_payment_term_manipulation", "T3_payment_term_manip") \
                   .replace("T4_shell_supplier_ring", "T4_shell_supplier_ring") \
                   .replace("T5_wire_redirection", "T5_wire_redirection")
        out[key] = {"n": n_t, "detected": det, "recall": (det / n_t) if n_t else None}
    # T2 per-event
    m2 = (test_df[TOPOLOGY_COL] == "T2_invoice_cycling").to_numpy()
    t2 = test_df.loc[m2, ["TransactionID", BUYER_COL, SUPPLIER_COL]].copy()
    t2["score"] = scores[m2]
    t2["base_tid"] = t2["TransactionID"].astype(str).str.split("_DUP_").str[0]
    grp = t2.groupby([BUYER_COL, SUPPLIER_COL, "base_tid"])
    n_ev = int(grp.ngroups)
    det_ev = int((grp.score.max() >= threshold).sum())
    out["T2_invoice_cycling_event"] = {
        "n": n_ev, "detected": det_ev,
        "recall": (det_ev / n_ev) if n_ev else None,
    }
    return out


# ---------------------------------------------------------------------------
# Split-graph packaging
# ---------------------------------------------------------------------------

@dataclass
class DataBundle:
    train_g: SplitGraph
    val_g: SplitGraph
    test_g: SplitGraph
    train_rows: pd.DataFrame
    val_rows: pd.DataFrame
    test_rows: pd.DataFrame
    train_edge_x: torch.Tensor
    val_edge_x: torch.Tensor
    test_edge_x: torch.Tensor
    train_bid: torch.Tensor
    train_sid: torch.Tensor
    val_bid: torch.Tensor
    val_sid: torch.Tensor
    test_bid: torch.Tensor
    test_sid: torch.Tensor
    y_train: torch.Tensor
    y_val: torch.Tensor
    y_test: torch.Tensor
    n_nodes: int
    dt_min: int
    dt_max: int


def prepare_data(input_path: str, train_frac: float = 0.70, val_frac: float = 0.15) -> DataBundle:
    df = pd.read_parquet(input_path)
    df = df.sort_values(TIME_COL, kind="mergesort").reset_index(drop=True)

    buyer_ids, supplier_ids, n_buyers, n_suppliers = build_node_index(df)
    n_nodes = n_buyers + n_suppliers
    logger.info("Nodes: %d buyers + %d suppliers = %d total", n_buyers, n_suppliers, n_nodes)

    n = len(df)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    train = df.iloc[:n_train].reset_index(drop=True)
    val   = df.iloc[n_train:n_train+n_val].reset_index(drop=True)
    test  = df.iloc[n_train+n_val:].reset_index(drop=True)
    # Preserve original global position (before reset_index)
    train.index = pd.RangeIndex(0, len(train))
    val.index   = pd.RangeIndex(len(train), len(train)+len(val))
    test.index  = pd.RangeIndex(len(train)+len(val), n)
    logger.info("Chronological split: train=%d val=%d test=%d", len(train), len(val), len(test))

    train_visible = df.iloc[:n_train]
    val_visible   = df.iloc[:n_train+n_val]
    test_visible  = df

    logger.info("Building train graph...")
    train_g = build_split_graph(train_visible, buyer_ids, supplier_ids, n_nodes, n_buyers, n_suppliers)
    logger.info("  train_g:  unique edges=%d, dt_range=[%d, %d]", train_g.n_unique_edges, train_g.dt_min, train_g.dt_max)
    logger.info("Building val graph...")
    val_g = build_split_graph(val_visible, buyer_ids, supplier_ids, n_nodes, n_buyers, n_suppliers)
    logger.info("  val_g:    unique edges=%d, dt_range=[%d, %d]", val_g.n_unique_edges, val_g.dt_min, val_g.dt_max)
    logger.info("Building test graph...")
    test_g = build_split_graph(test_visible, buyer_ids, supplier_ids, n_nodes, n_buyers, n_suppliers)
    logger.info("  test_g:   unique edges=%d, dt_range=[%d, %d]", test_g.n_unique_edges, test_g.dt_min, test_g.dt_max)

    dt_min = train_g.dt_min
    dt_max = test_g.dt_max
    train_edge_x = build_edge_features_for_rows(train, dt_min, dt_max)
    val_edge_x   = build_edge_features_for_rows(val,   dt_min, dt_max)
    test_edge_x  = build_edge_features_for_rows(test,  dt_min, dt_max)

    return DataBundle(
        train_g=train_g, val_g=val_g, test_g=test_g,
        train_rows=train, val_rows=val, test_rows=test,
        train_edge_x=train_edge_x, val_edge_x=val_edge_x, test_edge_x=test_edge_x,
        train_bid=torch.from_numpy(buyer_ids[:n_train]).long(),
        train_sid=torch.from_numpy(supplier_ids[:n_train]).long(),
        val_bid=torch.from_numpy(buyer_ids[n_train:n_train+n_val]).long(),
        val_sid=torch.from_numpy(supplier_ids[n_train:n_train+n_val]).long(),
        test_bid=torch.from_numpy(buyer_ids[n_train+n_val:]).long(),
        test_sid=torch.from_numpy(supplier_ids[n_train+n_val:]).long(),
        y_train=torch.from_numpy(train[LABEL_COL].to_numpy()).float(),
        y_val=torch.from_numpy(val[LABEL_COL].to_numpy()).float(),
        y_test=torch.from_numpy(test[LABEL_COL].to_numpy()).float(),
        n_nodes=n_nodes, dt_min=dt_min, dt_max=dt_max,
    )


# ---------------------------------------------------------------------------
# One training step — used by the smoke-test entry point
# ---------------------------------------------------------------------------

def one_step_demo(bundle: DataBundle, seed: int, device: torch.device) -> dict:
    """Build model, do one forward + loss + backward + val AUC. Prints and returns diagnostics."""
    torch.manual_seed(seed)

    model = GraphSAGEFraudModel(node_in=N_NODE_FEATS, edge_in=N_EDGE_FEATS,
                                 hidden=64, emb=64, dropout=0.5).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model params: %d (encoder + edge-MLP)", n_params)

    train_g = bundle.train_g
    node_x = train_g.node_x.to(device)
    edge_index = train_g.edge_index.to(device)
    train_bid = bundle.train_bid.to(device); train_sid = bundle.train_sid.to(device)
    y_train = bundle.y_train.to(device); edge_x = bundle.train_edge_x.to(device)

    pos = float(y_train.sum().item()); neg = float(len(y_train) - pos)
    pos_weight = torch.tensor(neg / max(1.0, pos), device=device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    t0 = time.perf_counter()
    logits = model(node_x, edge_index, train_bid, train_sid, edge_x)
    loss = criterion(logits, y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()
    fw_bw_s = time.perf_counter() - t0
    logger.info("Step 1: train loss = %.6f  (fw+bw+step = %.2fs)", loss.item(), fw_bw_s)

    # One-shot val eval on the val graph (still after 1 step, expect noise)
    model.eval()
    with torch.no_grad():
        val_node_x = bundle.val_g.node_x.to(device)
        val_edge_index = bundle.val_g.edge_index.to(device)
        val_bid = bundle.val_bid.to(device); val_sid = bundle.val_sid.to(device)
        val_edge_x = bundle.val_edge_x.to(device)
        val_logits = model(val_node_x, val_edge_index, val_bid, val_sid, val_edge_x)
        val_scores = torch.sigmoid(val_logits).cpu().numpy()
    val_m = aggregate_metrics(bundle.y_val.numpy(), val_scores)
    logger.info("Step 1: val ROC-AUC=%.4f  PR-AUC=%.4f  F1_best=%.4f  R@5%%FPR=%.4f (untrained-ish baseline)",
                val_m["roc_auc"], val_m["pr_auc"], val_m["f1_best"], val_m["recall_at_5pct_fpr"])

    return {
        "train_loss_step1": float(loss.item()),
        "fw_bw_step_seconds": round(fw_bw_s, 3),
        "val_metrics_step1": val_m,
        "n_model_params": int(n_params),
    }


# ---------------------------------------------------------------------------
# Full training + eval
# ---------------------------------------------------------------------------

def train_and_evaluate(
    input_path: str, output_path: str,
    seed: int = 42, train_frac: float = 0.70, val_frac: float = 0.15,
    epochs: int = 60, hidden: int = 64, emb: int = 64, lr: float = 1e-3,
    weight_decay: float = 1e-5, dropout: float = 0.5, patience: int = 8,
) -> dict:
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    logger.info("Device: %s", device)
    torch.manual_seed(seed); np.random.seed(seed)

    logger.info("Loading and preparing %s ...", input_path)
    bundle = prepare_data(input_path, train_frac=train_frac, val_frac=val_frac)

    model = GraphSAGEFraudModel(node_in=N_NODE_FEATS, edge_in=N_EDGE_FEATS,
                                 hidden=hidden, emb=emb, dropout=dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %d params", n_params)

    train_node_x = bundle.train_g.node_x.to(device)
    train_edge_index = bundle.train_g.edge_index.to(device)
    val_node_x = bundle.val_g.node_x.to(device)
    val_edge_index = bundle.val_g.edge_index.to(device)
    test_node_x = bundle.test_g.node_x.to(device)
    test_edge_index = bundle.test_g.edge_index.to(device)

    tr_b = bundle.train_bid.to(device); tr_s = bundle.train_sid.to(device)
    va_b = bundle.val_bid.to(device);   va_s = bundle.val_sid.to(device)
    te_b = bundle.test_bid.to(device);  te_s = bundle.test_sid.to(device)
    tr_e = bundle.train_edge_x.to(device)
    va_e = bundle.val_edge_x.to(device)
    te_e = bundle.test_edge_x.to(device)
    y_tr = bundle.y_train.to(device); y_va = bundle.y_val.to(device); y_te = bundle.y_test.to(device)

    pos = float(y_tr.sum().item()); neg = float(len(y_tr) - pos)
    pos_weight = torch.tensor(neg / max(1.0, pos), device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_pr = -1.0
    best_state = None
    epochs_since_improve = 0
    history = []
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(train_node_x, train_edge_index, tr_b, tr_s, tr_e)
        loss = criterion(logits, y_tr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_node_x, val_edge_index, va_b, va_s, va_e)
            val_scores = torch.sigmoid(val_logits).cpu().numpy()
        val_m = aggregate_metrics(bundle.y_val.numpy(), val_scores)
        history.append({"epoch": ep, "train_loss": float(loss.item()), **val_m})

        improved = val_m["pr_auc"] > best_val_pr + 1e-6
        if improved:
            best_val_pr = val_m["pr_auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if ep % 5 == 0 or improved:
            logger.info("epoch %3d  loss=%.4f  val ROC=%.4f  PR=%.4f  F1=%.4f  R@5%%=%.4f%s",
                        ep, loss.item(), val_m["roc_auc"], val_m["pr_auc"], val_m["f1_best"],
                        val_m["recall_at_5pct_fpr"], "  *" if improved else "")
        if epochs_since_improve >= patience:
            logger.info("Early stop at epoch %d (patience=%d)", ep, patience)
            break
    train_time = time.perf_counter() - t0

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_logits = model(test_node_x, test_edge_index, te_b, te_s, te_e)
        test_scores = torch.sigmoid(test_logits).cpu().numpy()
        val_logits2 = model(val_node_x, val_edge_index, va_b, va_s, va_e)
        val_scores2 = torch.sigmoid(val_logits2).cpu().numpy()

    y_test_np = bundle.y_test.numpy()
    y_val_np  = bundle.y_val.numpy()
    test_m = aggregate_metrics(y_test_np, test_scores)
    val_m  = aggregate_metrics(y_val_np, val_scores2)

    thr_f1, _ = best_f1_threshold_from_curve(y_val_np, val_scores2)
    fpr_arr, tpr_arr, thr_roc = roc_curve(y_test_np, test_scores)
    mask = fpr_arr <= 0.05
    i_5 = int(np.where(mask)[0][-1])
    thr_5fpr = float(thr_roc[i_5])

    def agg_at(threshold: float) -> dict:
        pred = (test_scores >= threshold).astype(int)
        tp = int(((pred==1)&(y_test_np==1)).sum()); fp = int(((pred==1)&(y_test_np==0)).sum())
        fn = int(((pred==0)&(y_test_np==1)).sum()); tn = int(((pred==0)&(y_test_np==0)).sum())
        rec  = tp/(tp+fn) if (tp+fn) else 0.0
        fpr  = fp/(fp+tn) if (fp+tn) else 0.0
        prec = tp/(tp+fp) if (tp+fp) else 0.0
        f1v  = (2*prec*rec/(prec+rec)) if (prec+rec) else 0.0
        return {"recall": rec, "fpr": fpr, "f1": f1v, "precision": prec}

    ops = {
        "f1_optimal": {"threshold": thr_f1, "threshold_source": "best F1 on validation set",
                       "aggregate": agg_at(thr_f1),
                       "per_topology": per_topology_recall(bundle.test_rows, test_scores, thr_f1)},
        "fpr_5pct":   {"threshold": thr_5fpr, "threshold_source": "largest threshold with test FPR <= 5%",
                       "aggregate": agg_at(thr_5fpr),
                       "per_topology": per_topology_recall(bundle.test_rows, test_scores, thr_5fpr)},
    }

    # Latency — per-transaction inference on test set
    logger.info("Measuring per-row inference latency (%d rows)...", len(test_scores))
    times_ms = _measure_latency(model, test_node_x, test_edge_index, te_b, te_s, te_e, device)
    latency_ms = {
        "p50": float(np.percentile(times_ms, 50)),
        "p95": float(np.percentile(times_ms, 95)),
        "p99": float(np.percentile(times_ms, 99)),
    }
    logger.info("Latency ms p50=%.3f p95=%.3f p99=%.3f", latency_ms["p50"], latency_ms["p95"], latency_ms["p99"])

    results = {
        "method": "graphsage",
        "metrics": test_m,
        "latency_ms": latency_ms,
        "operating_points": ops,
        "n_train": int(len(bundle.train_rows)),
        "n_val":   int(len(bundle.val_rows)),
        "n_test":  int(len(bundle.test_rows)),
        "config": {
            "seed": seed, "train_frac": train_frac, "val_frac": val_frac,
            "device": str(device),
            "hidden_dim": hidden, "emb_dim": emb, "dropout": dropout,
            "lr": lr, "weight_decay": weight_decay, "epochs_max": epochs, "patience": patience,
            "n_node_features": N_NODE_FEATS,
            "n_edge_features": N_EDGE_FEATS,
            "n_model_params": int(n_params),
            "graph_stats": {
                "n_buyers": bundle.train_g.n_buyers,
                "n_suppliers": bundle.train_g.n_suppliers,
                "n_nodes_total": bundle.n_nodes,
                "train_unique_edges": bundle.train_g.n_unique_edges,
                "val_unique_edges":   bundle.val_g.n_unique_edges,
                "test_unique_edges":  bundle.test_g.n_unique_edges,
            },
            "training_time_seconds": round(train_time, 2),
            "history": history,
            "val_metrics_at_best": val_m,
            "best_f1_threshold": thr_f1,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s", output_path)
    return results


def _measure_latency(model, node_x, edge_index, b_idx, s_idx, e_x, device, warmup=100):
    # Pre-compute node embeddings once (standard for transductive inference):
    # per-transaction latency is then just the head forward on that pair.
    model.eval()
    with torch.no_grad():
        h_all = model.encoder(node_x, edge_index)
        n = b_idx.shape[0]
        # warmup
        for i in range(min(warmup, n)):
            _ = model.head(h_all[b_idx[i:i+1]], h_all[s_idx[i:i+1]], e_x[i:i+1])
        # sync device between iterations for accurate measurement on MPS/CUDA
        if device.type in ("cuda", "mps"):
            (torch.cuda.synchronize if device.type == "cuda"
             else torch.mps.synchronize)()
        times_ms = np.empty(n, dtype=np.float64)
        for i in range(n):
            t0 = time.perf_counter()
            _ = model.head(h_all[b_idx[i:i+1]], h_all[s_idx[i:i+1]], e_x[i:i+1])
            if device.type in ("cuda", "mps"):
                (torch.cuda.synchronize if device.type == "cuda"
                 else torch.mps.synchronize)()
            times_ms[i] = (time.perf_counter() - t0) * 1000.0
    return times_ms


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="STREAM-BSG GraphSAGE baseline")
    p.add_argument("--input", required=True, type=str)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--emb", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--smoke-test", action="store_true",
                   help="Build graphs and run ONE training step; report diagnostics without saving results.")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    if args.smoke_test:
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        logger.info("Device: %s", device)
        bundle = prepare_data(args.input, args.train_frac, args.val_frac)
        info = one_step_demo(bundle, args.seed, device)
        print("\n=== SMOKE-TEST DIAGNOSTICS ===")
        print(json.dumps(info, indent=2))
        print("\nGraph summary (from logs above): train/val/test edge counts and node counts.")
        print("If these look reasonable, rerun without --smoke-test for full training.")
        return

    if args.output is None:
        raise SystemExit("--output is required for full training")
    train_and_evaluate(
        input_path=args.input, output_path=args.output,
        seed=args.seed, train_frac=args.train_frac, val_frac=args.val_frac,
        epochs=args.epochs, hidden=args.hidden, emb=args.emb,
        lr=args.lr, weight_decay=args.weight_decay,
        dropout=args.dropout, patience=args.patience,
    )


if __name__ == "__main__":
    main()
