"""
classifier.py
=============

STREAM-BSG classifier — XGBoost on the 49 graph features produced by
``code/stream_bsg/features.py``. Reports the standard schema metrics plus
two STREAM-BSG-specific diagnostics:

  * per-topology recall at the val-selected best-F1 threshold
  * per-event T2 recall (an event = an original/duplicate pair; the event
    is detected if either row in the pair is flagged)

USAGE:
    python classifier.py \\
        --input  data/ieee_cis_with_features.parquet \\
        --output results/streambsg_results.json \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

LABEL_COL = "fraud_injected"
TOPOLOGY_COL = "fraud_topology"
TIME_COL = "TransactionDT"
FEATURE_PREFIX = "feat_"
LATENCY_WARMUP = 100

TOPOLOGIES = [
    "T1_vendor_injection",
    "T2_invoice_cycling",
    "T3_payment_term_manipulation",
    "T4_shell_supplier_ring",
    "T5_wire_redirection",
]


@dataclass
class SplitConfig:
    train_frac: float = 0.70
    val_frac: float = 0.15


def chronological_split(
    df: pd.DataFrame, cfg: SplitConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if TIME_COL not in df.columns:
        raise ValueError(f"Missing required time column '{TIME_COL}'")
    df_sorted = df.sort_values(TIME_COL, kind="mergesort").reset_index(drop=True)
    n = len(df_sorted)
    n_train = int(n * cfg.train_frac)
    n_val = int(n * cfg.val_frac)
    return (
        df_sorted.iloc[:n_train].reset_index(drop=True),
        df_sorted.iloc[n_train : n_train + n_val].reset_index(drop=True),
        df_sorted.iloc[n_train + n_val :].reset_index(drop=True),
    )


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = sorted([c for c in df.columns if c.startswith(FEATURE_PREFIX)])
    if not cols:
        raise ValueError(
            f"No columns with prefix '{FEATURE_PREFIX}' — run features.py before classifier.py"
        )
    return cols


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    roc_auc = float(roc_auc_score(y_true, y_score))
    pr_auc = float(average_precision_score(y_true, y_score))
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    denom = precision + recall
    f1_curve = np.where(denom > 0, 2 * precision * recall / np.where(denom > 0, denom, 1), 0.0)
    f1_best = float(np.max(f1_curve)) if f1_curve.size else 0.0
    fpr, tpr, _ = roc_curve(y_true, y_score)
    mask = fpr <= 0.05
    recall_at_5pct_fpr = float(tpr[mask].max()) if mask.any() else 0.0
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1_best": f1_best,
        "recall_at_5pct_fpr": recall_at_5pct_fpr,
    }


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Return (threshold, F1) maximizing F1 on a precision_recall_curve sweep."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision/recall arrays are length T+1; thresholds length T (last point is recall=0 by convention)
    denom = precision + recall
    f1 = np.where(denom > 0, 2 * precision * recall / np.where(denom > 0, denom, 1), 0.0)
    idx = int(np.argmax(f1[:-1])) if len(f1) > 1 else 0
    thr = float(thresholds[idx]) if idx < len(thresholds) else 0.5
    return thr, float(f1[idx])


def per_topology_recall(
    df_test: pd.DataFrame,
    y_score: np.ndarray,
    threshold: float,
) -> dict:
    out: dict = {}
    for topo in TOPOLOGIES:
        mask = (df_test[TOPOLOGY_COL].astype(str) == topo).to_numpy()
        n = int(mask.sum())
        if n == 0:
            out[topo] = {"n": 0, "recall": None}
            continue
        flagged = int((y_score[mask] >= threshold).sum())
        out[topo] = {"n": n, "detected": flagged, "recall": flagged / n}
    return out


def t2_per_event_recall(
    df_test: pd.DataFrame,
    y_score: np.ndarray,
    threshold: float,
) -> dict:
    """Per-event T2 recall.

    Each T2 fraud event is a pair (original row + _DUP_ row) sharing the same
    (buyer, supplier, base TransactionID). The event is detected if EITHER row
    in the pair scores at/above threshold.
    """
    mask = (df_test[TOPOLOGY_COL].astype(str) == "T2_invoice_cycling").to_numpy()
    n_rows = int(mask.sum())
    if n_rows == 0:
        return {"n_events": 0, "events_detected": 0, "recall_event": None,
                "n_rows": 0, "recall_row": None}

    sub = df_test.loc[mask, ["TransactionID", "buyer_id", "supplier_id"]].copy()
    sub["score"] = y_score[mask]
    sub["flagged"] = sub["score"] >= threshold
    sub["base_tid"] = sub["TransactionID"].astype(str).str.split("_DUP_").str[0]

    event_groups = sub.groupby(["buyer_id", "supplier_id", "base_tid"])
    n_events = int(event_groups.ngroups)
    events_detected = int(event_groups["flagged"].any().sum())

    return {
        "n_events": n_events,
        "events_detected": events_detected,
        "recall_event": events_detected / n_events if n_events else None,
        "n_rows": n_rows,
        "recall_row": float(sub["flagged"].mean()),
    }


def measure_per_row_latency(model: xgb.XGBClassifier, X_test: np.ndarray) -> dict:
    n = X_test.shape[0]
    if n == 0:
        return {"p50": None, "p95": None, "p99": None}
    for i in range(min(LATENCY_WARMUP, n)):
        model.predict_proba(X_test[i : i + 1])
    times_ms = np.empty(n, dtype=np.float64)
    for i in range(n):
        row = X_test[i : i + 1]
        t0 = time.perf_counter()
        model.predict_proba(row)
        times_ms[i] = (time.perf_counter() - t0) * 1000.0
    return {
        "p50": float(np.percentile(times_ms, 50)),
        "p95": float(np.percentile(times_ms, 95)),
        "p99": float(np.percentile(times_ms, 99)),
    }


def train_and_evaluate(
    data_path: str,
    output_path: str,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> dict:
    logger.info("Loading %s", data_path)
    df = pd.read_parquet(data_path)
    logger.info("Loaded %d rows, %d columns", len(df), df.shape[1])

    if LABEL_COL not in df.columns:
        raise ValueError(f"Input missing label column '{LABEL_COL}'")

    feature_cols = select_feature_columns(df)
    logger.info("Using %d STREAM-BSG features (feat_*)", len(feature_cols))

    cfg = SplitConfig(train_frac=train_frac, val_frac=val_frac)
    train, val, test = chronological_split(df, cfg)
    logger.info("Split: train=%d val=%d test=%d", len(train), len(val), len(test))

    X_train = train[feature_cols].to_numpy(dtype=np.float32)
    X_val = val[feature_cols].to_numpy(dtype=np.float32)
    X_test = test[feature_cols].to_numpy(dtype=np.float32)
    y_train = train[LABEL_COL].to_numpy().astype(int)
    y_val = val[LABEL_COL].to_numpy().astype(int)
    y_test = test[LABEL_COL].to_numpy().astype(int)

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0
    logger.info(
        "Class balance: train pos=%d/%d (%.3f%%) -> scale_pos_weight=%.2f",
        pos, len(y_train), 100 * y_train.mean(), scale_pos_weight,
    )

    xgb_params = dict(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=-1,
    )
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_scores = model.predict_proba(X_val)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, test_scores)
    val_metrics = compute_metrics(y_val, val_scores)

    thr, val_f1 = best_f1_threshold(y_val, val_scores)
    logger.info(
        "Val-selected best-F1 threshold = %.6f (val F1=%.4f)", thr, val_f1
    )

    topo_recall = per_topology_recall(test, test_scores, thr)
    t2_event = t2_per_event_recall(test, test_scores, thr)

    logger.info(
        "Test  ROC-AUC=%.4f  PR-AUC=%.4f  F1_best=%.4f  Recall@5%%FPR=%.4f",
        metrics["roc_auc"], metrics["pr_auc"], metrics["f1_best"], metrics["recall_at_5pct_fpr"],
    )
    for topo, info in topo_recall.items():
        if info["recall"] is None:
            logger.info("  %-32s n=0 (no test rows)", topo)
        else:
            logger.info("  %-32s n=%d  detected=%d  recall=%.4f",
                        topo, info["n"], info["detected"], info["recall"])
    if t2_event["recall_event"] is not None:
        logger.info(
            "  T2 per-event:                  n_events=%d  detected=%d  recall_event=%.4f  (row-level recall=%.4f)",
            t2_event["n_events"], t2_event["events_detected"],
            t2_event["recall_event"], t2_event["recall_row"],
        )

    logger.info("Measuring per-row inference latency on %d test rows...", len(X_test))
    latency_ms = measure_per_row_latency(model, X_test)
    logger.info(
        "Latency ms p50=%.3f p95=%.3f p99=%.3f",
        latency_ms["p50"], latency_ms["p95"], latency_ms["p99"],
    )

    results = {
        "method": "streambsg",
        "metrics": metrics,
        "latency_ms": latency_ms,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "config": {
            "seed": seed,
            "train_frac": train_frac,
            "val_frac": val_frac,
            "n_features": len(feature_cols),
            "feature_cols": feature_cols,
            "xgb_params": xgb_params,
            "latency_warmup": LATENCY_WARMUP,
            "val_metrics": val_metrics,
            "best_f1_threshold": thr,
            "val_f1_at_threshold": val_f1,
        },
        "per_topology_recall": topo_recall,
        "t2_per_event_recall": t2_event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s", out)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="STREAM-BSG XGBoost classifier on 49 graph features")
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    train_and_evaluate(
        data_path=args.input,
        output_path=args.output,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )


if __name__ == "__main__":
    main()
