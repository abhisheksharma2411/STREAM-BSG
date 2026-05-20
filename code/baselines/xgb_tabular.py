"""
xgb_tabular.py
==============

XGBoost tabular baseline for STREAM-BSG.

Consumes the labeled parquet produced by ``code/synth/synth_b2b_injection.py``
and reports classification metrics plus per-row inference latency under a
chronological 70/15/15 train/val/test split.

USAGE:
    python xgb_tabular.py \\
        --input  data/ieee_cis_with_synthetic_b2b.parquet \\
        --output results/xgb_tabular_results.json \\
        --seed 42

Outputs a JSON results file conforming to the STREAM-BSG common schema
(see CLAUDE.md). Latency p50/p95/p99 are measured by predicting on each
test row INDIVIDUALLY in a tight loop after a 100-prediction warm-up,
per the CLAUDE.md latency protocol.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)

LABEL_COL = "fraud_injected"
TOPOLOGY_COL = "fraud_topology"
TIME_COL = "TransactionDT"
LEAKY_COLS = {LABEL_COL, TOPOLOGY_COL, "isFraud"}
MAX_CAT_CARDINALITY = 50
LATENCY_WARMUP = 100


@dataclass
class SplitConfig:
    train_frac: float = 0.70
    val_frac: float = 0.15


def chronological_split(
    df: pd.DataFrame, cfg: SplitConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if TIME_COL not in df.columns:
        raise ValueError(f"Missing required time column '{TIME_COL}' for chronological split")
    df_sorted = df.sort_values(TIME_COL, kind="mergesort").reset_index(drop=True)
    n = len(df_sorted)
    n_train = int(n * cfg.train_frac)
    n_val = int(n * cfg.val_frac)
    return (
        df_sorted.iloc[:n_train],
        df_sorted.iloc[n_train : n_train + n_val],
        df_sorted.iloc[n_train + n_val :],
    )


def select_feature_columns(
    df: pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    dropped_cols: list[str] = []
    for col in df.columns:
        if col in LEAKY_COLS:
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
        else:
            n_unique = series.nunique(dropna=True)
            if n_unique < MAX_CAT_CARDINALITY:
                categorical_cols.append(col)
            else:
                dropped_cols.append(col)
    return numeric_cols, categorical_cols, dropped_cols


def build_design_matrix(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numeric: keep NaN (XGBoost handles it). Categorical: one-hot encode."""
    parts_train: list[np.ndarray] = []
    parts_val: list[np.ndarray] = []
    parts_test: list[np.ndarray] = []

    if numeric_cols:
        parts_train.append(train[numeric_cols].to_numpy(dtype=np.float32, na_value=np.nan))
        parts_val.append(val[numeric_cols].to_numpy(dtype=np.float32, na_value=np.nan))
        parts_test.append(test[numeric_cols].to_numpy(dtype=np.float32, na_value=np.nan))

    if categorical_cols:
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        cat_train = train[categorical_cols].astype(object).fillna("__missing__")
        cat_val = val[categorical_cols].astype(object).fillna("__missing__")
        cat_test = test[categorical_cols].astype(object).fillna("__missing__")
        parts_train.append(ohe.fit_transform(cat_train).astype(np.float32))
        parts_val.append(ohe.transform(cat_val).astype(np.float32))
        parts_test.append(ohe.transform(cat_test).astype(np.float32))

    X_train = np.hstack(parts_train) if parts_train else np.zeros((len(train), 0), dtype=np.float32)
    X_val = np.hstack(parts_val) if parts_val else np.zeros((len(val), 0), dtype=np.float32)
    X_test = np.hstack(parts_test) if parts_test else np.zeros((len(test), 0), dtype=np.float32)
    return X_train, X_val, X_test


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


def measure_per_row_latency(
    model: xgb.XGBClassifier,
    X_test: np.ndarray,
    warmup: int = LATENCY_WARMUP,
) -> dict:
    """Per-row inference latency: predict on each test row INDIVIDUALLY.

    Per CLAUDE.md: warm-up of 100 predictions, then a tight loop calling
    predict_proba on a single row (shape (1, d)) using time.perf_counter().
    """
    n = X_test.shape[0]
    if n == 0:
        return {"p50": None, "p95": None, "p99": None}

    for i in range(min(warmup, n)):
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
        raise ValueError(f"Input parquet is missing required label column '{LABEL_COL}'")

    cfg = SplitConfig(train_frac=train_frac, val_frac=val_frac)
    train, val, test = chronological_split(df, cfg)
    logger.info("Split: train=%d val=%d test=%d", len(train), len(val), len(test))

    numeric_cols, categorical_cols, dropped_cols = select_feature_columns(df)
    logger.info(
        "Features: %d numeric, %d categorical (<%d uniques), %d dropped (high-card)",
        len(numeric_cols),
        len(categorical_cols),
        MAX_CAT_CARDINALITY,
        len(dropped_cols),
    )

    X_train, X_val, X_test = build_design_matrix(
        train, val, test, numeric_cols, categorical_cols
    )
    y_train = train[LABEL_COL].to_numpy().astype(int)
    y_val = val[LABEL_COL].to_numpy().astype(int)
    y_test = test[LABEL_COL].to_numpy().astype(int)

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0
    logger.info(
        "Class balance: train pos=%d/%d (%.3f%%) -> scale_pos_weight=%.2f",
        pos,
        len(y_train),
        100 * y_train.mean(),
        scale_pos_weight,
    )

    xgb_params = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=-1,
    )
    model = xgb.XGBClassifier(**xgb_params)
    fit_kwargs: dict = {}
    if len(X_val) > 0:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["verbose"] = False
    model.fit(X_train, y_train, **fit_kwargs)

    test_scores = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, test_scores)
    val_metrics = compute_metrics(y_val, model.predict_proba(X_val)[:, 1]) if len(X_val) else None

    logger.info(
        "Test  ROC-AUC=%.4f  PR-AUC=%.4f  F1_best=%.4f  Recall@5%%FPR=%.4f",
        metrics["roc_auc"],
        metrics["pr_auc"],
        metrics["f1_best"],
        metrics["recall_at_5pct_fpr"],
    )

    logger.info("Measuring per-row inference latency on %d test rows...", len(X_test))
    latency_ms = measure_per_row_latency(model, X_test)
    logger.info(
        "Latency ms p50=%.3f p95=%.3f p99=%.3f",
        latency_ms["p50"],
        latency_ms["p95"],
        latency_ms["p99"],
    )

    results = {
        "method": "xgb_tabular",
        "metrics": metrics,
        "latency_ms": latency_ms,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "config": {
            "seed": seed,
            "train_frac": train_frac,
            "val_frac": val_frac,
            "max_cat_cardinality": MAX_CAT_CARDINALITY,
            "n_numeric_features": len(numeric_cols),
            "n_categorical_features": len(categorical_cols),
            "n_dropped_high_card": len(dropped_cols),
            "xgb_params": xgb_params,
            "latency_warmup": LATENCY_WARMUP,
            "val_metrics": val_metrics,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s", out)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="STREAM-BSG XGBoost tabular baseline")
    parser.add_argument("--input", required=True, type=str, help="Path to labeled parquet")
    parser.add_argument("--output", required=True, type=str, help="Path to write results JSON")
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
