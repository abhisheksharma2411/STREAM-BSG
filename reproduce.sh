#!/usr/bin/env bash
# =============================================================================
# STREAM-BSG end-to-end reproduction script
# IEEE InC4 2026 — Paper 3092
# =============================================================================
#
# Runs the full pipeline that produces every JSON in results/:
#   1. Environment check       (Python >= 3.10, pip)
#   2. Install pinned requirements
#   3. Verify IEEE-CIS data is present (manual download step)
#   4. Synthetic B2B injection at 1 % rate, seed 42
#   5. 49-feature streaming extractor (dominant cost: ~85 min)
#   6. STREAM-BSG XGBoost classifier   -> results/streambsg_results.json
#   7. Baselines (LR, XGBoost-tabular, GraphSAGE)
#   8. Print output-verification pointers
#
# Total wall time: ~2 hours on a 2024-era Mac / mid-range Linux workstation.
# The feature extractor is by far the slowest step; individual scripts can be
# rerun independently. See REPRODUCE.md for step-by-step guidance and for how
# to reproduce individual ablations (WI1, WI3, WI5, WI8, seed sweep).
# =============================================================================

set -euo pipefail

# ------------------------------ configuration --------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${REPO_ROOT}/data"
IEEE_CSV="${DATA_DIR}/ieee-cis/train_transaction.csv"
LEGACY_CSV="${DATA_DIR}/ieee-fraud-detection/train_transaction.csv"
INJECTED_PARQUET="${DATA_DIR}/ieee_cis_with_synthetic_b2b.parquet"
FEATURES_PARQUET="${DATA_DIR}/ieee_cis_with_features.parquet"
RESULTS_DIR="${REPO_ROOT}/results"
SEED=42

BLUE='\033[0;34m'; GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[0;33m'; NC='\033[0m'

banner() {
    echo
    echo -e "${BLUE}==============================================================${NC}"
    echo -e "${BLUE}$*${NC}"
    echo -e "${BLUE}==============================================================${NC}"
}
ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}!${NC} $*"; }
die()  { echo -e "${RED}✗${NC} $*" >&2; exit 1; }

banner "STREAM-BSG reproduction pipeline (Paper 3092, IEEE InC4 2026)"

# ------------------------------ 1. Python check ------------------------------
banner "1/8  Python version check"
if ! command -v python3 >/dev/null 2>&1; then
    die "python3 not found on PATH. Install Python 3.10+ and retry."
fi
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_OK=$(python3 -c "import sys; print(1 if sys.version_info >= (3,10) else 0)")
if [ "$PY_OK" != "1" ]; then
    die "Python ${PY_VER} detected; STREAM-BSG requires Python >= 3.10."
fi
ok "python3 = ${PY_VER}"

# ------------------------------ 2. Install deps ------------------------------
banner "2/8  Install requirements (pinned versions)"
if [ ! -f "${REPO_ROOT}/requirements.txt" ]; then
    die "requirements.txt missing at repo root"
fi
python3 -m pip install --upgrade pip >/dev/null 2>&1 || warn "pip self-upgrade failed; continuing"
if ! python3 -m pip install -r "${REPO_ROOT}/requirements.txt"; then
    die "pip install failed. If PyTorch wheels are the issue, install torch and torch-geometric manually per your platform, then re-run."
fi
ok "requirements installed"

# ------------------------------ 3. Data check --------------------------------
banner "3/8  IEEE-CIS dataset check"
if [ -f "${IEEE_CSV}" ]; then
    INPUT_CSV="${IEEE_CSV}"
    ok "found ${IEEE_CSV}"
elif [ -f "${LEGACY_CSV}" ]; then
    INPUT_CSV="${LEGACY_CSV}"
    ok "found ${LEGACY_CSV} (legacy path from initial repo layout)"
else
    cat <<EOF >&2

$(echo -e "${RED}✗${NC}") IEEE-CIS dataset not found. STREAM-BSG cannot reproduce without it.

The dataset is NOT redistributed in this repository (per the Kaggle license).
Fetch it via ONE of the following:

  A) kaggle CLI (auto):
       kaggle competitions download -c ieee-fraud-detection -p data/ieee-cis/
       unzip data/ieee-cis/ieee-fraud-detection.zip -d data/ieee-cis/

  B) Manual: visit https://www.kaggle.com/c/ieee-fraud-detection/data ,
     accept the competition rules, then place train_transaction.csv into
     ${DATA_DIR}/ieee-cis/ (create the directory if needed).

Re-run this script after the file exists.
EOF
    exit 1
fi

# ------------------------------ 4. Synth injection ---------------------------
banner "4/8  Synthetic B2B fraud injection (1 %, seed=${SEED})  [~40 s]"
mkdir -p "${DATA_DIR}"
if ! python3 "${REPO_ROOT}/code/synth/synth_b2b_injection.py" \
        --input "${INPUT_CSV}" \
        --output "${INJECTED_PARQUET}" \
        --inject-rate 0.01 --seed ${SEED}; then
    die "Synthetic injection failed. Check that ${INPUT_CSV} is the raw IEEE-CIS train_transaction.csv (not a Parquet or subset)."
fi
ok "wrote ${INJECTED_PARQUET}"

# ------------------------------ 5. Feature extraction ------------------------
banner "5/8  49-feature streaming extractor  [~85 minutes — this is the slow step]"
if ! python3 "${REPO_ROOT}/code/stream_bsg/features.py" \
        --input "${INJECTED_PARQUET}" \
        --output "${FEATURES_PARQUET}"; then
    die "Feature pipeline failed. Did the synth step above complete? Check ${INJECTED_PARQUET} exists and has fraud_injected + fraud_topology columns."
fi
ok "wrote ${FEATURES_PARQUET}"

# ------------------------------ 6. STREAM-BSG classifier ---------------------
banner "6/8  STREAM-BSG classifier (49 features, seed=${SEED})  [~30 s]"
mkdir -p "${RESULTS_DIR}"
if ! python3 "${REPO_ROOT}/code/stream_bsg/classifier.py" \
        --input "${FEATURES_PARQUET}" \
        --output "${RESULTS_DIR}/streambsg_results.json" \
        --seed ${SEED}; then
    die "STREAM-BSG classifier failed. Feature pipeline output missing or malformed?"
fi
ok "wrote ${RESULTS_DIR}/streambsg_results.json"

# ------------------------------ 7. Baselines ---------------------------------
banner "7/8  Baselines (LR, XGB-tabular, GraphSAGE)"

echo "  → Logistic Regression..."
python3 "${REPO_ROOT}/code/baselines/lr_baseline.py" \
    --input "${INJECTED_PARQUET}" \
    --output "${RESULTS_DIR}/lr_results.json" \
    --seed ${SEED} \
    || die "LR baseline failed. Injected parquet missing or column selection error?"
ok "wrote ${RESULTS_DIR}/lr_results.json"

echo "  → XGBoost-tabular..."
python3 "${REPO_ROOT}/code/baselines/xgb_tabular.py" \
    --input "${INJECTED_PARQUET}" \
    --output "${RESULTS_DIR}/xgb_tabular_results.json" \
    --seed ${SEED} \
    || die "XGB-tabular baseline failed."
ok "wrote ${RESULTS_DIR}/xgb_tabular_results.json"

echo "  → GraphSAGE (transductive, bipartite)..."
if ! python3 "${REPO_ROOT}/code/baselines/graphsage_baseline.py" \
        --input "${INJECTED_PARQUET}" \
        --output "${RESULTS_DIR}/graphsage_results.json" \
        --seed ${SEED}; then
    warn "GraphSAGE failed (torch / torch-geometric install issue on this platform?). Skipping — other baselines and STREAM-BSG results are still valid."
else
    ok "wrote ${RESULTS_DIR}/graphsage_results.json"
fi

# ------------------------------ 8. Verify ------------------------------------
banner "8/8  Verify results"
cat <<EOF

Results written to: ${RESULTS_DIR}

Headline artifacts (Table II source of truth):
  streambsg_results.json      STREAM-BSG (49 features)
  lr_results.json             Logistic Regression tabular baseline
  xgb_tabular_results.json    XGBoost tabular baseline
  graphsage_results.json      GraphSAGE baseline (if step 7 succeeded)

Expected STREAM-BSG headline metrics at seed=42 (from Paper 3092):
  ROC-AUC       0.8885
  PR-AUC        0.6778
  F1_best       0.7767
  Recall@5%FPR  0.7111

Quick check:
  python3 -c "import json; m=json.load(open('${RESULTS_DIR}/streambsg_results.json'))['metrics']; print(m)"

For ablation reruns (WI1 / WI3 / WI5 / WI8, seed sweeps), see REPRODUCE.md.
EOF

ok "reproduce.sh completed successfully"
