# STREAM-BSG
 
**A Streaming Graph Architecture for Real-Time Fraud Detection in B2B Payment Networks**
 
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
 
This repository contains the companion code, figures, datasets, and full reproduction pipeline for the STREAM-BSG paper, accepted at the **2026 IEEE International Conference on Contemporary Computing and Communications (InC4)** (Paper ID 3092; to appear in IEEE Xplore).
 
## What is STREAM-BSG?
 
STREAM-BSG (Streaming Buyer–Supplier Graph) is a streaming graph architecture for detecting fraud in business-to-business (B2B) payment networks in real time. Unlike most academic fraud-detection work — which targets business-to-consumer (B2C) credit-card transactions — STREAM-BSG explicitly models the **buyer–supplier–invoice graph** that characterizes B2B payments, and formalizes five fraud-topology patterns documented in industry practice:
 
1. **Vendor Injection (T1)** — a new shell supplier receives concentrated payments from a single buyer, then goes quiet.
2. **Invoice Cycling (T2)** — repeated, near-duplicate invoices on the same buyer–supplier edge within a short window.
3. **Payment-Term Manipulation (T3)** — an abrupt shift in payment terms on an established edge.
4. **Shell-Supplier Ring (T4)** — a buyer connected only to several coordinated, low-history suppliers.
5. **Wire Redirection / BEC (T5)** — a bank/remittance-attribute change followed by an unusually high-value payment.
The architecture maintains an incremental buyer–supplier graph in a streaming state store (Redis) and computes a **49-dimensional feature vector** online — **15 node + 18 edge + 14 subgraph + 2 current-row change-detection features** — before classification with XGBoost. The complete per-feature definitions are in [`paper/feature_table.md`](paper/feature_table.md); the formal topology rules are in [`paper/topology_definitions.md`](paper/topology_definitions.md).
 
**On latency (stated precisely):** the reported **0.37 ms p99** is *measured* per-row model-inference latency. The **sub-100 ms p99** figure is an *architectural budget* across the full pipeline (ingest → state → features → inference → decision), **not** a measured end-to-end benchmark; a full Kafka/Flink/Redis deployment benchmark is future work.
 
## Repository structure
 
```
STREAM-BSG/
├── code/
│   ├── synth/
│   │   ├── synth_b2b_injection.py       # Inject the 5 B2B topologies into IEEE-CIS   [runnable]
│   │   └── synth_b2b_injection_ood.py   # Out-of-distribution attack variants (Abl. C) [runnable]
│   ├── figures/
│   │   ├── fig1_taxonomy.py             # Taxonomy figure                             [runnable]
│   │   └── fig2_architecture.py         # Architecture figure                         [runnable]
│   ├── stream_bsg/
│   │   ├── features.py                  # 49-feature streaming extractor              [runnable]
│   │   └── classifier.py                # XGBoost + metrics + per-topology + latency   [runnable]
│   └── baselines/
│       ├── lr_baseline.py               # Logistic Regression                         [runnable]
│       ├── xgb_tabular.py               # XGBoost on raw tabular features             [runnable]
│       └── graphsage_baseline.py        # 2-layer heterogeneous GraphSAGE (PyG)        [runnable]
├── results/                             # All committed result JSONs + summary tables
│   ├── table_accuracy.csv               #   Table II
│   ├── table_latency.csv                #   Table III
│   ├── table_per_topology.csv           #   Table IV
│   ├── graphsage_seed*.json             #   GraphSAGE seed-stability runs
│   ├── seed_stability.json              #   STREAM-BSG seed-stability summary
│   ├── *_no_synth_leak.json             #   Leakage ablation (Table V)
│   ├── ablation_a_no_topology_feats.json, ablation_0p5pct/, ablation_c_ood_injection.json  # Table VI
│   ├── leakage_analysis_report.md
│   └── all_experiments_summary.md
├── figures/                             # Pre-generated figures (PDF + PNG)
├── data/                                # Data placement instructions (IEEE-CIS not redistributed)
├── paper/
│   ├── feature_table.md                 # Full 49-feature definitions
│   ├── topology_definitions.md          # Formal T1–T5 firing rules
│   └── README.md
├── reproduce.sh                         # One-command end-to-end reproduction
├── REPRODUCE.md                         # Step-by-step reproduction guide
├── CITATION.cff
├── requirements.txt
├── LICENSE
└── README.md
```
 
## Reproducibility
 
The full pipeline that regenerates the paper's results runs end to end. The quickest path:
 
```bash
pip install -r requirements.txt        # includes torch + torch-geometric for GraphSAGE
bash reproduce.sh                       # see REPRODUCE.md for the step-by-step breakdown
```
 
Or run the stages individually:
 
```bash
# 1. Inject the five B2B topologies into IEEE-CIS
python3 code/synth/synth_b2b_injection.py --input data/train_transaction.csv \
    --output data/ieee_cis_with_synthetic_b2b.parquet --fraud-rate 0.01
# 2. Compute the 49 streaming features (single chronological pass, no future leakage)
python3 code/stream_bsg/features.py --input data/ieee_cis_with_synthetic_b2b.parquet \
    --output data/ieee_cis_with_features.parquet
# 3. Train + score STREAM-BSG and the baselines
python3 code/stream_bsg/classifier.py  ...    # -> results/streambsg_results.json
python3 code/baselines/lr_baseline.py  ...    # -> results/lr_results.json
python3 code/baselines/xgb_tabular.py  ...    # -> results/xgb_tabular_results.json
python3 code/baselines/graphsage_baseline.py ...  # -> results/graphsage_results.json
```
 
Committed outputs in [`results/`](results/) map directly to the paper: **Table II** (accuracy) → `table_accuracy.csv`; **Table III** (latency) → `table_latency.csv`; **Table IV** (per-topology recall) → `table_per_topology.csv`; **Table V** (synthetic-attribute leakage) → `*_no_synth_leak.json` + `leakage_analysis_report.md`; **Table VI** (feature / prevalence / OOD ablations) → `ablation_*`; and the GraphSAGE seed-stability footnote → `graphsage_seed*.json` + `seed_stability.json`. A consolidated write-up is in [`results/all_experiments_summary.md`](results/all_experiments_summary.md).
 
Results were produced under pinned dependencies (see `requirements.txt`, incl. `xgboost==3.2.0`, `scikit-learn==1.8.0`) and reproduce exactly (|Δ| = 0 across the four aggregate metrics and the top-20 feature-importance gains).
 
## Quick start
 
**Prerequisites:** Python 3.10+, pip. GraphSAGE additionally needs `torch` + `torch-geometric` (in `requirements.txt`).
 
```bash
git clone https://github.com/abhisheksharma2411/STREAM-BSG.git
cd STREAM-BSG
pip install -r requirements.txt
```
 
The IEEE-CIS Fraud Detection dataset is **not** redistributed here — download `train_transaction.csv` and `train_identity.csv` from Kaggle (<https://www.kaggle.com/c/ieee-fraud-detection/data>) into `data/`, then run `bash reproduce.sh`.
 
## Dataset
 
A standalone, DOI-citable synthetic B2B payment-fraud dataset (**SynB2B-Fraud**), generated with the same topology rules, is being released separately on Zenodo — a link will be added here when available.
 
## Roadmap
 
- [x] Synthetic B2B fraud injection (in-distribution + OOD variants)
- [x] Figure generation (taxonomy, architecture)
- [x] STREAM-BSG 49-feature streaming extractor + XGBoost classifier
- [x] Baselines: Logistic Regression, XGBoost-tabular, GraphSAGE
- [x] Ablation studies (leakage, feature, prevalence, OOD — Tables V & VI)
- [x] Seed-stability experiments
- [x] End-to-end reproduction script (`reproduce.sh` + `REPRODUCE.md`)
- [ ] Streaming deployment example (Kafka + Flink + Redis)
- [ ] Standalone SynB2B-Fraud dataset (Zenodo DOI)
## Citation
 
If you use this code or methodology, please cite:
 
```bibtex
@inproceedings{sharma2026streambsg,
  title     = {{STREAM-BSG}: A Streaming Graph Architecture for Real-Time
               Fraud Detection in {B2B} Payment Networks},
  author    = {Sharma, Abhishek},
  booktitle = {Proceedings of the 2026 IEEE International Conference on
               Contemporary Computing and Communications (InC4)},
  year      = {2026},
  publisher = {IEEE},
  note      = {To appear}
}
```
 
## License
 
MIT License — see [LICENSE](LICENSE).
 
## Contact
 
Abhishek Sharma — <abhicse24@gmail.com> · ORCID: [0009-0007-1103-2103](https://orcid.org/0009-0007-1103-2103)
 
---
 
*The IEEE-CIS dataset is NOT redistributed in this repository. Users must download it directly from Kaggle under its original license terms.*
