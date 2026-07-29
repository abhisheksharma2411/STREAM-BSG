# STREAM-BSG
 
**A Streaming Graph Architecture for Real-Time Fraud Detection in B2B Payment Networks**
 
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
 
This repository contains the companion code, figures, and reproduction scripts for the STREAM-BSG paper, accepted at the **2026 IEEE International Conference on Contemporary Computing and Communications (InC4)** (Paper ID 3092; to appear in IEEE Xplore).
 
## What is STREAM-BSG?
 
STREAM-BSG (Streaming Buyer–Supplier Graph) is a streaming graph architecture for detecting fraud in business-to-business (B2B) payment networks in real time. Unlike most academic fraud-detection work — which targets business-to-consumer (B2C) credit-card transactions — STREAM-BSG explicitly models the **buyer–supplier–invoice graph** that characterizes B2B payments, and formalizes five fraud-topology patterns documented in industry practice:
 
1. **Vendor Injection (T1)** — a new shell supplier receives concentrated payments from a single buyer, then goes quiet.
2. **Invoice Cycling (T2)** — repeated, near-duplicate invoices on the same buyer–supplier edge within a short window.
3. **Payment-Term Manipulation (T3)** — an abrupt shift in payment terms on an established edge.
4. **Shell-Supplier Ring (T4)** — a buyer connected only to several coordinated, low-history suppliers.
5. **Wire Redirection / BEC (T5)** — a bank/remittance-attribute change followed by an unusually high-value payment.
The architecture maintains an incremental buyer–supplier graph in a streaming state store (Redis) and computes a **49-dimensional feature vector** online — **15 node + 18 edge + 14 subgraph + 2 current-row change-detection features** — before classification with XGBoost.
 
**On latency (stated precisely):** the reported **0.37 ms p99** is *measured* per-row model-inference latency. The **sub-100 ms p99** figure is an *architectural budget* across the full pipeline (ingest → state → features → inference → decision), **not** a measured end-to-end benchmark; a full Kafka/Flink/Redis deployment benchmark is future work.
 
## Repository structure
 
```
STREAM-BSG/
├── code/
│   ├── synth/                  # Synthetic B2B fraud injection on IEEE-CIS   [runnable]
│   │   └── synth_b2b_injection.py
│   ├── figures/                # Figure generation scripts                   [runnable]
│   │   ├── fig1_taxonomy.py
│   │   └── fig2_architecture.py
│   ├── stream_bsg/             # 49-feature incremental pipeline + XGBoost    [in progress]
│   ├── baselines/              # Logistic Regression, XGBoost-tabular, GraphSAGE [in progress]
│   └── eval/                   # Evaluation harness (Tables II–VI)            [in progress]
├── figures/                    # Pre-generated figures (PDF + PNG)
├── data/                       # Data placement instructions (IEEE-CIS not redistributed)
├── paper/                      # Paper artifacts
├── CITATION.cff
├── requirements.txt
├── LICENSE
└── README.md
```
 
## Reproducibility
 
**Runnable today:** the synthetic B2B fraud-injection script (`code/synth/`) and the figure-generation scripts (`code/figures/`).
 
**Being finalized:** the full 49-feature pipeline (`code/stream_bsg/`), the baseline implementations (`code/baselines/`), and the evaluation harness (`code/eval/`) that regenerates the paper's Tables II–VI are being completed for the camera-ready and extended-journal versions.
 
Reported STREAM-BSG results were produced under pinned dependencies — `xgboost==3.2.0`, `scikit-learn==1.8.0` — and reproduce exactly (|Δ| = 0 across all four aggregate metrics and the top-20 feature-importance gains) under that environment.
 
## Quick start
 
### Prerequisites
- Python 3.10+
- pip
### Setup
```bash
git clone https://github.com/abhisheksharma2411/STREAM-BSG.git
cd STREAM-BSG
pip install -r requirements.txt
```
 
### Reproduce the figures
```bash
cd code/figures
python3 fig1_taxonomy.py
python3 fig2_architecture.py
```
Output PDFs and PNGs are written to `figures/`.
 
### Run synthetic B2B fraud injection on IEEE-CIS
1. Download the IEEE-CIS Fraud Detection dataset from Kaggle: <https://www.kaggle.com/c/ieee-fraud-detection/data> (requires a Kaggle account).
2. Place `train_transaction.csv` and `train_identity.csv` in the `data/` directory.
3. Run:
```bash
python3 code/synth/synth_b2b_injection.py \
    --input data/train_transaction.csv \
    --output data/ieee_cis_with_synthetic_b2b.parquet \
    --fraud-rate 0.01
```
This produces a labeled Parquet file with two added columns: `fraud_injected` (binary) and `fraud_topology` (one of T1–T5, or NULL for non-injected rows).
 
## Dataset
 
The IEEE-CIS dataset is **not** redistributed here; download it directly from Kaggle under its original license. A standalone, DOI-citable synthetic B2B payment-fraud dataset (**SynB2B-Fraud**) generated with the same topology rules is being released separately on Zenodo — a link will be added here when available.
 
## Roadmap
 
- [x] Synthetic B2B fraud injection script
- [x] Figure generation (taxonomy, architecture)
- [ ] Baseline implementations (Logistic Regression, XGBoost-tabular, GraphSAGE)
- [ ] STREAM-BSG 49-feature pipeline
- [ ] End-to-end evaluation harness (Tables II–VI)
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
 
