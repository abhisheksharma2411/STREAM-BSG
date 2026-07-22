# STREAM-BSG

**A Streaming Graph Architecture for Real-Time Fraud Detection in B2B Payment Networks**

This repository contains the companion code, figures, and reproduction scripts for the STREAM-BSG paper (submitted to IEEE InC4 2026).

## What is STREAM-BSG?

STREAM-BSG (Streaming Buyer-Supplier Graph) is a streaming graph architecture for detecting fraud in business-to-business (B2B) payment networks in real time. Unlike most academic fraud detection work — which targets business-to-consumer (B2C) credit card transactions — STREAM-BSG explicitly models the buyer-supplier-invoice graph structure that characterizes B2B payments and formalizes five fraud topology patterns documented in industry literature:

1. **Vendor Injection** — a new shell supplier receives concentrated payments
2. **Invoice Cycling** — duplicate-attribute invoices within a short window
3. **Payment-Term Manipulation** — abrupt shift in payment terms on an established edge
4. **Shell-Supplier Ring** — coordinated low-history suppliers
5. **Wire Redirection (BEC)** — bank attribute change followed by high-value payment

The architecture maintains an incremental buyer-supplier graph in a streaming state store (Redis) and computes 47 graph features online before classification, targeting sub-100ms p99 latency end-to-end.

## Repository structure

```
stream-bsg/
├── code/
│   ├── synth/                  # Synthetic B2B fraud injection on IEEE-CIS
│   │   └── synth_b2b_injection.py
│   ├── figures/                # Figure generation scripts
│   │   ├── fig1_taxonomy.py
│   │   └── fig2_architecture.py
│   ├── stream_bsg/             # Main STREAM-BSG implementation (WIP)
│   │   └── README.md
│   ├── baselines/              # Baseline implementations (WIP)
│   │   └── README.md
│   └── eval/                   # Evaluation scripts (WIP)
│       └── README.md
├── figures/                    # Pre-generated figures (PDF + PNG)
│   ├── fig1_taxonomy.pdf
│   ├── fig1_taxonomy.png
│   ├── fig2_architecture.pdf
│   └── fig2_architecture.png
├── data/                       # Data placement instructions
│   └── README.md
├── paper/                      # Paper artifacts
│   └── README.md
├── requirements.txt
├── LICENSE
└── README.md
```

## Quick start

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
git clone https://github.com/<your-username>/stream-bsg.git
cd stream-bsg
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

This produces a labeled parquet file with two added columns: `fraud_injected` (binary) and `fraud_topology` (one of T1-T5 or NULL for non-injected rows).

## Roadmap

This repository accompanies the InC4 2026 submission. Items marked WIP will be completed for the camera-ready version and the extended journal version.

- [x] Synthetic B2B fraud injection script
- [x] Figure generation (taxonomy, architecture)
- [ ] Baseline implementations (Logistic Regression, XGBoost-tabular, GraphSAGE)
- [ ] STREAM-BSG 49-feature pipeline
- [ ] End-to-end evaluation harness
- [ ] Streaming deployment example (Kafka + Flink + Redis)
- [ ] Native B2B benchmark dataset (extended version)

## Citation

If you use this code or methodology, please cite:

```bibtex
@inproceedings{sharma2026streambsg,
  title     = {STREAM-BSG: A Streaming Graph Architecture for Real-Time Fraud
               Detection in B2B Payment Networks},
  author    = {Sharma, Abhishek},
  booktitle = {IEEE International Conference on Contemporary Computing and Communications (InC4)},
  year      = {2026},
  note      = {To appear}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contact

Abhishek Sharma — abhicse24@gmail.com
ORCID: [0009-0007-1103-2103](https://orcid.org/0009-0007-1103-2103)

---

**Note:** The IEEE-CIS dataset is NOT redistributed in this repository. Users must download it directly from Kaggle under its original license terms.
