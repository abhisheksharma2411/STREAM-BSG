# Evaluation

 
Evaluation is driven end-to-end from the repository root — see [`../../reproduce.sh`](../../reproduce.sh) and [`../../REPRODUCE.md`](../../REPRODUCE.md).
 
Each stage writes its metrics to [`../../results/`](../../results/):
 
- **Accuracy (Table II)** — `results/table_accuracy.csv`, produced by `stream_bsg/classifier.py` and the three baselines (`baselines/lr_baseline.py`, `baselines/xgb_tabular.py`, `baselines/graphsage_baseline.py`).
- **Per-row latency (Table III)** — `results/table_latency.csv`, measured inside `classifier.py` after a 100-prediction warm-up.
- **Per-topology recall (Table IV)** — `results/table_per_topology.csv`, including T2 per-event recall.
- **Synthetic-attribute leakage (Table V)** — `results/*_no_synth_leak.json` + `results/leakage_analysis_report.md`.
- **Feature / prevalence / OOD ablations (Table VI)** — `results/ablation_a_no_topology_feats.json`, `results/ablation_0p5pct/`, `results/ablation_c_ood_injection.json`.
- **Seed stability** — `results/seed_stability.json` and `results/graphsage_seed*.json`.
Metric definitions (ROC-AUC, PR-AUC, best-F1, recall @ 5% FPR) are implemented in `stream_bsg/classifier.py::compute_metrics`. A consolidated write-up of every experiment is in [`../../results/all_experiments_summary.md`](../../results/all_experiments_summary.md).
 
