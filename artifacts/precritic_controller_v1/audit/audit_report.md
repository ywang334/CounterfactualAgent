# Pre-Critic Controller v1 Offline Audit

`offline_audit=true`, `controller_retrained=false`, `model_calls=0`, `final_test_evaluated=false`, and `deployable=false`.

The audit recomputes metrics from saved JSONL predictions and replays only the frozen Controller checkpoint plus local MiniLM on CPU. It does not train, initialize an LLM/backend, alter thresholds, or read Final Test examples.

## Run and artifacts

- Exit code: 0
- Process exited: True
- Observed wall clock: 48.019998 s
- Training device: cpu (GPU used: False)

| Artifact | Bytes | SHA256 |
|---|---:|---|
| primary_model.pt | 107244 | `f67e2ebbb0987fe1dce81e3ab9aed50abe4ade9fd17234b4dbca1dfb38e26052` |
| seed_metrics.json | 745500 | `57a7579a8fc4f5d82495f0d6befceb8c33360c3730e29b9190018f9ba003c5f7` |
| oof_predictions.jsonl | 3967581 | `b36ae7a9d858f6033534ad05e278faefe74f24f2bc85772fbcd79ee19d762324` |
| validation_predictions.jsonl | 430345 | `e3453c0270134788c5f2a7240e713e90ecc889ee30725d52d8311d6444a48259` |
| summary.json | 163325 | `16fc44f56072a7b09e794365e59cbd99ef7401184ca56943e72650565066f6e4` |
| report.md | 2478 | `59bcb4330bc4ababd1cf15bfaeb312bc3826f09403bc478006762d53fb4766b2` |

## Frozen protocol

- Training examples: 1000
- Training SHA256: `52d87713773308bc6085fa7d35f3c1be29f6de6e35633a286c68c0e241007303`
- Primary seed: 20260816
- Stability seeds: 20260817, 20260818, 20260819, 20260820
- Cost targets available/masked: 800/200
- Validation used for training/OOF/thresholds: false
- Hyperparameter search / best-seed selection: false / false
- Final Test: manifest-only verification; sealed and never evaluated

## Validation 100 recomputation

| Policy | Accuracy | Corrected | Degraded | Net | Critic rate | Mean total tokens | Mean calls | Mean latency (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| STOP | 0.6100 | 0 | 0 | 0 | 0.0000 | 397.73 | 1.00 | 1.9182 |
| ALWAYS_CRITIC_ONLY | 0.6400 | 8 | 5 | 3 | 1.0000 | 1182.92 | 2.00 | 4.7269 |
| OLD_PROBE | 0.6300 | 2 | 0 | 2 | 0.1100 | 485.30 | 1.11 | 2.2985 |
| CONTROLLER_V1_PRIMARY | 0.6300 | 2 | 0 | 2 | 0.0800 | 465.75 | 1.08 | 2.2444 |
| POSTHOC_ORACLE | 0.6900 | 8 | 0 | 8 | 0.0800 | 472.25 | 1.08 | 2.2393 |

`POSTHOC_ORACLE` is a minimum-cost retrospective oracle and remains `deployable=false`.

## Primary-seed head metrics

| Split | Helpful PR-AUC | Harmful PR-AUC | Macro-F1 | Cost MAE |
|---|---:|---:|---:|---:|
| OOF | 0.071162 | 0.087553 | 0.258674 | 813.711 |
| Validation | 0.239292 | 0.069748 | 0.290750 | 973.543 |

## Primary OOF-derived budget curve

All points are audit-only; no operating point is selected.

| Budget | Threshold | OOF calls | OOF acc. | OOF corr./degr./net | Val calls | Val acc. | Val corr./degr./net | Mean val tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5% | 0.35864978 | 50 | 0.7180 | 6/2/4 | 7 | 0.6200 | 1/0/1 | 455.81 |
| 10% | 0.23894896 | 100 | 0.7190 | 9/4/5 | 7 | 0.6200 | 1/0/1 | 455.81 |
| 20% | 0.12722360 | 200 | 0.7170 | 12/9/3 | 17 | 0.6200 | 2/1/1 | 537.68 |
| 30% | 0.06489222 | 300 | 0.7110 | 16/19/-3 | 29 | 0.6300 | 3/1/2 | 630.19 |
| 50% | -0.02881460 | 500 | 0.7000 | 30/44/-14 | 47 | 0.6300 | 4/2/2 | 773.80 |
| 100% | -0.71437487 | 1000 | 0.6990 | 64/79/-15 | 100 | 0.6400 | 8/5/3 | 1182.92 |

## Stability across five fixed seeds

Population mean ± standard deviation; no best seed was selected.

- accuracy: 0.620000 ± 0.006325
- corrected: 1.200000 ± 0.400000
- degraded: 0.200000 ± 0.400000
- net_benefit: 1.000000 ± 0.632456
- critic_call_rate: 0.054000 ± 0.032619
- mean_total_tokens: 442.372000 ± 27.850179
- mean_calls: 1.054000 ± 0.032619
- mean_latency_seconds: 2.124568 ± 0.140211
- helpful_pr_auc: 0.225717 ± 0.038309
- harmful_pr_auc: 0.075828 ± 0.009605
- four_class_macro_f1: 0.301979 ± 0.018292
- critic_cost_mae: 890.662767 ± 69.733912

## Paired exact McNemar

- Controller vs STOP: p=0.50000000
- Controller vs Always Critic-only: p=1.00000000

## CPU checkpoint replay

- Validation rows replayed: 100
- Actions exact match: True
- Maximum score difference: 2.980e-08
- Maximum probability difference: 2.384e-07

Audit complete. No threshold or final operating point was selected.
