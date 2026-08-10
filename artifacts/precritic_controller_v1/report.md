# Pre-Critic Controller v1

Formal supervised training on Training 1000 with frozen MiniLM. `development_validation=true`, `final_test_evaluated=false`, `deployable=false`, and `model_calls=0`.

Validation 100 is evaluation-only. It is not used for training, OOF thresholds, or model/seed selection. Final Test 500 remains sealed.

## Development Validation 100

| Policy | Accuracy | Corrected | Degraded | Net | Critic rate | Mean tokens | Mean calls | Mean latency (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| STOP | 0.6100 | 0 | 0 | 0 | 0.0000 | 397.73 | 1.00 | 1.9182 |
| ALWAYS_CRITIC_ONLY | 0.6400 | 8 | 5 | 3 | 1.0000 | 1182.92 | 2.00 | 4.7269 |
| OLD_PROBE | 0.6300 | 2 | 0 | 2 | 0.1100 | 485.30 | 1.11 | 2.2985 |
| CONTROLLER_V1_PRIMARY | 0.6300 | 2 | 0 | 2 | 0.0800 | 465.75 | 1.08 | 2.2444 |
| POSTHOC_ORACLE | 0.6900 | 8 | 0 | 8 | 0.0800 | 472.25 | 1.08 | 2.2393 |

## Primary seed head metrics

- Seed: 20260816
- Helpful PR-AUC: 0.239292
- Harmful PR-AUC: 0.069748
- Four-class macro-F1: 0.290750
- Critic incremental total-token MAE: 973.543

## Primary OOF budget points

All points are shown; no deployment operating point is selected.

| Budget | Threshold | OOF calls | OOF net | Validation calls | Validation accuracy | Validation net |
|---:|---:|---:|---:|---:|---:|---:|
| 5% | 0.35864979 | 50 | 4 | 7 | 0.6200 | 1 |
| 10% | 0.23894897 | 100 | 5 | 7 | 0.6200 | 1 |
| 20% | 0.12722360 | 200 | 3 | 17 | 0.6200 | 1 |
| 30% | 0.06489222 | 300 | -3 | 29 | 0.6300 | 2 |
| 50% | -0.02881460 | 500 | -14 | 47 | 0.6300 | 2 |
| 100% | -0.71437488 | 1000 | -15 | 100 | 0.6400 | 3 |

## Stability

Seeds 20260817-20260820 are stability reports only; the primary model is always seed 20260816 and no best seed is selected.

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
- critic_cost_mae: 890.662769 ± 69.733918

## Paired tests

- Controller v1 vs STOP exact McNemar p=0.50000000
- Controller v1 vs Always Critic-only exact McNemar p=1.00000000

`POSTHOC_ORACLE` uses gold after generation and is deployable=false.
No Final Test example or model inference was read or executed.
