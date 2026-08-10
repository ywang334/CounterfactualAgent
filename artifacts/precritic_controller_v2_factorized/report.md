# Factorized Pre-Critic Controller v2

Offline supervised label-factorization experiment. final_test_evaluated=false, deployable=false, and model_calls=0.

Training data, frozen MiniLM, feature schema, 64d trunk, folds, seeds, optimizer, and epochs match Controller v1. Validation 100 is evaluation-only.

## Validation 100

| Policy | Accuracy | Corrected | Degraded | Net | Critic rate | Mean tokens | Mean calls | Mean latency (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| STOP | 0.6100 | 0 | 0 | 0 | 0.0000 | 397.73 | 1.00 | 1.9182 |
| ALWAYS_CRITIC_ONLY | 0.6400 | 8 | 5 | 3 | 1.0000 | 1182.92 | 2.00 | 4.7269 |
| OLD_PROBE | 0.6300 | 2 | 0 | 2 | 0.1100 | 485.30 | 1.11 | 2.2985 |
| CONTROLLER_V1_PRIMARY | 0.6300 | 2 | 0 | 2 | 0.0800 | 465.75 | 1.08 | 2.2444 |
| CONTROLLER_V2_FACTORIZED_PRIMARY | 0.6300 | 3 | 1 | 2 | 0.1700 | 543.39 | 1.17 | 2.5868 |
| POSTHOC_ORACLE | 0.6900 | 8 | 0 | 8 | 0.0800 | 472.25 | 1.08 | 2.2393 |

## Cost protection

- v1 target normalization: none
- Best constant baseline: training_median (MAE=74.906)
- Primary OOF cost-head MAE: 731.587
- Relative improvement: -8.766704
- cost_model_enabled: False
- Effective source: fixed_training_median_total_tokens

## Primary factorized heads

| Split | Error PR/F1 | Fix PR/F1 | Harm PR/F1 | Helpful PR | Harmful PR | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| OOF | 0.386591/0.398176 | 0.212996/0.240000 | 0.133015/0.202247 | 0.070917 | 0.084643 | 0.272228 |
| VALIDATION | 0.543353/0.465753 | 0.370733/0.434783 | 0.141876/0.166667 | 0.260898 | 0.083107 | 0.325502 |

## Primary OOF-derived budget curve

All points are shown. No final operating point is selected.

| Budget | Threshold | OOF calls | OOF C/D/Net | Val calls | Val accuracy | Val C/D/Net | Mean val tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5% | 0.23128647 | 50 | 5/2/3 | 8 | 0.6200 | 1/0/1 | 465.00 |
| 10% | 0.16899554 | 100 | 8/5/3 | 8 | 0.6200 | 1/0/1 | 465.00 |
| 20% | 0.09425546 | 200 | 16/10/6 | 18 | 0.6300 | 3/1/2 | 551.67 |
| 30% | 0.04740036 | 300 | 17/19/-2 | 30 | 0.6300 | 3/1/2 | 646.90 |
| 50% | -0.03456134 | 500 | 34/44/-10 | 52 | 0.6300 | 5/3/2 | 821.90 |
| 100% | -0.43275387 | 1000 | 64/79/-15 | 100 | 0.6400 | 8/5/3 | 1182.92 |

## Five-seed stability

- accuracy: 0.622000 ± 0.004000
- corrected: 1.400000 ± 0.800000
- degraded: 0.200000 ± 0.400000
- net_benefit: 1.200000 ± 0.400000
- critic_call_rate: 0.066000 ± 0.057480
- mean_total_tokens: 454.510000 ± 49.021242
- mean_calls: 1.066000 ± 0.057480
- mean_latency_seconds: 2.195229 ± 0.220494
- solver_error_pr_auc: 0.540323 ± 0.008368
- solver_error_f1: 0.476589 ± 0.017001
- critic_fix_pr_auc: 0.360413 ± 0.032395
- critic_fix_f1: 0.402831 ± 0.028512
- critic_harm_pr_auc: 0.154174 ± 0.024888
- critic_harm_f1: 0.155704 ± 0.017153
- final_helpful_pr_auc: 0.254402 ± 0.004210
- final_harmful_pr_auc: 0.079439 ± 0.006461
- four_class_macro_f1: 0.337734 ± 0.009740
- cost_head_mae: 845.585962 ± 54.118102

## Exact McNemar

- v2 vs STOP: p=0.62500000
- v2 vs Always Critic: p=1.00000000
- v2 vs v1: p=1.00000000

The posthoc oracle uses gold after generation and is deployable=false.
Final Test 500 remains sealed and was not read or evaluated.
