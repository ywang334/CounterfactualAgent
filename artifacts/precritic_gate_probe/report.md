# Pre-Critic Gate Learnability Probe

Pure offline supervised probe. `controller_probe=true`, `deployable=false`, `final_test=false`. No LLM/backend was initialized or called.

Collection 200 is used only for stratified OOF threshold selection and final Probe training. Validation 100 is evaluated once after the threshold is fixed.

## OOF threshold

- Threshold: 0.09621601
- OOF corrected/degraded/net: 5/2/3
- OOF Critic call rate: 0.1900
- Safe always-STOP fallback: False

## Independent Validation 100

| Policy | Accuracy | Corrected | Degraded | Net | Critic rate | Mean total tokens | Mean calls | Mean latency (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| STOP | 0.6100 | 0 | 0 | 0 | 0.0000 | 397.73 | 1.00 | 1.9182 |
| ALWAYS_CRITIC_ONLY | 0.6400 | 8 | 5 | 3 | 1.0000 | 1182.92 | 2.00 | 4.7269 |
| LEARNED_GATE | 0.6300 | 2 | 0 | 2 | 0.1100 | 485.30 | 1.11 | 2.2985 |
| POSTHOC_ORACLE | 0.6900 | 8 | 0 | 8 | 0.0800 | 472.25 | 1.08 | 2.2393 |

`POSTHOC_ORACLE` uses gold after generation and is deployable=false.

## OOF-derived budget curve on Validation 100

Validation labels were not used to choose any threshold.

| OOF budget | Threshold | Validation Critic rate | Accuracy | Corrected | Degraded | Mean tokens |
|---:|---:|---:|---:|---:|---:|---:|
| 10% | 0.19777165 | 0.0500 | 0.6200 | 1 | 0 | 438.91 |
| 20% | 0.09463838 | 0.1100 | 0.6300 | 2 | 0 | 485.30 |
| 30% | 0.02111789 | 0.2400 | 0.6500 | 5 | 1 | 595.67 |
| 50% | -0.01242808 | 0.3900 | 0.6500 | 6 | 2 | 710.19 |
| 100% | -0.82245646 | 1.0000 | 0.6400 | 8 | 5 | 1182.92 |

## Learnability assessment

- Continue collection/formal controller training: True
- Basis: OOF threshold has positive corrected-degraded and the independently evaluated Learned Gate also has positive net benefit.
- Caveat: The signal is based on 200 training and 100 policy-selection validation samples; this Probe remains deployable=false and final_test=false.

## Boundary

This is a learnability probe, not a deployable controller and not a final test. No hyperparameter search, prompt change, extra collection, or follow-up tuning was performed.
