# Critic Gating Offline Audit

This is a saved-output counterfactual audit. `offline_audit=true`, `deployable=false`; no backend was initialized, no model was called, and no controller was trained.

Gating uses the historical `effective_verdict`: KEEP preserves the Solver; REVISE uses the valid effective proposed answer or the saved structured_v2 Refiner output, depending on policy.

## Validation 100

| Strategy | Accuracy | Corrected | Degraded | c→c | c→w | w→c | w→w | Calls | Mean total tokens | Mean latency (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STOP | 0.6100 | 0 | 0 | 61 | 0 | 0 | 39 | 100 | 397.73 | 1.9182 |
| MINIMAL_V1_ABLATION | 0.5800 | 15 | 18 | 43 | 18 | 15 | 24 | 300 | 1499.69 | 6.3766 |
| CRITIC_ONLY | 0.6400 | 8 | 5 | 56 | 5 | 8 | 31 | 200 | 1182.92 | 4.7269 |
| CONDITIONAL_REFINE | 0.6400 | 8 | 5 | 56 | 5 | 8 | 31 | 220 | 1374.99 | 5.1064 |
| ALWAYS_FULL | 0.6400 | 8 | 5 | 56 | 5 | 8 | 31 | 300 | 1975.17 | 6.4580 |

- Effective Critic KEEP/REVISE: 80/20
- Critic parse failures: 38
- Proposed/Refiner agreement: 20/20 (1.0000)
- Refiner changed Solver answer: 16
- Refiner beneficial/harmful changes: 8/5
- ALWAYS_FULL net benefit: 3
- CONDITIONAL_REFINE net benefit: 3

## Collection 200

| Strategy | Accuracy | Corrected | Degraded | c→c | c→w | w→c | w→w | Calls | Mean total tokens | Mean latency (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STOP | 0.7500 | 0 | 0 | 150 | 0 | 0 | 50 | 200 | unavailable | unavailable |
| MINIMAL_V1_ABLATION | 0.5800 | 21 | 55 | 95 | 55 | 21 | 29 | 600 | unavailable | unavailable |
| CRITIC_ONLY | 0.7500 | 14 | 14 | 136 | 14 | 14 | 36 | 400 | unavailable | unavailable |
| CONDITIONAL_REFINE | 0.7550 | 14 | 13 | 137 | 13 | 14 | 36 | 433 | unavailable | unavailable |
| ALWAYS_FULL | 0.7550 | 14 | 13 | 137 | 13 | 14 | 36 | 600 | unavailable | unavailable |

- Effective Critic KEEP/REVISE: 167/33
- Critic parse failures: 84
- Proposed/Refiner agreement: 32/33 (0.9697)
- Refiner changed Solver answer: 28
- Refiner beneficial/harmful changes: 14/13
- ALWAYS_FULL net benefit: 1
- CONDITIONAL_REFINE net benefit: 1

Collection stage token/latency cost is **unavailable** because Critic and Refiner usage was not saved separately. No estimate or subtraction was used.

## Interpretation boundary

These policies are posthoc simulations over already generated outputs. They are not deployable policy results and do not select or modify a production prompt or ActionController.
