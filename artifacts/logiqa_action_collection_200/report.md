# LogiQA 2.0 train three-action paired rollout collection

Real paired collection only; no controller was trained and no budget-bound labels were generated.

Samples: 200 (seed=20260812)
Actual calls: 1000

## Accuracy and outcomes

| Action | Strict acc. | Strict failures | Tolerant acc. | Tolerant failures | Corrected | Degraded | Helpful | Harmful | Neutral correct | Neutral wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STOP | 0.6300 | 36 | 0.7500 | 0 | 0 | 0 | 0 | 0 | 150 | 50 |
| SHORT | 0.4650 | 36 | 0.5800 | 0 | 21 | 55 | 21 | 55 | 95 | 29 |
| FULL | 0.7550 | 0 | 0.7550 | 0 | 14 | 13 | 14 | 13 | 137 | 36 |

## Incremental cost distributions

### STOP

| Cost | Mean | P50 | P90 | P95 | Max |
|---|---:|---:|---:|---:|---:|
| prompt_tokens | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| completion_tokens | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| total_tokens | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| calls | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| latency_seconds | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### SHORT

| Cost | Mean | P50 | P90 | P95 | Max |
|---|---:|---:|---:|---:|---:|
| prompt_tokens | 873.4500 | 852.5000 | 1075.1000 | 1125.1500 | 1381.0000 |
| completion_tokens | 205.7050 | 202.5000 | 243.1000 | 263.5000 | 349.0000 |
| total_tokens | 1079.1550 | 1053.5000 | 1295.1000 | 1350.2000 | 1675.0000 |
| calls | 2.0000 | 2.0000 | 2.0000 | 2.0000 | 2.0000 |
| latency_seconds | 4.4674 | 4.3621 | 5.2044 | 5.7390 | 6.9298 |

### FULL

| Cost | Mean | P50 | P90 | P95 | Max |
|---|---:|---:|---:|---:|---:|
| prompt_tokens | 1345.1150 | 1314.0000 | 1555.0000 | 1648.5500 | 1847.0000 |
| completion_tokens | 204.9550 | 180.0000 | 298.0000 | 322.1000 | 575.0000 |
| total_tokens | 1550.0700 | 1516.5000 | 1806.5000 | 1926.2500 | 2177.0000 |
| calls | 2.0000 | 2.0000 | 2.0000 | 2.0000 | 2.0000 |
| latency_seconds | 4.5547 | 4.1511 | 6.1078 | 6.5049 | 10.2866 |

## Corrected/degraded overlap

- Corrected intersection: 14222, 2042, 3466, 3663, 4396, 497, 5829, 6234, 7302, 7727
- Corrected Jaccard: 0.4000
- Degraded intersection: 12097, 14215, 15019, 1672, 5513, 8538, 8904
- Degraded Jaccard: 0.1148

## Minimum-cost posthoc oracle

- Accuracy: 0.8750
- STOP/SHORT/FULL selections: 175/21/4
- posthoc_oracle=true; deployable=false.

## Actual service cost

- Prompt tokens: 503633
- Completion tokens: 99560
- Total tokens: 603193
- Calls: 1000
- Summed request latency: 2194.2333s

All usage is service-reported; no missing usage is estimated.
