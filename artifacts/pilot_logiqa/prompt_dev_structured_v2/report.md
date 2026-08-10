# LogiQA prompt replay: structured_v2

**Prompt development set only. deployable_result=false; this is not a final test result.**

Samples: 50
Solver reused: true; Solver called: false

## Strategy comparison

| Strategy | Strict acc. | Strict parse failures | Tolerant acc. | Tolerant parse failures | Corrected | Degraded | Unchanged | Avg tokens | Avg calls | Avg latency (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Solver Only | 0.6600 | 6 | 0.7400 | 0 | 0 | 0 | 50 | 383.44 | 1.00 | 1.7738 |
| Full minimal_v1 | 0.5200 | 6 | 0.5400 | 0 | 4 | 14 | 32 | 1449.78 | 3.00 | 5.8463 |
| Full structured_v2 | 0.6600 | 0 | 0.6600 | 0 | 3 | 7 | 40 | 1948.86 | 3.00 | 6.4053 |

## Critic and Refiner protocol

- Valid KEEP: 14
- Valid REVISE: 12
- Critic parse failures: 24
- Effective KEEP after safe fallback: 38
- false_revise: 8
- helpful_revise: 3
- REVISE followed by Refiner KEEP_ORIGINAL: 0
- Refiner protocol violations: 0

## Real token and latency cost

| Component | Prompt tokens | Completion tokens | Total tokens | Calls | Wall latency (s) |
|---|---:|---:|---:|---:|---:|
| Critic | 31595 | 7572 | 39167 | 50 | 148.6640 |
| Refiner | 35791 | 3313 | 39104 | 50 | 82.9098 |
| Replay actual (Critic + Refiner) | 67386 | 10885 | 78271 | 100 | 231.5738 |
| Complete structured_v2 including saved Solver | 82403 | 15040 | 97443 | 150 | 320.2652 |

All usage values are reported by the real OpenAI-compatible backend. No missing token usage was estimated.
