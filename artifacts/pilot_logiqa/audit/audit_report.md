# LogiQA Pilot Offline Audit

Source: `/data/wy/agent/CounterfactualAgent/artifacts/pilot_logiqa/predictions.jsonl`
Samples: 50

No LLM or semantic-error classifier was used. Strict results are preserved; tolerant results only use explicit `FINAL_ANSWER:` markers.

## Accuracy and formatting

| Parser | Strategy | Accuracy | Parse failures | Format compliance |
|---|---|---:|---:|---:|
| Strict | Solver Only | 0.6600 | 6 | 0.8800 |
| Strict | Full | 0.5200 | 6 | 0.8800 |
| Tolerant | Solver Only | 0.7400 | 0 | 1.0000 |
| Tolerant | Full | 0.5400 | 0 | 1.0000 |

## Tolerant transition matrix

- `correct_to_correct`: 23 — 2745, 11656, 7475, 11625, 3197, 14198, 7261, 5181, 6581, 11618, 13351, 12776, 11674, 11484, 2221, 10680, 1391, 7564, 10985, 12476, 2900, 1522, 1062
- `correct_to_wrong`: 14 — 8898, 13622, 6137, 7776, 10583, 14532, 11202, 6643, 14352, 1222, 1444, 8685, 4219, 6773
- `wrong_to_correct`: 4 — 2288, 7040, 11554, 7450
- `wrong_to_wrong`: 9 — 6310, 3417, 5027, 3541, 2586, 8844, 2767, 486, 2941

## Format recovery

- Solver Only: 5027, 11202, 8685, 1522, 4219, 2941
- Full: 5027, 11202, 8685, 1522, 4219, 2941
- Conflicting Solver markers: None
- Conflicting Full markers: None

## Exact McNemar test (tolerant)

- Correct→wrong: 14
- Wrong→correct: 4
- Two-sided exact p-value: 0.03088379

## Recorded strategy costs

| Strategy | Avg total tokens | Avg calls | Avg latency (s) |
|---|---:|---:|---:|
| Solver Only | 383.44 | 1.00 | 1.7738 |
| Full | 1449.78 | 3.00 | 5.8463 |

## Post-hoc Oracle

**Warning: `posthoc_oracle=true`. This uses gold outcomes after inference and is not deployable.**

- Accuracy: 0.8200 (41/50)
- Full usage: 0.0800 (4/50)
- Average total tokens: 470.02
- Average calls: 1.16
- Average latency: 2.1143 seconds
